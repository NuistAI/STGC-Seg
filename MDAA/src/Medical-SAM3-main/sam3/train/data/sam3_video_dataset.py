# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import copy

import io
import json
import logging
import math
import os
import pickle
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torchvision


from iopath.common.file_io import PathManager
from PIL import Image as PILImage

from .sam3_image_dataset import Datapoint, Sam3ImageDataset


SEED = 42


class VideoGroundingDataset(Sam3ImageDataset):
    def __init__(
        self,
        num_stages_sample: int = 4,
        stage_stride_min: int = 1,
        stage_stride_max: int = 5,
        random_reverse_time_axis: bool = True,
        use_all_frames_in_train: bool = False,
        allow_ptr_remap: bool = False,
        is_tiling_single_image: bool = False,
        tile_img_keep_find_queries_with_geo_inputs: bool = False,
        tile_img_keep_get_queries: bool = False,
        max_query_num: int = -1,
        override_query_is_exhaustive_to_true: bool = False,
        max_masklet_num_in_video: int = 500,
        **kwargs,
    ):
        """
        Loading video grounding data

        Video frame sampling parameters (for training only):
        - num_stages_sample: number of frames to sample from the video during training
        - stage_stride_min: minimum stride between sampled frames during training
        - stage_stride_max: maximum stride between sampled frames during training (if it's
          greater than stage_stride_min, the actual stride is sampled uniformly between min
          and max; during inference, we always use all frames in the video with stride=1)
        - random_reverse_time_axis: whether to randomly invert the video's temporal axis
          (i.e. playing it backwards) during training
        """
        super().__init__(**kwargs)
        assert num_stages_sample >= 1
        assert stage_stride_min >= 1
        assert stage_stride_max >= stage_stride_min
        self.num_stages_sample = num_stages_sample
        self.stage_stride_min = stage_stride_min
        self.stage_stride_max = stage_stride_max
        self.random_reverse_time_axis = random_reverse_time_axis
        self.use_all_frames_in_train = use_all_frames_in_train
        self.allow_ptr_remap = allow_ptr_remap
        self.is_tiling_single_image = is_tiling_single_image
        self.tile_img_keep_find_queries_with_geo_inputs = (
            tile_img_keep_find_queries_with_geo_inputs
        )
        self.tile_img_keep_get_queries = tile_img_keep_get_queries
        self.max_query_num = max_query_num
        self.override_query_is_exhaustive_to_true = override_query_is_exhaustive_to_true
        self.max_masklet_num_in_video = max_masklet_num_in_video
        self.rng = random.Random()
        self.set_curr_epoch(0)

    def set_curr_epoch(self, epoch: int):
        super().set_curr_epoch(epoch)
        self.rng.seed(SEED + epoch)

    def _normalize_frame_local_ids(self, queries, annotations):
        annotations = [ann.copy() for ann in annotations]
        queries = [query.copy() for query in queries]

        ann_id_to_frame_id = {}
        for ann in annotations:
            frame_id = ann.get("frame_index", ann.get("image_id", 0))
            ann["image_id"] = frame_id
            ann_id_to_frame_id[ann["id"]] = frame_id

        for query in queries:
            obj_ids = query.get("object_ids_output") or []
            if obj_ids:
                frame_id = ann_id_to_frame_id.get(
                    obj_ids[0], query.get("image_id", 0)
                )
            else:
                frame_id = query.get("frame_index", query.get("image_id", 0))
            query["image_id"] = frame_id
            if query.get("query_processing_order", 0) != frame_id:
                query["query_processing_order"] = frame_id

        return queries, annotations

    def _load_datapoint(self, index: int) -> Datapoint:
        id = self.ids[index].item()
        queries, annotations = self.coco.loadQueriesAndAnnotationsFromDatapoint(id)
        queries, annotations = self._normalize_frame_local_ids(queries, annotations)

        if (
            self.training
            and not self.is_tiling_single_image
            and not self.use_all_frames_in_train
        ):
            stage_stride = self.rng.randint(
                self.stage_stride_min, self.stage_stride_max
            )
            stage_ids_to_keep = self._sample_stage_ids(
                queries, self.num_stages_sample, stage_stride
            )
            reverse_time_axis = (
                self.rng.random() < 0.5 if self.random_reverse_time_axis else False
            )
            queries, annotations, kept_img_ids = self._filter_query_and_anns(
                queries,
                annotations,
                stage_ids_to_keep,
                remap_stage_id=True,
                reverse_time_axis=reverse_time_axis,
            )
            pil_images, img_metadata = self._load_images(id, kept_img_ids)
            if reverse_time_axis:
                pil_images = pil_images[::-1]
                img_metadata = img_metadata[::-1]
        else:
            pil_images, img_metadata = self._load_images(id)

        if not self.is_tiling_single_image:
            stage_to_image_id = self._get_stage_to_image_id(queries)
            stage_ids_match_image_ids = all(
                stage_id == image_id
                for stage_id, image_id in stage_to_image_id.items()
            )
            if stage_ids_match_image_ids:
                for image_id, _ in pil_images:
                    stage_to_image_id.setdefault(image_id, image_id)
            queries = self._merge_and_pad_queries_across_stages(
                queries, stage_to_image_id
            )

        assert all(p.size == pil_images[0][1].size for _, p in pil_images)

        queries.sort(key=lambda q: q["query_processing_order"])
        if self.override_query_is_exhaustive_to_true:
            for query in queries:
                query["is_exhaustive"] = True
        try:
            datapoint = self.load_queries(pil_images, annotations, queries, img_metadata)
        except AssertionError as e:
            if "Number of queries in stage" not in str(e):
                raise
            logging.warning(
                f"Datapoint {id} has inconsistent query counts across stages: {e}. "
                "Skipping this datapoint."
            )
            next_index = (index + 1) % len(self)
            return self._load_datapoint(next_index)

        num_masklets_in_video = len(datapoint.images[0].objects)
        if num_masklets_in_video > self.max_masklet_num_in_video > 0:
            logging.warning(
                f"Datapoint {id} has num_masklets_in_video={num_masklets_in_video}, "
                f"exceeding the maximum allowed ({self.max_masklet_num_in_video}). "
                "Skipping this datapoint."
            )
            next_index = (index + 1) % len(self)
            return self._load_datapoint(next_index)

        if self.is_tiling_single_image:
            datapoint = self._tile_single_image_data(datapoint, self.num_stages_sample)
        if self.max_query_num > 0:
            datapoint = self._subsample_queries(datapoint, self.max_query_num)

        for query in datapoint.find_queries:
            assert query.image_id == query.query_processing_order, (
                "find query has inconsistent image_id and "
                f"query_processing_order: image_id={query.image_id} vs "
                f"query_processing_order={query.query_processing_order}"
            )
        return datapoint

    def _sample_stage_ids(self, queries, num_stages_sample, stage_stride):
        """Sample a subset of stage ids from all queries."""
        all_stage_ids = sorted(set(q["query_processing_order"] for q in queries))
        num_stages_total = len(all_stage_ids)
        if num_stages_total < num_stages_sample:
            return all_stage_ids

        b_e_gap = (num_stages_sample - 1) * stage_stride
        if b_e_gap > num_stages_total - 1:
            prev_stage_stride = stage_stride
            stage_stride = math.floor((num_stages_total - 1) / (num_stages_sample - 1))
            logging.info(
                f"lowering stride from {prev_stage_stride} to {stage_stride} to "
                f"sample {num_stages_sample} stages (from {num_stages_total} total)"
            )
            b_e_gap = (num_stages_sample - 1) * stage_stride

        b_max = len(all_stage_ids) - 1 - b_e_gap
        b = self.rng.randint(0, b_max)
        e = b + b_e_gap
        stage_ids_to_keep = all_stage_ids[b : e + 1 : stage_stride]
        return stage_ids_to_keep

    def _get_stage_to_image_id(self, queries):
        stage_to_image_id = {}
        for query in queries:
            stage_id = query["query_processing_order"]
            image_id = query["image_id"]
            prev_image_id = stage_to_image_id.get(stage_id)
            if prev_image_id is not None:
                assert prev_image_id == image_id, (
                    f"Inconsistent image ids for stage {stage_id}: "
                    f"{prev_image_id} vs {image_id}"
                )
            stage_to_image_id[stage_id] = image_id
        return stage_to_image_id

    def _query_slot_key(self, query):
        original_cat_id = query.get("original_cat_id")
        if original_cat_id is not None:
            return ("original_cat_id", original_cat_id)
        query_text = query.get("query_text")
        if query_text not in [None, ""]:
            return ("query_text", query_text)
        return ("query_id", query.get("id"))

    def _ordered_unique(self, values):
        seen = set()
        deduped = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    def _first_non_empty_query_field(self, queries, field_name):
        for query in queries:
            value = query.get(field_name)
            if value not in [None, [], ()]:
                return copy.deepcopy(value)
        return None

    def _merge_queries_for_slot(self, queries, stage_id, image_id):
        merged_query = copy.deepcopy(queries[0])
        merged_query["image_id"] = image_id
        merged_query["query_processing_order"] = stage_id
        merged_query["object_ids_output"] = self._ordered_unique(
            object_id
            for query in queries
            for object_id in query.get("object_ids_output", []) or []
        )
        merged_query["ptr_x_query_id"] = None
        merged_query["ptr_y_query_id"] = None
        merged_query["input_box"] = self._first_non_empty_query_field(
            queries, "input_box"
        )
        merged_query["input_box_label"] = self._first_non_empty_query_field(
            queries, "input_box_label"
        )
        merged_query["input_points"] = self._first_non_empty_query_field(
            queries, "input_points"
        )
        merged_query["is_exhaustive"] = all(
            query.get("is_exhaustive", True) for query in queries
        )
        return merged_query

    def _make_empty_query(self, template_query, stage_id, image_id):
        empty_query = copy.deepcopy(template_query)
        empty_query["id"] = None
        empty_query["image_id"] = image_id
        empty_query["query_processing_order"] = stage_id
        empty_query["object_ids_output"] = []
        empty_query["ptr_x_query_id"] = None
        empty_query["ptr_y_query_id"] = None
        empty_query["input_box"] = None
        empty_query["input_box_label"] = None
        empty_query["input_points"] = None
        empty_query["is_exhaustive"] = template_query.get("is_exhaustive", True)
        return empty_query

    def _merge_and_pad_queries_across_stages(self, queries, stage_to_image_id):
        """
        Align video queries to a fixed set of semantic slots:
        1. Merge duplicate queries with the same slot key on the same stage.
        2. Pad missing slots on a stage with empty queries.

        This keeps the per-stage query count and ordering consistent, while letting
        medical slices represent a structure as absent on frames where it is not visible.
        """
        if not queries:
            return queries

        queries_by_stage_and_slot = defaultdict(lambda: defaultdict(list))
        slot_templates = {}
        slot_order = []
        for query in queries:
            stage_id = query["query_processing_order"]
            slot_key = self._query_slot_key(query)
            queries_by_stage_and_slot[stage_id][slot_key].append(query)
            if slot_key not in slot_templates:
                slot_templates[slot_key] = copy.deepcopy(query)
                slot_order.append(slot_key)

        aligned_queries = []
        for stage_id in sorted(stage_to_image_id):
            image_id = stage_to_image_id[stage_id]
            stage_queries = queries_by_stage_and_slot.get(stage_id, {})
            for slot_key in slot_order:
                slot_queries = stage_queries.get(slot_key)
                if slot_queries:
                    aligned_queries.append(
                        self._merge_queries_for_slot(slot_queries, stage_id, image_id)
                    )
                else:
                    aligned_queries.append(
                        self._make_empty_query(
                            slot_templates[slot_key], stage_id, image_id
                        )
                    )
        return aligned_queries

    def _filter_query_and_anns(
        self, queries, annotations, stage_ids_to_keep, remap_stage_id, reverse_time_axis
    ):
        """Filter queries and annotations to only keep those in `stage_ids_to_keep`."""
        stage_ids_to_keep = set(stage_ids_to_keep)
        kept_img_ids = set()
        kept_stage_ids = set()

        filtered_queries = []
        for query in queries:
            input_box = query.get("input_box", None)
            input_points = query.get("input_points", None)
            has_geo_input = input_box is not None or input_points is not None
            if has_geo_input and not self.tile_img_keep_find_queries_with_geo_inputs:
                continue
            stage_id = query["query_processing_order"]
            if stage_id in stage_ids_to_keep:
                kept_img_ids.add(query["image_id"])
                kept_stage_ids.add(stage_id)
                filtered_queries.append(query)
        all_frame_present = kept_stage_ids == stage_ids_to_keep
        assert all_frame_present, (
            f"kept_stage_ids={kept_stage_ids} vs stage_ids_to_keep={stage_ids_to_keep}"
        )
        if remap_stage_id:
            old_stage_ids = sorted(kept_stage_ids, reverse=reverse_time_axis)
            stage_id_old2new = {old: new for new, old in enumerate(old_stage_ids)}
            kept_query_ids = {q.get("id") for q in filtered_queries if q.get("id") is not None}
            for query in filtered_queries:
                if self.allow_ptr_remap:
                    ptr_x = query.get("ptr_x_query_id")
                    ptr_y = query.get("ptr_y_query_id")
                    if ptr_x not in kept_query_ids:
                        query["ptr_x_query_id"] = None
                    if ptr_y not in kept_query_ids:
                        query["ptr_y_query_id"] = None
                else:
                    ptr_x_is_empty = query["ptr_x_query_id"] in [None, -1]
                    ptr_y_is_empty = query["ptr_y_query_id"] in [None, -1]
                    assert (
                        ptr_x_is_empty and ptr_y_is_empty
                    ), "Remapping stage ids is not supported for queries with non-empty ptr_x or ptr_y pointers"
                query["query_processing_order"] = stage_id_old2new[
                    query["query_processing_order"]
                ]

        filtered_annotations = [
            ann for ann in annotations if ann["image_id"] in kept_img_ids
        ]

        return filtered_queries, filtered_annotations, kept_img_ids

    def _tile_single_image_data(self, datapoint: Datapoint, num_stages_sample: int):
        """
        Tile a single image and its queries to simulate video frames. The output is a
        datapoint with *identical video frames* (i.e. the same static image) and needs
        further transforms (e.g. affine) to get video frames with different content.
        """
        assert len(datapoint.images) == 1, "Expected only one single image"
        tiled_images = [
            copy.deepcopy(datapoint.images[0]) for _ in range(num_stages_sample)
        ]
        for stage_id, img in enumerate(tiled_images):
            for obj in img.objects:
                obj.frame_index = stage_id

        tiled_raw_images = None
        if datapoint.raw_images is not None:
            assert len(datapoint.raw_images) == 1, "Expected only one single image"
            tiled_raw_images = [
                datapoint.raw_images[0].copy() for _ in range(num_stages_sample)
            ]

        tiled_find_queries_per_stage = [[] for _ in range(num_stages_sample)]
        for query in datapoint.find_queries:
            assert query.image_id == 0
            assert query.query_processing_order == 0
            assert query.ptr_x is None and query.ptr_y is None
            assert query.ptr_mem is None
            has_geo_input = (
                query.input_bbox is not None or query.input_points is not None
            )
            if has_geo_input and not self.tile_img_keep_find_queries_with_geo_inputs:
                continue
            for stage_id in range(num_stages_sample):
                new_query = copy.deepcopy(query)
                new_query.image_id = stage_id
                new_query.query_processing_order = stage_id
                if new_query.inference_metadata is not None:
                    new_query.inference_metadata.frame_index = stage_id
                tiled_find_queries_per_stage[stage_id].append(new_query)

        tiled_find_queries = sum(tiled_find_queries_per_stage, [])

        if self.tile_img_keep_get_queries:
            raise NotImplementedError("Tiling get queries is not implemented yet")
        else:
            tiled_get_queries = []

        return Datapoint(
            images=tiled_images,
            raw_images=tiled_raw_images,
            find_queries=tiled_find_queries,
            get_queries=tiled_get_queries,
        )

    def _subsample_queries(self, datapoint: Datapoint, max_query_num: int):
        """Subsample to keep at most `max_query_num` queries per frame in a datapoint."""
        num_frames = max(q.query_processing_order for q in datapoint.find_queries) + 1
        find_queries_per_stage = [[] for _ in range(num_frames)]
        for query in datapoint.find_queries:
            find_queries_per_stage[query.query_processing_order].append(query)

        num_queries_per_stage = len(find_queries_per_stage[0])
        for queries in find_queries_per_stage:
            assert len(queries) == num_queries_per_stage
        if max_query_num <= 0 or num_queries_per_stage <= max_query_num:
            return datapoint

        sampled_inds = self.rng.sample(range(num_queries_per_stage), max_query_num)
        sampled_find_queries_per_stage = [
            [queries[idx] for idx in sampled_inds] for queries in find_queries_per_stage
        ]
        sampled_find_queries = sum(sampled_find_queries_per_stage, [])
        return Datapoint(
            images=datapoint.images,
            raw_images=datapoint.raw_images,
            find_queries=sampled_find_queries,
            get_queries=datapoint.get_queries,
        )
