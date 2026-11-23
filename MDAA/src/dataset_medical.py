import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from sam3.model.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage,
)


class MedicalTextSegDataset(Dataset):
    """QaTa/MosMed image-mask-text loader with deterministic labeled subset."""

    def __init__(
        self,
        image_dir,
        mask_dir,
        csv_path,
        image_size=1008,
        labeled_ratio=1.0,
        seed=0,
        split="train",
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.csv_path = Path(csv_path)
        self.image_size = int(image_size)
        self.split = split

        for path in (self.image_dir, self.mask_dir, self.csv_path):
            if not path.exists():
                raise FileNotFoundError(path)

        frame = pd.read_csv(self.csv_path)
        if frame.shape[1] < 2:
            raise ValueError(f"CSV must contain image and text columns: {self.csv_path}")

        samples = []
        missing = []
        for row in frame.itertuples(index=False, name=None):
            raw_name, raw_text = row[0], row[1]
            if pd.isna(raw_name) or pd.isna(raw_text):
                continue
            csv_name = str(raw_name)
            text = str(raw_text)
            direct_image = self.image_dir / csv_name
            stripped_image = self.image_dir / (
                csv_name[5:] if csv_name.startswith("mask_") else csv_name
            )
            image_path = direct_image if direct_image.exists() else stripped_image
            direct_mask = self.mask_dir / csv_name
            prefixed_mask = self.mask_dir / f"mask_{csv_name}"
            mask_path = direct_mask if direct_mask.exists() else prefixed_mask
            if not image_path.exists() or not mask_path.exists():
                missing.append((csv_name, str(image_path), str(mask_path)))
                continue
            samples.append((image_path, mask_path, text, csv_name))

        if missing:
            preview = missing[:5]
            raise FileNotFoundError(
                f"{len(missing)} CSV rows have missing image/mask files; first={preview}"
            )
        if not samples:
            raise RuntimeError(f"No usable samples found in {self.csv_path}")

        self.full_count = len(samples)
        if split == "train":
            ratio = float(labeled_ratio)
            if not 0.0 < ratio <= 1.0:
                raise ValueError(f"labeled_ratio must be in (0,1], got {ratio}")
            count = max(1, int(round(ratio * len(samples))))
            rng = np.random.default_rng(int(seed))
            keep = np.sort(rng.permutation(len(samples))[:count])
            samples = [samples[i] for i in keep]
        self.samples = samples

        print(
            f"DATASET split={split} csv={self.csv_path} full={self.full_count} "
            f"selected={len(self.samples)} labeled_ratio={labeled_ratio if split == 'train' else 1.0}",
            flush=True,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, text, sample_id = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        mask = torch.from_numpy((np.asarray(mask) > 0).astype(np.bool_))

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        ys, xs = torch.where(mask)
        if len(xs):
            x0, x1 = xs.min().float(), xs.max().float() + 1.0
            y0, y1 = ys.min().float(), ys.max().float() + 1.0
            width, height = x1 - x0, y1 - y0
            bbox = torch.stack(
                [
                    (x0 + x1) / (2.0 * self.image_size),
                    (y0 + y1) / (2.0 * self.image_size),
                    width / self.image_size,
                    height / self.image_size,
                ]
            )
            valid = True
        else:
            bbox = torch.zeros(4, dtype=torch.float32)
            valid = False

        return {
            "image": image,
            "mask": mask,
            "bbox": bbox,
            "valid": valid,
            "text": text,
            "sample_id": sample_id,
        }


def medical_collate(batch):
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    boxes = torch.stack([item["bbox"] for item in batch])
    valid = torch.tensor([item["valid"] for item in batch], dtype=torch.bool)
    texts = [item["text"] for item in batch]
    batch_size = len(batch)

    find_stage = FindStage(
        img_ids=torch.arange(batch_size, dtype=torch.long),
        text_ids=torch.arange(batch_size, dtype=torch.long),
        input_boxes=torch.zeros(0, batch_size, 4, dtype=torch.float32),
        input_boxes_label=torch.zeros(0, batch_size, dtype=torch.long),
        input_boxes_mask=torch.ones(batch_size, 0, dtype=torch.bool),
        input_points=torch.empty(0, batch_size, 2, dtype=torch.float32),
        input_points_mask=torch.empty(batch_size, 0, dtype=torch.bool),
        object_ids=[[0] for _ in range(batch_size)],
    )
    target = BatchedFindTarget(
        num_boxes=torch.ones(batch_size, dtype=torch.long),
        boxes=boxes,
        boxes_padded=boxes.unsqueeze(1),
        is_exhaustive=torch.ones(batch_size, dtype=torch.bool),
        segments=masks,
        semantic_segments=masks.clone(),
        is_valid_segment=valid,
        repeated_boxes=[],
        object_ids=list(range(batch_size)),
        object_ids_padded=torch.zeros(batch_size, 1, dtype=torch.long),
    )
    metadata = BatchedInferenceMetadata(
        coco_image_id=torch.arange(batch_size, dtype=torch.long),
        original_size=[(images.shape[-2], images.shape[-1])] * batch_size,
        object_id=torch.zeros(batch_size, dtype=torch.long),
        frame_index=torch.zeros(batch_size, dtype=torch.long),
        original_image_id=torch.arange(batch_size, dtype=torch.long),
        original_category_id=torch.zeros(batch_size, dtype=torch.long),
        is_conditioning_only=torch.zeros(batch_size, dtype=torch.bool),
    )
    datapoint = BatchedDatapoint(
        img_batch=images,
        find_text_batch=texts,
        find_inputs=[find_stage],
        find_targets=[target],
        find_metadatas=[metadata],
        raw_images=None,
    )
    return datapoint
