import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

from sam3.model_builder import build_sam3_image_model
from train_mdaa import (
    apply_mdaa_training,
    last_output,
    make_model_adapter_config,
    make_loader,
    move_to_device,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an adapter-only checkpoint with overlap and surface metrics."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def surface_metrics(pred, target):
    """Return HD95 and ASSD in resized-image pixels for a non-empty pair."""
    if not pred.any() or not target.any():
        return None, None
    structure = generate_binary_structure(2, 1)
    pred_surface = np.logical_xor(
        pred, binary_erosion(pred, structure=structure, border_value=0)
    )
    target_surface = np.logical_xor(
        target, binary_erosion(target, structure=structure, border_value=0)
    )
    target_distance = distance_transform_edt(~target_surface)
    pred_distance = distance_transform_edt(~pred_surface)
    distances = np.concatenate(
        [target_distance[pred_surface], pred_distance[target_surface]]
    )
    return float(np.percentile(distances, 95)), float(np.mean(distances))


def load_adapter_checkpoint(path, model):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("checkpoint_format") != "trainable_parameters_only_v1":
        raise RuntimeError(
            f"Unsupported checkpoint format: {state.get('checkpoint_format')!r}"
        )
    supplied = set(state["trainable_model"])
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if supplied != expected:
        raise RuntimeError(
            "Trainable checkpoint/model key mismatch: "
            f"missing={sorted(expected - supplied)[:20]} "
            f"unexpected={sorted(supplied - expected)[:20]}"
        )
    incompatible = model.load_state_dict(state["trainable_model"], strict=False)
    unexpected = [name for name in incompatible.unexpected_keys if name in supplied]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:20]}")
    return state


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    metrics_path = output_dir / "metrics.json"
    per_image_path = output_dir / "per_image_metrics.csv"
    if metrics_path.exists() or per_image_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing evaluation in {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-resolution evaluation")

    set_seed(int(cfg["training"]["seed"]))
    device = torch.device("cuda")
    loader = make_loader(cfg, "test")
    model = build_sam3_image_model(
        device="cuda",
        eval_mode=False,
        checkpoint_path=str(Path(cfg["pretrained_checkpoint"])),
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
        context_length=32,
        adapter_config=make_model_adapter_config(cfg["adapter"]),
    )
    apply_mdaa_training(model, cfg["strategy"])
    checkpoint_state = load_adapter_checkpoint(checkpoint_path, model)
    model.eval()

    sample_ids = [sample[3] for sample in loader.dataset.samples]
    rows = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    both_empty = 0
    pred_empty_target_nonempty = 0
    pred_nonempty_target_empty = 0
    offset = 0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if args.max_steps is not None and step >= args.max_steps:
                break
            batch = move_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                outputs = model(batch)
                out_dict = last_output(outputs)
                targets = batch.find_targets[0].semantic_segments.bool()
                logits = out_dict["semantic_seg"]
                if logits.shape[-2:] != targets.shape[-2:]:
                    logits = F.interpolate(
                        logits.float(),
                        size=targets.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
            predictions = (logits.float().sigmoid().squeeze(1) > 0.5).cpu().numpy()
            target_arrays = targets.cpu().numpy()

            for pred, target in zip(predictions, target_arrays):
                sample_id = sample_ids[offset]
                offset += 1
                pred = pred.astype(bool, copy=False)
                target = target.astype(bool, copy=False)
                tp = int(np.logical_and(pred, target).sum())
                fp = int(np.logical_and(pred, ~target).sum())
                fn = int(np.logical_and(~pred, target).sum())
                total_tp += tp
                total_fp += fp
                total_fn += fn
                dice_denominator = 2 * tp + fp + fn
                iou_denominator = tp + fp + fn
                dice = 1.0 if dice_denominator == 0 else (2.0 * tp) / dice_denominator
                iou = 1.0 if iou_denominator == 0 else tp / iou_denominator

                pred_nonempty = bool(pred.any())
                target_nonempty = bool(target.any())
                if pred_nonempty and target_nonempty:
                    hd95, assd = surface_metrics(pred, target)
                    boundary_status = "valid"
                else:
                    hd95, assd = None, None
                    if not pred_nonempty and not target_nonempty:
                        both_empty += 1
                        boundary_status = "both_empty"
                    elif not pred_nonempty:
                        pred_empty_target_nonempty += 1
                        boundary_status = "pred_empty_target_nonempty"
                    else:
                        pred_nonempty_target_empty += 1
                        boundary_status = "pred_nonempty_target_empty"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "pred_pixels": int(pred.sum()),
                        "target_pixels": int(target.sum()),
                        "dice": dice,
                        "iou": iou,
                        "hd95_pixels": hd95,
                        "assd_pixels": assd,
                        "boundary_status": boundary_status,
                    }
                )
            if (step + 1) % 25 == 0:
                print(
                    f"EVAL_PROGRESS batches={step + 1}/{len(loader)} images={len(rows)}",
                    flush=True,
                )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not rows:
        raise RuntimeError("Evaluation produced no samples")
    boundary_rows = [row for row in rows if row["boundary_status"] == "valid"]
    global_dice_denominator = 2 * total_tp + total_fp + total_fn
    global_iou_denominator = total_tp + total_fp + total_fn
    metrics = {
        "dataset": cfg["dataset"],
        "strategy": cfg["strategy"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_state["epoch"]),
        "threshold": 0.5,
        "test_n": len(rows),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "global_dice": (
            1.0
            if global_dice_denominator == 0
            else (2.0 * total_tp) / global_dice_denominator
        ),
        "global_iou": (
            1.0 if global_iou_denominator == 0 else total_tp / global_iou_denominator
        ),
        "hd95_pixels": (
            float(np.mean([row["hd95_pixels"] for row in boundary_rows]))
            if boundary_rows
            else None
        ),
        "assd_pixels": (
            float(np.mean([row["assd_pixels"] for row in boundary_rows]))
            if boundary_rows
            else None
        ),
        "n_hd95": len(boundary_rows),
        "n_assd": len(boundary_rows),
        "n_both_empty": both_empty,
        "n_pred_empty_target_nonempty": pred_empty_target_nonempty,
        "n_pred_nonempty_target_empty": pred_nonempty_target_empty,
        "surface_metric_policy": (
            "HD95 and ASSD are per-image symmetric surface distances averaged over "
            "samples where prediction and target are both non-empty; undefined cases "
            "are excluded and counted explicitly. Distances use 4-connected surfaces "
            "with unit spacing on 1008x1008 resized masks, so units are pixels."
        ),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "evaluation_seconds": elapsed,
        "milliseconds_per_image": 1000.0 * elapsed / len(rows),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_steps": args.max_steps,
    }

    with per_image_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("EXTENDED_METRICS " + json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
