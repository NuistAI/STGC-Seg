#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Run Medical-SAM3 text-prompt inference on OCT samples.

Filename rule for this OCT dataset:
    CSV / GTs:  subject_1_10.png
    Images:     patient_1_B_scan_10.png

Put this file under:
    D:\Users\admin\Pycharmprojects\experiments2\Medical-SAM3-main\inference\octtest_patient_subject_fixed.py

Then run:
    C:\Users\admin\anaconda3\envs\SAM\python.exe octtest_patient_subject_fixed.py
"""

import os
import re
import json
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sam3_inference as sam3_mod

if not hasattr(sam3_mod, "SAM3_ROOT"):
    sam3_mod.SAM3_ROOT = sam3_mod._REPO_ROOT

from sam3_inference import SAM3Model, resize_mask
from metrics import compute_dice, compute_iou


CHECKPOINT_PATH = Path(r"D:\Users\admin\Pycharmprojects\experiments2\Medical-SAM3-main\checkpoint\checkpoint.pt")
IMAGE_DIR = Path(r"D:\Users\admin\Data\OCTtext\datasets\2015_BOE_Chiu\segment_annotation\All Set\Images")
TEXT_CSV = Path(r"D:\Users\admin\Data\OCTtext\datasets\2015_BOE_Chiu\segment_annotation\descriptions.csv")
GT_DIR = Path(r"D:\Users\admin\Data\OCTtext\datasets\2015_BOE_Chiu\segment_annotation\All Set\GTs")

MAX_SAMPLES = 20
CONFIDENCE_THRESHOLD = 0.1

OUT_DIR = Path(r"D:\Users\admin\Pycharmprojects\experiments2\Medical-SAM3-main\oct_medsam3_results_20")
PRED_MASK_DIR = OUT_DIR / "pred_masks"
OVERLAY_DIR = OUT_DIR / "overlays"
VIS_PATH = OUT_DIR / "comparison_20x4.png"
CSV_PATH = OUT_DIR / "metrics_20.csv"
JSON_PATH = OUT_DIR / "summary_20.json"

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def check_paths() -> None:
    for name, p in [
        ("checkpoint", CHECKPOINT_PATH),
        ("image dir", IMAGE_DIR),
        ("text csv", TEXT_CSV),
        ("gt dir", GT_DIR),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def read_binary_mask(path: Path, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Read mask as binary uint8 array of shape HxW."""
    mask_img = Image.open(path).convert("L")
    if target_shape is not None:
        h, w = target_shape
        if mask_img.size != (w, h):
            mask_img = mask_img.resize((w, h), Image.NEAREST)
    mask = np.array(mask_img)
    return (mask > 127).astype(np.uint8)


def unique_keep_order(items: List[str]) -> List[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def subject_stem_to_patient_stem(stem: str) -> Optional[str]:
    """
    CSV/GT stem -> image stem.

    Example:
        subject_1_10  -> patient_1_B_scan_10
        subject_10_25 -> patient_10_B_scan_25
    """
    m = re.match(r"^subject_(\d+)_(\d+)$", stem)
    if not m:
        return None
    return f"patient_{m.group(1)}_B_scan_{m.group(2)}"


def patient_stem_to_subject_stem(stem: str) -> Optional[str]:
    """
    Image stem -> CSV/GT stem.

    Example:
        patient_1_B_scan_10  -> subject_1_10
        patient_10_B_scan_25 -> subject_10_25
    """
    m = re.match(r"^patient_(\d+)_B_scan_(\d+)$", stem)
    if not m:
        return None
    return f"subject_{m.group(1)}_{m.group(2)}"


def _find_with_common_extensions(root: Path, filename: str, recursive: bool = False) -> Optional[Path]:
    """Find a file by exact filename, then by same stem with common extensions."""
    filename = Path(str(filename).strip()).name
    if not filename:
        return None

    p = root / filename
    if p.exists():
        return p

    stem = Path(filename).stem
    suffixes = []
    if Path(filename).suffix:
        suffixes.append(Path(filename).suffix)
    suffixes.extend(IMG_EXTS)

    seen = set()
    for ext in suffixes:
        for e in (ext, ext.lower(), ext.upper()):
            if not e or e in seen:
                continue
            seen.add(e)
            p = root / f"{stem}{e}"
            if p.exists():
                return p

    if recursive:
        for ext in suffixes:
            pattern = f"{stem}{ext}"
            matches = list(root.rglob(pattern))
            if matches:
                return matches[0]
            pattern = f"{stem}{ext.upper()}"
            matches = list(root.rglob(pattern))
            if matches:
                return matches[0]

    return None


def _candidate_image_names_from_csv(csv_image_name: str) -> List[str]:
    """
    Build candidate original-image filenames from CSV filename.

    Your OCT rule:
        CSV:    subject_1_10.png
        Images: patient_1_B_scan_10.png
    """
    name = Path(str(csv_image_name).strip()).name
    stem = Path(name).stem
    suffix = Path(name).suffix or ".png"

    candidates = []

    candidates.append(name)

    stem_no_mask = stem[5:] if stem.lower().startswith("mask_") else stem

    patient_stem = subject_stem_to_patient_stem(stem_no_mask)
    if patient_stem is not None:
        candidates.append(f"{patient_stem}{suffix}")
        candidates.append(f"{patient_stem}.png")

    if re.match(r"^patient_\d+_B_scan_\d+$", stem_no_mask):
        candidates.append(f"{stem_no_mask}{suffix}")
        candidates.append(f"{stem_no_mask}.png")

    return unique_keep_order(candidates)


def _candidate_gt_names_from_csv(csv_image_name: str) -> List[str]:
    """
    Build candidate GT filenames from CSV filename.

    Your OCT rule:
        CSV: subject_1_10.png
        GTs: subject_1_10.png
    """
    name = Path(str(csv_image_name).strip()).name
    stem = Path(name).stem
    suffix = Path(name).suffix or ".png"

    candidates = []

    candidates.append(name)

    if stem.lower().startswith("mask_"):
        no_mask_stem = stem[5:]
        candidates.append(f"{no_mask_stem}{suffix}")
        candidates.append(f"{no_mask_stem}.png")
    else:
        candidates.append(f"mask_{stem}{suffix}")
        candidates.append(f"mask_{stem}.png")

    subject_stem = patient_stem_to_subject_stem(stem)
    if subject_stem is not None:
        candidates.append(f"{subject_stem}{suffix}")
        candidates.append(f"{subject_stem}.png")
        candidates.append(f"mask_{subject_stem}{suffix}")
        candidates.append(f"mask_{subject_stem}.png")

    return unique_keep_order(candidates)


def find_image_path(csv_image_name: str) -> Optional[Path]:
    """Find original image under IMAGE_DIR using subject->patient mapping."""
    for candidate in _candidate_image_names_from_csv(csv_image_name):
        p = _find_with_common_extensions(IMAGE_DIR, candidate, recursive=False)
        if p is not None:
            return p
    return None


def find_gt_path(csv_image_name: str) -> Optional[Path]:
    """Find GT mask under GT_DIR. For your OCT dataset, direct subject filename is expected."""
    for candidate in _candidate_gt_names_from_csv(csv_image_name):
        p = _find_with_common_extensions(GT_DIR, candidate, recursive=False)
        if p is not None:
            return p
    return None


def get_csv_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Support both Image/Description and image/description column names."""
    lower_to_real = {c.lower(): c for c in df.columns}

    if "image" not in lower_to_real:
        raise ValueError(f"CSV must contain an image column. Current columns: {list(df.columns)}")

    if "description" in lower_to_real:
        return lower_to_real["image"], lower_to_real["description"]

    if "text" in lower_to_real:
        return lower_to_real["image"], lower_to_real["text"]

    if "prompt" in lower_to_real:
        return lower_to_real["image"], lower_to_real["prompt"]

    raise ValueError(
        "CSV must contain one text column: Description / description / text / prompt. "
        f"Current columns: {list(df.columns)}"
    )


def mask_to_rgb(mask: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    mask = mask.astype(bool)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = np.array(color, dtype=np.uint8)
    return out


def make_overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay masks on image:
      - GT only: green
      - Prediction only: red
      - GT ∩ Prediction: yellow
    """
    image = image.copy().astype(np.float32)
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)

    gt_only = gt_bool & ~pred_bool
    pred_only = pred_bool & ~gt_bool
    both = gt_bool & pred_bool

    overlay = image.copy()
    overlay[gt_only] = (1 - alpha) * overlay[gt_only] + alpha * np.array([0, 255, 0])
    overlay[pred_only] = (1 - alpha) * overlay[pred_only] + alpha * np.array([255, 0, 0])
    overlay[both] = (1 - alpha) * overlay[both] + alpha * np.array([255, 255, 0])

    return np.clip(overlay, 0, 255).astype(np.uint8)


def resize_for_grid(arr: np.ndarray, size: int = 256) -> Image.Image:
    img = Image.fromarray(arr.astype(np.uint8))
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_comparison_grid(rows, out_path: Path, cell_size: int = 256, header_h: int = 34, label_h: int = 26) -> None:
    """
    Save one Nx4 grid:
      col1 original, col2 GT, col3 prediction, col4 overlay.
    """
    headers = ["Image", "GT", "Prediction", "Overlay: GT green / Pred red / Overlap yellow"]
    n = len(rows)
    w = cell_size * 4
    h = header_h + n * (cell_size + label_h)
    grid = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for j, title in enumerate(headers):
        x = j * cell_size + 6
        draw.text((x, 10), title, fill=(0, 0, 0), font=font)

    for i, row in enumerate(rows):
        y0 = header_h + i * (cell_size + label_h)
        panels = [
            resize_for_grid(row["image"], cell_size),
            resize_for_grid(mask_to_rgb(row["gt"], (255, 255, 255)), cell_size),
            resize_for_grid(mask_to_rgb(row["pred"], (255, 255, 255)), cell_size),
            resize_for_grid(row["overlay"], cell_size),
        ]
        for j, panel in enumerate(panels):
            grid.paste(panel, (j * cell_size, y0))

        label = (
            f"{i+1:02d} csv={row['csv_name']} | img={row['image_file']} | "
            f"Dice={row['dice']:.4f}, IoU={row['iou']:.4f}"
        )
        draw.text((6, y0 + cell_size + 5), label[:190], fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main() -> None:
    check_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_MASK_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TEXT_CSV, encoding="utf-8-sig")
    image_col, text_col = get_csv_columns(df)

    print(f"CSV image column: {image_col}")
    print(f"CSV text column: {text_col}")
    print("Filename rule: subject_X_Y.png in CSV/GTs -> patient_X_B_scan_Y.png in Images")

    samples = []
    for _, r in df.iterrows():
        csv_name = str(r[image_col]).strip()
        desc = str(r[text_col]).strip()

        if not csv_name or csv_name.lower() in ["nan", "none"]:
            continue
        if not desc or desc.lower() in ["nan", "none"]:
            desc = "OCT lesion"

        img_path = find_image_path(csv_name)
        gt_path = find_gt_path(csv_name)

        if img_path is None:
            print(
                f"[Skip] image not found for CSV name: {csv_name}. "
                f"Expected image like: patient_X_B_scan_Y.png"
            )
            continue

        if gt_path is None:
            print(
                f"[Skip] GT not found for CSV name: {csv_name}. "
                f"Expected GT like: subject_X_Y.png"
            )
            continue

        samples.append((csv_name, desc, img_path, gt_path))
        if len(samples) >= MAX_SAMPLES:
            break

    if not samples:
        raise RuntimeError(
            "No valid samples found. Check filename mapping: "
            "CSV/GTs should be subject_X_Y.png, Images should be patient_X_B_scan_Y.png."
        )

    print(f"Found {len(samples)} valid samples. Running Medical-SAM3 text-prompt inference...")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output dir: {OUT_DIR}")

    model = SAM3Model(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        checkpoint_path=str(CHECKPOINT_PATH),
    )

    results = []
    grid_rows = []

    for idx, (csv_name, desc, img_path, gt_path) in enumerate(tqdm(samples, desc="OCT"), start=1):
        image = read_rgb(img_path)
        h, w = image.shape[:2]
        gt = read_binary_mask(gt_path, target_shape=(h, w))

        state = model.encode_image(image)
        pred = model.predict_text(state, desc)

        if pred is None:
            print(f"[Warn] no prediction for {csv_name}; use all-zero mask")
            pred = np.zeros((h, w), dtype=np.uint8)
        else:
            pred = np.squeeze(pred).astype(np.uint8)
            if pred.shape != (h, w):
                pred = resize_mask(pred, (h, w))
            pred = (pred > 0).astype(np.uint8)

        dice = float(compute_dice(pred, gt))
        iou = float(compute_iou(pred, gt))

        csv_stem = Path(csv_name).stem
        pred_path = PRED_MASK_DIR / f"{idx:02d}_{csv_stem}_pred.png"
        overlay_path = OVERLAY_DIR / f"{idx:02d}_{csv_stem}_overlay.png"

        Image.fromarray((pred * 255).astype(np.uint8)).save(pred_path)
        overlay = make_overlay(image, gt, pred)
        Image.fromarray(overlay).save(overlay_path)

        results.append({
            "index": idx,
            "csv_name": csv_name,
            "image_file": img_path.name,
            "gt_file": gt_path.name,
            "description": desc,
            "image_path": str(img_path),
            "gt_path": str(gt_path),
            "pred_mask_path": str(pred_path),
            "overlay_path": str(overlay_path),
            "dice": dice,
            "miou": iou,
        })

        grid_rows.append({
            "csv_name": csv_name,
            "image_file": img_path.name,
            "image": image,
            "gt": gt,
            "pred": pred,
            "overlay": overlay,
            "dice": dice,
            "iou": iou,
        })

    res_df = pd.DataFrame(results)
    mean_dice = float(res_df["dice"].mean())
    mean_miou = float(res_df["miou"].mean())

    res_df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    make_comparison_grid(grid_rows, VIS_PATH, cell_size=256)

    summary = {
        "num_samples": int(len(res_df)),
        "mean_dice": mean_dice,
        "mean_miou": mean_miou,
        "checkpoint": str(CHECKPOINT_PATH),
        "image_dir": str(IMAGE_DIR),
        "gt_dir": str(GT_DIR),
        "text_csv": str(TEXT_CSV),
        "metrics_csv": str(CSV_PATH),
        "comparison_image": str(VIS_PATH),
        "filename_rule": "CSV/GTs subject_X_Y.png -> Images patient_X_B_scan_Y.png",
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean mIoU: {mean_miou:.4f}")
    print(f"Per-image metrics CSV: {CSV_PATH}")
    print(f"20x4 comparison image: {VIS_PATH}")
    print(f"Prediction masks: {PRED_MASK_DIR}")
    print(f"Overlays: {OVERLAY_DIR}")


if __name__ == "__main__":
    main()
