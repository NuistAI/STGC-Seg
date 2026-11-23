#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Pure Medical-SAM3 inference script for QaTa-COV19.

功能：
1. 读取 CSV 里的图片名；
2. 在 Images 文件夹里找原图；
3. 在 GTs 文件夹里找对应 mask；
4. 所有样本统一使用固定文本 prompt："lung lesion"；
5. 跑 Medical-SAM3 text-only 推理；
6. 计算 Dice / mIoU；
7. 保存预测 mask、overlay、20x4 对比图、CSV 和 JSON。

不需要 pywt / monai / transformers / cv2。

建议放到：
D:\Users\admin\Pycharmprojects\experiments2\Medical-SAM3-main\inference\run_qata_cov19_fixed_lung_lesion_prompt.py

运行：
C:\Users\admin\anaconda3\envs\SAM\python.exe D:\Users\admin\Pycharmprojects\experiments2\Medical-SAM3-main\inference\run_qata_cov19_fixed_lung_lesion_prompt.py
"""

import os
import sys
import json
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
IMAGE_DIR = Path(r"D:\Users\admin\Data\QaTa-COV19-ori\QaTa-COV19\QaTa-COV19-v2\All_Set\Images")
TEXT_CSV = Path(r"D:\Users\admin\Data\QaTa-COV19-ori\QaTa-COV19\QaTa-COV19-v2\prompt\All_text.csv")
GT_DIR = Path(r"D:\Users\admin\Data\QaTa-COV19-ori\QaTa-COV19\QaTa-COV19-v2\All_Set\GTs")

MAX_SAMPLES = 20

CONFIDENCE_THRESHOLD = 0.1

FIXED_TEXT_PROMPT = "lung lesion"

OUT_DIR = Path(r"D:\Users\admin\Pycharmprojects\experiments2\Medical-SAM3-main\qata_cov19_medsam3_fixed_lung_lesion_results_20")
PRED_MASK_DIR = OUT_DIR / "pred_masks"
OVERLAY_DIR = OUT_DIR / "overlays"
VIS_PATH = OUT_DIR / "comparison_20x4.png"
CSV_PATH = OUT_DIR / "metrics_20.csv"
JSON_PATH = OUT_DIR / "summary_20.json"


def check_paths() -> None:
    for name, path in [
        ("checkpoint", CHECKPOINT_PATH),
        ("image dir", IMAGE_DIR),
        ("text csv", TEXT_CSV),
        ("gt dir", GT_DIR),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def read_binary_mask(path: Path, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """读取 GT mask，转成 0/1 二值图。"""
    mask_img = Image.open(path).convert("L")
    if target_shape is not None:
        h, w = target_shape
        if mask_img.size != (w, h):
            mask_img = mask_img.resize((w, h), Image.NEAREST)
    mask = np.array(mask_img)
    return (mask > 127).astype(np.uint8)


def get_image_column(df: pd.DataFrame) -> str:
    """自动识别 CSV 里的图片名列。优先 Image，其次 image，否则第一列。"""
    for col in ["Image", "image", "img", "filename", "file", "name"]:
        if col in df.columns:
            return col
    return df.columns[0]


def candidate_image_names(csv_name: str) -> List[str]:
    """
    根据 CSV 里的名字生成候选原图名。

    支持：
    CSV 是 covid_1.png      -> Images/covid_1.png
    CSV 是 mask_covid_1.png -> Images/covid_1.png 或 Images/mask_covid_1.png
    """
    name = Path(str(csv_name).strip()).name
    names = []

    if name.lower().startswith("mask_"):
        names.append(name[5:])

    names.append(name)

    out = []
    for n in names:
        if n and n not in out:
            out.append(n)
    return out


def find_with_common_extensions(root: Path, filename: str) -> Optional[Path]:
    """先按完整文件名找，再按同 stem 的常见扩展名找。"""
    filename = Path(str(filename).strip()).name
    direct = root / filename
    if direct.exists():
        return direct

    stem = Path(filename).stem
    suffixes = []
    if Path(filename).suffix:
        suffixes.append(Path(filename).suffix)
    suffixes.extend([".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"])

    seen = set()
    for ext in suffixes:
        for e in [ext, ext.lower(), ext.upper()]:
            if not e or e in seen:
                continue
            seen.add(e)
            p = root / f"{stem}{e}"
            if p.exists():
                return p

    return None


def find_image_path(csv_image_name: str) -> Optional[Path]:
    """根据 CSV 图片名找 Images 里的原图。"""
    for candidate in candidate_image_names(csv_image_name):
        p = find_with_common_extensions(IMAGE_DIR, candidate)
        if p is not None:
            return p
    return None


def find_gt_path(csv_image_name: str) -> Optional[Path]:
    """
    根据 CSV 图片名找 GT mask。

    支持：
    CSV: covid_1.png      -> GTs/mask_covid_1.png
    CSV: mask_covid_1.png -> GTs/mask_covid_1.png
    """
    name = Path(str(csv_image_name).strip()).name

    candidates = []
    if name.lower().startswith("mask_"):
        candidates.append(name)
    else:
        candidates.append(f"mask_{name}")
        candidates.append(name)

    if name.lower().startswith("mask_"):
        candidates.append(name[5:])

    out = []
    for c in candidates:
        if c and c not in out:
            out.append(c)

    for candidate in out:
        p = find_with_common_extensions(GT_DIR, candidate)
        if p is not None:
            return p

    return None


def mask_to_rgb(mask: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    mask_bool = mask.astype(bool)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask_bool] = np.array(color, dtype=np.uint8)
    return out


def make_overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    叠加显示：
    GT only: 绿色
    Pred only: 红色
    GT ∩ Pred: 黄色
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
    headers = ["Image", "GT", "Prediction", "Overlay: GT green / Pred red / Overlap yellow"]
    n = len(rows)
    w = cell_size * 4
    h = header_h + n * (cell_size + label_h)
    grid = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for j, title in enumerate(headers):
        draw.text((j * cell_size + 6, 10), title, fill=(0, 0, 0), font=font)

    for i, row in enumerate(rows):
        y0 = header_h + i * (cell_size + label_h)
        panels = [
            resize_for_grid(row["image"], cell_size),
            resize_for_grid(mask_to_rgb(row["gt"]), cell_size),
            resize_for_grid(mask_to_rgb(row["pred"]), cell_size),
            resize_for_grid(row["overlay"], cell_size),
        ]
        for j, panel in enumerate(panels):
            grid.paste(panel, (j * cell_size, y0))

        label = f"{i + 1:02d} {row['image_name']} | Dice={row['dice']:.4f}, IoU={row['iou']:.4f} | Prompt={FIXED_TEXT_PROMPT}"
        draw.text((6, y0 + cell_size + 5), label[:180], fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main() -> None:
    check_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_MASK_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TEXT_CSV, encoding="utf-8-sig")
    image_col = get_image_column(df)

    print(f"CSV columns: {list(df.columns)}")
    print(f"Use image column: {image_col}")
    print(f"Fixed text prompt for all samples: {FIXED_TEXT_PROMPT}")

    samples = []
    for _, row in df.iterrows():
        image_name = str(row[image_col]).strip()
        if not image_name or image_name.lower() in ["nan", "none"]:
            continue

        img_path = find_image_path(image_name)
        gt_path = find_gt_path(image_name)

        if img_path is None:
            print(f"[Skip] image not found: {image_name}")
            continue
        if gt_path is None:
            print(f"[Skip] GT not found for: {image_name}")
            continue

        samples.append((image_name, FIXED_TEXT_PROMPT, img_path, gt_path))
        if len(samples) >= MAX_SAMPLES:
            break

    if not samples:
        raise RuntimeError("No valid samples found. Check image filenames, GT filenames, and paths.")

    print(f"Found {len(samples)} valid samples. Running Medical-SAM3 text-prompt inference...")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Image dir: {IMAGE_DIR}")
    print(f"GT dir: {GT_DIR}")
    print(f"Output dir: {OUT_DIR}")

    model = SAM3Model(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        checkpoint_path=str(CHECKPOINT_PATH),
    )

    results = []
    grid_rows = []

    for idx, (image_name, prompt, img_path, gt_path) in enumerate(tqdm(samples, desc="Medical-SAM3"), start=1):
        image = read_rgb(img_path)
        h, w = image.shape[:2]
        gt = read_binary_mask(gt_path, target_shape=(h, w))

        state = model.encode_image(image)
        pred = model.predict_text(state, prompt)

        if pred is None:
            print(f"[Warn] no prediction for {image_name}; use all-zero mask")
            pred = np.zeros((h, w), dtype=np.uint8)
        else:
            pred = np.squeeze(pred).astype(np.uint8)
            if pred.shape != (h, w):
                pred = resize_mask(pred, (h, w))
            pred = (pred > 0).astype(np.uint8)

        dice = float(compute_dice(pred, gt))
        iou = float(compute_iou(pred, gt))

        stem = Path(image_name).stem
        pred_path = PRED_MASK_DIR / f"{idx:02d}_{stem}_pred.png"
        overlay_path = OVERLAY_DIR / f"{idx:02d}_{stem}_overlay.png"

        Image.fromarray((pred * 255).astype(np.uint8)).save(pred_path)
        overlay = make_overlay(image, gt, pred)
        Image.fromarray(overlay).save(overlay_path)

        results.append({
            "index": idx,
            "image": image_name,
            "fixed_prompt": prompt,
            "image_path": str(img_path),
            "gt_path": str(gt_path),
            "pred_mask_path": str(pred_path),
            "overlay_path": str(overlay_path),
            "dice": dice,
            "miou": iou,
        })

        grid_rows.append({
            "image_name": image_name,
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
        "fixed_prompt": FIXED_TEXT_PROMPT,
        "mean_dice": mean_dice,
        "mean_miou": mean_miou,
        "checkpoint": str(CHECKPOINT_PATH),
        "image_dir": str(IMAGE_DIR),
        "gt_dir": str(GT_DIR),
        "text_csv": str(TEXT_CSV),
        "metrics_csv": str(CSV_PATH),
        "comparison_image": str(VIS_PATH),
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Fixed prompt: {FIXED_TEXT_PROMPT}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean mIoU: {mean_miou:.4f}")
    print(f"Per-image metrics CSV: {CSV_PATH}")
    print(f"20x4 comparison image: {VIS_PATH}")
    print(f"Prediction masks: {PRED_MASK_DIR}")
    print(f"Overlays: {OVERLAY_DIR}")


if __name__ == "__main__":
    main()
