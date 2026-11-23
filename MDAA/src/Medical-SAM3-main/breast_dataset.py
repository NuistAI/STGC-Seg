"""BrEaST-Lesions-USG dataset adapter for the SAM3 fine-tuning project.

This module is a *new* dataset layer. It does NOT modify ``dataset.py``.

The parsing logic is an equivalent re-implementation of the already-validated
BrEaST handling found in the reference project:

    /home/Disk4toCheckpoint/gx/DFTSEG_breast/utils/dataset.py  ->  ``_init_breast``

Key rules (kept identical to the reference):

* Table columns used: ``Image`` (image filename, e.g. ``case001.png``) and
  ``text`` (English ultrasound report, used as the text prompt).
* Image files: ``*.png`` whose stem contains **no** underscore -> ``case001.png``.
* Mask files: ``{stem}_tumor.png`` -> ``case001_tumor.png``.
* An image is kept only when (a) its filename is present in the report table and
  (b) the corresponding ``*_tumor.png`` mask exists on disk.
* Extra-lesion files such as ``case022_other1.png`` are excluded by the
  underscore rule, so each patient contributes exactly one (image, mask) pair
  -> the split is effectively patient-level.
* Data split: a fixed-seed (0) random permutation, 70% train / 10% valid /
  20% test, exactly matching DFTSEG_breast.

The returned ``__getitem__`` dict has the same keys/format as the existing
``QaTaDataset`` (``image`` / ``mask`` / ``bbox`` / ``prompt``), so it can be fed
straight into the existing ``qata_collate_fn``.
"""

import os
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _read_xlsx_stdlib(path):
    """Minimal stdlib-only .xlsx reader.

    Fallback for environments where ``openpyxl`` is not installed (e.g. the
    SAM3 conda env ``GXSAM310``). Reads the first worksheet and the shared
    string table, returning a ``pandas.DataFrame`` with the first row as header.
    """
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root:
                shared.append(
                    "".join(t.text or "" for t in si.iter(f"{{{_NS_MAIN}}}t"))
                )

        sheets = [
            n
            for n in z.namelist()
            if n.startswith("xl/worksheets/")
            and n.endswith(".xml")
            and "_rels" not in n
        ]
        sheet_root = ET.fromstring(z.read(sheets[0]))

        rows = []
        for row in sheet_root.iter(f"{{{_NS_MAIN}}}row"):
            cells = []
            for c in row.findall(f"{{{_NS_MAIN}}}c"):
                t = c.get("t")
                v_el = c.find(f"{{{_NS_MAIN}}}v")
                if t == "inlineStr":
                    is_el = c.find(f"{{{_NS_MAIN}}}is")
                    val = (
                        "".join(x.text or "" for x in is_el.iter(f"{{{_NS_MAIN}}}t"))
                        if is_el is not None
                        else ""
                    )
                elif t == "s" and v_el is not None:
                    idx = int(v_el.text)
                    val = shared[idx] if 0 <= idx < len(shared) else ""
                else:
                    val = v_el.text if v_el is not None else ""
                cells.append(val)
            rows.append(cells)

    header = rows[0] if rows else []
    data = rows[1:] if len(rows) > 1 else []
    return pd.DataFrame(data, columns=header)


class BrEaSTDataset(Dataset):
    """BrEaST-Lesions-USG dataset.

    Args:
        csv_path: path to ``BrEaST_Lesions_USG_English_Reports_3cols.xlsx``
            (also accepts CSV).
        root_path: directory containing ``caseXXX.png`` and ``caseXXX_tumor.png``.
        split: one of ``train`` / ``valid`` (or ``val``) / ``test``.
        labeled_ratio: only used for ``split == 'train'``. If not None, selects
            ``labeled_ratio`` fraction of the training split as labeled samples
            (reproducible via ``seed``). None = use all training samples.
        seed: controls the labeled-subset permutation (NOT the train/val/test
            split, which is fixed to seed 0 to match DFTSEG_breast).
        image_size: (width, height) to which images/masks are resized.
    """

    def __init__(
        self,
        csv_path,
        root_path,
        split="train",
        labeled_ratio=None,
        seed=0,
        transform=None,
        image_size=(1008, 1008),
    ):
        super().__init__()
        self.root_path = root_path
        self.split = split
        self.image_size = tuple(image_size)
        self.transform = transform

        report_df = self._read_table(csv_path)

        image_col = "Image" if "Image" in report_df.columns else report_df.columns[0]
        if "text" in report_df.columns:
            text_col = "text"
        elif "Description" in report_df.columns:
            text_col = "Description"
        else:
            text_col = report_df.columns[1]

        reports = {str(row[image_col]): str(row[text_col]) for _, row in report_df.iterrows()}

        image_files = sorted(
            f
            for f in os.listdir(root_path)
            if f.lower().endswith(".png") and "_" not in os.path.splitext(f)[0]
        )

        all_pairs = []
        for image_name in image_files:
            stem = os.path.splitext(image_name)[0]
            mask_name = f"{stem}_tumor.png"
            if image_name in reports and os.path.exists(os.path.join(root_path, mask_name)):
                all_pairs.append((mask_name, image_name))

        rng = np.random.default_rng(0)
        indices = np.arange(len(all_pairs))
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(n_total * 0.7)
        n_valid = int(n_total * 0.1)
        split_map = {
            "train": indices[:n_train],
            "valid": indices[n_train : n_train + n_valid],
            "val": indices[n_train : n_train + n_valid],
            "test": indices[n_train + n_valid :],
        }
        chosen = split_map.get(split, indices)
        self.data_pairs = [all_pairs[i] for i in chosen]

        if split == "train" and labeled_ratio is not None:
            labeled_ratio = float(labeled_ratio)
            n_labeled = max(1, int(round(labeled_ratio * len(self.data_pairs))))
            label_rng = np.random.default_rng(seed)
            perm = label_rng.permutation(len(self.data_pairs))
            keep = np.sort(perm[:n_labeled])
            self.data_pairs = [self.data_pairs[i] for i in keep]

        self.samples = []
        for mask_name, image_name in self.data_pairs:
            self.samples.append(
                {
                    "img_path": os.path.join(root_path, image_name),
                    "mask_path": os.path.join(root_path, mask_name),
                    "prompt": reports[image_name],
                }
            )
        print(f"[{split}] BrEaST data ready: {len(self.samples)}/{len(all_pairs)} samples")

    @staticmethod
    def _read_table(path):
        if str(path).lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        return pd.read_csv(path, encoding="utf-8-sig")

    @staticmethod
    def _load_mask(mask_path, image_size):
        mask = Image.open(mask_path).convert("RGBA")
        arr = np.array(mask)
        binary = ((arr[..., 3] > 0) | (arr[..., :3].sum(axis=-1) > 0)).astype(np.uint8)
        mask_img = Image.fromarray(binary * 255).resize(image_size, Image.Resampling.NEAREST)
        return (np.array(mask_img) > 128).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = Image.open(sample["img_path"]).convert("RGB")
        img = img.resize(self.image_size, Image.Resampling.BILINEAR)

        mask = self._load_mask(sample["mask_path"], self.image_size)

        if self.transform:
            img = self.transform(img)
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)

        mask = torch.tensor(mask, dtype=torch.bool)

        y_indices, x_indices = torch.where(mask)
        if len(y_indices) > 0:
            y_min, y_max = y_indices.min().item(), y_indices.max().item()
            x_min, x_max = x_indices.min().item(), x_indices.max().item()
            bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        else:
            bbox = torch.zeros(4, dtype=torch.float32)

        return {
            "image": img,
            "mask": mask,
            "bbox": bbox,
            "prompt": sample["prompt"],
        }
