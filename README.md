# STGC-Seg

PyTorch implementation accompanying **STGC-Seg: Spatialized Text Guidance and Collaborative Knowledge Transfer for Semi-Supervised Pneumonia Lesion Segmentation**.

STGC-Seg combines spatial text guidance with a dual-student, single-Teacher framework. Student-A learns from reliable Teacher predictions, and Student-B learns from a frozen, domain-adapted Medical SAM3 expert. The students also supervise each other through cross-pseudo supervision. The final Teacher predicts from the medical image alone.

## Manuscript results

The main comparison tables report the following results at 5%, 10%, and 25% labeled data. mDice and mIoU are per-image means, while gDice and gIoU aggregate pixels across the test set. HD95 and ASSD are boundary-distance measures.

| Dataset | Labeled data | mDice (%) | mIoU (%) | gDice (%) | gIoU (%) | HD95 | ASSD |
|---|---:|---:|---:|---:|---:|---:|---:|
| QaTa-COV19 | 5% | 78.83 | 69.24 | 87.16 | 77.25 | 29.78 | 8.18 |
| MosMedData+ | 5% | 68.44 | 54.75 | 76.41 | 61.83 | 29.55 | 7.95 |
| QaTa-COV19 | 10% | 80.34 | 71.19 | 88.31 | 79.07 | 28.17 | 7.47 |
| MosMedData+ | 10% | 71.19 | 57.62 | 77.45 | 63.20 | 19.39 | 4.54 |
| QaTa-COV19 | 25% | 80.90 | 72.12 | 88.62 | 79.57 | 27.43 | 7.48 |
| MosMedData+ | 25% | 72.19 | 58.66 | 77.92 | 63.82 | 18.32 | 4.23 |

The manuscript reports five-fold means separately in its label-ratio analysis. This checkout provides segmentation-stage code and configurations. Reproducing the table requires the corresponding split manifests, priors, checkpoints, and evaluation protocol.

## Method overview

The Teacher and both students use the same segmentation architecture, with a shared image encoder and complementary foreground and background decoders. The supplied segmentation configurations use ConvNeXt-Tiny.

| Component | Full name | Role |
|---|---|---|
| URS | Uncertainty-aware Region Selection | Selects reliable Teacher pseudo-labels for Student-A using agreement and relative confidence between the two decoders. |
| CPS | Cross-pseudo supervision | Lets the two students supervise each other with hard pseudo-labels from their predictions. |
| TSL | Text-guided Semantic Localization | Uses FiLM to turn paired images and diagnostic descriptions into image-aligned class activation maps (CAMs). Grad-CAM from the adapted expert provides training targets for TSL. |
| SCA | Synchronized Cross-modal Augmentation | Aligns images, masks, and CAMs under shared geometric transformations. The CAMs provide pixel-wise weighting for foreground training losses. |
| ASR | Adaptive Semantic Refinement | Refreshes the stored CAMs from Teacher-derived Grad-CAM estimates during semi-supervised training. |
| MDAA | Medical Domain Adaptation Adapter | Adapts Medical SAM3 with shallow spatial adapters, deep semantic adapters, and a prompt adapter. The frozen expert provides pseudo-labels for Student-B. |
| WEMA | Weighted Exponential Moving Average | Aggregates the updated Student-A and Student-B parameters into the Teacher, with weights of 0.75 and 0.25. |

The manuscript pipeline first adapts Medical SAM3, trains TSL, and initializes the segmentation network on labeled images and their spatial priors. During semi-supervised training, both students also learn from the available ground-truth masks. At inference, only the WEMA Teacher is used.

## Repository layout

```text
config/
data/
engine/
models/
modules/
TSL/
train.py
evaluate.py
requirements.txt
```

The main segmentation stage starts in [train.py](train.py). [TSL/](TSL/) contains TSL training and CAM export scripts. The outer Git tree records `MDAA` as a separate commit reference. A checkout with the MDAA sources includes `MDAA/src/train_mdaa.py`, `MDAA/src/evaluate_extended_metrics.py`, and the adapted Medical SAM3 source tree.

## 1. Environment

Python 3.10 is recommended. The supplied training and evaluation commands use an NVIDIA GPU. Install a CUDA-compatible PyTorch build for your system before installing the remaining dependencies.

```bash
conda create -n stgcseg python=3.10 -y
conda activate stgcseg
pip install -r requirements.txt
```

The supplied segmentation configurations reference `facebook/convnext-tiny-224` and `microsoft/BiomedVLP-CXR-BERT-specialized`. For offline runs, set `vision_type` and `bert_type` to the corresponding local model directories. Resolve relative paths from the repository root.

## 2. Dataset preparation

The manuscript uses QaTa-COV19 (chest radiographs) and MosMedData+ (CT slices) for in-domain experiments. It reports direct-transfer evaluation on COVID-QU-Ex and KISEG. Diagnostic descriptions for the in-domain datasets come from LViT.

| Dataset | Training | Validation | Test |
|---|---:|---:|---:|
| QaTa-COV19 | 5,716 | 1,429 | 2,113 |
| MosMedData+ | 2,183 | 273 | 273 |

Obtain the images and masks from the original providers. For each of `datasets/qata` and `datasets/mosmed`, use the following directory layout or edit the dataset paths in the YAML.

```text
prompt/
  Train.csv
  Test.csv
Train Set/
  Images/
  GTs/
Test Set/
  Images/
  GTs/
```

Each training CSV needs an `Image` column and a diagnostic-text column. The segmentation configs use `Description` for QaTa-COV19 and `text` for MosMedData+. For example, an `Image` entry of `mask_case_0001.png` maps to `Images/case_0001.png` and `GTs/mask_case_0001.png`. Masks can use values of 0 and 1 or 0 and 255. The segmentation stage resizes images and masks to 224 × 224.

## 3. Training priors

Prepare an initial TSL CAM and a frozen MDAA-adapted Medical SAM3 mask for each training sample required by the configuration. Strict loading also requires the configured priors for validation rows read from the same training CSV.

- **TSL CAMs.** Train TSL with [TSL/train_tsl.py](TSL/train_tsl.py) using the corresponding dataset config in [TSL/configs](TSL/configs). Set `DATA.train_cam_path` to the expert Grad-CAM targets. Export CAMs with [TSL/export_tsl_cam.py](TSL/export_tsl_cam.py). Pass the training CSV and image root using `--csv_path` and `--root_path` because the exporter otherwise uses the test paths from its config. Point `DATA.cam_prior_path` in the segmentation YAML to the export's `npy/` directory.
- **MDAA expert masks.** In a checkout containing the MDAA sources, `MDAA/src/train_mdaa.py` trains the expert adapters. Supply the exported masks from the frozen adapted expert in `DATA.sam_prior_path`. The segmentation entry point consumes these masks as files.

The loader supports single-channel NumPy arrays (`.npy` or `.npz`) and images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`). For `.npz`, it prefers the `cam` or `sam` key and falls back to the first array. For `mask_case_0001.png`, use `case_0001.npy` for the corresponding prior. The supplied segmentation configs use strict prior loading, which reports missing files instead of proceeding with empty priors.

## 4. Segmentation training

The following settings describe the supplied segmentation configurations.

| Setting | Value |
|---|---:|
| Image size | 224 × 224 |
| Batch size | 8 |
| Optimizer and initial learning rate | AdamW, `1e-4` |
| URS agreement threshold | `0.40` |
| URS conflict threshold | `0.80` |
| URS confidence margin | `0.10` |
| Random seed | `0` (set in `train.py`) |
| ASR warm-up | 10 epochs |
| ASR refresh interval | 10 epochs |
| WEMA Student-A weight | `0.75` |
| EMA decay | `0.99` |
| CPS loss weight | `1.0` |
| Maximum epochs per stage | 200 |
| Early-stopping patience | 20 |

Edit the data, prior, and output paths and the GPU index in the selected config. Leave the test-prior paths empty to run the final test in `train.py` without loading those priors. Edit these keys in the existing `DATA` section.

```yaml
DATA:
  test_cam_prior_path: ''
  test_sam_prior_path: ''
```

Run the command for your dataset.

```bash
python train.py --config config/qata.yaml
python train.py --config config/mosmed.yaml
```

The entry point initializes the segmentation network on the labeled subset, initializes the students and Teacher from that checkpoint, and runs semi-supervised training with the prepared priors. The trainer selects checkpoints by validation loss. By default, `cast_seg_pretrain.ckpt` is stored under `outputs/<dataset>/pretrain/` and `cast_seg.ckpt` is stored under `outputs/<dataset>/semi_supervised/`. ASR writes refreshed CAM banks and manifests under `outputs/<dataset>/asr_cam_priors/`. Lightning logs are written under `lightning_logs/`.

To start semi-supervised training from an existing supervised checkpoint, set `pretrain: False` and set `pretrained_ckpt_path` to that file in the `TRAIN` section.

For the separate label-ratio analysis, the loader supports `num_folds: 5` and `fold_index` from 0 through 4. Check the exact fold membership and use separate checkpoint and ASR output directories for each fold.

## 5. Image-only Teacher evaluation

The manuscript averages the Teacher's foreground prediction and the inverted background prediction. The supplied image-only evaluation uses a decision threshold of 0.5.

```math
P_{\mathrm{final}} = \frac{P_f + (1-P_b)}{2},
\qquad
\hat{Y}_{\mathrm{final}} = \mathbf{1}[P_{\mathrm{final}} > 0.5].
```

Use a Teacher checkpoint selected on the validation split. The ground-truth test masks are used to score the predictions.

```bash
python evaluate.py --config config/qata.yaml --ckpt outputs/qata/semi_supervised/cast_seg.ckpt --disable-cam-conflict-fusion
python evaluate.py --config config/mosmed.yaml --ckpt outputs/mosmed/semi_supervised/cast_seg.ckpt --disable-cam-conflict-fusion
```

These commands use the image-only Lightning test path. It reports accuracy, global Dice, and global IoU in the terminal. The full manuscript table also requires per-image Dice and IoU and the HD95 and ASSD boundary metrics. The image-only path does not currently export that six-metric summary as JSON.

Use `--disable-cam-conflict-fusion` for manuscript-style evaluation. The legacy CAM arbitration branch uses a different prediction rule, and with the supplied dataset layout its batch parser can treat the ground-truth mask as the arbitration CAM. Its metrics and JSON should not be used for manuscript comparisons.

## Release contents

Raw datasets, pretrained and trained checkpoints, generated CAM banks, expert masks, and experiment outputs are prepared separately. The main training command assumes the initial priors already exist. ASR refreshes the CAM bank during semi-supervised training.

Before a full run, verify the dataset and prior paths against the split manifests, check CUDA availability, and use separate output directories for each dataset, label ratio, fold, and seed.

## Citation

```bibtex
@article{gu2026stgcseg,
  title  = {STGC-Seg: Spatialized Text Guidance and Collaborative Knowledge Transfer for Semi-Supervised Pneumonia Lesion Segmentation},
  author = {Gu, Xun and Zhang, Zichen and Yang, Fan and Huang, Yan and Chen, Yunjie},
  year   = {2026},
  note   = {Manuscript}
}
```

## License

The repository currently has no top-level license. External datasets, pretrained models, and third-party sources are governed by their respective licenses.
