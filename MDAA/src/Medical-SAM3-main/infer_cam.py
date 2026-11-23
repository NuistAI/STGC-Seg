import os
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from sam3.model_builder import build_sam3_image_model


IMAGE_DIR = r"/home/Disk3toData/DATAVLM/QaTa-COV19-ori/QaTa-COV19/QaTa-COV19-v2/Train Set/Images"
CSV_PATH = r"/home/Disk3toData/DATAVLM/QaTa-COV19-ori/QaTa-COV19/QaTa-COV19-v2/prompt/Train.csv"
WEIGHT_PATH = "best_model_0.05.pth"

OUTPUT_DIR = r"//home/users/gx202512151439/DATA/QaTa-COV19/QaTa-COV19-v2/Train Set/SAM3_0.05_gradcam"
HEATMAP_DIR_NAME = "heatmap_png"
NPY_DIR_NAME = "npy"

TARGET_LAYER_NAME = None
TARGET_CONV_INDEX = -2
PRINT_CONV_LAYERS = True

BATCH_SIZE = 1
INPUT_SIZE = (1008, 1008)
EPS = 1e-8


class InferDataset(Dataset):
    def __init__(self, image_dir, csv_path):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.samples = []

        for _, row in self.df.iterrows():
            csv_img_name = row.iloc[0]
            prompt = str(row.iloc[1])
            if pd.isna(csv_img_name) or pd.isna(prompt):
                continue

            csv_img_name = str(csv_img_name)
            real_img_name = csv_img_name.replace("mask_", "", 1) if csv_img_name.startswith("mask_") else csv_img_name

            img_path = os.path.join(image_dir, real_img_name)
            if os.path.exists(img_path):
                self.samples.append({
                    "img_path": img_path,
                    "prompt": prompt,
                    "name": real_img_name,
                    "csv_name": csv_img_name,
                })

        print(f"Loaded {len(self.samples)} samples from {image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = Image.open(sample["img_path"]).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize(INPUT_SIZE, Image.Resampling.BILINEAR)

        img = transforms.ToTensor()(img)

        mask = torch.zeros(INPUT_SIZE[1], INPUT_SIZE[0], dtype=torch.bool)
        bbox = torch.tensor([0.0, 0.0, float(INPUT_SIZE[0] - 1), float(INPUT_SIZE[1] - 1)], dtype=torch.float32)

        return {
            "image": img,
            "mask": mask,
            "bbox": bbox,
            "prompt": sample["prompt"],
            "name": sample["name"],
            "orig_size": (orig_w, orig_h),
        }


def infer_collate_fn(batch):
    from sam3.model.data_misc import BatchedDatapoint, BatchedFindTarget, BatchedInferenceMetadata, FindStage

    images, masks, bboxes, texts, names, orig_sizes = [], [], [], [], [], []

    for item in batch:
        images.append(item["image"])
        masks.append(item["mask"])
        bboxes.append(item["bbox"])
        texts.append(item["prompt"])
        names.append(item["name"])
        orig_sizes.append(item["orig_size"])

    img_batch = torch.stack(images)
    bs = len(batch)

    find_stage = FindStage(
        img_ids=torch.arange(bs, dtype=torch.long),
        text_ids=torch.arange(bs, dtype=torch.long),
        input_boxes=torch.zeros(0, bs, 4, dtype=torch.float32),
        input_boxes_label=torch.zeros(0, bs, dtype=torch.long),
        input_boxes_mask=torch.ones(bs, 0, dtype=torch.bool),
        input_points=torch.empty(0, bs, 2, dtype=torch.float32),
        input_points_mask=torch.empty(bs, 0, dtype=torch.bool),
        object_ids=[[0] for _ in range(bs)],
    )

    find_target = BatchedFindTarget(
        num_boxes=torch.ones(bs, dtype=torch.long),
        boxes=torch.stack(bboxes).unsqueeze(1),
        boxes_padded=torch.stack(bboxes).unsqueeze(1),
        is_exhaustive=torch.ones(bs, dtype=torch.bool),
        segments=masks,
        semantic_segments=[],
        is_valid_segment=torch.ones(bs, dtype=torch.long),
        repeated_boxes=[],
        object_ids=[0] * bs,
        object_ids_padded=torch.zeros(bs, 1, dtype=torch.long),
    )

    find_metadata = BatchedInferenceMetadata(
        coco_image_id=torch.arange(bs, dtype=torch.long),
        original_size=[INPUT_SIZE for _ in images],
        object_id=torch.zeros(bs, dtype=torch.long),
        frame_index=torch.zeros(bs, dtype=torch.long),
        original_image_id=torch.arange(bs, dtype=torch.long),
        original_category_id=torch.zeros(bs, dtype=torch.long),
        is_conditioning_only=torch.zeros(bs, dtype=torch.bool),
    )

    batched_datapoint = BatchedDatapoint(
        img_batch=img_batch,
        find_text_batch=texts,
        find_inputs=[find_stage],
        find_targets=[find_target],
        find_metadatas=[find_metadata],
        raw_images=None,
    )

    return batched_datapoint, names, orig_sizes


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import fields
        for field in fields(obj):
            val = getattr(obj, field.name)
            setattr(obj, field.name, move_to_device(val, device))
        return obj
    return obj


def first_tensor_4d(x):
    if isinstance(x, torch.Tensor):
        return x if x.dim() == 4 else None
    if isinstance(x, (list, tuple)):
        for v in x:
            t = first_tensor_4d(v)
            if t is not None:
                return t
    if isinstance(x, dict):
        for v in x.values():
            t = first_tensor_4d(v)
            if t is not None:
                return t
    return None


class GradCAMCapture:
    def __init__(self, target_layer):
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, inputs, output):
        tensor = first_tensor_4d(output)
        if tensor is None:
            self.activations = None
            self.gradients = None
            return

        self.activations = tensor
        self.gradients = None

        if tensor.requires_grad:
            tensor.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    def close(self):
        self.handle.remove()


def list_conv_layers(model):
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    print("\n[Conv2d layers]")
    for i, (name, module) in enumerate(conv_layers):
        print(f"{i:4d}: {name} | {module}")
    print(f"Total Conv2d layers: {len(conv_layers)}\n")
    return conv_layers


def get_all_conv_layers(model):
    """返回所有 Conv2d 层（按深度顺序），用于自动搜索。"""
    return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]


def get_target_layer(model, target_layer_name=None, target_conv_index=-2):
    """获取首选目标层（可能不一定在梯度路径上）。"""
    if target_layer_name is not None and str(target_layer_name).strip() != "":
        modules = dict(model.named_modules())
        if target_layer_name not in modules:
            raise ValueError(f"TARGET_LAYER_NAME not found: {target_layer_name}")
        return target_layer_name, modules[target_layer_name]

    conv_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    if len(conv_layers) == 0:
        raise RuntimeError("No Conv2d layer found. Please set TARGET_LAYER_NAME manually.")

    if not (-len(conv_layers) <= target_conv_index < len(conv_layers)):
        target_conv_index = -1

    name, layer = conv_layers[target_conv_index]
    return name, layer


def extract_out_dict(outputs):
    """尽量兼容 SAM3 不同返回格式，递归找到包含 pred_masks 的 dict。"""
    if isinstance(outputs, dict):
        if "pred_masks" in outputs:
            return outputs
        for v in outputs.values():
            found = extract_out_dict(v)
            if found is not None:
                return found

    if hasattr(outputs, "pred_masks"):
        return {"pred_masks": outputs.pred_masks}

    if hasattr(outputs, "output"):
        found = extract_out_dict(outputs.output)
        if found is not None:
            return found

    if isinstance(outputs, (list, tuple)):
        for v in reversed(outputs):
            found = extract_out_dict(v)
            if found is not None:
                return found

    return None


def build_gradcam_from_capture(capture, target_size):
    if capture.activations is None or capture.gradients is None:
        raise RuntimeError(
            "Grad-CAM capture failed: activations or gradients is None. "
            "Try setting TARGET_LAYER_NAME manually or change TARGET_CONV_INDEX."
        )

    activations = capture.activations
    gradients = capture.gradients

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)

    b = cam.size(0)
    cam_flat = cam.view(b, -1)
    cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    cam = (cam - cam_min) / (cam_max - cam_min + EPS)
    return cam


def build_segmentation_target_score(pred_logits):
    """
    用 SAM 的分割输出作为 Grad-CAM 的目标。
    优先取预测前景区域内的平均 logit；如果没有前景，则退化为全图平均 logit。
    """
    prob = torch.sigmoid(pred_logits)
    fg = (prob > 0.5).float()

    scores = []
    for i in range(pred_logits.size(0)):
        fg_i = fg[i:i + 1]
        logit_i = pred_logits[i:i + 1]
        if fg_i.sum() > 0:
            score_i = (logit_i * fg_i).sum() / (fg_i.sum() + EPS)
        else:
            score_i = logit_i.mean()
        scores.append(score_i)

    return torch.stack(scores).sum()


def apply_jet_colormap(cam_01):
    """
    cam_01: float array, [H,W], range [0,1]
    返回 uint8 RGB JET-like heatmap，不依赖 cv2 / matplotlib。
    """
    x = np.clip(cam_01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def save_cam(cam_1008, orig_size, out_name, heatmap_dir, npy_dir):
    orig_w, orig_h = orig_size
    stem, ext = os.path.splitext(out_name)
    if ext == "":
        ext = ".png"
        out_name = stem + ext

    cam_img = Image.fromarray((cam_1008 * 255).astype(np.uint8))
    cam_img = cam_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    cam_orig = np.asarray(cam_img).astype(np.float32) / 255.0
    cam_orig = np.clip(cam_orig, 0.0, 1.0).astype(np.float32)

    npy_path = os.path.join(npy_dir, stem + ".npy")
    np.save(npy_path, cam_orig)

    heatmap = apply_jet_colormap(cam_orig)
    heatmap_path = os.path.join(heatmap_dir, out_name)
    Image.fromarray(heatmap).save(heatmap_path)


def load_weights_with_shape_fix(model, weight_path, device):
    if not os.path.exists(weight_path):
        print(f"Weight file {weight_path} not found!")
        return False

    state_dict = torch.load(weight_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in model_dict and state_dict[k].shape != model_dict[k].shape:
            print(f"Handling size mismatch for {k}: checkpoint {state_dict[k].shape}, model {model_dict[k].shape}")
            if len(state_dict[k].shape) == 2 and state_dict[k].shape[1] == model_dict[k].shape[1]:
                ckpt_shape = state_dict[k].shape[0]
                model_shape = model_dict[k].shape[0]
                if ckpt_shape < model_shape:
                    new_tensor = torch.zeros_like(model_dict[k])
                    new_tensor[:ckpt_shape] = state_dict[k]
                    state_dict[k] = new_tensor
                else:
                    state_dict[k] = state_dict[k][:model_shape]
            else:
                print(f"Skipping key {k} entirely")
                state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights from {weight_path}")
    return True


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    heatmap_dir = os.path.join(OUTPUT_DIR, HEATMAP_DIR_NAME)
    npy_dir = os.path.join(OUTPUT_DIR, NPY_DIR_NAME)
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    test_dataset = InferDataset(
        image_dir=IMAGE_DIR,
        csv_path=CSV_PATH,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=infer_collate_fn,
    )

    print("Building model...")
    model = build_sam3_image_model(
        device=device,
        eval_mode=False,
        load_from_HF=False,
        enable_inst_interactivity=False,
    )

    if not load_weights_with_shape_fix(model, WEIGHT_PATH, device):
        return

    model.to(device)
    model.eval()

    all_conv = get_all_conv_layers(model)
    if PRINT_CONV_LAYERS:
        list_conv_layers(model)

    candidates = list(enumerate(all_conv))

    preferred_idx = None
    if TARGET_LAYER_NAME is not None and str(TARGET_LAYER_NAME).strip() != "":
        for i, (name, _) in enumerate(all_conv):
            if name == TARGET_LAYER_NAME:
                preferred_idx = i
                break
    elif -len(all_conv) <= TARGET_CONV_INDEX < len(all_conv):
        preferred_idx = TARGET_CONV_INDEX % len(all_conv)

    if preferred_idx is not None:
        rest = [c for i, c in enumerate(candidates) if i != preferred_idx]
        ordered = [candidates[preferred_idx]] + list(reversed(rest))
    else:
        ordered = list(reversed(candidates))

    working_name = None
    working_layer = None
    first_batch_data = next(iter(test_loader))
    batched_datapoint, names, orig_sizes = first_batch_data
    batched_datapoint = move_to_device(batched_datapoint, device)
    if hasattr(batched_datapoint, "img_batch"):
        batched_datapoint.img_batch.requires_grad_(True)

    for _, (conv_name, conv_module) in ordered:
        capture = GradCAMCapture(conv_module)
        model.zero_grad(set_to_none=True)

        outputs = model(batched_datapoint)
        out_dict = extract_out_dict(outputs)
        if out_dict is None or "pred_masks" not in out_dict:
            capture.close()
            continue

        pred_logits = out_dict["pred_masks"][:, 0:1, :, :]
        if pred_logits.shape[-2:] != INPUT_SIZE:
            pred_logits_for_score = F.interpolate(
                pred_logits, size=INPUT_SIZE, mode="bilinear", align_corners=False,
            )
        else:
            pred_logits_for_score = pred_logits

        score = build_segmentation_target_score(pred_logits_for_score)
        score.backward()

        if capture.activations is not None and capture.gradients is not None:
            working_name = conv_name
            working_layer = conv_module
            print(f"[Auto-select] Grad-CAM target layer: {working_name}")
            capture.close()
            break

        capture.close()
        print(f"  Skip {conv_name}: activations={capture.activations is not None}, gradients={capture.gradients is not None}")

    if working_layer is None:
        print("WARNING: No Conv2d layer yields gradients. Falling back to input-image Grad-CAM.")
        working_name = "__input_image__"

    use_input_fallback = (working_name == "__input_image__")
    capture = None if use_input_fallback else GradCAMCapture(working_layer)

    print("Starting inference and saving Grad-CAM heatmaps + npy files...")
    try:
        for batched_datapoint, names, orig_sizes in tqdm(test_loader):
            batched_datapoint = move_to_device(batched_datapoint, device)

            if hasattr(batched_datapoint, "img_batch"):
                batched_datapoint.img_batch.requires_grad_(True)

            model.zero_grad(set_to_none=True)
            if not use_input_fallback:
                capture.activations = None
                capture.gradients = None

            outputs = model(batched_datapoint)
            out_dict = extract_out_dict(outputs)
            if out_dict is None or "pred_masks" not in out_dict:
                print("Warning: pred_masks not found, skip this batch.")
                continue

            pred_logits = out_dict["pred_masks"][:, 0:1, :, :]
            if pred_logits.shape[-2:] != INPUT_SIZE:
                pred_logits_for_score = F.interpolate(
                    pred_logits, size=INPUT_SIZE, mode="bilinear", align_corners=False,
                )
            else:
                pred_logits_for_score = pred_logits

            score = build_segmentation_target_score(pred_logits_for_score)
            score.backward()

            if use_input_fallback:
                cams = batched_datapoint.img_batch.grad
                if cams is None:
                    print("Warning: input grad is None, skip this batch.")
                    continue
                cams = cams.mean(dim=1, keepdim=True)
                cams = F.relu(cams)
                cams = F.interpolate(cams, size=INPUT_SIZE, mode="bilinear", align_corners=False)
                b = cams.size(0)
                cams_flat = cams.view(b, -1)
                cams_min = cams_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
                cams_max = cams_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
                cams = (cams - cams_min) / (cams_max - cams_min + EPS)
                cams_np = cams.detach().cpu().numpy()
            else:
                cams = build_gradcam_from_capture(capture, target_size=INPUT_SIZE)
                cams_np = cams.detach().cpu().numpy()

            for i in range(cams_np.shape[0]):
                cam_i = cams_np[i, 0]
                save_cam(
                    cam_1008=cam_i,
                    orig_size=orig_sizes[i],
                    out_name=names[i],
                    heatmap_dir=heatmap_dir,
                    npy_dir=npy_dir,
                )

            del outputs, out_dict, pred_logits, pred_logits_for_score, score
            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:
        if capture is not None:
            capture.close()

    print("Grad-CAM export complete!")
    print(f"Heatmaps saved to: {heatmap_dir}")
    print(f"Npy files saved to: {npy_dir}")


if __name__ == "__main__":
    main()
