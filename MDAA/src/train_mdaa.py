"""Train the paper MDAA Full model on Medical SAM3.

Only the released Full path is supported: shallow spatial-aware adapters,
deep semantic adapters, learnable residual scales, and the text/prompt adapter
applied after Medical SAM3 text encoding.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset_medical import MedicalTextSegDataset, medical_collate
from sam3.model_builder import build_sam3_image_model
from sam3.train.loss.loss_fns import (
    CORE_LOSS_KEY,
    Boxes,
    IABCEMdetr,
    SemanticSegCriterion,
)
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryOneToManyMatcher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--max-test-steps", type=int, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, list):
        return [move_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(value, device) for value in obj)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        for field in fields(obj):
            setattr(obj, field.name, move_to_device(getattr(obj, field.name), device))
        return obj
    return obj


def last_output(outputs):
    if isinstance(outputs, dict):
        return outputs
    try:
        candidate = outputs[0]
    except Exception as exc:
        raise TypeError(f"Unsupported model output type: {type(outputs)}") from exc
    if isinstance(candidate, dict):
        return candidate
    if isinstance(candidate, (list, tuple)) and candidate and isinstance(candidate[-1], dict):
        return candidate[-1]
    raise TypeError(f"Could not find output dictionary in {type(outputs)}")


def build_loss(model):
    find_losses = [
        Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
        IABCEMdetr(
            weak_loss=False,
            weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
            pos_weight=10.0,
            alpha=0.25,
            gamma=2.0,
            use_presence=True,
            pos_focal=False,
            pad_n_queries=200,
        ),
    ]
    semantic_loss = SemanticSegCriterion(
        weight_dict={"loss_semantic_seg": 20.0, "loss_semantic_dice": 30.0},
        focal=True,
        focal_alpha=0.6,
        focal_gamma=2.0,
        downsample=False,
        presence_head=False,
        presence_loss=False,
    )
    return Sam3LossWrapper(
        loss_fns_find=find_losses,
        normalization="local",
        matcher=model.matcher,
        o2m_matcher=BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4),
        o2m_weight=2.0,
        use_o2m_matcher_on_o2m_aux=False,
        loss_fn_semantic_seg=semantic_loss,
        scale_by_find_batch_size=True,
        normalize_by_stage_num=True,
    )


def get_vit(model):
    return model.backbone.vision_backbone.trunk


def is_residual_scale_parameter(name):
    return name.endswith(".alpha_attn") or name.endswith(".alpha_mlp")


def make_model_adapter_config(adapter_config):
    """Translate the paper-facing MDAA config to the upstream SAM3 API."""
    model_adapter_config = dict(adapter_config)
    learnable_scale = model_adapter_config.pop("learnable_residual_scale", None)
    if learnable_scale is None:
        raise ValueError("adapter.learnable_residual_scale must be specified")
    model_adapter_config["use_layer_gate"] = bool(learnable_scale)
    return model_adapter_config


def apply_mdaa_training(model, strategy):
    if strategy != "mdaa_full":
        raise ValueError(
            f"This public entry point only supports strategy='mdaa_full', got {strategy!r}."
        )
    prompt_adapter = getattr(model, "HyP_Adapter", None)
    if prompt_adapter is None:
        raise RuntimeError(
            "Medical SAM3 does not expose HyP_Adapter after text encoding; "
            "the paper MDAA prompt adapter is missing."
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("Adapter" in name or is_residual_scale_parameter(name))

    vit = get_vit(model)
    if len(vit.blocks) != 32:
        raise RuntimeError(f"Expected current ViT-H depth=32, got {len(vit.blocks)}")
    adapter_kinds = [block.adapter_kind for block in vit.blocks]
    if adapter_kinds[:16] != ["spatial"] * 16 or adapter_kinds[16:] != ["semantic"] * 16:
        raise RuntimeError(
            "Full MDAA requires spatial adapters in blocks 1-16 and semantic "
            f"adapters in blocks 17-32, got {adapter_kinds}"
        )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError("No Adapter/residual-scale parameters were selected for training")
    invalid = [
        name
        for name in trainable_names
        if "Adapter" not in name and not is_residual_scale_parameter(name)
    ]
    if invalid:
        raise RuntimeError(f"Non-Adapter backbone parameters became trainable: {invalid}")
    prompt_trainable = [
        name for name in trainable_names if name.startswith("HyP_Adapter.")
    ]
    if not prompt_trainable:
        raise RuntimeError("The text/prompt HyP_Adapter is not trainable in MDAA Full.")


def parameter_audit(model, optimizer, strategy, output_path):
    named = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in named)
    trainable = sum(parameter.numel() for _, parameter in named if parameter.requires_grad)
    frozen = total - trainable
    trainable_names = [name for name, parameter in named if parameter.requires_grad]
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_ids = {id(parameter) for _, parameter in named if parameter.requires_grad}
    missing_from_optimizer = [
        name for name, parameter in named if parameter.requires_grad and id(parameter) not in optimizer_ids
    ]
    frozen_in_optimizer = [
        name for name, parameter in named if not parameter.requires_grad and id(parameter) in optimizer_ids
    ]
    encoder_in_optimizer = [
        name for name, parameter in named if name.startswith("backbone.") and id(parameter) in optimizer_ids
    ]
    head_in_optimizer = [
        name for name, parameter in named if name.startswith("segmentation_head.") and id(parameter) in optimizer_ids
    ]
    if optimizer_ids != trainable_ids or missing_from_optimizer or frozen_in_optimizer:
        raise RuntimeError("Optimizer parameter registration does not match requires_grad")
    adapter_names = [name for name in trainable_names if "Adapter" in name]
    residual_scale_names = [
        name for name in trainable_names if is_residual_scale_parameter(name)
    ]
    spatial_names = [name for name in adapter_names if ".depthwise_conv." in name]
    prompt_names = [name for name in adapter_names if name.startswith("HyP_Adapter.")]
    adapter_parameters = sum(
        parameter.numel()
        for name, parameter in named
        if parameter.requires_grad and "Adapter" in name
    )
    residual_scale_parameters = sum(
        parameter.numel()
        for name, parameter in named
        if parameter.requires_grad and is_residual_scale_parameter(name)
    )
    spatial_parameters = sum(
        parameter.numel()
        for name, parameter in named
        if parameter.requires_grad and ".depthwise_conv." in name
    )
    prompt_parameters = sum(
        parameter.numel()
        for name, parameter in named
        if parameter.requires_grad and name.startswith("HyP_Adapter.")
    )
    if any(
        "Adapter" not in name and not is_residual_scale_parameter(name)
        for name in trainable_names
    ):
        raise RuntimeError("MDAA Full contains unexpected trainable parameters")
    if not adapter_names:
        raise RuntimeError("MDAA Full has no trainable Adapter parameters")
    if not prompt_names or prompt_parameters <= 0:
        raise RuntimeError("MDAA Full has no trainable text/prompt Adapter parameters")
    audit = {
        "strategy": strategy,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_ratio": trainable / total,
        "optimizer_parameter_tensors": len(optimizer_ids),
        "trainable_parameter_tensors": len(trainable_ids),
        "encoder_parameter_tensors_in_optimizer": len(encoder_in_optimizer),
        "head_parameter_tensors_in_optimizer": len(head_in_optimizer),
        "adapter_parameters": adapter_parameters,
        "residual_scale_parameters": residual_scale_parameters,
        "spatial_depthwise_parameters": spatial_parameters,
        "prompt_adapter_parameters": prompt_parameters,
        "adapter_parameter_tensors": len(adapter_names),
        "residual_scale_parameter_tensors": len(residual_scale_names),
        "spatial_parameter_tensors": len(spatial_names),
        "prompt_adapter_parameter_tensors": len(prompt_names),
        "trainable_parameter_names": trainable_names,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"PARAM_AUDIT total={total} trainable={trainable} frozen={frozen} "
        f"ratio={trainable / total:.8%}",
        flush=True,
    )
    print(
        f"OPTIMIZER_AUDIT trainable_tensors={len(trainable_ids)} optimizer_tensors={len(optimizer_ids)} "
        f"encoder_in_optimizer={len(encoder_in_optimizer)} head_in_optimizer={len(head_in_optimizer)}",
        flush=True,
    )
    print(
        f"ADAPTER_AUDIT adapter={adapter_parameters} "
        f"residual_scale={residual_scale_parameters} "
        f"spatial_depthwise={spatial_parameters} prompt={prompt_parameters}",
        flush=True,
    )
    print("TRAINABLE_MODULE_PREFIXES " + ", ".join(sorted({name.split(".")[0] for name in trainable_names})), flush=True)
    for name in trainable_names[:80]:
        print(f"TRAINABLE_PARAM {name}", flush=True)
    if len(trainable_names) > 80:
        print(f"TRAINABLE_PARAM ... {len(trainable_names) - 80} additional names saved in {output_path}", flush=True)
    return audit


def make_loader(cfg, split):
    data_cfg = cfg["data"][split]
    dataset = MedicalTextSegDataset(
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        csv_path=data_cfg["csv_path"],
        image_size=cfg["training"]["image_size"],
        labeled_ratio=cfg["training"]["labeled_ratio"] if split == "train" else 1.0,
        seed=cfg["training"]["seed"],
        split=split,
    )
    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=split == "train",
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=False,
        collate_fn=medical_collate,
        persistent_workers=cfg["training"]["num_workers"] > 0,
    )


def semantic_logits(out_dict, target_size):
    if "semantic_seg" not in out_dict:
        raise KeyError(f"semantic_seg missing; keys={sorted(out_dict)}")
    logits = out_dict["semantic_seg"]
    if logits.shape[-2:] != target_size:
        logits = F.interpolate(logits.float(), size=target_size, mode="bilinear", align_corners=False)
    return logits


def batch_metrics(logits, targets):
    pred = logits.sigmoid().squeeze(1) > 0.5
    target = targets.bool()
    dims = tuple(range(1, pred.ndim))
    intersection = (pred & target).sum(dim=dims).double()
    pred_sum = pred.sum(dim=dims).double()
    target_sum = target.sum(dim=dims).double()
    union = (pred | target).sum(dim=dims).double()
    dice = torch.where(pred_sum + target_sum > 0, 2 * intersection / (pred_sum + target_sum), torch.ones_like(intersection))
    iou = torch.where(union > 0, intersection / union, torch.ones_like(intersection))
    return dice.cpu().tolist(), iou.cpu().tolist()


@torch.no_grad()
def evaluate(model, loader, device, max_steps=None):
    model.eval()
    dices, ious = [], []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = move_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            outputs = model(batch)
            out_dict = last_output(outputs)
            targets = batch.find_targets[0].semantic_segments
            logits = semantic_logits(out_dict, targets.shape[-2:])
        dice, iou = batch_metrics(logits.float(), targets)
        dices.extend(dice)
        ious.extend(iou)
    if not dices:
        raise RuntimeError("Evaluation produced no batches")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "n": len(dices),
        "elapsed_seconds": elapsed,
        "milliseconds_per_image": 1000.0 * elapsed / len(dices),
    }


def save_checkpoint(path, model, epoch, metrics, cfg, audit):
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state = {
        "checkpoint_format": "trainable_parameters_only_v1",
        "trainable_model": trainable_state,
        "epoch": epoch,
        "metrics": metrics,
        "config": cfg,
        "parameter_audit": {k: v for k, v in audit.items() if k != "trainable_parameter_names"},
    }
    torch.save(state, path)


def load_trainable_checkpoint(path, model):
    state = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state["trainable_model"], strict=False)
    supplied = set(state["trainable_model"])
    unexpected = [name for name in incompatible.unexpected_keys if name in supplied]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable checkpoint keys: {unexpected}")
    return state


def residual_scale_snapshot(model):
    rows = []
    for index, block in enumerate(get_vit(model).blocks):
        rows.append(
            {
                "block_id": index + 1,
                "adapter_kind": block.adapter_kind,
                "adapter_enabled": bool(block.adapter_enabled),
                "attn_scale": float(block.alpha_attn.detach().cpu()),
                "mlp_scale": float(block.alpha_mlp.detach().cpu()),
            }
        )
    return rows


def export_residual_scales(model, result_dir):
    rows = residual_scale_snapshot(model)
    csv_path = result_dir / "layer_residual_scale.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blocks = [row["block_id"] for row in rows]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(blocks, [row["attn_scale"] for row in rows], marker="o", label="Attention")
    axis.plot(blocks, [row["mlp_scale"] for row in rows], marker="s", label="MLP")
    axis.axvline(16.5, color="0.5", linestyle="--", linewidth=1, label="shallow/deep")
    axis.set_xlabel("Transformer block index")
    axis.set_ylabel("Learned residual scale")
    axis.set_xticks(range(1, len(rows) + 1, 2))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "layer_residual_scale.png", dpi=200)
    plt.close(fig)
    return rows


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_cfg = cfg["training"]
    adapter_cfg = cfg["adapter"]
    output_dir = Path(cfg["output_dir"])
    result_dir = Path(cfg["result_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    if cfg.get("strategy") != "mdaa_full":
        raise ValueError("Only the paper Full MDAA strategy is retained in this release")
    if not bool(cfg.get("prompt_adapter", False)):
        raise ValueError("prompt_adapter must be true for the paper Full MDAA")
    required_full = {
        "hierarchical_adapter": True,
        "learnable_residual_scale": True,
        "adapter_policy": "all",
    }
    for key, expected in required_full.items():
        if adapter_cfg.get(key) != expected:
            raise ValueError(
                f"Full MDAA requires adapter.{key}={expected!r}, "
                f"got {adapter_cfg.get(key)!r}"
            )
    if train_cfg["labeled_ratio"] != 0.04:
        raise ValueError("MDAA Full is locked to code labeled_ratio=0.04 (paper 5% setting)")
    if train_cfg["image_size"] != 1008:
        raise ValueError("Image size is locked to 1008")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; do not run long training on a login CPU")

    set_seed(int(train_cfg["seed"]))
    device = torch.device("cuda")
    print(json.dumps({
        "config": str(config_path),
        "experiment_name": cfg["experiment_name"],
        "dataset": cfg["dataset"],
        "strategy": cfg["strategy"],
        "adapter": adapter_cfg,
        "paper_setting": "5% labeled data",
        "code_labeled_ratio": train_cfg["labeled_ratio"],
        "batch_size": train_cfg["batch_size"],
        "epochs": train_cfg["epochs"],
        "early_stopping_patience": train_cfg.get("early_stopping_patience"),
        "optimizer": "AdamW",
        "learning_rate": train_cfg["learning_rate"],
        "lr_schedule": "constant",
        "image_size": train_cfg["image_size"],
        "shard": int(os.environ.get("EXPERIMENT_SHARD", "2")),
        "dry_run": args.dry_run,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
    }, indent=2), flush=True)

    train_loader = make_loader(cfg, "train")
    val_loader = make_loader(cfg, "val")
    test_loader = make_loader(cfg, "test")

    checkpoint = Path(cfg["pretrained_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    print(f"PRETRAINED_CHECKPOINT {checkpoint} bytes={checkpoint.stat().st_size}", flush=True)
    model = build_sam3_image_model(
        device="cuda",
        eval_mode=False,
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
        context_length=32,
        adapter_config=make_model_adapter_config(adapter_cfg),
    )
    apply_mdaa_training(model, cfg["strategy"])
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    audit = parameter_audit(model, optimizer, cfg["strategy"], result_dir / "parameter_audit.json")
    criterion = build_loss(model).to(device)
    print(
        "VIT_AUDIT "
        + json.dumps(
            {
                "class": type(get_vit(model)).__name__,
                "block_class": type(get_vit(model).blocks[0]).__name__,
                "blocks": len(get_vit(model).blocks),
                "feature_layout": "BHWC inside ViT; BCHW after trunk",
                "embed_dim": 1024,
                "bottleneck_rank": 256,
                "activation": "GELU",
                "fixed_attn_scale": adapter_cfg["attn_adapter_scale"],
                "fixed_mlp_scale": adapter_cfg["mlp_adapter_scale"],
                "spatial_layers": adapter_cfg["spatial_layers"],
                "prompt_adapter": type(model.HyP_Adapter).__name__,
            }
        ),
        flush=True,
    )

    epochs = 1 if args.dry_run else int(train_cfg["epochs"])
    early_stopping_patience = train_cfg.get("early_stopping_patience")
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
        if early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be a positive integer")
    max_train_steps = args.max_train_steps if args.dry_run else None
    max_val_steps = args.max_val_steps if args.dry_run else None
    best_dice = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    last_epoch = 0
    history_path = result_dir / ("dry_run_history.csv" if args.dry_run else "training_history.csv")
    residual_scale_initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and is_residual_scale_parameter(name)
    }
    adapter_probe_name, adapter_probe_initial = next(
        (
            (name, parameter.detach().cpu().clone())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "Adapter" in name
        )
    )
    residual_scale_update_verified = not residual_scale_initial
    adapter_update_verified = False
    torch.cuda.reset_peak_memory_stats(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_dice", "val_iou", "val_n"])
        writer.writeheader()
        for epoch in range(epochs):
            last_epoch = epoch + 1
            model.train()
            losses_for_epoch = []
            for step, batch in enumerate(train_loader):
                if max_train_steps is not None and step >= max_train_steps:
                    break
                batch = move_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    outputs = model(batch)
                    converted_targets = [
                        model.back_convert(target) for target in batch.find_targets
                    ]
                    loss_dict = criterion(outputs, converted_targets)
                    loss = loss_dict[CORE_LOSS_KEY]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch={epoch} step={step}: {loss}")
                loss.backward()
                if step == 0:
                    grad_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is not None]
                    bad_grads = [name for name, parameter in model.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
                    print(f"GRAD_AUDIT gradients={len(grad_names)} nonfinite={len(bad_grads)} first={grad_names[:12]}", flush=True)
                    if not grad_names or bad_grads:
                        raise RuntimeError("Gradient audit failed")
                torch.nn.utils.clip_grad_norm_(parameters, float(train_cfg["gradient_clip_norm"]))
                optimizer.step()
                if epoch == 0 and step == 0:
                    named_parameters = dict(model.named_parameters())
                    adapter_update_verified = not torch.equal(
                        adapter_probe_initial,
                        named_parameters[adapter_probe_name].detach().cpu(),
                    )
                    residual_scale_update_verified = not residual_scale_initial or all(
                        not torch.equal(before, named_parameters[name].detach().cpu())
                        for name, before in residual_scale_initial.items()
                    )
                    print(
                        f"UPDATE_AUDIT adapter_updated={adapter_update_verified} "
                        f"residual_scale_updated={residual_scale_update_verified} "
                        f"adapter_probe={adapter_probe_name}",
                        flush=True,
                    )
                    if not adapter_update_verified or not residual_scale_update_verified:
                        raise RuntimeError("Adapter/residual-scale optimizer update audit failed")
                losses_for_epoch.append(float(loss.detach().cpu()))
                print(f"TRAIN epoch={epoch + 1}/{epochs} step={step + 1}/{len(train_loader)} loss={losses_for_epoch[-1]:.8f}", flush=True)

            val_metrics = evaluate(model, val_loader, device, max_steps=max_val_steps)
            row = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses_for_epoch)),
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_n": val_metrics["n"],
            }
            writer.writerow(row)
            stream.flush()
            print(f"VALID {json.dumps(row)}", flush=True)
            if not args.dry_run:
                if val_metrics["dice"] > best_dice:
                    best_dice = val_metrics["dice"]
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    save_checkpoint(output_dir / "best.pt", model, best_epoch, val_metrics, cfg, audit)
                elif early_stopping_patience is not None:
                    epochs_without_improvement += 1
                    print(
                        f"EARLY_STOP_STATUS epoch={epoch + 1} best_epoch={best_epoch} "
                        f"epochs_without_improvement={epochs_without_improvement}/"
                        f"{early_stopping_patience}",
                        flush=True,
                    )
                    if epochs_without_improvement >= early_stopping_patience:
                        print(
                            f"EARLY_STOP_TRIGGERED epoch={epoch + 1} best_epoch={best_epoch} "
                            f"best_val_dice={best_dice:.8f} patience={early_stopping_patience}",
                            flush=True,
                        )
                        break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    peak_gpu_bytes = torch.cuda.max_memory_allocated(device)

    if not args.dry_run:
        save_checkpoint(output_dir / "last.pt", model, last_epoch, val_metrics, cfg, audit)

    if args.dry_run:
        test_metrics = evaluate(model, test_loader, device, max_steps=args.max_test_steps or 1)
        dry_result = {
            "status": "success",
            "train_steps": len(losses_for_epoch),
            "val": val_metrics,
            "test": test_metrics,
            "adapter_updated": adapter_update_verified,
            "residual_scale_updated": residual_scale_update_verified,
            "peak_gpu_memory_bytes": peak_gpu_bytes,
            "parameter_audit": {k: v for k, v in audit.items() if k != "trainable_parameter_names"},
        }
        (result_dir / "dry_run_success.json").write_text(json.dumps(dry_result, indent=2), encoding="utf-8")
        print(f"DRY_RUN_SUCCESS {json.dumps(dry_result)}", flush=True)
        return

    load_trainable_checkpoint(output_dir / "best.pt", model)
    model.to(device)
    test_metrics = evaluate(model, test_loader, device)
    residual_scale_rows = (
        export_residual_scales(model, result_dir) if residual_scale_initial else []
    )
    final = {
        "dataset": cfg["dataset"],
        "fine_tuning_strategy": cfg["strategy"],
        "labeled_data_setting": "paper 5%; code labeled_ratio=0.04",
        "dice": test_metrics["dice"],
        "iou": test_metrics["iou"],
        "hd95": None,
        "hd95_note": "use evaluate_extended_metrics.py for boundary metrics",
        "test_n": test_metrics["n"],
        "inference_seconds": test_metrics["elapsed_seconds"],
        "inference_milliseconds_per_image": test_metrics["milliseconds_per_image"],
        "training_seconds": training_seconds,
        "peak_gpu_memory_bytes": peak_gpu_bytes,
        "best_epoch": best_epoch,
        "stopped_epoch": last_epoch,
        "early_stopping_patience": early_stopping_patience,
        "trainable_parameters": audit["trainable_parameters"],
        "total_parameters": audit["total_parameters"],
        "adapter_parameters": audit["adapter_parameters"],
        "residual_scale_parameters": audit["residual_scale_parameters"],
        "spatial_depthwise_parameters": audit["spatial_depthwise_parameters"],
        "prompt_adapter_parameters": audit["prompt_adapter_parameters"],
        "adapter_updated": adapter_update_verified,
        "residual_scale_updated": residual_scale_update_verified,
        "residual_scale_min": min(
            (min(row["attn_scale"], row["mlp_scale"]) for row in residual_scale_rows),
            default=None,
        ),
        "residual_scale_max": max(
            (max(row["attn_scale"], row["mlp_scale"]) for row in residual_scale_rows),
            default=None,
        ),
    }
    (result_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    with (result_dir / "final_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(final))
        writer.writeheader()
        writer.writerow(final)
    print(f"FINAL_METRICS {json.dumps(final)}", flush=True)


if __name__ == "__main__":
    main()
