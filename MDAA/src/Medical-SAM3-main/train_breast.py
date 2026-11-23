"""SAM3 fine-tuning entry point for the BrEaST-Lesions-USG dataset.

This is a *new* script. It reuses the existing training machinery
(model builder, collate function, dice loss, device move helper) and only adds
the BrEaST data-adaptation layer + a ``labeled_ratio`` knob.

The original ``train.py``, ``dataset.py`` and their configs are left untouched.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from dataset import qata_collate_fn
from breast_dataset import BrEaSTDataset
from sam3.model_builder import build_sam3_image_model
from train import move_to_device, dice_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 fine-tuning on BrEaST-Lesions-USG")
    parser.add_argument("--config", type=str, default="train_breast.yaml",
                        help="path to the BrEaST config file")
    parser.add_argument("--labeled_ratio", type=float, default=None,
                        help="fraction of the training split that is labeled (0.1 = 10%%)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (controls labeled-subset selection)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="CUDA device id (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="override TRAIN.train_batch_size")
    parser.add_argument("--lr", type=float, default=None,
                        help="override TRAIN.learning_rate")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override TRAIN.max_epochs")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="override MODEL.checkpoint_path")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    train_cfg = config["TRAIN"]
    model_cfg = config.get("MODEL", {})
    data_cfg = config["DATA"]

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 0)
    set_seed(seed)

    train_batch_size = args.batch_size if args.batch_size is not None else train_cfg["train_batch_size"]
    valid_batch_size = train_cfg.get("valid_batch_size", train_cfg["test_batch_size"])
    test_batch_size = train_cfg["test_batch_size"]
    max_epochs = args.epochs if args.epochs is not None else train_cfg["max_epochs"]
    early_stopping_epochs = train_cfg["early_stopping_epochs"]
    learning_rate = args.lr if args.lr is not None else train_cfg["learning_rate"]
    labeled_ratio = args.labeled_ratio if args.labeled_ratio is not None else train_cfg.get("labeled_ratio", 1.0)
    image_size = tuple(train_cfg.get("image_size", [1008, 1008]))
    checkpoint_path = args.checkpoint_path if args.checkpoint_path is not None else model_cfg.get(
        "checkpoint_path", "checkpoint/checkpoint.pt"
    )
    context_length = model_cfg.get("context_length", 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"seed={seed} labeled_ratio={labeled_ratio} "
          f"batch_size={train_batch_size} lr={learning_rate} epochs={max_epochs}")

    train_dataset = BrEaSTDataset(
        csv_path=data_cfg["train_csv_path"],
        root_path=data_cfg["train_root_path"],
        split="train",
        labeled_ratio=labeled_ratio,
        seed=seed,
        image_size=image_size,
    )
    val_dataset = BrEaSTDataset(
        csv_path=data_cfg["valid_csv_path"],
        root_path=data_cfg["valid_root_path"],
        split="valid",
        image_size=image_size,
    )
    test_dataset = BrEaSTDataset(
        csv_path=data_cfg["test_csv_path"],
        root_path=data_cfg["test_root_path"],
        split="test",
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=qata_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        collate_fn=qata_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=qata_collate_fn,
    )

    print("Building model...")
    model = build_sam3_image_model(
        device=device,
        eval_mode=False,
        load_from_HF=False,
        checkpoint_path=checkpoint_path,
        enable_inst_interactivity=False,
        context_length=context_length,
    )
    model.to(device)

    for name, param in model.named_parameters():
        if "Adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    best_loss = float("inf")
    patience_counter = 0
    save_name = f"best_model_breast_lr{labeled_ratio}.pth"

    print("Starting training...")
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, batched_datapoint in enumerate(train_loader):
            batched_datapoint = move_to_device(batched_datapoint, device)

            optimizer.zero_grad()

            outputs = model(batched_datapoint)
            if hasattr(outputs, "output") and isinstance(outputs.output, list):
                out_dict = outputs[0]
            elif isinstance(outputs, list):
                out_dict = outputs[0][-1] if isinstance(outputs[0], list) else outputs[0]
            elif isinstance(outputs, dict):
                out_dict = outputs
            else:
                out_dict = {}

            if "loss" in out_dict:
                loss = out_dict["loss"]
            else:
                targets = []
                for tgt in batched_datapoint.find_targets:
                    targets.extend(tgt.segments)

                if len(targets) > 0:
                    targets = torch.stack(targets).unsqueeze(1).to(device)
                else:
                    targets = torch.zeros(
                        batched_datapoint.img_batch.size(0), 1, image_size[0], image_size[1], device=device
                    )

                if "pred_masks" in out_dict:
                    pred_masks = out_dict["pred_masks"][:, 0:1, :, :]
                else:
                    pred_masks = torch.rand_like(targets)

                pred_masks = torch.sigmoid(pred_masks)
                if pred_masks.shape != targets.shape:
                    pred_masks = F.interpolate(
                        pred_masks, size=targets.shape[-2:], mode="bilinear", align_corners=False
                    )

                loss = dice_loss(pred_masks, targets.float())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}/{max_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch} finished. Avg Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batched_datapoint in val_loader:
                batched_datapoint = move_to_device(batched_datapoint, device)
                outputs = model(batched_datapoint)
                if hasattr(outputs, "output") and isinstance(outputs.output, list):
                    out_dict = outputs[0]
                elif isinstance(outputs, list):
                    out_dict = outputs[0][-1] if isinstance(outputs[0], list) else outputs[0]
                elif isinstance(outputs, dict):
                    out_dict = outputs
                else:
                    out_dict = {}

                if "loss" in out_dict:
                    loss = out_dict["loss"]
                else:
                    targets = []
                    for tgt in batched_datapoint.find_targets:
                        targets.extend(tgt.segments)

                    if len(targets) > 0:
                        targets = torch.stack(targets).unsqueeze(1).to(device)
                    else:
                        targets = torch.zeros(
                            batched_datapoint.img_batch.size(0), 1, image_size[0], image_size[1], device=device
                        )

                    if "pred_masks" in out_dict:
                        pred_masks = out_dict["pred_masks"][:, 0:1, :, :]
                    else:
                        pred_masks = torch.rand_like(targets)

                    pred_masks = torch.sigmoid(pred_masks)
                    if pred_masks.shape != targets.shape:
                        pred_masks = F.interpolate(
                            pred_masks, size=targets.shape[-2:], mode="bilinear", align_corners=False
                        )
                    loss = dice_loss(pred_masks, targets.float())

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_name)
            print(f"Saved new best model -> {save_name}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_epochs:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Training finished.")


if __name__ == "__main__":
    main()
