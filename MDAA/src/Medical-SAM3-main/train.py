import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import QaTaDataset, qata_collate_fn
from sam3.model_builder import build_sam3_image_model
import os

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif hasattr(obj, "__dataclass_fields__"):
        from dataclasses import fields
        for field in fields(obj):
            val = getattr(obj, field.name)
            setattr(obj, field.name, move_to_device(val, device))
        return obj
    return obj

def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2. * intersection + smooth) / (union + smooth)).mean()

def main():
    with open('train.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_dataset = QaTaDataset(
        image_dir=config['train_dataset_path'] + '/Images',
        gt_dir=config['train_dataset_path'] + '/GTs',
        csv_path=config['train_text_path'],
        split='train'
    )
    
    val_dataset = QaTaDataset(
        image_dir=config['train_dataset_path'] + '/Images',
        gt_dir=config['train_dataset_path'] + '/GTs',
        csv_path=config['train_text_path'],
        split='val'
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['test_batch_size'], 
        shuffle=False,
        collate_fn=qata_collate_fn
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train_batch_size'], 
        shuffle=True,
        collate_fn=qata_collate_fn
    )

    test_dataset = QaTaDataset(
        image_dir=config['test_dataset_path'] + '/Images',
        gt_dir=config['test_dataset_path'] + '/GTs',
        csv_path=config['test_text_path']
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['test_batch_size'], 
        shuffle=False,
        collate_fn=qata_collate_fn
    )

    print("Building model...")
    model = build_sam3_image_model(
        device=device,
        eval_mode=False,
        load_from_HF=False,
        checkpoint_path="checkpoint/checkpoint.pt",
        enable_inst_interactivity=False,
        context_length=32,
    )
    model.to(device)

    for name, param in model.named_parameters():
        if "Adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])
    bce_loss = nn.BCEWithLogitsLoss()

    max_epochs = config['max_epochs']
    early_stopping_epochs = config['early_stopping_epochs']
    
    best_loss = float('inf')
    patience_counter = 0

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
                    targets = torch.zeros(batched_datapoint.img_batch.size(0), 1, 1008, 1008, device=device)
                
                if "pred_masks" in out_dict:
                    pred_masks = out_dict["pred_masks"][:, 0:1, :, :]
                else:
                    pred_masks = torch.rand_like(targets)
                
                pred_masks = torch.sigmoid(pred_masks)
                if pred_masks.shape != targets.shape:
                    pred_masks = torch.nn.functional.interpolate(pred_masks, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                
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
            for batch_idx, batched_datapoint in enumerate(val_loader):
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
                        targets = torch.zeros(batched_datapoint.img_batch.size(0), 1, 1008, 1008, device=device)
                    
                    if "pred_masks" in out_dict:
                        pred_masks = out_dict["pred_masks"][:, 0:1, :, :]
                    else:
                        pred_masks = torch.rand_like(targets)
                        
                    pred_masks = torch.sigmoid(pred_masks)
                    if pred_masks.shape != targets.shape:
                        pred_masks = torch.nn.functional.interpolate(pred_masks, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                    loss = dice_loss(pred_masks, targets.float())
                
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_qata_test.pth")
            print("Saved new best model.")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_epochs:
                print(f"Early stopping at epoch {epoch}")
                break

if __name__ == '__main__':
    main()
