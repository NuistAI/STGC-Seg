import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from sam3.train.data.collator import pad_tensor_list_to_longest
from sam3.model.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage,
)

class QaTaDataset(Dataset):
    def __init__(self, image_dir, gt_dir, csv_path, transform=None, split='all'):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.df = pd.read_csv(csv_path)
        
        n = len(self.df)
        if split == 'train':
            self.df = self.df.iloc[:int(0.4 * n)]
        elif split == 'unlabeled':
            self.df = self.df.iloc[int(0.4 * n):int(0.80 * n)]
        elif split == 'val':
            self.df = self.df.iloc[int(0.80 * n):]
        
        self.transform = transform

        self.samples = []
        for idx, row in self.df.iterrows():
            csv_img_name = row.iloc[0]
            prompt = str(row.iloc[1])
            if pd.isna(csv_img_name) or pd.isna(prompt):
                continue
            
            real_img_name = csv_img_name.replace("mask_", "", 1) if str(csv_img_name).startswith("mask_") else csv_img_name
            
            img_path = os.path.join(image_dir, real_img_name)
            mask_path = os.path.join(gt_dir, csv_img_name)
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.samples.append({
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'prompt': prompt
                })
        print(f"Loaded {len(self.samples)} samples from {image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img = Image.open(sample['img_path']).convert('RGB')
        img = img.resize((1008, 1008), Image.Resampling.BILINEAR)
        
        mask = Image.open(sample['mask_path']).convert('L')
        mask = mask.resize((1008, 1008), Image.Resampling.NEAREST)
        mask = np.array(mask)
        mask = (mask > 128).astype(np.float32)
        
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
            'image': img,
            'mask': mask,
            'bbox': bbox,
            'prompt': sample['prompt']
        }

def qata_collate_fn(batch):
    images = []
    masks = []
    bboxes = []
    texts = []
    
    for item in batch:
        images.append(item['image'])
        masks.append(item['mask'])
        bboxes.append(item['bbox'])
        texts.append(item['prompt'])
        
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
        object_ids=[[0] for _ in range(bs)]
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
        object_ids_padded=torch.zeros(bs, 1, dtype=torch.long)
    )
    
    find_metadata = BatchedInferenceMetadata(
        coco_image_id=torch.arange(bs, dtype=torch.long),
        original_size=[(img.shape[1], img.shape[2]) for img in images],
        object_id=torch.zeros(bs, dtype=torch.long),
        frame_index=torch.zeros(bs, dtype=torch.long),
        original_image_id=torch.arange(bs, dtype=torch.long),
        original_category_id=torch.zeros(bs, dtype=torch.long),
        is_conditioning_only=torch.zeros(bs, dtype=torch.bool)
    )
    
    batched_datapoint = BatchedDatapoint(
        img_batch=img_batch,
        find_text_batch=texts,
        find_inputs=[find_stage],
        find_targets=[find_target],
        find_metadatas=[find_metadata],
        raw_images=None
    )
    
    return batched_datapoint
