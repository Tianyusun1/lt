# File: tianyusun1/test2/test2-5.1/data/dataset.py (V6.0: QUANTITY EXPANSION + DYNAMIC MATRIX)

import os
import torch
import pandas as pd
import yaml 
from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np 
import random 

# --- 导入知识图谱模型 ---
from models.kg import PoetryKnowledgeGraph
# --- 导入位置引导信号生成器 ---
from models.location import LocationSignalGenerator
# ---------------------------

# 类别定义
CLASS_NAMES = {
    2: "mountain", 3: "water", 4: "people", 5: "tree",
    6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"
}
VALID_CLASS_IDS = set(CLASS_NAMES.keys())

def _load_num_bins_from_config():
    """尝试从 configs/default.yaml 中读取 num_bbox_bins"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs/default.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config['model']['num_bbox_bins']
    except Exception:
        return 1000

class PoegraphLayoutDataset(Dataset):
    def __init__(
        self,
        xlsx_path: str,
        labels_dir: str,
        bert_model_path: str = "/home/sty/pyfile/huggingface/bert-base-chinese",
        max_layout_length: int = 30, 
        max_text_length: int = 64, 
        preload: bool = False
    ):
        super().__init__()
        self.xlsx_path = xlsx_path
        self.labels_dir = Path(labels_dir)
        self.max_layout_length = max_layout_length # 这里指最大物体数量
        self.max_text_length = max_text_length
        
        self.num_classes = 9 

        print("Initializing Knowledge Graph...")
        self.pkg = PoetryKnowledgeGraph()
        print("✅ Knowledge Graph initialized.")
        
        # 初始化位置信号生成器 (8x8)
        self.location_gen = LocationSignalGenerator(grid_size=8)
        print("✅ Location Signal Generator (8x8) initialized.")
        
        # 加载 Excel
        df = pd.read_excel(xlsx_path)
        self.data = []

        for _, row in df.iterrows():
            raw_img_name = str(row['image']).strip()
            poem = str(row['poem']).strip()
            img_stem = Path(raw_img_name).stem
            label_path = self.labels_dir / f"{img_stem}.txt"

            if not label_path.exists():
                continue

            # 读取并验证标注
            boxes = []
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5: continue
                        cls_id = int(float(parts[0]))
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        if cls_id in VALID_CLASS_IDS and \
                           0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1:
                            boxes.append((float(cls_id), cx, cy, w, h)) 
            except Exception:
                continue

            if boxes:
                self.data.append({
                    'poem': poem,
                    'boxes': boxes # List[(cls, cx, cy, w, h)]
                })

        print(f"✅ PoegraphLayoutDataset 加载完成，共 {len(self.data)} 个样本")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        poem = sample['poem']
        gt_boxes = sample['boxes'] # List[(cls_id, cx, cy, w, h)]

        # 1. 文本编码
        tokenized = self.tokenizer(
            poem,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )

        # ---------------------------------------------------------
        # 2. KG 推理 & 数量扩展 (Quantity Expansion)
        # ---------------------------------------------------------
        # A. 基础提取 (Base Extraction)
        kg_vector = self.pkg.extract_visual_feature_vector(poem) # [9] multi-hot
        raw_indices = torch.nonzero(kg_vector > 0).squeeze(1).tolist()
        raw_ids = [i + 2 for i in raw_indices]
        
        # B. [INNOVATION] 数量扩展
        # "千峰" -> [Mountain, Mountain, Mountain, ...]
        if not raw_ids:
            expanded_ids = []
        else:
            expanded_ids = self.pkg.expand_ids_with_quantity(raw_ids, poem)
            
        # C. 截断 (Truncate)
        if len(expanded_ids) > self.max_layout_length:
            expanded_ids = expanded_ids[:self.max_layout_length]
            
        num_boxes = len(expanded_ids)
        
        # [验证输出] 随机打印变长样本 (仅调试用)
        if len(expanded_ids) > len(raw_ids) and random.random() < 0.01: 
            print(f"\n[Dataset Debug] ID Expansion: {raw_ids} -> {expanded_ids} ('{poem}')")

        # ---------------------------------------------------------
        # 3. 提取空间关系矩阵 (Instance-level Spatial Matrix)
        # ---------------------------------------------------------
        # [INNOVATION] 生成 T x T 的实例关系矩阵
        if num_boxes > 0:
            spatial_matrix = self.pkg.extract_spatial_matrix(poem, obj_ids=expanded_ids)
        else:
            spatial_matrix = torch.zeros((0, 0), dtype=torch.long)

        # ---------------------------------------------------------
        # 4. GT 对齐与构建 (Alignment & Target Construction)
        # ---------------------------------------------------------
        target_boxes = []
        loss_mask = [] 
        kg_class_weights = []

        # 将 GT 按类别分组
        gt_dict = {}
        for item in gt_boxes:
            cid, cx, cy, w, h = item
            cid = int(cid)
            if cid not in gt_dict: gt_dict[cid] = []
            gt_dict[cid].append([cx, cy, w, h])

        # 全局数据增强决策
        do_flip = random.random() < 0.5
        
        # 遍历扩展后的 IDs
        for i, k_cls in enumerate(expanded_ids):
            k_cls = int(k_cls)
            
            # 设置权重 (简单处理：存在即为 1.0)
            idx = k_cls - 2
            if 0 <= idx < self.num_classes:
                kg_class_weights.append(kg_vector[idx].item())
            else:
                kg_class_weights.append(1.0)

            # 尝试分配 GT Box
            if k_cls in gt_dict and len(gt_dict[k_cls]) > 0:
                box = gt_dict[k_cls].pop(0) # [cx, cy, w, h]
                
                # === 脏数据过滤 ===
                if box[2] * box[3] > 0.90: # 太大
                    target_boxes.append([0.0, 0.0, 0.0, 0.0])
                    loss_mask.append(0.0) 
                    continue
                
                aspect_ratio = box[2] / (box[3] + 1e-6)
                if aspect_ratio > 10.0 or aspect_ratio < 0.1: # 太扁或太高
                    target_boxes.append([0.0, 0.0, 0.0, 0.0])
                    loss_mask.append(0.0)
                    continue
                # =================
                
                # [几何增强]
                if do_flip:
                    box[0] = 1.0 - box[0]
                
                # Jitter
                noise = np.random.uniform(-0.02, 0.02, size=4)
                box_aug = [
                    np.clip(box[0] + noise[0], 0.0, 1.0),
                    np.clip(box[1] + noise[1], 0.0, 1.0),
                    np.clip(box[2] + noise[2], 0.01, 1.0),
                    np.clip(box[3] + noise[3], 0.01, 1.0)
                ]
                target_boxes.append(box_aug)
                loss_mask.append(1.0)
            else:
                # 这是一个由数量扩展产生的新物体，但在 GT 里没有对应的框
                # (例如 "两只鸟"，GT只标了1个框)
                # 这种情况下，我们不计算 Regression Loss，但它仍然参与 Attention 和 Spatial Reasoning
                target_boxes.append([0.0, 0.0, 0.0, 0.0])
                loss_mask.append(0.0)

        # 处理空数据情况
        if not expanded_ids:
            expanded_ids = [0]
            kg_class_weights = [0.0]
            target_boxes = [[0.0]*4]
            loss_mask = [0.0]
            spatial_matrix = torch.zeros((1, 1), dtype=torch.long) # Pad 1x1

        # ---------------------------------------------------------
        # 5. 生成位置引导信号 (Location Grids - Stateful)
        # ---------------------------------------------------------
        location_grids_list = [] 
        current_occupancy = torch.zeros((8, 8), dtype=torch.float32)
        
        for i, cls_id in enumerate(expanded_ids):
            cls_id = int(cls_id)
            if cls_id == 0: # PAD
                location_grids_list.append(torch.zeros((8, 8), dtype=torch.float32))
                continue
                
            # 获取该物体在 spatial_matrix 中的行和列 (T x T 矩阵)
            # 注意: 如果是空数据情况，matrix 大小可能不匹配，需保护
            if i < spatial_matrix.shape[0]:
                spatial_row = spatial_matrix[i, :]  
                spatial_col = spatial_matrix[:, i] 
            else:
                spatial_row = torch.zeros(len(expanded_ids))
                spatial_col = torch.zeros(len(expanded_ids))
            
            # 推理
            signal, current_occupancy = self.location_gen.infer_stateful_signal(
                i, spatial_row, spatial_col, current_occupancy,
                mode='sample', top_k=3 
            )
            
            # Jitter
            if random.random() < 0.7: 
                shift = random.randint(-2, 2) 
                signal = torch.roll(signal, shifts=shift, dims=1) 
            
            location_grids_list.append(signal)
            
        location_grids = torch.stack(location_grids_list) # [T, 8, 8]
        if do_flip:
            location_grids = torch.flip(location_grids, dims=[2])

        return {
            'input_ids': tokenized['input_ids'].squeeze(0), 
            'attention_mask': tokenized['attention_mask'].squeeze(0), 
            'kg_class_ids': torch.tensor(expanded_ids, dtype=torch.long),
            'kg_class_weights': torch.tensor(kg_class_weights, dtype=torch.float32), 
            'target_boxes': torch.tensor(target_boxes, dtype=torch.float32),
            'loss_mask': torch.tensor(loss_mask, dtype=torch.float32),
            'kg_spatial_matrix': spatial_matrix, # [T, T] Tensor
            'kg_vector': kg_vector,
            'num_boxes': torch.tensor(len(gt_boxes), dtype=torch.long), # 原始 GT 数量
            'location_grids': location_grids 
        }

def layout_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function to handle variable length fields.
    Specially handles padding of 2D spatial matrices.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    kg_vectors = torch.stack([item['kg_vector'] for item in batch])
    num_boxes = torch.stack([item['num_boxes'] for item in batch])

    # 获取 Batch 中的最大序列长度
    lengths = [len(item['kg_class_ids']) for item in batch]
    max_len = max(lengths)
    if max_len == 0: max_len = 1

    batched_class_ids = []
    batched_class_weights = [] 
    batched_target_boxes = []
    batched_loss_mask = []
    batched_padding_mask = [] 
    batched_location_grids = []
    
    # [NEW] 批量化空间矩阵
    batched_spatial_matrices = torch.zeros((len(batch), max_len, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        cur_len = len(item['kg_class_ids'])
        pad_len = max_len - cur_len
        
        # 1. IDs
        padded_ids = torch.cat([
            item['kg_class_ids'], 
            torch.zeros(pad_len, dtype=torch.long)
        ])
        batched_class_ids.append(padded_ids)
        
        # 2. Weights 
        padded_weights = torch.cat([
            item['kg_class_weights'],
            torch.zeros(pad_len, dtype=torch.float32)
        ])
        batched_class_weights.append(padded_weights)
        
        # 3. Boxes
        padded_boxes = torch.cat([
            item['target_boxes'], 
            torch.zeros((pad_len, 4), dtype=torch.float32)
        ])
        batched_target_boxes.append(padded_boxes)
        
        # 4. Mask
        padded_loss_mask = torch.cat([
            item['loss_mask'], 
            torch.zeros(pad_len, dtype=torch.float32)
        ])
        batched_loss_mask.append(padded_loss_mask)

        # 5. Location Grids
        padded_grids = torch.cat([
            item['location_grids'],
            torch.zeros((pad_len, 8, 8), dtype=torch.float32)
        ])
        batched_location_grids.append(padded_grids)
        
        # 6. Pad Mask
        pad_mask = torch.zeros(max_len, dtype=torch.bool)
        if pad_len > 0:
            pad_mask[cur_len:] = True
        batched_padding_mask.append(pad_mask)
        
        # 7. [NEW] Spatial Matrix Padding (2D)
        mat = item['kg_spatial_matrix']
        # 将 [T, T] 的矩阵填入 [B, Max, Max] 的左上角
        batched_spatial_matrices[i, :cur_len, :cur_len] = mat

    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'kg_class_ids': torch.stack(batched_class_ids),      
        'kg_class_weights': torch.stack(batched_class_weights), 
        'target_boxes': torch.stack(batched_target_boxes),   
        'loss_mask': torch.stack(batched_loss_mask),         
        'padding_mask': torch.stack(batched_padding_mask),   
        'kg_spatial_matrix': batched_spatial_matrices, # [B, Max, Max]
        'kg_vector': kg_vectors,
        'num_boxes': num_boxes,
        'location_grids': torch.stack(batched_location_grids) 
    }

if __name__ == "__main__":
    pass