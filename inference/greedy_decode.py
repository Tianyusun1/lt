# File: tianyusun1/test2/test2-5.1/inference/greedy_decode.py (V5.3: QUANTITY AWARE + JITTER)

import torch
import numpy as np
import random # [NEW] Needed for jitter logic

# --- Import KG & Location ---
try:
    from models.kg import PoetryKnowledgeGraph
except ImportError:
    print("[Error] Could not import PoetryKnowledgeGraph. Make sure models/kg.py is accessible.")
    PoetryKnowledgeGraph = None

# 导入位置生成器 (V4.5+ Gaussian Support)
try:
    from models.location import LocationSignalGenerator
except ImportError:
    print("[Error] Could not import LocationSignalGenerator. Make sure models/location.py is accessible.")
    LocationSignalGenerator = None
# -----------------

# [NEW V5.2] 定义所有类别的形状先验 (Shape Priors)
# 防止物体塌缩或形状畸形。这些是物理/常识约束。
# 2:mtn, 3:water, 4:ppl, 5:tree, 6:bldg, 7:bridge, 8:flower, 9:bird, 10:animal
CLASS_SHAPE_PRIORS = {
    # 背景大物体 (山、水) -> 至少要有一定规模
    2: {'min_w': 0.20, 'min_h': 0.20}, # Mountain
    3: {'min_w': 0.20, 'min_h': 0.10}, # Water
    
    # 瘦高物体 (人、树)
    4: {'min_w': 0.02, 'min_h': 0.08, 'max_w': 0.15}, # People (人应该是瘦高的)
    5: {'min_w': 0.05, 'min_h': 0.15}, # Tree (树应该是比较高的)
    
    # 块状物体 (建筑)
    6: {'min_w': 0.05, 'min_h': 0.05}, # Building
    
    # 扁平物体 (桥)
    7: {'min_w': 0.15, 'max_h': 0.08}, # Bridge (桥应该是宽而扁的)
    
    # 小物体 (花、鸟、兽) -> 防止变成不可见的点
    8: {'min_w': 0.03, 'min_h': 0.03}, # Flower
    9: {'min_w': 0.03, 'min_h': 0.03}, # Bird
    10: {'min_w': 0.04, 'min_h': 0.04} # Animal
}

def greedy_decode_poem_layout(model, tokenizer, poem: str, max_elements=None, device='cuda', mode='greedy', top_k=3):
    """
    Query-Based decoding with Location Guidance & CVAE Diversity.
    """
    if PoetryKnowledgeGraph is None:
        return []

    model.eval()
    device = torch.device(device)
    model.to(device)
    
    # 1. 实例化组件
    pkg = PoetryKnowledgeGraph()
    
    if LocationSignalGenerator is not None:
        location_gen = LocationSignalGenerator(grid_size=8)
    else:
        location_gen = None
    
    # 2. KG 提取内容 (含数量扩展逻辑)
    kg_vector = pkg.extract_visual_feature_vector(poem)
    existing_indices = torch.nonzero(kg_vector > 0).squeeze(1)
    raw_class_ids = (existing_indices + 2).tolist()
    
    if not raw_class_ids:
        return []
        
    # [MODIFIED] 调用 KG 进行数量扩展 (解决"两只鸟"等问题)
    # 需确保 models/kg.py 已更新包含此方法
    if hasattr(pkg, 'expand_ids_with_quantity'):
        kg_class_ids = pkg.expand_ids_with_quantity(raw_class_ids, poem)
        # [验证输出]
        if len(kg_class_ids) > len(raw_class_ids):
            print(f"[Infer Debug] Quantity Rules Applied: {raw_class_ids} -> {kg_class_ids}")
    else:
        kg_class_ids = raw_class_ids
        
    # Limit max length
    if max_elements:
        kg_class_ids = kg_class_ids[:max_elements]
        
    # 3. 准备模型输入 Tensor
    kg_class_tensor = torch.tensor([kg_class_ids], dtype=torch.long).to(device)
    
    kg_spatial_matrix_raw = pkg.extract_spatial_matrix(poem)
    kg_spatial_matrix = kg_spatial_matrix_raw.unsqueeze(0).to(device) 
    
    # === 生成位置引导信号 (含 Jitter) ===
    location_grids_tensor = None
    
    if location_gen is not None:
        current_occupancy = torch.zeros((8, 8), dtype=torch.float32)
        grids_list = []
        
        for i, cls_id in enumerate(kg_class_ids):
            # Class ID (2-10) -> Matrix Index (0-8)
            matrix_idx = int(cls_id) - 2
            if matrix_idx < 0 or matrix_idx >= 9: matrix_idx = 0 # Safety
            
            row = kg_spatial_matrix_raw[matrix_idx]
            col = kg_spatial_matrix_raw[:, matrix_idx]
            
            signal, current_occupancy = location_gen.infer_stateful_signal(
                i, row, col, current_occupancy, 
                mode=mode, top_k=top_k 
            )
            
            # [NEW V5.3] 引入位置抖动 (Jitter)
            # 打破 "Location Grid 总是指向正中间" 的先验偏差
            # 仅在 sample 模式下启用，保持 greedy 的确定性
            shift_val = 0
            if mode == 'sample':
                if random.random() < 0.6: # 60% 概率发生偏移
                    shift_val = random.randint(-2, 2)
                    signal = torch.roll(signal, shifts=shift_val, dims=1)
            
            if shift_val != 0:
                # 可选：打印调试信息
                # print(f"[Infer Debug] Obj {i} (Class {cls_id}): Jitter shift {shift_val}")
                pass

            grids_list.append(signal)
            
        location_grids_tensor = torch.stack(grids_list).unsqueeze(0).to(device)
    # ========================
    
    # 文本编码
    inputs = tokenizer(poem, return_tensors='pt', padding=True, truncation=True, max_length=64)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    padding_mask = torch.zeros(kg_class_tensor.shape, dtype=torch.bool).to(device)
    
    # 4. 单次前向传播
    with torch.no_grad():
        _, _, pred_boxes, _ = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            kg_class_ids=kg_class_tensor, 
            padding_mask=padding_mask, 
            kg_spatial_matrix=kg_spatial_matrix,
            location_grids=location_grids_tensor
        )
        
    # 5. 格式化输出 (应用形状先验)
    layout = []
    boxes_flat = pred_boxes[0].cpu().tolist()
    
    for cls_id, box in zip(kg_class_ids, boxes_flat):
        cid = int(cls_id)
        cx, cy, w, h = box
        
        # === [FIX V5.2] 通用形状约束逻辑 ===
        # 1. 基础兜底：防止塌缩成点 (至少 2%)
        w = max(w, 0.02)
        h = max(h, 0.02)
        
        if cid in CLASS_SHAPE_PRIORS:
            prior = CLASS_SHAPE_PRIORS[cid]
            if 'min_w' in prior: w = max(w, prior['min_w'])
            if 'min_h' in prior: h = max(h, prior['min_h'])
            if 'max_w' in prior: w = min(w, prior['max_w'])
            if 'max_h' in prior: h = min(h, prior['max_h'])
        # =================================
        
        layout.append((float(cls_id), cx, cy, w, h))
        
    return layout