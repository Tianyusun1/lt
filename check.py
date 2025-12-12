# check_pipeline.py
import torch
from models.kg import PoetryKnowledgeGraph
from models.poem2layout import Poem2LayoutGenerator

# 1. 测试 KG 数量扩展
pkg = PoetryKnowledgeGraph()
poem = "两只黄鹂鸣翠柳"
vec = pkg.extract_visual_feature_vector(poem)
raw_ids = [i+2 for i,v in enumerate(vec) if v > 0]
expanded_ids = pkg.expand_ids_with_quantity(raw_ids, poem)
print(f"KG Expansion: {raw_ids} -> {expanded_ids}")
assert len(expanded_ids) >= len(raw_ids), "数量扩展失败"

# 2. 测试 KG 矩阵构建
matrix = pkg.extract_spatial_matrix(poem, obj_ids=expanded_ids)
print(f"Spatial Matrix Shape: {matrix.shape}") # 应该是 [T, T]
assert matrix.shape[0] == len(expanded_ids), "矩阵维度错误"

# 3. 测试模型前向传播
model = Poem2LayoutGenerator(
    bert_path="bert-base-chinese", # 确保路径正确或使用 huggingface 名称
    num_classes=9
)
# 模拟输入
B, T = 1, len(expanded_ids)
input_ids = torch.randint(0, 100, (B, 10))
att_mask = torch.ones((B, 10))
kg_ids = torch.tensor([expanded_ids])
pad_mask = torch.zeros((B, T), dtype=torch.bool)
spatial_mat = torch.tensor(matrix).unsqueeze(0) # [1, T, T]

# 这一步如果不报错，说明所有修改都兼容了！
mu, logvar, boxes, dec_out = model(input_ids, att_mask, kg_ids, pad_mask, kg_spatial_matrix=spatial_mat)
print("Model Forward Pass: Success!")
print(f"Boxes shape: {boxes.shape}, Decoder Out shape: {dec_out.shape}")