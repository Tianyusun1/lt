# models/decoder.py (V5.3 Fixed Initialization)

import torch
import torch.nn as nn
from .transformer_layers import PoemLayoutDecoderLayer

class LayoutDecoder(nn.Module):
    def __init__(self, hidden_size: int, bb_size: int, num_layers: int, num_heads: int, ff_size: int = None, dropout: float = 0.1):
        super(LayoutDecoder, self).__init__()
        if ff_size is None:
            ff_size = hidden_size * 4
        self.bb_size = bb_size
        
        self.layers = nn.ModuleList([
            PoemLayoutDecoderLayer(
                hidden_size=hidden_size,
                bb_size=bb_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # The final layer norm input size is hidden_size + bb_size
        self.layer_norm = nn.LayerNorm(hidden_size + bb_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    # [MODIFIED] Added spatial_bias parameter
    def forward(self, layout_embed: torch.Tensor, text_features: torch.Tensor, src_mask: torch.Tensor, trg_mask: torch.Tensor, spatial_bias: torch.Tensor = None):
        # layout_embed: [B, T, bb_size]
        # text_features: [B, L_text, hidden_size]
        B, T, bb_size_dim = layout_embed.shape
        B_text, L_text, hidden_size_dim = text_features.shape

        # === 关键修复区域 (V5.2 Fix) ===
        # 原问题：全 0 初始化导致第一层 Text 流缺乏语义引导，Cross-Attention 退化。
        # 修正：使用 BERT 的 [CLS] token (Index 0) 初始化 text_x 流。
        # 这为 Text 流提供了全局语义上下文作为起点。
        
        # 1. 提取 [CLS] token: [B, L_text, H] -> [B, 1, H]
        cls_token = text_features[:, 0, :].unsqueeze(1)
        
        # 2. 扩展到序列长度 T: [B, 1, H] -> [B, T, H]
        # 使用 clone() 确保后续 inplace 操作安全
        text_x = cls_token.expand(-1, T, -1).clone()
        
        layout_x = layout_embed # [B, T, bb_size]
        # ==================

        # Iterate through the stack of layers
        for layer in self.layers:
            # layout_x 和 text_x 都是 Query，text_features 是 Memory
            # [MODIFIED] Pass spatial_bias to each layer
            layout_x, text_x = layer(layout_x, text_x, text_features, src_mask, trg_mask, spatial_bias=spatial_bias)

        # After all layers, concatenate the final streams
        # Order: [text_features_stream, layout_features_stream]
        output = torch.cat([text_x, layout_x], dim=-1) # [B, T, hidden_size + bb_size]
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output # [B, T, hidden_size + bb_size]