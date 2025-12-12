import torch
import torch.nn.functional as F

def layout_loss(pred_cls, pred_coord, target_layout_seq, target_layout_mask, coord_loss_weight=1.0):
    """
    计算布局生成的混合损失。
    Args:
        pred_cls: [B, num_elements, num_classes] 预测的类别 logits
        pred_coord: [B, num_elements, 4] 预测的坐标 (cx, cy, w, h)
        target_layout_seq: [B, S] (S = 5 * num_elements) 真实布局序列
        target_layout_mask: [B, S] 真实布局掩码
        coord_loss_weight: 坐标损失权重
    Returns:
        total_loss, cls_loss, coord_loss
    """
    batch_size, seq_len = target_layout_seq.shape
    num_elements = seq_len // 5

    reshaped_target = target_layout_seq.view(batch_size, num_elements, 5)
    target_cls_ids = reshaped_target[:, :, 0].long() # [B, num_elements]
    target_coords = reshaped_target[:, :, 1:5].float() # [B, num_elements, 4]

    reshaped_mask = target_layout_mask.view(batch_size, num_elements, 5)
    cls_mask = reshaped_mask[:, :, 0].bool() # [B, num_elements]

    # Classification loss
    cls_loss = F.cross_entropy(pred_cls, target_cls_ids, reduction='none') # [B, num_elements]
    cls_loss = cls_loss * cls_mask.float()
    cls_loss = cls_loss.sum() / cls_mask.sum().clamp(min=1)

    # Coordinate loss
    coord_loss = F.smooth_l1_loss(pred_coord, target_coords, reduction='none') # [B, num_elements, 4]
    coord_loss = coord_loss * cls_mask.unsqueeze(-1).float()
    coord_loss = coord_loss.sum() / (cls_mask.sum().clamp(min=1) * 4)

    total_loss = cls_loss + coord_loss_weight * coord_loss
    return total_loss, cls_loss, coord_loss
