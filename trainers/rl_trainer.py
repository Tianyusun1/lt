# File: tianyusun1/test2/test2-5.2/trainers/rl_trainer.py (V5.23: DYNAMIC REWARDS + CONSISTENCY)

import torch
import torch.nn.functional as F
import torch.optim as optim
from .trainer import LayoutTrainer
import time
import os
import matplotlib.pyplot as plt

# [MODIFIED] Import robust KL calculation
from trainers.loss import compute_kl_loss

class RLTrainer(LayoutTrainer):
    """
    强化学习训练器 (RL Fine-tuning Trainer)。
    继承自 LayoutTrainer，但重写了训练循环以支持 SCST。
    
    [V5.23 Updates]
    1. 动态尺寸奖励 (Dynamic Size Reward): 强制小物体(鸟/花)微小，独占物体放大。
    2. 一致性损失监控 (Consistency Loss): 监控 Text/Layout Stream 对齐。
    3. 接口适配: 兼容 Poem2LayoutGenerator V6.0 的返回值。
    """
    def __init__(self, model, train_loader, val_loader, config, tokenizer, example_poem, test_loader):
        super().__init__(model, train_loader, val_loader, config, tokenizer, example_poem, test_loader)
        
        # === 1. RL 超参数设置 ===
        self.rl_lr = float(config['training'].get('rl_learning_rate', 5e-6))
        
        # 重新定义优化器 (针对 RL 微调)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.rl_lr)
        
        # === 2. 奖励权重 (Reward Weights) ===
        reward_cfg = config['training'].get('reward_weights', {})
        
        self.w_iou = float(reward_cfg.get('iou', 2.0))              
        self.w_rel = float(reward_cfg.get('relation', 5.0)) # 强语义约束       
        self.w_dispersion = float(reward_cfg.get('dispersion', 0.5)) 
        self.w_overlap = float(reward_cfg.get('overlap', -0.5))     
        
        # [NEW] 尺寸奖励权重 (引导形状合理性)
        self.w_size = 2.0 

        # 用于记录最近一次 batch 的奖励明细
        self.last_reward_stats = {}

        # 奖励历史记录，用于画图
        self.reward_history = []
        self.plot_path_reward = os.path.join(self.output_dir, "rl_reward_trajectory.png")

        print(f"[RLTrainer] Initialized. LR={self.rl_lr:.2e}")
        print(f"[RLTrainer] Reward Weights: IoU={self.w_iou}, Rel={self.w_rel}, Disp={self.w_dispersion}, Size={self.w_size}")

    def compute_reward(self, pred_boxes, batch):
        """
        计算每个样本的奖励值 (Batch-wise Reward Calculation)。
        """
        B, T, _ = pred_boxes.shape
        device = pred_boxes.device
        
        # 提取 Batch 信息
        loss_mask = batch['loss_mask']          
        target_boxes = batch['target_boxes']
        kg_spatial_matrix = batch['kg_spatial_matrix'] 
        kg_class_ids = batch['kg_class_ids']    
        
        # 初始化奖励矩阵 [B, T]
        obj_rewards = torch.zeros(B, T, device=device)
        
        # ===========================================================
        # A. 监督奖励 (Supervised Reward)
        # ===========================================================
        iou = self._calculate_iou(pred_boxes, target_boxes) 
        r_iou = iou * loss_mask * self.w_iou
        obj_rewards += r_iou

        # ===========================================================
        # B. 关系奖励 (Relation Reward)
        # ===========================================================
        rel_scores = self._calculate_relation_reward(pred_boxes, kg_spatial_matrix, kg_class_ids) 
        r_rel = rel_scores * self.w_rel
        obj_rewards += r_rel

        # ===========================================================
        # C. 分散度奖励 (Dispersion Reward)
        # ===========================================================
        centers = pred_boxes[..., :2] 
        dists = torch.cdist(centers, centers)
        eye = torch.eye(T, device=device).unsqueeze(0)
        dists = dists + eye * 10.0
        min_dist, _ = dists.min(dim=2)
        disp_score = torch.clamp(min_dist, max=0.3) 
        r_disp = disp_score * self.w_dispersion
        obj_rewards += r_disp

        # ===========================================================
        # D. 构图美学奖励：三分法
        # ===========================================================
        cx = centers[..., 0]
        cy = centers[..., 1]
        dist_x_third = torch.min(torch.abs(cx - 0.333), torch.abs(cx - 0.667))
        dist_y_third = torch.min(torch.abs(cy - 0.333), torch.abs(cy - 0.667))
        r_composition = (0.2 - (dist_x_third + dist_y_third)).clamp(min=0) * 2.5
        obj_rewards += r_composition 

        # ===========================================================
        # E. 边缘惩罚 & 重叠惩罚
        # ===========================================================
        dist_to_edge = torch.min(centers, 1.0 - centers) 
        is_too_close_to_edge = (dist_to_edge < 0.05).float().sum(dim=-1) 
        r_border_penalty = is_too_close_to_edge * -0.5 
        obj_rewards += r_border_penalty

        overlap_penalty = self._calculate_overlap_penalty(pred_boxes) 
        r_over = overlap_penalty * self.w_overlap
        obj_rewards += r_over
        
        # ===========================================================
        # G. [NEW] 动态尺寸奖励 (Dynamic Size Reward)
        # ===========================================================
        size_reward = self._calculate_dynamic_size_reward(pred_boxes, batch['loss_mask'], kg_class_ids)
        r_size = size_reward * self.w_size
        obj_rewards += r_size

        # 记录明细
        self.last_reward_stats = {
            'IoU': (r_iou.sum() / (loss_mask.sum() + 1e-6)).item(),
            'Rel': (r_rel.sum() / (B * T)).item(),
            'Disp': disp_score.mean().item(),
            'Size': size_reward.mean().item(),
            'Comp': r_composition.mean().item()
        }

        # === 汇总 ===
        total_sample_reward = obj_rewards.sum(dim=1) / (T + 1e-6)
        
        return total_sample_reward

    def _calculate_dynamic_size_reward(self, pred_boxes, mask, class_ids):
        """
        [INNOVATION] 计算动态尺寸奖励。
        """
        B, T, _ = pred_boxes.shape
        w = pred_boxes[..., 2]
        h = pred_boxes[..., 3]
        areas = w * h 
        
        rewards = torch.zeros_like(areas)
        
        # 小物体 ID (Flower:8, Bird:9, Animal:10)
        SMALL_OBJECT_IDS = [8, 9, 10] 
        
        for b in range(B):
            valid_mask = (mask[b] > 0)
            # 如果是 Quantity Expansion 生成的，可能 mask=0 但我们依然希望约束它
            # 这里我们使用 class_ids > 0 来判断是否是有效物体
            valid_obj_mask = (class_ids[b] > 0)
            
            num_valid = valid_obj_mask.sum().item()
            if num_valid == 0: continue
            
            # 规则 1: 基准面积随数量递减
            base_target_area = 0.4 / (num_valid ** 0.6) 
            
            for t in range(T):
                if not valid_obj_mask[t]: continue
                
                cid = class_ids[b, t].item()
                current_area = areas[b, t]
                
                # 规则 2: 小物体强制约束
                if cid in SMALL_OBJECT_IDS:
                    target_min, target_max = 0.005, 0.03
                    if target_min <= current_area <= target_max:
                        r = 1.0
                    else:
                        dist = min(abs(current_area - target_min), abs(current_area - target_max))
                        r = max(0.0, 1.0 - dist * 10.0)
                    rewards[b, t] = r
                    continue 
                
                # 规则 3: 独占画面 (Solitary)
                if num_valid == 1:
                    if current_area > 0.3:
                        rewards[b, t] = 1.0
                    else:
                        rewards[b, t] = current_area / 0.3 
                    continue
                
                # 规则 4: 普通物体
                t_min = base_target_area * 0.5
                t_max = base_target_area * 1.5
                
                if t_min <= current_area <= t_max:
                    rewards[b, t] = 1.0
                else:
                    dist = min(abs(current_area - t_min), abs(current_area - t_max))
                    rewards[b, t] = max(0.0, 1.0 - dist * 5.0)

        return rewards

    def _calculate_iou(self, pred, target):
        p_x1, p_y1 = pred[..., 0]-pred[..., 2]/2, pred[..., 1]-pred[..., 3]/2
        p_x2, p_y2 = pred[..., 0]+pred[..., 2]/2, pred[..., 1]+pred[..., 3]/2
        t_x1, t_y1 = target[..., 0]-target[..., 2]/2, target[..., 1]-target[..., 3]/2
        t_x2, t_y2 = target[..., 0]+target[..., 2]/2, target[..., 1]+target[..., 3]/2
        
        i_x1 = torch.max(p_x1, t_x1); i_y1 = torch.max(p_y1, t_y1)
        i_x2 = torch.min(p_x2, t_x2); i_y2 = torch.min(p_y2, t_y2)
        
        i_area = (i_x2 - i_x1).clamp(min=0) * (i_y2 - i_y1).clamp(min=0)
        p_area = pred[..., 2] * pred[..., 3]
        t_area = target[..., 2] * target[..., 3]
        u_area = p_area + t_area - i_area
        return i_area / (u_area + 1e-6)

    def _calculate_relation_reward(self, boxes, matrix, class_ids):
        B, T, _ = boxes.shape
        rewards = torch.zeros(B, T, device=boxes.device)
        if matrix is None: return rewards

        for b in range(B):
            for i in range(T):
                cid_i = class_ids[b, i].item(); idx_i = int(cid_i) - 2
                if not (0 <= idx_i < 9): continue
                
                for j in range(T):
                    if i == j: continue
                    cid_j = class_ids[b, j].item(); idx_j = int(cid_j) - 2
                    if not (0 <= idx_j < 9): continue
                    
                    # [Updated] Matrix is now T x T (Instance Level)
                    # We can index directly if matrix shape matches, otherwise map
                    if matrix.shape[1] == T:
                        rel = matrix[b, i, j].item()
                    else:
                        rel = matrix[b, idx_i, idx_j].item()
                        
                    if rel == 0: continue
                    
                    box_i = boxes[b, i]; box_j = boxes[b, j]
                    reward_val = 0.0
                    
                    if rel == 1: # ABOVE
                        diff = box_j[1] - box_i[1]
                        reward_val = 1.0 if diff > 0 else 0.2 * diff 
                    elif rel == 2: # BELOW
                        diff = box_i[1] - box_j[1]
                        reward_val = 1.0 if diff > 0 else 0.2 * diff
                    elif rel == 3: # INSIDE
                        dx = abs(box_i[0] - box_j[0]); dy = abs(box_i[1] - box_j[1])
                        if dx < box_j[2]/2 and dy < box_j[3]/2: reward_val = 1.0
                    elif rel == 4: # SURROUNDS
                        dx = abs(box_i[0] - box_j[0]); dy = abs(box_i[1] - box_j[1])
                        if dx < box_i[2]/2 and dy < box_i[3]/2: reward_val = 1.0
                    elif rel == 5: # ON TOP
                        diff = box_j[1] - box_i[1]
                        reward_val = 1.0 if diff > 0 else 0.2 * diff

                    rewards[b, i] += reward_val
                        
        return torch.clamp(rewards, min=-1.0, max=3.0)

    def _calculate_overlap_penalty(self, boxes):
        B, T, _ = boxes.shape
        centers = boxes[..., :2]
        dist = torch.cdist(centers, centers)
        
        too_close = (dist < 0.1).float()
        mask = torch.eye(T, device=boxes.device).unsqueeze(0).expand(B, -1, -1)
        too_close = too_close * (1 - mask)
        
        penalty = too_close.sum(dim=2) 
        return penalty

    def _plot_reward_history(self):
        if not self.reward_history: return
        epochs = range(1, len(self.reward_history) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.reward_history, label='Avg Epoch Reward', color='purple', marker='o', linestyle='-')
        plt.title('RL Training Reward Trajectory', fontsize=14)
        plt.xlabel('Epoch', fontsize=12); plt.ylabel('Average Reward', fontsize=12)
        plt.legend(loc='upper left'); plt.grid(True, linestyle='--', alpha=0.5)
        try: plt.savefig(self.plot_path_reward); plt.close()
        except Exception: pass

    def train_rl_epoch(self, epoch):
        self.model.train()
        total_reward = 0; steps = 0
        
        print(f"\n--- [RL] Starting RL Epoch {epoch+1} (Mixed Training) ---")
        
        for step, batch in enumerate(self.train_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(self.device)
            
            # ==========================================
            # Part A: Reinforcement Learning (RL)
            # ==========================================
            # 1. Baseline (Greedy)
            self.model.eval()
            with torch.no_grad():
                baseline_boxes, _ = self.model.forward_rl(
                    batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                    batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'], sample=False)
                reward_baseline = self.compute_reward(baseline_boxes, batch)
            
            # 2. Sampling (Exploration)
            self.model.train()
            sample_boxes, log_probs = self.model.forward_rl(
                batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'], sample=True)
            reward_sample = self.compute_reward(sample_boxes, batch)
            
            # 3. Advantage & Loss
            advantage = reward_sample - reward_baseline
            log_prob_sum = log_probs.sum(dim=1)
            rl_loss = -(log_prob_sum * advantage).mean()
            
            # ==========================================
            # Part B: Supervised Anchor (Mixed Training)
            # ==========================================
            # [MODIFIED] Unpack 4 values including decoder_output
            mu, logvar, pred_boxes_sup, decoder_output = self.model(
                batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                batch['padding_mask'], batch['kg_spatial_matrix'], batch['location_grids'],
                target_boxes=batch['target_boxes'])
            
            # [MODIFIED] Pass decoder_output and unpack 11 values
            loss_tuple = self.model.get_loss(
                None, None, pred_boxes_sup, None, None, 
                batch['loss_mask'], batch['num_boxes'].to(self.device), 
                target_coords_gt=batch['target_boxes'],
                kg_spatial_matrix=batch['kg_spatial_matrix'],
                kg_class_weights=batch['kg_class_weights'],
                kg_class_ids=batch['kg_class_ids'],
                decoder_output=decoder_output # [NEW]
            )
            
            supervised_loss = loss_tuple[0]
            consistency_loss = loss_tuple[10] # Get consistency loss
            
            # 3. KL Divergence (Robust)
            if mu is not None and logvar is not None:
                kl_loss = compute_kl_loss(mu, logvar, free_bits=1.0)
            else:
                kl_loss = torch.tensor(0.0, device=self.device)

            # ==========================================
            # Part C: Total Loss & Update
            # ==========================================
            alpha = 1.0 
            
            # 最终 Loss = RL + 监督 + KL
            # 注意: consistency_loss 已经包含在 supervised_loss 中了 (在 model.get_loss 里加的)
            total_combined_loss = alpha * rl_loss + (0.3 * supervised_loss + 0.05 * kl_loss)
            
            self.optimizer.zero_grad()
            total_combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_reward += reward_sample.mean().item()
            steps += 1
            
            if (step + 1) % 10 == 0:
                stats = self.last_reward_stats
                print(f"[RL] Epoch {epoch+1} Step {step+1} | "
                      f"R_Avg: {reward_sample.mean().item():.3f} | "
                      f"Adv: {advantage.mean().item():.3f} | "
                      f"Loss: {total_combined_loss.item():.4f} || "
                      f"IoU:{stats.get('IoU', 0):.2f} "
                      f"Rel:{stats.get('Rel', 0):.2f} "
                      f"Disp:{stats.get('Disp', 0):.2f} "
                      f"Size:{stats.get('Size', 0):.2f} " # Log Size
                      f"Cons: {consistency_loss.item():.3f}") # Log Consistency

        avg_reward = total_reward / steps if steps > 0 else 0
        print(f"--- [RL] Epoch {epoch+1} Finished. Avg Epoch Reward: {avg_reward:.4f} ---")
        
        self.reward_history.append(avg_reward)
        self._plot_reward_history()
        
        return avg_reward