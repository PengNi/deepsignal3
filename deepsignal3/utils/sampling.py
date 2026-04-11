import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Downsample(nn.Module):
    def __init__(self, d_model, ratio, mode):
        super().__init__()
        self.mode = mode
        self.ratio = ratio
        if self.mode == 'concat':
            self.lin = nn.Linear(d_model * 2, d_model)

    def forward(self, x, x_mask, idx_b, idx_t, idx_c, imp):
        """
        Static Downsampling optimized for Fixed Length & Continuous Time.
        Uses Replication Padding (Last Value Padding) instead of Zero Padding.
        """
        B, T, C, D = x.shape
        r = self.ratio

        # 1. 结构性 Padding：确保 T 能被 r 整除
        if T % r != 0:
            pad_len = r - (T % r)
            
            # [修改点]：使用复制填充 (Replication Padding) 替代 0 填充
            # 取出最后一个时间步 (B, 1, C, D)
            last_x = x[:, -1:, :, :] 
            last_t = idx_t[:, -1:, :] # (B, 1, C)
            
            # 使用 expand 创建填充视图 (零拷贝，编译友好)
            x_pad = last_x.expand(-1, pad_len, -1, -1)
            t_pad = last_t.expand(-1, pad_len, -1)

            # 在时间维度拼接
            x = torch.cat([x, x_pad], dim=1)
            idx_t = torch.cat([idx_t, t_pad], dim=1)
            
            # Mask 依然需要填充 True (表示这些位置是 Padding)
            # 虽然我们填充了真实值以保持统计稳定性，但逻辑上它们不应参与 Pooling 的分母计算
            x_mask = F.pad(x_mask, (0, 0, 0, pad_len), mode='constant', value=True)
            
            # 更新 T
            T = T + pad_len

        # 2. 静态 Reshape (零拷贝，编译极快)
        # 将时间轴 T 切分为 (L, r)
        L = T // r
        x_reshaped = x.view(B, L, r, C, D)
        mask_reshaped = x_mask.view(B, L, r, C) # (B, L, r, C)

        # 3. 计算 Block Mask
        # 如果一个 r 窗口内全是 Padding (True)，则聚合后的结果也该是 Padding
        mask_block = mask_reshaped.all(dim=2) # (B, L, C)

        # 准备广播用的 Mask (B, L, r, C, 1)
        mask_broad = mask_reshaped.unsqueeze(-1)

        # 4. 执行 Pooling (数值稳定版)
        if self.mode == 'max':
            # 将 Padding 填为 -inf，防止 max 选中它
            x_filled = x_reshaped.masked_fill(mask_broad, float("-inf"))
            x_out = x_filled.max(dim=2).values
            
            # [关键修复] 将全 Padding 产生的 -inf 强制置为 0 (或最后一个有效值，视下游任务而定，通常0是安全的特征基准)
            x_out = x_out.masked_fill(mask_block.unsqueeze(-1), 0.0)

        elif self.mode == 'avg':
            # 将 Padding 填为 0，防止影响 Sum
            x_filled = x_reshaped.masked_fill(mask_broad, 0.0)
            x_sum = x_filled.sum(dim=2)
            
            # 计算分母（有效元素个数）
            valid_count = (~mask_reshaped).sum(dim=2, keepdim=True) # (B, L, 1, C)
            valid_count = valid_count.squeeze(2).unsqueeze(-1) # (B, L, C, 1)
            
            # [关键修复] 避免除以 0
            x_out = x_sum / valid_count.clamp(min=1.0)

        elif self.mode == 'concat':
            # Max Branch
            x_max_filled = x_reshaped.masked_fill(mask_broad, float("-inf"))
            x_max = x_max_filled.max(dim=2).values
            x_max = x_max.masked_fill(mask_block.unsqueeze(-1), 0.0)
            
            # Avg Branch
            x_avg_filled = x_reshaped.masked_fill(mask_broad, 0.0)
            x_sum = x_avg_filled.sum(dim=2)
            valid_count = (~mask_reshaped).sum(dim=2, keepdim=True).squeeze(2).unsqueeze(-1)
            x_avg = x_sum / valid_count.clamp(min=1.0)
            
            # Concat & Project
            x_out = self.lin(torch.cat([x_max, x_avg], dim=-1))

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # 5. 更新 Mask 和 idx_t
        x_mask_out = mask_block

        # 静态生成新的 idx_t
        # 对于连续时间，新的 idx_t 就是旧的每隔 r 取一个
        idx_t_reshaped = idx_t.view(B, L, r, C)
        idx_t_out = idx_t_reshaped[:, :, 0, :] # 取每个窗口的第 0 个时间

        return x_out, x_mask_out, idx_t_out


class DownsampleLayer(nn.Module):
    def __init__(self, d_model, ratio, mode):
        super().__init__()
        self.mode = mode
        self.ratio = ratio
        
        if self.mode in ['max', 'avg', 'concat']:
            self.down = Downsample(d_model, ratio, mode)
        else:
            self.down = None 

    def forward(self, x, x_mask, idx_b, idx_t, idx_c, imp):
        if self.down is not None:
            return self.down(x, x_mask, idx_b, idx_t, idx_c, imp)
        return x, x_mask, idx_t