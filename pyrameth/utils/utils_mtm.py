import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers import DropPath

def precompute_rpe(dim: int, max_len=640, theta=10000.0):
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(max_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    rpe = torch.polar(torch.ones_like(freqs), freqs)
    return rpe

def apply_rel_pe_qk(xq, xk, pos, rpe):
    d_model = xq.shape[-1]
    xq_ = xq.float().reshape(-1, d_model // 2, 2)
    xk_ = xk.float().reshape(-1, d_model // 2, 2)
    pos = pos.reshape(-1)
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    xq_ = torch.view_as_real(xq_ * rpe[pos, :])
    xk_ = torch.view_as_real(xk_ * rpe[pos, :])
    return xq_.type_as(xq).flatten(1), xk_.type_as(xk).flatten(1)

def precompute_ape(d_model, max_len=640, theta=10000.0):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(theta) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def apply_abs_pe(x, pos, ape):
    pe = ape[pos.flatten(), :]
    return x + pe.reshape(*(pos.shape), -1)

PE_QK_FUNC = {'rel': apply_rel_pe_qk}

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    def forward(self, x):
        return x * self.gamma

class MLP(nn.Module):
    def __init__(self, d_model, r_hid, drop=0.1, norm_first=True, layer_scale=True):
        super().__init__()
        ls = LayerScale(d_model) if layer_scale else nn.Identity()
        self.net = nn.Sequential(nn.Linear(d_model, d_model * r_hid), nn.GELU(),
                                 nn.Linear(d_model * r_hid, d_model), ls,
                                 DropPath(drop))
        self.norm = nn.LayerNorm(d_model)
        self.norm_first = norm_first
    def forward(self, x, x_mask=None):
        if self.norm_first:
            x = x + self.net(self.norm(x))
        else:
            x = self.norm(x + self.net(x))
        return x

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, r_hid, drop=0.1, norm_first=True, layer_scale=True):
        super().__init__()
        hidden_dim = int(d_model * r_hid * 2 / 3) 
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(d_model)
        self.norm_first = norm_first
        
        self.layer_scale = LayerScale(d_model) if layer_scale else nn.Identity()
        self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()

        # [重要] 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x, x_mask=None):
        # 1. Norm
        input_x = self.norm(x) if self.norm_first else x
        
        x1 = self.w1(input_x)
        x2 = self.w2(input_x)
        hidden = self.act(x1) * x2
        out = self.drop(self.w3(hidden))
        
        # 3. Residual & Post-Process
        if self.norm_first:
            x = x + self.drop_path(self.layer_scale(out))
        else:
            x = self.norm(x + self.drop_path(self.layer_scale(out)))
        return x


# ==========================================
# 优化组件 1: RMSNorm (替代 LayerNorm，收敛更快更稳)
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g



class TemporalAttn(nn.Module):
    def __init__(self, d_model, drop=0.1, norm_first=False, layer_scale=True):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(drop)
        self.layer_norm = RMSNorm(d_model)#nn.LayerNorm
        self.norm_first = norm_first
        self.layer_scale = LayerScale(d_model) if layer_scale else nn.Identity()
        self.d_head = d_model

    def _attn_block(self, x, x_mask, pos, pe, pe_type='rel'):
        bsz, nt, nc, nd = x.shape
        if pe_type in PE_QK_FUNC:
            xq, xk = PE_QK_FUNC[pe_type](self.wq(x), self.wk(x), pos, pe)
            xq = rearrange(xq, "(b t c) d -> (b c) t d", b=bsz, t=nt)
            xk = rearrange(xk, "(b t c) d -> (b c) t d", b=bsz, t=nt)
        else:
            xq, xk = self.wq(x), self.wk(x)
            xq = rearrange(xq, "b t c d -> (b c) t d")
            xk = rearrange(xk, "b t c d -> (b c) t d")
        
        xv = rearrange(self.wv(x), "b t c d -> b c t d")
        
        # Einsum 是最稳的
        attn = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(self.d_head)
        attn = rearrange(attn, "(b c) tq tk -> b c tq tk", b=bsz)

        mask = x_mask.transpose(1, 2)
        mask = mask[:, :, :, None] | mask[:, :, None, :]
        attn = torch.masked_fill(attn, mask, float('-inf'))
        # nan_to_num 保护
        attn = self.drop(F.softmax(attn, -1).nan_to_num(0))

        out = torch.einsum("bcmn,bcnd->bmcd", attn, xv)
        return out, attn

    def forward(self, x, x_mask, pos, pe, pe_type='rel'):
        if self.norm_first:
            out, attn = self._attn_block(self.layer_norm(x), x_mask, pos, pe, pe_type)
            out = x + self.layer_scale(out)
        else:
            out, attn = self._attn_block(x, x_mask, pos, pe, pe_type)
            out = self.layer_norm(x + self.layer_scale(self.drop(out)))
        return out, attn


class CLSHead(nn.Module):
    def __init__(self, d_model, d_static, num_cls, drop=0.1):
        super().__init__()
        d_out = d_model + d_static
        if d_static > 0:
            self.net = nn.Sequential(
                nn.Linear(d_out, d_model * 4), nn.GELU(), nn.Dropout(drop),
                nn.Linear(d_model * 4, num_cls)
            )
            nn.init.xavier_uniform_(self.net[0].weight)
            nn.init.xavier_uniform_(self.net[3].weight)
        else:
            self.net = nn.Linear(d_out, num_cls)
            nn.init.xavier_uniform_(self.net.weight)
    def forward(self, x):
        return self.net(x)