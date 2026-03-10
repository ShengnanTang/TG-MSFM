import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, RMSNorm, GELU2, series_decomp
import os



class SafeGELU(nn.Module):
    def __init__(self, x_clip: float = 6.0, y_clip: float = 6.0):
        super().__init__()
        self.x_clip = float(x_clip)
        self.y_clip = float(y_clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 入口夹紧，避免极端输入造成 erf/tanh 溢出
        x = x.clamp(min=-self.x_clip, max=self.x_clip)
        y = F.gelu(x, approximate='tanh')  # 更平滑的近似
        # 出口再夹紧，给下游一个硬上限
        y = y.clamp(min=-self.y_clip, max=self.y_clip)
        return y
    

def _chk(name, t):
    if not torch.isfinite(t).all():
        t_min = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).min().item()
        t_max = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).max().item()
        raise RuntimeError(f"[NaN@{name}] non-finite detected; approx min={t_min:.3e} max={t_max:.3e} shape={tuple(t.shape)}")

def _chk_finite(name: str, t: torch.Tensor):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).sum().item()
        t_ = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        t_min, t_max = t_.min().item(), t_.max().item()
        raise RuntimeError(f"[NaN@{name}] non-finite ({bad} vals); approx min={t_min:.3e} max={t_max:.3e} shape={tuple(t.shape)}")


def _masked_softmax_stable(logits, key_mask_bt):  # logits: [B, h, Tq, Tk], key_mask_bt: [B, Tk] (bool)
    # broadcast 到 [B,1,1,Tk]
    km = key_mask_bt[:, None, None, :].to(logits.dtype)

    # 先把被遮位置设成大负数，未遮位置减去行内最大值，避免 exp 溢出
    logits = logits.masked_fill(km == 0, float('-inf'))
    max_per_row = logits.amax(dim=-1, keepdim=True)
    # 全遮一行时 max 为 -inf，替换成 0，避免 -inf - (-inf)
    max_per_row = torch.where(torch.isfinite(max_per_row), max_per_row, torch.zeros_like(max_per_row))
    logits = logits - max_per_row

    # 只对未遮位置做 exp
    exp_logits = torch.exp(logits) * km
    denom = exp_logits.sum(dim=-1, keepdim=True)

    # 分母为 0（整行全遮）→ 该行返回全 0，不做除法
    att = torch.where(denom > 0, exp_logits / denom, torch.zeros_like(exp_logits))
    return att

class TrendBlock(nn.Module):
    """
    输入:  [B, in_dim=n_channel, in_feat=n_embd]
    输出:  [B, out_dim=n_channel, out_feat=n_feat]
    """
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super().__init__()
        trend_poly = 3

        # ① 预归一化（按通道），无仿射，稳定输入尺度
        self.pre_norm = nn.GroupNorm(
            num_groups=min(32, in_dim),
            num_channels=in_dim,
            eps=1e-5,
            affine=False
        )

        # ② 第一层卷积（无 bias）+ 权重范数约束
        conv1 = nn.Conv1d(in_channels=in_dim, out_channels=trend_poly,
                          kernel_size=3, padding=1, bias=False)
        self.conv1 = conv1
        try:
            # 加 weight_norm（想完全干净可注释掉）
            self.conv1 = parametrize.register_parametrization(self.conv1, "weight",
                                                              nn.utils.parametrizations.weight_norm(self.conv1).parametrizations[0])
        except Exception:
            # 不支持就忽略
            pass

        # ③ 安全 GELU（替代原激活）
        self.act = SafeGELU(x_clip=6.0, y_clip=6.0)

        # ④ 保持你的转置 + 第二层 1D 卷积
        self.transpose = Transpose(shape=(1, 2))           # -> [B, in_feat, 3]
        self.conv2 = nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1, bias=True)

        # 多项式基（buffer）
        lin_space = torch.arange(1, out_dim + 1, 1, dtype=torch.float32) / (out_dim + 1)
        poly = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)  # [3, out_dim]
        self.register_buffer("poly_space", poly)

        # ⑤ 保守初始化
        with torch.no_grad():
            # conv1: 小权重
            if hasattr(self.conv1, "weight"):
                self.conv1.weight.mul_(0.05)
            # conv2: 小权重 + 零 bias
            self.conv2.weight.mul_(0.1)
            if self.conv2.bias is not None:
                self.conv2.bias.zero_()

        # ⑥ （可选）对 conv1 的梯度做夹紧 —— 最后一层保险
        self._enable_grad_fuse = True
        if self._enable_grad_fuse:
            for p in self.parameters():
                if p.requires_grad:
                    p.register_hook(lambda g: torch.clamp(g, -1e3, 1e3))  # 需要更紧就把 1e3 改小

        # 输出缩放（第一层后面的一个小门，进一步限幅）
        self.post_scale = 0.1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: [B, C=n_channel, H=n_embd]
        dtype_in = input.dtype
        with torch.cuda.amp.autocast(False):  # ★ 全程 FP32
            x = input.float()                         # [B,C,H]
            x = self.pre_norm(x)                      # 预归一化
            x = self.conv1(x)                         # [B,3,H]
            x = self.post_scale * x                   # 小尺度输出
            x = self.act(x)                           # 安全 GELU
            x = self.transpose(x)                     # [B,H,3]
            x = self.conv2(x)                         # [B,out_feat,3]
            ps = self.poly_space.to(device=x.device, dtype=x.dtype)  # [3,out_dim]
            trend_vals = torch.matmul(x, ps).transpose(1, 2)  # [B,out_dim,out_feat]

            # 出口再做一次轻微夹紧，防止回传链条过激
            trend_vals = trend_vals.clamp(min=-50., max=50.)
        return trend_vals.to(dtype_in)

    

class MovingBlock(nn.Module):
    """
    Model trend of time series using the moving average.
    """
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals


class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple
    

class SeasonBlock(nn.Module):
    """
    Model seasonality of time series using the Fourier series.
    """
    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


### precompute_freqs_cis/reshape_for_broadcast/apply_rotary_emb are rope code adapted from LLama code
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
                 max_len=None
    ):
        super().__init__()
        assert n_embd % n_head == 0

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

        self.q_norm = RMSNorm(n_embd)
        self.k_norm = RMSNorm(n_embd)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.freqs_cis = precompute_freqs_cis(
            n_embd // n_head,
            max_len * 4 if max_len is not None else 2048 * 4,
            50000,
        )

        self.regi_num = 128
        self.register = nn.Parameter(torch.randn([1, self.regi_num, n_embd]))



    def forward(self, x, mask=None):
        
        
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        k = self.k_norm(k) 
        q = self.q_norm(q)
        _chk("att.q", q); _chk("att.k", k); _chk("att.v", v)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        _chk("att.q.hsplit", q); _chk("att.k.hsplit", k); _chk("att.v.hsplit", v)

        # ---- Rotary Positional Embedding (ROPE) 可选 ----
        if int(os.environ.get('hucfg_attention_rope_use', '-1')) == 1:
            freqs_cis = self.freqs_cis.to(self.device)[0: T]
            q, k = apply_rotary_emb(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), freqs_cis=freqs_cis)
            q, k = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        _chk("att.logits", att)

        # ============ Mask Attention 实现 ============

        if mask is not None:
            # 接受 [B,T] 或 [B,T,D]
            if mask.dim() == 3:
                key_mask_bt = mask.any(dim=-1)               # [B,T]
            else:
                key_mask_bt = mask                           # [B,T]
            att = _masked_softmax_stable(att, key_mask_bt)   # ★ 不会 NaN
        else:
            # 纯 softmax 也减个最大值，数值更稳
            att = F.softmax(att - att.amax(dim=-1, keepdim=True), dim=-1)

        #assert torch.isfinite(att).all(), "NaN/Inf after masked softmax"

        #att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)
        y = self.resid_drop(self.proj(y))
        return y, att



class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
                 max_len = None
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

        self.q_norm = RMSNorm(n_embd)
        self.k_norm = RMSNorm(n_embd)

        self.freqs_cis = precompute_freqs_cis(
            n_embd // n_head,
            max_len * 4,
            50000,  ## hucfg913
        )
     

        self.regi_num = 128
        self.register = nn.Parameter(torch.randn([1, self.regi_num, n_embd]))
        self.register_2 = nn.Parameter(torch.randn([1, self.regi_num, n_embd]))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def forward(self, x, encoder_output, mask=None):
        
        # x = torch.cat([self.register.repeat(x.shape[0],1,1), x], 1)

        # encoder_output = torch.cat([self.register_2.repeat(x.shape[0],1,1), encoder_output], 1)

        
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output)
        q = self.query(x)



        k = self.k_norm(k) 
        q = self.q_norm(q) 

        k = k.view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        freqs_cis = self.freqs_cis.to(self.device)[0 : T]
        q, k = apply_rotary_emb(q.permute(0,2,1,3), k.permute(0,2,1,3), freqs_cis=freqs_cis)
        q, k = q.permute(0,2,1,3), k.permute(0,2,1,3)

        

        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        # ============ Mask Attention 实现 ============

        if mask is not None:
            # 接受 [B,T] 或 [B,T,D]
            if mask.dim() == 3:
                key_mask_bt = mask.any(dim=-1)               # [B,T]
            else:
                key_mask_bt = mask                           # [B,T]
            att = _masked_softmax_stable(att, key_mask_bt)   # ★ 不会 NaN
        else:
            # 纯 softmax 也减个最大值，数值更稳
            att = F.softmax(att - att.amax(dim=-1, keepdim=True), dim=-1)
        # att = torch.sigmoid(att)  ## sigmoid attention infact 
            

        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)


        y = self.resid_drop(self.proj(y))
        # y = y[:,self.regi_num:,:]

        return y, att
        

class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 max_len = None
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                max_len = max_len
            )
        
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        
    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + 0.70710678*a
        x = x + 0.70710678*self.mlp(self.ln2(x))   # only one really use encoder_output
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.,
        resid_pdrop=0.,
        mlp_hidden_times=4,
        block_activate='GELU',
        max_len = None
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                max_len = max_len
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_channel,
                 n_feat,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 condition_dim=1024,
                 max_len = None
                 ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.ln2 = AdaLayerNorm(n_embd)
        self.trend_gate = nn.Parameter(torch.tensor(0.0))

        self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                max_len = max_len
                )
        self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                max_len = max_len
                )
        
        self.ln1_1 = AdaLayerNorm(n_embd)
        # self.ln1_1 = nn.LayerNorm(n_embd)

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()
        self.n_feat    = n_feat
        self.n_channel = n_channel
        self.d_model   = n_embd
        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)
        self.seasonal = FourierLayer(d_model=n_embd)


        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        res_scale = 2**-0.5   # ≈0.707

        a, _ = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + res_scale * a

        a, _ = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + res_scale * a

        # proj 仍保持你的缩放（0.1）与 /√d
        #x1, x2 = (0.1 * self.proj(x)).chunk(2, dim=1)
        x1, x2 = self.proj(x).chunk(2, dim=1)
        #scale = (self.d_model ** 0.5)
        #x1 = x1 / scale
        #x2 = x2 / scale

        trend  = self.trend(x1).clamp_(-50., 50.)
        g = torch.sigmoid(self.trend_gate)
        trend = g * trend
        season = self.seasonal(x2)

        #x = x + res_scale * self.mlp(self.ln2(x))   # ★ MLP 残差也衰减
        #season = 0.0 * x[:, :, :self.d_model]  # 形状对齐即可
        #B, C, _ = x.shape
        #trend = x.new_zeros(B, C, self.n_feat)   # [B, C, n_feat]

        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season

    

class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd=1024,
        n_head=16,
        n_layer=10,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        condition_dim=512,
        max_len = None
    ):
      super().__init__()
      self.d_model = n_embd
      self.n_feat = n_feat
      self.blocks = nn.Sequential(*[DecoderBlock(
                n_feat=n_feat,
                n_channel=n_channel,
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                condition_dim=condition_dim,
                max_len = max_len
        ) for _ in range(n_layer)])
      
    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, c, _ = x.shape
        # att_weights = []
        mean = []
        season = torch.zeros((b, c, self.d_model), device=x.device)
        trend = torch.zeros((b, c, self.n_feat), device=x.device)
        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = \
                self.blocks[block_idx](x, enc, t, mask=padding_masks, label_emb=label_emb)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season


class Transformer(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        n_layer_enc=5,
        n_layer_dec=14,
        n_embd=1024,
        n_heads=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        max_len=2048,
        conv_params=None,
        **kwargs
    ):
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)

        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.combine_s = nn.Conv1d(n_embd, n_feat, kernel_size=kernel_size, stride=1, padding=padding,
                                   padding_mode='circular', bias=False)
        self.combine_m = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0,
                                   padding_mode='circular', bias=False)
        self.max_len = max_len
        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate, max_len = self.max_len)

        self.decoder = Decoder(n_channel, n_feat, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times,
                               block_activate, condition_dim=n_embd, max_len = self.max_len)

    def forward(self, input, t, padding_masks=None, return_res=False):
        _chk_finite("emb_input", input)
        emb = self.emb(input); _chk("emb", emb)
        _chk_finite("emb", emb)
        enc_cond = self.encoder(emb, t, padding_masks=padding_masks); _chk("encoder.out", enc_cond)
        output, mean, trend, season = self.decoder(emb, t, enc_cond, padding_masks=padding_masks)
        _chk("decoder.out", output); _chk("decoder.mean", mean); _chk("decoder.trend", trend); _chk("decoder.season", season)

        res = self.inverse(output); _chk("inverse(res)", res)
        res_m = torch.mean(res, dim=1, keepdim=True); _chk("res.mean", res_m)
        season_error = self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        _chk("season_error", season_error)
        trend = self.combine_m(mean) + res_m + trend; _chk("trend", trend)
        out = trend + season_error; _chk("out(final)", out)

        return out


if __name__ == '__main__':
    pass