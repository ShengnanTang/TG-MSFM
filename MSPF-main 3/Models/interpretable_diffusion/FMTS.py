import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from Models.interpretable_diffusion.transformer import Transformer
import os
from typing import Optional

# ==== Multi-Scale building blocks ====
class AntiAlias1D(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.channels = channels
        self.pad = kernel_size // 2
        k = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
        k = (k / k.sum()).view(1, 1, -1)
        self.register_buffer("kernel", k)

    def forward(self, x):  # x: [B,C,T]
        w = self.kernel.expand(self.channels, 1, -1)      # [C,1,K]
        return F.conv1d(x, w, padding=self.pad, groups=self.channels)


class Pyramid1D(nn.Module):
    """
    固定多尺度金字塔：
      - 下采样：AvgPool1d
      - 上采样：线性插值
    """
    def __init__(self, scales=(1, 2, 4)):
        super().__init__()
        self.scales = tuple(scales)

    def down(self, x: torch.Tensor, s: int) -> torch.Tensor:
        # x: [B, D, T] -> [B, D, ceil(T/s)]
        if s == 1:
            return x
        return F.avg_pool1d(x, kernel_size=s, stride=s, ceil_mode=True)

    def up(self, x: torch.Tensor, T_target: int, s: int) -> torch.Tensor:
        # x: [B, D, T_s] -> [B, D, T_target]
        if s == 1:
            return x
        return F.interpolate(x, size=T_target, mode='linear', align_corners=False)


class ScaleHead(nn.Module):
    """
    每个尺度的轻量“速度头”：Conv1d -> GELU -> Conv1d
    输入/输出通道数都为 D_all（与 Transformer 输出对齐）
    """
    def __init__(self, d_in: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, d_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_hidden, d_in, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T_s] -> [B, D, T_s]
        return self.net(x)


class Gate(nn.Module):
    """
    多尺度门控：根据 (t, 轻统计) 输出各尺度权重 alpha（Softmax）
    统计项：每尺度的能量 + 一阶差分能量（稳定、廉价）
    """
    def __init__(self, n_scales: int, stat_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1 + stat_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_scales)
        )

    def forward(self, t_b: torch.Tensor, stats_bf: torch.Tensor) -> torch.Tensor:
        """
        t_b:      [B]  —— 0~1 的时间标量（若传入已*1000，先归一化）
        stats_bf: [B, stat_dim]
        return:   [B, n_scales]（Softmax 后）
        """
        h = torch.cat([t_b.view(-1, 1), stats_bf], dim=1)
        return torch.softmax(self.mlp(h), dim=-1)


class FM_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=5,
            n_layer_dec=6,
            d_model=None,
            n_heads=4,
            mlp_hidden_times=4,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            **kwargs
    ):
        super(FM_TS, self).__init__()

        self.seq_length = seq_length
        self.feature_size = feature_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.filter_infer = False
        self.s_max = nn.Parameter(torch.tensor(2.0))

        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], **kwargs)

        self.alpha = 3  ## t shifting, change to 1 is the uniform sampling during inference
        self.time_scalar = 1000 ## scale 0-1 to 0-1000 for time embedding

        self.num_timesteps = int(os.environ.get('hucfg_num_steps', '100'))

        # ==== Multi-Scale（零参默认）====
        self.ms_enable  = True
        self.ms_scales  = (1, 2, 4)
        self.ms_pyr     = Pyramid1D(scales=self.ms_scales)

        hidden_dim = 64  # 轻量稳定
        self.ms_heads = nn.ModuleList([
            ScaleHead(d_in=self.feature_size, d_hidden=hidden_dim)
            for _ in self.ms_scales
        ])
        self.aa_fine = AntiAlias1D(self.feature_size, kernel_size=5)  # 细尺度轻滤波
        self.ms_gate = Gate(n_scales=len(self.ms_scales), stat_dim=0)

        # ==== 自动损失权重（代替手工 λ）====
        self.log_sigma_smooth = nn.Parameter(torch.tensor(-2.3))  # ≈0.10
        self.log_sigma_curv   = nn.Parameter(torch.tensor(-3.9))  # ≈0.02

        self.log_sigma_bcl    = nn.Parameter(torch.tensor(-2.8))  # ≈0.06


    def output(self, x, t, mask: Optional[torch.Tensor] = None, padding_masks=None):
        """
        统一维度：
        - 进入 Transformer 前： [B,D,T]
        - 返回：                 [B,T,D_all]   （与 _train_loss 的使用一致）
        B0 路线不使用 mask/padding。
        """
        # ---- 1) 统一输入为 [B,D,T] 再喂主干 ----
        if x.dim() != 3:
            raise ValueError(f"x must be 3D, got {x.shape}")

        if x.shape[1] == self.seq_length and x.shape[2] == self.feature_size:
            # x is [B,T,D] -> [B,D,T]
            x_cf = x
        elif x.shape[1] == self.feature_size and x.shape[2] == self.seq_length:
            # x is [B,D,T]
            x_cf = x.permute(0, 2, 1).contiguous()
        else:
            # 容错：若最后一维像 T
            if x.shape[-1] == self.seq_length:
                x_cf = x if x.shape[1] == self.feature_size else x.permute(0, 2, 1).contiguous()
            else:
                raise ValueError(f"ambiguous x shape {x.shape} vs (T={self.seq_length}, D={self.feature_size})")

        # 主干（Transformer 期望 [B,D,T]）
        h = self.model(x_cf, t, padding_masks=mask)  # 输出可能是 [B,D,T] 或 [B,T,D]

        # ---- 2) 转成 [B,D,T] 做多尺度 ----
        if h.shape[1] == self.feature_size and h.shape[2] == self.seq_length:
            h_cf = h
        elif h.shape[1] == self.seq_length and h.shape[2] == self.feature_size:
            h_cf = h.permute(0, 2, 1).contiguous()
        else:
            # 默认当作 [B,D,T]
            h_cf = h

        B, D_all, T = h_cf.shape
        if D_all != self.feature_size or T != self.seq_length:
            raise ValueError(f"backbone out shape {h.shape} mismatches (D={self.feature_size}, T={self.seq_length})")

        if not self.ms_enable:
            # 不做多尺度，直接回 [B,T,D_all]
            return h_cf.permute(0, 2, 1).contiguous()

        # ---- 3) 多尺度：down -> head -> up -> (细尺度轻滤波) ----
        v_per_scale = []
        for s_idx, s in enumerate(self.ms_scales):
            h_s  = self.ms_pyr.down(h_cf, s)        # [B,D,T_s]
            v_s  = self.ms_heads[s_idx](h_s)        # [B,D,T_s]
            v_up = self.ms_pyr.up(v_s, T, s)        # [B,D,T]
            if s == 1:
                v_up = self.aa_fine(v_up)           # 细尺度小滤波，抑制锯齿
            v_per_scale.append(v_up)

        # ---- 4) 零参粗→细时间门控（只依赖尺度数 S）----
        S = len(self.ms_scales)
        t_norm = (t.float() / self.time_scalar).clamp_(0., 1.)
        # stats 先留空（维度 0），后续要的话再扩展
        stats = torch.zeros(h_cf.size(0), 0, device=h_cf.device)
        alpha = self.ms_gate(t_norm, stats).view(B, S, 1, 1)     # [B,S,1,1]
        v_ms_cf = (alpha * torch.stack(v_per_scale, dim=1)).sum(dim=1)  # [B,D,T]
        v_ms_cf = torch.tanh(v_ms_cf / 2.0) * self.s_max

        return v_ms_cf.permute(0, 2, 1).contiguous()




    @torch.no_grad()
    def sample(self, shape):
        self.eval()
        zt = torch.randn(shape, device=self.device)
        timesteps = torch.linspace(0, 1, self.num_timesteps+1, device=self.device)
        t_shifted = 1 - (self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0)

        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            step = t_prev - t_curr
            v = self.output(zt.clone(),
                            torch.full((shape[0],), t_curr*self.time_scalar,
                                    device=self.device),
                            mask=None)
            if v.shape[-1] > shape[-1]:
                v = v[:, :, :shape[-1]]
            zt = zt + step * v
        return zt

    def generate_mts(self, batch_size=16):
        feature_size, seq_length = self.feature_size, self.seq_length
        return self.sample((batch_size, seq_length, feature_size))


    def _train_loss(self, x_start, mask=None):
        """
        x_start: [B, D_all, T]   （C1 时 D_all = 数据D + 3）
        mask:    [B, T, D_data]  True=观测
        """
        # —— 超参：混合权重（可调）——
        w_gap    = 0.8
        w_global = 1.0 - w_gap

        # 1) 到 [B,T,D_all]，并确定“参与监督的”数据通道数 D
        x_start = x_start.permute(0, 2, 1)          # [B,T,D_all]
        D_all   = x_start.shape[2]
        D       = D_all if mask is None else (D_all - 3)  # C1: 只监督数据通道

        # 2) FM 训练对：z0, z1, z_t, target
        z0 = torch.randn_like(x_start)
        z1 = x_start
        t  = torch.rand(z0.shape[0], 1, 1, device=z0.device)
        z_t    = (1. - t) * z0 + t * z1
        target = z1 - z0

        # 3) 注意力掩码（给 Transformer 用 2D），主损失掩码（3D 用于挑 gap）
        attn_mask_bt, mask_exp = None, None
        if mask is not None:
            # mask: [B,T,D_data]
            attn_mask_bt = mask.any(dim=-1)                # [B,T] 供注意力
            mask_exp     = mask[:, :, :D].permute(0, 2, 1) # [B,D,T] 供主损失

        # 4) 前向，拿速度场
        '''
        print_once = (torch.randint(0, 100, ()).item() == 0)  # 只偶尔打，避免刷屏
        if print_once:
            print(f"[CHK] t_raw min/max={t.min().item():.3e}/{t.max().item():.3e}, time_scalar={self.time_scalar}")
            print(f"[CHK] z_t stats: min={z_t.min().item():.3e} max={z_t.max().item():.3e} std={z_t.std().item():.3e}")
        if not torch.isfinite(z_t).all():
            raise RuntimeError("NaN/Inf in z_t BEFORE Transformer")
        '''
        model_out = self.output(z_t, t.squeeze() * self.time_scalar, attn_mask_bt)  # [B,T,D_all]
        model_out_feat = model_out.permute(0, 2, 1)[:, :D, :]       # [B,D,T]
        target_feat    = target[:, :, :D].permute(0, 2, 1)          # [B,D,T]

        mse_feat = (model_out_feat - target_feat) ** 2              # [B,D,T]

        # 5) 混合损失：gap-only + 全局
        global_loss = mse_feat.mean()
        if mask_exp is not None:
            inv = ~mask_exp                                         # gap 区
            gap_loss = mse_feat[inv].mean() if inv.any() else global_loss
        else:
            gap_loss = global_loss

        main_loss = w_gap * gap_loss + w_global * global_loss

        # 6) 轻正则（可保留 / 可关）
        diff1 = model_out_feat[:, :, 1:] - model_out_feat[:, :, :-1]
        curv  = model_out_feat[:, :, 2:] - 2 * model_out_feat[:, :, 1:-1] + model_out_feat[:, :, :-2]
        smooth_loss = (diff1 ** 2).mean()
        curv_loss   = (curv  ** 2).mean()

        Vf  = torch.fft.rfft(model_out_feat, dim=-1)
        cut = int(0.25 * Vf.shape[-1])
        spec_loss = (Vf[:, :, cut:].abs() ** 2).mean()

        total_loss = main_loss + 0.05 * smooth_loss + 0.01 * curv_loss + 1e-4 * spec_loss

        '''
        # ==================  新增：BCL（不确定性权重）  ==================
        # 思路：在“缺口边界”时刻（观测↔缺失的交界处），约束 v_theta(z_t, t) ≈ (z1 - z0)。
        # 具体实现：在 [B,D,T] 上找边界位置，把这些位置上的 (model_out_feat - target_feat)^2 取均值。
        bcl = torch.zeros((), device=model_out_feat.device)
        if mask_exp is not None:
            # 边界：t 与 t-1 的可见性不同处（XOR）
            edges = mask_exp[:, :, 1:] ^ mask_exp[:, :, :-1]            # [B, D, T-1] (bool)
            if edges.any():
                bcl_idx = torch.zeros_like(mask_exp, dtype=torch.bool)  # [B, D, T]
                bcl_idx[:, :, 1:] |= edges
                bcl_idx[:, :, :-1] |= edges
                bcl = ((model_out_feat - target_feat) ** 2)[bcl_idx].mean()

        # 不确定性加权：exp(-2 logσ) * L + logσ
        w_bcl = torch.exp(-2.0 * self.log_sigma_bcl)
        total_loss = total_loss + w_bcl * bcl + self.log_sigma_bcl
        '''

        return total_loss.view([])


        
        # mask 展开 [B, D, T]
        if mask is not None:
            mask_exp = mask.unsqueeze(1).expand(-1, D, -1)
            assert mask_exp.shape == model_out_feat.shape, f"mask_exp.shape={mask_exp.shape}, model_out_feat.shape={model_out_feat.shape}"
        else:
            mask_exp = None

        '''
        print("[_train_loss] model_out shape:", model_out.shape)         # [B, D_all, T]
        print("[_train_loss] model_out_feat shape:", model_out_feat.shape)  # [B, D, T]
        print("[_train_loss] mask_exp shape:", mask_exp.shape)           # [B, D, T]
        '''
        # 主损失
        if mask_exp is not None:
            main_loss = mse_feat[~mask_exp].mean() if (~mask_exp).sum() > 0 else torch.tensor(0., device=mse_feat.device)
        else:
            main_loss = mse_feat.reshape(-1).mean()

        # 全局平滑
        smooth_loss = ((model_out_feat[:, :, 1:] - model_out_feat[:, :, :-1]) ** 2).mean()
        # 二阶曲率
        curv = model_out_feat[:, :, 2:] - 2 * model_out_feat[:, :, 1:-1] + model_out_feat[:, :, :-2]
        curv_loss = (curv ** 2).mean()

        # 自动权重（Kendall & Gal, 2018）
        w_smooth = torch.exp(-2.0 * self.log_sigma_smooth)
        w_curv   = torch.exp(-2.0 * self.log_sigma_curv)

        # gap区一阶二阶差分
        if mask_exp is not None:
            diff1 = model_out_feat[:, :, 1:] - model_out_feat[:, :, :-1]
            gap_inside = (~mask_exp[:, :, 1:] & ~mask_exp[:, :, :-1])
            gap_smooth_loss_1 = (diff1[gap_inside] ** 2).mean() if gap_inside.sum() > 0 else torch.tensor(0., device=main_loss.device)

            diff2 = model_out_feat[:, :, 2:] - 2 * model_out_feat[:, :, 1:-1] + model_out_feat[:, :, :-2]
            gap_inside2 = (~mask_exp[:, :, 2:] & ~mask_exp[:, :, 1:-1] & ~mask_exp[:, :, :-2])
            gap_smooth_loss_2 = (diff2[gap_inside2] ** 2).mean() if gap_inside2.sum() > 0 else torch.tensor(0., device=main_loss.device)
        else:
            gap_smooth_loss_1 = torch.tensor(0., device=main_loss.device)
            gap_smooth_loss_2 = torch.tensor(0., device=main_loss.device)

        # 上下文均值loss
        if mask_exp is not None:
            context_left = (model_out_feat[:, :, :10] * mask_exp[:, :, :10]).sum(dim=2) / (mask_exp[:, :, :10].sum(dim=2) + 1e-6)
            context_right = (model_out_feat[:, :, -10:] * mask_exp[:, :, -10:]).sum(dim=2) / (mask_exp[:, :, -10:].sum(dim=2) + 1e-6)
            gap_mean = (model_out_feat * (~mask_exp)).sum(dim=2) / ((~mask_exp).sum(dim=2) + 1e-6)
            mean_loss = ((gap_mean - context_left) ** 2 + (gap_mean - context_right) ** 2).mean()
        else:
            mean_loss = torch.tensor(0., device=main_loss.device)

        # gap区方差loss
        if mask_exp is not None:
            gap_output = model_out_feat[~mask_exp]
            var_loss = ((gap_output - gap_output.mean()) ** 2).mean() if gap_output.numel() > 0 else torch.tensor(0., device=main_loss.device)
        else:
            var_loss = torch.tensor(0., device=main_loss.device)

        # gap区极值惩罚
        if mask_exp is not None:
            gap_output = model_out_feat[~mask_exp]
            reg_penalty = (gap_output.abs() > 1.5).float() * (gap_output.abs() - 1.5)
            reg_loss = reg_penalty.mean() if reg_penalty.numel() > 0 else torch.tensor(0., device=main_loss.device)
        else:
            reg_loss = torch.tensor(0., device=main_loss.device)

        # 边缘上下文loss
        if mask_exp is not None:
            context_mask = mask.clone()
            context_mask[:, 10:-10] = False
            context_mask_exp = context_mask.unsqueeze(1).expand(-1, D, -1)
            assert context_mask_exp.shape == model_out_feat.shape
            context_target = target_feat[context_mask_exp]
            context_pred = model_out_feat[context_mask_exp]
            context_loss = F.mse_loss(context_pred, context_target) if context_target.numel() > 0 else torch.tensor(0., device=main_loss.device)
        else:
            context_loss = torch.tensor(0., device=main_loss.device)

        # Gap边界loss
        if mask_exp is not None:
            diff = model_out_feat[:, :, 1:] - model_out_feat[:, :, :-1]
            edge_mask = (~mask_exp).float()
            edge_weight = (edge_mask[:, :, 1:] - edge_mask[:, :, :-1]).abs()
            boundary_loss = (diff ** 2 * edge_weight).mean()
        else:
            boundary_loss = torch.tensor(0., device=main_loss.device)

        total_loss = (
            1.2 * main_loss +
            w_smooth * smooth_loss + self.log_sigma_smooth +
            w_curv   * curv_loss   + self.log_sigma_curv +
            0.2 * context_loss +
            0.2 * boundary_loss +
            2.0 * gap_smooth_loss_1 +
            2.0 * gap_smooth_loss_2 +
            2.0 * reg_loss +
            1.0 * mean_loss +
            0.1 * var_loss
        )
        return total_loss.view([])


    def forward(self, x, mask=None):
        #print("[DEBUG] FMTS forward called!")
        return self._train_loss(x_start=x, mask=mask)


    @torch.no_grad() # 确保在推理模式下不计算梯度
    def fast_sample_infill(self, shape, target_x1_bdt, partial_mask_btd=None):
        """s
        使用 ODE 求解器进行条件插值 (Flow Matching) - 完整版。
        该版本对所有通道 (数据 + 条件) 进行联合ODE演化，并通过数据一致性步骤
        确保已知信息被尊重。

        参数:
        shape (tuple): 原始数据部分的形状 [B, T_seq_len, D_data]。
                       B: 批次大小, T_seq_len: 序列长度, D_data: 原始数据特征维度。
        target_x1_bdt (torch.Tensor): 目标 x1，形状为 [B, D_total, T_seq_len]。
                                    D_total 是模型的总特征维度 (D_data + 条件通道数)。
                                    这代表了在 t=1 时的目标状态，包含了已知数据和条件信息。
        partial_mask_btd (torch.Tensor, optional):
                                    数据部分已知位置的布尔掩码，形状 [B, T_seq_len, D_data]。
                                    True 表示对应位置的数据是已知的。默认为 None。

        返回:
        torch.Tensor: 插值后的数据，形状为 [B, D_data, T_seq_len]。
        """
        self.eval() # 确保模型处于评估模式

        B, T_seq_len, D_data = shape
        _B_check, D_total, _T_check = target_x1_bdt.shape

        assert B == _B_check, f"Batch size mismatch in shape ({B}) and target ({_B_check})"
        assert T_seq_len == _T_check, f"Sequence length mismatch in shape ({T_seq_len}) and target ({_T_check})"
        assert D_total == self.feature_size, f"Feature size mismatch: model expects {self.feature_size}, target has {D_total}"

        is_c1 = (D_total > D_data) # 判断是否为C1模式 (带额外条件通道)

        # 1. 初始化 z_t 在 t=0 时的状态为纯高斯噪声 (z0)
        #    形状为 [B, D_total, T_seq_len]
        current_zt_bdt = torch.randn((B, D_total, T_seq_len), device=self.device)
        current_zt_bdt = current_zt_bdt.permute(0, 2, 1)
        # 保存初始噪声，用于数据一致性步骤中的路径重投影
        initial_noise_bdt = current_zt_bdt.clone() # 这就是 z0
        initial_noise_bdt = initial_noise_bdt.permute(0, 2, 1)

        # 2. 设置 ODE 求解的时间步长和 Kscale
        # self.num_timesteps 从环境变量 hucfg_num_steps 获取
        # k_scale 用于调整时间步的非线性进展，从环境变量 hucfg_Kscale 获取
        k_scale = float(os.environ.get('hucfg_Kscale', 1.0))

        # 3. ODE 迭代求解
        for step_idx in tqdm(range(self.num_timesteps), desc="Conditional Infilling Steps (Full ODE)"):
            # 计算当前物理时间 t_curr 和有效时间步长 dt
            t_curr_scalar = step_idx / self.num_timesteps       # 物理时间 t，从 0 增加到接近 1
            t_next_scalar = (step_idx + 1) / self.num_timesteps # 下一个物理时间点

            # 应用 Kscale 变换
            t_eff_curr = t_curr_scalar ** k_scale
            t_eff_next = t_next_scalar ** k_scale
            dt = t_eff_next - t_eff_curr # 有效的ODE步长，应为正

            # 为模型准备时间输入 t_input_model，形状为 [B]
            t_input_model_b = torch.full((B,), t_eff_curr * self.time_scalar, device=self.device)
            
            # 准备 Transformer 的注意力掩码 (如果 partial_mask 提供了)
            # mask_1d_bt: [B, T_seq_len]。假设 Transformer 期望 True=不mask, False=mask。
            mask_1d_bt_for_transformer = None
            if partial_mask_btd is not None:
                mask_1d_bt_for_transformer = partial_mask_btd.any(dim=-1) # True 表示该时间点至少有一个数据特征是已知的

            # 从模型获取预测的完整速度场 v(z_t, t, condition)
            # 输入 current_zt_bdt 的形状是 [B, D_total, T_seq_len]
            # 输出 v_bdt 的形状也应该是 [B, D_total, T_seq_len]
            v_bdt = self.output(current_zt_bdt.clone(),
                                t_input_model_b,
                                mask=mask_1d_bt_for_transformer)
            

            # 执行一步欧拉法
            
            #print("shape:v_bdt",v_bdt.shape)
            v1 = self.output(current_zt_bdt.clone(), t_input_model_b, mask=mask_1d_bt_for_transformer)

            # predictor
            z_pred = current_zt_bdt + dt * v1
            t_input_next_b = torch.full((B,), t_eff_next * self.time_scalar, device=self.device)

            # v2 at t_eff_next
            v2 = self.output(z_pred.clone(), t_input_next_b, mask=mask_1d_bt_for_transformer)

            # Heun update
            next_zt_bdt_predicted_by_ode = current_zt_bdt + 0.5 * dt * (v1 + v2)

            
            # 数据一致性步骤：
            # 对于已知的观测部分 (数据通道) 和 固定的条件通道 (C1模式下)，
            # 将 z_t "拉回" 到从初始噪声 z0 到目标 x1 的理想（通常是直线）路径上。
            # z_t_ideal = (1 - t_eff_next) * z0 + t_eff_next * x1
            if partial_mask_btd is not None: # 如果有数据部分的掩码
                # 创建一个针对 D_total 通道的完整掩码，标记哪些值是“已知”或需要被强制的
                # 初始时，假设所有通道都是未知的（即不由掩码直接约束）
                full_known_mask_bdt = torch.zeros_like(target_x1_bdt, dtype=torch.bool, device=self.device)

                # 1. 设置数据部分的已知掩码
                # partial_mask_btd: [B, T_seq_len, D_data] -> permute to [B, D_data, T_seq_len]
                data_known_mask_bdt = partial_mask_btd.permute(0, 2, 1)  # [B,D_data,T]
                full_known_mask_bdt[:, :D_data, :] = data_known_mask_bdt
                # 2. C1模式下，假设条件通道总是“已知”的，它们的值在 target_x1_bdt 中是确定的
                if is_c1:
                    full_known_mask_bdt[:, D_data:, :] = True # 将条件通道标记为始终已知
                
                full_known_mask_bdt = full_known_mask_bdt.permute(0, 2, 1)
                #print("▶ full_known_ms :", full_known_mask_bdt.shape)  
                
                # 计算在 t_eff_next 时刻，这些"已知"通道的理想 z_t 值
                # target_x1_bdt 是 t=1 时的目标状态 (x1)
                # initial_noise_bdt 是 t=0 时的初始噪声 (z0)
                ideal_path_zt_at_t_next = (1 - t_eff_next) * initial_noise_bdt + \
                                          t_eff_next * target_x1_bdt
                
                ideal_path_zt_at_t_next = ideal_path_zt_at_t_next.permute(0, 2, 1)
                #print("▶ ideal_path_zt_at_t_next :", ideal_path_zt_at_t_next.shape)  
                #print("▶ next_zt_bdt_predicted_by_ode :", next_zt_bdt_predicted_by_ode.shape)  

                # 将 next_zt_bdt_predicted_by_ode 中对应于 "已知" 通道的部分，替换为理想路径上的值
                next_zt_bdt = torch.where(full_known_mask_bdt, ideal_path_zt_at_t_next, next_zt_bdt_predicted_by_ode)
            else:
                # 如果没有提供 partial_mask (例如纯无条件生成，虽然此函数主要用于条件)，
                # 或者不执行数据一致性，则直接使用ODE的预测结果
                next_zt_bdt = next_zt_bdt_predicted_by_ode
            
            # 更新 z_t 到下一个时间步的状态
            current_zt_bdt = next_zt_bdt
            
            # 在迭代过程中限制 z_t 的范围
            current_zt_bdt = torch.clamp(current_zt_bdt, min=-1.1, max=1.1)
            #print("shape:current_zt_bdt",current_zt_bdt.shape)

        # 4. ODE求解完成，current_zt_bdt 近似于 t=1 时的 z1
        final_output_bdt = torch.clamp(current_zt_bdt, min=-1.0, max=1.0)
        #print(f"[DEBUG FMTS Infill Full] Final output shape (data part): {final_output_bdt.shape}")

        final_output_bdt=final_output_bdt.permute(0, 2, 1)
        #print(f"[DEBUG FMTS Infill Full] Final output shape (data part): {final_output_bdt.shape}")

        # 5. 如果是C1模式，只返回数据部分的通道
        if is_c1:
            final_output_bdt = final_output_bdt[:, :D_data, :]
        
        #print(f"[DEBUG FMTS Infill Full] Final output shape (data part): {final_output_bdt.shape}")
        #print(f"[TRACE FMTS Infill Full] Final output (data part, norm) min={final_output_bdt.min().item():.3f}, max={final_output_bdt.max().item():.3f}")
        return final_output_bdt