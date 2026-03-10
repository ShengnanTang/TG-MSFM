#!/usr/bin/env python
# ----------------------------------------------------------
# eval_infill.py ― 连续长缺口（gap_len）插值评估 + 绘图
# 用法示例：
#   python scripts/eval_infill.py \
#          --config  configs/etth_gap1000.yaml \
#          --ckpt    10 \
#          --name    etth_gap1k \
#          --plot_num 3
# ----------------------------------------------------------
import os, argparse, yaml, json, numpy as np, matplotlib.pyplot as plt, torch
from pathlib import Path
from Utils.io_utils import instantiate_from_config
from Data.build_dataloader import build_dataloader_cond
from engine.solver import Trainer # Trainer is needed for make_ctx_feature and interpolate_with_noise
from scipy.stats import pearsonr
import math
from tqdm.auto import tqdm

# import torch.nn.functional as F # Not used directly here

# ----------------------- CLI -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config',    required=True)
parser.add_argument('--ckpt',      type=int, required=True, help='checkpoint‑id')
parser.add_argument('--name',      required=True,          help='实验简称，用于输出目录')
parser.add_argument('--plot_num',  type=int, default=1,    help='绘图样本数')
parser.add_argument('--num_steps', type=int, default=400, help='ODE steps (Heun)')
parser.add_argument('--k_scale',   type=float, default=1.5, help='time-warp exponent')
parser.add_argument('--rand_ratios', type=str, default='0.1,0.3,0.5,0.7',
                    help='random missing ratios (comma-separated)')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()


torch.cuda.empty_cache()

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


# ---------------- configuration --------------------
with open(args.config) as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model structure, weights will be loaded via Trainer
model_for_trainer  = instantiate_from_config(cfg['model']).to(device)

# -------- build TEST dataloader --------------------
ns = argparse.Namespace(
    save_dir='OUTPUT', # This might be used by dataset if it saves anything during init
    mode='infill',
    # period='test', # period is usually handled by dataset config in YAML
    missing_ratio=None, # For infill, missing_ratio/pred_len not directly used by dataloader_cond
    pred_len=None,      # if gap generation is handled by CustomDataset's test mode
    long_len=cfg['model']['params']['seq_length'] # Not sure where this is used
)

dl_info   = build_dataloader_cond(cfg, ns)
test_dl   = dl_info['dataloader']
dataset_instance = dl_info['dataset'] # Get the dataset instance for unnormalize and shape
shape     = [dataset_instance.window, dataset_instance.var_num]
os.environ['results_folder'] = f"{args.name}" # Ensure this matches saving path
print(f"[INFO] Attempting to load checkpoint from base folder: {os.environ['results_folder']}")
# ---------------- Trainer & checkpoint -------------
# We need a Trainer instance to easily access make_ctx_feature and interpolate_with_noise,
# and also for its load method.
# The dataloader passed to Trainer here is not strictly for training, but its 'load' method uses it.
# For inference, we iterate over test_dl directly.
trainer = Trainer(cfg,
                  argparse.Namespace(name=args.name), # dummy Args for Trainer init if it expects 'name'
                  model_for_trainer, # Pass the model instance
                  dataloader={'dataloader': test_dl}, # Dummy dataloader for Trainer init
                  logger=None)


trainer.load(args.ckpt, verbose=True)
# After loading, the model for inference is trainer.ema.ema_model or trainer.model
inference_model = trainer.ema.ema_model if trainer.ema is not None else trainer.model
#inference_model = trainer.model
inference_model.eval() # Ensure model is in eval mode


@torch.no_grad()
def heun_infill(inference_model,
                shape_btd,         # [B, T, D_data]
                target_x1_bdt,     # [B, D_total, T]  —— 与 fast_sample_infill 保持一致
                partial_mask_btd,  # [B, T, D_data] (bool) 或 None
                num_steps=200,
                k_scale=1.5,
                clamp_mid=(-1.1, 1.1),
                clamp_final=(-1.0, 1.0)):
    """
    Heun(2阶) + 数据一致性；内部状态严格用 [B, T, D_total]，以匹配你现有的 fast_sample_infill / FMTS.output().
    """
    device = target_x1_bdt.device
    B, T_seq, D_data = shape_btd
    _B, D_total, _T = target_x1_bdt.shape
    assert B == _B and T_seq == _T, f"shape mismatch: shape_btd={shape_btd}, target_x1_bdt={target_x1_bdt.shape}"

    # --- 工作布局：current_zt 用 [B, T, D_total]（与 fast_sample_infill 相同）---
    current_zt_btd = torch.randn((B, T_seq, D_total), device=device)  # z_t in [B,T,D*]
    # 保留 z0 的 [B, D_total, T] 版本用于 DC 直线路径
    z0_bdt = current_zt_btd.permute(0, 2, 1).contiguous()            # [B,D*,T]

    # --- 构造“已知通道”布尔掩码：始终用 [B, T, D_total] 做 where ---
    full_known_mask_btd = None
    is_c1 = (D_total > D_data)
    if partial_mask_btd is not None:
        full_known_mask_btd = torch.zeros((B, T_seq, D_total), dtype=torch.bool, device=device)
        # 数据通道按提供的 mask
        full_known_mask_btd[:, :, :D_data] = partial_mask_btd.to(torch.bool)
        # C1：条件通道恒“已知”
        if is_c1:
            full_known_mask_btd[:, :, D_data:] = True

    # 也与 fast_sample_infill 一致：给 Transformer 的 1D mask 是 [B,T]
    mask_1d_bt = partial_mask_btd.any(dim=-1) if partial_mask_btd is not None else None  # [B,T] or None

    time_scalar = float(getattr(inference_model, "time_scalar", 1000.0))

    for s in range(num_steps):
        if not torch.isfinite(current_zt_btd).all():
            raise RuntimeError(f"NaN detected in ODE state `current_zt_btd` at step {s} BEFORE model call.")
        t_curr = s / num_steps
        t_next = (s + 1) / num_steps
        t_eff_c = t_curr ** k_scale
        t_eff_n = t_next ** k_scale
        dt = t_eff_n - t_eff_c

        t_in_c = torch.full((B,), t_eff_c * time_scalar, device=device)
        t_in_n = torch.full((B,), t_eff_n * time_scalar, device=device)

        # ---- v(t)：FMTS.output 期望 [B,T,D*]，返回也按该布局（如返回 [B,D*,T] 就转回来）----
        with torch.cuda.amp.autocast(True):
            v_c = inference_model.output(current_zt_btd.clone(), t_in_c, mask=mask_1d_bt)
        if v_c.shape[1] == D_total and v_c.shape[2] == T_seq:
            v_c = v_c.permute(0, 2, 1).contiguous()  # 统一到 [B,T,D*]
        #v_c = torch.tanh(v_c)

        # 预测一步
        z_pred_btd = current_zt_btd + dt * v_c

        # ---- v(t+dt) ----
        with torch.cuda.amp.autocast(True):
            v_n = inference_model.output(z_pred_btd.clone(), t_in_n, mask=mask_1d_bt)
        if v_n.shape[1] == D_total and v_n.shape[2] == T_seq:
            v_n = v_n.permute(0, 2, 1).contiguous()
        #v_n = torch.tanh(v_n)

        # Heun 校正
        z_next_btd = current_zt_btd + 0.5 * dt * (v_c + v_n)

        # ---- 数据一致性（与 fast_sample_infill 完全对齐的维度处理）----
        if full_known_mask_btd is not None:
            # 直线路径在 [B,D*,T] 上计算
            ideal_path_bdt = (1.0 - t_eff_n) * z0_bdt + t_eff_n * target_x1_bdt  # [B,D*,T]
            ideal_path_btd = ideal_path_bdt.permute(0, 2, 1).contiguous()        # [B,T,D*]
            z_next_btd = torch.where(full_known_mask_btd, ideal_path_btd, z_next_btd)

        current_zt_btd = torch.clamp(z_next_btd, *clamp_mid)

    # 结束：回到 [B, D_total, T]，并裁掉条件通道（C1）
    final_bdt = torch.clamp(current_zt_btd, *clamp_final).permute(0, 2, 1).contiguous()  # [B,D*,T]
    if is_c1:
        final_bdt = final_bdt[:, :D_data, :]  # 只返回数据通道
    return final_bdt


def evaluate_random_missing_ratios(inference_model, trainer, dataset_instance, test_dl,
                                   ratios=(0.1, 0.3, 0.5, 0.7),
                                   num_steps=200, k_scale=1.5, seed=42):
    """
    在归一化空间上评测随机缺失率（0.1/0.3/0.5/0.7），返回各比率的 MSE/MAE/RMSE/PCC。
    注意：
      - 本函数不使用 dataloader 自带的 mask（忽略！），而是自己随机生成；
      - 结果在“归一化”尺度上统计；
      - 推理用 heun_infill（二阶 + 数据一致性）。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = next(inference_model.parameters()).device
    time_scalar = float(getattr(inference_model, "time_scalar", 1000.0))

    # 准备聚合器（按比率分别统计）
    stats = {}
    for r in ratios:
        stats[r] = dict(
            mse_sum=0.0, mae_sum=0.0, n=0,
            # 用总体公式统计 Pearson r
            sum_x=0.0, sum_y=0.0, sum_x2=0.0, sum_y2=0.0, sum_xy=0.0
        )

    is_c1_model = hasattr(inference_model, "feature_size")

    with torch.inference_mode():
        for batch_data in test_dl:
            # 统一拿 x_gt （已经是“归一化后”的张量）
            if torch.is_tensor(batch_data):
                x_gt = batch_data
            elif isinstance(batch_data, (list, tuple)):
                x_gt = batch_data[0]
            else:
                raise ValueError(f"Unexpected batch type: {type(batch_data)}")

            x_gt = x_gt.to(device)                 # [B,T,D_data]
            B, T, D_data = x_gt.shape
            D_total = getattr(inference_model, "feature_size", D_data)

            for ratio in ratios:
                # --------- 随机 mask（True=已观测，False=缺失）---------
                mask_rand = (torch.rand((B, T, D_data), device=device) > ratio)
                # 保证边界与少量点可用（避免全缺、全观测）
                mask_rand[:, 0, :] = True
                mask_rand[:, -1, :] = True
                # 如果某个变量全缺，强制打开一个点
                all_missing = ~mask_rand.view(B * D_data, T).any(dim=1)
                if all_missing.any():
                    idxs = all_missing.nonzero(as_tuple=True)[0]
                    # 给这些通道随机开一个点
                    rand_t = torch.randint(low=0, high=T, size=(idxs.numel(),), device=device)
                    mask_flat = mask_rand.view(B * D_data, T)
                    mask_flat[idxs, rand_t] = True
                    mask_rand = mask_flat.view(B, D_data, T).permute(0, 2, 1).contiguous()
                    mask_rand = mask_rand.permute(0, 2, 1)

                # --------- 构造 target_x1_bdt ----------
                if is_c1_model and D_total > D_data:
                    left_ctx, right_ctx = trainer.make_ctx_feature(x_gt, mask_rand, window=10)
                    
                    mask_chan = mask_rand.any(dim=-1, keepdim=True).float()
                    #left_ctx  = torch.zeros_like(mask_chan, dtype=x_gt.dtype, device=x_gt.device)
                    #right_ctx = torch.zeros_like(mask_chan, dtype=x_gt.dtype, device=x_gt.device)
                    data_part = x_gt * mask_rand.float()
                    target_btd = torch.cat([data_part, mask_chan, left_ctx, right_ctx], dim=-1)
                else:
                    target_btd = x_gt * mask_rand.float()


                target_bdt = target_btd.permute(0, 2, 1)   # [B,D_total,T] 或 [B,D_data,T]

                # --------- 推理（Heun + 数据一致性）----------
                pred_bdt = heun_infill(
                    inference_model=inference_model,
                    shape_btd=x_gt.shape,                 # [B,T,D_data]
                    target_x1_bdt=target_bdt,            # [B,D*,T]
                    partial_mask_btd=mask_rand,          # [B,T,D_data] (bool)
                    num_steps=num_steps,
                    k_scale=k_scale
                )
                # 只留数据通道
                if pred_bdt.shape[1] > D_data:
                    pred_bdt = pred_bdt[:, :D_data, :]

                pred_btd = pred_bdt                       # [B,D_data,T]
                pred_btd = pred_btd                      # already normalized
                pred_btd = pred_btd.permute(0, 2, 1)     # [B,T,D_data] for metric indexing
                # 归一化空间上计算 gap 区域
                gap = ~mask_rand                          # [B,T,D_data]
                if not gap.any():
                    continue

                diff = (pred_btd - x_gt)
                mse_sum = (diff[gap] ** 2).sum().item()
                mae_sum = diff[gap].abs().sum().item()
                n = gap.sum().item()

                # Pearson 统计量（一次性全局统计）
                x = pred_btd[gap].double()
                y = x_gt[gap].double()
                sum_x = x.sum().item()
                sum_y = y.sum().item()
                sum_x2 = (x * x).sum().item()
                sum_y2 = (y * y).sum().item()
                sum_xy = (x * y).sum().item()

                st = stats[ratio]
                st['mse_sum'] += mse_sum
                st['mae_sum'] += mae_sum
                st['n']       += n
                st['sum_x']   += sum_x
                st['sum_y']   += sum_y
                st['sum_x2']  += sum_x2
                st['sum_y2']  += sum_y2
                st['sum_xy']  += sum_xy

    # 汇总
    results = {}
    for r, st in stats.items():
        if st['n'] == 0:
            results[r] = dict(MSE=None, MAE=None, RMSE=None, PCC=None)
            continue
        mse = st['mse_sum'] / st['n']
        mae = st['mae_sum'] / st['n']
        rmse = math.sqrt(mse)

        n = float(st['n'])
        num = st['sum_xy'] - (st['sum_x'] * st['sum_y'] / n)
        den_x = st['sum_x2'] - (st['sum_x'] ** 2) / n
        den_y = st['sum_y2'] - (st['sum_y'] ** 2) / n
        denom = math.sqrt(max(den_x, 0.0) * max(den_y, 0.0))
        pcc = (num / denom) if denom > 0 else None

        results[r] = dict(MSE=mse, MAE=mae, RMSE=rmse, PCC=pcc)

    return results



# ---------------- inference ------------------------
all_samples_np, all_reals_np, all_masks_np = [], [], []

with torch.inference_mode():
    for idx, batch_data in enumerate(tqdm(test_dl, total=len(test_dl), desc="Infill (test)", dynamic_ncols=True)):
        # ---- 统一解包 (x_gt, x_in_cond, t_m) ----
        x_gt_batch = x_input_for_cond_batch = t_m_batch = None

        if torch.is_tensor(batch_data):
            # 只返回 x 的情况
            x_gt_batch = batch_data
        elif isinstance(batch_data, (list, tuple)):
            if len(batch_data) == 3:
                x_gt_batch, x_input_for_cond_batch, t_m_batch = batch_data
            elif len(batch_data) == 2:
                # 最常见： (x_gt, mask)
                x_gt_batch, t_m_batch = batch_data
            elif len(batch_data) == 1:
                (x_gt_batch,) = batch_data
            else:
                raise ValueError(f"Unexpected batch structure: len={len(batch_data)} types={[type(x) for x in batch_data]}")
        else:
            raise ValueError(f"Unexpected batch type: {type(batch_data)}")

        # ---- 送显卡 / 类型修正 ----
        x_batch = x_gt_batch.to(device)
        if t_m_batch is None:
            # 做“长缺口插值”必须要有 mask
            raise ValueError("Mask (t_m) not provided by test dataloader, but required for infill.")
        t_m_batch = t_m_batch.to(device)
        if t_m_batch.dtype != torch.bool:
            t_m_batch = t_m_batch.bool()

        # ---- 归一化后的 GT 即 x_normed_gt ----
        x_normed_gt = x_batch  # [B, T, D_data]

        # ---- 判断是否 C1 模式（数据通道+3个条件通道）----
        num_data_features = x_normed_gt.shape[2]
        is_c1 = hasattr(inference_model, "feature_size") and (inference_model.feature_size > num_data_features)

       # ---- 构造 target_x1_bdt（无插值版）----
        if is_c1:
            # 左右上下文直接用原序列 + mask 计算（make_ctx_feature 内部会用 mask 做加权）
            left_ctx_cond, right_ctx_cond = trainer.make_ctx_feature(x_normed_gt, t_m_batch, window=10)
            
            mask_channel_cond = t_m_batch.any(dim=-1, keepdim=True).float()
            #left_ctx_cond  = torch.zeros_like(mask_channel_cond, dtype=x_normed_gt.dtype, device=x_normed_gt.device)
            #right_ctx_cond = torch.zeros_like(mask_channel_cond, dtype=x_normed_gt.dtype, device=x_normed_gt.device)
            data_part = x_normed_gt * t_m_batch.float()  # 观测处放 GT，缺失处为 0
            target_btd_c1 = torch.cat([data_part, mask_channel_cond, left_ctx_cond, right_ctx_cond], dim=-1)
            target_bdt_for_infill = target_btd_c1.permute(0, 2, 1)
        else:
            target_btd_b0 = x_normed_gt * t_m_batch.float()
            target_bdt_for_infill = target_btd_b0.permute(0, 2, 1)


        # ---- 推理 ----
        '''
        sample_output_bdt = inference_model.fast_sample_infill(
            shape=x_normed_gt.shape,           # [B, T, D_data]
            target_x1_bdt=target_bdt_for_infill,
            partial_mask_btd=t_m_batch         # [B, T, D_data] (bool)
        )
        '''
        sample_output_bdt = heun_infill(
            inference_model=inference_model,
            shape_btd=x_normed_gt.shape,
            target_x1_bdt=target_bdt_for_infill,
            partial_mask_btd=t_m_batch,
            num_steps=args.num_steps,
            k_scale=args.k_scale,
        )
        # 若是 C1，只取前 D_data 通道
        if target_bdt_for_infill.shape[1] > num_data_features:
            sample_output_bdt = sample_output_bdt[:, :num_data_features, :]

        sample_btd_normed = sample_output_bdt.permute(0, 2, 1)  # [B, T, D_data]

        # ---- 反归一化 ----
        sample_to_unnorm_cpu = sample_btd_normed.detach().cpu().numpy()
        reals_to_unnorm_cpu  = x_normed_gt.detach().cpu().numpy()

        if hasattr(dataset_instance, 'unnormalize'):
            sample_final_np = dataset_instance.unnormalize(sample_to_unnorm_cpu)
            reals_final_np  = dataset_instance.unnormalize(reals_to_unnorm_cpu)
        else:
            sample_final_np = sample_to_unnorm_cpu
            reals_final_np  = reals_to_unnorm_cpu
        
        sample_final_np = sample_to_unnorm_cpu
        reals_final_np  = reals_to_unnorm_cpu

        print("caculating...")
        all_samples_np.append(sample_final_np)
        all_reals_np.append(reals_final_np)
        all_masks_np.append(t_m_batch.detach().cpu().numpy())  # bool


samples_arr = np.vstack(all_samples_np)
reals_arr = np.vstack(all_reals_np)
masks_arr = np.vstack(all_masks_np).astype(bool) # Ensure boolean type for masking

gap_region = ~masks_arr       # Bool mask – True where gap
# print("[EVAL] samples_arr (all):", samples_arr.min(), samples_arr.max())
# print("[EVAL] gap samples_arr:", samples_arr[gap_region].min() if gap_region.any() else "N/A", samples_arr[gap_region].max() if gap_region.any() else "N/A")
# print("[EVAL] reals_arr (all):", reals_arr.min(), reals_arr.max())

# print("GAP REALS min/max/mean:", reals_arr[gap_region].min() if gap_region.any() else "N/A", reals_arr[gap_region].max() if gap_region.any() else "N/A", reals_arr[gap_region].mean() if gap_region.any() else "N/A")

if not gap_region.any():
    print("[ERROR] No gap regions found in the test data according to masks. Metrics cannot be calculated for gaps.")
    mse, mae, rmse, pcc = float('nan'), float('nan'), float('nan'), float('nan')
else:
    mse  = np.mean(((samples_arr - reals_arr) ** 2)[gap_region])
    mae  = np.mean(np.abs(samples_arr - reals_arr)[gap_region])
    rmse = np.sqrt(mse)
    gap_preds = samples_arr[gap_region]
    gap_truth = reals_arr[gap_region]

    # print("Num gap preds < 0:", (gap_preds < 0).sum(), "总gap数:", gap_preds.size)
    # print("Num gap preds < -5:", (gap_preds < -5).sum())
    # print("Num gap preds < -10:", (gap_preds < -10).sum())
    # print("Num gap preds < -20:", (gap_preds < -20).sum())
    # print("gap preds mean/std:", gap_preds.mean(), gap_preds.std())

    # === CLIP后分布与metrics ===
    clip_min_val = reals_arr[gap_region].min()
    clip_max_val = reals_arr[gap_region].max()
    gap_preds_clip = np.clip(gap_preds, clip_min_val, clip_max_val)

    # print("[EVAL][CLIP] gap_preds_clip min/max:", gap_preds_clip.min(), gap_preds_clip.max())
    # print("[EVAL][CLIP] gap区 MSE:", np.mean((gap_preds_clip - gap_truth) ** 2))
    # print("[EVAL][CLIP] gap区 MAE:", np.mean(np.abs(gap_preds_clip - gap_truth)))
    # print("[EVAL][CLIP] clip区间: [{:.2f}, {:.2f}]".format(clip_min_val, clip_max_val))
    # print("[EVAL][CLIP] 被clip的个数:", np.sum((gap_preds < clip_min_val) | (gap_preds > clip_max_val)), "总gap数:", gap_preds.size)

    if gap_preds.size == gap_truth.size and gap_preds.size > 1: # Pearson r needs at least 2 points
        pcc = pearsonr(gap_preds.flatten(), gap_truth.flatten())[0]
    else:
        pcc = float('nan')

print(f'✅  Infill done:  MSE={mse:.6f}  MAE={mae:.6f} PCC={pcc:.6f}')

# -------------- save arrays & metrics -------------
out_dir = Path(f'OUTPUT/{args.name}')
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / 'samples.npy', samples_arr)
np.save(out_dir / 'reals.npy',   reals_arr)
np.save(out_dir / 'masks.npy',   masks_arr) # Save boolean masks
with open(out_dir / 'metrics.json', 'w') as f:
    json.dump({
        'MSE': float(mse) if not np.isnan(mse) else None,
        'MAE': float(mae) if not np.isnan(mae) else None,
        'RMSE': float(rmse) if not np.isnan(rmse) else None,
        'PCC': float(pcc) if not np.isnan(pcc) else None
    }, f, indent=2)

# -------------------- plotting ---------------------
if gap_region.any(): # Only plot if there are gaps
    plt.rcParams["font.size"] = 12
    # Assuming all samples in the batch have the same gap structure for plotting example
    # This might not be true if gaps are generated randomly per sample in test_dl
    # For consistent plotting, CustomDataset should generate a fixed gap for test period.
    # Your CustomDataset's test mode for long_gap creates a fixed center gap.
    
    # Find first actual gap start/end from the first sample's mask for plotting lines
    first_sample_mask_1d = masks_arr[0, :, 0] # Mask for var 0 of sample 0
    if (~first_sample_mask_1d).any(): # If there is a gap for var0 of sample0
        gap_indices_for_plot = np.where(~first_sample_mask_1d)[0]
        plot_gap_start = gap_indices_for_plot[0]
        plot_gap_end = gap_indices_for_plot[-1]
    else: # No gap in the first sample, can't draw lines properly for this example
        plot_gap_start, plot_gap_end = shape[0] // 2 - 10, shape[0] // 2 + 10 # Fallback if no gap
        print("[WARNING PLOT] No gap found in the first sample's first variable for plotting lines.")


    for plot_idx in range(min(args.plot_num, samples_arr.shape[0])):
        for d_var in range(shape[1]): # Iterate over D_data variables
            plt.figure(figsize=(15,3))
            # History + GT 全体
            plt.plot(reals_arr[plot_idx, :, d_var], c='c', label='History & GT')
            
            # Check if this specific variable in this sample has a gap
            current_var_has_gap = (~masks_arr[plot_idx, :, d_var]).any()
            if current_var_has_gap:
                current_gap_indices = np.where(~masks_arr[plot_idx, :, d_var])[0]
                current_plot_gap_start = current_gap_indices[0]
                current_plot_gap_end = current_gap_indices[-1]

                # 缺口真实
                plt.plot(range(current_plot_gap_start, current_plot_gap_end + 1),
                         reals_arr[plot_idx, current_plot_gap_start : current_plot_gap_end + 1, d_var], c='g', label='GT Gap', linewidth=1.5)
                # 模型插值
                plt.plot(range(current_plot_gap_start, current_plot_gap_end + 1),
                         samples_arr[plot_idx, current_plot_gap_start : current_plot_gap_end + 1, d_var], c='r', label='Infill', linewidth=1.5, linestyle='--')
                
                plt.axvline(current_plot_gap_start,  c='k', ls='--');
                plt.axvline(current_plot_gap_end, c='k', ls='--') # End of gap region
            else:
                print(f"[INFO PLOT] Sample {plot_idx} Var {d_var} has no gap, skipping gap-specific plotting.")


            plt.title(f'seq#{plot_idx} var#{d_var}  Gap MSE={mse:.4f}') # Show gap MSE if calculated
            plt.xlabel('Time'); plt.ylabel('Value'); plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f'seq{plot_idx}_var{d_var}.png', dpi=150)
            plt.close()
else:
    print("No gap regions found. Skipping plotting.")


# ====== 随机缺失评测（归一化空间）======
ratios = tuple(float(x) for x in args.rand_ratios.split(','))
rand_results = evaluate_random_missing_ratios(
    inference_model=inference_model,
    trainer=trainer,
    dataset_instance=dataset_instance,
    test_dl=test_dl,
    ratios=ratios,
    num_steps=args.num_steps,
    k_scale=args.k_scale,
    seed=args.seed
)

print("🔎 Random-missing (normalized) results:")
for r in ratios:
    rdict = rand_results[r]
    print(f"  ratio={r:.1f}  MSE={rdict['MSE']:.6f}  MAE={rdict['MAE']:.6f}  RMSE={rdict['RMSE']:.6f}  PCC={rdict['PCC']:.6f}")

# 追加写到 metrics.json（或另存一个文件）
metrics_path = out_dir / 'metrics_random_norm.json'
with open(metrics_path, 'w') as f:
    json.dump({str(r): rand_results[r] for r in ratios}, f, indent=2)
print(f'📝  Random-missing normalized metrics saved to {metrics_path}')



print(f'📊  All results saved to {out_dir}')