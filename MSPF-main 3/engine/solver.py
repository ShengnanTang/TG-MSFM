import os
import sys
import time
import math
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info

from torch.cuda.amp import autocast, GradScaler


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data

def _nan_grad_hook(name):
    def _hook(module, grad_input, grad_output):
        def has_bad(g):
            # g 可能是 None / Tensor / 稀疏 Tensor
            if g is None:
                return False
            if not torch.is_tensor(g):
                return False
            t = g
            if t.is_sparse:
                t = t.coalesce().values()
            t = t.detach()
            # 返回 Python bool
            return (~torch.isfinite(t)).any().item()

        def max_abs(g):
            if g is None or (not torch.is_tensor(g)):
                return 0.0
            t = g.coalesce().values() if g.is_sparse else g
            return t.detach().abs().max().item()

        bad_in  = any(has_bad(g) for g in grad_input)
        bad_out = any(has_bad(g) for g in grad_output)

        if bad_in or bad_out:
            # 打点一些有用信息
            mi = max((max_abs(g) for g in grad_input), default=0.0)
            mo = max((max_abs(g) for g in grad_output), default=0.0)
            print(f"[NaN-Grad] at {name} (bad_in={bad_in}, bad_out={bad_out}, "
                  f"max|gin|={mi:.3e}, max|gout|={mo:.3e})")
            # 可选：再看看该模块参数的范数
            try:
                pn = sum((p.detach().abs().max().item() for p in module.parameters(recurse=False)), 0.0)
                print(f"[NaN-Grad] param max(abs) sum={pn:.3e}")
            except Exception:
                pass
            raise RuntimeError(f"NaN gradient hit at {name}")
    return _hook


def register_nan_debug_hooks(root_model):
    """
    root_model: 你的顶层模型（FM_TS 实例或 DataParallel 包裹）
    只在最可疑的几处挂 backward hook，定位第一处 NaN 梯度来源
    """
    # 如果用了 DataParallel，要拿 .module
    fm = root_model.module if isinstance(root_model, torch.nn.DataParallel) else root_model

    # 你的 FM_TS 里真正的 Transformer 在 fm.model
    core = getattr(fm, "model", fm)
    dec  = core.decoder

    handles = []

    # 只钩几个最敏感的位置：trend 的 conv1 + gelu、自注意/交叉注意
    for i, block in enumerate(dec.blocks):
        # TrendBlock 的 Sequential: [0]=Conv1d, [1]=GELU, [2]=Transpose, [3]=Conv1d
        try:
            handles.append(block.trend.trend[0].register_full_backward_hook(
                _nan_grad_hook(f"decoder.blocks[{i}].trend.conv1")))
        except Exception: pass
        try:
            handles.append(block.trend.trend[1].register_full_backward_hook(
                _nan_grad_hook(f"decoder.blocks[{i}].trend.gelu")))
        except Exception: pass

        # 注意力两条支路
        try:
            handles.append(block.attn1.register_full_backward_hook(
                _nan_grad_hook(f"decoder.blocks[{i}].attn1")))
        except Exception: pass
        try:
            handles.append(block.attn2.register_full_backward_hook(
                _nan_grad_hook(f"decoder.blocks[{i}].attn2")))
        except Exception: pass

        # MLP 入口（常见放大点）
        try:
            handles.append(block.mlp[0].register_full_backward_hook(
                _nan_grad_hook(f"decoder.blocks[{i}].mlp.fc1")))
        except Exception: pass

        # 只钩前 2~3 个 block 往往就够定位；想全钩也可
        if i >= 2:
            break

    return handles



class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = next(self.model.parameters()).device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.dataloader = dataloader['dataloader']
        self.step = 0
        self.milestone = 0
        self.args, self.config = args, config
        self.logger = logger

        if isinstance(model, torch.nn.DataParallel):
            seq_len = getattr(model.module, "seq_length", None)
        else:
            seq_len = getattr(model, "seq_length", None)
        suffix = f"_{seq_len}" if seq_len is not None else ""
        base = config['solver'].get('results_folder', os.environ.get('results_folder', './Checkpoints'))
        self.results_folder = Path(base + suffix)
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        # 允许 TF32（Ampere+ 可显著加速，数值对 FM 问题足够稳）
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        # AdamW（支持 fused 就开启）
        use_fused = hasattr(AdamW, "fused") and torch.cuda.is_available()
        self.opt = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=start_lr, betas=(0.9, 0.96), weight_decay=1e-4,
            fused=use_fused
        )

        # EMA 原样
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        # AMP
        self.use_amp = torch.cuda.is_available()
        self.use_amp = False
        self.scaler  = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # “虚拟 epoch” 步数：用 dataloader 长度和梯度累积一起折算
        num_batches = max(1, len(self.dataloader))
        self.steps_per_epoch = max(1, math.ceil(num_batches / self.gradient_accumulate_every))

        # 统一用 StepLR，每“2 个虚拟 epoch”衰一次（避免每步衰减）
        #self.sch = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.steps_per_epoch * 2, gamma=0.9)
        self.sch = None


        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

        # __init__ 里其他东西都配好后……
        self._nan_hook_handles = []
        if os.environ.get("DEBUG_NAN", "1") == "1":  # 用环境变量控制开关
            try:
                self._nan_hook_handles = register_nan_debug_hooks(self.model)
                print(f"[DEBUG] NaN hooks registered: {len(self._nan_hook_handles)}")
            except Exception as e:
                print(f"[WARN] failed to register NaN hooks: {e}")


    
    def make_ctx_feature(self, seq, mask, window=10):
        """
        seq:  [B, T, D]
        mask: [B, T, D]  (bool)  True=observed, False=missing
        return: left_ctx, right_ctx  [B, T, 1]
        """
        B, T, D = seq.shape
        device = seq.device
        kernel = torch.ones(1, 1, window, device=device)

        # --- Left context ---
        seq_left = seq * mask.float()
        mask_left = mask.float()
        # 变成 [B*D, 1, T]
        seq_left = seq_left.permute(0,2,1).reshape(B*D, 1, T)
        mask_left = mask_left.permute(0,2,1).reshape(B*D, 1, T)
        # pad左侧
        seq_cumsum = F.conv1d(F.pad(seq_left, (window, 0)), kernel)[:, :, :T]
        mask_cumsum = F.conv1d(F.pad(mask_left, (window, 0)), kernel)[:, :, :T] + 1e-8
        left_ctx_raw = seq_cumsum / mask_cumsum
        # 回 [B, T, D]
        left_ctx_raw = left_ctx_raw.reshape(B, D, T).permute(0,2,1)
        # [B, T, 1]  (可用 mean/first/last，这里取 mean)
        left_ctx = left_ctx_raw.mean(dim=2, keepdim=True)

        # --- Right context ---
        seq_right = seq * mask.float()
        mask_right = mask.float()
        seq_right = seq_right.permute(0,2,1).reshape(B*D, 1, T)
        mask_right = mask_right.permute(0,2,1).reshape(B*D, 1, T)
        # pad右侧
        seq_cumsum_r = F.conv1d(F.pad(seq_right, (0, window)), kernel)[:, :, :T]
        mask_cumsum_r = F.conv1d(F.pad(mask_right, (0, window)), kernel)[:, :, :T] + 1e-8
        right_ctx_raw = seq_cumsum_r / mask_cumsum_r
        right_ctx_raw = right_ctx_raw.reshape(B, D, T).permute(0,2,1)
        right_ctx = right_ctx_raw.mean(dim=2, keepdim=True)

        return left_ctx, right_ctx

    

    def interpolate_with_noise(self, x, mask, noise_level=0.05):
        """
        用结构性插值填补缺口，并添加微小高斯噪声。
        x: [B, T, D] 实际数据
        mask: [B, T, D] bool mask, True=observed
        """
        x_filled = x.clone()

        for b in range(x.shape[0]):
            for d in range(x.shape[2]):
                observed = mask[b, :, d].bool()
                if observed.sum() < 2:
                    x_filled[b, :, d] = 0.0
                    continue

                obs_idx = observed.nonzero(as_tuple=True)[0]
                obs_vals = x[b, obs_idx, d]

                # 使用 numpy 插值
                full_idx = torch.arange(x.shape[1], device=x.device)
                interp_vals = np.interp(full_idx.cpu().numpy(),
                                        obs_idx.cpu().numpy(),
                                        obs_vals.cpu().numpy())
                x_filled[b, :, d] = torch.from_numpy(interp_vals).to(x.device)

        # 添加轻微高斯噪声（仅对缺失部分）
        noise = torch.randn_like(x_filled) * noise_level
        x_filled = x_filled + (~mask).float() * noise
        #print("[DEBUG] interpolate_with_noise out:", x_filled.min().item(), x_filled.max().item())

        return x_filled



    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        state_dict = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) \
             else self.model.state_dict()
        data = {'step': self.step, 'model': state_dict, 'ema': self.ema.state_dict(), 'opt': self.opt.state_dict()}

        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))
    
    def load(self, milestone, verbose=False):
        checkpoint_path = str(self.results_folder / f'checkpoint-{milestone}.pt')
        if self.logger is not None and verbose:
            self.logger.log_info(f'Resume from {checkpoint_path}')
        else:
            print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

        device = self.device
        data = torch.load(checkpoint_path, map_location=device)

        # ------- 加载主模型 -------
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(data['model'])
        else:
            self.model.load_state_dict(data['model'])

        self.step = data.get('step', 0)
        if 'opt' in data:
            try:
                self.opt.load_state_dict(data['opt'])
            except Exception as e:
                print(f"[WARN] Failed to load optimizer state: {e}")

        # ------- 加载 EMA --------
        if 'ema' in data:
            try:
                # 临时包装 DataParallel（即使当前不使用多卡）
                tmp_model = torch.nn.DataParallel(self.model)
                tmp_ema = EMA(tmp_model,
                            beta=self.config['solver']['ema']['decay'],
                            update_every=self.config['solver']['ema']['update_interval']).to(device)

                tmp_ema.load_state_dict(data['ema'])
                #print("[INFO] EMA state loaded with DataParallel wrapper.")

                # 从 DataParallel 中解包回来
                tmp_ema.ema_model = tmp_ema.ema_model.module
                self.ema = tmp_ema

            except Exception as e:
                #print(f"[WARN] Failed to load EMA state_dict even after fallback: {e}")
                #print("[INFO] EMA will be disabled.")
                self.ema = None
        else:
            #print("[INFO] No EMA state found in checkpoint.")
            self.ema = None

        self.milestone = milestone



    def train(self):
        torch.autograd.set_detect_anomaly(True)
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info(f'{self.args.name}: start training...', check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps, mininterval=2.0, dynamic_ncols=True) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.dl)

                    if isinstance(batch, (list, tuple)):
                        if len(batch) == 3:
                            # 兼容旧管线：(gt, input_x, mask)
                            gt, input_x, mask = [b.to(device, non_blocking=True) for b in batch]  # [B,T,D]
                            left_ctx, right_ctx = self.make_ctx_feature(input_x, mask, window=10)  # [B,T,1]
                            mask_channel = mask.any(dim=-1, keepdim=True).float()                  # [B,T,1]
                            # 目标：gt + 条件通道（C1）

                            #left_ctx  = torch.zeros_like(mask_channel, dtype=input_x.dtype, device=input_x.device)
                            #right_ctx = torch.zeros_like(mask_channel, dtype=input_x.dtype, device=input_x.device)

                            z1_target = torch.cat([gt, mask_channel, left_ctx, right_ctx], dim=-1).permute(0, 2, 1)  # [B,D+3,T]
                            loss = self.model(z1_target, mask)  # [B,T] True=observed
                        elif len(batch) == 2:
                            # 推荐新管线：(x, mask)
                            x, mask = [b.to(device, non_blocking=True) for b in batch]             # [B,T,D], [B,T,D]bool
                            left_ctx, right_ctx = self.make_ctx_feature(x, mask, window=10)        # [B,T,1]
                            mask_channel = mask.any(dim=-1, keepdim=True).float()                  # [B,T,1]
                            # 目标：x + 条件通道（C1）

                            #left_ctx  = torch.zeros_like(mask_channel, dtype=x.dtype, device=x.device)
                            #right_ctx = torch.zeros_like(mask_channel, dtype=x.dtype, device=x.device)

                            z1_target = torch.cat([x, mask_channel, left_ctx, right_ctx], dim=-1).permute(0, 2, 1)  # [B,D+3,T]
                            loss = self.model(z1_target, mask)
                        else:
                            raise ValueError(f"Unexpected batch length {len(batch)}")
                    else:
                        # B0 退化路径（仅当数据集真的只给了 x）
                        seq = batch.to(device, non_blocking=True)                                    # [B,T,D]
                        loss = self.model(seq.permute(0, 2, 1), mask=None)

                    loss = loss / self.gradient_accumulate_every


                    # AMP 梯度缩放
                    self.scaler.scale(loss.mean()).backward()
                    total_loss += loss.mean().item()

                # —— 梯度累积结束，做一次真正的优化步骤 ——
                # 1) 反缩放 + 裁剪（可选）
                self.scaler.unscale_(self.opt)
                # 统计梯度异常 & 最大梯度来源
                bad, max_g, max_name = False, 0.0, ""
                for n, p in self.model.named_parameters():
                    if p.grad is None: 
                        continue
                    if not torch.isfinite(p.grad).all():
                        print(f"[GRAD NaN] {n} grad has non-finite values")
                        bad = True
                        break
                    g = p.grad.data.detach().abs().max().item()
                    if g > max_g:
                        max_g, max_name = g, n
                if bad:
                    raise RuntimeError("Non-finite gradient detected — see above")

                #print(f"[GRAD] max |grad|={max_g:.3e} @ {max_name}")  # 每个 step 或每N步打印一次
                clip_grad_norm_(self.model.parameters(), 1.0)

                # 2) 优化器 step + AMP scaler 更新
                self.scaler.step(self.opt)
                self.scaler.update()

                # 3) 清梯度
                self.opt.zero_grad(set_to_none=True)

                # 4) EMA
                self.ema.update()

                # 5) 计步 &（可选）调度器
                self.step += 1
                step += 1
                if self.sch:
                    self.sch.step()

                # 6) 保存 & 记录
                if self.step % self.save_cycle == 0:
                    self.milestone += 1
                    self.save(self.milestone)

                if self.logger is not None and self.step % self.log_frequency == 0:
                    lr = self.opt.param_groups[0]['lr']
                    self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)
                    print(f"[DBG] step={self.step} lr={lr:.2e} loss={total_loss:.4f}")


                if self.step % 10 == 0:
                    pbar.set_description(f'loss: {total_loss:.6f}')
                pbar.update(1)


        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))



    def sample(self, num, size_every, shape=None, model_kwargs=None, cond_fn=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples


    # ---------------------------------------------------------
    #  Trainer.restore
    # ---------------------------------------------------------
    def restore(self, raw_dataloader, shape=None,
                coef=1e-1, stepsize=1e-1, sampling_steps=50):

        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')

        # shape = [T , D]  —— 方便预先申请 numpy 容器
        samples = np.empty([0, shape[0], shape[1]])
        reals   = np.empty([0, shape[0], shape[1]])
        masks   = np.empty([0, shape[0], shape[1]])

        inference_model = self.ema.ema_model if self.ema is not None else self.model
        inference_model.eval()

        for idx, batch in enumerate(raw_dataloader):
            # ============================================================
            # ❶ 统一拆包：支持
            #    (gt , input_x , mask)  ← 与训练保持一致           推荐
            #    (input_x , mask)       ← 若你以后改成 2 元组亦可
            # ------------------------------------------------------------
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:                      # 三元组
                    gt, input_x, t_m = batch             # [B,T,D] / bool
                    x     = input_x.to(self.device)      # 模型输入（带 gap）
                    x_real= gt.to(self.device)           # 真值 (评估用)
                elif len(batch) == 2:                    # 二元组
                    x, t_m = batch
                    x      = x.to(self.device)
                    x_real = x                           # 无 gt 时退化
                else:
                    raise ValueError(f"Unexpected batch length {len(batch)}")
            else:
                raise TypeError("Batch must be tuple / list")

            t_m = t_m.to(self.device)                    # [B,T,D] bool

            if torch.all(t_m):
                #print(f"[INFO] Batch {idx} 无缺口，跳过。")
                continue
            # ============================================================

            # ---------- C1 / B0 检测 ----------
            num_feat = x.shape[2]
            is_c1 = hasattr(inference_model, "feature_size") and \
                    inference_model.feature_size > num_feat

            # ---------- Step-1 归一化 ----------
            # 你的 dataset 已经做 normalize，故直接使用
            x_normed = x

            # ---------- Step-2 构造 target ----------
            if is_c1:
                left_ctx, right_ctx = self.make_ctx_feature(x_normed, t_m, window=10)
                mask_chan = t_m.any(dim=-1, keepdim=True).float()
                # 关掉 left/right：用全零占位
                #left_ctx  = torch.zeros_like(mask_chan, dtype=x_normed.dtype, device=x_normed.device)
                #right_ctx = torch.zeros_like(mask_chan, dtype=x_normed.dtype, device=x_normed.device)
                target_btd = torch.cat([x_normed, mask_chan, left_ctx, right_ctx], dim=-1)  # [B,T,D+3]
                                        # [B,T,D+3]
            else:
                target_btd = x_normed * t_m.float()      # [B,T,D]

            # ---------- Step-3 fast_sample_infill ----------
            target_bdt = target_btd.permute(0, 2, 1)     # [B,D*,T]

            with torch.no_grad():
                sample_bdt = inference_model.fast_sample_infill(
                    shape=x.shape,
                    target_x1_bdt=target_bdt,
                    partial_mask_btd=t_m
                )
                sample_btd = sample_bdt.permute(0, 2, 1) # → [B,T,D]

            # ---------- Step-4 反归一化 ----------
            sample_np = sample_btd.detach().cpu().numpy()
            real_np   = x_real.detach().cpu().numpy()

            if hasattr(raw_dataloader.dataset, 'unnormalize'):
                sample_np = raw_dataloader.dataset.unnormalize(sample_np)
                real_np   = raw_dataloader.dataset.unnormalize(real_np)

            # ---------- 收集 ----------
            samples = np.row_stack([samples, sample_np])
            reals   = np.row_stack([reals,   real_np])
            masks   = np.row_stack([masks,   t_m.detach().cpu().numpy()])

            print(f"[{idx+1}/{len(raw_dataloader)}] batch done. "
                  f"sample min/max: {sample_np.min():.3f}/{sample_np.max():.3f}")

        if self.logger is not None:
            self.logger.log_info(f'Imputation done, time: {time.time() - tic:.2f}')

        return samples, reals, masks
