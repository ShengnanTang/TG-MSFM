import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=4000, 
        proportion=0.8, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=8,          # ★ 几何段均值，8~16 可调
        long_gap=False,              # 是否启用连续缺口（一般只在 test 用）
        gap_len=1000,                # 连续缺口长度
        apply_prob=1.0,              # 生成缺口的样本比例（只对 long_gap 生效）
        return_mask=True,            # 训练/测试是否返回 mask
        # === 新增：训练期对齐式随机缺失配置 ===
        train_mask_random_ratios=None,       # 例： [0.1, 0.3, 0.5, 0.7]
        train_mask_prob_concurrent=0.0,      # 小概率并发 blackout（0.0~0.2）
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            # 训练期不走旧的 missing_ratio/predict_length 接口（避免分布失配）
            assert not (predict_length is not None or missing_ratio is not None), \
                'For training, please use random-missing config instead of predict_length/missing_ratio.'

        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.train_mask_random_ratios = train_mask_random_ratios or [0.3]   # 默认 0.3
        self.train_mask_prob_concurrent = float(train_mask_prob_concurrent)

        self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one
        self.gap_len_choices = [100, 250, 500, 750, 1000]  # 训练若要随机长缺口可用，但默认不用
        # print(f"[DEBUG] Raw data shape = {self.rawdata.shape}")

        self.data = self.__normalize(self.rawdata)
        train_data, test_data = self.__getsamples(self.data, proportion, seed)
        if self.period == 'train':
            self.samples = train_data
        else:
            # ★ 测试期保留全部样本（之前只取 [:1] 会导致评测极不稳定）
            self.samples = test_data[:1]

        # ---------- 仅在需要 old masking 方案时进入（保兼容） ----------
        if period == 'test' and (missing_ratio is not None or predict_length is not None):
            if missing_ratio is not None:          # 稀疏缺口方案（老接口）
                self.masking = self.mask_data(seed)
            elif predict_length is not None:       # 末尾置零预测方案（老接口）
                masks = np.ones(self.samples.shape, dtype=bool)
                masks[:, -predict_length:, :] = False
                self.masking = masks
        # ------------------------------------------------------
        self.sample_num = self.samples.shape[0]
        self.long_gap   = bool(long_gap)
        self.gap_len    = int(gap_len)
        self.apply_prob = float(apply_prob)
        self.return_mask= bool(return_mask)
        # print(f"[DEBUG] period={self.period} | proportion={proportion} | sample_total={self.sample_num_total} | loaded={self.samples.shape[0]}")

    # --------------------- utils ---------------------
    def __getsamples(self, data, proportion, seed):
        x = np.zeros((self.sample_num_total, self.window, self.var_num), dtype=np.float32)
        for i in range(self.sample_num_total):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]

        train_data, test_data = self.divide(x, proportion, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)
        # print(f"[DEBUG] __getsamples() train_data.shape={train_data.shape}, test_data.shape={test_data.shape}")
        return train_data, test_data

    def _make_long_gap(self, length:int):
        g = np.random.choice(self.gap_len_choices)
        g = min(g, length - 2)
        s = np.random.randint(0, length - g - 1)
        m = np.ones(length, bool)
        m[s:s+g] = False
        return m

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data.astype(np.float32)

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num).astype(np.float32)

    def unnormalize(self, sq):
        if isinstance(sq, torch.Tensor):
            sq = sq.detach().cpu().numpy()
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        st0 = np.random.get_state()
        np.random.seed(seed)

        split_num = int(np.ceil(size * ratio))
        id_rdm = np.random.permutation(size)

        part1_id = id_rdm[:split_num]
        part2_id = id_rdm[split_num:]

        part1 = data[part1_id, :]
        part2 = data[part2_id, :]

        np.random.set_state(st0)

        if ratio >= 0.5:
            return part1, part2  # train, test
        else:
            return part2, part1  # train, test

    @staticmethod
    def read_data(filepath, name=''):
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler

    # 旧接口（保兼容）
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples, dtype=bool)
        st0 = np.random.get_state()
        np.random.seed(seed)
        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style, self.distribution)
            masks[idx, :, :] = mask
        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)
        np.random.set_state(st0)
        return masks

    # —— 新增：确保每个变量至少有 1 个观测 —— #
    @staticmethod
    def _ensure_not_all_missing(mask_2d: np.ndarray):
        # mask_2d: (T, D) bool
        T, D = mask_2d.shape
        for d in range(D):
            if not mask_2d[:, d].any():
                t = np.random.randint(0, T)
                mask_2d[t, d] = True
        return mask_2d

    def __getitem__(self, ind):
        """
        返回三元组 (gt, input_x, mask) 以支持 C1 训练 / 推理；
        若 self.return_mask=False，则仅返回 gt（与旧逻辑兼容）。
        形状：
          gt/input_x: torch.FloatTensor, [T, D]
          mask:       torch.BoolTensor,  [T, D]  (True=观测，False=缺失)
        """
        x = self.samples[ind]                               # np.ndarray, (T, D)
        T, D = x.shape
        mask = np.ones_like(x, dtype=bool)                  # 默认全观测

        if self.period == 'train':
            # === 对齐式随机缺失（推荐用于所有常规实验） ===
            # 1) 随机抽取缺失比例
            ratio = float(np.random.choice(self.train_mask_random_ratios))
            # 2) 小概率并发 blackout；否则按 feature 独立（separate）
            mode = 'concurrent' if (np.random.rand() < self.train_mask_prob_concurrent) else self.style
            mask = noise_mask(x, masking_ratio=ratio, lm=self.mean_mask_length,
                              mode=mode, distribution=self.distribution)
            # 3) 确保每个变量至少保留 1 个观测，避免全缺通道
            mask = self._ensure_not_all_missing(mask)

        else:  # period == 'test'
            if self.long_gap:
                # 固定中心长缺口（评测用）
                center = self.window // 2
                half = min(self.gap_len // 2, self.window // 2 - 1)
                start = max(0, center - half)
                end   = min(self.window, center + half)
                mask[:] = True
                mask[start:end, :] = False
            elif hasattr(self, 'masking'):
                # 兼容老接口：若 __init__ 构造了 self.masking
                mask = self.masking[ind]
            else:
                # 否则保持全观测（或你也可以在这里做轻度随机缺失评测）
                mask = np.ones_like(x, dtype=bool)

        x_tensor    = torch.from_numpy(x).float()           # [T, D]
        mask_tensor = torch.from_numpy(mask).bool()         # [T, D]
        if self.return_mask:
            # C1：input_x 用零填缺的“可见输入”
            input_x = x_tensor * mask_tensor.float()
            return x_tensor, input_x, mask_tensor
        else:
            return x_tensor

    def __len__(self):
        return self.sample_num


class fMRIDataset(CustomDataset):
    def __init__(self, proportion=1., **kwargs):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
