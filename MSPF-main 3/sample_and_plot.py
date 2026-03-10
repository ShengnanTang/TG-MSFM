import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from engine.solver import Trainer
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one

# === 设置 ===
CONFIG_PATH = "Config/debug.yaml"
CHECKPOINT_ID = 1
RESULTS_DIR = Path("results_debug")
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ.setdefault("hucfg_Kscale", "0.5")
os.environ.setdefault("hucfg_num_steps", "10")
os.environ.setdefault("hucfg_t_sampling", "logitnorm")

# === 加载配置 ===
args = type("Args", (), {})()
args.config_path = CONFIG_PATH
args.name = "debug_run"
args.save_dir = RESULTS_DIR
args.gpu = None

config = load_yaml_config(CONFIG_PATH)

# === 构建模型 ===
model = instantiate_from_config(config['model']).to(DEVICE)

# === 构建 dataloader（测试集） ===
test_dataset_cfg = config["dataloader"]["test_dataset"]
test_dataset_cfg["params"]["output_dir"] = str(RESULTS_DIR)
test_dataset = instantiate_from_config(test_dataset_cfg)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=config["dataloader"]["sample_size"],
    shuffle=False,
    num_workers=0
)

# === 创建 trainer 并加载模型 ===
trainer = Trainer(config=config, args=args, model=model, dataloader={"dataloader": test_dataloader})
trainer.load(CHECKPOINT_ID)

# === 执行插值 ===
seq_len, feat_dim = test_dataset.window, test_dataset.var_num
samples, reals, masks = trainer.restore(test_dataloader, shape=[seq_len, feat_dim])
samples = test_dataset.unnormalize(samples)
reals = test_dataset.unnormalize(reals)

# === 计算 MSE 并保存 ===
masks = masks.astype(bool)  # 确保为布尔类型
mse = mean_squared_error(samples[~masks], reals[~masks])
with open(RESULTS_DIR / "metrics.txt", "w") as f:
    f.write(f"MSE (missing only): {mse:.6f}\n")

# === 绘图保存 ===
# === 绘图保存（只突出插值区域） ===
for i in range(min(5, feat_dim)):  # 只画前2个变量
    plt.figure(figsize=(15, 3))
    t = np.arange(seq_len)
    real = reals[0, :, i]
    pred = samples[0, :, i]
    mask = masks[0, :, i]  # True 表示未缺失，False 表示缺失

    # 画完整的 Ground Truth
    plt.plot(t, real, label="Ground Truth", color="g")

    # 只画缺失区域的预测值
    pred_masked = np.where(mask, np.nan, pred)
    plt.plot(t, pred_masked, label="Prediction (missing only)", color="r", linestyle="dashed")

    # 可视化 mask 区域（灰背景）
    plt.fill_between(t, real.min(), real.max(), where=~mask, color="gray", alpha=0.1, label="Missing Region")

    plt.title(f"Feature {i} Inpainting")
    plt.legend()
    plt.savefig(RESULTS_DIR / f"feature_{i}.png")
    plt.close()

print("Sample std:", np.std(samples))
print("Sample mean:", np.mean(samples))
