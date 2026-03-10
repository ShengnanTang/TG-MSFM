# Makefile for FMTS interpolation tasks
ENV_DIR := .venv
PYTHON_VERSION := 3.8.10

# ---------- 通用设置 ----------
ENV_DIR        := .venv
PYTHON         := $(ENV_DIR)/bin/python
PYTHON_VERSION := 3.8.10

# ---------- 配置文件 ----------
CFG_B0 := Config/etth_B0.yaml
CFG_C1 := Config/etth_C1.yaml

# ---------- 实验别名 ----------
NAME_B0 := etth_b0_gap1k
NAME_C1 := etth_c1_gap1k

CKPT_DIR_B0 := .Checkpoints/Checkpoints_$(NAME_B0)_8000
CKPT_DIR_C1 := .Checkpoints/Checkpoints_$(NAME_C1)_8000

# 保存周期 1800 ⇒ 第一次 checkpoint-1.pt，第二次 checkpoint-2.pt…
CKPT_TAG := 5        # 评估第几号 checkpoint；按需改
.PHONY: setup clean activate run train evaluate \
        train-gap1k  infer-gap1k  full-gap1k


# ---------- 原有目标保持不变 ----------
setup:
	@command -v pyenv >/dev/null 2>&1 || { echo '❌ pyenv not found. Please install pyenv.'; exit 1; }
	@if ! pyenv versions --bare | grep -qx "$(PYTHON_VERSION)"; then \
		echo "⬇️ Installing Python $(PYTHON_VERSION)..."; \
		pyenv install $(PYTHON_VERSION); \
	fi
	@echo "$(PYTHON_VERSION)" > .python-version
	@rm -rf $(ENV_DIR)
	@pyenv exec python -m venv $(ENV_DIR)

	@$(ENV_DIR)/bin/pip install --upgrade pip setuptools wheel
	@$(ENV_DIR)/bin/pip install -r requirements.txt || echo "⚠️ Please provide a requirements.txt file."
	@echo "✅ Setup complete. Activate with: source $(ENV_DIR)/bin/activate"

run:
	bash scripts/run_interpolation.sh

train:
	@$(ENV_DIR)/bin/python train.py

evaluate:
	@$(ENV_DIR)/bin/python evaluate.py

clean:
	rm -rf $(ENV_DIR)

debug:
	@export results_folder=./Checkpoints_debug && \
	PYTHONPATH=. .venv/bin/python main.py \
		--train \
		--config_file Config/etth.yaml \
		--name debug_run \
		--long_len 24 \
		--missing_ratio 0.3

sample:
	@echo "🚀 Running inference and plotting..."
	@$(ENV_DIR)/bin/python sample_and_plot.py


# ==================== 目标 ====================
.PHONY: train-b0 train-c1 infer-b0 infer-c1 \
        full-b0 full-c1 full-all

# ------- Baseline-B0 -------
train-b0:
	@echo "🏋️  Train B0 ..."
	@CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. results_folder=$(CKPT_DIR_B0) \
	$(PYTHON) main.py \
	    --train \
	    --config_file $(CFG_B0) \
	    --name $(NAME_B0) \
	    --gpu 0           # 可选：显式绑定“可见索引0”（=物理#3）


infer-b0:
	@echo "🔍  Inference B0 ..."
	@hucfg_num_steps=400 hucfg_Kscale=0.03 results_folder=$(CKPT_DIR_B0) PYTHONPATH=. \
	$(PYTHON) eval_infill.py \
	    --config $(CFG_B0) \
	    --ckpt $(CKPT_TAG) \
	    --name  $(CKPT_DIR_B0) \
	    --plot_num 3

full-b0: train-b0 infer-b0

# ------- Baseline-C1 -------
train-c1:
	@echo "🏋️  Train C1 on GPU1 ..."
	@CUDA_VISIBLE_DEVICES=0,1 \
	 export results_folder=$(CKPT_DIR_C1) && \
	 PYTHONPATH=. $(PYTHON) main.py \
	    --train \
	    --config_file $(CFG_C1) \
	    --name $(NAME_C1)

infer-c1:
	@echo "🔍  Inference C1 ..."
	@hucfg_num_steps=400 hucfg_Kscale=0.05 results_folder=$(CKPT_DIR_C1) PYTHONPATH=. \
	$(PYTHON) eval_infill.py \
	    --config $(CFG_C1) \
	    --ckpt  $(CKPT_TAG) \
	    --name  $(CKPT_DIR_C1) \
	    --plot_num 3

full-c1: train-c1 infer-c1

# ------- 同时跑两个 baseline（可 -j 并行） -------
full-all: full-b0 full-c1
