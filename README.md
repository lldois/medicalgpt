# MedicalGPT - Full RLHF Training Pipeline

Complete implementation of the RLHF (Reinforcement Learning from Human Feedback) pipeline:

**SFT → Reward Model → PPO**

Based on Qwen2.5-3B with LoRA fine-tuning, designed for A100 40GB GPU.

## Pipeline Overview

| Stage | Method | Data | Expected Time |
|-------|--------|------|---------------|
| Stage 1 | Supervised Fine-Tuning (SFT) | UltraChat 200k (10K subset) | ~3-4h |
| Stage 2 | Reward Model Training (RM) | Anthropic HH-RLHF (5K subset) | ~2h |
| Merge | LoRA → Full Model | - | ~15min |
| Stage 3 | PPO Training (RLHF) | SFT data prompts | ~4-5h |
| Eval | Perplexity + Generation | UltraChat test (200) | ~15min |

**Total: ~10-12 hours on A100 40GB**

## Project Structure

```
medicalgpt/
├── src/
│   ├── supervised_finetuning.py   # Stage 1: SFT training
│   ├── reward_modeling.py         # Stage 2: Reward model
│   ├── ppo_training.py            # Stage 3: PPO/RLHF
│   ├── merge_peft_adapter.py      # Merge LoRA into base model
│   ├── evaluate.py                # Evaluation pipeline
│   ├── prepare_data.py            # Dataset download & conversion
│   ├── template.py                # Chat prompt templates
│   └── utils.py                   # Shared utilities
├── scripts/
│   ├── run_sft.sh                 # SFT training script
│   ├── run_rm.sh                  # Reward model training
│   ├── run_ppo.sh                 # PPO training
│   ├── run_merge.sh               # Merge LoRA adapters
│   ├── run_eval.sh                # Evaluation
│   ├── run_all.sh                 # Full pipeline (all stages)
│   └── run_test_cpu.sh            # CPU minimal test
├── notebooks/
│   └── run_on_colab.ipynb         # Google Colab notebook
├── data/                          # Populated by prepare_data.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick Start (Colab)

1. Open `notebooks/run_on_colab.ipynb` in Google Colab
2. Select A100 GPU runtime
3. Set your API keys in the config cell:
   ```python
   os.environ["HF_TOKEN"] = "hf_..."
   os.environ["HF_USERNAME"] = "your-username"
   os.environ["WANDB_API_KEY"] = "..."
   ```
4. Run cells sequentially

## Local Development (CPU Debugging)

```bash
# Install uv
pip install uv

# Create environment
uv venv --python 3.11
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies (CPU only)
uv pip install -e ".[cpu]"

# Generate test data
python src/prepare_data.py --mode minimal

# Run minimal CPU test
bash scripts/run_test_cpu.sh
```

## Features

### Automatic Checkpoint Resume
All training stages support automatic resume:
1. Checks local `output_dir` for existing checkpoints
2. If not found, checks HuggingFace Hub for matching repo
3. Downloads and resumes from Hub checkpoint if available

### Hyperparameter-Based Naming
Models are named based on hyperparameters:
```
sft-qwen2.5-3b-instruct-lr2e-5-r16-a32-bs4-ep3-ml2048
rm-qwen2.5-3b-instruct-lr2e-5-r16-a32-bs4-ep2
ppo-qwen2.5-3b-instruct-bs1-ep1
```

### HuggingFace Hub Integration
- Checkpoints uploaded at fixed step intervals
- Final models uploaded after each stage
- Public repos for easy access

### wandb Logging
All training metrics logged to Weights & Biases:
- Loss curves, learning rate
- Evaluation metrics (perplexity, reward accuracy)
- GPU utilization

## Hyperparameter Tuning

All hyperparameters are configurable at the top of each shell script:

```bash
# scripts/run_sft.sh
LEARNING_RATE=2e-5     # Try: 1e-5, 2e-5, 5e-5
LORA_RANK=16           # Try: 8, 16, 32, 64
LORA_ALPHA=32          # Try: 16, 32, 64
BATCH_SIZE=4           # Adjust based on GPU memory
NUM_EPOCHS=3           # Try: 1, 2, 3
MAX_LENGTH=2048        # Try: 1024, 2048, 4096
```

When you change hyperparameters, the output directory and HF Hub repo name automatically change, so different experiments don't overwrite each other.

## Datasets

| Dataset | Source | Usage | Samples |
|---------|--------|-------|---------|
| UltraChat 200k | HuggingFaceH4/ultrachat_200k | SFT Training | 10,000 |
| Anthropic HH-RLHF | Anthropic/hh-rlhf | Reward Model | 5,000 |
| UltraChat Test | HuggingFaceH4/ultrachat_200k (test) | Evaluation | 200 |

## Hardware Requirements

| Component | A100 40GB | A100 80GB |
|-----------|-----------|-----------|
| SFT (LoRA) | ✅ | ✅ |
| Reward Model (LoRA) | ✅ | ✅ |
| PPO (4 models) | ✅ (tight) | ✅ |

## 技术要点 (面试参考)

1. **SFT阶段**: 使用LoRA (rank=16) 对Qwen2.5-3B进行指令微调，训练数据为UltraChat多轮对话，使用ShareGPT格式
2. **Reward Model**: 基于InstructGPT的pairwise logloss训练偏好排序模型: $L = -\log\sigma(r_{chosen} - r_{rejected})$
3. **PPO阶段**: 使用TRL库的PPOTrainer，加载4个模型(policy, ref_policy, reward, value)进行强化学习对齐
4. **LoRA**: 低秩适配器微调，仅训练约0.5%的参数，大幅降低显存需求
5. **Checkpoint设计**: 基于超参数命名的自动断点续训机制，支持本地+HuggingFace Hub双重检查
6. **评估**: 困惑度(Perplexity)定量评估 + 生成样本定性评估

## License

Apache License 2.0
