# MedicalGPT - 基于RLHF的医疗健康对话助手

## 项目背景

随着大语言模型（LLM）在各领域的广泛应用，**医疗健康**成为最具价值也最具挑战性的应用场景之一。通用大模型直接应用于医疗场景面临三个核心问题：

1. **领域知识缺失**：通用模型缺乏系统的医学专业知识，回答不够准确
2. **安全性风险**：模型可能生成有害的医疗建议，在医疗领域一个错误建议可能危及生命
3. **专业度不足**：缺乏医患对话的专业风格，回答过于笼统

本项目通过完整的 **三阶段RLHF训练管道**，将通用大模型（Qwen2.5-3B）对齐为安全、准确、专业的医疗对话助手：

**SFT（注入医学知识）→ Reward Model（学习安全偏好）→ PPO（强化学习对齐）**

## 训练管道

| 阶段 | 方法 | 数据 | 目标 | 预期时间 |
|-------|--------|------|------|----------|
| 阶段1 | 监督微调 (SFT) | ChatDoctor 医患对话 (10K) | 注入医学领域知识和对话风格 | ~3-4小时 |
| 阶段2 | 奖励模型 (RM) | Anthropic HH-RLHF (5K) | 学习区分有帮助vs有害的回答 | ~2小时 |
| 合并 | LoRA → 完整模型 | - | 合并适配器用于下游推理 | ~15分钟 |
| 阶段3 | PPO训练 (RLHF) | SFT数据的prompts | 优化生成策略，提升安全性 | ~4-5小时 |
| 评估 | 困惑度 + 医疗问答生成 | 医疗评估集 (200) | 定量+定性评估效果 | ~15分钟 |

**总计：约10-12小时（A100 40GB）**

## 技术架构

```
                    ┌─────────────────────────────┐
                    │   Qwen2.5-3B-Instruct       │
                    │   (基座模型)                  │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Stage 1: SFT               │
                    │   ChatDoctor 医患对话 (10K)   │
                    │   LoRA rank=16, α=32         │
                    │   → 注入医学知识              │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   LoRA Merge                 │
                    │   合并适配器→完整模型          │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                                 │
    ┌─────────▼────────┐              ┌─────────▼────────┐
    │  Stage 2: RM      │              │  Policy Model    │
    │  Anthropic RLHF   │              │  (SFT模型)       │
    │  pairwise loss    │              │                  │
    │  → 偏好打分器      │              │                  │
    └─────────┬────────┘              └─────────┬────────┘
              │                                 │
              └────────────────┬────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Stage 3: PPO (RLHF)       │
                    │   4个模型同时运行:             │
                    │   policy + ref + reward + value│
                    │   → 安全性对齐                │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Evaluation                 │
                    │   困惑度 + 医疗问答生成       │
                    └─────────────────────────────┘
```

## 项目结构

```
medicalgpt/
├── src/
│   ├── supervised_finetuning.py   # 阶段1：SFT训练
│   ├── reward_modeling.py         # 阶段2：奖励模型
│   ├── ppo_training.py            # 阶段3：PPO/RLHF
│   ├── merge_peft_adapter.py      # 合并LoRA到基础模型
│   ├── evaluate.py                # 评估管道
│   ├── prepare_data.py            # 数据集下载与转换
│   ├── template.py                # 聊天模板
│   └── utils.py                   # 通用工具
├── scripts/
│   ├── run_sft.sh                 # SFT训练脚本
│   ├── run_rm.sh                  # 奖励模型训练
│   ├── run_ppo.sh                 # PPO训练
│   ├── run_merge.sh               # 合并LoRA适配器
│   ├── run_eval.sh                # 评估
│   ├── run_all.sh                 # 全流程（所有阶段）
│   └── run_test_cpu.sh            # CPU最小化测试
├── notebooks/
│   └── run_on_colab.ipynb         # Google Colab笔记本
├── data/                          # 由prepare_data.py生成
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 快速开始（Colab）

1. 在Google Colab中打开`notebooks/run_on_colab.ipynb`
2. 选择A100 GPU运行时
3. 在配置单元中设置API密钥：
   ```python
   os.environ["HF_TOKEN"] = "hf_..."
   os.environ["HF_USERNAME"] = "your-username"
   os.environ["WANDB_API_KEY"] = "..."
   ```
4. 按顺序运行单元格

## 本地开发（CPU调试）

```bash
# 安装uv
pip install uv

# 创建环境
uv venv --python 3.11
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖（仅CPU）
uv pip install -e ".[cpu]"

# 生成测试数据
python src/prepare_data.py --mode minimal

# 运行最小化CPU测试
bash scripts/run_test_cpu.sh
```

## 功能特性

### 自动断点续训
所有训练阶段支持自动续训：
1. 检查本地`output_dir`是否存在检查点
2. 如果未找到，则检查HuggingFace Hub中是否有匹配的仓库
3. 如果可用，从Hub检查点下载并续训

### 基于超参数的命名
模型名称基于超参数自动生成：
```
sft-qwen2.5-3b-instruct-lr2e-5-r16-a32-bs4-ep3-ml2048
rm-qwen2.5-3b-instruct-lr2e-5-r16-a32-bs4-ep2
ppo-qwen2.5-3b-instruct-bs1-ep1
```

### HuggingFace Hub集成
- 检查点在固定步数间隔上传
- 每个阶段结束后上传最终模型
- 公共仓库便于访问

### wandb日志记录
所有训练指标记录到Weights & Biases：
- 损失曲线、学习率
- 评估指标（困惑度、奖励准确率）
- GPU利用率

## 超参数调优

所有超参数可在每个脚本顶部配置：

```bash
# scripts/run_sft.sh
LEARNING_RATE=2e-5     # 尝试：1e-5, 2e-5, 5e-5
LORA_RANK=16           # 尝试：8, 16, 32, 64
LORA_ALPHA=32          # 尝试：16, 32, 64
BATCH_SIZE=4           # 根据GPU显存调整
NUM_EPOCHS=3           # 尝试：1, 2, 3
MAX_LENGTH=2048        # 尝试：1024, 2048, 4096
```

更改超参数时，输出目录和HF Hub仓库名称会自动更改，因此不同实验不会覆盖彼此。

## 数据集

| 数据集 | 来源 | 用途 | 样本数 | 说明 |
|---------|--------|-------|---------|------|
| ChatDoctor | `lavita/ChatDoctor-HealthCareMagic-100k` | SFT训练 | 10,000 | 真实医患对话，注入医学领域知识 |
| Anthropic HH-RLHF | `Anthropic/hh-rlhf` | 奖励模型 | 5,000 | 人类偏好数据，训练安全对齐偏好 |
| 医疗评估集 | ChatDoctor hold-out | 评估 | 200 | 计算困惑度 + 定性评估 |

**数据选择理由**：
- **SFT用ChatDoctor**：来自HealthCareMagic在线医疗咨询平台的真实医患对话，覆盖内科、外科、儿科、妇科、皮肤科等多个专科，格式为单轮问答，数据质量高
- **RM用Anthropic HH-RLHF**：医疗领域的偏好标注数据稀缺且标注成本极高（需要医学专家）。Anthropic HH-RLHF包含"有帮助"vs"有害"的通用偏好标注，其安全对齐能力可以泛化到医疗领域——让模型学会拒绝给出危险建议、避免幻觉

## 硬件需求

| 组件 | A100 40GB | A100 80GB |
|-----------|-----------|-----------|
| SFT (LoRA) | ✅ | ✅ |
| 奖励模型 (LoRA) | ✅ | ✅ |
| PPO (4个模型) | ✅（紧张） | ✅ |

## 技术要点

1. **SFT阶段**: 使用LoRA (rank=16) 对Qwen2.5-3B进行指令微调，训练数据为ChatDoctor医患对话，使用ShareGPT格式，为模型注入医学领域知识
2. **Reward Model**: 基于InstructGPT的pairwise logloss训练偏好排序模型: $L = -\log\sigma(r_{chosen} - r_{rejected})$，学习区分有帮助vs有害的回答
3. **PPO阶段**: 使用TRL库的PPOTrainer，加载4个模型(policy, ref_policy, reward, value)进行强化学习对齐，使医疗回答更安全
4. **LoRA**: 低秩适配器微调，将权重更新分解为 $\Delta W = BA$，其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d,k)$，仅训练约0.5%的参数
5. **Checkpoint设计**: 基于超参数命名的自动断点续训机制，支持本地+HuggingFace Hub双重检查
6. **评估**: 困惑度(Perplexity)定量评估 + 医疗问答生成定性评估

## 面试指南

### Q: 这个项目是做什么的？

> "这是一个完整的医疗大模型对齐项目。核心问题是：通用大模型在医疗场景下缺乏专业知识且存在安全风险。我实现了从SFT到奖励模型到PPO的三阶段RLHF管道，将Qwen2.5-3B对齐为安全、准确的医疗对话助手。"

### Q: 为什么需要RLHF，SFT不够吗？

> "SFT的本质是最大似然估计（MLE），通过模仿学习掌握知识，但它有两个关键局限：
>
> 1. **无法区分回答质量的相对好坏**：SFT对所有训练样本一视同仁，不能学习'这个回答比那个好'这种偏好
> 2. **Exposure bias**：训练时看到的是正确回答，但推理时依赖自己之前的输出，导致误差累积
>
> 在医疗领域这尤其关键——一个'基本正确但有微小错误'的回答和一个'完全正确'的回答，SFT可能给两者相似的概率。RLHF通过奖励模型学习偏好排序，再用PPO优化策略，让模型学会生成更安全、更有帮助的回答。"

### Q: LoRA的原理是什么？为什么用LoRA？

> "LoRA的核心假设是：模型微调时的权重变化矩阵是低秩的。它将权重更新分解为两个低秩矩阵的乘积 $\Delta W = BA$，其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d,k)$。
>
> 选择LoRA的原因是PPO阶段需要**同时加载4个模型**（policy、reference、reward、value），如果全参数微调，3B模型×4 = 12B参数 + 优化器状态，A100 40GB完全放不下。使用 rank=16 的LoRA，可训练参数仅占总参数的0.5%左右。"

### Q: 奖励模型怎么训练的？

> "基于InstructGPT论文的方法。输入一个question和一对回答（chosen和rejected），模型分别对两个回答打分 $r_{chosen}$ 和 $r_{rejected}$，使用 pairwise ranking loss：
>
> $$L = -\log\sigma(r_{chosen} - r_{rejected})$$
>
> 直觉是：让chosen的得分尽可能高于rejected。训练数据来自Anthropic的HH-RLHF数据集，包含人类标注的偏好对——标注员判断哪个回答更有帮助、更安全。"

### Q: PPO的训练流程？

> "PPO阶段使用4个模型协同工作：
>
> 1. **Policy模型**（被优化的SFT模型）：生成response
> 2. **Reference模型**（冻结的SFT模型）：计算KL散度惩罚，防止policy偏离太远
> 3. **Reward模型**（打分器）：对response打分
> 4. **Value模型**（状态价值网络）：估计状态价值，用于计算优势函数
>
> 训练过程：给定prompt → policy生成response → reward打分 → 计算KL惩罚 → 用GAE估计优势 → 更新policy和value网络。KL惩罚项确保模型不会为了追求高reward而生成离谱的内容（reward hacking）。"

### Q: 为什么RM不用医疗专用数据？

> "医疗领域的偏好标注数据极其稀缺且标注成本高——需要有资质的医学专家来判断哪个回答更好，每条标注成本可能是普通标注的10倍以上。Anthropic HH-RLHF虽然是通用偏好数据，但它的核心是训练模型区分'有帮助的回答'和'有害的回答'，这种安全对齐能力具有很强的泛化性。在medical domain，让模型学会拒绝给出不确定的诊断、不推荐未经验证的治疗方案，本质上就是helpfulness vs harmlessness的判断。"

### Q: 工程上有什么亮点？

> "三点：
> 1. **自动断点续训**：先查本地检查点，再查HuggingFace Hub。训练意外中断后，在任何机器上都能自动恢复，不浪费算力
> 2. **超参数命名系统**：输出目录自动包含核心超参数（如 `sft-qwen2.5-3b-lr2e-5-r16-a32-bs4-ep3`），不同实验互不干扰，方便对比
> 3. **全流程自动化**：`run_all.sh` 一键从数据准备到最终评估。Colab notebook可直接在云端A100上运行完整管道"

## License

Apache License 2.0
