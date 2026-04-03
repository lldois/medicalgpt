# MedicalGPT - 基于RLHF的医疗健康对话助手

## 项目背景

随着大语言模型（LLM）在各领域的广泛应用，**医疗健康**成为最具价值也最具挑战性的应用场景之一。通用大模型直接应用于医疗场景面临三个核心问题：

1. **领域知识缺失**：通用模型缺乏系统的医学专业知识，回答不够准确
2. **安全性风险**：模型可能生成有害的医疗建议，在医疗领域一个错误建议可能危及生命
3. **专业度不足**：缺乏医患对话的专业风格，回答过于笼统

本项目通过完整的 **四阶段RLHF训练管道**，将通用大模型（Qwen2.5-3B）对齐为安全、准确、专业的医疗对话助手：

**继续预训练（注入医学知识）→ SFT（学习问答风格）→ Reward Model（学习偏好）→ PPO（强化对齐）**

## 训练管道

| 阶段 | 方法 | 数据 | 评估指标 | 预期时间 |
|-------|--------|------|----------|----------|
| Stage 0 | 继续预训练 (PT) | 医学百科 (361K) + 医学教材 (8K) | PPL (困惑度) | ~3小时 |
| Stage 1 | 监督微调 (SFT) | 中文医疗问答 (195万, 取子集) | PPL + BLEU-4 + ROUGE-L | ~3小时 |
| — | LoRA合并 | — | — | ~15分钟 |
| Stage 2 | 奖励模型 (RM) | 医疗偏好排序 (3,800) | Accuracy + Mean Reward Gap | ~1小时 |
| Stage 3 | PPO训练 (RLHF) | SFT prompts + RM打分 | Mean Reward + BLEU/ROUGE | ~4小时 |

**总计：约10-12小时（A100 40GB）** ｜ 灵活设计：可跳过预训练，直接从SFT开始

## 技术架构

```
                    ┌─────────────────────────────┐
                    │   Qwen2.5-3B                │
                    │   (基座模型)                  │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Stage 0: 继续预训练 (可选)  │
                    │   医学百科 + 医学教材          │
                    │   CLM, LoRA rank=16          │
                    │   → 注入医学领域知识           │
                    │   评估: PPL ↓                 │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Stage 1: SFT               │
                    │   中文医疗问答 (20K子集)       │
                    │   LoRA rank=16, α=32         │
                    │   → 学习医患对话风格           │
                    │   评估: PPL↓ BLEU↑ ROUGE↑    │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                                 │
    ┌─────────▼────────┐              ┌─────────▼────────┐
    │  Stage 2: RM      │              │  Policy Model    │
    │  医疗偏好排序      │              │  (SFT模型)       │
    │  pairwise loss    │              │                  │
    │  → 偏好打分器      │              │                  │
    │  评估: Acc↑ Gap↑  │              │                  │
    └─────────┬────────┘              └─────────┬────────┘
              │                                 │
              └────────────────┬────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Stage 3: PPO (RLHF)       │
                    │   4个模型协同:                 │
                    │   policy + ref + reward + value│
                    │   → 安全性对齐                │
                    │   评估: Reward↑ BLEU↑ ROUGE↑ │
                    └─────────────────────────────┘
```

## 评估体系

| 阶段 | 指标 | 含义 | 期望方向 |
|------|------|------|----------|
| PT | PPL (困惑度) | 模型对医学文本的预测能力 | ↓ 越低越好 |
| SFT | PPL | 模型对问答对的预测能力 | ↓ 越低越好 |
| SFT | BLEU-4 | 生成回答与参考答案的n-gram重合度 | ↑ 越高越好 |
| SFT | ROUGE-L | 生成回答与参考答案的最长公共子序列 | ↑ 越高越好 |
| RM | Accuracy | 正确判断 chosen > rejected 的比例 | ↑ 越高越好 |
| RM | Mean Reward Gap | chosen与rejected的平均分数差 | ↑ 越大越好 |
| RLHF | Mean Reward | RM对PPO生成回答的平均打分 | ↑ 越高越好 |
| RLHF | BLEU-4 / ROUGE-L | PPO生成回答的质量 | ↑ 对比SFT |

## 数据集

使用 [shibing624/medical](https://huggingface.co/datasets/shibing624/medical) 数据集，涵盖训练全阶段：

| 阶段 | 数据 | 格式 | 训练集 | 验证/测试 |
|------|------|------|--------|-----------|
| 预训练 | 医学百科 + 医学教材 | `{"text": "..."}` | 361K + 8K | 500 / 500 |
| SFT | 中文医疗问答 | `{"instruction", "input", "output"}` → ShareGPT | 195万 (取子集) | 500 / 500 |
| 奖励 | 医疗偏好排序 | `{"question", "response_chosen", "response_rejected"}` | 3,800 | 100 / 100 |

**数据选择理由**：
- **统一数据源**：预训练、SFT、RM三阶段数据来自同一个医疗数据集，领域一致性好
- **中文医疗数据**：195万中文医疗问答覆盖内、外、妇、儿、皮肤等多个专科
- **真实偏好标注**：3800条医疗偏好排序数据，由专业标注员判断答案质量

## 项目结构

```
medicalgpt/
├── src/
│   ├── pretrain.py                # Stage 0：继续预训练 (CLM)
│   ├── supervised_finetuning.py   # Stage 1：SFT训练
│   ├── reward_modeling.py         # Stage 2：奖励模型
│   ├── ppo_training.py            # Stage 3：PPO/RLHF
│   ├── merge_peft_adapter.py      # 合并LoRA到基础模型
│   ├── evaluate.py                # 多阶段评估管道
│   ├── prepare_data.py            # 数据集下载与转换
│   ├── template.py                # 聊天模板
│   └── utils.py                   # 通用工具
├── scripts/
│   ├── run_pretrain.sh            # 预训练脚本
│   ├── run_sft.sh                 # SFT训练脚本
│   ├── run_rm.sh                  # 奖励模型训练
│   ├── run_ppo.sh                 # PPO训练
│   ├── run_merge.sh               # 合并LoRA (支持 pt/sft/rm/all)
│   ├── run_eval.sh                # 评估 (支持 pretrain/sft/rm/rlhf)
│   ├── run_all.sh                 # 全流程（所有阶段 + 每阶段评估）
│   └── run_test_cpu.sh            # CPU最小化测试
├── notebooks/
│   └── run_on_colab.ipynb         # Google Colab 灵活训练笔记本
├── data/                          # 由 prepare_data.py 生成
│   ├── pretrain_prepared/         #   预训练数据
│   ├── sft/                       #   SFT对话数据 (ShareGPT格式)
│   └── reward_converted/          #   奖励偏好数据
├── eval_results/                  # 各阶段评估结果
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 快速开始（Colab）

1. 在Google Colab中打开 `notebooks/run_on_colab.ipynb`
2. 选择 A100 GPU 运行时
3. 在配置单元中设置API密钥
4. 按顺序运行 — 每个阶段可独立运行，可跳过预训练

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

# 运行最小化CPU测试（预训练 + SFT + RM + 各阶段评估）
bash scripts/run_test_cpu.sh
```

## 功能特性

### 灵活的阶段跳过
- 预训练效果不好？直接跳过，SFT会自动使用原始基座模型
- PPO不稳定？跳过PPO，使用SFT模型即可
- Colab notebook 中每个阶段带配置说明，支持独立运行

### 自动断点续训
所有训练阶段支持自动续训：
1. 检查本地 `output_dir` 是否存在检查点
2. 如果未找到，则检查 HuggingFace Hub 中是否有匹配的仓库
3. 如果可用，从Hub检查点下载并续训

### 基于超参数的命名
模型名称基于超参数自动生成，不同实验互不覆盖：
```
pt-qwen2.5-3b-lr2e-5-r16-a32-bs4-ep1-bl1024
sft-qwen2.5-3b-lr2e-5-r16-a32-bs4-ep3-ml2048
rm-qwen2.5-3b-lr2e-5-r16-a32-bs4-ep2
```

### 多阶段评估
训练完成后，每个阶段自动运行对应的评估脚本：
```bash
bash scripts/run_eval.sh pretrain ./merged-pt          # PPL
bash scripts/run_eval.sh sft ./merged-sft              # PPL + BLEU + ROUGE
bash scripts/run_eval.sh rm ./merged-rm                # Accuracy + Reward Gap
bash scripts/run_eval.sh rlhf ./ppo-output ./merged-rm # Reward + BLEU + ROUGE
```

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

## 硬件需求

| 组件 | A100 40GB | A100 80GB |
|-----------|-----------|-----------|
| 预训练 (LoRA) | ✅ | ✅ |
| SFT (LoRA) | ✅ | ✅ |
| 奖励模型 (LoRA) | ✅ | ✅ |
| PPO (4个模型) | ✅（紧张） | ✅ |

## 技术要点

1. **继续预训练**: 在医学百科和教材上进行因果语言建模 (CLM)，使用 LoRA (rank=16)，让基座模型学习医学领域知识。使用 `DataCollatorForLanguageModeling`，将文本分组为固定长度 block
2. **SFT阶段**: 在195万中文医疗问答（取子集）上进行指令微调，数据转换为 ShareGPT 对话格式，使模型学会以医生视角回答问题
3. **Reward Model**: 基于InstructGPT的 pairwise logloss 训练偏好排序模型: $L = -\log\sigma(r_{chosen} - r_{rejected})$，在3800条医疗偏好对上学习区分好坏回答
4. **PPO阶段**: 使用TRL库的PPOTrainer，加载4个模型(policy, ref_policy, reward, value)进行强化学习对齐，使医疗回答更安全
5. **LoRA**: 低秩适配器微调，将权重更新分解为 $\Delta W = BA$，其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d,k)$，仅训练约0.5%的参数
6. **评估体系**: 每阶段有针对性的评估指标 — 预训练看PPL，SFT看PPL+BLEU+ROUGE，RM看准确率，RLHF看Reward分数。BLEU/ROUGE 使用字符级实现，无额外依赖

## 面试指南

### Q: 这个项目是做什么的？

> "这是一个完整的医疗大模型对齐项目，实现了四阶段训练管道：继续预训练注入医学知识 → SFT学习问答风格 → 奖励模型学习偏好 → PPO强化对齐。使用 shibing624/medical 中文医疗数据集，将 Qwen2.5-3B 对齐为安全、准确的医疗对话助手。每个阶段都有对应的评估指标（PPL/BLEU/ROUGE/Accuracy/Reward）。"

### Q: 为什么要加继续预训练？

> "通用大模型的预训练语料中医学知识占比很少。继续预训练的目的是通过大量医学百科文本和教材，让模型'补课'——学习医学术语、疾病机理、药物知识等。这一步通过PPL评估效果：如果PPL明显下降，说明模型对医学文本的理解能力提升了。如果效果不明显，可以跳过直接SFT，项目设计支持灵活跳过。"

### Q: 为什么需要RLHF，SFT不够吗？

> "SFT的本质是最大似然估计（MLE），通过模仿学习掌握知识，但它有两个关键局限：
>
> 1. **无法区分回答质量的相对好坏**：SFT对所有训练样本一视同仁，不能学习'这个回答比那个好'
> 2. **Exposure bias**：训练时看到的是正确回答，但推理时依赖自己之前的输出，导致误差累积
>
> 在医疗领域，一个'基本正确但有微小错误'的回答和一个'完全正确'的回答，SFT可能给两者相似的概率。RLHF通过奖励模型学习偏好排序，再用PPO优化策略，让模型学会生成更安全、更专业的回答。"

### Q: LoRA的原理是什么？为什么用LoRA？

> "LoRA的核心假设是：模型微调时的权重变化矩阵是低秩的。它将权重更新分解为两个低秩矩阵的乘积 $\Delta W = BA$，其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d,k)$。
>
> 选择LoRA的原因：PPO阶段需要**同时加载4个模型**（policy、reference、reward、value），如果全参数微调，3B模型×4 = 12B参数 + 优化器状态，A100 40GB完全放不下。使用 rank=16 的LoRA，可训练参数仅占总参数的0.5%左右。"

### Q: 奖励模型怎么训练的？

> "基于InstructGPT论文的方法。输入一个question和一对回答（chosen和rejected），模型分别对两个回答打分 $r_{chosen}$ 和 $r_{rejected}$，使用 pairwise ranking loss：
>
> $$L = -\log\sigma(r_{chosen} - r_{rejected})$$
>
> 训练数据来自 shibing624/medical 的3800条医疗偏好排序数据，标注员判断哪个回答质量更高。评估时看准确率（chosen分数是否高于rejected）和平均分数差距。"

### Q: PPO的训练流程？

> "PPO阶段使用4个模型协同工作：
>
> 1. **Policy模型**（被优化的SFT模型）：生成response
> 2. **Reference模型**（冻结的SFT模型）：计算KL散度惩罚，防止policy偏离太远
> 3. **Reward模型**（打分器）：对response打分
> 4. **Value模型**（状态价值网络）：估计状态价值，用于计算优势函数
>
> 训练过程：给定prompt → policy生成response → reward打分 → 计算KL惩罚 → 用GAE估计优势 → 更新policy和value网络。KL惩罚项确保模型不会为了追求高reward而生成离谱的内容（reward hacking）。"

### Q: 每个阶段怎么评估效果？

> "每个阶段有针对性的定量评估：
>
> - **预训练**：看PPL（困惑度），衡量模型对医学文本的预测能力，越低说明学到的医学知识越多
> - **SFT**：PPL + BLEU-4 + ROUGE-L。BLEU衡量生成答案与参考答案的n-gram重合度，ROUGE-L看最长公共子序列。因为是中文，使用字符级计算
> - **RM**：准确率（chosen得分>rejected得分的比例，理想值>0.7）+ 平均reward差距（越大说明模型区分能力越强）
> - **RLHF**：RM对PPO生成回答的平均打分 + BLEU/ROUGE（对比SFT是否有提升）"

### Q: 工程上有什么亮点？

> "四点：
> 1. **灵活的阶段设计**：预训练效果不好可以跳过，SFT自动检测是否有预训练模型并选择基座。PPO也可跳过
> 2. **自动断点续训**：先查本地检查点，再查HuggingFace Hub。训练中断后在任何机器上都能自动恢复
> 3. **超参数命名系统**：输出目录自动包含核心超参数，不同实验互不干扰
> 4. **完整评估体系**：每个阶段训练后自动运行对应评估，结果保存为JSON便于对比。BLEU/ROUGE使用字符级原生实现，不需要额外依赖"

## License

Apache License 2.0
