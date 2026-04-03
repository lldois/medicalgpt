# -*- coding: utf-8 -*-
"""
Download and prepare shibing624/medical dataset for the full RLHF pipeline.

Dataset: https://huggingface.co/datasets/shibing624/medical
Contains:
  - pretrain/: 医学百科 + 医学教材 ({"text": "..."})
  - finetune/: 医患对话 ({"instruction": "...", "input": "", "output": "..."})
  - reward/:   偏好对比 ({"question": "...", "response_chosen": "...", "response_rejected": "..."})

Data conversion:
  - SFT data is converted to ShareGPT format: {"conversations": [...]}
  - Reward data gets "system" and "history" fields added for compatibility
  - Pretrain data is kept as-is: {"text": "..."}
"""

import json
import os
import argparse
from pathlib import Path
from loguru import logger


def download_dataset(data_dir: str):
    """Download shibing624/medical dataset files from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    repo_id = "shibing624/medical"
    files = {
        "pretrain": [
            "pretrain/train_encyclopedia.json",
            "pretrain/valid_encyclopedia.json",
            "pretrain/test_encyclopedia.json",
            "pretrain/medical_book_zh.json",
        ],
        "finetune": [
            "finetune/train_zh_0.json",
            "finetune/valid_zh_0.json",
            "finetune/test_zh_0.json",
            "finetune/train_en_1.json",
            "finetune/valid_en_1.json",
            "finetune/test_en_1.json",
        ],
        "reward": [
            "reward/train.json",
            "reward/valid.json",
            "reward/test.json",
        ],
    }

    for stage, file_list in files.items():
        stage_dir = os.path.join(data_dir, stage)
        os.makedirs(stage_dir, exist_ok=True)
        for filepath in file_list:
            filename = os.path.basename(filepath)
            local_path = os.path.join(stage_dir, filename)
            if os.path.exists(local_path):
                logger.info(f"Already exists: {local_path}")
                continue
            logger.info(f"Downloading {filepath}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filepath,
                repo_type="dataset",
                local_dir=data_dir,
            )
    logger.info("Dataset download complete!")


def convert_sft_data(data_dir: str, max_samples: int = -1, seed: int = 42):
    """
    Convert finetune data (instruction/input/output) to ShareGPT conversation format.
    Uses Chinese data (train_zh_0.json) by default.
    Creates train/valid/test splits in conversations format.
    """
    import random

    random.seed(seed)
    finetune_dir = os.path.join(data_dir, "finetune")
    output_dir = os.path.join(data_dir, "sft")
    os.makedirs(output_dir, exist_ok=True)

    for split_prefix, output_name in [
        ("train_zh_0", "train.jsonl"),
        ("valid_zh_0", "valid.jsonl"),
        ("test_zh_0", "test.jsonl"),
    ]:
        input_path = os.path.join(finetune_dir, f"{split_prefix}.json")
        output_path = os.path.join(output_dir, output_name)

        if not os.path.exists(input_path):
            logger.warning(f"File not found: {input_path}")
            continue

        records = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                instruction = (item.get("instruction") or "").strip()
                input_text = (item.get("input") or "").strip()
                output_text = (item.get("output") or "").strip()
                if not instruction or not output_text:
                    continue
                patient_query = f"{instruction}\n{input_text}".strip() if input_text else instruction
                records.append({
                    "conversations": [
                        {"from": "human", "value": patient_query},
                        {"from": "gpt", "value": output_text},
                    ]
                })

        if max_samples > 0 and "train" in split_prefix and len(records) > max_samples:
            random.shuffle(records)
            records = records[:max_samples]

        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info(f"SFT {output_name}: {len(records)} samples -> {output_path}")


def convert_reward_data(data_dir: str):
    """
    Add 'system' and 'history' fields to reward data for compatibility with RM training script.
    """
    reward_dir = os.path.join(data_dir, "reward")
    output_dir = os.path.join(data_dir, "reward_converted")
    os.makedirs(output_dir, exist_ok=True)

    for filename in ["train.json", "valid.json", "test.json"]:
        input_path = os.path.join(reward_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".json", ".jsonl"))

        if not os.path.exists(input_path):
            logger.warning(f"File not found: {input_path}")
            continue

        count = 0
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                record = {
                    "system": "",
                    "history": [],
                    "question": item.get("question", ""),
                    "response_chosen": item.get("response_chosen", ""),
                    "response_rejected": item.get("response_rejected", ""),
                }
                if record["question"] and record["response_chosen"]:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

        logger.info(f"Reward {filename}: {count} samples -> {output_path}")


def prepare_pretrain_data(data_dir: str, max_samples: int = -1, seed: int = 42):
    """
    Prepare pretrain data splits. The raw data is already in {"text": "..."} format.
    """
    import random

    random.seed(seed)
    pretrain_dir = os.path.join(data_dir, "pretrain")
    output_dir = os.path.join(data_dir, "pretrain_prepared")
    os.makedirs(output_dir, exist_ok=True)

    # Training: combine encyclopedia + medical books, optionally subsample
    train_files = ["train_encyclopedia.json", "medical_book_zh.json"]
    records = []
    for fname in train_files:
        fpath = os.path.join(pretrain_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(line)

    if max_samples > 0 and len(records) > max_samples:
        random.shuffle(records)
        records = records[:max_samples]

    train_out = os.path.join(output_dir, "train.jsonl")
    with open(train_out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec + "\n")
    logger.info(f"Pretrain train: {len(records)} samples -> {train_out}")

    # Valid & Test: use encyclopedia valid/test
    for split in ["valid", "test"]:
        src = os.path.join(pretrain_dir, f"{split}_encyclopedia.json")
        dst = os.path.join(output_dir, f"{split}.jsonl")
        if not os.path.exists(src):
            continue
        count = 0
        with open(src, "r", encoding="utf-8") as fin, \
             open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                if line.strip():
                    fout.write(line.strip() + "\n")
                    count += 1
        logger.info(f"Pretrain {split}: {count} samples -> {dst}")


def prepare_minimal_test_data(base_dir: str):
    """
    Create tiny medical datasets for CPU debugging (5 samples each).
    """
    # Pretrain: medical text
    pretrain_dir = os.path.join(base_dir, "pretrain_prepared")
    os.makedirs(pretrain_dir, exist_ok=True)
    medical_texts = [
        "高血压是一种常见的慢性疾病，长期高血压可导致心脑血管疾病。治疗方法包括生活方式调整和药物治疗。常用降压药物有ACEI类、ARB类、钙通道阻滞剂等。",
        "糖尿病分为1型和2型。1型糖尿病是由于胰岛β细胞被破坏导致胰岛素缺乏，2型糖尿病是由于胰岛素抵抗和胰岛素分泌不足所致。",
        "感冒是由病毒引起的上呼吸道感染，常见症状有鼻塞、流涕、咳嗽、咽痛等。治疗以对症处理为主，注意休息和补充水分。",
        "心肌梗死是由于冠状动脉急性闭塞导致心肌缺血坏死。主要表现为持续性胸骨后疼痛，可伴有大汗、恶心等。需立即就医进行急诊介入治疗。",
        "肺炎是指终末气道、肺泡和肺间质的炎症，可由细菌、病毒、真菌等引起。细菌性肺炎最常见，治疗以抗生素为主。",
    ]
    for split in ["train", "valid", "test"]:
        with open(os.path.join(pretrain_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for text in medical_texts:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    # SFT: medical Q&A conversations
    sft_dir = os.path.join(base_dir, "sft")
    os.makedirs(sft_dir, exist_ok=True)
    medical_qa = [
        ("头痛应该怎么办？", "头痛的原因较多，建议首先排除器质性病变。如果是紧张性头痛，可以通过休息、按摩、适当运动来缓解。如果头痛持续不缓解或伴有其他症状，建议及时就医。"),
        ("什么是高血压？", "高血压是指动脉血压持续升高的慢性疾病。成人收缩压≥140mmHg和/或舒张压≥90mmHg即为高血压。需要通过药物和生活方式调整进行治疗。"),
        ("感冒了怎么办？", "感冒多为病毒感染，以对症治疗为主。注意休息、多饮水，可服用解热镇痛药缓解症状。如发热超过3天或症状加重，建议就医。"),
        ("糖尿病有哪些症状？", "糖尿病的典型症状为多饮、多食、多尿、体重减轻，即'三多一少'。还可出现视力模糊、皮肤瘙痒、伤口不易愈合等。"),
        ("孩子发烧怎么处理？", "儿童发热时，先物理降温（温水擦浴），体温超过38.5°C可口服退热药（如对乙酰氨基酚）。注意补充水分，密切观察精神状态，持续高热应就医。"),
    ]
    for split in ["train", "valid", "test"]:
        with open(os.path.join(sft_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for q, a in medical_qa:
                record = {"conversations": [{"from": "human", "value": q}, {"from": "gpt", "value": a}]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Reward: preference pairs
    reward_dir = os.path.join(base_dir, "reward_converted")
    os.makedirs(reward_dir, exist_ok=True)
    reward_data = [
        ("头痛应该怎么办？",
         "头痛的原因较多，建议首先排除器质性病变。如果是紧张性头痛，可以通过休息、按摩来缓解。",
         "吃点药就好了。"),
        ("什么是高血压？",
         "高血压是指动脉血压持续升高的慢性疾病，需要通过药物和生活方式调整进行治疗。",
         "血压高一点没关系。"),
        ("感冒了怎么办？",
         "感冒多为病毒感染，注意休息、多饮水，可服用解热镇痛药缓解症状。",
         "不用管它。"),
        ("糖尿病有哪些症状？",
         "糖尿病的典型症状为多饮、多食、多尿、体重减轻，即三多一少。",
         "没什么症状。"),
        ("孩子发烧怎么处理？",
         "儿童发热时，先物理降温，体温超过38.5°C可口服退热药，持续高热应就医。",
         "小孩发烧不用管。"),
    ]
    for split in ["train", "valid", "test"]:
        with open(os.path.join(reward_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for q, chosen, rejected in reward_data:
                record = {
                    "system": "",
                    "history": [],
                    "question": q,
                    "response_chosen": chosen,
                    "response_rejected": rejected,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Minimal medical test data created in {base_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare shibing624/medical dataset")
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "download", "convert", "minimal"],
        help="download: download raw data; convert: convert formats; all: both; minimal: test data only",
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--sft_max_samples", type=int, default=-1,
                        help="Max SFT training samples (-1 for all)")
    parser.add_argument("--pretrain_max_samples", type=int, default=-1,
                        help="Max pretrain training samples (-1 for all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "minimal":
        prepare_minimal_test_data(args.data_dir)
    elif args.mode in ["all", "download"]:
        download_dataset(args.data_dir)
        if args.mode == "all":
            convert_sft_data(args.data_dir, max_samples=args.sft_max_samples, seed=args.seed)
            convert_reward_data(args.data_dir)
            prepare_pretrain_data(args.data_dir, max_samples=args.pretrain_max_samples, seed=args.seed)
    elif args.mode == "convert":
        convert_sft_data(args.data_dir, max_samples=args.sft_max_samples, seed=args.seed)
        convert_reward_data(args.data_dir)
        prepare_pretrain_data(args.data_dir, max_samples=args.pretrain_max_samples, seed=args.seed)

    logger.info("Data preparation complete!")
