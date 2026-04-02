# -*- coding: utf-8 -*-
"""
Download and prepare datasets for the Medical RLHF pipeline.

SFT: lavita/ChatDoctor-HealthCareMagic-100k (medical patient-doctor conversations)
Reward/PPO: Anthropic/hh-rlhf (human preference data for safety alignment)
Eval: Hold-out from ChatDoctor for perplexity computation

All data is converted to the formats expected by the training scripts:
- SFT: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
- Reward: {"system": "", "history": [], "question": "...", "response_chosen": "...", "response_rejected": "..."}
"""

import json
import os
import argparse
from pathlib import Path
from loguru import logger


def prepare_sft_data(output_dir: str, max_samples: int = 10000, seed: int = 42):
    """
    Download and convert ChatDoctor-HealthCareMagic medical dialogues to ShareGPT format.
    """
    from datasets import load_dataset

    logger.info(
        f"Loading lavita/ChatDoctor-HealthCareMagic-100k (max_samples={max_samples})..."
    )
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "medical_sft.jsonl")

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in ds:
            # ChatDoctor format: instruction (patient question) + input (context) + output (doctor response)
            instruction = (example.get("instruction") or "").strip()
            input_text = (example.get("input") or "").strip()
            output_text = (example.get("output") or "").strip()

            if not instruction or not output_text:
                continue

            # Combine instruction and input as the patient query
            patient_query = instruction
            if input_text:
                patient_query = f"{instruction}\n{input_text}"

            conversations = [
                {"from": "human", "value": patient_query},
                {"from": "gpt", "value": output_text},
            ]

            f.write(
                json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n"
            )
            count += 1

    logger.info(f"Medical SFT data saved: {output_path} ({count} samples)")
    return output_path


def prepare_reward_data(output_dir: str, max_samples: int = 5000, seed: int = 42):
    """
    Download and convert Anthropic/hh-rlhf to reward model format.
    """
    from datasets import load_dataset

    logger.info(f"Loading Anthropic/hh-rlhf (max_samples={max_samples})...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hh_rlhf_reward.jsonl")

    def parse_hh_rlhf(text: str):
        """Parse Anthropic hh-rlhf format: '\n\nHuman: ...\n\nAssistant: ...'"""
        parts = text.strip().split("\n\nAssistant:")
        if len(parts) < 2:
            return None, None
        # Get the question (everything before last Assistant response)
        question_parts = parts[0].split("\n\nHuman:")
        if len(question_parts) < 2:
            return None, None

        question = question_parts[-1].strip()
        answer = parts[-1].strip()

        # Build history from earlier turns
        history = []
        for i in range(1, len(question_parts) - 1):
            q = question_parts[i].strip()
            # Find corresponding assistant response
            if i - 1 < len(parts) - 1:
                a = (
                    parts[i].split("\n\nHuman:")[0].strip()
                    if "\n\nHuman:" in parts[i]
                    else parts[i].strip()
                )
                if q and a:
                    history.append([q, a])

        return question, answer, history

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in ds:
            chosen_text = example.get("chosen", "")
            rejected_text = example.get("rejected", "")

            result_chosen = parse_hh_rlhf(chosen_text)
            result_rejected = parse_hh_rlhf(rejected_text)

            if result_chosen is None or result_rejected is None:
                continue
            if len(result_chosen) != 3 or len(result_rejected) != 3:
                continue

            question_c, answer_c, history_c = result_chosen
            question_r, answer_r, history_r = result_rejected

            if not question_c or not answer_c or not answer_r:
                continue

            record = {
                "system": "",
                "history": history_c[:3],  # Keep at most 3 turns of history
                "question": question_c,
                "response_chosen": answer_c,
                "response_rejected": answer_r,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Reward data saved: {output_path} ({count} samples)")
    return output_path


def prepare_eval_data(output_dir: str, max_samples: int = 200, seed: int = 42):
    """
    Download eval data from ChatDoctor (hold-out portion, different seed for separation).
    """
    from datasets import load_dataset

    logger.info(f"Loading medical eval data (max_samples={max_samples})...")
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    # Use a different seed offset to get different samples from the SFT set
    ds = ds.shuffle(seed=seed + 1000)
    # Skip first 10000 (reserved for SFT) then take eval samples
    start_idx = min(10000, len(ds) - max_samples)
    ds = ds.select(range(start_idx, min(start_idx + max_samples, len(ds))))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "medical_eval.jsonl")

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in ds:
            instruction = (example.get("instruction") or "").strip()
            input_text = (example.get("input") or "").strip()
            output_text = (example.get("output") or "").strip()

            if not instruction or not output_text:
                continue

            patient_query = instruction
            if input_text:
                patient_query = f"{instruction}\n{input_text}"

            conversations = [
                {"from": "human", "value": patient_query},
                {"from": "gpt", "value": output_text},
            ]

            f.write(
                json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n"
            )
            count += 1

    logger.info(f"Medical eval data saved: {output_path} ({count} samples)")
    return output_path


def prepare_minimal_test_data(base_dir: str):
    """
    Create tiny medical datasets for CPU debugging (5 samples each).
    """
    medical_qa = [
        ("What are the common symptoms of a cold?", "Common symptoms include runny nose, sore throat, cough, sneezing, mild body aches, and low-grade fever."),
        ("How should I treat a minor burn?", "For minor burns, run cool water over the area for 10-15 minutes, apply aloe vera or burn cream, and cover with a sterile bandage."),
        ("What causes high blood pressure?", "High blood pressure can be caused by obesity, high salt intake, lack of exercise, stress, genetics, and chronic kidney disease."),
        ("When should I see a doctor for a headache?", "See a doctor if headaches are severe, frequent, accompanied by vision changes, or not relieved by over-the-counter medication."),
        ("What foods help lower cholesterol?", "Foods that help lower cholesterol include oats, nuts, fatty fish, olive oil, beans, and fruits rich in soluble fiber."),
    ]

    # SFT
    sft_dir = os.path.join(base_dir, "finetune")
    os.makedirs(sft_dir, exist_ok=True)
    sft_path = os.path.join(sft_dir, "test_sft.jsonl")
    with open(sft_path, "w", encoding="utf-8") as f:
        for q, a in medical_qa:
            record = {
                "conversations": [
                    {"from": "human", "value": q},
                    {"from": "gpt", "value": a},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Reward
    reward_dir = os.path.join(base_dir, "reward")
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "test_reward.jsonl")
    with open(reward_path, "w", encoding="utf-8") as f:
        for q, a in medical_qa:
            record = {
                "system": "",
                "history": [],
                "question": q,
                "response_chosen": a,
                "response_rejected": "I'm not sure, you should ask someone else.",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Eval
    eval_dir = os.path.join(base_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    eval_path = os.path.join(eval_dir, "test_eval.jsonl")
    with open(eval_path, "w", encoding="utf-8") as f:
        for q, a in medical_qa:
            record = {
                "conversations": [
                    {"from": "human", "value": q},
                    {"from": "gpt", "value": a},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Minimal medical test data created in {base_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for RLHF pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "sft", "reward", "eval", "minimal"],
        help="Which data to prepare",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Base data directory"
    )
    parser.add_argument(
        "--sft_samples", type=int, default=10000, help="Number of SFT samples"
    )
    parser.add_argument(
        "--reward_samples", type=int, default=5000, help="Number of reward samples"
    )
    parser.add_argument(
        "--eval_samples", type=int, default=200, help="Number of eval samples"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode in ["all", "minimal"]:
        prepare_minimal_test_data(args.data_dir)

    if args.mode in ["all", "sft"]:
        prepare_sft_data(
            os.path.join(args.data_dir, "finetune"),
            max_samples=args.sft_samples,
            seed=args.seed,
        )

    if args.mode in ["all", "reward"]:
        prepare_reward_data(
            os.path.join(args.data_dir, "reward"),
            max_samples=args.reward_samples,
            seed=args.seed,
        )

    if args.mode in ["all", "eval"]:
        prepare_eval_data(
            os.path.join(args.data_dir, "eval"),
            max_samples=args.eval_samples,
            seed=args.seed,
        )

    logger.info("Data preparation complete!")
