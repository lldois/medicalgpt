# -*- coding: utf-8 -*-
"""
Comprehensive evaluation pipeline for each stage of the RLHF pipeline.

Supported stages and their metrics:
  - pretrain: Perplexity (PPL) on medical encyclopedia test set
  - sft:      PPL + BLEU-4 + ROUGE-L on SFT test set + sample generation
  - rm:       Accuracy + Mean Reward Gap on reward test set
  - rlhf:     Mean Reward + BLEU/ROUGE + sample generation with reward scores
"""

import argparse
import json
import math
import os
import sys
from collections import Counter

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template import get_conv_template


# ============================================================
# Metric Functions
# ============================================================

def compute_ppl(model, tokenizer, texts, max_length=512, device="cpu"):
    """Compute perplexity on raw texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            if inputs["input_ids"].shape[1] < 2:
                continue
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def compute_bleu4(predictions, references):
    """
    Character-level BLEU-4 for Chinese text.
    Each prediction/reference is a string.
    """
    def ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    scores = []
    for pred, ref in zip(predictions, references):
        pred_chars = list(pred.strip())
        ref_chars = list(ref.strip())
        if len(pred_chars) == 0:
            scores.append(0.0)
            continue

        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_chars) / max(len(pred_chars), 1)))

        # n-gram precisions
        log_precision = 0.0
        valid = True
        for n in range(1, 5):
            pred_ng = Counter(ngrams(pred_chars, n))
            ref_ng = Counter(ngrams(ref_chars, n))
            clipped = sum(min(count, ref_ng[ng]) for ng, count in pred_ng.items())
            total = sum(pred_ng.values())
            if total == 0:
                valid = False
                break
            log_precision += math.log(max(clipped, 1e-10) / total) / 4

        if valid:
            scores.append(bp * math.exp(log_precision))
        else:
            scores.append(0.0)

    return sum(scores) / max(len(scores), 1)


def compute_rouge_l(predictions, references):
    """
    ROUGE-L (F1) based on Longest Common Subsequence for Chinese text.
    """
    def lcs_length(x, y):
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    scores = []
    for pred, ref in zip(predictions, references):
        pred_chars = list(pred.strip())
        ref_chars = list(ref.strip())
        if len(pred_chars) == 0 or len(ref_chars) == 0:
            scores.append(0.0)
            continue
        lcs_len = lcs_length(pred_chars, ref_chars)
        precision = lcs_len / len(pred_chars)
        recall = lcs_len / len(ref_chars)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))

    return sum(scores) / max(len(scores), 1)


def generate_response(model, tokenizer, prompt_text, template_name="qwen",
                      max_new_tokens=256, device="cpu"):
    """Generate a single response using the chat template."""
    prompt_template = get_conv_template(template_name)
    formatted = prompt_template.get_prompt(messages=[[prompt_text, ""]])
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response


# ============================================================
# Stage Evaluators
# ============================================================

def evaluate_pretrain(model, tokenizer, test_data_path, max_length=512, device="cpu"):
    """
    Evaluate pretrain model: Perplexity on medical text.
    test_data_path: JSONL file with {"text": "..."} per line.
    """
    texts = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                texts.append(item.get("text", ""))

    texts = [t for t in texts if t]
    logger.info(f"Evaluating pretrain on {len(texts)} texts...")
    ppl = compute_ppl(model, tokenizer, texts, max_length=max_length, device=device)
    logger.info(f"Pretrain PPL: {ppl:.4f}")
    return {"perplexity": ppl, "num_samples": len(texts)}


def evaluate_sft(model, tokenizer, test_data_path, template_name="qwen",
                 max_length=512, max_new_tokens=256, max_eval_samples=100, device="cpu"):
    """
    Evaluate SFT model: PPL + BLEU-4 + ROUGE-L + sample generation.
    test_data_path: JSONL with {"conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}
    """
    test_data = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))

    if max_eval_samples > 0:
        test_data = test_data[:max_eval_samples]

    # 1. PPL on full text
    logger.info(f"Computing PPL on {len(test_data)} samples...")
    texts = []
    for item in test_data:
        convs = item.get("conversations", [])
        text = " ".join(msg.get("value", "") for msg in convs)
        if text.strip():
            texts.append(text)
    ppl = compute_ppl(model, tokenizer, texts, max_length=max_length, device=device)
    logger.info(f"SFT PPL: {ppl:.4f}")

    # 2. BLEU-4 + ROUGE-L: generate answers and compare with references
    questions = []
    references = []
    for item in test_data:
        convs = item.get("conversations", [])
        if len(convs) >= 2 and convs[0]["from"] == "human" and convs[1]["from"] == "gpt":
            questions.append(convs[0]["value"])
            references.append(convs[1]["value"])

    # Limit generation samples for speed
    gen_limit = min(len(questions), 50)
    logger.info(f"Generating {gen_limit} responses for BLEU/ROUGE...")
    predictions = []
    for i in range(gen_limit):
        resp = generate_response(
            model, tokenizer, questions[i],
            template_name=template_name, max_new_tokens=max_new_tokens, device=device
        )
        predictions.append(resp)
        if i < 3:
            logger.info(f"\n[Q] {questions[i]}\n[Ref] {references[i]}\n[Gen] {resp}\n")

    bleu = compute_bleu4(predictions, references[:gen_limit])
    rouge = compute_rouge_l(predictions, references[:gen_limit])
    logger.info(f"SFT BLEU-4: {bleu:.4f}, ROUGE-L: {rouge:.4f}")

    return {
        "perplexity": ppl,
        "bleu4": bleu,
        "rouge_l": rouge,
        "num_samples": len(test_data),
        "num_generated": gen_limit,
    }


def evaluate_rm(model, tokenizer, test_data_path, template_name="qwen",
                max_length=512, device="cpu"):
    """
    Evaluate Reward Model: Accuracy (chosen score > rejected score) + Mean Reward Gap.
    test_data_path: JSONL with {"question": "...", "response_chosen": "...", "response_rejected": "..."}
    """
    test_data = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))

    prompt_template = get_conv_template(template_name)
    correct = 0
    total = 0
    reward_gaps = []

    logger.info(f"Evaluating RM on {len(test_data)} preference pairs...")
    model.eval()
    with torch.no_grad():
        for item in test_data:
            question = item.get("question", "")
            chosen = item.get("response_chosen", "")
            rejected = item.get("response_rejected", "")
            if not question or not chosen or not rejected:
                continue

            # Format with template
            chosen_prompt = prompt_template.get_prompt(messages=[[question, chosen]])
            rejected_prompt = prompt_template.get_prompt(messages=[[question, rejected]])

            inputs_chosen = tokenizer(
                chosen_prompt, return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)
            inputs_rejected = tokenizer(
                rejected_prompt, return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)

            r_chosen = model(**inputs_chosen).logits.squeeze().item()
            r_rejected = model(**inputs_rejected).logits.squeeze().item()

            if r_chosen > r_rejected:
                correct += 1
            reward_gaps.append(r_chosen - r_rejected)
            total += 1

    accuracy = correct / max(total, 1)
    mean_gap = sum(reward_gaps) / max(len(reward_gaps), 1)
    logger.info(f"RM Accuracy: {accuracy:.4f} ({correct}/{total})")
    logger.info(f"RM Mean Reward Gap: {mean_gap:.4f}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "mean_reward_gap": mean_gap,
    }


def evaluate_rlhf(model, tokenizer, reward_model, reward_tokenizer,
                   test_data_path, template_name="qwen",
                   max_length=512, max_new_tokens=256, max_eval_samples=50, device="cpu"):
    """
    Evaluate RLHF (PPO) model: Mean Reward + BLEU/ROUGE + reward-scored samples.
    """
    test_data = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))

    if max_eval_samples > 0:
        test_data = test_data[:max_eval_samples]

    questions = []
    references = []
    for item in test_data:
        convs = item.get("conversations", [])
        if len(convs) >= 2:
            questions.append(convs[0]["value"])
            references.append(convs[1]["value"])

    prompt_template = get_conv_template(template_name)

    # Generate responses and compute rewards
    logger.info(f"Generating {len(questions)} RLHF responses and scoring...")
    predictions = []
    rewards = []
    model.eval()
    reward_model.eval()

    with torch.no_grad():
        for i, q in enumerate(questions):
            resp = generate_response(
                model, tokenizer, q,
                template_name=template_name, max_new_tokens=max_new_tokens, device=device
            )
            predictions.append(resp)

            # Score with reward model
            scored_prompt = prompt_template.get_prompt(messages=[[q, resp]])
            inputs = reward_tokenizer(
                scored_prompt, return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)
            reward_score = reward_model(**inputs).logits.squeeze().item()
            rewards.append(reward_score)

            if i < 3:
                logger.info(f"\n[Q] {q}\n[Gen] {resp}\n[Reward] {reward_score:.4f}\n")

    mean_reward = sum(rewards) / max(len(rewards), 1)
    bleu = compute_bleu4(predictions, references)
    rouge = compute_rouge_l(predictions, references)

    logger.info(f"RLHF Mean Reward: {mean_reward:.4f}")
    logger.info(f"RLHF BLEU-4: {bleu:.4f}, ROUGE-L: {rouge:.4f}")

    return {
        "mean_reward": mean_reward,
        "bleu4": bleu,
        "rouge_l": rouge,
        "num_samples": len(questions),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate RLHF pipeline models")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["pretrain", "sft", "rm", "rlhf"],
                        help="Which stage to evaluate")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (merged or base)")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Optional LoRA adapter path to merge before eval")
    parser.add_argument("--reward_model_path", type=str, default=None,
                        help="Path to reward model (required for --stage rlhf)")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="Path to evaluation data file")
    parser.add_argument("--template_name", type=str, default="qwen")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_eval_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    logger.info(f"Loading model from {args.model_path} (stage={args.stage})")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if args.stage == "rm":
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=1, torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()

    if device == "cpu":
        model = model.to("cpu")

    # --- Evaluate ---
    if args.stage == "pretrain":
        results = evaluate_pretrain(
            model, tokenizer, args.eval_data,
            max_length=args.max_length, device=device
        )

    elif args.stage == "sft":
        results = evaluate_sft(
            model, tokenizer, args.eval_data,
            template_name=args.template_name,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            max_eval_samples=args.max_eval_samples,
            device=device,
        )

    elif args.stage == "rm":
        results = evaluate_rm(
            model, tokenizer, args.eval_data,
            template_name=args.template_name,
            max_length=args.max_length,
            device=device,
        )

    elif args.stage == "rlhf":
        if not args.reward_model_path:
            raise ValueError("--reward_model_path required for RLHF evaluation")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model_path, num_labels=1, torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(
            args.reward_model_path, trust_remote_code=True
        )
        if device == "cpu":
            reward_model = reward_model.to("cpu")
        results = evaluate_rlhf(
            model, tokenizer, reward_model, reward_tokenizer,
            args.eval_data,
            template_name=args.template_name,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            max_eval_samples=args.max_eval_samples,
            device=device,
        )

    # --- Save ---
    results["stage"] = args.stage
    results["model_path"] = args.model_path
    output_path = os.path.join(args.output_dir, f"eval_{args.stage}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Final results: {json.dumps(results, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
