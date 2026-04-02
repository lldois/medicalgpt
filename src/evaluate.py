# -*- coding: utf-8 -*-
"""
Evaluation pipeline for the RLHF model.
Computes perplexity, generates sample outputs, and saves evaluation results.
"""

import argparse
import json
import os
import sys

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template import get_conv_template


def compute_perplexity(model, tokenizer, eval_data, max_length=512, device="cuda"):
    """Compute perplexity on evaluation data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for item in eval_data:
            conversations = item.get("conversations", [])
            if len(conversations) < 2:
                continue

            text = ""
            for msg in conversations:
                text += msg.get("value", "") + " "
            text = text.strip()

            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_length).to(device)
            if inputs["input_ids"].shape[1] < 2:
                continue

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]

    if total_tokens == 0:
        return float("inf")

    import math
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def generate_samples(model, tokenizer, prompts, template_name="qwen",
                     max_new_tokens=256, device="cuda"):
    """Generate sample responses for qualitative evaluation."""
    model.eval()
    prompt_template = get_conv_template(template_name)
    results = []

    for prompt_text in prompts:
        messages = [[prompt_text, ""]]
        formatted = prompt_template.get_prompt(messages=messages)

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

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
        results.append({"prompt": prompt_text, "response": response})

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLHF model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model (merged or base + LoRA)")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Optional LoRA adapter path")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Path to eval JSONL file")
    parser.add_argument("--template_name", type=str, default="qwen")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True, device_map="auto" if device == "cuda" else None)

    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()

    if device == "cpu":
        model = model.to("cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    # 1. Perplexity
    if args.eval_data and os.path.exists(args.eval_data):
        logger.info("Computing perplexity...")
        eval_data = []
        with open(args.eval_data, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_data.append(json.loads(line))

        ppl = compute_perplexity(model, tokenizer, eval_data,
                                 max_length=args.max_length, device=device)
        results["perplexity"] = ppl
        logger.info(f"Perplexity: {ppl:.4f}")

    # 2. Sample generation
    logger.info("Generating sample responses...")
    sample_prompts = [
        "What are the common symptoms of diabetes?",
        "Explain the difference between Type 1 and Type 2 diabetes.",
        "What is the recommended treatment for high blood pressure?",
        "How does the immune system fight infections?",
        "What are the side effects of aspirin?",
    ]
    samples = generate_samples(model, tokenizer, sample_prompts,
                               template_name=args.template_name,
                               max_new_tokens=args.max_new_tokens, device=device)
    results["generated_samples"] = samples

    for s in samples:
        logger.info(f"\n[Prompt] {s['prompt']}\n[Response] {s['response']}\n")

    # 3. Save results
    output_path = os.path.join(args.output_dir, "eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    main()
