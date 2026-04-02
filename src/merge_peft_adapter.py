# -*- coding: utf-8 -*-
"""
Merge LoRA adapter weights into the base model.
"""

import argparse
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', required=True, type=str)
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--lora_model', required=True, type=str)
    parser.add_argument('--resize_emb', action='store_true')
    parser.add_argument('--output_dir', default='./merged', type=str)
    parser.add_argument('--hf_hub_model_id', default='', type=str)
    parser.add_argument('--hf_hub_token', default=None, type=str)
    args = parser.parse_args()

    peft_config = PeftConfig.from_pretrained(args.lora_model)
    if peft_config.task_type == "SEQ_CLS":
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model, num_labels=1, load_in_8bit=False,
            torch_dtype=torch.float32, trust_remote_code=True, device_map="auto")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype='auto', trust_remote_code=True, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.base_model, trust_remote_code=True)
    if args.resize_emb and base_model.get_input_embeddings().weight.size(0) != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))

    new_model = PeftModel.from_pretrained(base_model, args.lora_model,
                                          device_map="auto", torch_dtype='auto')
    new_model.eval()
    base_model = new_model.merge_and_unload()

    tokenizer.save_pretrained(args.output_dir)
    base_model.save_pretrained(args.output_dir, max_shard_size='10GB')
    print(f"Merged model saved to {args.output_dir}")

    if args.hf_hub_model_id:
        base_model.push_to_hub(args.hf_hub_model_id, token=args.hf_hub_token, max_shard_size="10GB")
        tokenizer.push_to_hub(args.hf_hub_model_id, token=args.hf_hub_token)
        print(f"Model pushed to HF Hub: {args.hf_hub_model_id}")


if __name__ == '__main__':
    main()
