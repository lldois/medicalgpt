# -*- coding: utf-8 -*-
"""
Stage 3: PPO Training (RLHF)
Use PPO to align the SFT model using the reward model.

Features:
  - wandb logging
  - HuggingFace Hub checkpoint upload
  - Hyperparameter-based run naming
  - Automatic resume from local/Hub checkpoints
"""

import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

from datasets import load_dataset
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForCausalLM,
)
from trl import (
    PPOConfig,
    PPOTrainer,
    ModelConfig,
    get_peft_config,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template import get_conv_template
from utils import (
    build_run_name,
    get_hf_repo_id,
    resolve_resume_checkpoint,
    upload_checkpoint_to_hub,
    setup_wandb,
    save_hyperparams,
)

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class PPOArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config: Optional[str] = field(default=None)
    dataset_train_split: str = field(default="train")
    dataset_test_split: str = field(default="test")
    train_file_dir: Optional[str] = field(default=None)
    validation_file_dir: Optional[str] = field(default=None)
    template_name: Optional[str] = field(default="qwen")
    max_source_length: Optional[int] = field(default=1024)
    upload_steps: Optional[int] = field(default=None)


def main():
    parser = HfArgumentParser((PPOArguments, PPOConfig, ModelConfig))
    args, training_args, model_args = parser.parse_args_into_dataclasses()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = local_rank == 0

    run_name = build_run_name(
        "ppo",
        model=training_args.sft_model_path,
        batch_size=training_args.per_device_train_batch_size,
    )
    repo_id = get_hf_repo_id(run_name)

    if is_main_process:
        logger.info(f"Run name: {run_name}")
        setup_wandb("medicalgpt", run_name)

    # Auto-resume checkpoint
    if training_args.resume_from_checkpoint is None:
        resolved = resolve_resume_checkpoint(training_args.output_dir, repo_id)
        if resolved:
            training_args.resume_from_checkpoint = resolved

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = (
            tokenizer.eos_token
            if tokenizer.eos_token is not None
            else tokenizer.sep_token
        )
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token

    # Load models
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    # Load datasets
    prompt_template = get_conv_template(args.template_name)
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name, args.dataset_config, split=args.dataset_train_split
        )
        eval_samples = 100
        train_dataset = dataset.select(range(len(dataset) - eval_samples))
        eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    else:
        data_files = {}
        if args.train_file_dir and os.path.exists(args.train_file_dir):
            train_data_files = glob(
                f"{args.train_file_dir}/**/*.json", recursive=True
            ) + glob(f"{args.train_file_dir}/**/*.jsonl", recursive=True)
            data_files["train"] = train_data_files
        if args.validation_file_dir and os.path.exists(args.validation_file_dir):
            eval_data_files = glob(
                f"{args.validation_file_dir}/**/*.json", recursive=True
            ) + glob(f"{args.validation_file_dir}/**/*.jsonl", recursive=True)
            data_files["validation"] = eval_data_files
        dataset = load_dataset("json", data_files=data_files)
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        eval_dataset = val_dataset.select(range(min(100, len(val_dataset))))

    if is_main_process:
        logger.info(f"Datasets: train={len(train_dataset)}, eval={len(eval_dataset)}")

    max_source_length = args.max_source_length

    def preprocess_function(examples):
        new_examples = {"input_ids": []}
        roles = ["human", "gpt"]

        def get_dialog(examples):
            system_prompts = examples.get("system_prompt", "")
            for i, source in enumerate(examples["conversations"]):
                if len(source) < 2:
                    continue
                data_role = source[0].get("from", "")
                if data_role not in roles or data_role != roles[0]:
                    source = source[1:]
                if len(source) < 2:
                    continue
                messages = []
                for j, sentence in enumerate(source):
                    data_role = sentence.get("from", "")
                    if data_role not in roles:
                        break
                    if data_role == roles[j % 2]:
                        messages.append(sentence["value"])
                if len(messages) < 2 or len(messages) % 2 != 0:
                    continue
                history_messages = [
                    [messages[k], messages[k + 1]] for k in range(0, len(messages), 2)
                ]
                system_prompt = system_prompts[i] if system_prompts else None
                yield prompt_template.get_dialog(
                    history_messages, system_prompt=system_prompt
                )

        for dialog in get_dialog(examples):
            for i in range(len(dialog) // 2):
                source_txt = dialog[2 * i]
                tokenized_question = tokenizer(source_txt, padding=False)
                new_examples["input_ids"].append(tokenized_question["input_ids"])
        return new_examples

    if is_main_process:
        tokenized_train = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=training_args.dataset_num_proc,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=False,
        )
        train_dataset = tokenized_train.filter(lambda x: len(x["input_ids"]) > 0)

        tokenized_eval = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=training_args.dataset_num_proc,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=False,
        )
        eval_dataset = tokenized_eval.filter(lambda x: len(x["input_ids"]) > 0)

        logger.info(f"Tokenized: train={len(train_dataset)}, eval={len(eval_dataset)}")

    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    if training_args.do_train:
        if is_main_process:
            logger.info("*** PPO Train ***")
            save_hyperparams(
                training_args.output_dir,
                stage="ppo",
                sft_model=training_args.sft_model_path,
                reward_model=training_args.reward_model_path,
                batch_size=training_args.per_device_train_batch_size,
                run_name=run_name,
            )
        trainer.train()

        if is_main_process:
            trainer.save_model(training_args.output_dir)
            upload_checkpoint_to_hub(
                training_args.output_dir,
                repo_id,
                commit_message=f"Final PPO model: {run_name}",
            )

    trainer.generate_completions()


if __name__ == "__main__":
    main()
