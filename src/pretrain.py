# -*- coding: utf-8 -*-
"""
Stage 0: Continuous Pretraining (PT)
Train the base model on medical domain text (encyclopedia, medical books)
using causal language modeling to inject domain knowledge.

Based on the SFT training structure, but simplified:
- No conversation template needed
- Loss computed on ALL tokens (no masking)
- Data format: {"text": "..."}
"""

import math
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import torch
from datasets import load_dataset
from loguru import logger
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from transformers.trainer import TRAINING_ARGS_NAME

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    build_run_name,
    get_hf_repo_id,
    resolve_resume_checkpoint,
    upload_checkpoint_to_hub,
    setup_wandb,
    save_hyperparams,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=False)
    torch_dtype: Optional[str] = field(default="float16")
    device_map: Optional[str] = field(default="auto")
    trust_remote_code: bool = field(default=True)

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required")


@dataclass
class DataArguments:
    train_file_dir: Optional[str] = field(default=None)
    validation_file_dir: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=1)
    preprocessing_num_workers: Optional[int] = field(default=None)
    block_size: int = field(default=1024, metadata={"help": "Max sequence length for CLM"})


@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True)
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    qlora: bool = field(default=False)
    upload_steps: Optional[int] = field(default=None)


class SavePeftModelTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def find_all_linear_names(peft_model, int4=False, int8=False):
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            if "lm_head" in name or "output_layer" in name:
                continue
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ScriptArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, script_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, script_args = (
            parser.parse_args_into_dataclasses(look_for_args_file=False)
        )

    is_main_process = training_args.local_rank in [-1, 0]

    run_name = build_run_name(
        "pt",
        model=model_args.model_name_or_path,
        lr=training_args.learning_rate,
        lora_rank=script_args.lora_rank,
        lora_alpha=script_args.lora_alpha,
        batch_size=training_args.per_device_train_batch_size,
        epochs=training_args.num_train_epochs,
        max_length=data_args.block_size,
    )
    repo_id = get_hf_repo_id(run_name)

    if is_main_process:
        logger.info(f"Run name: {run_name}")
        logger.info(f"HF Hub repo: {repo_id}")
        setup_wandb("medicalgpt", run_name)

    if not training_args.run_name:
        training_args.run_name = run_name

    if training_args.resume_from_checkpoint is None:
        resolved = resolve_resume_checkpoint(training_args.output_dir, repo_id)
        if resolved:
            training_args.resume_from_checkpoint = resolved

    set_seed(training_args.seed)

    # Load tokenizer
    tokenizer_name_or_path = (
        model_args.tokenizer_name_or_path or model_args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token

    block_size = data_args.block_size

    # Load datasets
    data_files = {}
    if data_args.train_file_dir and os.path.exists(data_args.train_file_dir):
        train_files = glob(f"{data_args.train_file_dir}/**/*.json", recursive=True) + \
                      glob(f"{data_args.train_file_dir}/**/*.jsonl", recursive=True)
        data_files["train"] = train_files
    if data_args.validation_file_dir and os.path.exists(data_args.validation_file_dir):
        eval_files = glob(f"{data_args.validation_file_dir}/**/*.json", recursive=True) + \
                     glob(f"{data_args.validation_file_dir}/**/*.jsonl", recursive=True)
        data_files["validation"] = eval_files
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)

    if "validation" not in raw_datasets.keys():
        shuffled = raw_datasets["train"].shuffle(seed=42)
        split = shuffled.train_test_split(
            test_size=data_args.validation_split_percentage / 100, seed=42
        )
        raw_datasets["train"] = split["train"]
        raw_datasets["validation"] = split["test"]

    logger.info(f"Raw datasets: {raw_datasets}")

    # Tokenize: concatenate texts and split into fixed-length blocks
    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        # If total tokens < block_size, use total_length as one block (pad-free)
        if total_length < block_size:
            if total_length == 0:
                return {k: [] for k in concatenated.keys()}
            result = {k: [v[:total_length]] for k, v in concatenated.items()}
            result["labels"] = result["input_ids"].copy()
            return result
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Prepare train dataset
    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        train_dataset = raw_datasets["train"].shuffle(seed=42)
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="Tokenize train"):
            train_dataset = train_dataset.map(
                tokenize_function, batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
            train_dataset = train_dataset.map(
                group_texts, batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )
        if is_main_process:
            logger.info(f"Train blocks: {len(train_dataset)} (block_size={block_size})")

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if data_args.max_eval_samples and data_args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="Tokenize eval"):
            eval_dataset = eval_dataset.map(
                tokenize_function, batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
            eval_dataset = eval_dataset.map(
                group_texts, batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )
        if is_main_process:
            logger.info(f"Eval blocks: {len(eval_dataset)}")

    # Load model
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    if ddp:
        model_args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
    )

    quantization_config = None
    if model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif model_args.load_in_4bit:
        if script_args.qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype,
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map=model_args.device_map,
    )

    # LoRA
    if script_args.use_peft:
        logger.info("Using LoRA (PEFT)")
        output_layer = getattr(model, "lm_head", None)
        if output_layer and isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            def fp32_forward_post_hook(module, args, output):
                return output.to(torch.float32)
            output_layer.register_forward_hook(fp32_forward_post_hook)

        if script_args.peft_path is not None:
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
        else:
            if model_args.load_in_8bit or model_args.load_in_4bit:
                model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)
            target_modules = (
                script_args.target_modules.split(",") if script_args.target_modules else None
            )
            if target_modules and "all" in target_modules:
                target_modules = find_all_linear_names(
                    model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit
                )
            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(",")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save,
            )
            model = get_peft_model(model, peft_config)
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()

    # Trainer setup
    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    model.enable_input_require_grads()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    trainer = SavePeftModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        if is_main_process:
            logger.info("*** Pretrain ***")
            save_hyperparams(
                training_args.output_dir,
                stage="pretrain",
                model=model_args.model_name_or_path,
                lr=training_args.learning_rate,
                lora_rank=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                batch_size=training_args.per_device_train_batch_size,
                epochs=training_args.num_train_epochs,
                block_size=data_args.block_size,
                run_name=run_name,
            )

        checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        model.config.use_cache = True

        if trainer.is_world_process_zero():
            logger.info(f"Saving model to {training_args.output_dir}")
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)

            upload_checkpoint_to_hub(
                training_args.output_dir,
                repo_id,
                commit_message=f"Final pretrain model: {run_name}",
            )

    # Evaluation
    if training_args.do_eval:
        if is_main_process:
            logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
