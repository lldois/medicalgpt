# -*- coding: utf-8 -*-
"""
Stage 2: Reward Model Training
Train a reward model using pairwise preference data (InstructGPT pairwise logloss).

Features:
  - wandb logging
  - HuggingFace Hub checkpoint upload
  - Hyperparameter-based run naming
  - Automatic resume from local/Hub checkpoints
"""

import math
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Any, List, Union, Optional, Dict

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import (
    AutoConfig,
    PreTrainedTokenizerBase,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer import TRAINING_ARGS_NAME

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template import get_conv_template
from utils import (
    build_run_name, get_hf_repo_id, resolve_resume_checkpoint,
    upload_checkpoint_to_hub, setup_wandb, save_hyperparams,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    load_in_4bit: bool = field(default=False)
    load_in_8bit: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=False)
    torch_dtype: Optional[str] = field(default=None,
        metadata={"choices": ["auto", "bfloat16", "float16", "float32"]})
    device_map: Optional[str] = field(default="auto")
    trust_remote_code: bool = field(default=True)

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required")


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file_dir: Optional[str] = field(default=None)
    validation_file_dir: Optional[str] = field(default=None)
    max_source_length: Optional[int] = field(default=2048)
    max_target_length: Optional[int] = field(default=512)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=1)
    preprocessing_num_workers: Optional[int] = field(default=4)


@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True)
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    template_name: Optional[str] = field(default="qwen")
    upload_steps: Optional[int] = field(default=None)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    return {"mse": mse, "mae": mae}


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        for feature in features:
            features_chosen.append({
                "input_ids": feature["input_ids_chosen"],
                "attention_mask": feature["attention_mask_chosen"],
            })
            features_rejected.append({
                "input_ids": feature["input_ids_rejected"],
                "attention_mask": feature["attention_mask_rejected"],
            })
        batch_chosen = self.tokenizer.pad(features_chosen, padding=self.padding,
            max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)
        batch_rejected = self.tokenizer.pad(features_rejected, padding=self.padding,
            max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)
        return {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rewards_chosen = model(input_ids=inputs["input_ids_chosen"],
                               attention_mask=inputs["attention_mask_chosen"])[0]
        rewards_rejected = model(input_ids=inputs["input_ids_rejected"],
                                 attention_mask=inputs["attention_mask_rejected"])[0]
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                metric_key_prefix=metric_key_prefix)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        device = model.device
        inputs_chosen = {
            "input_ids": inputs["input_ids_chosen"].to(device),
            "attention_mask": inputs["attention_mask_chosen"].to(device),
        }
        outputs_chosen = model(**inputs_chosen)
        rewards_chosen = outputs_chosen.logits.detach()
        inputs_rejected = {
            "input_ids": inputs["input_ids_rejected"].to(device),
            "attention_mask": inputs["attention_mask_rejected"].to(device),
        }
        outputs_rejected = model(**inputs_rejected)
        rewards_rejected = outputs_rejected.logits.detach()
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, rewards_chosen, rewards_rejected)

    def save_model(self, output_dir=None, _internal_call=False):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def save_model(model, tokenizer, args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


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
            if 'lm_head' in name or 'score' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ScriptArguments))
    model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()

    is_main_process = training_args.local_rank in [-1, 0]

    run_name = build_run_name(
        "rm", model=model_args.model_name_or_path,
        lr=training_args.learning_rate, lora_rank=script_args.lora_rank,
        lora_alpha=script_args.lora_alpha,
        batch_size=training_args.per_device_train_batch_size,
        epochs=training_args.num_train_epochs,
    )
    repo_id = get_hf_repo_id(run_name)

    if is_main_process:
        logger.info(f"Run name: {run_name}")
        setup_wandb("medicalgpt", run_name)

    if not training_args.run_name:
        training_args.run_name = run_name

    if training_args.resume_from_checkpoint is None:
        resolved = resolve_resume_checkpoint(training_args.output_dir, repo_id)
        if resolved:
            training_args.resume_from_checkpoint = resolved

    set_seed(training_args.seed)

    # Load model
    torch_dtype = (model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
                   else getattr(torch, model_args.torch_dtype))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        model_args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=1,
        torch_dtype=torch_dtype, trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir)

    from transformers import BitsAndBytesConfig
    quantization_config = None
    if model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, config=config, torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=model_args.device_map, trust_remote_code=model_args.trust_remote_code)

    # Load tokenizer
    tokenizer_name_or_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path,
        cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer,
        trust_remote_code=model_args.trust_remote_code)
    prompt_template = get_conv_template(script_args.template_name)

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = prompt_template.stop_str
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token

    # LoRA
    if script_args.use_peft:
        logger.info("Using LoRA (PEFT)")
        if script_args.peft_path is not None:
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
        else:
            if model_args.load_in_8bit:
                model = prepare_model_for_kbit_training(model)
            target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model, int4=False, int8=model_args.load_in_8bit)
            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS, target_modules=target_modules,
                inference_mode=False, r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha, lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()

    # Load datasets
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name,
                                    cache_dir=model_args.cache_dir)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir)
            raw_datasets["train"] = load_dataset(data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        if data_args.train_file_dir and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(f'{data_args.train_file_dir}/**/*.json', recursive=True) + \
                               glob(f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
            data_files["train"] = train_data_files
        if data_args.validation_file_dir and os.path.exists(data_args.validation_file_dir):
            eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.json', recursive=True) + \
                              glob(f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset('json', data_files=data_files, cache_dir=model_args.cache_dir)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset('json', data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]", cache_dir=model_args.cache_dir)
            raw_datasets["train"] = load_dataset('json', data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]", cache_dir=model_args.cache_dir)

    logger.info(f"Raw datasets: {raw_datasets}")

    full_max_length = data_args.max_source_length + data_args.max_target_length

    def preprocess_reward_function(examples):
        new_examples = {
            "input_ids_chosen": [], "attention_mask_chosen": [],
            "input_ids_rejected": [], "attention_mask_rejected": [],
        }
        for system, history, question, chosen, rejected in zip(
                examples["system"], examples["history"], examples["question"],
                examples["response_chosen"], examples["response_rejected"]):
            system_prompt = system or ""
            chosen_messages = history + [[question, chosen]] if history else [[question, chosen]]
            chosen_prompt = prompt_template.get_prompt(messages=chosen_messages, system_prompt=system_prompt)
            rejected_messages = history + [[question, rejected]] if history else [[question, rejected]]
            rejected_prompt = prompt_template.get_prompt(messages=rejected_messages, system_prompt=system_prompt)
            tokenized_chosen = tokenizer(chosen_prompt)
            tokenized_rejected = tokenizer(rejected_prompt)
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        return new_examples

    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        train_dataset = raw_datasets['train']
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="Train dataset tokenization"):
            tokenized = train_dataset.shuffle().map(preprocess_reward_function, batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache)
            train_dataset = tokenized.filter(
                lambda x: 0 < len(x['input_ids_rejected']) <= full_max_length and
                          0 < len(x['input_ids_chosen']) <= full_max_length)
            logger.info(f"Train samples: {len(train_dataset)}")

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if data_args.max_eval_samples and data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            tokenized = eval_dataset.map(preprocess_reward_function, batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache)
            eval_dataset = tokenized.filter(
                lambda x: 0 < len(x['input_ids_rejected']) <= full_max_length and
                          0 < len(x['input_ids_chosen']) <= full_max_length)
            logger.info(f"Eval samples: {len(eval_dataset)}")

    # Trainer
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    model.enable_input_require_grads()
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = RewardTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer, compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=full_max_length, padding="max_length"))

    if training_args.do_train:
        if is_main_process:
            logger.info("*** Train ***")
            save_hyperparams(training_args.output_dir,
                stage="rm", model=model_args.model_name_or_path,
                lr=training_args.learning_rate, lora_rank=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                batch_size=training_args.per_device_train_batch_size,
                epochs=training_args.num_train_epochs, run_name=run_name)

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
            save_model(model, tokenizer, training_args)
            upload_checkpoint_to_hub(training_args.output_dir, repo_id,
                                    commit_message=f"Final RM model: {run_name}")

    if training_args.do_eval:
        if is_main_process:
            logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
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
