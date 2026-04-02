# -*- coding: utf-8 -*-
"""
Stage 1: Supervised Fine-Tuning (SFT)
Based on MedicalGPT/CustomGPT by shibing624, with additions:
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
from types import MethodType
from typing import Literal, Optional, Tuple

import torch
import torch.utils.data
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
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import LabelSmoother
from transformers.integrations import is_deepspeed_zero3_enabled

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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: Optional[str] = field(default="main")
    hf_hub_token: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=False)
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={"choices": ["auto", "bfloat16", "float16", "float32"]},
    )
    device_map: Optional[str] = field(default="auto")
    trust_remote_code: bool = field(default=True)
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(default=None)
    flash_attn: Optional[bool] = field(default=False)
    shift_attn: Optional[bool] = field(default=False)
    neft_alpha: Optional[float] = field(default=0)

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required")


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file_dir: Optional[str] = field(default=None)
    validation_file_dir: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    ignore_pad_token_for_loss: bool = field(default=True)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=1)
    preprocessing_num_workers: Optional[int] = field(default=None)


@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True)
    train_on_inputs: bool = field(default=False)
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    qlora: bool = field(default=False)
    model_max_length: int = field(default=512)
    template_name: Optional[str] = field(default="qwen")
    # Custom: upload checkpoint every N steps
    upload_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Upload checkpoint to HF Hub every N steps. If None, only upload at end."
        },
    )

    def __post_init__(self):
        if self.model_max_length < 60:
            raise ValueError("model_max_length must be >= 60")


class SavePeftModelTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


class HubUploadCallback:
    """Callback to upload checkpoints to HuggingFace Hub at fixed intervals."""

    def __init__(self, repo_id: str, upload_steps: int = None):
        self.repo_id = repo_id
        self.upload_steps = upload_steps

    def on_save(self, args, state, control, **kwargs):
        if self.upload_steps and state.global_step % self.upload_steps == 0:
            checkpoint_dir = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )
            if os.path.exists(checkpoint_dir):
                upload_checkpoint_to_hub(
                    checkpoint_dir,
                    self.repo_id,
                    commit_message=f"checkpoint-{state.global_step}",
                )


def save_model(model, tokenizer, args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_zero3(model, tokenizer, args, trainer):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)
    tokenizer.save_pretrained(output_dir)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param}"
    )


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
            if "lm_head" in name:
                continue
            if "output_layer" in name:
                continue
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments)
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

    # Build run name & setup tracking
    run_name = build_run_name(
        "sft",
        model=model_args.model_name_or_path,
        lr=training_args.learning_rate,
        lora_rank=script_args.lora_rank,
        lora_alpha=script_args.lora_alpha,
        batch_size=training_args.per_device_train_batch_size,
        epochs=training_args.num_train_epochs,
        max_length=script_args.model_max_length,
    )
    repo_id = get_hf_repo_id(run_name)

    if is_main_process:
        logger.info(f"Run name: {run_name}")
        logger.info(f"HF Hub repo: {repo_id}")
        setup_wandb("medicalgpt", run_name)

    # Set wandb run name
    if not training_args.run_name:
        training_args.run_name = run_name

    # Auto-resume checkpoint
    if training_args.resume_from_checkpoint is None:
        resolved = resolve_resume_checkpoint(training_args.output_dir, repo_id)
        if resolved:
            training_args.resume_from_checkpoint = resolved

    set_seed(training_args.seed)

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = (
        model_args.tokenizer_name_or_path or model_args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, **tokenizer_kwargs
    )
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

    IGNORE_INDEX = (
        LabelSmoother.ignore_index
        if data_args.ignore_pad_token_for_loss
        else tokenizer.pad_token_id
    )

    # Load datasets
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            shuffled = raw_datasets["train"].shuffle(seed=42)
            split = shuffled.train_test_split(
                test_size=data_args.validation_split_percentage / 100, seed=42
            )
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
    else:
        data_files = {}
        if data_args.train_file_dir and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(
                f"{data_args.train_file_dir}/**/*.json", recursive=True
            ) + glob(f"{data_args.train_file_dir}/**/*.jsonl", recursive=True)
            data_files["train"] = train_data_files
        if data_args.validation_file_dir and os.path.exists(
            data_args.validation_file_dir
        ):
            eval_data_files = glob(
                f"{data_args.validation_file_dir}/**/*.json", recursive=True
            ) + glob(f"{data_args.validation_file_dir}/**/*.jsonl", recursive=True)
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            "json", data_files=data_files, cache_dir=model_args.cache_dir
        )
        if "validation" not in raw_datasets.keys():
            shuffled = raw_datasets["train"].shuffle(seed=42)
            split = shuffled.train_test_split(
                test_size=float(data_args.validation_split_percentage / 100), seed=42
            )
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]

    logger.info(f"Raw datasets: {raw_datasets}")

    max_length = script_args.model_max_length

    def preprocess_function(examples):
        input_ids_list = []
        attention_mask_list = []
        targets_list = []
        roles = ["human", "gpt"]

        def get_dialog(examples):
            system_prompts = examples.get("system_prompt", "")
            for i, source in enumerate(examples["conversations"]):
                system_prompt = ""
                if len(source) < 2:
                    continue
                data_role = source[0].get("from", "")
                if data_role == "system":
                    system_prompt = source[0]["value"]
                    source = source[1:]
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
                if len(messages) % 2 != 0:
                    continue
                history_messages = [
                    [messages[k], messages[k + 1]] for k in range(0, len(messages), 2)
                ]
                if not system_prompt:
                    system_prompt = system_prompts[i] if system_prompts else ""
                yield prompt_template.get_dialog(
                    history_messages, system_prompt=system_prompt
                )

        for dialog in get_dialog(examples):
            input_ids, labels = [], []
            for i in range(len(dialog) // 2):
                source_ids = tokenizer.encode(
                    text=dialog[2 * i], add_special_tokens=(i == 0)
                )
                target_ids = tokenizer.encode(
                    text=dialog[2 * i + 1], add_special_tokens=False
                )

                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(max_length * (len(source_ids) / total_len))
                max_target_len = int(max_length * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len - 1:
                    target_ids = target_ids[: max_target_len - 1]
                if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
                    source_ids = source_ids[1:]
                if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
                    target_ids = target_ids[:-1]
                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
                    break

                input_ids += source_ids + target_ids + [tokenizer.eos_token_id]
                if script_args.train_on_inputs:
                    labels += source_ids + target_ids + [tokenizer.eos_token_id]
                else:
                    labels += (
                        [IGNORE_INDEX] * len(source_ids)
                        + target_ids
                        + [tokenizer.eos_token_id]
                    )

            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)

        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=targets_list,
        )

    def filter_empty_labels(example):
        return not all(label == IGNORE_INDEX for label in example["labels"])

    # Prepare datasets
    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        train_dataset = raw_datasets["train"].shuffle(seed=42)
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="Train dataset tokenization"):
            tokenized = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
            train_dataset = tokenized.filter(
                filter_empty_labels, num_proc=data_args.preprocessing_num_workers
            )
            if is_main_process:
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
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
            eval_dataset = eval_dataset.filter(
                filter_empty_labels, num_proc=data_args.preprocessing_num_workers
            )
            if is_main_process:
                logger.info(f"Eval samples: {len(eval_dataset)}")

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
        training_args.gradient_accumulation_steps = (
            training_args.gradient_accumulation_steps // world_size or 1
        )

    config_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    load_in_4bit = model_args.load_in_4bit
    load_in_8bit = model_args.load_in_8bit
    quantization_config = None
    if load_in_8bit or load_in_4bit:
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif load_in_4bit:
            if script_args.qlora:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype
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

    # NEFTune
    if model_args.neft_alpha > 0:
        input_embed = model.get_input_embeddings()
        if isinstance(input_embed, torch.nn.Embedding):

            def noisy_forward(self, x):
                embeddings = torch.nn.Embedding.forward(self, x)
                dims = self.num_embeddings * self.embedding_dim
                mag_norm = model_args.neft_alpha / (dims**0.5)
                embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
                return embeddings

            input_embed.forward = MethodType(noisy_forward, input_embed)

    # LoRA
    if script_args.use_peft:
        logger.info("Using LoRA (PEFT)")
        output_layer = getattr(model, "lm_head")
        if (
            isinstance(output_layer, torch.nn.Linear)
            and output_layer.weight.dtype != torch.float32
        ):

            def fp32_forward_post_hook(module, args, output):
                return output.to(torch.float32)

            output_layer.register_forward_hook(fp32_forward_post_hook)

        if script_args.peft_path is not None:
            model = PeftModel.from_pretrained(
                model, script_args.peft_path, is_trainable=True
            )
        else:
            if load_in_8bit or load_in_4bit:
                model = prepare_model_for_kbit_training(
                    model, training_args.gradient_checkpointing
                )
            target_modules = (
                script_args.target_modules.split(",")
                if script_args.target_modules
                else None
            )
            if target_modules and "all" in target_modules:
                target_modules = find_all_linear_names(
                    model, int4=load_in_4bit, int8=load_in_8bit
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
    else:
        model = model.float()
        print_trainable_parameters(model)

    # Trainer setup
    if training_args.gradient_checkpointing and getattr(
        model, "supports_gradient_checkpointing", False
    ):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    model.enable_input_require_grads()

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,
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
            logger.info("*** Train ***")
            save_hyperparams(
                training_args.output_dir,
                stage="sft",
                model=model_args.model_name_or_path,
                lr=training_args.learning_rate,
                lora_rank=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                batch_size=training_args.per_device_train_batch_size,
                epochs=training_args.num_train_epochs,
                max_length=script_args.model_max_length,
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
            if is_deepspeed_zero3_enabled():
                save_model_zero3(model, tokenizer, training_args, trainer)
            else:
                save_model(model, tokenizer, training_args)

            # Upload final model to HF Hub
            upload_checkpoint_to_hub(
                training_args.output_dir,
                repo_id,
                commit_message=f"Final SFT model: {run_name}",
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
