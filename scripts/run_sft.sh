#!/bin/bash
# ==============================================================================
# Stage 1: Supervised Fine-Tuning (SFT)
# Model: Qwen2.5-3B → LoRA fine-tuning on medical dialogue data (ChatDoctor)
# Expected time: ~3-4 hours on A100 40GB
# ==============================================================================
set -e

# ============ Configurable Hyperparameters ============
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
TEMPLATE="qwen"
LEARNING_RATE=2e-5
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
BATCH_SIZE=4
GRAD_ACCUM=8
NUM_EPOCHS=3
MAX_LENGTH=2048
MAX_TRAIN_SAMPLES=-1   # -1 for all
MAX_EVAL_SAMPLES=50
SAVE_STEPS=500
EVAL_STEPS=100

# ============ Paths ============
TRAIN_DIR="./data/finetune"
EVAL_DIR="./data/eval"
OUTPUT_DIR="outputs-sft-qwen2.5-3b-lr${LEARNING_RATE}-r${LORA_RANK}-a${LORA_ALPHA}-bs${BATCH_SIZE}-ep${NUM_EPOCHS}"

# ============ Training ============
CUDA_VISIBLE_DEVICES=0 python src/supervised_finetuning.py \
    --model_name_or_path ${MODEL_NAME} \
    --train_file_dir ${TRAIN_DIR} \
    --validation_file_dir ${EVAL_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --do_train \
    --do_eval \
    --template_name ${TEMPLATE} \
    --use_peft True \
    --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --max_eval_samples ${MAX_EVAL_SAMPLES} \
    --model_max_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps ${EVAL_STEPS} \
    --eval_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --preprocessing_num_workers 4 \
    --output_dir ${OUTPUT_DIR} \
    --logging_first_step True \
    --target_modules all \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --torch_dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to wandb \
    --gradient_checkpointing True \
    --cache_dir ./cache

echo "SFT training complete. Output: ${OUTPUT_DIR}"
