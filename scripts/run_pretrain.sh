#!/bin/bash
# ==============================================================================
# Stage 0: Continuous Pretraining (PT)
# Model: Qwen2.5-3B → LoRA pretrain on medical encyclopedia & books
# Expected time: ~3-4 hours on A100 40GB
# ==============================================================================
set -e

# ============ Configurable Hyperparameters ============
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
LEARNING_RATE=2e-5
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
BATCH_SIZE=4
GRAD_ACCUM=8
NUM_EPOCHS=1
BLOCK_SIZE=1024
MAX_TRAIN_SAMPLES=-1   # -1 for all
MAX_EVAL_SAMPLES=100
SAVE_STEPS=500
EVAL_STEPS=200

# ============ Paths ============
TRAIN_DIR="./data/pretrain_prepared"
EVAL_DIR="./data/pretrain_prepared"
OUTPUT_DIR="outputs-pt-qwen2.5-3b-lr${LEARNING_RATE}-r${LORA_RANK}-a${LORA_ALPHA}-bs${BATCH_SIZE}-ep${NUM_EPOCHS}"

# ============ Training ============
CUDA_VISIBLE_DEVICES=0 python src/pretrain.py \
    --model_name_or_path ${MODEL_NAME} \
    --train_file_dir ${TRAIN_DIR} \
    --validation_file_dir ${EVAL_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --max_eval_samples ${MAX_EVAL_SAMPLES} \
    --block_size ${BLOCK_SIZE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --eval_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --output_dir ${OUTPUT_DIR} \
    --target_modules all \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --torch_dtype bfloat16 \
    --bf16 \
    --gradient_checkpointing True \
    --report_to wandb

echo "Pretrain complete! Output: ${OUTPUT_DIR}"
