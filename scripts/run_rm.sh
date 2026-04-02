#!/bin/bash
# ==============================================================================
# Stage 2: Reward Model Training
# Train reward model on preference data for safety alignment (pairwise ranking)
# Expected time: ~2 hours on A100 40GB
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
NUM_EPOCHS=2
MAX_SOURCE_LENGTH=1024
MAX_TARGET_LENGTH=256
MAX_TRAIN_SAMPLES=-1
MAX_EVAL_SAMPLES=50
SAVE_STEPS=500
EVAL_STEPS=100

# ============ Paths ============
TRAIN_DIR="./data/reward"
EVAL_DIR="./data/reward"
OUTPUT_DIR="outputs-rm-qwen2.5-3b-lr${LEARNING_RATE}-r${LORA_RANK}-a${LORA_ALPHA}-bs${BATCH_SIZE}-ep${NUM_EPOCHS}"

# ============ Training ============
# Note: reward model training does not support torchrun multi-GPU
CUDA_VISIBLE_DEVICES=0 python src/reward_modeling.py \
    --model_name_or_path ${MODEL_NAME} \
    --train_file_dir ${TRAIN_DIR} \
    --validation_file_dir ${EVAL_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --max_eval_samples ${MAX_EVAL_SAMPLES} \
    --num_train_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio 0.05 \
    --weight_decay 0.001 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps ${EVAL_STEPS} \
    --eval_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --save_strategy steps \
    --save_total_limit 3 \
    --max_source_length ${MAX_SOURCE_LENGTH} \
    --max_target_length ${MAX_TARGET_LENGTH} \
    --output_dir ${OUTPUT_DIR} \
    --logging_first_step True \
    --target_modules all \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --template_name ${TEMPLATE} \
    --bf16 \
    --torch_dtype bfloat16 \
    --device_map auto \
    --report_to wandb \
    --remove_unused_columns False \
    --gradient_checkpointing True

echo "Reward model training complete. Output: ${OUTPUT_DIR}"
