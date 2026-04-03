#!/bin/bash
# ==============================================================================
# Stage 3: PPO Training (RLHF)
# Align medical SFT model using reward model via PPO for safety
# Expected time: ~4-5 hours on A100 40GB
# ==============================================================================
set -e

# ============ Configurable Hyperparameters ============
TEMPLATE="qwen"
BATCH_SIZE=1
GRAD_ACCUM=4
MAX_SOURCE_LENGTH=1024
RESPONSE_LENGTH=512
TOTAL_EPISODES=10000
NUM_EPOCHS=1
EVAL_STEPS=200

# ============ Paths (must match SFT and RM output dirs) ============
# These should point to the MERGED SFT and RM models
SFT_MODEL_PATH="./merged-sft"
REWARD_MODEL_PATH="./merged-rm"
TRAIN_DIR="./data/sft"
EVAL_DIR="./data/sft"
OUTPUT_DIR="outputs-ppo-qwen2.5-3b-bs${BATCH_SIZE}-ep${NUM_EPOCHS}"

# ============ Training ============
CUDA_VISIBLE_DEVICES=0 python src/ppo_training.py \
    --sft_model_path ${SFT_MODEL_PATH} \
    --reward_model_path ${REWARD_MODEL_PATH} \
    --template_name ${TEMPLATE} \
    --torch_dtype bfloat16 \
    --train_file_dir ${TRAIN_DIR} \
    --validation_file_dir ${EVAL_DIR} \
    --max_source_length ${MAX_SOURCE_LENGTH} \
    --response_length ${RESPONSE_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --gradient_checkpointing True \
    --do_train \
    --total_episodes ${TOTAL_EPISODES} \
    --output_dir ${OUTPUT_DIR} \
    --missing_eos_penalty 1.0 \
    --eval_strategy steps \
    --eval_steps ${EVAL_STEPS} \
    --num_train_epochs ${NUM_EPOCHS} \
    --report_to wandb

echo "PPO training complete. Output: ${OUTPUT_DIR}"
