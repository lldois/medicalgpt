#!/bin/bash
# ==============================================================================
# Evaluation pipeline - supports all stages
# Usage:
#   bash scripts/run_eval.sh pretrain <model_path>
#   bash scripts/run_eval.sh sft <model_path>
#   bash scripts/run_eval.sh rm <model_path>
#   bash scripts/run_eval.sh rlhf <model_path> <reward_model_path>
# ==============================================================================
set -e

STAGE="${1:?Usage: run_eval.sh <stage> <model_path> [reward_model_path]}"
MODEL_PATH="${2:?Usage: run_eval.sh <stage> <model_path> [reward_model_path]}"
REWARD_MODEL_PATH="${3:-}"
OUTPUT_DIR="./eval_results"

echo "=== Evaluating stage=${STAGE}, model=${MODEL_PATH} ==="

case $STAGE in
    pretrain)
        python src/evaluate.py \
            --stage pretrain \
            --model_path ${MODEL_PATH} \
            --eval_data ./data/pretrain_prepared/test.jsonl \
            --max_length 1024 \
            --output_dir ${OUTPUT_DIR}
        ;;
    sft)
        python src/evaluate.py \
            --stage sft \
            --model_path ${MODEL_PATH} \
            --eval_data ./data/sft/test.jsonl \
            --template_name qwen \
            --max_length 512 \
            --max_new_tokens 256 \
            --max_eval_samples 100 \
            --output_dir ${OUTPUT_DIR}
        ;;
    rm)
        python src/evaluate.py \
            --stage rm \
            --model_path ${MODEL_PATH} \
            --eval_data ./data/reward_converted/test.jsonl \
            --template_name qwen \
            --max_length 512 \
            --output_dir ${OUTPUT_DIR}
        ;;
    rlhf)
        if [ -z "$REWARD_MODEL_PATH" ]; then
            echo "Error: RLHF evaluation requires reward_model_path as 3rd argument"
            exit 1
        fi
        python src/evaluate.py \
            --stage rlhf \
            --model_path ${MODEL_PATH} \
            --reward_model_path ${REWARD_MODEL_PATH} \
            --eval_data ./data/sft/test.jsonl \
            --template_name qwen \
            --max_length 512 \
            --max_new_tokens 256 \
            --max_eval_samples 50 \
            --output_dir ${OUTPUT_DIR}
        ;;
    *)
        echo "Unknown stage: $STAGE (use: pretrain, sft, rm, rlhf)"
        exit 1
        ;;
esac

echo "Evaluation complete. Results: ${OUTPUT_DIR}/eval_${STAGE}.json"
