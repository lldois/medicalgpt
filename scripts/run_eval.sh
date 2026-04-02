#!/bin/bash
# ==============================================================================
# Evaluation pipeline
# Run after PPO training (or after any stage for intermediate eval)
# ==============================================================================
set -e

MODEL_PATH="${1:-./merged-sft}"
EVAL_DATA="./data/eval"
OUTPUT_DIR="./eval_results"

echo "=== Evaluating model: ${MODEL_PATH} ==="

# Find eval data file
EVAL_FILE=$(find ${EVAL_DATA} -name "*.jsonl" | head -1)
if [ -z "$EVAL_FILE" ]; then
    echo "Warning: No eval data found in ${EVAL_DATA}, running without perplexity"
    EVAL_FILE=""
fi

python src/evaluate.py \
    --model_path ${MODEL_PATH} \
    --eval_data "${EVAL_FILE}" \
    --template_name qwen \
    --max_length 512 \
    --max_new_tokens 256 \
    --output_dir ${OUTPUT_DIR}

echo "Evaluation complete. Results: ${OUTPUT_DIR}/eval_results.json"
