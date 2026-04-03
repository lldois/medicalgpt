#!/bin/bash
# ==============================================================================
# Merge LoRA adapters into base model
# Usage:
#   bash scripts/run_merge.sh           # merge all available stages
#   bash scripts/run_merge.sh pt        # merge pretrain only
#   bash scripts/run_merge.sh sft       # merge SFT only
#   bash scripts/run_merge.sh rm        # merge RM only
# ==============================================================================
set -e

STAGE="${1:-all}"
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"

merge_stage() {
    local stage=$1
    local prefix=$2
    local output=$3
    
    echo "=== Merging ${stage} model ==="
    STAGE_OUTPUT=$(ls -d ${prefix}* 2>/dev/null | head -1)
    if [ -z "$STAGE_OUTPUT" ]; then
        echo "Warning: No ${stage} output directory found (${prefix}*), skipping"
        return 0
    fi
    
    # For pretrain, the base model for SFT should be the pretrained model
    local base=${BASE_MODEL}
    if [ "$stage" = "sft" ] && [ -d "merged-pt" ]; then
        base="merged-pt"
        echo "  Using pretrained base: ${base}"
    fi
    
    python src/merge_peft_adapter.py \
        --base_model ${base} \
        --lora_model ${STAGE_OUTPUT} \
        --output_dir ${output}
    echo "  Merged to: ${output}/"
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "pt" ]; then
    merge_stage "pretrain" "outputs-pt-" "merged-pt"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "sft" ]; then
    merge_stage "SFT" "outputs-sft-" "merged-sft"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "rm" ]; then
    merge_stage "RM" "outputs-rm-" "merged-rm"
fi

echo "Merge complete!"
