#!/bin/bash
# ==============================================================================
# Merge LoRA adapters into base model
# Run after SFT and RM training to create merged models for PPO
# ==============================================================================
set -e

BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"

# Merge SFT LoRA
echo "=== Merging SFT model ==="
SFT_OUTPUT=$(ls -d outputs-sft-* 2>/dev/null | head -1)
if [ -z "$SFT_OUTPUT" ]; then
    echo "Error: No SFT output directory found"
    exit 1
fi
python src/merge_peft_adapter.py \
    --base_model ${BASE_MODEL} \
    --lora_model ${SFT_OUTPUT} \
    --output_dir merged-sft

# Merge RM LoRA
echo "=== Merging Reward model ==="
RM_OUTPUT=$(ls -d outputs-rm-* 2>/dev/null | head -1)
if [ -z "$RM_OUTPUT" ]; then
    echo "Error: No RM output directory found"
    exit 1
fi
python src/merge_peft_adapter.py \
    --base_model ${BASE_MODEL} \
    --lora_model ${RM_OUTPUT} \
    --output_dir merged-rm

echo "Merge complete!"
echo "  SFT merged: merged-sft/"
echo "  RM merged:  merged-rm/"
