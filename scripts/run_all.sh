#!/bin/bash
# ==============================================================================
# Full RLHF Pipeline: SFT → Merge → RM → Merge → PPO → Eval
#
# Prerequisites:
#   - Set HF_TOKEN, WANDB_API_KEY, HF_USERNAME environment variables
#   - Run: python src/prepare_data.py --mode all
#
# Expected total time: ~10-12 hours on A100 40GB
# ==============================================================================
set -e

echo "============================================"
echo "  MedicalGPT Full RLHF Training Pipeline"
echo "============================================"
echo ""
echo "HF_USERNAME: ${HF_USERNAME:-NOT SET}"
echo "HF_TOKEN:    ${HF_TOKEN:+SET}"
echo "WANDB:       ${WANDB_API_KEY:+SET}"
echo ""

# Step 0: Prepare data
echo "=== Step 0: Preparing data ==="
python src/prepare_data.py --mode all --data_dir ./data

# Step 1: SFT
echo ""
echo "=== Step 1: Supervised Fine-Tuning ==="
bash scripts/run_sft.sh

# Step 2: Merge SFT + Train RM
echo ""
echo "=== Step 2: Merge SFT model ==="
bash scripts/run_merge.sh

# Step 3: RM Training
echo ""
echo "=== Step 3: Reward Model Training ==="
bash scripts/run_rm.sh

# Step 4: Merge RM (re-run merge for RM)
echo ""
echo "=== Step 4: Re-merge models (RM updated) ==="
bash scripts/run_merge.sh

# Step 5: PPO
echo ""
echo "=== Step 5: PPO Training (RLHF) ==="
bash scripts/run_ppo.sh

# Step 6: Evaluation
echo ""
echo "=== Step 6: Evaluation ==="
# Eval SFT model
bash scripts/run_eval.sh ./merged-sft

# Eval final PPO model
PPO_OUTPUT=$(ls -d outputs-ppo-* 2>/dev/null | head -1)
if [ -n "$PPO_OUTPUT" ]; then
    bash scripts/run_eval.sh ${PPO_OUTPUT}
fi

echo ""
echo "============================================"
echo "  Pipeline Complete!"
echo "============================================"
echo ""
echo "Outputs:"
echo "  SFT:     outputs-sft-*/"
echo "  RM:      outputs-rm-*/"
echo "  PPO:     outputs-ppo-*/"
echo "  Merged:  merged-sft/, merged-rm/"
echo "  Eval:    eval_results/"
