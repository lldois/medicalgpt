#!/bin/bash
# ==============================================================================
# Full RLHF Pipeline: PT → SFT → Merge → RM → Merge → PPO → Eval
#
# Prerequisites:
#   - Set HF_TOKEN, WANDB_API_KEY, HF_USERNAME environment variables
#   - Data prepared via: python src/prepare_data.py --mode all
#
# Expected total time: ~10-12 hours on A100 40GB
#
# To skip pretrain, comment out Step 1 and Step 1.5 below.
# ==============================================================================
set -e

echo "============================================"
echo "  MedicalGPT - Medical RLHF Training Pipeline"
echo "============================================"
echo ""
echo "HF_USERNAME: ${HF_USERNAME:-NOT SET}"
echo "HF_TOKEN:    ${HF_TOKEN:+SET}"
echo "WANDB:       ${WANDB_API_KEY:+SET}"
echo ""

# Step 0: Prepare data
echo "=== Step 0: Preparing data ==="
python src/prepare_data.py --mode all --data_dir ./data

# Step 1: Continuous Pretrain (OPTIONAL - comment out to skip)
echo ""
echo "=== Step 1: Continuous Pretraining ==="
bash scripts/run_pretrain.sh

# Step 1.5: Evaluate pretrain + Merge
echo ""
echo "=== Step 1.5: Merge pretrain & Evaluate ==="
bash scripts/run_merge.sh pt
bash scripts/run_eval.sh pretrain ./merged-pt

# Step 2: SFT
echo ""
echo "=== Step 2: Supervised Fine-Tuning ==="
bash scripts/run_sft.sh

# Step 2.5: Merge SFT & Evaluate
echo ""
echo "=== Step 2.5: Merge SFT & Evaluate ==="
bash scripts/run_merge.sh sft
bash scripts/run_eval.sh sft ./merged-sft

# Step 3: Reward Model
echo ""
echo "=== Step 3: Reward Model Training ==="
bash scripts/run_rm.sh

# Step 3.5: Merge RM & Evaluate
echo ""
echo "=== Step 3.5: Merge RM & Evaluate ==="
bash scripts/run_merge.sh rm
bash scripts/run_eval.sh rm ./merged-rm

# Step 4: PPO
echo ""
echo "=== Step 4: PPO Training (RLHF) ==="
bash scripts/run_ppo.sh

# Step 4.5: Evaluate RLHF
echo ""
echo "=== Step 4.5: Evaluate RLHF ==="
PPO_OUTPUT=$(ls -d outputs-ppo-* 2>/dev/null | head -1)
if [ -n "$PPO_OUTPUT" ]; then
    bash scripts/run_eval.sh rlhf ${PPO_OUTPUT} ./merged-rm
fi

echo ""
echo "============================================"
echo "  Pipeline Complete!"
echo "============================================"
echo ""
echo "Results in: eval_results/"
echo "  eval_pretrain.json - Pretrain PPL"
echo "  eval_sft.json      - SFT PPL + BLEU + ROUGE"
echo "  eval_rm.json       - RM Accuracy + Reward Gap"
echo "  eval_rlhf.json     - RLHF Reward + BLEU + ROUGE"
