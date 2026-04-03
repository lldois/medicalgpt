#!/bin/bash
# ==============================================================================
# Minimal CPU test: verify all training scripts & evaluation run with tiny data
# Used for local debugging before GPU training
# ==============================================================================
set -e

BASE_MODEL="Qwen/Qwen2.5-0.5B"

echo "=== Step 0: Creating minimal test data ==="
python src/prepare_data.py --mode minimal --data_dir ./data

# ---------- Pretrain ----------
echo ""
echo "=== Testing Pretrain (CPU, 2 samples, 2 steps) ==="
python src/pretrain.py \
    --model_name_or_path ${BASE_MODEL} \
    --train_file_dir ./data/pretrain_prepared \
    --validation_file_dir ./data/pretrain_prepared \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 2 \
    --max_eval_samples 2 \
    --block_size 128 \
    --num_train_epochs 1 \
    --max_steps 2 \
    --learning_rate 2e-5 \
    --logging_steps 1 \
    --save_steps 999999 \
    --gradient_accumulation_steps 1 \
    --output_dir outputs-test-pt \
    --target_modules all \
    --lora_rank 4 \
    --lora_alpha 8 \
    --torch_dtype float32 \
    --use_cpu \
    --report_to none \
    --gradient_checkpointing False

echo ""
echo "=== Testing Pretrain Eval ==="
python src/evaluate.py \
    --stage pretrain \
    --model_path ${BASE_MODEL} \
    --eval_data ./data/pretrain_prepared/test.jsonl \
    --max_eval_samples 2 \
    --max_length 128 \
    --device cpu

# ---------- SFT ----------
echo ""
echo "=== Testing SFT (CPU, 2 samples, 2 steps) ==="
python src/supervised_finetuning.py \
    --model_name_or_path ${BASE_MODEL} \
    --train_file_dir ./data/sft \
    --validation_file_dir ./data/sft \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --max_train_samples 2 \
    --max_eval_samples 2 \
    --model_max_length 128 \
    --num_train_epochs 1 \
    --max_steps 2 \
    --learning_rate 2e-5 \
    --logging_steps 1 \
    --save_steps 999999 \
    --gradient_accumulation_steps 1 \
    --output_dir outputs-test-sft \
    --target_modules all \
    --lora_rank 4 \
    --lora_alpha 8 \
    --torch_dtype float32 \
    --use_cpu \
    --report_to none \
    --gradient_checkpointing False

echo ""
echo "=== Testing SFT Eval ==="
python src/evaluate.py \
    --stage sft \
    --model_path ${BASE_MODEL} \
    --eval_data ./data/sft/test.jsonl \
    --template_name qwen \
    --max_eval_samples 2 \
    --max_length 128 \
    --max_new_tokens 32 \
    --device cpu

# ---------- Reward Model ----------
echo ""
echo "=== Testing Reward Model (CPU, 2 samples, 2 steps) ==="
python src/reward_modeling.py \
    --model_name_or_path ${BASE_MODEL} \
    --train_file_dir ./data/reward_converted \
    --validation_file_dir ./data/reward_converted \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 2 \
    --max_eval_samples 2 \
    --num_train_epochs 1 \
    --max_steps 2 \
    --learning_rate 2e-5 \
    --logging_steps 1 \
    --save_steps 999999 \
    --max_source_length 128 \
    --max_target_length 64 \
    --output_dir outputs-test-rm \
    --target_modules all \
    --lora_rank 4 \
    --lora_alpha 8 \
    --template_name qwen \
    --torch_dtype float32 \
    --use_cpu \
    --report_to none \
    --remove_unused_columns False \
    --gradient_checkpointing False

echo ""
echo "=== Testing RM Eval ==="
python src/evaluate.py \
    --stage rm \
    --model_path ${BASE_MODEL} \
    --eval_data ./data/reward_converted/test.jsonl \
    --template_name qwen \
    --max_eval_samples 2 \
    --max_length 128 \
    --device cpu

echo ""
echo "=========================================="
echo "  All CPU tests passed!"
echo "=========================================="
echo "Cleaning up test outputs..."
rm -rf outputs-test-pt outputs-test-sft outputs-test-rm eval_results
echo "Done."
