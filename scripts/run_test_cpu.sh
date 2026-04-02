#!/bin/bash
# ==============================================================================
# Minimal CPU test: verify all scripts run with tiny data
# Used for local debugging before GPU training
# ==============================================================================
set -e

echo "=== Creating minimal test data ==="
python src/prepare_data.py --mode minimal --data_dir ./data

echo ""
echo "=== Testing SFT (CPU, 2 samples, 1 step) ==="
python src/supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --train_file_dir ./data/finetune \
    --validation_file_dir ./data/finetune \
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
echo "=== Testing Reward Model (CPU, 2 samples, 1 step) ==="
python src/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --train_file_dir ./data/reward \
    --validation_file_dir ./data/reward \
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
echo "=== Testing Evaluation (CPU) ==="
python src/evaluate.py \
    --model_path Qwen/Qwen2.5-0.5B \
    --eval_data ./data/eval/test_eval.jsonl \
    --template_name qwen \
    --max_length 128 \
    --max_new_tokens 32 \
    --output_dir eval_results_test \
    --device cpu

echo ""
echo "=== All CPU tests passed! ==="
echo "Cleaning up test outputs..."
rm -rf outputs-test-sft outputs-test-rm eval_results_test
echo "Done."
