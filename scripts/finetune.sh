#!/bin/bash

mkdir -p /data/chatglm3-6b-lora

export PYTORCH_JIT=0
LOWER_LIST=scripts/ops_bf16.txt python3 finetune/finetune.py \
    --model_name_or_path /data/chatglm3-6b \
    --train_file AdvertiseGen/train.json \
    --validation_file AdvertiseGen/dev.json \
    --bf16 True \
    --output_dir /data/chatglm3-6b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-4 \
    --warmup_ratio  0.03 \
    --lr_scheduler_type "constant" \
    --max_grad_norm  0.3 \
    --logging_steps 1 \
    --do_train \
    --use_habana \
    --use_lazy_mode \
    --throughput_warmup_steps 3 \
    --lora_rank=8 \
    --lora_alpha=16 \
    --lora_dropout=0.05 \
    --lora_target_modules "query_key_value" \
    --max_step 500 \
    --max_seq_length 512 \
    --low_cpu_mem_usage True \
    --adam_epsilon 1e-08