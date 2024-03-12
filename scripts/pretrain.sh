#!/bin/bash

export TOKENIZERS_PARALLELISM=false

run_name=pretrain-test
sft_output_dir=output/${run_name}
train_data_dir=dataset/pretrain

CUDA_VISIBLE_DEVICES="" torchrun --nproc_per_node=1 --master_port=25701 -m protllm.train.train \
  --train_data_dir ${train_data_dir} \
  --output_dir ${sft_output_dir} \
  --bf16 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 8 \
  --eval_accumulation_steps 8 \
  --eval_steps 1000 \
  --evaluation_strategy "steps" \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --dataloader_num_workers 4 \
  --ddp_find_unused_parameters False \
  --save_steps 5000 \
  --save_strategy "steps" \
  --max_steps 50000 \
  --logging_steps 1 --llm_max_len 512 --seed 2 \
  --report_to "wandb" \
  --run_name ${run_name} \
  --llm_name_or_path meta-llama/Llama-2-7b-hf \
  --lora_r 32 --lora_alpha 64 --lora_bias "none" --text2prot_prob 0.2