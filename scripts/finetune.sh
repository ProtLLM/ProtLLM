#!/bin/bash

export TOKENIZERS_PARALLELISM=false

run_name=sft-mf
sft_output_dir=output/${run_name}
protllm_path=ckpt/protllm
protein_model_checkpoint=ckpt/protst_esm2.pth
train_data_dir=dataset/finetune/cc.json

torchrun --nproc_per_node=1 --master_port=25690 -m protllm.train.train \
  --output_dir ${sft_output_dir} \
  --main_training_task sft --task mf --train_data_dir ${train_data_dir} \
  --protein_model_checkpoint ${protein_model_checkpoint} \
  --bf16 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 4 \
  --eval_accumulation_steps 8 \
  --eval_steps 2000 --do_eval True \
  --evaluation_strategy "steps" \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --dataloader_num_workers 4 \
  --ddp_find_unused_parameters True \
  --save_steps 5000 \
  --save_strategy "steps" \
  --max_steps 50000 \
  --logging_steps 1 --llm_max_len 512 --seed 1 \
  --llm_name_or_path meta-llama/Llama-2-7b-hf \
  --sft_lora_r 128 --sft_lora_alpha 256 --sft_with_lora True \
  --save_safetensors False \
  --report_to "wandb" \
  --run_name ${run_name} \
  --resume_from_checkpoint ${protllm_path} \
  --protein_model_name "protst" --prot_output_size 512 \
  --lora_r 32 --lora_alpha 64 \
  --lr_ratio 0.1 \
  --learn_protst True \

