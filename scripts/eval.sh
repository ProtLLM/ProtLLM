#!/bin/bash

export TOKENIZERS_PARALLELISM=false

protein_model_checkpoint=ckpt/protst_esm2.pth
pretrain_ckpt=ckpt/protllm
sft_ckpt=ckpt/protllm
output_dir=output/eval/test
data_path=dataset/finetune/mf.json

# n_labels: mf: 489, bp: 1943, cc: 320, ec:538
torchrun --nproc_per_node=1 --master_port=25686 -m protllm.eval.run_sft_eval \
  --output_dir ${output_dir} \
  --protein_model_checkpoint ${protein_model_checkpoint} \
  --data_path ${data_path} --task "mf" --n_labels 489 \
  --llm_name_or_path meta-llama/Llama-2-7b-hf --lora_r 32 --lora_alpha 64 --bf16 \
  --batch_size 32 \
  --logging_steps 1000 \
  --save_every_n_samples 100000 \
  --report_to "none" \
  --resume_from_checkpoint ${pretrain_ckpt} \
  --resume_from_sft_checkpoint ${sft_ckpt} --sft_lora_r 128 --sft_lora_alpha 256