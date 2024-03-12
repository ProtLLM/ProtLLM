#!/bin/bash

export TOKENIZERS_PARALLELISM=false

protein_model_checkpoint=ckpt/protst_esm2.pth
pretrain_ckpt=ckpt/protllm
output_dir=output/eval/test
data_path=dataset/finetune/ppi.json

torchrun --nproc_per_node=1 --master_port=25686 -m protllm.eval.run_sft_eval \
  --output_dir ${output_dir} \
  --protein_model_checkpoint ${protein_model_checkpoint} \
  --data_path ${data_path} --task "ppi" --n_demo 16 \
  --llm_name_or_path meta-llama/Llama-2-7b-hf --lora_r 32 --lora_alpha 64 --bf16 \
  --report_to "none" \
  --resume_from_checkpoint ${pretrain_ckpt} \