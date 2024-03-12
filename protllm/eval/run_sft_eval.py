import os
import math
import torch
import transformers
import random

from tqdm import tqdm
from functools import partial
from typing import Optional
from dataclasses import dataclass, field
from protllm.eval.eval_util import f1_max, area_under_prc
from protllm.model.protllm import ProtLlmForBinaryCls
from protllm.data.data_eval import Ppi4ProtLlmDataset, GO4ProtLlmDataset
from protllm.train.train import MyModelArguments, MyTrainingArguments, Trainer4ProtLlm
from peft import get_peft_model, LoraConfig


@dataclass
class MyEvaluationArguments:
  task: Optional[str] = field(default="ppi", metadata={"help": "ppi, mf, cc, bp, ec"})
  data_path: Optional[str] = field(default="")
  eval_seed: Optional[int] = field(default=42)
  resume_from_sft_checkpoint: Optional[str] = field(default="")
  n_demo: Optional[int] = field(default=0)
  n_labels: Optional[int] = field(default=1)
  batch_size: Optional[int] = field(default=4)
  save_every_n_samples: Optional[int] = field(default=200000)
  eval_split: Optional[str] = field(default="test")


def eval_main():
  parser = transformers.HfArgumentParser((MyModelArguments, MyEvaluationArguments, MyTrainingArguments))
  model_args, eval_args, training_args = parser.parse_args_into_dataclasses()
  
  random.seed(eval_args.eval_seed)

  model = ProtLlmForBinaryCls(model_args, device=training_args.device)
  model.init_prot_model(model_args, device=training_args.device)

  if eval_args.task == "ppi":
    if eval_args.n_demo == 0:
      eval_dataset = Ppi4ProtLlmDataset(model.llm_tok, prot_tok=model.prot_tok, data_path=eval_args.data_path, split=eval_args.eval_split)
    else:
      Ppi4ProtLlmInContextDataset(model.llm_tok, prot_tok=model.prot_tok, data_path=eval_args.data_path, n_demo=eval_args.n_demo)
  else:
    eval_dataset = GO4ProtLlmDataset(model.llm_tok, prot_tok=model.prot_tok, data_path=eval_args.data_path, split="test", task=eval_args.task)
  data_collator = partial(eval_dataset.collator, lm_pad_token_id=model.llm_tok.eos_token_id, prot_pad_token_id=model.prot_tok.padding_idx)
  eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_args.batch_size, collate_fn=data_collator, shuffle=False, num_workers=4)

  if training_args.resume_from_checkpoint:
    peft_config = LoraConfig(
        inference_mode=False, r=model_args.lora_r, lora_alpha=model_args.lora_alpha, lora_dropout=model_args.lora_dropout, bias=model_args.lora_bias, target_modules=model_args.pretrain_target_modules.split(","))
    model = get_peft_model(model, peft_config)
    training_args.report_to = []
    trainer = Trainer4ProtLlm(
      model=model,
      args=training_args,
      eval_dataset=eval_dataset,
      tokenizer=model.llm_tok,
      data_collator=data_collator,
    )
    trainer._load_from_checkpoint(training_args.resume_from_checkpoint)
    model = trainer.model.merge_and_unload()

  if eval_args.resume_from_sft_checkpoint:
    peft_config = LoraConfig(
      inference_mode=False, r=model_args.sft_lora_r, lora_alpha=model_args.sft_lora_alpha, lora_dropout=model_args.lora_dropout, bias=model_args.lora_bias, target_modules=model_args.sft_target_modules.split(","))
    model = get_peft_model(model, peft_config)
    model.base_model.post_init_prot_model(model_args, device=training_args.device, learn_protst=False)
    model.base_model.init_cls_head(label2tok_ids=eval_dataset.label2tok_ids)

    trainer = Trainer4ProtLlm(
      model=model,
      args=training_args,
      eval_dataset=eval_dataset,
      tokenizer=model.llm_tok,
      data_collator=data_collator,
    )
    trainer._load_from_checkpoint(eval_args.resume_from_sft_checkpoint)
  else:
    model.post_init_prot_model(model_args, device=training_args.device, learn_protst=False)
    model.init_cls_head(label2tok_ids=eval_dataset.label2tok_ids)

  trainer._move_model_to_device(trainer.model, training_args.device)
  trainer.model.eval()
  
  n_prev = 0
  n_chunk = 0
  n_example = 0
  n_correct = 0
  n_answer0 = 0
  avg_loss = 0
  all_preds = []
  all_labels = []
  for step, batch in tqdm(enumerate(eval_loader), total=len(eval_loader)):
    labels = batch["labels"]
    del batch["labels"]
    batch = trainer._prepare_inputs(batch)
    batch.update({"return_dict": True})
    with torch.no_grad():
      logits = trainer.model(**batch).logits.detach().cpu()
    pred = logits.argmax(dim=1)

    n_correct += (pred.reshape(-1).cpu() == labels.reshape(-1)).sum().item()
    n_example += pred.shape[0]
    n_answer0 += (pred.reshape(-1) == 0).sum().item()
    
    if eval_args.task in ["mf", "cc", "bp", "ec"]:
      probs = torch.softmax(logits, dim=1)[:, 1]
      all_preds.append(probs.detach().cpu())
      all_labels.append(labels.detach().cpu())
      # save the accumulated predictions to avoid memory overflow
      if n_example >= eval_args.save_every_n_samples * (n_chunk + 1):
        all_preds = torch.cat(all_preds, dim=0).view(-1)
        all_labels = torch.cat(all_labels, dim=0).view(-1)
        torch.save(all_preds, os.path.join(training_args.output_dir, f"all_preds_{n_chunk}.pt"))
        torch.save(all_labels, os.path.join(training_args.output_dir, f"all_labels_{n_chunk}.pt"))
        st_idx = math.ceil(n_prev / eval_args.n_labels) * eval_args.n_labels - n_prev
        all_labels = all_labels[st_idx:]
        all_labels = all_labels[:all_labels.shape[0] // eval_args.n_labels * eval_args.n_labels]
        all_preds = all_preds[st_idx:]
        all_preds = all_preds[:all_preds.shape[0] // eval_args.n_labels * eval_args.n_labels]
        auprc = area_under_prc(all_preds, all_labels)
        f1_score = f1_max(all_preds.reshape(-1, eval_args.n_labels), all_labels.reshape(-1, eval_args.n_labels))
        print(f'Chunk {n_chunk}: f1: {round(f1_score.item(), 4)}, auprc: {round(auprc.item(), 4)}', flush=True)
        all_preds = []
        all_labels = []
        n_chunk += 1  
        n_prev = n_example
    
    if step % training_args.logging_steps == 0:  
      print(f"step={step} n_example={n_example} n_correct={n_correct} acc={n_correct/n_example} avg_loss={avg_loss/n_example} n_answer0={n_answer0}", flush=True)


  if len(all_preds) > 0:
    all_preds = torch.cat(all_preds, dim=0).view(-1)
    all_labels = torch.cat(all_labels, dim=0).view(-1)
    torch.save(all_preds, os.path.join(training_args.output_dir, f"all_preds_{n_chunk}.pt"))
    torch.save(all_labels, os.path.join(training_args.output_dir, f"all_labels_{n_chunk}.pt"))
    n_chunk += 1
    all_preds = []
    all_labels = []
  
  if eval_args.task in ["mf", "cc", "bp", "ec"]:
    # load all predictions and labels
    for i in range(n_chunk):
      all_preds.append(torch.load(os.path.join(training_args.output_dir, f"all_preds_{i}.pt")))
      all_labels.append(torch.load(os.path.join(training_args.output_dir, f"all_labels_{i}.pt")))

    all_labels = torch.cat(all_labels, dim=0).view(-1)
    all_preds = torch.cat(all_preds, dim=0).view(-1)
    all_labels = all_labels[:all_labels.shape[0] // eval_args.n_labels * eval_args.n_labels]
    all_preds = all_preds[:all_preds.shape[0] // eval_args.n_labels * eval_args.n_labels]
    auprc = area_under_prc(all_preds, all_labels)
    f1_score = f1_max(all_preds.reshape(-1, eval_args.n_labels), all_labels.reshape(-1, eval_args.n_labels))
    print(f'n_example={all_preds.shape[0]} f1: {round(f1_score.item(), 4)}, auprc: {round(auprc.item(), 4)}', flush=True)

  print(f"n_example={n_example} n_correct={n_correct} acc={n_correct/n_example} avg_loss={avg_loss/n_example} n_answer0={n_answer0}", flush=True)


if __name__ == "__main__":
  eval_main()
