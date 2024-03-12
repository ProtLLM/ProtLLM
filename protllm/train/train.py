import pathlib
import transformers
import random
import torch
import os

from torch.utils.data import random_split
from functools import partial
from typing import Optional
from dataclasses import dataclass, field
from transformers import Trainer
from transformers.trainer import get_parameter_names, is_sagemaker_mp_enabled, ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from peft import get_peft_model, LoraConfig

from protllm.model.protllm import ProtLlm, ProtLlmForBinaryCls
from protllm.data.data import ProtLlmDataset
from protllm.data.data_eval import Ppi4ProtLlmDataset, GO4ProtLlmDataset
  
  
@dataclass
class MyModelArguments:
  model_name_or_path: Optional[str] = field(default="")
  esm_model_name: Optional[str] = field(default="ESM-2-650M")
  esm_model_file_path: Optional[str] = field(default="")
  protein_model_checkpoint: Optional[str] = field(default="")
  esm_tok_arch_name: Optional[str] = field(default="ESM-1b")
  protein_model_name: Optional[str] = field(default="protst")
  prot_output_size: Optional[int] = field(default=512)
  # sft arguments
  learn_protst: Optional[bool] = field(default=False)
  sft_with_lora: Optional[bool] = field(default=False)
  llm_name_or_path: Optional[str] = field(default="")
  # pretrain lora arguments
  lora_r: Optional[int] = field(default=32)
  lora_alpha: Optional[int] = field(default=64)
  lora_dropout: Optional[float] = field(default=0.1)
  lora_bias: Optional[str] = field(default="none")
  sft_lora_r: Optional[int] = field(default=32)
  sft_lora_alpha: Optional[int] = field(default=64)
  sft_target_modules: Optional[str] = field(default="down_proj,up_proj,q_proj,v_proj,k_proj,o_proj,gate_proj")
  pretrain_target_modules: Optional[str] = field(default="down_proj,up_proj,q_proj,v_proj,k_proj,o_proj,gate_proj")


@dataclass
class MyTrainingArguments(transformers.TrainingArguments):
  main_training_task: Optional[str] = field(default="pretrain")
  exclude_test_prot: Optional[bool] = field(default=False)
  lr_ratio: Optional[float] = field(default=1.0)


@dataclass
class MyDataArguments:
  task: Optional[str] = field(default="ppi", metadata={"help": "ppi, mf, cc, bp, fold"})
  task_type: Optional[str] = field(default="binary", metadata={"help": "binary, multi"})
  train_data_dir: Optional[str] = field(default="/scratch/chizewen/res/string-db")
  max_n_prot: Optional[int] = field(default=5)
  prot_max_len: Optional[int] = field(default=512)
  llm_max_len: Optional[int] = field(default=512)
  text2prot_prob: Optional[float] = field(default=0.2)
  

class Trainer4ProtLlm(transformers.Trainer):

  def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
    super()._load_from_checkpoint(resume_from_checkpoint, model=model)
    if model is None:
      model = self.model
    
    prot2llm_linear_file = os.path.join(resume_from_checkpoint, "prot2llm_linear.bin")
    state_dict = torch.load(prot2llm_linear_file, map_location=torch.device("cpu")).state_dict()
    load_result = model.prot2llm_linear.load_state_dict(state_dict, strict=True)
    self._issue_warnings_after_load(load_result)

    if os.path.exists(os.path.join(resume_from_checkpoint, "llm2prot_linear.bin")):
      llm2prot_linear_file = os.path.join(resume_from_checkpoint, "llm2prot_linear.bin")
      state_dict = torch.load(llm2prot_linear_file, map_location=torch.device("cpu")).state_dict()
      load_result = model.llm2prot_linear.load_state_dict(state_dict, strict=True)
      self._issue_warnings_after_load(load_result)
    
    # load protst if the ckpt exists in the checkpoint folder
    if os.path.exists(os.path.join(resume_from_checkpoint, "protst.bin")):
      protst_file = os.path.join(resume_from_checkpoint, "protst.bin")
      state_dict = torch.load(protst_file, map_location=torch.device("cpu")).state_dict()
      load_result = model.prot_encoder.load_state_dict(state_dict, strict=True)
      self._issue_warnings_after_load(load_result)
  
  def create_optimizer(self):
    
    if is_sagemaker_mp_enabled():
        return super().create_optimizer()
    
    opt_model = self.model

    prot_params = []
    llm_params = []
    prot_params_wo_decay = []
    llm_params_wo_decay = []
    decay_params = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_params = [n for n in decay_params if "bias" not in n]
    for k, v in opt_model.named_parameters():
      if v.requires_grad:
        if "prot_encoder" in k or "prot2llm_linear" in k or "llm2prot_linear" in k:
          if k in decay_params:
            prot_params.append(v)
          else:
            prot_params_wo_decay.append(v)
        else:
          if k in decay_params:
            llm_params.append(v)
          else:
            llm_params_wo_decay.append(v)
    
    if self.args.lr_ratio != 1.0:
      optimizer_grouped_parameters = [
          {"params": prot_params, "lr": self.args.learning_rate * self.args.lr_ratio, "weight_decay": self.args.weight_decay},
          {"params": prot_params_wo_decay, "lr": self.args.learning_rate * self.args.lr_ratio, "weight_decay": 0.0},
          {"params": llm_params, "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay},
          {"params": llm_params_wo_decay, "lr": self.args.learning_rate, "weight_decay": 0.0},
      ]
    else:
      optimizer_grouped_parameters = [
          {"params": prot_params + llm_params, "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay},
          {"params": prot_params_wo_decay + llm_params_wo_decay, "lr": self.args.learning_rate, "weight_decay": 0.0},
      ]
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

    self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer
    

class Callback4ProtLlm(TrainerCallback):
  
  def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs,):
    checkpoint_folder = os.path.join(
      args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
    )       

    peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
    kwargs["model"].save_pretrained(peft_model_path)

    pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
    if os.path.exists(pytorch_model_path):
      os.remove(pytorch_model_path)

    prot2llm_linear_path = os.path.join(checkpoint_folder, "prot2llm_linear.bin")
    torch.save(kwargs["model"].prot2llm_linear, prot2llm_linear_path)

    if hasattr(kwargs["model"], "llm2prot_linear"):
      llm2prot_linear_path = os.path.join(checkpoint_folder, "llm2prot_linear.bin")
      torch.save(kwargs["model"].llm2prot_linear, llm2prot_linear_path)
    
    if kwargs["model"].base_model.learn_protst:
      protst_path = os.path.join(checkpoint_folder, "protst.bin")
      torch.save(kwargs["model"].prot_encoder, protst_path)

    return control
  
  
def pretrain(model_args, data_args, training_args):
  global local_rank
  transformers.trainer_utils.set_seed(training_args.seed)
  local_rank = training_args.local_rank
  
  model = ProtLlm(model_args, device=training_args.device)
  peft_config = LoraConfig(
    inference_mode=False, r=model_args.lora_r, lora_alpha=model_args.lora_alpha, lora_dropout=model_args.lora_dropout, bias=model_args.lora_bias, target_modules=model_args.pretrain_target_modules.split(","))
  model = get_peft_model(model, peft_config)
  model.init_prot_model(data_args, training_args.device)
  model.print_trainable_parameters()
  model.config = model._llm_config
  
  # get the filename of prot cache path
  dataset = ProtLlmDataset(model.llm_tok, prot_tok=1, data_dir=data_args.train_data_dir, max_len=data_args.llm_max_len, max_n_prot=data_args.max_n_prot, text2prot_prob=data_args.text2prot_prob, exculde_test_prot=training_args.exclude_test_prot)
  collator = partial(dataset.collator, lm_pad_token_id=model.llm_tok.eos_token_id)
  torch_rand_gen4dataset_split = torch.Generator().manual_seed(42)
  train_dataset, eval_dataset = random_split(dataset, [len(dataset) - 1000, 1000], generator=torch_rand_gen4dataset_split)

  trainer = Trainer4ProtLlm(
    model=model, tokenizer=model.llm_tok, args=training_args, data_collator=collator, train_dataset=train_dataset, eval_dataset=eval_dataset)
  trainer.add_callback(Callback4ProtLlm)

  print("Model: 👇")
  print(model, flush=True)
  
  if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    trainer.train(resume_from_checkpoint=True)
  else:
    trainer.train()
  trainer.save_state()


def sft(model_args, data_args, training_args):
  global local_rank
  transformers.trainer_utils.set_seed(training_args.seed)
  local_rank = training_args.local_rank

  model = ProtLlmForBinaryCls(model_args, device=training_args.device)
  model.init_prot_model(model_args, device=training_args.device)

  if data_args.task == "ppi":
    train_dataset = Ppi4ProtLlmDataset(model.llm_tok, prot_tok=model.prot_tok, data_path=data_args.train_data_dir, split="train")
    eval_dataset = Ppi4ProtLlmDataset(model.llm_tok, prot_tok=model.prot_tok, data_path=data_args.train_data_dir, split="valid")
  elif data_args.task in ["mf", "bp", "cc", "ec"]:
    train_dataset = GO4ProtLlmDataset(model.llm_tok, prot_tok=model.prot_tok, data_path=data_args.train_data_dir, split="train", task=data_args.task)
    eval_dataset = GO4ProtLlmDataset(model.llm_tok, prot_tok=model.prot_tok, data_path=data_args.train_data_dir, split="valid", task=data_args.task)
  else:
    raise ValueError
  data_collator = partial(train_dataset.collator, lm_pad_token_id=model.llm_tok.eos_token_id, prot_pad_token_id=model.prot_tok.padding_idx)

  model.init_cls_head(label2tok_ids=train_dataset.label2tok_ids)

  # load pre-trained weights: linear and lora weights
  if training_args.resume_from_checkpoint:
    peft_config = LoraConfig(
      inference_mode=False, r=model_args.lora_r, lora_alpha=model_args.lora_alpha, lora_dropout=model_args.lora_dropout, bias=model_args.lora_bias, target_modules=model_args.pretrain_target_modules.split(","))
    model = get_peft_model(model, peft_config)

    trainer = Trainer4ProtLlm(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      tokenizer=model.llm_tok,
      data_collator=data_collator,
    )

    trainer._load_from_checkpoint(training_args.resume_from_checkpoint)
    model = trainer.model.merge_and_unload()

  if model_args.sft_with_lora:
    peft_config = LoraConfig(
      inference_mode=False, r=model_args.sft_lora_r, lora_alpha=model_args.sft_lora_alpha, lora_dropout=model_args.lora_dropout, bias=model_args.lora_bias, target_modules=model_args.sft_target_modules.split(","))
    model = get_peft_model(model, peft_config)
  
  model.post_init_prot_model(model_args, device=training_args.device, learn_protst=model_args.learn_protst)
  model.print_trainable_parameters()
  model.config = model._llm_config

  trainer = Trainer4ProtLlm(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=model.llm_tok,
    data_collator=data_collator,
  )
  trainer.add_callback(Callback4ProtLlm)

  print("Model: 👇")
  print(model)
  
  train_result = trainer.train(resume_from_checkpoint=False)
  
  metrics = train_result.metrics
  metrics["train_samples"] = len(train_dataset)

  trainer.log_metrics("train", metrics)
  trainer.save_metrics("train", metrics)
  trainer.save_state()


def main():
  parser = transformers.HfArgumentParser(
    (MyModelArguments, MyDataArguments, MyTrainingArguments))
  model_args, data_args, training_args = parser.parse_args_into_dataclasses()

  if training_args.main_training_task == "pretrain":
    pretrain(model_args, data_args, training_args)
  elif training_args.main_training_task == "sft":
    sft(model_args, data_args, training_args)
  else:
    raise ValueError
  

if __name__ == "__main__":
  main()