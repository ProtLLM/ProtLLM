import torch
import transformers
import random

from tqdm import tqdm
from functools import partial
from typing import Optional
from dataclasses import dataclass, field
from protllm.projs.protllm.model import ProtLlm, ProtLlmProtstFalcon, ProtLlmNoEncoder, ProtLlmV2ForDownstreamTask, ProtLlmV3ForDownstreamTask, ProtLlmV3ForBinaryCls, ProtLlmV4ForDownstreamTask
# from xnlp.projs.protllm.data import ProtLmDataset, ProtLmDataset4ProtSt
from protllm.projs.protllm.data_eval import Yeast4InContextInferenceDataset, Ppi4ProtLlmInContextDataset
from protllm.projs.protllm.train import MyModelArguments, MyTrainingArguments, Trainer4ProtLlmV3Resume
from peft import get_peft_model, LoraConfig


@dataclass
class MyModelArguments4Eval(MyModelArguments):
  prot_llm_checkpoint: Optional[str] = field(default="")


@dataclass
class MyEvaluationArguments:
  data_path: Optional[str] = field(default="")
  n_demo: Optional[int] = field(default=0)
  eval_seed: Optional[int] = field(default=42)
  eval_protllm_v2: Optional[bool] = field(default=False)


def eval_main():
  parser = transformers.HfArgumentParser((MyModelArguments4Eval, MyEvaluationArguments, MyTrainingArguments))
  model_args, eval_args, training_args = parser.parse_args_into_dataclasses()

  random.seed(eval_args.eval_seed)

  model = ProtLlmV4ForDownstreamTask(model_args, device=training_args.device)
  peft_config = peft_config = LoraConfig(
    inference_mode=False, r=model_args.lora_r, lora_alpha=model_args.lora_alpha, lora_dropout=0.1, bias=model_args.lora_bias, target_modules=model_args.pretrain_target_modules.split(","))
  
  model = get_peft_model(model, peft_config)
  model.init_prot_model(model_args, training_args.device)
  model.config = model._llm_config

  dataset = Ppi4ProtLlmInContextDataset(model.llm_tok, model.prot_tok, data_path=eval_args.data_path, n_demo=eval_args.n_demo)
  data_collator = partial(dataset.collator, lm_pad_token_id=model.llm_tok.eos_token_id, prot_pad_token_id=model.prot_tok.padding_idx)


  data_module = dict(train_dataset=dataset, eval_dataset=dataset)
  print("training_args.report_to=")
  print(training_args.report_to, flush=True)
  training_args.report_to = []
  
  trainer = Trainer4ProtLlmV3Resume(
    model=model, tokenizer=None, args=training_args, data_collator=data_collator, **data_module)

  resume_from_checkpoint = model_args.prot_llm_checkpoint
  trainer._load_from_checkpoint(resume_from_checkpoint)
  trainer._move_model_to_device(trainer.model, training_args.device)

  model.base_model.post_init_prot_model(model_args, device=training_args.device, learn_protst=False)

  n_example = 0
  n_correct = 0
  n_answer0 = 0
  avg_loss = 0
  for item, label in tqdm(data_module["eval_dataset"]):
    min_loss_index = -1
    min_loss = 1000000000
    for prompt_index in range(2):
      batch = data_collator([item[prompt_index]])
      batch = trainer._prepare_inputs(batch)
      batch.update({"return_dict": True})
      trainer.model.eval()
      with torch.no_grad():
        output = trainer.model(**batch)
      loss = output.loss.item()
      if loss < min_loss:
        min_loss = loss
        min_loss_index = prompt_index
    
    n_example += 1
    avg_loss += min_loss
    if min_loss_index == 0:
      n_answer0 += 1
    if min_loss_index == label:
      n_correct += 1
    assert min_loss_index != -1
    print(f"n_example={n_example} n_correct={n_correct} acc={n_correct/n_example} avg_loss={avg_loss/n_example} n_answer0={n_answer0}", flush=True)

  print(f"n_example={n_example} n_correct={n_correct} acc={n_correct/n_example} avg_loss={avg_loss/n_example} n_answer0={n_answer0}", flush=True)


if __name__ == "__main__":
  eval_main()