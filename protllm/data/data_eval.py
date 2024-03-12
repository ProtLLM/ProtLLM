import copy
import os
import json
import torch
import random
import numpy as np
import esm

from copy import deepcopy
from typing import Iterator, Iterable, Dict, List, Tuple
from dataclasses import dataclass, field
from functools import partial
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from .data_util import general_collate_fn


@dataclass
class ProtLlmExample:
  input_ids: List[int] = field(default_factory=lambda :[])
  labels: List[int] = field(default_factory=lambda :[])
  prot_masks: List[bool] = field(default_factory=lambda :[])
  prot_input_ids_batch: List[List[int]] = field(default_factory=lambda :[])
  prot_residue_mask_batch: List[List[bool]] = field(default_factory=lambda :[])

  def extend(self, item):
    self.input_ids.extend(item.input_ids)
    self.labels.extend(item.labels)
    self.prot_masks.extend(item.prot_masks)
    self.prot_input_ids_batch.extend(item.prot_input_ids_batch)
    self.prot_residue_mask_batch.extend(item.prot_residue_mask_batch)
  
  def prepend_bos(self, bos_token_id, update_labels=True):
    self.input_ids = [bos_token_id] + self.input_ids
    self.prot_masks = [False] + self.prot_masks
    if update_labels:
      self.labels = [-100] + self.labels
  

class Ppi4ProtLlmInContextDataset(Dataset):

  def __init__(self, tok: PreTrainedTokenizer, prot_tok: esm.Alphabet, max_len=1024, prot_max_len=512, data_path=None, n_demo=0, prepend_bos=True) -> None:
    super().__init__()
    with open(data_path) as fp:
      yeast_ds = json.load(fp)
    self.demo_ds = yeast_ds["valid"]
    self.ds = yeast_ds["test"]

    self.n_demo = n_demo
    self.demo_indices = list(range(len(self.demo_ds)))

    self.tok = tok
    self.prot_tok = prot_tok
    self.max_len = max_len
    self.prot_max_len = prot_max_len
    self.prepend_bos = prepend_bos
    
    self._load_ds_prompts()
    
  def _load_ds_prompts(self):
    prompt_templates = {
      "label0_and_label1": [
        "Question: Do ",
        " and ",
        " interact with each other? ",
      ],
      "label1": ["Yes"],
      "label0": ["No"],
      "prot_bos": ["<PROT>"],
      "prot_eos": ["</PROT>"],
    }
    self.prompt_input_ids = {}
    for key, texts in prompt_templates.items():
      id_lists = []
      for text in texts:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        id_lists.append(ids)
      self.prompt_input_ids[key] = id_lists
    self.prot_token_len = len(self.prompt_input_ids["prot_bos"][0]) + 1 + len(self.prompt_input_ids["prot_eos"][0])
    
    print(f"prompts loaded. prot_token_len={self.prot_token_len} prompts={prompt_templates} prompt_input_ids={self.prompt_input_ids}", flush=True)

  def prompted_example(self, protein1: str, protein2: str, label: int, compute_loss=False) -> ProtLlmExample:
    assert label == 0 or label == 1
    input_ids = []
    labels = []
    prot_masks = []
    prot_input_ids_batch = []
    prot_residue_mask_batch = []
    
    def _append_text(ids, compute_loss=False):
      assert isinstance(ids, list)
      input_ids.extend(ids)
      if compute_loss:
        labels.extend(ids)
      else:
        labels.extend([-100] * len(ids))
      prot_masks.extend([False] * len(ids))
    
    def _append_prot_without_bos_eos(_prot):
      input_ids.append(0)
      labels.append(-100)
      prot_masks.append(True)
      prot_input_ids = self.prot_tok.encode(_prot)
      prot_residue_mask = [True] * len(prot_input_ids)
      if self.prot_tok.prepend_bos:
        prot_input_ids = [self.prot_tok.cls_idx] + prot_input_ids
        prot_residue_mask = [False] + prot_residue_mask
      if self.prot_tok.append_eos:
        prot_input_ids = prot_input_ids + [self.prot_tok.eos_idx]
        prot_residue_mask = prot_residue_mask + [False]
      prot_input_ids_batch.append(prot_input_ids)
      prot_residue_mask_batch.append(prot_residue_mask)
    
    def _append_prot(_prot):
      _append_text(self.prompt_input_ids["prot_bos"][0])
      _append_prot_without_bos_eos(_prot)
      _append_text(self.prompt_input_ids["prot_eos"][0])

    common_prompt_key = "label0_and_label1"
    prompt_key = "label1" if label == 1  else "label0"
    # append protein1
    _append_text(self.prompt_input_ids[common_prompt_key][0])
    _append_prot(protein1)
    _append_text(self.prompt_input_ids[common_prompt_key][1])
    _append_prot(protein2)
    _append_text(self.prompt_input_ids[common_prompt_key][2])
    _append_text(self.prompt_input_ids[prompt_key][0], compute_loss=compute_loss)

    return ProtLlmExample(input_ids, labels, prot_masks, prot_input_ids_batch, prot_residue_mask_batch)

  def _load_ds_prompts(self):
    self.cached_prompt_text = {
      k: self.tok(k, add_special_tokens=False)["input_ids"] for k in ["Protein1 and ", "Protein2 can interact with each other.\n", "Protein2 cannot interact with each other.\n"]
    }
  
  def __len__(self):
    return len(self.ds)
  
  def __getitem__(self, index) -> Tuple[List[ProtLlmExample], int]:
    demo_ret = ProtLlmExample()
    if self.n_demo > 0:
      for demo_index in random.sample(self.demo_indices, k=self.n_demo):
        demo_item = self.demo_ds[demo_index]
        demo_example = self.prompted_example(demo_item["protein1"], demo_item["protein2"], demo_item["interaction"])
        demo_ret.extend(demo_example)
    
    all_ret = []
    item = self.ds[index]
    for label_index in range(2):
      ret = copy.deepcopy(demo_ret)
      example = self.prompted_example(item["protein1"], item["protein2"], label_index, compute_loss=True)
      ret.extend(example)
      if self.prepend_bos:
        ret.prepend_bos(self.tok.bos_token_id)
      all_ret.append(ret)

    return all_ret, item["interaction"]
  
  @staticmethod
  def collator(features: List[ProtLlmExample], lm_pad_token_id=None, prot_pad_token_id=None, **unused):
    all_input_ids = []
    all_labels = []
    all_attn_masks = []
    all_prot_masks = []
    all_prot_input_ids = []
    all_prot_attn_masks = []
    all_residue_masks = []
    
    for feature in features:
      all_input_ids.append(feature.input_ids)
      all_attn_masks.append([True] * len(feature.input_ids))
      all_labels.append(feature.labels)
      all_prot_masks.append(feature.prot_masks)
      all_prot_input_ids.extend(feature.prot_input_ids_batch)
      all_prot_attn_masks.extend([[True] * len(prot_input_ids) for prot_input_ids in feature.prot_input_ids_batch])
      all_residue_masks.extend(feature.prot_residue_mask_batch)

    _collate_fn = partial(general_collate_fn, pad_to_multiple_of=1, return_pt=True)
    batch = {
      "input_ids": _collate_fn(all_input_ids, pad_token_id=lm_pad_token_id),
      "labels": _collate_fn(all_labels, pad_token_id=-100),
      "attention_mask": _collate_fn(all_attn_masks, pad_token_id=False),
      "prot_emb_mask": _collate_fn(all_prot_masks, pad_token_id=False),
      "prot_input_ids": _collate_fn(all_prot_input_ids, pad_token_id=prot_pad_token_id),
      "prot_attention_mask": _collate_fn(all_prot_attn_masks, pad_token_id=False),
      "residue_mask": _collate_fn(all_residue_masks, pad_token_id=False).float(),
      "return_loss": True,
    }
    return batch
  

class Ppi4ProtLlmDataset(Dataset):

  def __init__(self, tok: PreTrainedTokenizer, prot_tok: esm.Alphabet, max_len=1024, prot_max_len=512, data_path=None, split="train", prepend_bos=True, **unused) -> None:
    super().__init__()
    with open(data_path) as fp:
      ppi_ds = json.load(fp)
    self.ds = ppi_ds[split]

    self.tok = tok
    self.prot_tok = prot_tok
    self.max_len = max_len
    self.prot_max_len = prot_max_len
    self.prepend_bos = prepend_bos
    
    self._load_ds_prompts()
  
  def __len__(self):
    return len(self.ds)

  def _load_ds_prompts(self):
    prompt_templates = {
      "label0_and_label1": [
        "Question: Do ",
        " and ",
        " interact with each other? ",
      ],
      "prot_bos": ["<PROT>"],
      "prot_eos": ["</PROT>"],
    }
    self.prompt_input_ids = {}
    for key, texts in prompt_templates.items():
      id_lists = []
      for text in texts:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        id_lists.append(ids)
      self.prompt_input_ids[key] = id_lists
    self.prot_token_len = len(self.prompt_input_ids["prot_bos"][0]) + 1 + len(self.prompt_input_ids["prot_eos"][0])
    
    print(f"prompts loaded. prot_token_len={self.prot_token_len} prompts={prompt_templates} prompt_input_ids={self.prompt_input_ids}", flush=True)

    self.label2tok_ids = [
      self.tok("No", add_special_tokens=False)["input_ids"],
      self.tok("Yes", add_special_tokens=False)["input_ids"],
    ]
    assert all(len(tok_ids) == 1 for tok_ids in self.label2tok_ids)
  
  def __getitem__(self, index) -> ProtLlmExample:
    item = self.ds[index]
    protein1 = item["protein1"]
    protein2 = item["protein2"]
    label = item["interaction"]

    assert label == 0 or label == 1
    input_ids = []
    labels = []
    prot_masks = []
    prot_input_ids_batch = []
    prot_residue_mask_batch = []
    
    def _append_text(ids):
      assert isinstance(ids, list)
      input_ids.extend(ids)
      prot_masks.extend([False] * len(ids))
    
    def _append_prot_without_bos_eos(_prot):
      input_ids.append(0)
      prot_masks.append(True)
      prot_input_ids = self.prot_tok.encode(_prot)
      prot_residue_mask = [True] * len(prot_input_ids)
      if self.prot_tok.prepend_bos:
        prot_input_ids = [self.prot_tok.cls_idx] + prot_input_ids
        prot_residue_mask = [False] + prot_residue_mask
      if self.prot_tok.append_eos:
        prot_input_ids = prot_input_ids + [self.prot_tok.eos_idx]
        prot_residue_mask = prot_residue_mask + [False]
      prot_input_ids_batch.append(prot_input_ids)
      prot_residue_mask_batch.append(prot_residue_mask)
    
    def _append_prot(_prot):
      _append_text(self.prompt_input_ids["prot_bos"][0])
      _append_prot_without_bos_eos(_prot)
      _append_text(self.prompt_input_ids["prot_eos"][0])

    prompt_key = "label0_and_label1"
    # append protein1
    _append_text(self.prompt_input_ids[prompt_key][0])
    _append_prot(protein1)
    _append_text(self.prompt_input_ids[prompt_key][1])
    _append_prot(protein2)
    _append_text(self.prompt_input_ids[prompt_key][2])
    labels = [label]

    ret = ProtLlmExample(input_ids, labels, prot_masks, prot_input_ids_batch, prot_residue_mask_batch)
    if self.prepend_bos:
      ret.prepend_bos(self.tok.bos_token_id, update_labels=False)
    assert len(ret.labels) == 1
    return ret
  
  @staticmethod
  def collator(features: List[ProtLlmExample], lm_pad_token_id=None, prot_pad_token_id=None, **unused):
    return Ppi4ProtLlmInContextDataset.collator(features, lm_pad_token_id=lm_pad_token_id, prot_pad_token_id=prot_pad_token_id, **unused)
  

class GO4ProtLlmDataset(Dataset):

  def __init__(self, tok=None, prot_tok=None, max_len=1024, prot_max_len=512, data_path=None, split="train", prepend_bos=True, task="mf", **unused) -> None:
    super().__init__()
    with open(data_path) as fp:
      go_ds = json.load(fp)
      
    self.ds = go_ds[split]

    self.tok = tok
    self.prot_tok = prot_tok
    self.max_len = max_len
    self.prot_max_len = prot_max_len
    self.prepend_bos = prepend_bos
    
    self._load_ds_prompts()
  
  def __len__(self):
    return len(self.ds)

  def _load_ds_prompts(self):
    prompt_templates = {
      "prot_bos": ["<PROT>"],
      "prot_eos": ["</PROT>"],
    }
    self.prompt_input_ids = {}
    for key, texts in prompt_templates.items():
      id_lists = []
      for text in texts:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        id_lists.append(ids)
      self.prompt_input_ids[key] = id_lists
    self.prot_token_len = len(self.prompt_input_ids["prot_bos"][0]) + 1 + len(self.prompt_input_ids["prot_eos"][0])

    self.label2tok_ids = [
      self.tok("No", add_special_tokens=False)["input_ids"],
      self.tok("Yes", add_special_tokens=False)["input_ids"],
    ]
    assert all(len(tok_ids) == 1 for tok_ids in self.label2tok_ids)
  
  def __getitem__(self, index) -> ProtLlmExample:
    item = self.ds[index]
    text = item["text"]
    protein = item["protein"]
    label = item["label"]

    assert label == 0 or label == 1
    input_ids = []
    labels = []
    prot_masks = []
    prot_input_ids_batch = []
    prot_residue_mask_batch = []
    
    def _append_text(ids):
      assert isinstance(ids, list)
      input_ids.extend(ids)
      prot_masks.extend([False] * len(ids))
    
    def _append_prot_without_bos_eos(_prot):
      input_ids.append(0)
      prot_masks.append(True)
      prot_input_ids = self.prot_tok.encode(_prot)
      prot_residue_mask = [True] * len(prot_input_ids)
      if self.prot_tok.prepend_bos:
        prot_input_ids = [self.prot_tok.cls_idx] + prot_input_ids
        prot_residue_mask = [False] + prot_residue_mask
      if self.prot_tok.append_eos:
        prot_input_ids = prot_input_ids + [self.prot_tok.eos_idx]
        prot_residue_mask = prot_residue_mask + [False]
      prot_input_ids_batch.append(prot_input_ids)
      prot_residue_mask_batch.append(prot_residue_mask)
    
    def _append_prot(_prot):
      _append_text(self.prompt_input_ids["prot_bos"][0])
      _append_prot_without_bos_eos(_prot)
      _append_text(self.prompt_input_ids["prot_eos"][0])

    _append_prot(protein)
    _append_text(self.tok(text, add_special_tokens=False)["input_ids"])
    labels = [label]

    ret = ProtLlmExample(input_ids, labels, prot_masks, prot_input_ids_batch, prot_residue_mask_batch)
    if self.prepend_bos:
      ret.prepend_bos(self.tok.bos_token_id, update_labels=False)
    assert len(ret.labels) == 1
    return ret
  
  @staticmethod
  def collator(features: List[ProtLlmExample], lm_pad_token_id=None, prot_pad_token_id=None, **unused):
    return Ppi4ProtLlmInContextDataset.collator(features, lm_pad_token_id=lm_pad_token_id, prot_pad_token_id=prot_pad_token_id, **unused)
  

class ProtLlmRetrievalDataset(Dataset):

  def __init__(self, tok: PreTrainedTokenizer, max_len=1024, data_path=None, prepend_bos=True, **unused) -> None:
    super().__init__()
    with open(data_path) as fp:
      test_data = json.load(fp)
    self.ds = []
    for index, text in test_data["index2text"].items():
      self.ds.append((index, text))
    
    self.tok = tok
    self.max_len = max_len
    self.prepend_bos = prepend_bos
    self.prompt_index = 0

    self._load_ds_prompts()
  
  def __len__(self):
    return len(self.ds)
  
  def _load_ds_prompts(self):
    prompt_templates = {
      "text2protein": [
        " What protein does it describe? ",
        " Can you specify the protein? ",
        " Protein: ",
        " Can you identify the protein? ",
        " What protein is being talked about in the description? ",
      ],
      "protein2text": [
        " Describe the function of the protein. ",
        " Could you describe the function of the protein?",
        " Function: ",
        " Write a brief explanation of the protein: ",
        " Explain the protein: ",
      ],
      "prot_bos": ["<PROT>"],
      "prot_eos": ["</PROT>"],
    }
    self.prompt_input_ids = {}
    for key, texts in prompt_templates.items():
      id_lists = []
      for text in texts:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        id_lists.append(ids)
      self.prompt_input_ids[key] = id_lists
    self.prot_token_len = len(self.prompt_input_ids["prot_bos"][0]) + 1 + len(self.prompt_input_ids["prot_eos"][0])

    print(f"prompts loaded. prot_token_len={self.prot_token_len} prompts={prompt_templates} prompt_input_ids={self.prompt_input_ids}", flush=True)

  def __getitem__(self, index) -> ProtLlmExample:
    cache_index, text = self.ds[index]
    
    input_ids = []

    def _append_text(ids):
      assert isinstance(ids, list)
      input_ids.extend(ids)
    
    # prompt_key
    _append_text(self.tok(text, add_special_tokens=False)["input_ids"])
    _append_text(self.prompt_input_ids["text2protein"][self.prompt_index])
    _append_text(self.prompt_input_ids["prot_bos"][0])

    ret = ProtLlmExample(input_ids)
    if self.prepend_bos:
      ret.input_ids = [self.tok.bos_token_id] + ret.input_ids
    
    return ret, cache_index

  @staticmethod
  def collator(features: List[ProtLlmExample], lm_pad_token_id=None, left_padding=False, **unused):
    
    if left_padding:
      raise NotImplementedError
    
    if len(features) > 1:
      raise NotImplementedError

    all_input_ids = []
    all_attn_masks = []
    all_cache_indices = []

    for protllm_example, cached_index in features:
      input_ids = protllm_example.input_ids
      all_input_ids.append(input_ids)
      all_attn_masks.append([True] * len(input_ids))
      all_cache_indices.append(cached_index)
    
    _collate_fn = partial(general_collate_fn, pad_to_multiple_of=1, return_pt=True)
    batch = {
      "input_ids": _collate_fn(all_input_ids, pad_token_id=lm_pad_token_id),
      "attention_mask": _collate_fn(all_attn_masks, pad_token_id=False),
      "cache_indices": all_cache_indices,
      "return_loss": False,
    }
    return batch