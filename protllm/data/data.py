import os
import json
import torch
import random
import numpy as np
import collections

import esm
from tqdm import tqdm
from typing import List, Optional, Union
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import pool
from torch import distributed
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BertTokenizer
from .data_util import general_collate_fn, general_collate_fn_flatten_1d



class ProtDataset(Dataset):

  CACHE_PREFIX = "ProtDataset"

  def __init__(self, tok: PreTrainedTokenizer, prot_tok: BertTokenizer=None, max_len=1024, prot_max_len=512, data_dir=None, n_process_worker=12, max_n_prot=5, **kwargs) -> None:
    super().__init__()
    self.tok = tok
    if prot_tok is None:
      prot_tok = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    self.prot_tok = prot_tok
    self.max_len = max_len
    self.prot_max_len = prot_max_len
    self.data_dir = data_dir
    self.n_process_worker = n_process_worker
    self.unused_kwargs = kwargs
    self.max_n_prot = max_n_prot
    self.load_ds(max_n_prot=self.max_n_prot)
    
  def _preprocess_node_info(self, node_info):
    ret = {"nid": node_info["nid"]}
    if node_info["protein"] == "": ret["prot_input_ids"] = None
    else:
      try:
        ret["prot_input_ids"] = self.prot_tok.encode(node_info["protein"])
      except KeyError:
        print(f"[W] KeyError while encoding protein seq: {node_info['protein']}", flush=True)
        ret["prot_input_ids"] = None
    if node_info["info"] == "": ret["prot_info_input_ids"] = None
    else: ret["prot_info_input_ids"] = self.tok(node_info["info"], add_special_tokens=False)["input_ids"]
    return ret

  def _preprocess_interleaved_data(self, item):
    # processed_interleaved should be input_ids or nid strings
    processed_interleaved = []

    def _pushback(input_ids_or_nid, prot_name_input_ids=None):
      if isinstance(input_ids_or_nid, str):
        if input_ids_or_nid in self.available_nodes:
          processed_interleaved.append(input_ids_or_nid)
          processed_interleaved.append(prot_name_input_ids)
        else:
          _pushback(prot_name_input_ids)
      elif isinstance(input_ids_or_nid, list):
        if len(processed_interleaved) > 0 and isinstance(processed_interleaved[-1], list):
          processed_interleaved[-1].extend(input_ids_or_nid)
        else:
          processed_interleaved.append(input_ids_or_nid)
      else:
        print(input_ids_or_nid, flush=True)
        raise ValueError

    str_prefix = " " if random.random() < 0.5 else ""
    for interleaved in item:
      for str_or_prot in interleaved:
        if isinstance(str_or_prot, str):
          if "<span" in str_or_prot: break
          str_or_prot = str_or_prot.replace("&quot;", '"')
          _pushback(self.tok(str_prefix + str_or_prot.strip(), add_special_tokens=False)["input_ids"])
          str_prefix = " "
        elif isinstance(str_or_prot, dict):
          if "<span" in str_or_prot["p_text"]: break
          p_text = str_or_prot["p_text"].replace("&quot;", '"')
          _pushback(str_or_prot["nid"], prot_name_input_ids=self.tok(
            p_text.strip(), add_special_tokens=False)["input_ids"])
        else:
          print(str_or_prot, flush=True)
          raise ValueError
        str_prefix = " ... "
    
    return processed_interleaved

  def ds_preprocess_or_load_preprocessed(self):
    processed_data_fn = os.path.join(self.data_dir, f"{self.CACHE_PREFIX}-preprocessed_interleaved.pt")
    if os.path.exists(processed_data_fn):
      pass
    else:
      if not distributed.is_initialized() or (distributed.is_initialized() and distributed.get_rank()) == 0:
        print("Load interleaved data ...", flush=True)
        all_interleaved_data = []
        interleaved_fn = os.path.join(self.data_dir, "interleaved-data.jsonl")
        with open(interleaved_fn) as fp:
          for line in fp:
            interleaved_items = json.loads(line)
            all_interleaved_data.append(interleaved_items)
        
        print("Load node info ...", flush=True)
        all_node_info = []
        # available_nodes for interleaved training
        self.available_nodes = set()
        node_info_fn = os.path.join(self.data_dir, "node_info_all.txt")
        with open(node_info_fn) as fp:
          for line in fp:
            node_info = json.loads(line)
            all_node_info.append(node_info)
            if len(node_info["protein"]) > 0:
              self.available_nodes.add(node_info["nid"])
        
        print("Preprocess node info ...", flush=True)
        
        all_node_info_processed = pool.Pool(self.n_process_worker).map(self._preprocess_node_info, all_node_info)

        print("Node info preprocessed.\n Preprocessing interleaved data ...", flush=True)
                
        all_interleaved_processed = pool.Pool(self.n_process_worker).map(
          self._preprocess_interleaved_data, all_interleaved_data)
        print(f"len(all_interleaved_data)={len(all_interleaved_data)} len(all_interleaved_processed)={len(all_interleaved_processed)} len(node)={len(all_node_info_processed)} processed_interleaved[:2]=")
        print(all_interleaved_processed[:2], flush=True)
        
        torch.save({"all_node_info_processed": all_node_info_processed, "all_interleaved_processed": all_interleaved_processed}, processed_data_fn)

    if distributed.is_initialized():
      print("Wating all processes to sync ...", flush=True)
      distributed.barrier()
    
    print("Load preprocessed data ...", flush=True)
    loaded = torch.load(processed_data_fn)
    all_node_info_processed = loaded["all_node_info_processed"]
    all_interleaved_processed = loaded["all_interleaved_processed"]
    return all_node_info_processed, all_interleaved_processed

  def _load_ds_chunk_prot_text_pairs(self, all_node_info_processed, max_n_prot=5):
    # chunking prot-text pairs
    concatenated_prot_text = []
    tot_n_prot = 0
    temp = {"n_prot": 0, "input_ids_or_nid": []}
    cur_len = 0
    self.nid2prot_input_ids = {}
    perms = np.random.RandomState(seed=42).permutation(len(all_node_info_processed))
    for i in range(len(all_node_info_processed)):
      processed_node_info = all_node_info_processed[perms[i]]
      if processed_node_info["prot_input_ids"]:
        self.nid2prot_input_ids[processed_node_info["nid"]] = processed_node_info["prot_input_ids"]
        if processed_node_info["prot_info_input_ids"] and processed_node_info["prot_input_ids"] and len(processed_node_info["prot_input_ids"]) > 0:
          if cur_len + 1 + len(processed_node_info["prot_info_input_ids"]) > self.max_len:
            if cur_len > 0:
              tot_n_prot += temp["n_prot"]
              concatenated_prot_text.append(temp)
              temp = {"n_prot": 0, "input_ids_or_nid": []}
              cur_len = 0
            temp["n_prot"] += 1
            temp["input_ids_or_nid"].append(processed_node_info["nid"])
            temp["input_ids_or_nid"].append(processed_node_info["prot_info_input_ids"][:self.max_len-1])
            cur_len = 1 + len(temp["input_ids_or_nid"][-1])
          else:
            temp["n_prot"] += 1
            temp["input_ids_or_nid"].append(processed_node_info["nid"])
            temp["input_ids_or_nid"].append(processed_node_info["prot_info_input_ids"])
            cur_len += 1 + len(temp["input_ids_or_nid"][-1])
          if temp["n_prot"] >= max_n_prot:
            tot_n_prot += temp["n_prot"]
            concatenated_prot_text.append(temp)
            temp = {"n_prot": 0, "input_ids_or_nid": []}
            cur_len = 0
    if cur_len > 0:
      tot_n_prot += temp["n_prot"]
      concatenated_prot_text.append(temp)
    
    self.concatenated_prot_text = concatenated_prot_text
    print(f"Chunking prot-text pair data into {len(concatenated_prot_text)} sequences. tot_n_prot = {tot_n_prot} avg_n_prot = {tot_n_prot/len(concatenated_prot_text)}", flush=True)
  
  def _load_ds_chunk_interleaved(self, all_interleaved_processed, max_n_prot=5):
    # chunking into max-len sequences
    concatenated_interleaved = []
    tot_n_prot = 0
    temp = {"n_prot": 0, "input_ids_or_nid": []}
    cur_len = 0
    perms = np.random.RandomState(seed=42).permutation(len(all_interleaved_processed))
    for i in range(len(all_interleaved_processed)):
      processed_interleaved = all_interleaved_processed[perms[i]]

      for nid_or_input_ids in processed_interleaved:
        if isinstance(nid_or_input_ids, str):
          if nid_or_input_ids in self.nid2prot_input_ids and self.nid2prot_input_ids[nid_or_input_ids] and len(self.nid2prot_input_ids[nid_or_input_ids]) > 0:
            if temp["n_prot"] < max_n_prot and cur_len + 1 > self.max_len:
              tot_n_prot += temp["n_prot"]
              # NOTE avoid appending 0-protein example
              if temp["n_prot"] > 0:
                concatenated_interleaved.append(temp)
              temp = {"n_prot": 1, "input_ids_or_nid": [nid_or_input_ids]}
              cur_len = 1
            else:
              temp["n_prot"] += 1
              temp["input_ids_or_nid"].append(nid_or_input_ids)
              cur_len += 1

        elif isinstance(nid_or_input_ids, list):
          if cur_len + len(nid_or_input_ids) > self.max_len:
            remain_len = max(self.max_len - cur_len, 0)
            temp["input_ids_or_nid"].append(nid_or_input_ids[:remain_len])
            tot_n_prot += temp["n_prot"]
            if temp["n_prot"] > 0:
              concatenated_interleaved.append(temp)
            left_input_ids = nid_or_input_ids[remain_len:]
            left_input_ids = left_input_ids[:self.max_len]
            temp = {"n_prot": 0, "input_ids_or_nid": [left_input_ids]}
            cur_len = len(left_input_ids)
          else:
            temp["input_ids_or_nid"].append(nid_or_input_ids)
            cur_len += len(nid_or_input_ids)
        else:
          print(nid_or_input_ids, flush=True)
          raise RuntimeError
    if cur_len == self.max_len:
      tot_n_prot += temp["n_prot"]
      if temp["n_prot"] > 0:
        concatenated_interleaved.append(temp)
    self.concatenated_interleaved = concatenated_interleaved
    print(f"Chunking loaded interleaved data into {len(concatenated_interleaved)}x{self.max_len} sequences. tot_n_prot = {tot_n_prot} avg_n_prot = {tot_n_prot/len(concatenated_interleaved)}", flush=True)
    
  def load_ds(self, max_n_prot = 5):
    all_node_info_processed, all_interleaved_processed = self.ds_preprocess_or_load_preprocessed()

    self._load_ds_chunk_prot_text_pairs(all_node_info_processed, max_n_prot=max_n_prot)
    self._load_ds_chunk_interleaved(all_interleaved_processed, max_n_prot=max_n_prot)
    self.all_data = self.concatenated_interleaved + self.concatenated_prot_text

  def __len__(self):
    return len(self.all_data)
  
  def __getitem__(self, index):
    item = self.all_data[index]
    assert item["n_prot"] > 0
    if item["n_prot"] <= self.max_n_prot:
      prot_ranks = set(range(item["n_prot"]))
    else:
      # randomly select self.max_n_prot prots
      prot_ranks = set(np.random.permutation(item["n_prot"])[:self.max_n_prot])
    
    input_ids = []
    labels = []
    prot_masks = []
    prot_input_ids_batch = []
    prot_residue_mask_batch = []
    cur_prot_rank = 0
    for input_ids_or_nid in item["input_ids_or_nid"]:
      if isinstance(input_ids_or_nid, str):
        if cur_prot_rank in prot_ranks:
          input_ids.append(0)
          labels.append(-100)
          prot_masks.append(True)
          prot_input_ids = self.nid2prot_input_ids[input_ids_or_nid][:self.prot_max_len]
          prot_residue_mask = [True] * len(prot_input_ids)
          if self.prot_tok.prepend_bos:
            prot_input_ids = [self.prot_tok.cls_idx] + prot_input_ids
            prot_residue_mask = [False] + prot_residue_mask
          if self.prot_tok.append_eos:
            prot_input_ids = prot_input_ids + [self.prot_tok.eos_idx]
            prot_residue_mask = prot_residue_mask + [False]
          prot_input_ids_batch.append(prot_input_ids)
          prot_residue_mask_batch.append(prot_residue_mask)
        cur_prot_rank += 1
      elif isinstance(input_ids_or_nid, list):
        input_ids.extend(input_ids_or_nid)
        labels.extend(input_ids_or_nid)
        prot_masks.extend([False] * len(input_ids_or_nid))
      else:
        raise RuntimeError
    
    assert len(input_ids) <= self.max_len, (len(input_ids), item)
    return input_ids, labels, prot_masks, prot_input_ids_batch, prot_residue_mask_batch
  
  @staticmethod
  def collator(features, lm_pad_token_id=None, prot_pad_token_id=None, **unused):
    all_input_ids = []
    all_labels = []
    all_attn_masks = []
    all_prot_masks = []
    all_prot_input_ids = []
    all_prot_attn_masks = []
    all_residue_masks = []

    for feature in features:
      input_ids, labels, prot_masks, prot_input_ids_batch, prot_residue_mask_batch = feature
      all_input_ids.append(input_ids)
      all_attn_masks.append([True] * len(input_ids))
      all_labels.append(labels)
      all_prot_masks.append(prot_masks)
      all_prot_input_ids.extend(prot_input_ids_batch)
      all_prot_attn_masks.extend([[True] * len(prot_input_ids) for prot_input_ids in prot_input_ids_batch])
      all_residue_masks.extend(prot_residue_mask_batch)

    _collate_fn = partial(general_collate_fn, pad_to_multiple_of=1, return_pt=True)
    batch = {
      "input_ids": _collate_fn(all_input_ids, pad_token_id=lm_pad_token_id),
      "labels": _collate_fn(all_labels, pad_token_id=-100),
      "attention_mask": _collate_fn(all_attn_masks, pad_token_id=False),
      "prot_emb_mask": _collate_fn(all_prot_masks, pad_token_id=False),
      "prot_input_ids": _collate_fn(all_prot_input_ids, pad_token_id=prot_pad_token_id),
      "prot_attention_mask": _collate_fn(all_prot_attn_masks, pad_token_id=False),
      "residue_mask": _collate_fn(all_residue_masks, pad_token_id=False).float(),
    }
    return batch


class ProtDataset4ProtStCache(ProtDataset):

  def load_ds(self, max_n_prot=5):
    super().load_ds(max_n_prot=max_n_prot)
    self.all_data = list(self.nid2prot_input_ids.items())
  
  def __getitem__(self, index):
    nid, prot_input_ids = self.all_data[index]
    prot_input_ids = prot_input_ids[:self.prot_max_len]
    prot_residue_mask = [True] * len(prot_input_ids)
    if self.prot_tok.prepend_bos:
      prot_input_ids = [self.prot_tok.cls_idx] + prot_input_ids
      prot_residue_mask = [False] + prot_residue_mask
    if self.prot_tok.append_eos:
      prot_input_ids = prot_input_ids + [self.prot_tok.eos_idx]
      prot_residue_mask = prot_residue_mask + [False]
    return nid, prot_input_ids, prot_residue_mask
  
  @staticmethod
  def collator(features, prot_pad_token_id=None, **unused):
    all_nids = []
    all_prot_input_ids = []
    all_prot_attn_masks = []
    all_residue_masks = []

    for nid, prot_input_ids, residue_mask in features:
      all_nids.append(nid)
      all_prot_input_ids.append(prot_input_ids)
      all_residue_masks.append(residue_mask)
      all_prot_attn_masks.append([True] * len(prot_input_ids))

    _collate_fn = partial(general_collate_fn, pad_to_multiple_of=1, return_pt=True)
    batch = {
      "all_nids": all_nids,
      "input_ids": _collate_fn(all_prot_input_ids, pad_token_id=prot_pad_token_id),
      "attention_mask": _collate_fn(all_prot_attn_masks, pad_token_id=False),
      "residue_mask": _collate_fn(all_residue_masks, pad_token_id=False).float(),
    }
    return batch


@dataclass
class ProtLlmInputSegment:
  is_prot: bool = field(default=False)
  req_lm_loss: bool = field(default=True)
  input_ids: List[int] = field(default_factory=lambda :[])

  def __post_init__(self):
    if self.is_prot:
      assert len(self.input_ids) == 1


@dataclass
class _ProtLlmTempData:
  n_prot: int = field(default=0)
  segments: List[ProtLlmInputSegment] = field(default_factory=lambda :[])


@dataclass
class ProtLlmInputData:
  input_ids: List[int] = field(default_factory=lambda :[])
  labels: List[int] = field(default_factory=lambda :[])
  prot_masks: List[bool] = field(default_factory=lambda :[])
  prot_ids: List[int] = field(default_factory=lambda :[])
  prot_labels: List[int] = field(default_factory=lambda :[])

  def extend(self, input_ids=None, labels=None, prot_masks=None, prot_ids=None, prot_labels=None):
    if input_ids: self.input_ids.extend(input_ids)
    if labels: self.labels.extend(labels)
    if prot_masks: self.prot_masks.extend(prot_masks)
    if prot_ids: self.prot_ids.extend(prot_ids)
    if prot_labels: self.prot_labels.extend(prot_labels)
  
  def prepend_bos(self, bos_token_id):
    # update input_ids, labels, and prot_masks
    self.input_ids = [bos_token_id] + self.input_ids
    self.labels = [-100] + self.labels
    self.prot_masks = [False] + self.prot_masks


class UniProtDataset4ProtStCache(ProtDataset4ProtStCache):

  def __init__(self, prot_tok: esm.Alphabet=None, data_dir: str=None, prot_max_len: int = 1024, **unused):
    self.prot_tok = prot_tok
    if self.prot_tok is None:
      self.prot_tok = esm.Alphabet.from_architecture("ESM-1b")
    self.prot_max_len = prot_max_len
    self.data_dir = data_dir
    self.load_ds()
  
  def load_ds(self):
    # load dataset from the a tsv file in the data_dir, and each line is separated by \t. There are many columns in the file. load the last column as for each line. The return results will be a list.
    self.all_data = {}
    num_tok_failed = 0
    with open(os.path.join(self.data_dir, "uniprot_sprot.tsv")) as fp:
      # rewrite the above code to skip the first line of tsv with an enumerate, the other code remain the same
      for i, line in tqdm(enumerate(fp)):
        if i == 0:
          continue
        items = line.split("\t")
        try:
          encoded_prot_seq = self.prot_tok.encode(items[-1].strip())
          # self.all_data.add((items[0].strip(), encoded_prot_seq))
          self.all_data[items[0].strip()] = encoded_prot_seq
        except Exception as e:
          # print what exception it is
          print(line, flush=True)
          print(e, flush=True)
          num_tok_failed += 1

    
    self.all_data = [(k, v) for k, v in self.all_data.items()]
    print(f"num_tok_failed = {num_tok_failed}", flush=True)
    print(self.all_data[:2], flush=True)
  
  def __len__(self):
    return len(self.all_data)


class MolInstructionDataset4ProtStCache(Dataset):

  def __init__(self, prot_tok: esm.Alphabet=None, data_dir: str=None, prot_max_len: int = 1024, **unused):
    self.prot_tok = prot_tok
    if self.prot_tok is None:
      self.prot_tok = esm.Alphabet.from_architecture("ESM-1b")
    self.prot_max_len = prot_max_len
    self.data_dir = os.path.join(data_dir, "mol-instructions")

    all_prot_seq_fn = os.path.join(self.data_dir, "all_prot_seq.json")
    if os.path.exists(all_prot_seq_fn):
      self.load_ds(all_prot_seq_fn, load_from_prot_seq=True)
    else:
      self.load_ds(all_prot_seq_fn, load_from_prot_seq=False, save=True)
  
  def load_ds(self, all_prot_seq_fn, load_from_prot_seq=False, save=False):
    if load_from_prot_seq:
      print(f"Load molinstruct data from {all_prot_seq_fn}...", flush=True)
      with open(all_prot_seq_fn) as fp:
        all_prot_seq = json.load(fp)
      self.all_data = all_prot_seq
      return

    self.all_data = None
    all_prot_set = set()
    instruction_dir = os.path.join(self.data_dir, "Protein-oriented_Instructions")
    for fn in os.listdir(instruction_dir):
      if fn == "protein_design.json":
        with open(os.path.join(instruction_dir, fn)) as fp:
          data = json.load(fp)
          for item in data:
            # all_prot_set.add(item["input"].strip("`").strip("\n"))
            prot_pos = item["output"].find("```")
            item_output = item["output"][prot_pos:].strip("`").strip("\n")
            all_prot_set.add(item_output)
      else:
        with open(os.path.join(instruction_dir, fn)) as fp:
          data = json.load(fp)
          for item in data:
            all_prot_set.add(item["input"].strip("`").strip("\n"))
    self.all_data = list(all_prot_set)

    if save:
      with open(all_prot_seq_fn, "w") as fp:
        json.dump(self.all_data, fp)
  
  def __len__(self):
    return len(self.all_data)
  
  def __getitem__(self, index):
    prot_input_ids = self.prot_tok.encode(self.all_data[index])
    prot_input_ids = prot_input_ids[:self.prot_max_len]
    prot_residue_mask = [True] * len(prot_input_ids)
    if self.prot_tok.prepend_bos:
      prot_input_ids = [self.prot_tok.cls_idx] + prot_input_ids
      prot_residue_mask = [False] + prot_residue_mask
    if self.prot_tok.append_eos:
      prot_input_ids = prot_input_ids + [self.prot_tok.eos_idx]
      prot_residue_mask = prot_residue_mask + [False]
    return index, prot_input_ids, prot_residue_mask
  
  @staticmethod
  def collator(features, prot_pad_token_id=None, **unused):
    return ProtDataset4ProtStCache.collator(features, prot_pad_token_id=prot_pad_token_id, **unused)



class ProtLlmDataset(ProtDataset):
  def __init__(self, tok: PreTrainedTokenizer, prot_tok=None, max_len=1024, prot_max_len=512, data_dir=None, n_process_worker=12, max_n_prot=5, text2prot_prob=0.15, exculde_test_prot=False, **kwargs) -> None:
    self.text2prot_prob = text2prot_prob
    self.exculde_test_prot = exculde_test_prot
    super().__init__(tok, prot_tok, max_len, prot_max_len, data_dir, n_process_worker, max_n_prot, **kwargs)
    
  def _roll_prot_text_prompt(self):
    is_text2protein = np.random.rand() < self.text2prot_prob
    prompts = self.prompt_input_ids["text2protein"] if is_text2protein else self.prompt_input_ids["protein2text"] 
    prompt_index = np.random.randint(0, len(prompts))
    prompt_input_ids = prompts[prompt_index]
    return is_text2protein, prompt_input_ids
  
  def _prot_bos_segment(self, req_lm_loss=False):
    return ProtLlmInputSegment(input_ids=self.prompt_input_ids["prot_bos"][0], req_lm_loss=req_lm_loss)

  def _prot_eos_segment(self, req_lm_loss=False):
    return ProtLlmInputSegment(input_ids=self.prompt_input_ids["prot_eos"][0], req_lm_loss=req_lm_loss)

  def _load_ds_chunk_prot_text_pairs_impl(self, all_node_info_processed, **unused):
    concatenated_prot_text = []
    cur_len = tot_n_prot = 0
    temp = _ProtLlmTempData()
    perms = np.random.RandomState(seed=42).permutation(len(all_node_info_processed))
    for i in range(len(all_node_info_processed)):
      processed_node_info = all_node_info_processed[perms[i]]
      if processed_node_info["nid"] in self.nid2prot_input_id and processed_node_info["prot_info_input_ids"]:
        prot_id = self.nid2prot_input_id[processed_node_info["nid"]]
        is_text2protein, prompt_input_ids = self._roll_prot_text_prompt()
        if cur_len + self.prot_token_len + len(prompt_input_ids) + len(processed_node_info["prot_info_input_ids"]) > self.max_len:
          if cur_len > 0:
            assert cur_len <= self.max_len, cur_len
            tot_n_prot += temp.n_prot
            concatenated_prot_text.append(temp)
            temp = _ProtLlmTempData()
            cur_len = 0
        if self.prot_token_len + len(prompt_input_ids) + len(processed_node_info["prot_info_input_ids"]) > self.max_len:
          trunc_pos = self.max_len - self.prot_token_len - len(prompt_input_ids)
          assert trunc_pos > 0
        else:
          trunc_pos = len(processed_node_info["prot_info_input_ids"])
        prot_info_input_ids = processed_node_info["prot_info_input_ids"][:trunc_pos]

        # append data concatenated_prot_text
        temp.n_prot += 1
        if is_text2protein:
          temp.segments.append(ProtLlmInputSegment(req_lm_loss=False, input_ids=prot_info_input_ids))
          temp.segments.append(ProtLlmInputSegment(req_lm_loss=False, input_ids=prompt_input_ids))
          temp.segments.append(self._prot_bos_segment(req_lm_loss=True))
          temp.segments.append(ProtLlmInputSegment(is_prot=True, req_lm_loss=True, input_ids=[prot_id]))
          temp.segments.append(self._prot_eos_segment(req_lm_loss=True))
          cur_len += self.prot_token_len + len(prompt_input_ids) + len(prot_info_input_ids)
        else:
          temp.segments.append(self._prot_bos_segment(req_lm_loss=False))
          temp.segments.append(ProtLlmInputSegment(is_prot=True, req_lm_loss=False, input_ids=[prot_id]))
          temp.segments.append(self._prot_eos_segment(req_lm_loss=False))
          temp.segments.append(ProtLlmInputSegment(req_lm_loss=False, input_ids=prompt_input_ids))
          temp.segments.append(ProtLlmInputSegment(req_lm_loss=True, input_ids=prot_info_input_ids))
          cur_len += self.prot_token_len + len(prompt_input_ids) + len(prot_info_input_ids)
        
    if cur_len > 0:
      tot_n_prot += temp.n_prot
      concatenated_prot_text.append(temp)
    
    print(f"Chunking prot-text pair data into {len(concatenated_prot_text)} sequences. tot_n_prot = {tot_n_prot} avg_n_prot = {tot_n_prot/len(concatenated_prot_text)}", flush=True)

    return concatenated_prot_text

  def _load_ds_chunk_prot_text_pairs(self, all_node_info_processed, **unused):
    concatenated_prot_text = self._load_ds_chunk_prot_text_pairs_impl(all_node_info_processed, **unused)
    self.concatenated_prot_text = concatenated_prot_text

  def _load_ds_chunk_interleaved(self, all_interleaved_processed, **unused):
    concatenated_interleaved = []
    cur_len = tot_n_prot = 0
    temp = _ProtLlmTempData()
    perms = np.random.RandomState(seed=42).permutation(len(all_interleaved_processed))
    for i in range(len(all_interleaved_processed)):
      processed_interleaved = all_interleaved_processed[perms[i]]

      for nid_or_input_ids in processed_interleaved:
        if isinstance(nid_or_input_ids, str):
          if nid_or_input_ids in self.nid2prot_input_id:
            if cur_len + self.prot_token_len > self.max_len:
              # if temp.n_prot > 0:
              tot_n_prot += temp.n_prot
              concatenated_interleaved.append(temp)
              temp = _ProtLlmTempData()
              cur_len = 0
            temp.n_prot += 1
            prot_id = self.nid2prot_input_id[nid_or_input_ids]
            temp.segments.append(self._prot_bos_segment(req_lm_loss=True))
            temp.segments.append(ProtLlmInputSegment(is_prot=True, req_lm_loss=True, input_ids=[prot_id]))
            temp.segments.append(self._prot_eos_segment(req_lm_loss=True))
            cur_len += self.prot_token_len
        elif isinstance(nid_or_input_ids, list):
          if cur_len + len(nid_or_input_ids) > self.max_len:
            remain_len = max(self.max_len - cur_len, 0)
            temp.segments.append(ProtLlmInputSegment(req_lm_loss=True, input_ids=nid_or_input_ids[:remain_len]))
            tot_n_prot += temp.n_prot
            concatenated_interleaved.append(temp)
            left_input_ids = nid_or_input_ids[remain_len:]
            left_input_ids = left_input_ids[:self.max_len]
            temp = _ProtLlmTempData()
            temp.segments.append(ProtLlmInputSegment(req_lm_loss=True, input_ids=left_input_ids))
            cur_len = len(left_input_ids)
          else:
            temp.segments.append(ProtLlmInputSegment(req_lm_loss=True, input_ids=nid_or_input_ids))
            cur_len += len(nid_or_input_ids)
        else:
          print(nid_or_input_ids, flush=True)
          raise RuntimeError
    if cur_len >= 0.3 * self.max_len:
      tot_n_prot += temp.n_prot
      concatenated_interleaved.append(temp)
    
    self.concatenated_interleaved = concatenated_interleaved
    print(f"Chunking loaded interleaved data into {len(concatenated_interleaved)}x{self.max_len} sequences. tot_n_prot = {tot_n_prot} avg_n_prot = {tot_n_prot/len(concatenated_interleaved)}", flush=True)

  def __getitem__(self, index) -> ProtLlmInputData:
    item: _ProtLlmTempData = self.all_data[index]
    ret = ProtLlmInputData()
    temp: ProtLlmInputSegment = None
    for temp in item.segments:
      if temp.is_prot:
        ret.extend(
          input_ids=[0],
          labels=[-100],
          prot_masks=[True],
          prot_ids=temp.input_ids,
          prot_labels=temp.input_ids if temp.req_lm_loss else [-100],)
      else:
        _len = len(temp.input_ids)
        ret.extend(
          input_ids=temp.input_ids,
          labels=temp.input_ids if temp.req_lm_loss else [-100] * _len,
          prot_masks=[False] * _len,)

    assert len(ret.input_ids) <= self.max_len, len(ret.input_ids)
    ret.prepend_bos(self.tok.bos_token_id)
    return ret
  
  @staticmethod
  def collator(features, lm_pad_token_id=None, **unused):
    all_input_ids = []
    all_labels = []
    all_attn_masks = []
    all_prot_masks = []
    all_prot_ids = []
    all_prot_labels = []

    feature: ProtLlmInputData = None
    for feature in features:
      all_input_ids.append(feature.input_ids)
      all_attn_masks.append([True] * len(feature.input_ids))
      all_labels.append(feature.labels)
      all_prot_masks.append(feature.prot_masks)
      all_prot_ids.append(feature.prot_ids)
      all_prot_labels.append(feature.prot_labels)
    
    _collate_fn = partial(general_collate_fn, pad_to_multiple_of=1, return_pt=True)
    _collate_fn_1d = partial(general_collate_fn_flatten_1d,  return_pt=True)

    batched_prot_ids = _collate_fn_1d(all_prot_ids)
    batch = {
      "input_ids": _collate_fn(all_input_ids, pad_token_id=lm_pad_token_id),
      "labels": _collate_fn(all_labels, pad_token_id=-100),
      "attention_mask": _collate_fn(all_attn_masks, pad_token_id=False),
      "prot_emb_mask": _collate_fn(all_prot_masks, pad_token_id=False),
      "prot_ids": None if len(batched_prot_ids) == 0 else batched_prot_ids,
      "prot_labels": _collate_fn_1d(all_prot_labels),
      "return_loss": True,
    }
    return batch

  def load_ds(self, **unused):
    all_node_info_processed, all_interleaved_processed = self.ds_preprocess_or_load_preprocessed()
    all_uniprot_processed = self.ds_preprocess_or_load_preprocessed_uniprot()
    all_molinstruct_processed = self.ds_preprocess_or_load_preprocessed_molinstruct()

    self._load_ds_prompts()
    self._load_ds_prot_dict(exculde_test_prot=self.exculde_test_prot)
    self._load_ds_chunk_prot_text_pairs(all_node_info_processed)
    self._load_ds_chunk_interleaved(all_interleaved_processed)

    self._load_ds_uniprot(all_uniprot_processed)
    self._load_ds_molinstruct(all_molinstruct_processed)

    self.all_data = self.concatenated_interleaved + self.concatenated_prot_text + self.concatenated_uniprot + self.concatenated_molinstruct
    
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
    
    print(f"Prompts loaded.", flush=True)
    
  # implement preprocessed uniprot data. 
  def ds_preprocess_or_load_preprocessed_uniprot(self):
    processed_data_fn = os.path.join(self.data_dir, f"{self.CACHE_PREFIX}-preprocessed_uniprot.pt")
    if os.path.exists(processed_data_fn):
      pass
    else:
      if not distributed.is_initialized() or (distributed.is_initialized() and distributed.get_rank()) == 0:
        print("Load uniprot data ...", flush=True)
        all_uniprot_data = []
        uniprot_fn = os.path.join(self.data_dir, "uniprot_sprot.tsv")
        with open(uniprot_fn) as fp:
          # rewirte to skip the first line of tsv with an enumerate, the other code remain the same
          for i, line in enumerate(fp):
            if i == 0:
              continue
            uniprot_items = line.split("\t")
            all_uniprot_data.append(uniprot_items)
        
        # unique lines if ajacent lines have over 90% word overlap in uniprot_items[2] (to avoid duplicated data)
        all_uniprot_data_unique = []
        # use countable set, e.g., if a word appears twice in a line, it will be counted as 2, so directly using set() doesn't work
        prev_words = collections.Counter()
        for uniprot_items in all_uniprot_data:
          cur_words = collections.Counter(uniprot_items[2].split())
          if len(cur_words) == 0 or len(prev_words) == 0:
            # filter example without words
            if len(cur_words) == 0:
              pass
            else:
              all_uniprot_data_unique.append(uniprot_items)
          else:
            _sum = sum((cur_words & prev_words).values())
            overlap_prev = _sum / sum(prev_words.values())
            overlap_cur = _sum / sum(cur_words.values())
            if overlap_prev < 0.9 or overlap_cur < 0.9:
              all_uniprot_data_unique.append(uniprot_items)
          prev_words = cur_words

        # print the statistics of the data before and after removing duplicated data
        print(f"len(all_uniprot_data) = {len(all_uniprot_data)}, all_uniprot_data_unique = {len(all_uniprot_data_unique)}", flush=True)
        
        print("Preprocess uniprot data ...", flush=True)
        all_uniprot_processed = pool.Pool(self.n_process_worker).map(self._preprocess_uniprot, all_uniprot_data_unique)

        print("Uniprot data preprocessed.", flush=True)
                
        torch.save({"all_uniprot_processed": all_uniprot_processed}, processed_data_fn)

    if distributed.is_initialized():
      print("Wating all processes to sync ...", flush=True)
      distributed.barrier()
    
    print("Load preprocessed uniprot data ...", flush=True)
    loaded = torch.load(processed_data_fn)
    all_uniprot_processed = loaded["all_uniprot_processed"]
    return all_uniprot_processed
  
  def _preprocess_uniprot(self, uniprot_items):
    ret = {"nid": "uniprot-" + uniprot_items[0].strip()}
    if uniprot_items[-1].strip() == "": ret["prot_input_ids"] = None
    else:
      try:
        ret["prot_input_ids"] = self.prot_tok.encode(uniprot_items[-1].strip())
      except KeyError:
        print(f"[W] KeyError while encoding protein seq: {uniprot_items[-1].strip()}", flush=True)
        ret["prot_input_ids"] = None
    
    text = uniprot_items[2].strip() + uniprot_items[3].strip()
    if text == "": ret["prot_info_input_ids"] = None
    else: ret["prot_info_input_ids"] = self.tok(text, add_special_tokens=False)["input_ids"]
    return ret

  def _load_ds_uniprot(self, all_uniprot_processed, max_n_prot=5):
    concatenated_prot_text = self._load_ds_chunk_prot_text_pairs_impl(all_uniprot_processed, max_n_prot=max_n_prot)
    self.concatenated_uniprot = concatenated_prot_text
  
  def prot_seq_to_nids_main(self, prot_seqs: List[str]):
    node_info_fn = os.path.join(self.data_dir, "node_info_all.txt")

    # load protein seq to nid dict from node_info_fn
    string_prot_to_nid = collections.defaultdict(list)
    with open(node_info_fn) as fp:
      for line in fp:
        node_info = json.loads(line)
        if len(node_info["protein"]) > 0:
          string_prot_to_nid[node_info["protein"]].append("string-" + node_info["nid"].strip())
    
    # load protein seq to nid dict from uniprot_fn
    uniprot_fn = os.path.join(self.data_dir, "uniprot_sprot.tsv")
    uniprot_prot_to_nid = collections.defaultdict(list)
    with open(uniprot_fn) as fp:
      for i, line in enumerate(fp):
        if i == 0:
          continue
        uniprot_items = line.split("\t")
        uniprot_prot_to_nid[uniprot_items[-1].strip()].append("uniprot-" + uniprot_items[0].strip())
    
    results = []
    for prot in prot_seqs:
      results.extend(string_prot_to_nid[prot])
      results.extend(uniprot_prot_to_nid[prot])
    
    # save results
    exculded_prot_id_fn = os.path.join(self.data_dir, "excluded_prot_ids.json")
    with open(exculded_prot_id_fn, "w") as fp:
      json.dump(results, fp)

  
  def _load_ds_prot_dict(self, exculde_test_prot=False):
    # replace the hard code above with the following code
    if exculde_test_prot:
      exculded_prot_id_fn = os.path.join(self.data_dir, "excluded_prot_ids.json")
      with open(exculded_prot_id_fn) as fp:
        excluded_prot_ids = json.load(fp)
    else:
      excluded_prot_ids = []
    
    prot_dict_path = os.path.join(self.data_dir, "protein_cache.pt")

    excluded_prot_ids = set(excluded_prot_ids)
    print(f"{len(excluded_prot_ids)} excluded_prot_ids loaded.", flush=True)

    loaded = torch.load(prot_dict_path, map_location="cpu")
    self.nid2prot_input_id = {}
    for nid, index in loaded["nid2index"].items():
      if nid in excluded_prot_ids:
        raise RuntimeError(f"nid {nid} should be excluded")
      if nid.startswith("uniprot-"):
        self.nid2prot_input_id[nid] = index
      elif nid.startswith("string-"):
        self.nid2prot_input_id[nid[len("string-"):]] = index
      # consider molinstruct data
      elif nid.startswith("molinstruct-"):
        self.nid2prot_input_id[nid] = index
      else:
        raise RuntimeError(f"nid {nid} not recognized")
    
    # assert tot_excluded_prot_ids == 0
    print(f"nid2prot_input_id loaded. len={len(self.nid2prot_input_id)}", flush=True)
  
  def _truncate_prot_dict(self, cache_fn=None, truncated_cache_fn=None):
    prot_dict_path = os.path.join(self.data_dir, cache_fn)
    loaded = torch.load(prot_dict_path, map_location="cpu")

    nid2index = loaded["nid2index"]
    outputs: torch.Tesnor = loaded["all_outputs"]
    prot_id2nids = collections.defaultdict(list)
    max_len_nids = 0
    for nid, index in nid2index.items():
      prot_id2nids[index].append(nid)
      max_len_nids = max(max_len_nids, len(prot_id2nids[index]))
    print(f"max_len_nids = {max_len_nids}", flush=True)
    index2new_index = {}

    output_trucated_prot_dict_path = os.path.join(self.data_dir, truncated_cache_fn)
    assert not os.path.exists(output_trucated_prot_dict_path), f"{output_trucated_prot_dict_path} already exists"
    truncated_nid2index = {}
    truncated_outputs = []

    # go through all data, and only keep the nid that is in the all_data
    item: ProtLlmInputData = None
    for item in self:
      for prot_id in item.prot_ids:
        nids = prot_id2nids[prot_id]
        if prot_id not in index2new_index:
          index2new_index[prot_id] = len(index2new_index)
          truncated_outputs.append(outputs[prot_id])
        for nid in nids:
          if nid not in truncated_nid2index:
            truncated_nid2index[nid] = index2new_index[prot_id]
          else:
            assert truncated_nid2index[nid] == index2new_index[prot_id]

    # print the statistics of the data before and after removing duplicated data
    print(f"len(nid2index) = {len(nid2index)}, truncated_nid2index = {len(truncated_nid2index)}", flush=True)

    # save truncated_nid2index and corresponding outputs
    truncated_outputs = torch.stack(truncated_outputs)

    # verify the truncated_outputs
    for nid, index in truncated_nid2index.items():
      # assert torch.allclose(truncated_outputs[index], outputs[nid2index[nid]])
      assert truncated_outputs[index].equal(outputs[nid2index[nid]])
    
    # save the truncated data
    torch.save({"nid2index": truncated_nid2index, "all_outputs": truncated_outputs}, output_trucated_prot_dict_path)
  
  def _truncate_prot_dict_exclude_test_prot(self, truncated_cache_fn=None, output_cache_fn=None):
    if truncated_cache_fn is None:
      truncated_cache_fn = "truncated_combined_string_uniprot_protst-esm2-cache.pt"
    if output_cache_fn is None:
      output_cache_fn = "truncated_combined_string_uniprot_protst_exclude_test_protst-esm2-cache.pt"
    assert not os.path.exists(os.path.join(self.data_dir, output_cache_fn)), f"{output_cache_fn} already exists"

    exculded_prot_id_fn = os.path.join(self.data_dir, "excluded_prot_ids.json")
    if os.path.exists(exculded_prot_id_fn):
      with open(exculded_prot_id_fn) as fp:
        excluded_prot_ids = json.load(fp)
    else:
      excluded_prot_ids = []
    excluded_prot_ids = set(excluded_prot_ids)
    print(f"{len(excluded_prot_ids)} excluded_prot_ids loaded.", flush=True)

    prot_dict_path = os.path.join(self.data_dir, truncated_cache_fn)
    loaded = torch.load(prot_dict_path, map_location="cpu")
    original_all_outputs = loaded["all_outputs"]
    original_nid2index = loaded["nid2index"]

    original_index2new_index = {}
    ret_nid2index = {}
    ret_all_outputs = []
    for nid, index in original_nid2index.items():
      if nid in excluded_prot_ids:
        pass
      else:
        if index not in original_index2new_index:
          original_index2new_index[index] = len(original_index2new_index)
          ret_all_outputs.append(original_all_outputs[index])
        new_index = original_index2new_index[index]
        ret_nid2index[nid] = new_index
        # ret_all_outputs.append(original_all_outputs[index])
    
    ret_all_outputs = torch.stack(ret_all_outputs)

    # verify the truncated_outputs
    for nid, index in ret_nid2index.items():
      assert ret_all_outputs[index].equal(original_all_outputs[original_nid2index[nid]]), (ret_all_outputs[index], original_all_outputs[original_nid2index[nid]])

    # save
    output_trucated_prot_dict_path = os.path.join(self.data_dir, output_cache_fn)

    torch.save({"nid2index": ret_nid2index, "all_outputs": ret_all_outputs}, output_trucated_prot_dict_path)
  
  def _preprocess_molinstruct(self, molinstruct_item, text_to_prot=False):

    if text_to_prot:
      output_text_and_prot = molinstruct_item["output"]
      _pos = output_text_and_prot.find("\n```\n")
      assert _pos != -1
      output_text = output_text_and_prot[:_pos]
      prot_seq = output_text_and_prot[_pos + 1:].strip("`").strip("\n")
      ret = {"nid": "molinstruct-" + str(self._molinstruct_prot2id[prot_seq])}

      ret["answer_input_ids"] = self.tok(output_text, add_special_tokens=False)["input_ids"]
      text = molinstruct_item["instruction"] + "\n" + molinstruct_item["input"]
      ret["prot_info_input_ids"] = self.tok(text, add_special_tokens=False)["input_ids"]
      ret["text_to_prot"] = True
    else:
      prot_seq = molinstruct_item["input"].strip("`").strip("\n")
      ret = {"nid": "molinstruct-" + str(self._molinstruct_prot2id[prot_seq])}
      ret["prot_info_input_ids"] = self.tok(molinstruct_item["instruction"], add_special_tokens=False)["input_ids"]
      ret["answer_input_ids"] = self.tok(molinstruct_item["output"], add_special_tokens=False)["input_ids"]
      ret["text_to_prot"] = False
    return ret

  def ds_preprocess_or_load_preprocessed_molinstruct(self):
    preprocessed_data_fn = os.path.join(self.data_dir, f"{self.CACHE_PREFIX}-preprocessed_molinstruct.pt")
    if os.path.exists(preprocessed_data_fn):
      pass
    else:
      if not distributed.is_initialized() or (distributed.is_initialized() and distributed.get_rank()) == 0:
        print("Load molinstruct data ...", flush=True)

        self._molinstruct_prot2id = {}
        all_prot_seq_fn = os.path.join(self.data_dir, "mol-instructions","all_prot_seq.json")
        with open(all_prot_seq_fn) as fp:
          all_prot_seq = json.load(fp)
        for i, prot_seq in enumerate(all_prot_seq):
          self._molinstruct_prot2id[prot_seq] = i

        all_molinstruct_data = []
        molinstruct_dir = os.path.join(self.data_dir, "mol-instructions", "Protein-oriented_Instructions")
        for fn in os.listdir(molinstruct_dir):
          if fn == "protein_design.json":
            continue
          with open(os.path.join(molinstruct_dir, fn)) as fp:
            data = json.load(fp)
            for item in data:
              all_molinstruct_data.append(item)
        
        print("Preprocess molinstruct data ...", flush=True)
        _preprocess_fn = partial(self._preprocess_molinstruct, text_to_prot=False)
        all_molinstruct_processed = pool.Pool(self.n_process_worker).map(_preprocess_fn, all_molinstruct_data)

        all_molinstruct_data_text_to_prot = []
        fn = os.path.join(molinstruct_dir, "protein_design.json")
        with open(fn) as fp:
          data = json.load(fp)
          for item in data:
            all_molinstruct_data_text_to_prot.append(item)
        
        print("Preprocess molinstruct data (text_to_prot) ...", flush=True)
        _preprocess_fn = partial(self._preprocess_molinstruct, text_to_prot=True)
        all_molinstruct_processed_text_to_prot = pool.Pool(self.n_process_worker).map(_preprocess_fn, all_molinstruct_data_text_to_prot)

        print("Molinstruct data preprocessed.", flush=True)
        
        all_molinstruct_processed = all_molinstruct_processed + all_molinstruct_processed_text_to_prot
        torch.save({"all_molinstruct_processed": all_molinstruct_processed}, preprocessed_data_fn)
    
    if distributed.is_initialized():
      print("Wating all processes to sync ...", flush=True)
      distributed.barrier()
    
    print("Load preprocessed molinstruct data ...", flush=True)
    loaded = torch.load(preprocessed_data_fn)
    all_molinstruct_processed = loaded["all_molinstruct_processed"]
    return all_molinstruct_processed

  def _load_ds_molinstruct(self, all_molinstruct_processed):
    concatenated_molinstruct = []
    cur_len = tot_n_prot = 0
    temp = _ProtLlmTempData()
    perms = np.random.RandomState(seed=42).permutation(len(all_molinstruct_processed))
    exceed_max_len = 0
    skip_nid_not_found = 0
    for i in range(len(all_molinstruct_processed)):
      processed_molinstruct = all_molinstruct_processed[perms[i]]
      if processed_molinstruct["nid"] not in self.nid2prot_input_id:
        skip_nid_not_found += 1
        continue
      prot_id = self.nid2prot_input_id[processed_molinstruct["nid"]]
      if cur_len + self.prot_token_len + len(processed_molinstruct["prot_info_input_ids"]) + len(processed_molinstruct["answer_input_ids"]) > self.max_len:
        if cur_len > 0:
          assert cur_len <= self.max_len, cur_len
          tot_n_prot += temp.n_prot
          concatenated_molinstruct.append(temp)
          temp = _ProtLlmTempData()
          cur_len = 0
      if self.prot_token_len + len(processed_molinstruct["prot_info_input_ids"]) + len(processed_molinstruct["answer_input_ids"]) > self.max_len:
        exceed_max_len += 1
        continue
      
      temp.n_prot += 1
      prot_info_input_ids = processed_molinstruct["prot_info_input_ids"]
      answer_input_ids = processed_molinstruct["answer_input_ids"]

      if processed_molinstruct["text_to_prot"]:
        temp.segments.append(ProtLlmInputSegment(req_lm_loss=False, input_ids=prot_info_input_ids))
        temp.segments.append(ProtLlmInputSegment(req_lm_loss=True, input_ids=answer_input_ids))
        temp.segments.append(self._prot_bos_segment(req_lm_loss=True))
        temp.segments.append(ProtLlmInputSegment(is_prot=True, req_lm_loss=True, input_ids=[prot_id]))
        temp.segments.append(self._prot_eos_segment(req_lm_loss=True))
      else:
        temp.segments.append(self._prot_bos_segment(req_lm_loss=False))
        temp.segments.append(ProtLlmInputSegment(is_prot=True, req_lm_loss=False, input_ids=[prot_id]))
        temp.segments.append(self._prot_eos_segment(req_lm_loss=False))
        temp.segments.append(ProtLlmInputSegment(req_lm_loss=False, input_ids=prot_info_input_ids))
        temp.segments.append(ProtLlmInputSegment(req_lm_loss=True, input_ids=answer_input_ids))

      cur_len += self.prot_token_len + len(prot_info_input_ids) + len(answer_input_ids)
  
    if cur_len > 0:
      tot_n_prot += temp.n_prot
      concatenated_molinstruct.append(temp)
    
    print(f"Chunking molinstruct data into {len(concatenated_molinstruct)} sequences. tot_n_prot = {tot_n_prot} avg_n_prot = {tot_n_prot/len(concatenated_molinstruct)} exceed_max_len={exceed_max_len} skip_nid_not_found (exclude test) = {skip_nid_not_found}", flush=True)

    self.concatenated_molinstruct = concatenated_molinstruct
    
  
  