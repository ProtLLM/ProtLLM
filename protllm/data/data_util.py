import torch
import numpy as np
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Iterable


def general_collate_fn(inputs, pad_to_multiple_of=1, pad_token_id=None, return_pt=True, model_max_length=10000000):
  _numpify_inputs = []
  for a in inputs:
    if isinstance(a, list):
      a = np.array(a)
    elif isinstance(a, np.ndarray):
      pass
    elif isinstance(a, str):
      return inputs
    else:
      raise ValueError
    _numpify_inputs.append(a)
  inputs = _numpify_inputs
  max_len = max(a.shape[-1] for a in inputs)
  if max_len % pad_to_multiple_of != 0:
    max_len += 8 - (max_len % pad_to_multiple_of)
  ret = np.empty(shape=(len(inputs), max_len), dtype=inputs[0].dtype)
  ret.fill(pad_token_id)
  for i, a, in enumerate(inputs):
    ret[i, :a.shape[-1]] = a
  
  if ret.shape[-1] > model_max_length:
    ret = ret[:, :model_max_length]
    print(f"[W] batch length exceed model max length, shape: f{ret.shape}", flush=True)
  
  if return_pt: ret = torch.from_numpy(ret)
  return ret


def general_collate_fn_1d(inputs, return_pt=True):
  assert all((not isinstance(item, Iterable)) for item in inputs)
  ret = np.array(inputs)
  if return_pt: ret = torch.from_numpy(ret)
  return ret


def general_collate_fn_flatten_1d(inputs, return_pt=True):
  all_inputs = []
  for a in inputs:
    assert isinstance(a, Iterable)
    all_inputs.extend(a)
    if len(a) > 0:
      assert not isinstance(all_inputs[-1], Iterable)
  ret = np.array(all_inputs)
  if return_pt: ret = torch.from_numpy(ret)
  return ret

