from typing import Callable
import torch


def apply_recursive(d, fn: Callable, filter: Callable = None):
    if isinstance(d, list):
        return [apply_recursive(da, fn, filter) for da in d]
    elif isinstance(d, tuple):
        return tuple(apply_recursive(list(d), fn, filter))
    elif isinstance(d, dict):
        return {k: apply_recursive(v, fn, filter) for k, v in d.items()}
    else:
        if filter is None or filter(d):
            return fn(d)
        else:
            return d


def apply_to_tensors(d, fn: Callable):
    return apply_recursive(d, fn, torch.is_tensor)