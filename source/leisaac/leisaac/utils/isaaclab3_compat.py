"""Compatibility layer for running leisaac (built for IsaacLab 2.3) on IsaacLab 3.0.

IsaacLab 3.0 switched asset/sensor data buffers (root_pos_w, target_quat_w, ...)
from torch.Tensors to warp-backed ProxyArray objects. The deprecation bridge on
ProxyArray covers indexing and arithmetic, but passing one into isaaclab.utils.math
helpers fails inside warp kernels (e.g. quat_inv). This module wraps every public
math helper so ProxyArray arguments are converted to torch tensors transparently.

Importing this module installs the patches (idempotent).
"""

from __future__ import annotations

import functools
import inspect

import torch

from isaaclab.utils import math as _math
from isaaclab.utils.warp.proxy_array import ProxyArray


def _to_torch(x):
    if isinstance(x, ProxyArray):
        return x.torch
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        converted = [_to_torch(i) for i in x]
        return type(x)(converted)
    if isinstance(x, dict):
        return {k: _to_torch(v) for k, v in x.items()}
    return x


def _patch(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*(_to_torch(a) for a in args), **{k: _to_torch(v) for k, v in kwargs.items()})

    return wrapper


_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    for name, obj in list(vars(_math).items()):
        if name.startswith("_") or not callable(obj):
            continue
        if getattr(obj, "__module__", None) != _math.__name__:
            continue
        try:
            signature_is_wrappable = bool(inspect.signature(obj).parameters)
        except (ValueError, TypeError):
            signature_is_wrappable = False
        if signature_is_wrappable:
            setattr(_math, name, _patch(obj))
    _INSTALLED = True


install()