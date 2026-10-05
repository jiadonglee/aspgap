"""Mixed precision: bf16 autocast on CUDA by default; losses stay in fp32.

fp16 autocast produced NaN train losses in the transformer runs (attention
logits / layernorm overflow in the SpecCLIP backbone and the unscaled NLL
exp(logvar)); bf16 has the fp32 exponent range. The forward runs under
autocast, but mu/logvar are cast to float32 and the NLL is evaluated OUTSIDE
autocast, so the loss never sees low-precision arithmetic.
"""

from __future__ import annotations

import contextlib

import torch

_DTYPES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
           "fp16": torch.float16, "float16": torch.float16}


def amp_settings(cfg: dict, device: str) -> tuple[bool, torch.dtype]:
    tr = cfg.get("train", {})
    name = str(tr.get("amp_dtype", "bf16")).lower()
    if name not in _DTYPES:
        raise ValueError(f"train.amp_dtype must be bf16|fp16, got {name!r}")
    return bool(tr.get("amp", True)) and str(device).startswith("cuda"), _DTYPES[name]


def autocast_ctx(enabled: bool, dtype: torch.dtype, device: str = "cuda"):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=str(device).split(":")[0], dtype=dtype)


def float_outputs(out: dict) -> dict:
    """Cast every tensor of a model output dict to fp32."""
    return {k: v.float() for k, v in out.items()}
