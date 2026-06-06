import re
from typing import Optional

import torch


_PISSA_NITER_RE = re.compile(r"^pissa_niter_(\d+)$")


def is_pissa_init(init_method: str) -> bool:
    return init_method == "pissa" or _PISSA_NITER_RE.match(init_method) is not None


def _parse_niter(init_method: str) -> Optional[int]:
    if init_method == "pissa":
        return None
    match = _PISSA_NITER_RE.match(init_method)
    if match is None:
        raise ValueError(
            f"PiSSA init method must be 'pissa' or 'pissa_niter_[number of iters]', got {init_method!r}."
        )
    return int(match.group(1))


def _check_weight_dtype(weight: torch.Tensor):
    if weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(
            "PiSSA initialization requires float32, float16, or bfloat16 base weights. "
            "For quantized training, initialize before quantization or use a non-PiSSA LoRA init."
        )


def _low_rank_svd(weight: torch.Tensor, rank: int, init_method: str):
    if weight.dim() != 2:
        raise ValueError(f"PiSSA expects a 2D weight matrix, got shape {tuple(weight.shape)}.")
    if rank <= 0 or rank > min(weight.shape):
        raise ValueError(f"PiSSA rank must be in [1, {min(weight.shape)}], got rank={rank}.")

    niter = _parse_niter(init_method)
    if niter is None:
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        return u[:, :rank], s[:rank], vh[:rank, :]

    u, s, v = torch.svd_lowrank(weight, q=rank, niter=niter)
    return u[:, :rank], s[:rank], v[:, :rank].transpose(0, 1)


@torch.no_grad()
def pissa_init_linear_weight_(
    base_weight: torch.Tensor,
    lora_A_weight: torch.Tensor,
    lora_B_weight: torch.Tensor,
    scale: float,
    init_method: str,
):
    """Apply PiSSA to an nn.Linear-style weight.

    Shapes:
      base_weight: [out_features, in_features]
      lora_A_weight: [rank, in_features]
      lora_B_weight: [out_features, rank]

    The residual base weight is updated so that
    ``base_weight + scale * (lora_B @ lora_A)`` initially reconstructs the
    original base weight up to the selected SVD rank.
    """
    _check_weight_dtype(base_weight)
    rank = lora_A_weight.shape[0]
    dtype = base_weight.dtype

    weight = base_weight.detach().to(torch.float32)
    u, s, vh = _low_rank_svd(weight, rank, init_method)
    s = s / scale
    sqrt_s = torch.sqrt(s)

    lora_A = sqrt_s.unsqueeze(1) * vh
    lora_B = u * sqrt_s.unsqueeze(0)

    lora_A_weight.copy_(lora_A.to(dtype=lora_A_weight.dtype, device=lora_A_weight.device))
    lora_B_weight.copy_(lora_B.to(dtype=lora_B_weight.dtype, device=lora_B_weight.device))
    base_weight.copy_((weight - scale * (lora_B @ lora_A)).to(dtype=dtype, device=base_weight.device))


@torch.no_grad()
def pissa_init_expert_weight_(
    base_weight: torch.Tensor,
    lora_A_weight: torch.Tensor,
    lora_B_weight: torch.Tensor,
    scale: float,
    init_method: str,
):
    """Apply PiSSA to grouped expert weights.

    Shapes:
      base_weight: [num_experts, in_features, out_features]
      lora_A_weight: [num_experts, in_features, rank]
      lora_B_weight: [num_experts, rank, out_features]

    Expert modules in AutoModel compute ``x @ W`` and LoRA computes
    ``x @ A @ B``, so the residual update is ``W -= scale * (A @ B)``.
    """
    _check_weight_dtype(base_weight)
    if base_weight.dim() != 3:
        raise ValueError(f"PiSSA expects grouped expert weights to be 3D, got shape {tuple(base_weight.shape)}.")
    if lora_A_weight.shape[0] != base_weight.shape[0] or lora_B_weight.shape[0] != base_weight.shape[0]:
        raise ValueError("PiSSA expert LoRA tensors must have the same number of experts as the base weight.")

    rank = lora_A_weight.shape[-1]
    dtype = base_weight.dtype
    residual = base_weight.detach().to(torch.float32).clone()

    for expert_idx in range(base_weight.shape[0]):
        weight = residual[expert_idx]
        u, s, vh = _low_rank_svd(weight, rank, init_method)
        s = s / scale
        sqrt_s = torch.sqrt(s)

        lora_A = u * sqrt_s.unsqueeze(0)
        lora_B = sqrt_s.unsqueeze(1) * vh

        lora_A_weight[expert_idx].copy_(lora_A.to(dtype=lora_A_weight.dtype, device=lora_A_weight.device))
        lora_B_weight[expert_idx].copy_(lora_B.to(dtype=lora_B_weight.dtype, device=lora_B_weight.device))
        residual[expert_idx] = weight - scale * (lora_A @ lora_B)

    base_weight.copy_(residual.to(dtype=dtype, device=base_weight.device))
