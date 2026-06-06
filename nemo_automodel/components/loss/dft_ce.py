from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


class DFTCrossEntropy(nn.Module):
    def __init__(self, fp32_upcast: bool = True, ignore_index: int = -100):
        super().__init__()
        self.fp32_upcast = fp32_upcast
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_label_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        if labels.device != logits.device:
            labels = labels.to(logits.device)  # pragma: no cover

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        if mask is not None:
            with torch.no_grad():
                if mask.device != labels.device:
                    mask = mask.to(labels.device)  # pragma: no cover
                labels = labels.masked_fill(mask.view(-1) == 0, self.ignore_index)

        if self.fp32_upcast:
            logits = logits.float()

        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        if isinstance(labels, DTensor):
            labels = labels.full_tensor()

        per_token_loss = F.cross_entropy(logits, labels, ignore_index=self.ignore_index, reduction="none")
        with torch.no_grad():
            target_probs = torch.exp(-per_token_loss)
        loss = per_token_loss * target_probs

        if num_label_tokens is not None:
            if num_label_tokens == 0:
                return loss.sum() * 0.0
            return loss.sum() / num_label_tokens

        valid_tokens = (labels != self.ignore_index).sum().clamp_min(1)
        return loss.sum() / valid_tokens
