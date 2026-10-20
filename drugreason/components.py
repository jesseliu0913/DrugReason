"""Frozen routing encoder and straight-through Gumbel router."""

import torch
from torch import nn
from torch.nn import functional as F


class FrozenEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return (
            (hidden * attention_mask.unsqueeze(-1)).sum(1)
            / attention_mask.sum(1, keepdim=True).clamp_min(1)
        ).float()


class Router(nn.Module):
    def __init__(self, dimension, tau=1.0):
        super().__init__()
        if tau <= 0:
            raise ValueError("Routing temperature must be positive.")
        self.tau = tau
        self.mlp = nn.Sequential(
            nn.Linear(dimension, dimension // 2), nn.GELU(), nn.Linear(dimension // 2, 2)
        )

    def forward(self, hidden, deterministic=False):
        logits = self.mlp(hidden)
        if deterministic:
            soft = (logits / self.tau).softmax(-1)
        else:
            noise = -torch.empty_like(logits).exponential_().log()
            soft = ((logits + noise) / self.tau).softmax(-1)
        hard = F.one_hot(soft.argmax(-1), num_classes=2).to(soft.dtype)
        return hard - soft.detach() + soft, soft
