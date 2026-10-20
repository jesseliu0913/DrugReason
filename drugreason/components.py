"""Frozen text encoder, conditional router, and LoRA experts."""

import torch
from torch import nn
from torch.nn import functional as F


class CotEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    @torch.no_grad()
    def forward(self, ids, mask):
        h = self.backbone(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
        num = (h * mask.unsqueeze(-1)).sum(1)
        den = mask.sum(1, keepdim=True).clamp(min=1)
        return (num / den).float()


class Router(nn.Module):
    def __init__(self, d, tau):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 2))
        self.tau = tau

    def forward(self, h, tau=None):
        current_tau = tau if tau is not None else self.tau
        return F.gumbel_softmax(self.mlp(h), tau=current_tau, hard=True)


class HiddenProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False, dtype=torch.float16)

    def forward(self, h):
        return self.proj(h)


class WeightLearner(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1, bias=False, dtype=torch.float16)

    def forward(self, h):
        return torch.sigmoid(self.fc(h)).squeeze(-1)


def make_student(base):
    from peft import get_peft_model, LoraConfig, TaskType

    cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(base, cfg)
