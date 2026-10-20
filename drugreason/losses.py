"""Per-example supervision for differentiable routing."""

import torch
from torch.nn import functional as F


def per_example_ce(logits, labels):
    shifted = labels[:, 1:]
    mask = shifted.ne(-100)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Every sequence needs at least one supervised next-token target.")
    losses = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2),
        shifted,
        reduction="none",
        ignore_index=-100,
    )
    return (losses * mask).sum(dim=1) / counts


def routed_ce(gate, logits1, labels1, logits2, labels2):
    """Keep straight-through gate weights in the supervised loss graph."""
    return (
        gate[:, 0] * per_example_ce(logits1, labels1)
        + gate[:, 1] * per_example_ce(logits2, labels2)
    ).mean()


def distillation_batch(tokenizer, prompts, answers, max_length, device):
    """Retokenize teacher text and reserve space for supervised answer tokens."""
    if len(prompts) != len(answers) or not prompts or max_length < 3:
        raise ValueError("Expected paired prompts/answers and max_length >= 3.")
    rows = []
    for prompt, answer in zip(prompts, answers):
        if not answer.strip():
            raise ValueError("Teacher generated an empty answer.")
        prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target = tokenizer(answer, add_special_tokens=False)["input_ids"]
        if not prefix or not target:
            raise ValueError("Empty tokenized prompt or answer.")
        target = target[: max_length - 2] + [tokenizer.eos_token_id]
        prefix = prefix[-(max_length - len(target)) :]
        rows.append((prefix + target, [-100] * len(prefix) + target))
    width = max(len(ids) for ids, _ in rows)
    return {
        "input_ids": torch.tensor(
            [ids + [tokenizer.pad_token_id] * (width - len(ids)) for ids, _ in rows],
            device=device,
        ),
        "attention_mask": torch.tensor(
            [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in rows],
            device=device,
        ),
        "labels": torch.tensor(
            [labels + [-100] * (width - len(labels)) for _, labels in rows],
            device=device,
        ),
    }
