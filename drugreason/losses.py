"""Canonical-label scores and the two objectives in the DrugReason draft."""

import torch
from torch.nn import functional as F


def verbalizer_batch(tokenizer, prompts, verbalizers, max_length, device, prompt_budget=None):
    if len(prompts) != len(verbalizers) or not prompts:
        raise ValueError("Provide paired prompts and verbalizers.")
    rows = []
    for prompt, label in zip(prompts, verbalizers):
        prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        if not prefix or not target or len(target) >= max_length:
            raise ValueError(
                "The complete verbalizer and at least one prompt token must fit max_length."
            )
        budget = max_length - len(target) if prompt_budget is None else prompt_budget
        if budget < 1 or budget + len(target) > max_length:
            raise ValueError("The prompt and complete verbalizer must fit max_length.")
        prefix = prefix[-budget:]
        rows.append((prefix + target, [0] * len(prefix) + [1] * len(target)))
    width = max(len(ids) for ids, _ in rows)
    return {
        "input_ids": torch.tensor(
            [[tokenizer.pad_token_id] * (width - len(ids)) + ids for ids, _ in rows], device=device
        ),
        "attention_mask": torch.tensor(
            [[0] * (width - len(ids)) + [1] * len(ids) for ids, _ in rows], device=device
        ),
        "answer_mask": torch.tensor(
            [[0] * (width - len(ids)) + mask for ids, mask in rows], device=device, dtype=torch.bool
        ),
    }


def label_prompt_budget(tokenizer, verbalizers, max_length):
    if not verbalizers:
        raise ValueError("The label space cannot be empty.")
    lengths = [
        len(tokenizer(" " + label, add_special_tokens=False)["input_ids"])
        for label in set(verbalizers)
    ]
    budget = max_length - max(lengths)
    if budget < 1 or min(lengths) < 1:
        raise ValueError("The complete label space must fit with at least one prompt token.")
    return budget


def score_verbalizers(
    model, tokenizer, prompts, verbalizers, max_length=768, batch_size=4, prompt_budget=None
):
    """Mean next-token log probability over the complete label, excluding prompt/padding/EOS."""
    if batch_size < 1:
        raise ValueError("Label batch size must be positive.")
    device = next(model.parameters()).device
    if prompt_budget is None:
        prompt_budget = label_prompt_budget(tokenizer, verbalizers, max_length)
    scores = []
    for start in range(0, len(prompts), batch_size):
        batch = verbalizer_batch(
            tokenizer,
            prompts[start : start + batch_size],
            verbalizers[start : start + batch_size],
            max_length,
            device,
            prompt_budget,
        )
        mask = batch.pop("answer_mask")
        # Llama/Qwen support tail logits, avoiding a vocabulary tensor for the whole prompt.
        keep = int(mask.sum(1).max()) + 1
        logits = model(**batch, logits_to_keep=keep).logits
        length = logits.shape[1] - 1
        valid = mask[:, -length:]
        targets = batch["input_ids"][:, -length:][valid]
        token_scores = -F.cross_entropy(logits[:, :-1][valid].float(), targets, reduction="none")
        owners = valid.nonzero(as_tuple=True)[0]
        summed = token_scores.new_zeros(len(valid)).scatter_add(0, owners, token_scores)
        scores.append(summed / valid.sum(1))
    return torch.cat(scores)


def score_label_space(model, tokenizer, prompts, verbalizers, max_length=768, batch_size=4):
    """Score the same ordered semantic label space with each expert's own tokenizer."""
    repeated = [prompt for prompt in prompts for _ in verbalizers]
    labels = list(verbalizers) * len(prompts)
    return score_verbalizers(model, tokenizer, repeated, labels, max_length, batch_size).reshape(
        len(prompts), len(verbalizers)
    )


def pair_mean(values, owners):
    """Average views within each pair, then average pairs."""
    count = int(owners.max()) + 1
    sums = values.new_zeros(count).scatter_add(0, owners, values)
    sizes = torch.bincount(owners, minlength=count).to(values.dtype)
    return (sums / sizes).mean()


def mutual_kl(scores1, scores2, temperature=3.0):
    """Per-view bidirectional KL, with detached teachers and the T²/2 factor."""
    if temperature <= 0 or scores1.shape != scores2.shape:
        raise ValueError("Use a positive temperature and matching semantic label spaces.")
    log1, log2 = (scores1 / temperature).log_softmax(-1), (scores2 / temperature).log_softmax(-1)
    forward = F.kl_div(log2, log1.exp().detach(), reduction="none").sum(-1)
    backward = F.kl_div(log1, log2.exp().detach(), reduction="none").sum(-1)
    return (forward + backward) * (temperature**2 / 2)


def objective(
    gold_scores,
    gates,
    soft_gates,
    owners,
    load_weight=0.01,
    scores1=None,
    scores2=None,
    mutual_weight=0.0,
    temperature=3.0,
):
    prediction = pair_mean(-(gates * gold_scores).sum(-1), owners)
    entropies = []
    for pair in owners.unique():
        average = soft_gates[owners == pair].mean(0)
        entropies.append(-(average * average.clamp_min(1e-12).log()).sum())
    entropy = torch.stack(entropies).mean()
    loss = prediction - load_weight * entropy
    if scores1 is not None:
        loss = loss + mutual_weight * pair_mean(mutual_kl(scores1, scores2, temperature), owners)
    return loss
