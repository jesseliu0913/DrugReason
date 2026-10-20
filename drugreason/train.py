"""Two-stage routed supervision and symmetric label-space mutual distillation."""

import argparse
import json
import random
from pathlib import Path
import torch
from .checkpoint import load_bundle, save_bundle, routing_inputs
from .data import read_jsonl, serialize_view, validate_training
from .losses import score_verbalizers, score_label_space, objective, label_prompt_budget

PRESETS = {
    "ddinter": {"router_lr": 3e-4, "mutual_lr": 5e-4, "mutual_weight": 0.01},
    "drugbank": {"router_lr": 1e-4, "mutual_lr": 1e-4, "mutual_weight": 0.10},
    "pharmaDB": {"router_lr": 5e-4, "mutual_lr": 1e-4, "mutual_weight": 0.0},
}


def training_loss(bundle, pairs, labels, stage, config):
    mapping = {item["id"]: i for i, item in enumerate(labels)}
    verbalizers = [item["text"] for item in labels]
    prompts, owners, targets = [], [], []
    for owner, pair in enumerate(pairs):
        for view in pair["views"]:
            prompts.append(serialize_view(pair, view))
            owners.append(owner)
            targets.append(mapping[pair["label_id"]])
    owners = torch.tensor(owners, device=bundle["device"])
    targets = torch.tensor(targets, device=bundle["device"])
    gates, soft = bundle["router"](routing_inputs(bundle, prompts))
    scores, gold = [], []
    for expert, tokenizer in zip(bundle["experts"], bundle["tokenizers"]):
        if stage == "mutual":
            all_scores = score_label_space(
                expert,
                tokenizer,
                prompts,
                verbalizers,
                config["max_length"],
                config["label_batch_size"],
            )
            scores.append(all_scores)
            gold.append(all_scores.gather(1, targets[:, None]).squeeze(1))
        else:
            gold.append(
                score_verbalizers(
                    expert,
                    tokenizer,
                    prompts,
                    [verbalizers[t] for t in targets.tolist()],
                    config["max_length"],
                    config["label_batch_size"],
                    prompt_budget=label_prompt_budget(tokenizer, verbalizers, config["max_length"]),
                )
            )
    return objective(
        torch.stack(gold, dim=1),
        gates,
        soft,
        owners,
        config["load_weight"],
        scores[0] if scores else None,
        scores[1] if scores else None,
        config["mutual_weight"] if stage == "mutual" else 0.0,
        config["temperature"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["router", "mutual"], required=True)
    p.add_argument("--dataset", choices=list(PRESETS), required=True)
    p.add_argument(
        "--data", required=True, help="Training pairs with four label-free teacher views."
    )
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", help="Stage 1 checkpoint, required for stage mutual.")
    p.add_argument("--model1", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--model2", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--encoder", default="FacebookAI/roberta-base")
    p.add_argument("--quantization", choices=["4bit", "none"], default="4bit")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--batch-size", type=int, default=1, help="Pairs per microbatch; every view is used."
    )
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--label-batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--router-learning-rate", type=float, default=1e-4)
    p.add_argument("--load-weight", type=float, default=0.01)
    p.add_argument("--mutual-weight", type=float)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.gradient_accumulation, args.label_batch_size) < 1:
        p.error("Epochs, batch sizes, and gradient accumulation must be positive.")
    if args.temperature <= 0 or args.tau <= 0:
        p.error("Temperatures must be positive.")
    if args.stage == "mutual" and not args.checkpoint:
        p.error("--checkpoint is required for mutual distillation.")
    if args.stage == "router" and args.checkpoint:
        p.error("Use stage mutual to continue a router checkpoint.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    pairs = read_jsonl(args.data)
    validate_training(pairs)
    labels = json.loads(Path(args.labels).read_text())
    known = {item["id"] for item in labels}
    if any(pair["label_id"] not in known for pair in pairs):
        raise ValueError("Unknown training label.")
    config = vars(args).copy()
    config["models"] = [args.model1, args.model2]
    if args.checkpoint:
        old = json.loads((Path(args.checkpoint) / "config.json").read_text())
        old_labels = json.loads((Path(args.checkpoint) / "labels.json").read_text())
        if labels != old_labels or old["dataset"] != args.dataset:
            raise ValueError("Checkpoint and training label spaces must match exactly.")
        for key in ("models", "encoder", "quantization", "tau"):
            config[key] = old[key]
    preset = PRESETS[args.dataset]
    config["mutual_weight"] = (
        args.mutual_weight if args.mutual_weight is not None else preset["mutual_weight"]
    )
    config["learning_rate"] = (
        args.learning_rate if args.learning_rate is not None else preset[f"{args.stage}_lr"]
    )
    bundle = load_bundle(config, args.checkpoint)
    experts = [
        param for model in bundle["experts"] for param in model.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": experts, "lr": config["learning_rate"]},
            {"params": bundle["router"].parameters(), "lr": args.router_learning_rate},
        ],
        weight_decay=0.0,
    )
    for epoch in range(args.epochs):
        random.shuffle(pairs)
        # Accumulate by pair count so the final short batch receives the correct weight.
        window = args.batch_size * args.gradient_accumulation
        for start in range(0, len(pairs), window):
            group = pairs[start : start + window]
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for offset in range(0, len(group), args.batch_size):
                microbatch = group[offset : offset + args.batch_size]
                loss = (
                    training_loss(bundle, microbatch, labels, args.stage, config)
                    * len(microbatch)
                    / len(group)
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss.")
                loss.backward()
                total += loss.detach().item()
            torch.nn.utils.clip_grad_norm_(experts + list(bundle["router"].parameters()), 1.0)
            optimizer.step()
            print(
                f"epoch={epoch + 1} pairs={min(start + window, len(pairs))}/{len(pairs)} loss={total:.4f}",
                flush=True,
            )
    save_bundle(bundle, config, labels, args.output)
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
