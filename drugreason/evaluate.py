"""Deterministic routed inference and mean aggregation over KG evidence views."""

import argparse
import json
from pathlib import Path
import torch
from .checkpoint import load_bundle, routing_inputs
from .data import read_jsonl, serialize_view
from .losses import score_label_space


@torch.inference_mode()
def predict_pair(bundle, pair, labels, max_length=768, label_batch_size=4):
    views = pair["views"]
    if not views or any(view["kind"] not in ("definition", "path") for view in views):
        raise ValueError("Inference uses only definition and KG-path views; no teacher rationales.")
    prompts = [serialize_view(pair, view) for view in views]
    gates, _ = bundle["router"](routing_inputs(bundle, prompts), deterministic=True)
    routes = gates.argmax(-1)
    distributions = torch.empty(len(views), len(labels), device=bundle["device"])
    verbalizers = [item["text"] for item in labels]
    for expert_id, (expert, tokenizer) in enumerate(zip(bundle["experts"], bundle["tokenizers"])):
        selected = (routes == expert_id).nonzero(as_tuple=True)[0]
        if selected.numel():
            scores = score_label_space(
                expert,
                tokenizer,
                [prompts[i] for i in selected.tolist()],
                verbalizers,
                max_length,
                label_batch_size,
            )
            distributions[selected] = scores.softmax(-1)
    probabilities = distributions.mean(0)
    prediction = labels[int(probabilities.argmax())]
    return {
        "label_id": prediction["id"],
        "label": prediction["text"],
        "probabilities": probabilities.cpu().tolist(),
        "routes": routes.cpu().tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="Held-out pairs with KG views only.")
    p.add_argument("--output", required=True)
    p.add_argument("--label-batch-size", type=int, default=4)
    args = p.parse_args()
    path = Path(args.checkpoint)
    config = json.loads((path / "config.json").read_text())
    labels = json.loads((path / "labels.json").read_text())
    bundle = load_bundle(config, args.checkpoint, trainable=False)
    pairs = read_jsonl(args.data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    correct, count = 0, 0
    with output.open("w") as f:
        for pair in pairs:
            prediction = predict_pair(
                bundle, pair, labels, config["max_length"], args.label_batch_size
            )
            record = {"entity_a": pair["entity_a"], "entity_b": pair["entity_b"], **prediction}
            if "label_id" in pair:
                record["target_label_id"] = pair["label_id"]
                correct += pair["label_id"] == prediction["label_id"]
                count += 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} predictions to {output}")
    if count:
        print(f"Label accuracy: {correct / count:.4f}")


if __name__ == "__main__":
    main()
