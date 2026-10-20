"""Prepare label spaces and KG views; keep labels outside all model inputs."""

import argparse
import json
import re
from pathlib import Path

DATASETS = ("ddinter", "drugbank", "pharmaDB")
GENERATED_VIEWS = ("chain_of_thought", "tree_of_thought", "forward_backward", "mechanism")


def read_jsonl(path):
    with Path(path).open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_rationale(text):
    """Deterministic formatting checks; never paraphrase a retained rationale."""
    if not isinstance(text, str) or not text.strip():
        return None
    if text.lstrip().startswith(("{", "[", "```")) or "None\nYour reasoning" in text:
        return None
    if re.search(r"please (?:provide|specify)|as an ai language model", text, re.I):
        return None
    lines = [
        line
        for line in text.splitlines()
        if not re.match(
            r"^\s*(?:[-#>*]+\s*)?(?:answer|label|gold label|relation label)\s*[*_:=-]*\s*:",
            line,
            re.I,
        )
    ]
    result = "\n".join(lines).strip()
    return result if len(result.split()) >= 8 else None


def serialize_view(record, view):
    # Explicitly select input fields: label_id and labels never enter this serialization.
    return f"{record['question']}\n\nEvidence:\n{view['text']}\n\nLabel:"


def canonical_label(row, dataset):
    if dataset != "drugbank":
        return row["label"]
    names = {row["drug1_name"]: "drug A", row["drug2_name"]: "drug B"}
    lookup = {name.lower(): value for name, value in names.items()}
    pattern = re.compile("|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)), re.I)
    return pattern.sub(lambda match: lookup[match.group().lower()], row["label"])


def label_space(rows, dataset):
    labels = {}
    for row in rows:
        key, text = int(row["label_idx"]), canonical_label(row, dataset)
        if key in labels and labels[key] != text:
            raise ValueError(f"Conflicting canonical verbalizers for label {key}.")
        labels[key] = text
    # Keep the benchmark's full closed label set even if a class is absent in train.
    fixed = {
        "ddinter": ("Major", "Moderate", "Minor", "No Interaction"),
        "pharmaDB": ("Disease-modifying", "Palliates", "Non-indication"),
    }
    if dataset in fixed:
        if not set(labels.values()).issubset(fixed[dataset]):
            raise ValueError("Unexpected label in the benchmark label space.")
        for text in fixed[dataset]:
            if text not in labels.values():
                key = next(i for i in range(len(fixed[dataset])) if i not in labels)
                labels[key] = text
    result = [{"id": key, "text": labels[key]} for key in sorted(labels)]
    if len({r["text"] for r in result}) != len(result):
        raise ValueError("Different labels have identical verbalizers.")
    return result


def pair_record(row, dataset, split, max_paths=4):
    if dataset == "pharmaDB":
        a, b, da, db = row["drug_name"], row["disease_name"], row["drug_desc"], row["disease_desc"]
        question = f"Drug A: {a}. Disease B: {b}. Predict the therapeutic relation."
    else:
        a, b, da, db = row["drug1_name"], row["drug2_name"], row["drug1_desc"], row["drug2_desc"]
        task = "interaction severity" if dataset == "ddinter" else "pharmacological relation"
        question = f"Drug A: {a}. Drug B: {b}. Predict the {task}."
    views = [{"kind": "definition", "text": f"Definition of {a}: {da}\nDefinition of {b}: {db}"}]
    for path in [s.strip() for s in row["path_str"].splitlines() if s.strip()][:max_paths]:
        views.append({"kind": "path", "text": path})
    return {
        "entity_a": a,
        "entity_b": b,
        "question": question,
        "label_id": int(row["label_idx"]),
        "split": split,
        "views": views,
    }


def validate_training(rows):
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Training requires records marked split=train.")
        if any(v["kind"] not in (*GENERATED_VIEWS, "definition", "path") for v in row["views"]):
            raise ValueError("Unknown reasoning view kind.")
        generated_list = [v for v in row["views"] if v["kind"] in GENERATED_VIEWS]
        if len(generated_list) != len(GENERATED_VIEWS):
            raise ValueError("Expected exactly one view for each teacher strategy.")
        generated = {v["kind"]: v for v in generated_list}
        if set(generated) != set(GENERATED_VIEWS):
            raise ValueError("Generate all four label-free teacher views before training.")
        if any(v.get("label_conditioned") is not False for v in generated.values()):
            raise ValueError(
                "Training requires label-free generated views; legacy conditioned annotations cannot be relabeled."
            )
        if any(not v.get("text", "").strip() for v in row["views"]):
            raise ValueError("Empty reasoning view.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-train-pairs", type=int, default=2000)
    p.add_argument("--max-test-pairs", type=int, default=1000)
    p.add_argument("--max-paths", type=int, default=4)
    args = p.parse_args()
    from datasets import load_dataset

    raw = load_dataset(f"Tassy24/K-Paths-inductive-reasoning-{args.dataset}")
    labels = label_space(raw["train"], args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    for split, limit in [("train", args.max_train_pairs), ("test", args.max_test_pairs)]:
        data = raw[split]
        if limit > 0 and len(data) > limit:
            data = (data.shuffle(seed=42) if split == "test" else data).select(range(limit))
        rows = [pair_record(row, args.dataset, split, args.max_paths) for row in data]
        known = {label["id"] for label in labels}
        if any(row["label_id"] not in known for row in rows):
            raise ValueError("A split contains a label absent from the training label space.")
        write_jsonl(output / f"{split}.jsonl", rows)
        print(f"{split}: {len(rows)} pairs, {len(labels)} canonical labels")


if __name__ == "__main__":
    main()
