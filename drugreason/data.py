"""Load grouped reasoning views and prepare the public training annotations."""

import argparse
import json
from pathlib import Path

DATASET_ID = "JesseLiu/DrugReason-Reasoning"
DATASETS = ("ddinter", "drugbank", "pharmaDB")


def load_prefixes(directory):
    samples = {}
    for file in sorted(Path(directory).glob("*.jsonl")):
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["A"], row["B"]
            prefix = row["prefix"]
            if isinstance(prefix, dict):
                prefix = prefix["prefix"]
            if not isinstance(prefix, str) or not prefix.strip():
                raise ValueError(f"Empty or invalid prefix in {file.name}")
            item = samples.setdefault(
                key,
                dict(
                    id=len(samples),
                    A=key[0],
                    B=key[1],
                    question=row["question"],
                    label=row["label"],
                    variants=[],
                ),
            )
            if item["label"] != row["label"] or item["question"] != row["question"]:
                raise ValueError(f"Inconsistent views for pair {key}")
            item["variants"].append(prefix)
    if not samples:
        raise ValueError(f"No reasoning views found in {directory}")
    return list(samples.values())


def prepare(dataset, output):
    from datasets import load_dataset

    annotations = load_dataset(DATASET_ID, dataset, split="train")
    upstream = load_dataset(f"Tassy24/K-Paths-inductive-reasoning-{dataset}", split="train")
    if dataset == "pharmaDB":
        a, b, desc_a, desc_b = "drug_name", "disease_name", "drug_desc", "disease_desc"
    else:
        a, b, desc_a, desc_b = "drug1_name", "drug2_name", "drug1_desc", "drug2_desc"
    evidence = {}
    for row in upstream:
        evidence.setdefault((row[a], row[b], row["label"]), row)
    views, pairs = {}, {}
    for row in annotations:
        pair = row["entity_a"], row["entity_b"]
        record = dict(A=pair[0], B=pair[1], question=row["question"], label=row["label"])
        pairs[pair] = record
        views.setdefault(row["view"], []).append(
            dict(record, prompt_type=row["view"], prefix=row["training_prefix"])
        )
    for pair, record in pairs.items():
        original = evidence[(*pair, record["label"])]
        if original["label"] != record["label"]:
            raise ValueError(f"Original label mismatch: {pair}")
        definition = f"{original[desc_a]}\n{original[desc_b]}"
        path = original["path_str"].split("\n")[:4]
        views.setdefault("definition", []).append(
            dict(
                record,
                prompt_type="definition",
                prefix=f"The two terminologies have definitions as follows:\n{definition}",
            )
        )
        views.setdefault("path", []).append(
            dict(
                record,
                prompt_type="path",
                prefix=f"The question has biological path as follows:\n{path}",
            )
        )
    directory = Path(output) / dataset
    directory.mkdir(parents=True, exist_ok=True)
    for name, records in views.items():
        with (directory / f"{name}.jsonl").open("w") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Prepared {len(pairs)} pairs with {len(views)} views each in {directory}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--output", default="data/prompts")
    args = p.parse_args()
    prepare(args.dataset, args.output)


if __name__ == "__main__":
    main()
