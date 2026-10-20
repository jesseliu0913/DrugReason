"""Export DrugReason reasoning annotations in the public dataset schema."""

import argparse
import csv
import json
from pathlib import Path

DATASETS = ("ddinter", "drugbank", "pharmaDB")
VIEWS = ("reason_chain", "cot", "tree_thought", "fb")


def export(source, output):
    source, output = Path(source), Path(output)
    counts = {}
    for dataset in DATASETS:
        with (source / "reasoning_data" / f"{dataset}.csv").open() as f:
            chains = list(csv.DictReader(f))
        a, b = ("drug_name", "disease_name") if dataset == "pharmaDB" else ("drugA", "drugB")
        chain_map = {(r[a], r[b]): r for r in chains}
        if len(chain_map) != len(chains):
            raise ValueError(f"Duplicate entity pairs: {dataset}")
        records, seen = [], set()
        for view in VIEWS:
            path = source / "prompt_collection_human" / dataset / f"{view}.jsonl"
            for line in path.read_text().splitlines():
                row = json.loads(line)
                key = row["A"], row["B"]
                original = chain_map[key]
                if original["label"] != row["label"] or (*key, view) in seen:
                    raise ValueError(f"Inconsistent annotation: {dataset} {key}")
                prefix = row["prefix"]["prefix"].strip()
                reasoning = (
                    original["reason_path"]
                    if view == "reason_chain"
                    else row["prefix"]["gpt_output"]
                ).strip()
                if not prefix or not reasoning:
                    raise ValueError(f"Empty annotation: {dataset} {key}")
                records.append(
                    dict(
                        entity_a=key[0],
                        entity_b=key[1],
                        question=row["question"].strip(),
                        label=row["label"],
                        view=view,
                        reasoning=reasoning,
                        training_prefix=prefix,
                        label_conditioned=True,
                    )
                )
                seen.add((*key, view))
        if len(records) != len(chains) * len(VIEWS):
            raise ValueError(f"Incomplete view coverage: {dataset}")
        dest = output / dataset
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / "train.jsonl").open("w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[dataset] = {"pairs": len(chains), "reasoning_annotations": len(records)}
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2))
