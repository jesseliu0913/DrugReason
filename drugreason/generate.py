"""Add the four label-free teacher views to training pairs."""

import argparse
import os
from pathlib import Path
import json
from .data import GENERATED_VIEWS, clean_rationale, read_jsonl
from .prompts import teacher_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True, help="Teacher model ID or Azure deployment name.")
    args = p.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        p.error("Use separate input and output files.")
    rows = read_jsonl(args.input)
    if any(row.get("split") != "train" for row in rows):
        p.error("Teacher-generated views are used only for training.")
    from openai import OpenAI, AzureOpenAI

    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    else:
        client = OpenAI()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def key(row):
        return row["entity_a"], row["entity_b"], row["label_id"]

    seen = (
        {key(row) for row in read_jsonl(output)}
        if output.exists() and output.stat().st_size
        else set()
    )
    with output.open("a") as f:
        for row in rows:
            if key(row) in seen:
                continue
            views = [v for v in row["views"] if v["kind"] in ("definition", "path")]
            for kind in GENERATED_VIEWS:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {
                            "role": "user",
                            "content": teacher_prompt(row["entity_a"], row["entity_b"], kind),
                        }
                    ],
                )
                rationale = clean_rationale(response.choices[0].message.content)
                if rationale is None:
                    raise ValueError(
                        f"Teacher returned an invalid {kind} rationale; completed pairs are saved for resuming."
                    )
                views.append({"kind": kind, "text": rationale, "label_conditioned": False})
            f.write(json.dumps(dict(row, views=views), ensure_ascii=False) + "\n")
            f.flush()
            seen.add(key(row))


if __name__ == "__main__":
    main()
