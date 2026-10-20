"""Generate optional new teacher annotations from prepared pair records."""

import argparse
import json
import os
from pathlib import Path
from .prompts import teacher_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="JSONL records with A, B, question and label.")
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True, help="Teacher model ID or Azure deployment name.")
    p.add_argument("--view", required=True, choices=["cot", "tree_thought", "fb"])
    args = p.parse_args()
    from openai import OpenAI, AzureOpenAI

    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    else:
        client = OpenAI()  # OPENAI_API_KEY; optional OPENAI_BASE_URL.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if output.exists():
        for line in output.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["A"], r["B"], r["prompt_type"]))
    with Path(args.input).open() as source, output.open("a") as target:
        for line in source:
            r = json.loads(line)
            key = r["A"], r["B"], args.view
            if key in seen:
                continue
            prompt = teacher_prompt(r["question"], r["A"], r["B"], r["label"], args.view)
            response = client.chat.completions.create(
                model=args.model, messages=[{"role": "user", "content": prompt}]
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Teacher returned an empty response.")
            target.write(
                json.dumps(
                    dict(
                        A=r["A"],
                        B=r["B"],
                        question=r["question"],
                        label=r["label"],
                        prompt_type=args.view,
                        prefix={"prefix": text.split("\n\n")[0], "gpt_output": text},
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            target.flush()
            seen.add(key)


if __name__ == "__main__":
    main()
