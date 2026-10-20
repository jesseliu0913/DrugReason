"""Evaluate a trained DrugReason expert with deterministic answer generation."""

import argparse
import json
from pathlib import Path


def question_for(row, dataset):
    if dataset == "pharmaDB":
        return (
            f"What is the therapeutic relationship between {row['drug_name']} and {row['disease_name']}?\n"
            "Choices: [Disease-modifying, Palliates, Non-indication]"
        )
    if dataset == "ddinter":
        return (
            f"What is the interaction severity between {row['drug1_name']} and {row['drug2_name']}?\n"
            "Choices: [Major, Moderate, Minor, No Interaction]"
        )
    return f"What is the pharmacological interaction between {row['drug1_name']} and {row['drug2_name']}?"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint", required=True, help="The final_model or final_distilled directory."
    )
    p.add_argument("--expert", choices=["student1", "student2"], required=True)
    p.add_argument("--dataset", choices=["ddinter", "drugbank", "pharmaDB"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    checkpoint = Path(args.checkpoint)
    config = torch.load(checkpoint / "components.pt", map_location="cpu", weights_only=True)
    config = config.get("args_stage3", config.get("args"))
    base_id = config["model1" if args.expert == "student1" else "model2"]
    tokenizer = AutoTokenizer.from_pretrained(base_id, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
    )
    model = PeftModel.from_pretrained(base, checkpoint / args.expert).eval()
    data = load_dataset(f"Tassy24/K-Paths-inductive-reasoning-{args.dataset}", split="test")
    if args.max_samples > 0 and len(data) > args.max_samples:
        data = data.shuffle(seed=42).select(range(args.max_samples))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f, torch.inference_mode():
        for start in range(0, len(data), args.batch_size):
            rows = [data[i] for i in range(start, min(start + args.batch_size, len(data)))]
            prompts = [
                question_for(row, args.dataset) + "\n\nDirectly answer it with ANSWER:"
                for row in rows
            ]
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).to(model.device)
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            answers = tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for row, answer in zip(rows, answers):
                f.write(
                    json.dumps(
                        {
                            "question": question_for(row, args.dataset),
                            "label": row["label"],
                            "answer": answer.strip(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(f"Saved {len(data)} predictions to {output}")


if __name__ == "__main__":
    main()
