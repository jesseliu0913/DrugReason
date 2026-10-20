# DrugReason

Implementation for **DrugReason: Dynamic Multi-View Reasoning over Knowledge Graph and Language Evidence for Drug Repurposing**.

DrugReason builds multiple reasoning views for drug–drug and drug–disease pairs, trains two LoRA experts with a conditional router, and performs mutual distillation between the experts. This repository contains the router and mutual-distillation workflow.

## Install

Use Python 3.11 or newer and a CUDA-enabled PyTorch installation:

```bash
pip install -e .
```

Training loads two 4-bit language models and a frozen RoBERTa encoder. Access to the selected base models is required. The examples use Llama 3.2 3B and Qwen 2.5 3B; both model IDs are configurable.

## Reasoning data

The reasoning annotations are public on Hugging Face:

**[JesseLiu/DrugReason-Reasoning](https://huggingface.co/datasets/JesseLiu/DrugReason-Reasoning)** — 5,136 entity pairs and 20,544 reasoning annotations across DDInter, DrugBank, and PharmacotherapyDB.

Each pair has four generated reasoning views. The release includes the full annotation and the training prefix separately. These are label-conditioned training rationales; do not use them as unlabeled test evidence.

```python
from datasets import load_dataset

reasoning = load_dataset("JesseLiu/DrugReason-Reasoning", "ddinter", split="train")
```

Prepare all six training views, including definitions and graph paths from the original datasets:

```bash
python -m drugreason.data --dataset ddinter --output data/prompts
```

Use `drugbank` or `pharmaDB` for the other configurations. Original data: [DDInter](https://huggingface.co/datasets/Tassy24/K-Paths-inductive-reasoning-ddinter), [DrugBank](https://huggingface.co/datasets/Tassy24/K-Paths-inductive-reasoning-drugbank), and [PharmacotherapyDB](https://huggingface.co/datasets/Tassy24/K-Paths-inductive-reasoning-pharmaDB).

## Train

First, train the router and experts:

```bash
python -m drugreason.train_router \
  --prompt_root data/prompts --dataset ddinter \
  --model1 meta-llama/Llama-3.2-3B-Instruct --model2 Qwen/Qwen2.5-3B-Instruct \
  --epochs 1 --bs 4 --max_len 768 --no_reasoning_prob 0 \
  --outdir outputs/router
```

The command prints the saved `final_model` directory. Pass that directory to mutual distillation:

```bash
python -m drugreason.distill \
  --prompt_root data/prompts --dataset ddinter \
  --stage3_model_dir path/to/final_model \
  --epochs 1 --bs 4 --outdir outputs/distilled
```

The router uses a straight-through Gumbel gate. Both reasoning views contribute to the routed supervision, with a routing-balance penalty and projected hidden-state consistency. The historical `--kl_weight` argument controls that consistency term; it is a mean-squared error, not a vocabulary KL divergence.

Mutual distillation freezes the router and auxiliary components. Each expert learns from the other expert's generated text, retokenized with its own tokenizer, alongside supervised training. `--alpha` controls the distillation proportion. This implementation uses sequence-level distillation rather than KL over label probabilities.

## Evaluate

Evaluate either trained expert on the original held-out questions:

```bash
python -m drugreason.evaluate --checkpoint path/to/final_distilled \
  --expert student1 --dataset ddinter --output outputs/student1_predictions.jsonl
```

Use `student2` for the second expert, or a `final_model` checkpoint to evaluate before mutual distillation. Evaluation uses deterministic generation and does not load the training rationales.

## Optional teacher generation

The released annotations are sufficient for training. To generate new views, install `pip install -e '.[teacher]'`, configure `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`), then:

```bash
python -m drugreason.generate --input data/pairs.jsonl --output data/new_views.jsonl \
  --model YOUR_TEACHER_MODEL --view tree_thought
```

Input records contain `A`, `B`, `question`, and `label`. Azure deployments use `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_API_VERSION`. Teacher generation makes API calls; training from the release does not require them.

## Code

- `drugreason/data.py`, `prompts.py`, `generate.py`: reasoning views and preparation.
- `drugreason/components.py`, `losses.py`: router, experts, and objectives.
- `drugreason/train_router.py`, `distill.py`: the two training stages.
- `drugreason/evaluate.py`: expert evaluation.
- `scripts/export_reasoning.py`: deterministic export of existing annotations.

Run the CPU checks with `python -m unittest discover -s tests -v`. They cover routing gradients, tokenizer-specific distillation targets, masking, and data consistency. Full-scale GPU training has not been rerun for this cleanup. Model weights, experiment outputs, and alternative methods are not included.
