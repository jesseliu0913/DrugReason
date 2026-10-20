from .components import CotEncoder, Router, HiddenProjector, WeightLearner
from .data import load_prefixes
import random
import argparse
import pathlib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta
from tqdm import tqdm
import bitsandbytes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prompt_root",
        type=str,
        required=True,
        help="Root dir of prompt_collection/<dataset> files",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ddinter", "drugbank", "pharmaDB"],
        help="Subfolder under prompt_root with JSONL prompts",
    )
    p.add_argument(
        "--stage3_model_dir",
        type=str,
        required=True,
        help="Directory with Stage 3 trained model (e.g., runs/ddinter/final_model)",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--max_len", type=int, default=768)
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=48,
        help="Tokens the teacher generates for sequence-level distillation",
    )
    p.add_argument("--lr_s1", type=float, default=1e-06)
    p.add_argument("--lr_s2", type=float, default=1e-06)
    p.add_argument("--distill_weight", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.7, help="Weight for distillation vs CE loss")
    p.add_argument("--outdir", type=str, default="runs_stage4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, cap number of training samples (for smoke tests)",
    )
    return p.parse_args()


class MutualDistillationDataset(Dataset):
    """Dataset for mutual distillation with original text preservation"""

    def __init__(self, data, tok_shared, tok1, tok2, max_len):
        self.data = data
        self.tok_shared = tok_shared
        self.tok1 = tok1
        self.tok2 = tok2
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample["question"]
        variants = sample["variants"]
        variant_idx = random.randint(0, len(variants) - 1)
        text = f"{question}\n\nReasoning:{variants[variant_idx]}\n\nAnswer:{sample['label']}"
        gen_prompt = f"{question}\n\nReasoning:"
        enc = self.tok_shared(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        s1 = self.tok1(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        s2 = self.tok2(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        def make_label_safe(tokens, tokenizer):
            lbl = tokens["input_ids"].clone()
            vocab_size = len(tokenizer)
            lbl = torch.clamp(lbl, 0, vocab_size - 1)
            lbl[tokens["attention_mask"] == 0] = -100
            return lbl

        def validate_tokens(tokens, tokenizer):
            vocab_size = len(tokenizer)
            input_ids = tokens["input_ids"]
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            tokens["input_ids"] = input_ids
            return tokens

        s1 = validate_tokens(s1, self.tok1)
        s2 = validate_tokens(s2, self.tok2)
        lbl1 = make_label_safe(s1, self.tok1)
        lbl2 = make_label_safe(s2, self.tok2)
        return {
            "enc_ids": enc["input_ids"].squeeze(0),
            "enc_mask": enc["attention_mask"].squeeze(0),
            "ids1": s1["input_ids"].squeeze(0),
            "mask1": s1["attention_mask"].squeeze(0),
            "labels1": lbl1.squeeze(0),
            "ids2": s2["input_ids"].squeeze(0),
            "mask2": s2["attention_mask"].squeeze(0),
            "labels2": lbl2.squeeze(0),
            "original_text": text,
            "gen_prompt": gen_prompt,
            "variant_idx": variant_idx,
        }


def collate_fn(batch):
    keys = batch[0].keys()
    result = {}
    for key in keys:
        if key in ["variant_idx", "original_text", "gen_prompt"]:
            result[key] = [item[key] for item in batch]
        else:
            result[key] = torch.stack([item[key] for item in batch])
    return result


def load_stage3_model(stage3_dir, device):
    """Load Stage 3 trained model components"""
    stage3_dir = pathlib.Path(stage3_dir)
    components_path = stage3_dir / "components.pt"
    print(f"Loading Stage 3 model from {stage3_dir}")
    checkpoint = torch.load(components_path, map_location=device)
    args_stage3 = checkpoint["args"]
    print(f"Stage 3 models: {args_stage3['model1']} and {args_stage3['model2']}")
    model1_name = args_stage3["model1"]
    model2_name = args_stage3["model2"]
    encoder_name = "roberta-base"
    tok1 = AutoTokenizer.from_pretrained(
        model1_name, local_files_only=pathlib.Path(model1_name).exists()
    )
    if tok1.pad_token is None:
        tok1.pad_token = tok1.eos_token
    tok2 = AutoTokenizer.from_pretrained(
        model2_name, local_files_only=pathlib.Path(model2_name).exists()
    )
    if tok2.pad_token is None:
        tok2.pad_token = tok2.eos_token
    tok_shared = AutoTokenizer.from_pretrained(
        encoder_name, local_files_only=pathlib.Path(encoder_name).exists()
    )
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading base models...")
    base_model1 = AutoModelForCausalLM.from_pretrained(
        model1_name,
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        local_files_only=pathlib.Path(model1_name).exists(),
    )
    base_model2 = AutoModelForCausalLM.from_pretrained(
        model2_name,
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        local_files_only=pathlib.Path(model2_name).exists(),
    )
    base_model1.eval()
    base_model2.eval()
    for p in base_model1.parameters():
        p.requires_grad = False
    for p in base_model2.parameters():
        p.requires_grad = False
    print("Loading LoRA adapters...")
    student1 = PeftModel.from_pretrained(base_model1, stage3_dir / "student1", is_trainable=True)
    student2 = PeftModel.from_pretrained(base_model2, stage3_dir / "student2", is_trainable=True)
    student1.train()
    student2.train()
    print("Enabling LoRA parameters for training...")
    for name, param in student1.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            param.requires_grad = True
    for name, param in student2.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            param.requires_grad = True
    student1_trainable = sum((p.numel() for p in student1.parameters() if p.requires_grad))
    student2_trainable = sum((p.numel() for p in student2.parameters() if p.requires_grad))
    if student1_trainable == 0 or student2_trainable == 0:
        print("ERROR: No trainable parameters found!")
        print("This might be due to:")
        print("1. LoRA adapters not properly saved/loaded")
        print("2. PEFT version incompatibility")
        print("3. Model architecture changes")
        raise ValueError("No trainable parameters found! Check LoRA adapter loading.")
    enc_backbone = AutoModel.from_pretrained(
        encoder_name,
        torch_dtype=torch.float16,
        local_files_only=pathlib.Path(encoder_name).exists(),
    ).eval()
    encoder = CotEncoder(enc_backbone)
    encoder.load_state_dict(checkpoint["encoder"])
    router = Router(enc_backbone.config.hidden_size, args_stage3["tau"])
    router.load_state_dict(checkpoint["router"])
    weight_net = WeightLearner(base_model1.config.hidden_size)
    weight_net.load_state_dict(checkpoint["weight_net"])
    proj1 = HiddenProjector(base_model1.config.hidden_size, base_model1.config.hidden_size)
    proj1.load_state_dict(checkpoint["proj1"])
    proj2 = HiddenProjector(base_model2.config.hidden_size, base_model1.config.hidden_size)
    proj2.load_state_dict(checkpoint["proj2"])
    return {
        "student1": student1,
        "student2": student2,
        "encoder": encoder,
        "router": router,
        "weight_net": weight_net,
        "proj1": proj1,
        "proj2": proj2,
        "tok1": tok1,
        "tok2": tok2,
        "tok_shared": tok_shared,
        "args_stage3": args_stage3,
    }


@torch.no_grad()
def teacher_generate(gen_model, tokenizer, prompts, max_new_tokens, device):
    """Let a teacher generate its own answer text with greedy decoding.

    Cross-tokenizer logit KL is invalid because vocabulary index i does not denote
    the same token in the Llama and Qwen vocabularies, and sequence positions from
    two tokenizers are not aligned. Sequence-level distillation avoids both problems:
    the teacher produces natural-language text, which the student re-tokenizes with
    its own tokenizer.
    """
    was_training, old_side = gen_model.training, tokenizer.padding_side
    try:
        gen_model.eval()
        tokenizer.padding_side = "left"
        enc = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = gen_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return [
            a.strip()
            for a in tokenizer.batch_decode(
                out[:, enc["input_ids"].size(1) :], skip_special_tokens=True
            )
        ]
    finally:
        tokenizer.padding_side = old_side
        gen_model.train(was_training)


def seqkd_ce(student, tokenizer, prompts, teacher_answers, max_len, device):
    """CE of the student on (prompt + teacher answer), supervising only the answer tokens."""
    from .losses import distillation_batch

    batch = distillation_batch(tokenizer, prompts, teacher_answers, max_len, device)
    return student(**batch).loss


def mutual_distillation_step(
    student1,
    student2,
    gen_model1,
    gen_model2,
    batch,
    args,
    device,
    tok1,
    tok2,
    gen_prompts,
):
    """One step of sequence-level mutual distillation.

    Each student is trained on (a) the gold supervised target and (b) the natural-language
    answer generated by the other expert, re-tokenized with the student's own tokenizer.
    """
    teacher2_answers = teacher_generate(gen_model2, tok2, gen_prompts, args.max_new_tokens, device)
    teacher1_answers = teacher_generate(gen_model1, tok1, gen_prompts, args.max_new_tokens, device)
    ce_loss1 = student1(
        input_ids=batch["ids1"], attention_mask=batch["mask1"], labels=batch["labels1"]
    ).loss
    distill_loss1 = seqkd_ce(student1, tok1, gen_prompts, teacher2_answers, args.max_len, device)
    loss1 = args.alpha * distill_loss1 + (1 - args.alpha) * ce_loss1
    ce_loss2 = student2(
        input_ids=batch["ids2"], attention_mask=batch["mask2"], labels=batch["labels2"]
    ).loss
    distill_loss2 = seqkd_ce(student2, tok2, gen_prompts, teacher1_answers, args.max_len, device)
    loss2 = args.alpha * distill_loss2 + (1 - args.alpha) * ce_loss2
    total_loss = args.distill_weight * loss1 + args.distill_weight * loss2
    return {
        "total_loss": total_loss,
        "loss1": loss1,
        "loss2": loss2,
        "ce_loss1": ce_loss1,
        "ce_loss2": ce_loss2,
        "distill_loss1": distill_loss1,
        "distill_loss2": distill_loss2,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    prompt_root = pathlib.Path(args.prompt_root) / args.dataset
    stage3_path = pathlib.Path(args.stage3_model_dir)
    model_name = stage3_path.parent.name if stage3_path.name == "final_model" else stage3_path.name
    out_dir = pathlib.Path(args.outdir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_prefixes(prompt_root)
    if args.limit and args.limit > 0:
        data = data[: args.limit]
    print(f"Loaded {len(data)} samples for mutual distillation")
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    pg = InitProcessGroupKwargs(timeout=timedelta(1800))
    acc = Accelerator(kwargs_handlers=[ddp, pg])
    components = load_stage3_model(args.stage3_model_dir, acc.device)
    student1 = components["student1"]
    student2 = components["student2"]
    encoder = components["encoder"]
    router = components["router"]
    weight_net = components["weight_net"]
    proj1 = components["proj1"]
    proj2 = components["proj2"]
    encoder = encoder.to(acc.device)
    router = router.to(acc.device)
    weight_net = weight_net.to(acc.device)
    proj1 = proj1.to(acc.device)
    proj2 = proj2.to(acc.device)
    for p in encoder.parameters():
        p.requires_grad = False
    for p in router.parameters():
        p.requires_grad = False
    for p in weight_net.parameters():
        p.requires_grad = False
    for p in proj1.parameters():
        p.requires_grad = False
    for p in proj2.parameters():
        p.requires_grad = False
    print("Setting up data loader...")
    dataset = MutualDistillationDataset(
        data,
        components["tok_shared"],
        components["tok1"],
        components["tok2"],
        args.max_len,
    )
    dl = DataLoader(dataset, batch_size=args.bs, shuffle=True, collate_fn=collate_fn)
    student1_params = [p for p in student1.parameters() if p.requires_grad]
    student2_params = [p for p in student2.parameters() if p.requires_grad]
    print(f"Student1 trainable params found: {len(student1_params)}")
    print(f"Student2 trainable params found: {len(student2_params)}")
    if len(student1_params) == 0 or len(student2_params) == 0:
        print("Detailed parameter analysis:")
        raise ValueError("No trainable parameters found for students!")
    opt_groups = [
        {"params": student1_params, "lr": args.lr_s1},
        {"params": student2_params, "lr": args.lr_s2},
    ]
    opt = bitsandbytes.optim.Adam8bit(opt_groups, betas=(0.9, 0.95))
    student1, student2, opt, dl = acc.prepare(student1, student2, opt, dl)
    gen_model1 = acc.unwrap_model(student1)
    gen_model2 = acc.unwrap_model(student2)

    def adapter_signature(model):
        total = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                total += p.detach().float().abs().sum().item()
        return total

    init_sig1 = adapter_signature(gen_model1)
    init_sig2 = adapter_signature(gen_model2)
    print("Starting mutual distillation training...")
    print("Router and other components are FROZEN")
    print("Only students will learn from each other")
    for epoch in range(1, args.epochs + 1):
        prog = tqdm(dl, desc=f"Distillation Epoch {epoch}", disable=not acc.is_main_process)
        for batch_idx, batch in enumerate(prog):
            gen_prompts = batch.pop("gen_prompt")
            batch.pop("original_text")
            batch.pop("variant_idx")
            batch = {k: v.to(acc.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            opt.zero_grad()
            losses = mutual_distillation_step(
                student1,
                student2,
                gen_model1,
                gen_model2,
                batch,
                args,
                acc.device,
                components["tok1"],
                components["tok2"],
                gen_prompts,
            )
            if not losses["total_loss"].requires_grad:
                raise RuntimeError(f"Loss has no gradients in batch {batch_idx}")
            acc.backward(losses["total_loss"])
            opt.step()
            if acc.is_main_process:
                prog.set_postfix(loss=f"{losses['total_loss'].item():.4f}")
        if acc.is_main_process:
            epoch_dir = out_dir / f"epoch_{epoch}"
            epoch_dir.mkdir(exist_ok=True)
            acc.unwrap_model(student1).save_pretrained(epoch_dir / "student1_distilled")
            acc.unwrap_model(student2).save_pretrained(epoch_dir / "student2_distilled")
            torch.save(
                {
                    "router": router.state_dict(),
                    "weight_net": weight_net.state_dict(),
                    "proj1": proj1.state_dict(),
                    "proj2": proj2.state_dict(),
                    "encoder": encoder.state_dict(),
                    "args_stage3": components["args_stage3"],
                    "args_stage4": vars(args),
                    "epoch": epoch,
                    "stage": "mutual_distillation",
                },
                epoch_dir / "components.pt",
            )
            print(f"Saved distilled models to {epoch_dir}")
    final_sig1 = adapter_signature(gen_model1)
    final_sig2 = adapter_signature(gen_model2)
    if abs(final_sig1 - init_sig1) < 1e-06 or abs(final_sig2 - init_sig2) < 1e-06:
        raise RuntimeError(
            f"Adapters did not change during distillation (s1 {init_sig1:.6f}->{final_sig1:.6f}, s2 {init_sig2:.6f}->{final_sig2:.6f}). Refusing to save a no-op distillation result."
        )
    if acc.is_main_process:
        final_dir = out_dir / "final_distilled"
        final_dir.mkdir(exist_ok=True)
        acc.unwrap_model(student1).save_pretrained(final_dir / "student1")
        acc.unwrap_model(student2).save_pretrained(final_dir / "student2")
        torch.save(
            {
                "router": router.state_dict(),
                "weight_net": weight_net.state_dict(),
                "proj1": proj1.state_dict(),
                "proj2": proj2.state_dict(),
                "encoder": encoder.state_dict(),
                "args_stage3": components["args_stage3"],
                "args_stage4": vars(args),
                "final_epoch": args.epochs,
                "stage": "mutual_distillation_final",
            },
            final_dir / "components.pt",
        )
        print(f"Saved final distilled models to {final_dir}")
        print("Final model contains:")
        print("  • Enhanced Student1 (learned from Student2)")
        print("  • Enhanced Student2 (learned from Student1)")
        print("  • Original Router from Stage 3 (frozen)")
    print("Mutual distillation complete!")


if __name__ == "__main__":
    main()
