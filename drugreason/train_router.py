from .losses import routed_ce
from .components import CotEncoder, Router, HiddenProjector, WeightLearner, make_student
from .data import load_prefixes
import os
import json
import random
import argparse
import pathlib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    RobertaTokenizer,
    RobertaModel,
)
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
import datetime
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
    p.add_argument("--model1", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--model2", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--max_len", type=int, default=768)
    p.add_argument("--lr_s1", type=float, default=5e-06)
    p.add_argument("--lr_s2", type=float, default=0.0002)
    p.add_argument("--lr_misc", type=float, default=0.0001)
    p.add_argument("--kl_weight", type=float, default=0.1)
    p.add_argument("--reg_weight", type=float, default=0.01)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_reasoning_prob",
        type=float,
        default=0.15,
        help="Probability of using no-reasoning version (question + answer only)",
    )
    return p.parse_args()


class CotDataset(Dataset):
    """Dataset with added no-reasoning version option"""

    def __init__(self, data, tok_shared, tok1, tok2, max_len, no_reasoning_prob=0.15):
        self.data = data
        self.tok_shared = tok_shared
        self.tok1 = tok1
        self.tok2 = tok2
        self.max_len = max_len
        self.no_reasoning_prob = no_reasoning_prob

    def __len__(self):
        return len(self.data)

    def _create_text(self, sample, variant_idx, use_no_reasoning=False):
        """Create text with or without reasoning"""
        question = sample["question"]
        label = sample["label"]
        if use_no_reasoning:
            return f"{question}\n\nAnswer:{label}"
        else:
            variants = sample["variants"]
            reasoning = variants[variant_idx] if variant_idx < len(variants) else variants[0]
            return f"{question}\n\nReasoning:{reasoning}\n\nAnswer:{label}"

    def __getitem__(self, idx):
        sample = self.data[idx]
        variants = sample["variants"]
        k = len(variants)
        use_no_reasoning_0 = random.random() < self.no_reasoning_prob
        use_no_reasoning_1 = random.random() < self.no_reasoning_prob
        if k < 2:
            idx0, idx1 = (0, 0)
        else:
            idx0, idx1 = random.sample(range(k), 2)
        text0 = self._create_text(sample, idx0, use_no_reasoning_0)
        text1 = self._create_text(sample, idx1, use_no_reasoning_1)
        enc0 = self.tok_shared(
            text0.rsplit("\n\nAnswer:", 1)[0],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        enc1 = self.tok_shared(
            text1.rsplit("\n\nAnswer:", 1)[0],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        s10 = self.tok1(
            text0,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        s20 = self.tok2(
            text0,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        s11 = self.tok1(
            text1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        s21 = self.tok2(
            text1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        def make_label(tokens):
            lbl = tokens["input_ids"].clone()
            lbl[tokens["attention_mask"] == 0] = -100
            return lbl

        lbl10 = make_label(s10)
        lbl20 = make_label(s20)
        lbl11 = make_label(s11)
        lbl21 = make_label(s21)
        variant_info_0 = -1 if use_no_reasoning_0 else idx0
        variant_info_1 = -1 if use_no_reasoning_1 else idx1
        return (
            enc0["input_ids"].squeeze(0),
            enc0["attention_mask"].squeeze(0),
            enc1["input_ids"].squeeze(0),
            enc1["attention_mask"].squeeze(0),
            lbl10.squeeze(0),
            s10["input_ids"].squeeze(0),
            s10["attention_mask"].squeeze(0),
            lbl20.squeeze(0),
            s20["input_ids"].squeeze(0),
            s20["attention_mask"].squeeze(0),
            lbl11.squeeze(0),
            s11["input_ids"].squeeze(0),
            s11["attention_mask"].squeeze(0),
            lbl21.squeeze(0),
            s21["input_ids"].squeeze(0),
            s21["attention_mask"].squeeze(0),
            variant_info_0,
            variant_info_1,
        )


def collate_fn(batch):
    cols = list(zip(*batch))

    def stack(i):
        return torch.stack(cols[i])

    return {
        "ids0": stack(0),
        "mask0": stack(1),
        "ids1": stack(2),
        "mask1": stack(3),
        "lbl10": stack(4),
        "sid10": stack(5),
        "sm10": stack(6),
        "lbl20": stack(7),
        "sid20": stack(8),
        "sm20": stack(9),
        "lbl11": stack(10),
        "sid11": stack(11),
        "sm11": stack(12),
        "lbl21": stack(13),
        "sid21": stack(14),
        "sm21": stack(15),
        "cot_idx0": cols[16],
        "cot_idx1": cols[17],
    }


def save_config(config_dict, save_path):
    """Save configuration to JSON file"""
    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_root = pathlib.Path(args.prompt_root) / args.dataset
    out_dir = (
        pathlib.Path(args.outdir) / f"{args.dataset}_{args.kl_weight}_{args.reg_weight}_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dict = {
        **vars(args),
        "timestamp": timestamp,
        "python_version": os.sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    save_config(config_dict, out_dir / "config.json")
    print(f"Saved config to {out_dir / 'config.json'}")
    data = load_prefixes(prompt_root)
    print(f"Loaded {len(data)} samples with CoT variants")
    print(f"No-reasoning probability: {args.no_reasoning_prob}")
    print("Loading models…")
    tok1 = AutoTokenizer.from_pretrained(args.model1)
    if tok1.pad_token is None:
        tok1.pad_token = tok1.eos_token
    tok2 = AutoTokenizer.from_pretrained(args.model2)
    if tok2.pad_token is None:
        tok2.pad_token = tok2.eos_token
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model1 = AutoModelForCausalLM.from_pretrained(
        args.model1,
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        trust_remote_code=False,
    ).eval()
    for p in model1.parameters():
        p.requires_grad = False
    model2 = AutoModelForCausalLM.from_pretrained(
        args.model2,
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        trust_remote_code=False,
    ).eval()
    for p in model2.parameters():
        p.requires_grad = False
    enc_tok = RobertaTokenizer.from_pretrained("roberta-base")
    enc_backbone = RobertaModel.from_pretrained(
        "roberta-base", torch_dtype=torch.float16, trust_remote_code=False
    ).eval()
    for p in enc_backbone.parameters():
        p.requires_grad = False
    student1 = make_student(model1)
    student2 = make_student(model2)
    encoder = CotEncoder(enc_backbone)
    router = Router(enc_backbone.config.hidden_size, args.tau)
    weight_net = WeightLearner(student1.model.config.hidden_size)
    proj1 = HiddenProjector(student1.model.config.hidden_size, student1.model.config.hidden_size)
    proj2 = HiddenProjector(student2.model.config.hidden_size, student1.model.config.hidden_size)
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    pg = InitProcessGroupKwargs(timeout=timedelta(1800))
    acc = Accelerator(kwargs_handlers=[ddp, pg])
    dataset = CotDataset(data, enc_tok, tok1, tok2, args.max_len, args.no_reasoning_prob)
    dl = DataLoader(dataset, batch_size=args.bs, shuffle=True, collate_fn=collate_fn)
    student1_params = [p for p in student1.parameters() if p.requires_grad]
    student2_params = [p for p in student2.parameters() if p.requires_grad]
    other_params = (
        list(router.parameters())
        + list(weight_net.parameters())
        + list(proj1.parameters())
        + list(proj2.parameters())
    )
    opt_groups = [
        {"params": student1_params, "lr": args.lr_s1},
        {"params": student2_params, "lr": args.lr_s2},
        {"params": other_params, "lr": args.lr_misc, "weight_decay": 0.0},
    ]
    opt = bitsandbytes.optim.Adam8bit(opt_groups, betas=(0.9, 0.95))
    student1, student2, proj1, proj2, router, weight_net, opt, dl = acc.prepare(
        student1, student2, proj1, proj2, router, weight_net, opt, dl
    )
    encoder = encoder.to(acc.device)
    print("Starting training...")
    sample_ctr = 0
    no_reasoning_count = 0
    total_count = 0
    student1.train()
    student2.train()
    for epoch in range(1, args.epochs + 1):
        logs = []
        epoch_ratio = epoch / args.epochs
        tau_current = args.tau * (1 - epoch_ratio) + 0.1 * epoch_ratio
        prog = tqdm(dl, desc=f"Epoch {epoch}", disable=not acc.is_main_process)
        for batch in prog:
            idx0, idx1 = (batch["cot_idx0"], batch["cot_idx1"])
            batch = {k: v.to(acc.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            B = batch["sid10"].size(0)
            h0 = encoder(batch["ids0"], batch["mask0"])
            h1 = encoder(batch["ids1"], batch["mask1"])
            gate0 = router(h0, tau_current)
            gate1 = router(h1, tau_current)
            opt.zero_grad()
            losses, balances = [], []
            for view, gate in enumerate((gate0, gate1)):
                out1 = student1(
                    input_ids=batch[f"sid1{view}"],
                    attention_mask=batch[f"sm1{view}"],
                    output_hidden_states=True,
                )
                out2 = student2(
                    input_ids=batch[f"sid2{view}"],
                    attention_mask=batch[f"sm2{view}"],
                    output_hidden_states=True,
                )
                supervised = routed_ce(
                    gate,
                    out1.logits,
                    batch[f"lbl1{view}"],
                    out2.logits,
                    batch[f"lbl2{view}"],
                )
                h1 = proj1(out1.hidden_states[-1])
                h2 = proj2(out2.hidden_states[-1])
                weights = torch.softmax(
                    torch.stack((weight_net(h1.detach()), weight_net(h2.detach())), dim=-1),
                    dim=-1,
                )
                target = weights[..., :1] * h1.detach() + weights[..., 1:] * h2.detach()
                # Preserve the implementation's projected hidden-state consistency term.
                consistency = F.mse_loss(h1.float(), target.float()) + F.mse_loss(
                    h2.float(), target.float()
                )
                losses.append(supervised + args.kl_weight * min(1.0, epoch / 5) * consistency)
                balances.append((gate.mean(dim=0) - 0.5).square().sum())
            loss = sum(losses) / 2 + args.reg_weight * sum(balances) / 2
            acc.backward(loss)
            opt.step()
            r0 = gate0.argmax(-1).tolist()
            r1 = gate1.argmax(-1).tolist()
            batch_no_reasoning = sum((1 for i in [idx0, idx1] for val in i if val == -1))
            no_reasoning_count += batch_no_reasoning
            total_count += 2 * B
            for i in range(B):
                logs.append(
                    {
                        "sample": sample_ctr + i,
                        "cot_idx_0": idx0[i],
                        "route_0": r0[i],
                        "cot_idx_1": idx1[i],
                        "route_1": r1[i],
                        "is_no_reasoning_0": idx0[i] == -1,
                        "is_no_reasoning_1": idx1[i] == -1,
                    }
                )
            sample_ctr += B
            if acc.is_main_process:
                prog.set_postfix(
                    loss=f"{loss.item():.4f}",
                    no_reason=f"{no_reasoning_count / total_count:.3f}"
                    if total_count > 0
                    else "0.000",
                )
        if acc.is_main_process:
            with open(out_dir / f"router_logs_epoch{epoch}.jsonl", "w") as fw:
                for rec in logs:
                    fw.write(json.dumps(rec) + "\n")
            print(f"Saved logs epoch {epoch}")
            print(f"No-reasoning usage ratio: {no_reasoning_count / total_count:.4f}")
        sample_ctr = 0
    if acc.is_main_process:
        final_dir = out_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        final_config = {
            **config_dict,
            "final_epoch": args.epochs,
            "training_completed": True,
            "final_no_reasoning_ratio": no_reasoning_count / total_count if total_count > 0 else 0,
        }
        save_config(final_config, final_dir / "config.json")
        acc.unwrap_model(student1).save_pretrained(final_dir / "student1")
        acc.unwrap_model(student2).save_pretrained(final_dir / "student2")
        torch.save(
            {
                "router": acc.unwrap_model(router).state_dict(),
                "weight_net": acc.unwrap_model(weight_net).state_dict(),
                "proj1": acc.unwrap_model(proj1).state_dict(),
                "proj2": acc.unwrap_model(proj2).state_dict(),
                "encoder": encoder.state_dict(),
                "args": vars(args),
                "final_epoch": args.epochs,
                "timestamp": timestamp,
                "final_no_reasoning_ratio": no_reasoning_count / total_count
                if total_count > 0
                else 0,
            },
            final_dir / "components.pt",
        )
        print(f"Saved final model to {final_dir}")
    print("Training complete.")
    print(f"Final no-reasoning usage ratio: {no_reasoning_count / total_count:.4f}")


if __name__ == "__main__":
    main()
