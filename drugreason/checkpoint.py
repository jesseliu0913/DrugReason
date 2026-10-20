"""Load two independent LoRA experts and persist the router and label space."""

import json
from pathlib import Path
import torch
from .components import FrozenEncoder, Router


def load_bundle(config, checkpoint=None, trainable=True):
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        PeftModel,
        prepare_model_for_kbit_training,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantized = config.get("quantization", "4bit") == "4bit"
    if quantized and device.type != "cuda":
        raise RuntimeError(
            "4-bit training requires CUDA. Use --quantization none for small CPU checks."
        )
    encoder_tokenizer = AutoTokenizer.from_pretrained(config["encoder"])
    backbone = AutoModel.from_pretrained(config["encoder"]).to(device)
    encoder = FrozenEncoder(backbone)
    router = Router(backbone.config.hidden_size, config["tau"]).to(device)
    experts, tokenizers = [], []
    for i, model_id in enumerate(config["models"], 1):
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        if tok.pad_token_id is None:
            raise ValueError("The tokenizer must provide a padding or EOS token.")
        kwargs = {"torch_dtype": torch.float16 if quantized else torch.float32}
        if quantized:
            kwargs.update(
                device_map={"": device.index or 0},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                ),
            )
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if not quantized:
            model = model.to(device)
        model.config.use_cache = False
        if quantized and trainable:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        else:
            model.requires_grad_(False)
        if checkpoint:
            model = PeftModel.from_pretrained(
                model, Path(checkpoint) / f"expert{i}", is_trainable=trainable
            )
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=8,
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    task_type=TaskType.CAUSAL_LM,
                ),
            )
        model.train(trainable)
        experts.append(model)
        tokenizers.append(tok)
    if checkpoint:
        router.load_state_dict(
            torch.load(Path(checkpoint) / "router.pt", map_location=device, weights_only=True)
        )
    router.train(trainable)
    return {
        "experts": experts,
        "tokenizers": tokenizers,
        "encoder": encoder,
        "encoder_tokenizer": encoder_tokenizer,
        "router": router,
        "device": device,
    }


def save_bundle(bundle, config, labels, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for i, (model, tokenizer) in enumerate(zip(bundle["experts"], bundle["tokenizers"]), 1):
        model.save_pretrained(output / f"expert{i}")
        tokenizer.save_pretrained(output / f"expert{i}")
    torch.save(bundle["router"].state_dict(), output / "router.pt")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")


def routing_inputs(bundle, prompts):
    encoded = bundle["encoder_tokenizer"](
        prompts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    encoded = {
        k: v.to(bundle["device"])
        for k, v in encoded.items()
        if k in ("input_ids", "attention_mask")
    }
    return bundle["encoder"](**encoded)
