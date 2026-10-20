from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    BertConfig,
    BertModel,
)
from drugreason.checkpoint import load_bundle, save_bundle
from drugreason.data import GENERATED_VIEWS, validate_training
from drugreason.train import training_loss
from drugreason.evaluate import predict_pair


class TrainingIntegrationTest(unittest.TestCase):
    def test_two_training_stages_and_routed_inference(self):
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        torch.manual_seed(7)
        with (
            patch("torch.cuda.is_available", return_value=False),
            tempfile.TemporaryDirectory(prefix="drugreason-paper-") as folder,
        ):
            root = Path(folder)
            vocab = [
                "[PAD]",
                "[UNK]",
                "positive",
                "negative",
                "relation",
                "Drug",
                "A",
                "B",
                "Evidence",
                "Label",
                ":",
                ".",
                "path",
                "mechanism",
                "example",
                "definition",
            ]
            for i in range(3):
                directory = root / f"model{i}"
                directory.mkdir()
                words = vocab if i != 1 else vocab[:2] + list(reversed(vocab[2:]))
                tokenizer = Tokenizer(
                    WordLevel({word: j for j, word in enumerate(words)}, unk_token="[UNK]")
                )
                tokenizer.pre_tokenizer = Whitespace()
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]"
                )
                tokenizer.save_pretrained(directory)
                options = dict(
                    vocab_size=len(vocab),
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                )
                if i == 0:
                    model = LlamaForCausalLM(LlamaConfig(**options))
                elif i == 1:
                    model = Qwen2ForCausalLM(Qwen2Config(**options))
                else:
                    model = BertModel(
                        BertConfig(
                            vocab_size=len(vocab),
                            hidden_size=16,
                            intermediate_size=32,
                            num_hidden_layers=1,
                            num_attention_heads=2,
                        )
                    )
                model.save_pretrained(directory)
            config = {
                "models": [str(root / "model0"), str(root / "model1")],
                "encoder": str(root / "model2"),
                "quantization": "none",
                "tau": 1.0,
                "max_length": 64,
                "label_batch_size": 4,
                "load_weight": 0.01,
                "mutual_weight": 0.1,
                "temperature": 3.0,
                "dataset": "ddinter",
            }
            labels = [
                {"id": 0, "text": "positive relation"},
                {"id": 1, "text": "negative relation"},
            ]
            pair = {
                "entity_a": "A",
                "entity_b": "B",
                "question": "Drug A. Drug B.",
                "label_id": 0,
                "split": "train",
                "views": [
                    {"kind": "definition", "text": "definition example"},
                    {"kind": "path", "text": "path example"},
                ]
                + [
                    {"kind": kind, "text": "mechanism example", "label_conditioned": False}
                    for kind in GENERATED_VIEWS
                ],
            }
            validate_training([pair])
            bundle = load_bundle(config)
            for stage in ["router", "mutual"]:
                parameters = [
                    p for m in bundle["experts"] for p in m.parameters() if p.requires_grad
                ] + list(bundle["router"].parameters())
                optimizer = torch.optim.AdamW(parameters, lr=0.01)
                optimizer.zero_grad()
                loss = training_loss(bundle, [pair], labels, stage, config)
                assert torch.isfinite(loss)
                loss.backward()
                assert sum(p.grad.abs().sum().item() for p in bundle["router"].parameters()) > 0
                if stage == "mutual":
                    for model in bundle["experts"]:
                        assert (
                            sum(
                                p.grad.abs().sum().item()
                                for p in model.parameters()
                                if p.grad is not None
                            )
                            > 0
                        )
                optimizer.step()
                save_bundle(bundle, config, labels, root / stage)
                bundle = load_bundle(config, root / stage, trainable=stage == "router")
                print(stage, "forward/backward and checkpoint reload passed", flush=True)
            test = dict(pair, views=pair["views"][:2], split="test")
            first = predict_pair(bundle, test, labels, 64, 4)
            second = predict_pair(bundle, dict(test, label_id=1), labels, 64, 4)
            assert first == second and abs(sum(first["probabilities"]) - 1) < 1e-6
            print("Deterministic routed inference and gold-label independence passed", flush=True)


if __name__ == "__main__":
    unittest.main()
