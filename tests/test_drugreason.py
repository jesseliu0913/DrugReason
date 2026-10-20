import json
from pathlib import Path
import tempfile
import unittest
import torch
from drugreason.components import Router
from drugreason.data import load_prefixes
from drugreason.losses import per_example_ce, routed_ce, distillation_batch
from drugreason.prompts import teacher_prompt


class Tokenizer:
    eos_token_id, pad_token_id = 1, 0

    def __init__(self, offset=2):
        self.offset = offset

    def __call__(self, text, **kwargs):
        return {"input_ids": [self.offset + ord(c) % 13 for c in text]}


class DrugReasonTests(unittest.TestCase):
    def test_task_loss_reaches_router(self):
        torch.manual_seed(4)
        router = Router(8, 1)
        gate = router(torch.randn(4, 8))
        labels = torch.tensor([[1, 2, 3]] * 4)
        good = torch.zeros(4, 3, 8, requires_grad=True)
        bad = torch.randn(4, 3, 11, requires_grad=True)
        loss = routed_ce(gate, good, labels, bad, labels)
        loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in router.parameters()), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.all(gate.sum(-1) == 1))

    def test_ce_ignores_padding(self):
        logits = torch.randn(1, 4, 8)
        labels = torch.tensor([[1, 2, 3, -100]])
        self.assertTrue(
            torch.allclose(
                per_example_ce(logits, labels), per_example_ce(logits[:, :3], labels[:, :3])
            )
        )
        with self.assertRaises(ValueError):
            per_example_ce(logits, torch.full_like(labels, -100))

    def test_distillation_uses_each_students_tokenizer_and_preserves_targets(self):
        prompts, answers = ["x" * 200, "q"], ["answer", "z"]
        a = distillation_batch(Tokenizer(2), prompts, answers, 12, "cpu")
        b = distillation_batch(Tokenizer(20), prompts, answers, 12, "cpu")
        self.assertEqual(a["input_ids"].shape[1], 12)
        self.assertFalse(torch.equal(a["input_ids"], b["input_ids"]))
        self.assertTrue(torch.all(a["labels"][:, 1:].ne(-100).sum(1) > 0))
        self.assertTrue(torch.all(a["labels"][a["attention_mask"] == 0] == -100))
        with self.assertRaises(ValueError):
            distillation_batch(Tokenizer(), ["q"], [""], 12, "cpu")

    def test_grouping_rejects_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "views.jsonl"
            row = dict(A="a", B="b", question="q", label="yes", prefix="reason")
            p.write_text(
                json.dumps(row) + "\n" + json.dumps(dict(row, prefix={"prefix": "second"})) + "\n"
            )
            self.assertEqual(len(load_prefixes(tmp)[0]["variants"]), 2)
            with p.open("a") as f:
                f.write(json.dumps(dict(row, label="no")) + "\n")
            with self.assertRaises(ValueError):
                load_prefixes(tmp)

    def test_all_teacher_templates_have_a_real_question(self):
        for view in ["cot", "tree_thought", "fb"]:
            prompt = teacher_prompt("Question?", "A", "B", "label", view)
            self.assertIn("Question?", prompt)
            self.assertIn("label", prompt)
            self.assertNotIn("None", prompt)


if __name__ == "__main__":
    unittest.main()
