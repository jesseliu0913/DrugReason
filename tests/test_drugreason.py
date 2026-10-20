import math
import unittest
from types import SimpleNamespace
import torch
from drugreason.components import Router
from drugreason.data import (
    GENERATED_VIEWS,
    canonical_label,
    clean_rationale,
    serialize_view,
    validate_training,
)
from drugreason.losses import mutual_kl, objective, pair_mean, score_verbalizers, verbalizer_batch
from drugreason.prompts import teacher_prompt


class Tokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [1 + ord(char) % 13 for char in text]}


class NextTokenModel(torch.nn.Module):
    def __init__(self, vocabulary=20):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(3.0))
        self.vocabulary = vocabulary

    def forward(self, input_ids, attention_mask, logits_to_keep=0):
        preferred = (input_ids + 1) % self.vocabulary
        logits = torch.nn.functional.one_hot(preferred, self.vocabulary).float() * self.scale
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)


class DrugReasonTests(unittest.TestCase):
    def test_complete_label_score_is_length_normalized_and_padding_independent(self):
        tokenizer, model = Tokenizer(), NextTokenModel()
        prompts, labels = ["q", "a much longer prompt"], ["ab", "long label"]
        scores = score_verbalizers(model, tokenizer, prompts, labels, max_length=64)
        for i, (prompt, label) in enumerate(zip(prompts, labels)):
            prefix = tokenizer(prompt)["input_ids"]
            target = tokenizer(" " + label)["input_ids"]
            previous = prefix[-1:] + target[:-1]
            expected = sum(
                (3 if t == (prev + 1) % 20 else 0) - math.log(math.exp(3) + 19)
                for prev, t in zip(previous, target)
            ) / len(target)
            self.assertAlmostEqual(scores[i].item(), expected, places=5)
            single = score_verbalizers(model, tokenizer, [prompt], [label], 64)
            self.assertTrue(torch.allclose(scores[i], single[0]))
        scores.sum().backward()
        self.assertIsNotNone(model.scale.grad)

    def test_candidate_labels_share_the_same_truncated_context(self):
        from drugreason.losses import label_prompt_budget

        tokenizer = Tokenizer()
        labels = ["a", "a longer label"]
        budget = label_prompt_budget(tokenizer, labels, 24)
        batch = verbalizer_batch(tokenizer, ["long context " * 10] * 2, labels, 24, "cpu", budget)
        prefixes = [
            batch["input_ids"][i][
                batch["attention_mask"][i].bool() & ~batch["answer_mask"][i]
            ].tolist()
            for i in range(2)
        ]
        self.assertEqual(prefixes[0], prefixes[1])
        self.assertEqual(len(prefixes[0]), budget)

    def test_verbalizers_are_never_silently_truncated(self):
        with self.assertRaises(ValueError):
            verbalizer_batch(Tokenizer(), ["q"], ["a long label"], 5, "cpu")
        batch = verbalizer_batch(Tokenizer(), ["q" * 100], ["ok"], 8, "cpu")
        self.assertEqual(batch["input_ids"].shape[1], 8)
        self.assertEqual(batch["answer_mask"].sum().item(), len(" ok"))

    def test_symmetric_kl_has_detached_teachers_and_temperature_scaling(self):
        a = torch.tensor([[1.0, 2.0, -1.0]], requires_grad=True)
        b = torch.tensor([[0.0, -1.0, 2.0]], requires_grad=True)
        temperature = 2.5
        loss = mutual_kl(a, b, temperature).sum()
        loss.backward()
        expected = (
            temperature
            / 2
            * ((a.detach() / temperature).softmax(-1) - (b.detach() / temperature).softmax(-1))
        )
        self.assertTrue(torch.allclose(a.grad, expected, atol=1e-6))
        self.assertTrue(torch.allclose(b.grad, -expected, atol=1e-6))
        self.assertTrue(torch.allclose(mutual_kl(a, a, temperature), torch.zeros(1), atol=1e-6))

    def test_entropy_is_of_average_routing_load(self):
        gates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        soft = torch.tensor([[0.99, 0.01], [0.01, 0.99]])
        result = objective(
            torch.zeros(2, 2), gates, soft, torch.zeros(2, dtype=torch.long), load_weight=1
        )
        self.assertAlmostEqual(result.item(), -math.log(2), places=6)
        self.assertEqual(
            pair_mean(torch.tensor([1.0, 3.0, 3.0, 3.0]), torch.tensor([0, 1, 1, 1])).item(), 2.0
        )

    def test_routed_supervision_trains_router_and_selected_expert(self):
        torch.manual_seed(4)
        router = Router(8)
        gates, soft = router(torch.randn(4, 8))
        gold = torch.tensor([[-1.0, -4.0]] * 4, requires_grad=True)
        objective(gold, gates, soft, torch.zeros(4, dtype=torch.long), load_weight=0).backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in router.parameters()), 0)
        self.assertTrue(torch.allclose(gold.grad, -gates.detach() / 4))
        hidden = torch.randn(4, 8)
        first, _ = router(hidden, deterministic=True)
        second, _ = router(hidden, deterministic=True)
        self.assertTrue(torch.equal(first, second))

    def test_input_and_teacher_prompts_do_not_use_gold_labels(self):
        row = {"question": "Drug A: aspirin. Drug B: warfarin.", "label_id": "SECRET"}
        view = {"text": "Grounded evidence."}
        before = serialize_view(row, view)
        row["label_id"] = "DIFFERENT"
        self.assertEqual(before, serialize_view(row, view))
        for kind in GENERATED_VIEWS:
            self.assertNotIn("SECRET", teacher_prompt("aspirin", "warfarin", kind))
            self.assertNotIn("Answer:", teacher_prompt("aspirin", "warfarin", kind))

    def test_filter_removes_explicit_label_fields_without_rewriting(self):
        reasoning = "The drug acts on a shared molecular target in the described pathway."
        self.assertEqual(clean_rationale("Answer: Major\n" + reasoning), reasoning)
        self.assertIsNone(clean_rationale("Answer: Major"))
        self.assertIsNone(clean_rationale("```json\n{}\n```"))

    def test_legacy_conditioned_rationales_cannot_be_relabelled_as_paper_inputs(self):
        row = {
            "split": "train",
            "views": [
                {"kind": kind, "text": "reasoning", "label_conditioned": False}
                for kind in GENERATED_VIEWS
            ],
        }
        validate_training([row])
        row["views"][0]["label_conditioned"] = True
        with self.assertRaises(ValueError):
            validate_training([row])

    def test_closed_label_space_includes_unobserved_no_interaction(self):
        from drugreason.data import label_space

        rows = [
            {"label_idx": i, "label": name} for i, name in enumerate(["Major", "Moderate", "Minor"])
        ]
        labels = label_space(rows, "ddinter")
        self.assertEqual(len(labels), 4)
        self.assertEqual(labels[-1], {"id": 3, "text": "No Interaction"})

    def test_drugbank_verbalizers_preserve_entity_roles(self):
        row = {
            "drug1_name": "Aspirin",
            "drug2_name": "Warfarin",
            "label": "Aspirin may increase the effects of Warfarin.",
        }
        self.assertEqual(
            canonical_label(row, "drugbank"), "drug A may increase the effects of drug B."
        )


if __name__ == "__main__":
    unittest.main()
