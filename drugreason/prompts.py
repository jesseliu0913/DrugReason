"""Label-free teacher prompts from the draft's prompt-template appendix."""

from .data import GENERATED_VIEWS


def teacher_prompt(entity_a, entity_b, view):
    templates = {
        "mechanism": f"Focus on the biomedical mechanism connecting {entity_a} and {entity_b}. Describe relevant molecular targets, pathways, pharmacological processes, or therapeutic mechanisms. Do not restate the answer directly; justify the relationship mechanistically.",
        "chain_of_thought": f"Reason step by step about the possible biomedical relationship between {entity_a} and {entity_b}. Consider pharmacological effects, disease mechanisms, pathway connections, and interaction evidence.",
        "tree_of_thought": f"Generate multiple plausible biomedical hypotheses for the relationship between {entity_a} and {entity_b}. Compare these hypotheses and identify which one is most biologically plausible.",
        "forward_backward": f"First reason forward from {entity_a} to {entity_b}. Then reason backward from {entity_b} to {entity_a}. Check whether the two directions support a consistent biomedical relationship.",
    }
    if view not in GENERATED_VIEWS:
        raise ValueError(f"Unknown reasoning view: {view}")
    return templates[view]
