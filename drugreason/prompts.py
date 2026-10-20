"""Teacher instructions for DrugReason's generated reasoning views."""


def teacher_prompt(question, entity_a, entity_b, label, view):
    instructions = {
        "cot": f"Focus on the biomedical mechanism that explains this relationship. Describe how {entity_a} and {entity_b} interact through known molecular targets, pathways, or therapeutic processes. Do not restate the answer directly—justify it mechanistically.",
        "tree_thought": f"Think in a tree of biomedical reasoning. Consider multiple plausible biological hypotheses that could explain how {entity_a} and {entity_b} are related—such as shared molecular targets, overlapping metabolic pathways, or downstream signaling effects. Compare these paths and select the one that most strongly supports the observed relationship.",
        "fb": f"Start with forward biomedical reasoning: infer the relationship based on the known properties, mechanisms, and actions of {entity_a}. Then reason backward from the perspective of {entity_b}, validating whether this connection is biologically supported by known interactions, therapeutic roles, or pathway overlap. Conclude only if both directions align.",
    }
    if view not in instructions:
        raise ValueError(f"Unknown generated reasoning view: {view}")
    return f"{question}\nAnswer: {label}\n\n{instructions[view]}\nYour reasoning should be in one paragraph."
