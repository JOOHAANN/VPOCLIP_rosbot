"""Decision policies and score utilities for realtime recognition."""

import math

import torch

UNKNOWN_LABEL = -1
UNKNOWN_NAME = "unknown"
ANALYSING_LABEL = -2
ANALYSING_NAME = "analysing"
PERSON_SLOT = 0


def normalized_entropy(cosine, temperature):
    """Softmax entropy over the prototype scores, rescaled to [0,1].

    The learned logit scale saturates the softmax, so the cosine scores are
    re-softmaxed at a fixed temperature instead. 1.0 means "all classes look
    equally likely", 0.0 means "one class dominates".
    """

    scores = torch.as_tensor(cosine, dtype=torch.float32).flatten()
    if scores.numel() < 2:
        return 0.0
    probs = torch.softmax(scores / max(temperature, 1e-6), dim=0)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    return float(entropy / math.log(probs.numel()))


def yolo_person_present(object_map, person_slot=PERSON_SLOT):
    """Read YOLO person presence from an object RS map.

    Supported layouts are [O,H,W], [B,O,H,W], and [B,1,O,H,W]. Class slot 0
    is person for the detector used by this project.
    """

    if object_map is None:
        return False
    values = torch.as_tensor(object_map)
    if values.ndim < 3:
        return False
    object_axis = values.ndim - 3
    if person_slot < 0 or person_slot >= values.shape[object_axis]:
        return False
    return bool(values.select(object_axis, int(person_slot)).amax().item() > 0.0)


class PersonEntropyDecisionGate:
    """Require a YOLO person, then accept a class only when H is low.

    Prototype cosine remains in the returned diagnostics, but it is not part
    of the decision. Entropy is computed from raw cosine similarities.
    """

    def __init__(self, candidate_labels, unseen_labels, config=None):
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.candidate_labels = [int(label) for label in candidate_labels]
        self.unseen_labels = {int(label) for label in unseen_labels}
        self.person_slot = int(config.get("person_slot", PERSON_SLOT))
        self.temperature = float(config.get("temperature", 0.1))
        self.entropy_threshold = float(config.get("entropy_threshold", 0.30))
        if self.temperature <= 0.0:
            raise ValueError("decision_gate.temperature must be positive")
        if not 0.0 <= self.entropy_threshold <= 1.0:
            raise ValueError(
                "decision_gate.entropy_threshold must be in [0, 1], "
                f"got {self.entropy_threshold}"
            )

    def __call__(
        self,
        raw_cosine,
        unit_cosine,
        top_label,
        *,
        person_present=None,
        object_map=None,
    ):
        raw_cosine = torch.as_tensor(raw_cosine, dtype=torch.float32).flatten().cpu()
        unit_cosine = torch.as_tensor(unit_cosine, dtype=torch.float32).flatten().cpu()
        if raw_cosine.numel() != len(self.candidate_labels):
            raise ValueError(
                f"Expected {len(self.candidate_labels)} raw scores, "
                f"got {raw_cosine.numel()}"
            )
        if unit_cosine.numel() != len(self.candidate_labels):
            raise ValueError(
                f"Expected {len(self.candidate_labels)} unit scores, "
                f"got {unit_cosine.numel()}"
            )

        top_label = int(top_label)
        if top_label not in self.candidate_labels:
            raise ValueError(f"Top label {top_label} is outside the candidate set")
        top_index = self.candidate_labels.index(top_label)
        prototype_score = float(unit_cosine[top_index])
        raw_prototype_score = float(raw_cosine[top_index])
        entropy = normalized_entropy(raw_cosine, self.temperature)
        is_unseen = top_label in self.unseen_labels
        if person_present is None:
            person_present = yolo_person_present(object_map, self.person_slot)
        person_present = bool(person_present)

        info = {
            "label": top_label,
            "raw_label": top_label,
            "cosine": prototype_score,
            "raw_cosine": raw_prototype_score,
            "entropy": entropy,
            "person": person_present,
            "is_unseen": is_unseen,
            "reason": "accepted",
            "decision": "class",
        }
        if not self.enabled:
            info["reason"] = "person-entropy gate off"
            return info

        if not person_present:
            info["label"] = ANALYSING_LABEL
            info["reason"] = "no person"
            info["decision"] = "analysing"
            return info

        if entropy < self.entropy_threshold:
            info["reason"] = "person present, low H"
            return info

        info["label"] = ANALYSING_LABEL
        info["reason"] = "H too high"
        info["decision"] = "analysing"
        return info


def label_name(label, names=None):
    if label == UNKNOWN_LABEL:
        return UNKNOWN_NAME
    if label == ANALYSING_LABEL:
        return ANALYSING_NAME
    if names and int(label) in names:
        return names[int(label)]
    return f"class {int(label)}"
