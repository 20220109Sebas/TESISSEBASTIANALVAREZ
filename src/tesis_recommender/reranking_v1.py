"""Bounded text pairs and deterministic optional shortlist reordering."""
from __future__ import annotations

import math


def service_text(item):
    return " ".join(x for x in (item.title, item.description, item.technical_text) if x).strip()


def bounded_pair(tokenizer, history, opportunity):
    query = tokenizer.backend_tokenizer.encode(history, add_special_tokens=False)
    document = tokenizer.backend_tokenizer.encode(opportunity, add_special_tokens=False)
    lengths = len(query.ids), len(document.ids)
    if not all(lengths):
        raise ValueError("Both service texts must be nonempty")
    query.truncate(128, direction="right")
    document.truncate(380, direction="right")
    pair = tokenizer.backend_tokenizer.post_process(query, document, add_special_tokens=True)
    encoded = {"input_ids": pair.ids, "attention_mask": pair.attention_mask}
    if len(encoded["input_ids"]) > 512:
        raise ValueError("Model special tokens exceed the frozen budget")
    return encoded, dict(history_tokens=lengths[0], opportunity_tokens=lengths[1],
        history_truncated=lengths[0] > 128, opportunity_truncated=lengths[1] > 380,
        input_tokens=len(encoded["input_ids"]))


def rerank_prefix(ranking, scores):
    if (not ranking or len(set(ranking)) != len(ranking) or not scores
            or set(scores) != set(ranking[:len(scores)])
            or any(not math.isfinite(v) for v in scores.values())):
        raise ValueError("Scores must cover exactly a unique finite ranking prefix")
    return sorted(scores, key=lambda oid: (-scores[oid], oid)) + ranking[len(scores):]
