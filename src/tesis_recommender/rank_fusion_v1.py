"""Optional rank-based fusion; archived weighted rankings remain unchanged."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def ordered_ids(ids: Sequence[str], scores: Sequence[float]) -> list[str]:
    if (not ids or len(set(ids)) != len(ids) or len(ids) != len(scores)
            or any(not isinstance(i, str) or not i for i in ids)
            or any(not math.isfinite(float(s)) for s in scores)):
        raise ValueError("Expected a complete unique population and finite scores")
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))]


def reciprocal_rank_fusion(rankings: Mapping[str, Sequence[str]], constant: int = 60):
    if isinstance(constant, bool) or not isinstance(constant, int) or constant < 1:
        raise ValueError("RRF constant must be a positive integer")
    if not rankings:
        raise ValueError("At least one complete ranking is required")
    population = None
    positions = {}
    for name, ranking in sorted(rankings.items()):
        if not name or not ranking or len(set(ranking)) != len(ranking):
            raise ValueError("Empty or duplicated ranking")
        if any(not isinstance(i, str) or not i for i in ranking):
            raise ValueError("Invalid document identity")
        current = set(ranking)
        if population is not None and current != population:
            raise ValueError("Fusion requires identical complete candidate populations")
        population = current
        positions[name] = {oid: rank for rank, oid in enumerate(ranking, 1)}
    scores = {oid: sum(1.0 / (constant + positions[name][oid]) for name in sorted(positions))
              for oid in sorted(population)}
    ranked = sorted(scores, key=lambda oid: (-scores[oid], oid))
    return ranked, scores, positions
