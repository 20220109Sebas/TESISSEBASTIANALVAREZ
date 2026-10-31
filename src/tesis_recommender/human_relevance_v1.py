"""Metrics for complete human pools; missing judgments never become negative labels."""
from __future__ import annotations

import math


def evaluate_pool(pool, rankings, judgments):
    expected = {p["pair_id"]: p for p in pool}
    if len(expected) != len(pool) or not expected:
        raise ValueError("Pool must contain unique pairs")
    received = {}
    for row in judgments:
        key, grade = row["pair_id"], row["grade"]
        if key not in expected or key in received:
            raise ValueError("Unknown or duplicate human judgment")
        if not ((type(grade) is int and grade in (0, 1, 2)) or grade == "NE"):
            raise ValueError("Use relevance 0, 1, 2 or NE")
        if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
            raise ValueError("Each human judgment needs a reason")
        received[key] = grade
    if set(received) != set(expected):
        raise ValueError("Incomplete human pool: no metrics or implicit negative labels")
    pairs = {(p["supplier_id"], p["opportunity_id"]): received[key] for key, p in expected.items()}
    if len(pairs) != len(pool):
        raise ValueError("Duplicate company-opportunity pair")
    profiles = sorted({sid for sid, _ in pairs})
    per = {}
    for method, rows in rankings.items():
        if set(rows) != set(profiles):
            raise ValueError("All methods need the same profiles")
        per[method] = {}
        for sid in profiles:
            ranking = rows[sid]
            if len(ranking) != 10 or len(set(ranking)) != 10:
                raise ValueError("Each method needs ten unique opportunities")
            if any((sid, oid) not in pairs for oid in ranking):
                raise ValueError("Ranking outside judged pool")
            grades = [pairs[sid, oid] for oid in ranking]
            profile_pool = [g for (s, _), g in pairs.items() if s == sid]
            values = {"pool_size": len(profile_pool), "pool_NE": profile_pool.count("NE")}
            for k in (5, 10):
                known = sum(g == 2 for g in grades[:k])
                unknown = grades[:k].count("NE")
                values[f"precision_clear_lower@{k}"] = known/k
                values[f"precision_clear_upper@{k}"] = (known+unknown)/k
                values[f"precision_clear@{k}"] = known/k if not unknown else None
                values[f"precision_broad@{k}"] = sum(g in (1, 2) for g in grades[:k])/k if not unknown else None
                values[f"top_NE@{k}"] = unknown
                if "NE" in profile_pool:
                    values[f"ndcg_pool@{k}"] = None
                    values[f"recall_clear_pool@{k}"] = None
                else:
                    dcg = lambda gs: sum((2**g-1)/math.log2(i+2) for i, g in enumerate(gs[:k]))
                    ideal = dcg(sorted(profile_pool, reverse=True))
                    positives = profile_pool.count(2)
                    values[f"ndcg_pool@{k}"] = dcg(grades)/ideal if ideal else None
                    values[f"recall_clear_pool@{k}"] = known/positives if positives else None
            per[method][sid] = values
    macro = {}
    for method, rows in per.items():
        macro[method] = {}
        for key in next(iter(rows.values())):
            if "@" in key:
                values = [r[key] for r in rows.values()]
                macro[method][key] = sum(values)/len(values) if all(v is not None for v in values) else None
    return dict(per_profile=per, macro=macro, profiles=len(profiles), pairs=len(pool),
                judgments_completed=len(judgments), NE=sum(g == "NE" for g in received.values()),
                recall_scope="judged_pool_only_not_entire_collection", relevance_threshold_clear=2,
                evaluator_count=1, inter_rater_agreement=None, statistical_superiority_claim=False)
