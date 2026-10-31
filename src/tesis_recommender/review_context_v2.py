"""Source-bound facts for human inspection, without suggested relevance judgments."""
from __future__ import annotations

import math
import unicodedata


def normalized(value):
    return " ".join("".join(c for c in unicodedata.normalize("NFD", value or "").lower()
                            if unicodedata.category(c) != "Mn").split())


def usable_amount(row):
    value = row.get("amount")
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0 and bool((row.get("currency") or "").strip()))


def facts(case, opportunity):
    history = case["history"]
    if not history or len({a["award_id"] for a in history}) != len(history):
        raise ValueError("Require unique historical awards")
    if any(a["award_date"] >= case["cutoff"] for a in history):
        raise ValueError("Historical context must precede cutoff")
    currency = (opportunity.get("currency") or "").strip().upper()
    positive = [a for a in history if usable_amount(a)]
    comparable = [a for a in positive if a["currency"].strip().upper() == currency]
    category = normalized(opportunity.get("category"))
    exact = [a for a in history if category and normalized(a.get("category")) == category]
    categories = []
    groups = {}
    for i, a in enumerate(history, 1):
        key = normalized(a.get("category"))
        if key not in groups:
            groups[key] = dict(label=a.get("category") or "Sin categoría disponible", antecedents=[])
            categories.append(groups[key])
        groups[key]["antecedents"].append(i)
    amount_valid = usable_amount(opportunity)
    maximum = max((a["amount"] for a in comparable), default=None)
    total = math.fsum(a["amount"] for a in comparable) if comparable else None
    exact_amounts = [a["amount"] for a in exact if usable_amount(a) and a["currency"].strip().upper() == currency]
    return dict(history_count=len(history), history_first=min(a["award_date"] for a in history),
        history_last=max(a["award_date"] for a in history), categories=categories,
        exact_category_antecedents=[i for i,a in enumerate(history, 1) if a in exact],
        category_comparison="normalized_description_only_not_taxonomy_equivalence",
        opportunity_currency=currency, opportunity_amount=opportunity.get("amount"),
        opportunity_amount_comparable=amount_valid, comparable_history_count=len(comparable),
        other_currency_history_count=len(positive)-len(comparable),
        missing_or_nonpositive_history_count=len(history)-len(positive),
        maximum=maximum, total=total, same_category_maximum=max(exact_amounts, default=None),
        ratio_to_maximum=opportunity["amount"]/maximum if amount_valid and maximum else None,
        ratio_to_total=opportunity["amount"]/total if amount_valid and total else None,
        duration_verified=False, scope_verified=False, relevance_grade_suggested=None,
        contract_requirement_extracted=False)
