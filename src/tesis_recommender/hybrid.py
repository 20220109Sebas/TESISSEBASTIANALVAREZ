from __future__ import annotations

import math

from .bm25 import BM25
from .schemas import CompanyProfile, Opportunity, RankedOpportunity
from .semantic import Embedder, cosine


DEFAULT_WEIGHTS = {"bm25": 0.35, "semantic": 0.50, "category": 0.08, "region": 0.04, "amount": 0.03}
GENERIC_CATEGORIES = {"services", "consultingservices"}


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def structured_signals(profile: CompanyProfile, opportunity: Opportunity) -> dict[str, float]:
    candidate_category = opportunity.category.strip().lower()
    informative_categories = profile.categories - GENERIC_CATEGORIES
    category = float(bool(candidate_category) and candidate_category not in GENERIC_CATEGORIES
                     and candidate_category in informative_categories)
    region = float(bool(opportunity.region.strip()) and opportunity.region.strip().lower() in profile.regions)
    amount = 0.0
    median = profile.median_amount_in(opportunity.currency)
    if median and opportunity.amount is not None and math.isfinite(opportunity.amount) and opportunity.amount > 0:
        amount = math.exp(-abs(math.log(opportunity.amount) - math.log(median)))
    return {"category": category, "region": region, "amount": amount}


class HybridRanker:
    def __init__(self, opportunities: list[Opportunity], embedder: Embedder) -> None:
        self.opportunities = opportunities
        self.embedder = embedder
        self.bm25 = BM25().fit([item.text for item in opportunities])
        self.document_embeddings = embedder.encode([item.text for item in opportunities])
        self._component_cache: dict[str, dict[str, list[float]]] = {}

    def component_scores(self, profile: CompanyProfile) -> dict[str, list[float]]:
        if profile.supplier_id in self._component_cache:
            return self._component_cache[profile.supplier_id]
        lexical = minmax(self.bm25.score(profile.text))
        profile_vector = self.embedder.encode([profile.text])[0]
        semantic = minmax([cosine(profile_vector, vector) for vector in self.document_embeddings])
        structured = [structured_signals(profile, item) for item in self.opportunities]
        result = {
            "bm25": lexical,
            "semantic": semantic,
            "category": [item["category"] for item in structured],
            "region": [item["region"] for item in structured],
            "amount": [item["amount"] for item in structured],
        }
        self._component_cache[profile.supplier_id] = result
        return result

    def rank(
        self, profile: CompanyProfile, weights: dict[str, float] | None = None, top_k: int | None = 10
    ) -> list[RankedOpportunity]:
        weights = DEFAULT_WEIGHTS if weights is None else weights
        if any(name not in DEFAULT_WEIGHTS or not math.isfinite(value) or value < 0 for name, value in weights.items()):
            raise ValueError("Los pesos deben ser finitos, no negativos y de señales conocidas")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
            raise ValueError("top_k debe ser un entero positivo")
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
            raise ValueError("Los pesos deben sumar 1")
        components = self.component_scores(profile)
        ranked: list[RankedOpportunity] = []
        for index, opportunity in enumerate(self.opportunities):
            contributions = {name: weights.get(name, 0.0) * values[index] for name, values in components.items()}
            score = sum(contributions.values())
            dominant = max(contributions, key=contributions.get)
            explanation: dict[str, float | str] = {
                **{f"{name}_score": round(components[name][index], 6) for name in components},
                **{f"{name}_contribution": round(value, 6) for name, value in contributions.items()},
                "category_status": "generic_not_informative" if opportunity.category.strip().lower() in GENERIC_CATEGORIES
                else "observed" if opportunity.category.strip() and profile.categories - GENERIC_CATEGORIES else "unavailable",
                "region_status": "observed" if opportunity.region.strip() and profile.regions else "unavailable",
                "amount_status": "observed_same_currency" if profile.median_amount_in(opportunity.currency) and opportunity.amount is not None and math.isfinite(opportunity.amount) and opportunity.amount > 0 else "unavailable_or_not_comparable",
                "missing_policy": "zero_contribution_fixed_weights_not_evidence_of_incompatibility",
                "dominant_signal": dominant,
            }
            ranked.append(RankedOpportunity(opportunity.opportunity_id, score, opportunity.source_url, explanation))
        ranked.sort(key=lambda item: (-item.score, item.opportunity_id))
        return ranked if top_k is None else ranked[:top_k]
