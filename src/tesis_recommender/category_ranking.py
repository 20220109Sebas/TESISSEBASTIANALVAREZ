"""Ranking opt-in por códigos publicados, separado del experimento original."""
from __future__ import annotations

import math

import numpy as np

from .category_review import code_keys

SIGNALS = ("bm25", "semantic", "category", "region", "amount")
VARIANTS = {
    "hybrid_cubso_first": {"scheme": "CUBSO", "first_item_only": True, "role": "primary"},
    "hybrid_unspsc_first": {"scheme": "UNSPSC", "first_item_only": True, "role": "primary"},
    "hybrid_cubso_any": {"scheme": "CUBSO", "first_item_only": False, "role": "exploratory_item_scope"},
    "hybrid_unspsc_any": {"scheme": "UNSPSC", "first_item_only": False, "role": "exploratory_item_scope"},
}


def scoped_keys(features: dict, scheme: str, first_item_only: bool) -> set[tuple[str, str]]:
    """Revalida el formato, sin confiar solo en el flag del archivo auxiliar."""
    lengths = {"CUBSO": 16, "UNSPSC": 8}
    if scheme not in lengths:
        raise ValueError("Esquema de clasificación no soportado")
    for entry in features["classifications"]:
        code = entry["code"]
        valid = (entry["scheme"] in lengths and isinstance(code, str) and code.isascii()
                 and code.isdecimal() and len(code) == lengths[entry["scheme"]])
        if entry["valid_supported_code"] != valid:
            raise ValueError("El flag de código no coincide con su formato")
        if not isinstance(entry["item_index"], int) or not 0 <= entry["item_index"] < features["item_count"]:
            raise ValueError("Ítem fuera de su proceso")
    return code_keys(features, scheme, first_item_only)


class CategoryCodeSignal:
    def __init__(self, history: list[dict], opportunities: list[dict], scheme: str, first_item_only: bool):
        self.scheme, self.first_item_only = scheme, first_item_only
        self.candidate_keys = [scoped_keys(r, scheme, first_item_only) for r in opportunities]
        self.profile_keys: dict[str, set] = {}
        self.history = history
        for record in history:
            self.profile_keys.setdefault(record["supplier_id"], set()).update(scoped_keys(record, scheme, first_item_only))

    def scores(self, supplier_id: str) -> np.ndarray:
        if supplier_id not in self.profile_keys:
            raise ValueError("Proveedor sin antecedentes en la representación auxiliar")
        history_keys = self.profile_keys[supplier_id]
        return np.fromiter((float(bool(keys & history_keys)) for keys in self.candidate_keys), dtype=np.float64)

    def explanation(self, supplier_id: str, candidate_index: int) -> dict:
        history_keys = self.profile_keys[supplier_id]
        candidate_keys = self.candidate_keys[candidate_index]
        shared = candidate_keys & history_keys
        status = "unavailable" if not history_keys or not candidate_keys else "observed_match" if shared else "observed_no_exact_match"
        evidence = [{"award_id": r["award_id"], "ocid": r["ocid"], "source_uri": r.get("source_uri", ""),
                     "source_sha256": r["source_sha256"],
                     "shared_codes": [code for _, code in sorted(scoped_keys(r, self.scheme, self.first_item_only) & shared)]}
                    for r in self.history if r["supplier_id"] == supplier_id
                    and scoped_keys(r, self.scheme, self.first_item_only) & shared]
        return {"category_scheme": self.scheme, "category_scope": "first_item" if self.first_item_only else "any_item",
                "category_status": status, "shared_category_codes": [code for _, code in sorted(shared)],
                "candidate_category_codes": [code for _, code in sorted(candidate_keys)],
                "profile_category_code_count": len(history_keys), "historical_category_evidence": evidence,
                "code_validation": "published_scheme_and_format_not_external_catalog_membership",
                "scope_limit": "Affinity to historical procurement processes, not proof of award of every listed item."}


def rank_components(components: dict, weights: dict, candidate_order: list[str], category_override=None):
    """Misma suma Python y desempate que el ranking congelado, sin mutarlo."""
    if len(candidate_order) != len(set(candidate_order)) or not candidate_order:
        raise ValueError("La colección debe ser no vacía y sin duplicados")
    if set(components) != set(SIGNALS):
        raise ValueError("Se requieren exactamente las cinco señales archivadas")
    if (any(name not in SIGNALS or not math.isfinite(value) or value < 0 for name, value in weights.items())
            or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6)):
        raise ValueError("Pesos inválidos")
    signals = dict(components)
    if category_override is not None:
        signals["category"] = np.asarray(category_override, dtype=np.float64)
        if not np.isin(signals["category"], [0., 1.]).all():
            raise ValueError("La categoría por código es una coincidencia binaria")
    for name in SIGNALS:
        values = np.asarray(signals[name])
        if values.shape != (len(candidate_order),) or not np.isfinite(values).all():
            raise ValueError("Señal no finita o de dimensión incorrecta")
    scores = np.fromiter((sum(weights.get(name, 0.0) * float(signals[name][i]) for name in SIGNALS)
                          for i in range(len(candidate_order))), dtype=np.float64, count=len(candidate_order))
    ranking = sorted(range(len(candidate_order)), key=lambda i: (-float(scores[i]), candidate_order[i]))
    return scores, ranking
