"""Representación de categorías para auditoría, no conectada al ranking vigente."""
from __future__ import annotations

import unicodedata

GENERIC_LABELS = {"", "services", "consultingservices"}
CODE_LENGTHS = {"CUBSO": 16, "UNSPSC": 8}


def normalize_label(value: str) -> str:
    """Normaliza Unicode, espacios y caja. Conserva tildes y términos técnicos."""
    return " ".join(unicodedata.normalize("NFC", value or "").split()).casefold()


def informative_label(value: str) -> bool:
    return normalize_label(value) not in GENERIC_LABELS


def legacy_category(tender: dict) -> str:
    items = tender.get("items") or []
    classification = (items[0].get("classification") or {}) if items else {}
    return classification.get("description") or classification.get("id") or tender.get("mainProcurementCategory", "")


def extract_categories(tender: dict) -> dict:
    """Conserva todos los ítems y ambos esquemas sin inferir equivalencias."""
    entries = []
    items = tender.get("items") or []
    for index, item in enumerate(items):
        classifications = [("classification", item.get("classification") or {})]
        classifications += [(f"additionalClassifications[{i}]", value)
                            for i, value in enumerate(item.get("additionalClassifications") or [])]
        for field, classification in classifications:
            if not classification:
                continue
            scheme = str(classification.get("scheme") or "").strip().upper()
            raw_code = classification.get("id")
            code = str(raw_code).strip() if raw_code is not None else ""
            valid = scheme in CODE_LENGTHS and code.isascii() and code.isdecimal() and len(code) == CODE_LENGTHS[scheme]
            entries.append({"item_index": index, "item_id": str(item.get("id") or ""),
                            "field": f"tender.items[{index}].{field}", "scheme": scheme,
                            "code": code, "description": str(classification.get("description") or ""),
                            "valid_supported_code": valid})
    return {"legacy_category": str(legacy_category(tender)), "item_count": len(items), "classifications": entries}


def code_keys(features: dict, scheme: str, first_item_only: bool = False) -> set[tuple[str, str]]:
    """Un identificador solo coincide dentro de su propio esquema."""
    return {(entry["scheme"], entry["code"]) for entry in features["classifications"]
            if entry["scheme"] == scheme and entry["valid_supported_code"]
            and (not first_item_only or entry["item_index"] == 0)}


def primary_labels(features: dict) -> set[str]:
    return {normalize_label(entry["description"]) for entry in features["classifications"]
            if entry["field"].endswith(".classification") and informative_label(entry["description"])}
