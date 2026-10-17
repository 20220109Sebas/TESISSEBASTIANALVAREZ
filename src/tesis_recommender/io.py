from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

from .schemas import Award, Opportunity


def _date(value: str) -> date:
    return date.fromisoformat(value[:10])


def _amount(value: str) -> float | None:
    return float(value) if value and value.strip() else None


def read_awards(path: str | Path) -> list[Award]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [
            Award(
                award_id=row["award_id"], supplier_id=row["supplier_id"], award_date=_date(row["award_date"]),
                title=row["title"], description=row.get("description", ""), technical_text=row.get("technical_text", ""),
                category=row.get("category", ""), amount=_amount(row.get("amount", "")), region=row.get("region", ""),
                object_type=row.get("object_type", "services"), source_url=row.get("source_url", ""),
                currency=row.get("currency", ""),
            ) for row in csv.DictReader(handle)
        ]


def read_opportunities(path: str | Path) -> list[Opportunity]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [
            Opportunity(
                opportunity_id=row["opportunity_id"], publication_date=_date(row["publication_date"]), title=row["title"],
                description=row.get("description", ""), technical_text=row.get("technical_text", ""), category=row.get("category", ""),
                amount=_amount(row.get("amount", "")), region=row.get("region", ""), object_type=row.get("object_type", "services"),
                source_url=row.get("source_url", ""), awarded_supplier_id=row.get("awarded_supplier_id", ""),
                currency=row.get("currency", ""),
            ) for row in csv.DictReader(handle)
        ]


def read_relevance_labels(path: str | Path) -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = defaultdict(set)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row.get("label", 0)) > 0:
                relevant[row["supplier_id"]].add(row["opportunity_id"])
    return dict(relevant)
