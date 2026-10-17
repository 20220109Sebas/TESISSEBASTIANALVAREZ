from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math


@dataclass(frozen=True)
class Award:
    award_id: str
    supplier_id: str
    award_date: date
    title: str
    description: str = ""
    technical_text: str = ""
    category: str = ""
    amount: float | None = None
    region: str = ""
    object_type: str = "services"
    source_url: str = ""
    currency: str = ""


@dataclass(frozen=True)
class Opportunity:
    opportunity_id: str
    publication_date: date
    title: str
    description: str = ""
    technical_text: str = ""
    category: str = ""
    amount: float | None = None
    region: str = ""
    object_type: str = "services"
    source_url: str = ""
    awarded_supplier_id: str = ""
    currency: str = ""

    @property
    def text(self) -> str:
        return " ".join(x for x in (self.title, self.description, self.technical_text) if x).strip()


@dataclass
class CompanyProfile:
    supplier_id: str
    cutoff_date: date
    awards: list[Award] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(
            " ".join(x for x in (a.title, a.description, a.technical_text) if x)
            for a in self.awards
        ).strip()

    @property
    def categories(self) -> set[str]:
        return {a.category.strip().lower() for a in self.awards if a.category.strip()}

    @property
    def regions(self) -> set[str]:
        return {a.region.strip().lower() for a in self.awards if a.region.strip()}

    @property
    def median_amount(self) -> float | None:
        values = sorted(a.amount for a in self.awards if a.amount is not None and a.amount > 0)
        if not values:
            return None
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2

    def median_amount_in(self, currency: str) -> float | None:
        """No mezcla monedas ni supone PEN cuando falta la moneda."""
        currency = currency.strip().upper()
        if not currency:
            return None
        values = sorted(a.amount for a in self.awards if a.currency.strip().upper() == currency
                        and a.amount is not None and math.isfinite(a.amount) and a.amount > 0)
        if not values:
            return None
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


@dataclass(frozen=True)
class RankedOpportunity:
    opportunity_id: str
    score: float
    source_url: str
    explanation: dict[str, float | str]
