from __future__ import annotations

from collections import Counter
from datetime import date
import math

from .schemas import Award, Opportunity


def audit_dataset(awards: list[Award], opportunities: list[Opportunity], cutoff: date) -> dict[str, object]:
    award_ids = [item.award_id for item in awards]
    opportunity_ids = [item.opportunity_id for item in opportunities]
    supplier_counts = Counter(item.supplier_id for item in awards if item.award_date < cutoff)
    usable = [item for item in opportunities if item.text.strip()]
    return {
        "counts": {"awards": len(awards), "opportunities": len(opportunities), "eligible_profiles_ge_2": sum(v >= 2 for v in supplier_counts.values())},
        "uniqueness": {
            "duplicate_award_ids": len(award_ids) - len(set(award_ids)),
            "duplicate_opportunity_ids": len(opportunity_ids) - len(set(opportunity_ids)),
        },
        "completeness": {
            "opportunities_with_usable_text_rate": len(usable) / len(opportunities) if opportunities else 0.0,
            "category_coverage": sum(bool(item.category) for item in opportunities) / len(opportunities) if opportunities else 0.0,
            "amount_coverage": sum(item.amount is not None for item in opportunities) / len(opportunities) if opportunities else 0.0,
            "usable_amount_currency_coverage": sum(item.amount is not None and math.isfinite(item.amount) and item.amount > 0 and bool(item.currency.strip()) for item in opportunities) / len(opportunities) if opportunities else 0.0,
            "informative_category_coverage": sum(bool(item.category.strip()) and item.category.strip().lower() not in {"services", "consultingservices"} for item in opportunities) / len(opportunities) if opportunities else 0.0,
            "region_coverage": sum(bool(item.region) for item in opportunities) / len(opportunities) if opportunities else 0.0,
            "source_url_coverage": sum(bool(item.source_url) for item in opportunities) / len(opportunities) if opportunities else 0.0,
        },
        "validity": {
            "non_service_records": sum(item.object_type.lower() != "services" for item in opportunities),
            "nonpositive_amounts": sum(item.amount is not None and item.amount <= 0 for item in opportunities),
            "test_records_before_cutoff": sum(item.publication_date < cutoff for item in opportunities),
            "historical_awards_on_or_after_cutoff": sum(item.award_date >= cutoff for item in awards),
        },
        "thresholds": {
            "opportunities_target_500_met": len(opportunities) >= 500,
            "profiles_target_50_met": sum(v >= 2 for v in supplier_counts.values()) >= 50,
            "profiles_minimum_30_met": sum(v >= 2 for v in supplier_counts.values()) >= 30,
        },
        "label_warning": "Una adjudicación futura es positiva; las demás oportunidades quedan no etiquetadas, no son negativos confirmados.",
    }
