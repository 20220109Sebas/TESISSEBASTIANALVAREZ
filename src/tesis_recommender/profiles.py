from __future__ import annotations

from collections import defaultdict
from datetime import date

from .schemas import Award, CompanyProfile


def build_profiles(
    awards: list[Award], cutoff_date: date, minimum_awards: int = 2
) -> dict[str, CompanyProfile]:
    grouped: dict[str, list[Award]] = defaultdict(list)
    for award in awards:
        if award.award_date < cutoff_date:
            grouped[award.supplier_id].append(award)
    return {
        supplier_id: CompanyProfile(supplier_id, cutoff_date, sorted(items, key=lambda x: x.award_date))
        for supplier_id, items in grouped.items()
        if len(items) >= minimum_awards
    }


def assert_no_temporal_leakage(profiles: dict[str, CompanyProfile]) -> None:
    leaked = [
        f"{profile.supplier_id}:{award.award_id}"
        for profile in profiles.values()
        for award in profile.awards
        if award.award_date >= profile.cutoff_date
    ]
    if leaked:
        raise ValueError(f"Fuga temporal detectada: {', '.join(leaked[:10])}")
