from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Iterator

from .schemas import Award, Opportunity


def iter_releases(path: str | Path) -> Iterator[dict]:
    """Lee release/record packages OCDS JSON sin asumir un único envoltorio."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "releases" in payload:
        yield from payload["releases"]
    for record in payload.get("records", []):
        if record.get("compiledRelease"):
            yield record["compiledRelease"]
        else:
            yield from record.get("releases", [])


def _region(release: dict) -> str:
    buyer_id = (release.get("buyer") or release.get("tender", {}).get("procuringEntity") or {}).get("id")
    for party in release.get("parties", []):
        if party.get("id") == buyer_id or "buyer" in party.get("roles", []):
            address = party.get("address", {})
            return address.get("department") or address.get("region") or address.get("locality") or ""
    return ""


def _technical_text(tender: dict) -> str:
    parts: list[str] = []
    for item in tender.get("items", []):
        parts.append(item.get("description", ""))
        parts.append(item.get("classification", {}).get("description", ""))
        parts.extend(value.get("description", "") for value in item.get("additionalClassifications", []))
    return " ".join(part for part in parts if part).strip()


def _category(tender: dict) -> str:
    items = tender.get("items", [])
    if items:
        classification = items[0].get("classification", {})
        return classification.get("description") or classification.get("id") or tender.get("mainProcurementCategory", "")
    return tender.get("mainProcurementCategory", "")


def _value(container: dict) -> float | None:
    amount = container.get("value", {}).get("amount")
    try:
        return float(amount) if amount is not None else None
    except (TypeError, ValueError):
        return None


def transform_release(release: dict) -> tuple[Opportunity | None, list[Award]]:
    tender = release.get("tender", {})
    category = tender.get("mainProcurementCategory", "")
    if category.lower() not in {"services", "consultingservices"}:
        return None, []
    published = (
        tender.get("datePublished")
        or tender.get("tenderPeriod", {}).get("startDate")
        or release.get("publishedDate")
        or release.get("date")
    )
    if not published:
        return None, []
    opportunity_id = release.get("ocid") or tender.get("id") or release.get("id", "")
    source_url = f"https://contratacionesabiertas.oece.gob.pe/api/v1/record/{opportunity_id}"
    region = _region(release)
    title = tender.get("title", "")
    description = tender.get("description", "")
    supplier_ids = [
        supplier.get("id") or supplier.get("name", "")
        for award in release.get("awards", []) for supplier in award.get("suppliers", [])
        if supplier.get("id") or supplier.get("name")
    ]
    opportunity = Opportunity(
        opportunity_id=opportunity_id,
        publication_date=date.fromisoformat(published[:10]),
        title=title,
        description=description, technical_text=_technical_text(tender),
        category=_category(tender),
        amount=_value(tender),
        currency=(tender.get("value") or {}).get("currency", ""),
        region=region,
        object_type="services",
        source_url=source_url, awarded_supplier_id=supplier_ids[0] if supplier_ids else "",
    )
    awards: list[Award] = []
    for award in release.get("awards", []):
        award_date = award.get("date")
        if not award_date:
            continue
        for supplier in award.get("suppliers", []):
            supplier_id = supplier.get("id") or supplier.get("name", "")
            if supplier_id:
                awards.append(Award(
                    award_id=f"{opportunity_id}:{award.get('id', '')}:{supplier_id}", supplier_id=supplier_id,
                    award_date=date.fromisoformat(award_date[:10]), title=title,
                    description=description, technical_text=_technical_text(tender), category=_category(tender), amount=_value(award),
                    region=region, object_type="services", source_url=source_url,
                    currency=(award.get("value") or {}).get("currency", ""),
                ))
    return opportunity, awards


def export_canonical_csv(snapshot_path: str | Path, output_dir: str | Path) -> dict[str, int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    opportunities: dict[str, Opportunity] = {}
    awards: dict[str, Award] = {}
    for release in iter_releases(snapshot_path):
        opportunity, release_awards = transform_release(release)
        if opportunity:
            opportunities[opportunity.opportunity_id] = opportunity
        for award in release_awards:
            awards[award.award_id] = award

    opportunity_fields = [
        "opportunity_id", "publication_date", "title", "description", "technical_text", "category", "amount",
        "region", "object_type", "source_url", "awarded_supplier_id", "currency",
    ]
    award_fields = [
        "award_id", "supplier_id", "award_date", "title", "description", "technical_text", "category", "amount",
        "region", "object_type", "source_url", "currency",
    ]
    with (output_dir / "test_opportunities.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=opportunity_fields)
        writer.writeheader()
        for item in opportunities.values():
            writer.writerow({name: getattr(item, name) for name in opportunity_fields})
    with (output_dir / "historical_awards.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=award_fields)
        writer.writeheader()
        for item in awards.values():
            writer.writerow({name: getattr(item, name) for name in award_fields})
    return {"service_opportunities": len(opportunities), "service_awards": len(awards)}
