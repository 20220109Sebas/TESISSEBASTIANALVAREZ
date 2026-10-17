from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from .ocds import transform_release
from .schemas import Award, Opportunity
from .validation import version_hint


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _initial_tender_reference(record: dict) -> tuple[str, str]:
    candidates = [item for item in record.get("releases", []) if "tender" in item.get("tag", []) and item.get("url")]
    if not candidates:
        return "", ""
    known_versions = [item for item in candidates if version_hint(item)]
    selected = min(known_versions, key=version_hint) if known_versions else min(candidates, key=lambda item: item.get("date", "9999"))
    return selected.get("url", ""), selected.get("date", "")


def _write_dataclasses(path: Path, rows: list[object], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            writer.writerow({field: payload[field] for field in fields})


def build_corpus(
    raw_root: str | Path,
    output_dir: str | Path,
    start_date: date = date(2022, 1, 1),
    cutoff_date: date = date(2025, 1, 1),
    end_date: date = date(2025, 12, 31),
) -> dict[str, object]:
    raw_root, output_dir = Path(raw_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_paths = sorted(raw_root.glob("*/*/*.json"))
    selected: dict[str, dict[str, object]] = {}
    version_counts: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    snapshot_stats: list[dict[str, object]] = []
    total_records = 0

    for snapshot_path in snapshot_paths:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        total_records += len(records)
        service_count = 0
        for record in records:
            ocid = record.get("ocid")
            if not ocid:
                exclusions["missing_ocid"] += 1
                continue
            release = record.get("compiledRelease")
            if not release:
                exclusions["missing_compiled_release"] += 1
                continue
            tender = release.get("tender", {})
            if tender.get("mainProcurementCategory", "").lower() not in {"services", "consultingservices"}:
                exclusions["outside_service_scope"] += 1
                continue
            service_count += 1
            published = tender.get("datePublished")
            if not published:
                exclusions["missing_tender_date_published"] += 1
                continue
            try:
                publication_date = _parse_datetime(published).date()
            except ValueError:
                exclusions["invalid_tender_date_published"] += 1
                continue
            if publication_date < start_date or publication_date > end_date:
                exclusions["outside_date_window"] += 1
                continue
            opportunity, awards = transform_release(release)
            if not opportunity or not opportunity.text.strip():
                exclusions["insufficient_text"] += 1
                continue
            version_counts[ocid] += 1
            initial_url, initial_date = _initial_tender_reference(record)
            compiled_date = release.get("date", "")
            snapshot_id = snapshot_path.parent.parent.name + "-" + snapshot_path.parent.name
            candidate = {
                "opportunity": opportunity,
                "awards": awards,
                "compiled_date": compiled_date,
                "snapshot_id": snapshot_id,
                "initial_release_url": initial_url,
                "initial_release_date": initial_date,
            }
            previous = selected.get(ocid)
            if previous is None or compiled_date > str(previous["compiled_date"]):
                selected[ocid] = candidate
        snapshot_stats.append({
            "snapshot": str(snapshot_path), "records": len(records), "service_records": service_count,
            "sha256": _hash(snapshot_path),
        })

    opportunities = sorted((item["opportunity"] for item in selected.values()), key=lambda item: (item.publication_date, item.opportunity_id))
    historical_awards: list[Award] = []
    labels: list[dict[str, str | int]] = []
    lineage: list[dict[str, str | int]] = []
    for ocid, item in selected.items():
        opportunity: Opportunity = item["opportunity"]  # type: ignore[assignment]
        awards: list[Award] = item["awards"]  # type: ignore[assignment]
        if opportunity.publication_date < cutoff_date:
            historical_awards.extend(award for award in awards if start_date <= award.award_date < cutoff_date)
        else:
            labels.extend({
                "supplier_id": award.supplier_id, "opportunity_id": opportunity.opportunity_id,
                "label": 1, "judgment_source": "historical_award", "award_date": award.award_date.isoformat(),
            } for award in awards)
        lineage.append({
            "ocid": ocid, "publication_date": opportunity.publication_date.isoformat(),
            "selected_snapshot_id": item["snapshot_id"], "version_count": version_counts[ocid],
            "compiled_release_date": item["compiled_date"], "initial_release_date": item["initial_release_date"],
            "initial_release_url": item["initial_release_url"],
            "selection_reason": "latest_compiled_for_outcome; initial_release_required_for_features",
        })

    duplicate_award_rows = len(historical_awards) - len({item.award_id for item in historical_awards})
    historical_awards = list({item.award_id: item for item in historical_awards}.values())
    duplicate_label_rows = len(labels) - len({(item["supplier_id"], item["opportunity_id"]) for item in labels})
    labels = list({(item["supplier_id"], item["opportunity_id"]): item for item in labels}.values())
    historical_awards.sort(key=lambda item: (item.award_date, item.award_id))
    test_opportunities = [item for item in opportunities if item.publication_date >= cutoff_date]
    historical_opportunities = [item for item in opportunities if item.publication_date < cutoff_date]
    opportunity_fields = [
        "opportunity_id", "publication_date", "title", "description", "technical_text", "category", "amount",
        "region", "object_type", "source_url", "awarded_supplier_id", "currency",
    ]
    award_fields = [
        "award_id", "supplier_id", "award_date", "title", "description", "technical_text", "category", "amount",
        "region", "object_type", "source_url", "currency",
    ]
    _write_dataclasses(output_dir / "historical_awards.csv", historical_awards, award_fields)
    _write_dataclasses(output_dir / "historical_opportunities.csv", historical_opportunities, opportunity_fields)
    _write_dataclasses(output_dir / "test_opportunities.csv", test_opportunities, opportunity_fields)
    with (output_dir / "historical_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["supplier_id", "opportunity_id", "label", "judgment_source", "award_date"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(labels)
    with (output_dir / "lineage.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["ocid", "publication_date", "selected_snapshot_id", "version_count", "compiled_release_date", "initial_release_date", "initial_release_url", "selection_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(sorted(lineage, key=lambda row: row["ocid"]))

    supplier_counts = Counter(item.supplier_id for item in historical_awards)
    report: dict[str, object] = {
        "window": {"start": start_date.isoformat(), "cutoff": cutoff_date.isoformat(), "end": end_date.isoformat()},
        "snapshot_count": len(snapshot_paths), "raw_records": total_records, "unique_service_ocids": len(selected),
        "duplicate_versions_removed": sum(count - 1 for count in version_counts.values()),
        "historical_opportunities": len(historical_opportunities), "test_opportunities": len(test_opportunities),
        "historical_awards_before_cutoff": len(historical_awards), "test_positive_labels": len(labels),
        "duplicate_award_rows_removed": duplicate_award_rows, "duplicate_label_rows_removed": duplicate_label_rows,
        "unique_historical_suppliers": len(supplier_counts),
        "profiles_ge_2_awards": sum(value >= 2 for value in supplier_counts.values()),
        "profiles_ge_3_awards": sum(value >= 3 for value in supplier_counts.values()),
        "exclusions": dict(exclusions), "snapshots": snapshot_stats,
        "temporal_warning": "Features usan compiledRelease. Recuperar y verificar release.publishedDate y evidencia de adjudicación anterior al corte. La fecha del evento no acredita disponibilidad histórica.",
    }
    output_hashes = {
        path.name: _hash(path) for path in output_dir.glob("*.csv")
    }
    report["output_sha256"] = output_hashes
    (output_dir / "corpus_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
