"""Control de completitud anual, exclusiones y cobertura. No abre etiquetas."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tesis_recommender.contrast_candidates import candidate_queue
from tesis_recommender.validation import read_csv, sha256


def verify(args):
    plan, queue = candidate_queue(args.selection, args.preparation)
    report_path = args.cohort / "candidate_validation_report.json"
    report = json.loads(report_path.read_text())
    fetch = json.loads((args.fetch / "fetch_manifest.json").read_text())
    recovered_path = args.fetch / "candidate_version_results.csv"
    assert sha256(recovered_path) == fetch["results_sha256"]
    recovered = read_csv(recovered_path)
    availability = read_csv(args.cohort / "opportunity_availability.csv")
    opportunities = read_csv(args.cohort / "test_opportunities.csv")
    expected_ids = {row["ocid"] for row in queue}
    assert len(expected_ids) == len(queue) == len(recovered)
    assert {row["url"] for row in recovered} == {row["url"] for row in queue}
    assert all(row["status"] == "ok" for row in recovered)
    assert fetch["retrieval_complete"] and fetch["pending_versions"] == 0
    assert len(availability) == len(expected_ids)
    assert {row["ocid"] for row in availability} == expected_ids
    statuses = dict(Counter(row["status"] for row in availability))
    assert not any(statuses.get(status, 0) for status in ("pending_download", "download_failed"))
    assert statuses == report["candidate_statuses"]
    assert len(opportunities) == report["retained_candidates"] == statuses.get("verified", 0)
    retained_ids = {row["opportunity_id"] for row in opportunities}
    assert len(retained_ids) == len(opportunities)
    assert retained_ids == {row["ocid"] for row in availability if row["status"] == "verified"}
    assert all(not row["awarded_supplier_id"] for row in opportunities)
    assert all(plan["cutoff"] <= row["publication_date"] <= plan["candidate_end"] for row in opportunities)
    for name, digest in report["output_sha256"].items():
        assert sha256(args.cohort / name) == digest
    # El snapshot usado por la auditoría debe coincidir con la recuperación final.
    snapshot_result = next(Path(path) for path in report["input_sha256"] if Path(path).name == "candidate_version_results.csv")
    assert sha256(snapshot_result) == sha256(recovered_path)
    assert report["retrieval_complete"] and report["evaluation_ready"]
    assert not report["labels_read"] and not fetch["labels_read"]
    assert not report["outcome_based_candidate_filter"]
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(), "candidate_versions_planned": len(queue),
            "candidate_versions_recovered": len(recovered), "candidate_opportunities_retained": len(opportunities),
            "exclusions": {key: value for key, value in statuses.items() if key != "verified"},
            "excluded_total": len(expected_ids) - len(opportunities), "pending": 0,
            "month_counts_by_event_date": dict(sorted(Counter(row["publication_date"][:7] for row in opportunities).items())),
            "text_quality": report["text_quality"], "attributes": report["attributes"],
            "publication_hint_discrepancies_over_1s": report["identifier_native_timestamp_discrepancies_over_1s"],
            "eligible_profiles": report["eligible_profiles"], "historical_awards": report["historical_awards"],
            "duplicates_by_ocid": 0, "labels_read": False, "all_candidate_ids_accounted_for": True,
            "no_outcome_based_candidate_filter": True,
            "data_preparation_complete_not_evaluation_complete": True,
            "code_sha256": sha256(Path(__file__)), "report_sha256": sha256(report_path),
            "final_fetch_result_sha256": sha256(recovered_path), "snapshot_fetch_result_sha256": sha256(snapshot_result)}


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Ejecutar sin -O para mantener activas las comprobaciones")
    parser = argparse.ArgumentParser()
    for name in ("selection", "preparation", "fetch", "cohort", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("No sobrescribir un informe anterior")
    report = verify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    code_copy = args.output.with_suffix(".code.py")
    if code_copy.exists():
        raise ValueError("No sobrescribir una copia de verificación anterior")
    shutil.copyfile(Path(__file__), code_copy)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
