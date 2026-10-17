from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .ocds import transform_release


def _read_queue(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_payload(payload: dict, expected_ocid: str) -> dict:
    releases = payload.get("releases", [])
    if len(releases) != 1:
        raise ValueError(f"Se esperaba una entrega y se recibieron {len(releases)}")
    release = releases[0]
    if release.get("ocid") != expected_ocid:
        raise ValueError(f"OCID inesperado: {release.get('ocid')}")
    if "tender" not in release.get("tag", []):
        raise ValueError("La primera entrega no contiene la etiqueta tender")
    tender = release.get("tender", {})
    if not tender.get("datePublished"):
        raise ValueError("La primera entrega no contiene tender.datePublished")
    return release


def _fetch_one(row: dict[str, str], cache_dir: Path, retries: int) -> dict[str, object]:
    cache_path = cache_dir / f"{row['cache_key']}.json"
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            release = _validate_payload(payload, row["ocid"])
            return _result(row, cache_path, payload, release, "cached", 0)
        except (ValueError, json.JSONDecodeError):
            cache_path.unlink()
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            temporary = cache_path.with_suffix(".json.part")
            process = subprocess.run(
                ["curl", "-fsSL", "--connect-timeout", "15", "--max-time", "60", "--retry", "2", "--retry-all-errors",
                 "-A", "tesis-recommender/0.1", row["initial_release_url"], "-o", str(temporary)],
                capture_output=True, text=True, timeout=210,
            )
            if process.returncode:
                raise RuntimeError(process.stderr.strip() or f"curl finalizó con código {process.returncode}")
            raw = temporary.read_bytes()
            payload = json.loads(raw)
            release = _validate_payload(payload, row["ocid"])
            temporary.replace(cache_path)
            return _result(row, cache_path, payload, release, "downloaded", attempt)
        except (RuntimeError, subprocess.TimeoutExpired, TimeoutError, ValueError, json.JSONDecodeError) as error:
            last_error = str(error)
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    return {**row, "status": "failed", "cache_path": str(cache_path), "sha256": "", "attempts": retries,
            "tender_date_published": "", "main_procurement_category": "", "text_length": 0, "error": last_error}


def _result(row: dict[str, str], cache_path: Path, payload: dict, release: dict, status: str, attempts: int) -> dict[str, object]:
    tender = release.get("tender", {})
    text = " ".join([tender.get("title", ""), tender.get("description", "")] + [item.get("description", "") for item in tender.get("items", [])]).strip()
    return {
        **row, "status": status, "cache_path": str(cache_path),
        "sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(), "attempts": attempts,
        "tender_date_published": tender.get("datePublished", ""),
        "main_procurement_category": tender.get("mainProcurementCategory", ""),
        "text_length": len(text), "error": "",
    }


def _write_results(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["ocid", "role", "initial_release_url", "initial_release_date", "cache_key", "status", "cache_path", "sha256", "attempts", "tender_date_published", "main_procurement_category", "text_length", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda row: str(row["ocid"])))


def download_initial_releases(
    queue_path: str | Path, cache_dir: str | Path, output_path: str | Path, workers: int = 8, retries: int = 3
) -> dict[str, object]:
    queue_path, cache_dir, output_path = Path(queue_path), Path(cache_dir), Path(output_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    queue = _read_queue(queue_path)
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, row, cache_dir, retries): row["ocid"] for row in queue}
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 25 == 0:
                _write_results(output_path, results)
    _write_results(output_path, results)
    counts = {status: sum(row["status"] == status for row in results) for status in ("downloaded", "cached", "failed")}
    return {"requested": len(queue), **counts, "output": str(output_path)}


def materialize_temporal_cohort(cohort_dir: str | Path, output_dir: str | Path) -> dict[str, object]:
    cohort_dir, output_dir = Path(cohort_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = _read_queue(cohort_dir / "initial_release_results.csv")
    failed = [row for row in results if row["status"] == "failed"]
    expected = {row["ocid"] for row in _read_queue(cohort_dir / "initial_release_queue.csv")}
    if failed or {row["ocid"] for row in results} != expected or len(results) != len(expected):
        raise ValueError(f"La cola no está completa: resultados={len(results)}, fallos={len(failed)}")
    release_by_ocid = {}
    for row in results:
        payload = json.loads(Path(row["cache_path"]).read_text(encoding="utf-8"))
        release_by_ocid[row["ocid"]] = _validate_payload(payload, row["ocid"])

    compiled_candidates = _read_queue(cohort_dir / "cohort_test_opportunities_compiled.csv")
    historical_awards = _read_queue(cohort_dir / "cohort_historical_awards.csv")
    cohort_profiles = _read_queue(cohort_dir / "cohort_profiles.csv")
    labels = _read_queue(cohort_dir / "cohort_labels.csv")
    date_mismatches = 0
    initial_candidates: list[dict[str, object]] = []
    opportunity_fields = ["opportunity_id", "publication_date", "title", "description", "technical_text", "category", "amount", "region", "object_type", "source_url", "awarded_supplier_id", "currency"]
    for compiled in compiled_candidates:
        opportunity, _ = transform_release(release_by_ocid[compiled["opportunity_id"]])
        if not opportunity:
            raise ValueError(f"Primera entrega no transformable: {compiled['opportunity_id']}")
        date_mismatches += int(opportunity.publication_date.isoformat() != compiled["publication_date"])
        payload = {field: getattr(opportunity, field) for field in opportunity_fields}
        payload["awarded_supplier_id"] = compiled["awarded_supplier_id"]
        initial_candidates.append(payload)

    award_fields = ["award_id", "supplier_id", "award_date", "title", "description", "technical_text", "category", "amount", "region", "object_type", "source_url", "currency"]
    initial_awards: list[dict[str, object]] = []
    document_rows: list[dict[str, object]] = []
    for award in historical_awards:
        ocid = award["source_url"].rstrip("/").rsplit("/", 1)[-1]
        opportunity, _ = transform_release(release_by_ocid[ocid])
        if not opportunity:
            raise ValueError(f"Primera entrega histórica no transformable: {ocid}")
        payload: dict[str, object] = {
            "award_id": award["award_id"], "supplier_id": award["supplier_id"], "award_date": award["award_date"],
            "title": opportunity.title, "description": opportunity.description, "technical_text": opportunity.technical_text,
            "category": opportunity.category, "amount": award["amount"], "region": opportunity.region,
            "object_type": "services", "source_url": opportunity.source_url,
            "currency": award.get("currency", ""),
        }
        initial_awards.append(payload)
        document_rows.append({
            "supplier_id": award["supplier_id"], "ocid": ocid, "publication_date": opportunity.publication_date,
            "award_date": award["award_date"], "title": opportunity.title, "description": opportunity.description,
            "technical_text": opportunity.technical_text, "category": opportunity.category, "region": opportunity.region,
            "amount": award["amount"], "initial_release_url": next(row["initial_release_url"] for row in results if row["ocid"] == ocid),
        })

    _write_results_csv(output_dir / "test_opportunities.csv", initial_candidates, opportunity_fields)
    _write_results_csv(output_dir / "historical_awards.csv", initial_awards, award_fields)
    document_fields = ["supplier_id", "ocid", "publication_date", "award_date", "title", "description", "technical_text", "category", "region", "amount", "initial_release_url"]
    _write_results_csv(output_dir / "profile_documents.csv", document_rows, document_fields)
    _write_results_csv(output_dir / "profiles.csv", cohort_profiles, list(cohort_profiles[0]))
    _write_results_csv(output_dir / "labels.csv", labels, list(labels[0]))

    awards_by_supplier: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in initial_awards:
        awards_by_supplier[str(row["supplier_id"])].append(row)
    model_profiles: list[dict[str, object]] = []
    for profile in cohort_profiles:
        supplier_awards = awards_by_supplier[profile["supplier_id"]]
        profile_text = " ".join(
            " ".join(str(row[field]) for field in ("title", "description", "technical_text") if row[field])
            for row in supplier_awards
        ).strip()
        model_profiles.append({
            **profile, "cutoff_date": "2025-01-01", "profile_text": profile_text,
            "categories": " | ".join(sorted({str(row["category"]) for row in supplier_awards if row["category"]})),
            "regions": " | ".join(sorted({str(row["region"]) for row in supplier_awards if row["region"]})),
        })
    model_profile_fields = list(cohort_profiles[0]) + ["cutoff_date", "profile_text", "categories", "regions"]
    _write_results_csv(output_dir / "profiles_model.csv", model_profiles, model_profile_fields)

    profile_ids = {row["supplier_id"] for row in cohort_profiles}
    candidate_ids = {str(row["opportunity_id"]) for row in initial_candidates}
    report = {
        "initial_releases": len(results), "profiles": len(profile_ids), "historical_awards": len(initial_awards),
        "test_candidates": len(initial_candidates), "positive_labels": len(labels),
        "labels_missing_candidate": sum(row["opportunity_id"] not in candidate_ids for row in labels),
        "profile_awards_after_cutoff": sum(str(row["award_date"]) >= "2025-01-01" for row in initial_awards),
        "test_candidates_before_cutoff": sum(str(row["publication_date"]) < "2025-01-01" for row in initial_candidates),
        "compiled_vs_initial_publication_date_mismatches": date_mismatches,
        "empty_candidate_text": sum(not (str(row["title"]) + str(row["description"]) + str(row["technical_text"])).strip() for row in initial_candidates),
        "empty_profile_document_text": sum(not (str(row["title"]) + str(row["description"]) + str(row["technical_text"])).strip() for row in document_rows),
        "empty_profile_text": sum(not str(row["profile_text"]).strip() for row in model_profiles),
        "feature_source": "first OCDS release tagged tender", "outcome_source": "compiledRelease awards",
        "availability_warning": "La fecha del evento no verifica la publicación de la versión. Ejecutar oece-audit-versions y oece-validate-cohort antes de sostener ausencia de fuga temporal.",
    }
    (output_dir / "temporal_cohort_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _write_results_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
