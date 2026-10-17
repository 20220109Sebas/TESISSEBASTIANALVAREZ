"""Recuperación por lotes y auditoría de candidatos sin usar adjudicatarios."""
from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .contrast_history import verify_history_queue
from .ocds import transform_release
from .validation import (read_csv, write_csv, sha256, normalized_url, fetch_versions,
                         load_release, publication, publication_day, timestamp, attribute_report)


def candidate_queue(selection_dir: Path, preparation_dir: Path) -> tuple[dict, list[dict]]:
    plan, _ = verify_history_queue(selection_dir, preparation_dir)
    rows = [r for r in read_csv(preparation_dir / "version_queue.csv") if "test_candidate" in r["role"].split("|")]
    if len({r["url"] for r in rows}) != len(rows):
        raise ValueError("URLs duplicadas en la cola de candidatos")
    selected = {r["opportunity_id"] for r in read_csv(selection_dir / "candidate_ids.csv")}
    if {r["ocid"] for r in rows} != selected:
        raise ValueError("La cola de candidatos no corresponde a la selección congelada")
    return plan, rows


def fetch_candidate_batch(selection_dir: Path, preparation_dir: Path, cache_dir: Path,
                          output_dir: Path, cache_indexes: list[Path], max_new: int = 4096,
                          workers: int = 4) -> dict:
    if max_new < 0 or not 1 <= workers <= 8:
        raise ValueError("Revisar tamaño de lote y trabajadores")
    _, queue = candidate_queue(selection_dir, preparation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "candidate_version_results.csv"
    prior_manifest = output_dir / "fetch_manifest.json"
    prior = read_csv(results_path) if results_path.exists() else []
    if results_path.exists() and (not prior_manifest.exists() or json.loads(prior_manifest.read_text())["results_sha256"] != sha256(results_path)):
        raise ValueError("Los resultados previos no coinciden con el manifiesto")
    expected = {r["url"]: r for r in queue}
    if len({r["url"] for r in prior}) != len(prior) or any(r["url"] not in expected for r in prior):
        raise ValueError("Resultados previos fuera de la cola")
    reusable = {}
    indexes = [*cache_indexes, *([results_path] if results_path.exists() else [])]
    for index in indexes:
        for row in read_csv(index):
            raw = row.get("url") or row.get("initial_release_url")
            if not raw or row.get("status", "ok") != "ok":
                continue
            url = normalized_url(raw)
            if url not in expected:
                continue
            path = Path(row["cache_path"])
            if not path.is_file() or not row.get("sha256"):
                continue
            if sha256(path) != row["sha256"]:
                raise ValueError(f"Hash de caché modificado: {path}")
            item = {"initial_release_url": url, "cache_path": str(path), "sha256": row["sha256"]}
            if url in reusable and reusable[url]["sha256"] != item["sha256"]:
                raise ValueError("Dos cachés difieren para la misma versión")
            reusable[url] = item
    already_ok = {r["url"] for r in prior if r["status"] == "ok"}
    # El orden solo organiza trabajo computacional. Ningún lote reemplaza el
    # universo anual, ni se vinculan positivos durante esta operación.
    cached = [r for r in queue if r["url"] not in already_ok and r["url"] in reusable]
    new = [r for r in queue if r["url"] not in already_ok and r["url"] not in reusable][:max_new]
    requested = sorted(cached + new, key=lambda r: (r["ocid"], r["version_hint"], r["url"]))
    batches = output_dir / "batches"
    batches.mkdir(exist_ok=True)
    numbers = [int(p.name.split("_")[-1]) for p in batches.glob("batch_*") if p.is_dir()]
    batch = batches / f"batch_{max(numbers, default=0) + 1:04d}"
    batch.mkdir()
    queue_path = batch / "queue.csv"
    fields = ["ocid", "role", "url", "version_hint", "reference_date", "cache_key"]
    write_csv(queue_path, requested, fields)
    index_path = batch / "reusable_index.csv"
    write_csv(index_path, [reusable[r["url"]] for r in cached], ["initial_release_url", "cache_path", "sha256"])
    summary = fetch_versions(queue_path, cache_dir, batch / "results.csv", index_path, workers)
    aggregate = {r["url"]: r for r in prior}
    for row in read_csv(batch / "results.csv"):
        aggregate[row["url"]] = row
    result_fields = [*fields, "status", "cache_path", "sha256", "release_published_at", "error"]
    rows = [r for _, r in sorted(aggregate.items())]
    write_csv(results_path, rows, result_fields)
    ok = sum(r["status"] == "ok" for r in rows)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "candidate_versions_planned": len(queue),
              "versions_attempted": len(rows), "ok": ok, "failed": len(rows) - ok,
              "pending_versions": len(queue) - ok, "retrieval_complete": ok == len(queue),
              "batch": str(batch), "batch_requested": len(requested), "batch_reused": len(cached),
              "batch_new_requested": len(new), "batch_result": summary, "labels_read": False,
              "queue_sha256": sha256(preparation_dir / "version_queue.csv"),
              "plan_sha256": sha256(preparation_dir / "frozen_plan_with_code.json"),
              "results_sha256": sha256(results_path)}
    (batch / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    prior_manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def audit_candidate_batch(selection_dir: Path, preparation_dir: Path, history_dir: Path,
                         fetch_dir: Path, output_dir: Path) -> dict:
    plan, queue = candidate_queue(selection_dir, preparation_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("No sobrescribir una auditoría de candidatos. Usar otra carpeta")
    history_report_path = history_dir / "history_validation_report.json"
    history = json.loads(history_report_path.read_text(encoding="utf-8"))
    for name, digest in history["output_sha256"].items():
        if sha256(history_dir / name) != digest:
            raise ValueError("Se modificó una salida del historial auditado")
    if history["input_sha256"].get(str(preparation_dir / "frozen_plan_with_code.json")) != sha256(preparation_dir / "frozen_plan_with_code.json"):
        raise ValueError("El historial no corresponde al plan")
    result_path = fetch_dir / "candidate_version_results.csv"
    fetch_report = json.loads((fetch_dir / "fetch_manifest.json").read_text(encoding="utf-8"))
    if fetch_report["results_sha256"] != sha256(result_path) or fetch_report["plan_sha256"] != sha256(preparation_dir / "frozen_plan_with_code.json"):
        raise ValueError("Se modificó la recuperación o su plan")
    expected = {r["url"]: r for r in queue}
    results = read_csv(result_path)
    if len({r["url"] for r in results}) != len(results) or any(r["url"] not in expected for r in results):
        raise ValueError("Resultados fuera del universo de candidatos")
    by_url = {r["url"]: r for r in results}
    by_ocid = defaultdict(list)
    for r in queue:
        by_ocid[r["ocid"]].append(r)
    cutoff, end = date.fromisoformat(plan["cutoff"]), date.fromisoformat(plan["candidate_end"])
    audit, opportunities, evidence = [], [], {}
    for ocid, refs in sorted(by_ocid.items()):
        row = {"ocid": ocid, "status": "pending_download", "event_date": "", "event_date_basis": "",
               "text_published_at": "", "source_path": "", "sha256": "", "hint_native_delta_seconds": "", "text_quality": ""}
        if any(r["url"] not in by_url for r in refs):
            audit.append(row)
            continue
        if any(by_url[r["url"]]["status"] != "ok" for r in refs):
            row["status"] = "download_failed"
            audit.append(row)
            continue
        sources = []
        for ref in refs:
            r = by_url[ref["url"]]
            if any(r.get(f) != ref.get(f) for f in ("ocid", "role", "cache_key")):
                raise ValueError("La fuente no corresponde a la referencia")
            path = Path(r["cache_path"])
            if sha256(path) != r["sha256"]:
                raise ValueError(f"Hash de fuente modificado: {path}")
            release = load_release(path, ocid)
            if release.get("id") != urlparse(ref["url"]).path.rsplit("/", 1)[-1]:
                raise ValueError("La respuesta corresponde a otra versión")
            evidence[str(path)] = r["sha256"]
            if "tender" in release.get("tag", []) and publication(release):
                sources.append((release, r))
        if not sources:
            row["status"] = "no_dated_tender_version"
            audit.append(row)
            continue
        release, r = min(sources, key=lambda pair: (publication(pair[0]), pair[1]["url"]))
        hint = timestamp(expected[r["url"]]["version_hint"])
        row.update({"text_published_at": release["publishedDate"], "source_path": r["cache_path"], "sha256": r["sha256"],
                    "hint_native_delta_seconds": abs((publication(release) - hint).total_seconds()) if hint else ""})
        # Whitelist: las adjudicaciones, contratos y proveedores nunca son
        # entradas de extracción ni criterios para elegir candidatos.
        safe = {k: release[k] for k in ("ocid", "tender", "buyer", "parties", "date", "publishedDate") if k in release}
        tender = safe.get("tender", {})
        row["event_date_basis"] = "tender.datePublished" if tender.get("datePublished") else "tender.tenderPeriod.startDate" if tender.get("tenderPeriod", {}).get("startDate") else "release.publishedDate_fallback"
        try:
            opportunity, _ = transform_release(safe)
        except (ValueError, TypeError, AttributeError):
            row["status"] = "unparseable_fields"
            audit.append(row)
            continue
        if opportunity:
            row["event_date"] = opportunity.publication_date.isoformat()
            row["text_quality"] = "description_or_technical_text" if opportunity.description or opportunity.technical_text else "title_only" if opportunity.title else "empty"
        native_day = publication_day(release)
        row["status"] = "out_of_scope" if opportunity is None else "outside_publication_window" if not cutoff <= native_day <= end else "outside_event_window" if not cutoff <= opportunity.publication_date <= end else "empty_text" if not opportunity.text else "verified"
        if row["status"] == "verified":
            opportunities.append({**asdict(opportunity), "awarded_supplier_id": ""})
        audit.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(history_dir / "historical_awards.csv", output_dir / "historical_awards.csv")
    fields = ["opportunity_id", "publication_date", "title", "description", "technical_text", "category", "amount", "region", "object_type", "source_url", "awarded_supplier_id", "currency"]
    write_csv(output_dir / "test_opportunities.csv", opportunities, fields)
    write_csv(output_dir / "opportunity_availability.csv", audit, list(audit[0]))
    sources = [Path(__file__), Path(__file__).with_name("ocds.py"), Path(__file__).with_name("contrast_history.py")]
    archive = output_dir / "audit_code_snapshot"
    archive.mkdir()
    for source in sources:
        shutil.copyfile(source, archive / source.name)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "cutoff": plan["cutoff"], "candidate_end": plan["candidate_end"],
              "candidate_opportunities_planned": len(by_ocid), "retained_candidates": len(opportunities),
              "eligible_profiles": history["eligible_profiles_ge_2"], "historical_awards": history["retained_historical_awards"],
              "candidate_statuses": dict(Counter(r["status"] for r in audit)),
              "text_quality": dict(Counter(r["text_quality"] for r in audit if r["status"] == "verified")),
              "identifier_native_timestamp_discrepancies_over_1s": sum(r["hint_native_delta_seconds"] != "" and r["hint_native_delta_seconds"] > 1 for r in audit),
              "retrieval_complete": fetch_report["retrieval_complete"], "evaluation_ready": fetch_report["retrieval_complete"] and not any(r["status"] in {"pending_download", "download_failed"} for r in audit),
              "labels_read": False, "outcome_based_candidate_filter": False,
              "evaluation_status": "partial_computational_batch_not_reduced_experimental_sample" if not fetch_report["retrieval_complete"] else "full_selected_candidate_pool_audited_no_labels",
              "attributes": attribute_report(read_csv(output_dir / "historical_awards.csv"), opportunities),
              "input_sha256": {str(p): sha256(p) for p in (history_report_path, result_path, preparation_dir / "frozen_plan_with_code.json")},
              "evidence_sha256": evidence, "output_sha256": {p.name: sha256(p) for p in sorted(output_dir.glob("*.csv"))},
              "audit_code_archive_sha256": {p.name: sha256(p) for p in archive.iterdir()}}
    (output_dir / "candidate_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
