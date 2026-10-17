"""Recuperación y auditoría histórica de la cohorte de contraste, sin etiquetas."""
from __future__ import annotations

import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .ocds import transform_release
from .validation import (read_csv, write_csv, sha256, normalized_url, fetch_versions,
                         load_release, publication, publication_day, attribute_report, timestamp)


def verify_history_queue(selection_dir: Path, preparation_dir: Path) -> tuple[dict, list[dict]]:
    selection_path = selection_dir / "selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    plan_path = preparation_dir / "frozen_plan_with_code.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    preparation = json.loads((preparation_dir / "preparation_manifest.json").read_text(encoding="utf-8"))
    original = preparation_dir / "frozen_evaluation_plan.json"
    if sha256(original) != preparation["plan_sha256"] or sha256(original) != plan["original_plan_sha256"]:
        raise ValueError("Se modificó el plan original")
    if sha256(selection_path) != plan["selection_manifest_sha256"]:
        raise ValueError("Se modificó el manifiesto de selección")
    if selection["selection_sha256"] != plan["selection_sha256"]:
        raise ValueError("La selección no coincide con el plan")
    for name, digest in selection["selection_sha256"].items():
        if sha256(selection_dir / name) != digest:
            raise ValueError(f"Se modificó la selección: {name}")
    for name, digest in plan["code_archive_sha256"].items():
        if sha256(Path(plan["code_archive"]) / name) != digest:
            raise ValueError(f"Se modificó el código archivado: {name}")
    queue_path = preparation_dir / "version_queue.csv"
    if sha256(queue_path) != preparation["queue_sha256"]:
        raise ValueError("Se modificó la cola completa")
    expected = [r for r in read_csv(queue_path) if "historical_profile" in r["role"].split("|")]
    actual = read_csv(preparation_dir / "historical_version_queue.csv")
    if actual != expected or len({r["url"] for r in actual}) != len(actual):
        raise ValueError("La cola histórica no corresponde al subconjunto congelado")
    return plan, actual


def fetch_contrast_history(selection_dir: Path, preparation_dir: Path, cache_dir: Path,
                           output_dir: Path, cache_indexes: list[Path], workers: int = 4) -> dict:
    if not 1 <= workers <= 8:
        raise ValueError("Usar de uno a ocho trabajadores")
    _, queue = verify_history_queue(selection_dir, preparation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "historical_version_results.csv"
    queue_urls = {r["url"] for r in queue}
    reusable = {}
    indexes = [*cache_indexes, *([output_path] if output_path.exists() else [])]
    for index in indexes:
        for row in read_csv(index):
            raw_url = row.get("url") or row.get("initial_release_url")
            if not raw_url or row.get("status", "ok") != "ok":
                continue
            url = normalized_url(raw_url)
            if url not in queue_urls:
                continue
            if not row.get("sha256") or not Path(row["cache_path"]).is_file():
                continue
            candidate = {"initial_release_url": url, "cache_path": row["cache_path"], "sha256": row["sha256"]}
            old = reusable.get(url)
            if old and old["sha256"] != candidate["sha256"]:
                raise ValueError(f"Dos cachés difieren para la misma versión: {url}")
            reusable[url] = candidate
    index_path = output_dir / "reusable_cache_index.csv"
    write_csv(index_path, [v for _, v in sorted(reusable.items())],
              ["initial_release_url", "cache_path", "sha256"])
    report = fetch_versions(preparation_dir / "historical_version_queue.csv", cache_dir,
                            output_path, index_path, workers)
    report.update({"created_at_utc": datetime.now(timezone.utc).isoformat(), "reusable_versions": len(reusable),
                   "labels_read": False, "publication_verified": False,
                   "queue_sha256": sha256(preparation_dir / "historical_version_queue.csv"),
                   "results_sha256": sha256(output_path),
                   "plan_sha256": sha256(preparation_dir / "frozen_plan_with_code.json")})
    (output_dir / "fetch_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def materialize_contrast_history(selection_dir: Path, preparation_dir: Path,
                                 fetch_dir: Path, output_dir: Path) -> dict:
    plan, queue = verify_history_queue(selection_dir, preparation_dir)
    cutoff = date.fromisoformat(plan["cutoff"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("No sobrescribir una auditoría terminada. Usar otra carpeta")
    result_path = fetch_dir / "historical_version_results.csv"
    fetch_manifest = json.loads((fetch_dir / "fetch_manifest.json").read_text(encoding="utf-8"))
    if fetch_manifest["results_sha256"] != sha256(result_path) or fetch_manifest["queue_sha256"] != sha256(preparation_dir / "historical_version_queue.csv"):
        raise ValueError("Se modificaron los resultados de recuperación")
    if fetch_manifest["plan_sha256"] != sha256(preparation_dir / "frozen_plan_with_code.json"):
        raise ValueError("El plan cambió después de la recuperación")
    results = read_csv(result_path)
    expected = {r["url"]: r for r in queue}
    if len(results) != len(expected) or {r["url"] for r in results} != set(expected) or any(r["status"] != "ok" for r in results):
        raise ValueError("Completar toda la cola histórica antes de materializar")
    releases = defaultdict(list)
    version_audit = []
    for row in results:
        if any(row.get(f) != expected[row["url"]].get(f) for f in ("ocid", "role", "cache_key")):
            raise ValueError("El resultado no corresponde a la referencia solicitada")
        path = Path(row["cache_path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Hash de evidencia modificado: {path}")
        release = load_release(path, row["ocid"])
        if release.get("id") != urlparse(row["url"]).path.rsplit("/", 1)[-1]:
            raise ValueError("La respuesta contiene otra versión")
        releases[row["ocid"]].append((release, row))
        published = publication(release)
        day = publication_day(release)
        hint = timestamp(expected[row["url"]].get("version_hint"))
        version_audit.append({"ocid": row["ocid"], "url": row["url"], "tag": "|".join(release.get("tag", [])),
                              "native_published_at": release.get("publishedDate", ""),
                              "hint_native_delta_seconds": round(abs((published - hint).total_seconds()), 6) if published and hint else "",
                              "status": "missing_native_publication" if published is None else "before_cutoff" if day < cutoff else "after_cutoff",
                              "source_path": row["cache_path"], "sha256": row["sha256"]})
    text_by_ocid = {}
    for ocid, entries in releases.items():
        texts = [(r, source) for r, source in entries if "tender" in r.get("tag", []) and publication(r)]
        text_by_ocid[ocid] = min(texts, key=lambda pair: (publication(pair[0]), pair[1]["url"])) if texts else None
    retained, audit = [], []
    for selected in read_csv(selection_dir / "profile_history_ids.csv"):
        ocid = selected["source_url"].rstrip("/").rsplit("/", 1)[-1]
        feature = text_by_ocid.get(ocid)
        matches = []
        for release, source in releases[ocid]:
            if publication(release) is None or publication_day(release) >= cutoff:
                continue
            # La evidencia de adjudicación puede ser una entrega parcial sin
            # tender. Se leen sus datos propios sin rellenarlos con el futuro.
            for award in release.get("awards", []):
                try:
                    event = date.fromisoformat(award.get("date", "")[:10])
                except (ValueError, TypeError):
                    continue
                if event >= cutoff:
                    continue
                for supplier in award.get("suppliers", []):
                    supplier_id = supplier.get("id") or supplier.get("name", "")
                    identity = f"{ocid}:{award.get('id', '')}:{supplier_id}"
                    if identity == selected["award_id"] and supplier_id == selected["supplier_id"]:
                        matches.append((release, source, award, event))
        evidence = min(matches, key=lambda item: (publication(item[0]), item[1]["url"])) if matches else None
        opportunity, _ = transform_release(feature[0]) if feature and publication_day(feature[0]) < cutoff else (None, [])
        status = "verified" if evidence and opportunity and opportunity.text else (
            "no_pre_cutoff_text_version" if not feature or publication_day(feature[0]) >= cutoff else
            "unusable_or_out_of_scope_text" if not opportunity or not opportunity.text else "no_pre_cutoff_award_evidence")
        audit.append({"supplier_id": selected["supplier_id"], "award_id": selected["award_id"], "ocid": ocid,
                      "status": status, "selected_event_date": selected["award_date"],
                      "evidence_event_date": evidence[3].isoformat() if evidence else "",
                      "text_published_at": feature[0].get("publishedDate", "") if feature else "",
                      "award_published_at": evidence[0].get("publishedDate", "") if evidence else "",
                      "award_status_reported": evidence[2].get("status", "") if evidence else "",
                      "text_source": feature[1]["cache_path"] if feature else "",
                      "award_source": evidence[1]["cache_path"] if evidence else "",
                      "text_sha256": feature[1]["sha256"] if feature else "",
                      "award_sha256": evidence[1]["sha256"] if evidence else ""})
        if status != "verified":
            continue
        value = evidence[2].get("value") or {}
        try:
            amount = float(value["amount"]) if value.get("amount") is not None else None
        except (TypeError, ValueError):
            amount = None
        if amount is not None and not math.isfinite(amount):
            amount = None
        retained.append({"award_id": selected["award_id"], "supplier_id": selected["supplier_id"],
                         "award_date": evidence[3].isoformat(), "title": opportunity.title,
                         "description": opportunity.description, "technical_text": opportunity.technical_text,
                         "category": opportunity.category, "region": opportunity.region,
                         "amount": amount, "currency": value.get("currency", ""),
                         "object_type": "services", "source_url": opportunity.source_url})
    counts = Counter(r["supplier_id"] for r in retained)
    selected_profiles = read_csv(selection_dir / "selected_profiles.csv")
    eligible = {s for s, n in counts.items() if n >= 2}
    profile_rows = [{"supplier_id": r["supplier_id"], "history_count_provisional": r["history_count_provisional"],
                     "verified_history_count": counts[r["supplier_id"]],
                     "status": "eligible_ge_2" if r["supplier_id"] in eligible else "insufficient_verified_history",
                     "replaced": False} for r in selected_profiles]
    usable = [r for r in retained if r["supplier_id"] in eligible]
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["award_id", "supplier_id", "award_date", "title", "description", "technical_text", "category", "amount", "region", "object_type", "source_url", "currency"]
    write_csv(output_dir / "historical_awards.csv", usable, fields)
    write_csv(output_dir / "award_availability.csv", audit, list(audit[0]))
    write_csv(output_dir / "profile_eligibility.csv", profile_rows, list(profile_rows[0]))
    write_csv(output_dir / "version_availability.csv", version_audit, list(version_audit[0]))
    source_paths = (Path(__file__), Path(__file__).with_name("ocds.py"), Path(__file__).with_name("validation.py"))
    code_dir = output_dir / "audit_code_snapshot"
    code_dir.mkdir()
    for source_path in source_paths:
        shutil.copyfile(source_path, code_dir / source_path.name)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "cutoff": cutoff.isoformat(),
              "input_profiles": len(profile_rows), "input_historical_awards": len(audit),
              "verified_awards_before_profile_filter": len(retained), "eligible_profiles_ge_2": len(eligible),
              "excluded_profiles": len(profile_rows) - len(eligible), "retained_historical_awards": len(usable),
              "award_statuses": dict(Counter(r["status"] for r in audit)),
              "reported_award_statuses_verified": dict(Counter(r["award_status_reported"] or "missing" for r in audit if r["status"] == "verified")),
              "version_statuses": dict(Counter(r["status"] for r in version_audit)),
              "identifier_native_timestamp_discrepancies_over_1s": sum(
                  r["hint_native_delta_seconds"] != "" and r["hint_native_delta_seconds"] > 1 for r in version_audit),
              "attributes": attribute_report(usable, []), "labels_read": False, "replaced_profiles": 0,
              "evaluation_status": "historical_availability_verified_company_disjoint_same_inspected_source_year",
              "candidates_validated": False, "models_executed_on_contrast": False,
              "evidence_limit": "Native publication metadata reported by OECE. Not an independent observation of historical public access. An observed historical award is not a complete measure of company capacity.",
              "input_sha256": {str(p): sha256(p) for p in (selection_dir / "selection_manifest.json", selection_dir / "profile_history_ids.csv", result_path, preparation_dir / "frozen_plan_with_code.json")},
              "evidence_sha256": {r["cache_path"]: r["sha256"] for r in results},
              "source_sha256": {p.name: sha256(p) for p in source_paths},
              "audit_code_archive_sha256": {p.name: sha256(p) for p in sorted(code_dir.iterdir())},
              "output_sha256": {p.name: sha256(p) for p in sorted(output_dir.glob("*.csv"))}}
    (output_dir / "history_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
