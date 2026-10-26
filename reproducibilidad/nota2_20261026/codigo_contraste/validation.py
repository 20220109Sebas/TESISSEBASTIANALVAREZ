"""Auditoría de versiones OCDS y preparación de anotación humana.

La fecha del evento, la fecha de versión y la fecha del paquete son distintas.
Solo ``release.publishedDate`` acredita aquí la publicación de una versión.
Los timestamps del identificador sirven para localizar archivos, no para
certificar disponibilidad histórica sin inspeccionar su contenido.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from .ocds import transform_release
from .metrics import extraction_metrics


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def publication(release: dict) -> datetime | None:
    # No sustituir por release.date ni por package.publishedDate.
    return timestamp(release.get("publishedDate"))


def publication_day(release: dict) -> date | None:
    value = publication(release)
    return value.astimezone(ZoneInfo("America/Lima")).date() if value else None


def version_hint(reference: dict) -> datetime | None:
    value = (reference.get("details") or {}).get("id") or reference.get("url", "")
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))$", value)
    return timestamp(match.group(1)) if match else None


def normalized_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "contratacionesabiertas.osce.gob.pe", "contratacionesabiertas.oece.gob.pe"
    } or not parsed.path.startswith("/api/v1/release/"):
        raise ValueError(f"Referencia de entrega fuera de la fuente esperada: {url}")
    return parsed._replace(netloc="contratacionesabiertas.oece.gob.pe").geturl()


def load_release(path: Path, ocid: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    releases = payload.get("releases", [])
    if len(releases) != 1 or releases[0].get("ocid") != ocid:
        raise ValueError(f"Entrega inesperada en {path}")
    return releases[0]


def audit_cached_versions(cohort_dir: Path, output_dir: Path, cutoff: date) -> dict:
    rows = []
    for item in read_csv(cohort_dir / "initial_release_results.csv"):
        path = Path(item["cache_path"])
        release = load_release(path, item["ocid"])
        published = publication(release)
        actual_hash = sha256(path)
        status = "missing_publication_timestamp" if published is None else (
            "historical_version_after_cutoff" if item["role"] == "historical_profile"
            and publication_day(release) >= cutoff else "timestamp_recorded"
        )
        rows.append({"ocid": item["ocid"], "role": item["role"],
                     "event_date": release.get("tender", {}).get("datePublished", ""),
                     "release_date": release.get("date", ""),
                     "release_published_at": release.get("publishedDate", ""),
                     "package_published_at": json.loads(path.read_text(encoding="utf-8")).get("publishedDate", ""),
                     "status": status, "hash_matches": actual_hash == item["sha256"],
                     "source_path": str(path), "sha256": actual_hash})
    write_csv(output_dir / "cached_version_audit.csv", rows, list(rows[0]))
    report = {"cutoff": cutoff.isoformat(), "records": len(rows),
              "statuses": dict(Counter(row["status"] for row in rows)),
              "hash_mismatches": sum(not row["hash_matches"] for row in rows)}
    (output_dir / "cached_version_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def prepare_version_audit(raw_root: Path, corpus_dir: Path, cohort_dir: Path,
                          output_dir: Path, cutoff: date = date(2025, 1, 1)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_cached_versions(cohort_dir, output_dir, cutoff)
    initial = read_csv(cohort_dir / "initial_release_results.csv")
    targets = {row["ocid"]: row["role"] for row in initial}
    snapshots: dict[str, set[str]] = defaultdict(set)
    for row in read_csv(corpus_dir / "lineage.csv"):
        if row["ocid"] in targets:
            snapshots[row["selected_snapshot_id"]].add(row["ocid"])
    references = {}
    sources = []
    for snapshot_id, ocids in sorted(snapshots.items()):
        year, month = snapshot_id.split("-")
        paths = sorted((raw_root / year / month).glob("*.json"))
        if len(paths) != 1:
            raise ValueError(f"Snapshot esperado único: {snapshot_id}")
        path = paths[0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            ocid = record.get("ocid")
            if ocid in ocids:
                refs = record.get("releases", [])
                references[ocid] = {"role": targets[ocid], "references": refs}
        sources.append({"path": str(path), "sha256": sha256(path)})
    missing = set(targets) - set(references)
    if missing:
        raise ValueError(f"Faltan {len(missing)} procesos en los snapshots")
    queue_by_url: dict[str, dict] = {}
    for ocid, item in sorted(references.items()):
        refs = item["references"]
        tender_refs = [ref for ref in refs if "tender" in ref.get("tag", []) and ref.get("url")]
        known = [ref for ref in tender_refs if version_hint(ref)]
        # Se inspecciona la versión más antigua identificable y las referencias
        # sin timestamp. La disponibilidad se confirmará dentro del paquete.
        selected = ([min(known, key=version_hint)] if known else [])
        selected += [ref for ref in tender_refs if not version_hint(ref)]
        if item["role"] == "historical_profile":
            selected += [ref for ref in refs if any(tag.startswith("award") for tag in ref.get("tag", []))
                         and ref.get("url") and (version_hint(ref) is None or version_hint(ref).astimezone(ZoneInfo("America/Lima")).date() < cutoff)]
        for ref in selected:
            url = normalized_url(ref["url"])
            queue_by_url[url] = {"ocid": ocid, "role": item["role"], "url": url,
                                 "version_hint": version_hint(ref).isoformat() if version_hint(ref) else "",
                                 "reference_date": ref.get("date", ""),
                                 "cache_key": hashlib.sha256(url.encode()).hexdigest()}
    queue = sorted(queue_by_url.values(), key=lambda row: (row["ocid"], row["version_hint"], row["url"]))
    write_csv(output_dir / "version_queue.csv", queue,
              ["ocid", "role", "url", "version_hint", "reference_date", "cache_key"])
    (output_dir / "reference_index.json").write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "cutoff": cutoff.isoformat(),
                "cached_audit": audit, "processes": len(references), "versions_to_inspect": len(queue),
                "sources": sources, "selection_rule": "earliest identifier timestamp for tender; all pre-cutoff award references; publishedDate verified from release content",
                "queue_sha256": sha256(output_dir / "version_queue.csv")}
    (output_dir / "preparation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: manifest[key] for key in ("processes", "versions_to_inspect", "cached_audit")}


def fetch_versions(queue_path: Path, cache_dir: Path, output_path: Path,
                   initial_results: Path, workers: int = 4) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = {row["initial_release_url"]: row for row in read_csv(initial_results)}

    def fetch(row: dict) -> dict:
        cached = existing.get(row["url"])
        path = Path(cached["cache_path"]) if cached else cache_dir / (row["cache_key"] + ".json")
        try:
            if not path.exists():
                temporary = path.with_suffix(".json.part")
                process = subprocess.run(["curl", "-fsSL", "--connect-timeout", "15", "--max-time", "45",
                                          "--retry", "1", "-A", "tesis-recommender/0.1", row["url"], "-o", str(temporary)],
                                         capture_output=True, text=True, timeout=100)
                if process.returncode:
                    raise ValueError(process.stderr.strip())
                load_release(temporary, row["ocid"])
                temporary.replace(path)
            release = load_release(path, row["ocid"])
            expected_id = urlparse(row["url"]).path.rsplit("/", 1)[-1]
            if release.get("id") != expected_id:
                raise ValueError("El endpoint devolvió una versión distinta de la solicitada")
            digest = sha256(path)
            if cached and digest != cached["sha256"]:
                raise ValueError("El hash del caché original no coincide")
            return {**row, "status": "ok", "cache_path": str(path), "sha256": digest,
                    "release_published_at": release.get("publishedDate", ""), "error": ""}
        except (ValueError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
            return {**row, "status": "failed", "cache_path": str(path), "sha256": "",
                    "release_published_at": "", "error": str(error)}

    results = []
    fields = ["ocid", "role", "url", "version_hint", "reference_date", "cache_key", "status",
              "cache_path", "sha256", "release_published_at", "error"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, row) for row in read_csv(queue_path)]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 25 == 0:
                write_csv(output_path, sorted(results, key=lambda row: row["url"]), fields)
    write_csv(output_path, sorted(results, key=lambda row: row["url"]), fields)
    return {"requested": len(results), "ok": sum(row["status"] == "ok" for row in results),
            "failed": sum(row["status"] == "failed" for row in results)}


def materialize_validated_cohort(cohort_dir: Path, audit_dir: Path, output_dir: Path,
                                 cutoff: date = date(2025, 1, 1),
                                 end: date = date(2025, 12, 31)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(audit_dir / "version_results.csv")
    expected = {row["url"] for row in read_csv(audit_dir / "version_queue.csv")}
    if {row["url"] for row in rows} != expected or any(row["status"] != "ok" for row in rows):
        raise ValueError("Completar las versiones solicitadas antes de materializar")
    releases: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for row in rows:
        path = Path(row["cache_path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Hash modificado: {path}")
        releases[row["ocid"]].append((load_release(path, row["ocid"]), row))

    def first_text(ocid: str) -> tuple[dict, dict] | None:
        valid = [(release, source) for release, source in releases[ocid]
                 if "tender" in release.get("tag", []) and publication(release)]
        return min(valid, key=lambda pair: publication(pair[0])) if valid else None

    retained_awards, award_audit = [], []
    for old in read_csv(cohort_dir / "cohort_historical_awards.csv"):
        ocid = old["source_url"].rstrip("/").rsplit("/", 1)[-1]
        feature = first_text(ocid)
        matches = []
        for release, source in releases[ocid]:
            published = publication(release)
            if not published or publication_day(release) >= cutoff:
                continue
            _, awards = transform_release(release)
            for award in awards:
                if award.award_id == old["award_id"] and award.award_date < cutoff:
                    matches.append((award, release, source))
        status = "verified" if matches and feature and publication_day(feature[0]) < cutoff else (
            "no_pre_cutoff_text_version" if not feature or publication_day(feature[0]) >= cutoff
            else "no_pre_cutoff_award_evidence")
        evidence = min(matches, key=lambda item: publication(item[1])) if matches else None
        feature_source = feature[1] if feature else {}
        award_audit.append({"award_id": old["award_id"], "supplier_id": old["supplier_id"], "ocid": ocid,
                            "status": status, "event_date": old["award_date"],
                            "text_published_at": feature[0].get("publishedDate", "") if feature else "",
                            "award_published_at": evidence[1].get("publishedDate", "") if evidence else "",
                            "text_source": feature_source.get("cache_path", ""),
                            "award_source": evidence[2]["cache_path"] if evidence else ""})
        if status == "verified":
            opportunity, _ = transform_release(feature[0])
            if opportunity is None or not opportunity.text:
                award_audit[-1]["status"] = "unusable_text"
                continue
            award = evidence[0]
            retained_awards.append({**asdict(award), "title": opportunity.title,
                                    "description": opportunity.description, "technical_text": opportunity.technical_text,
                                    "category": opportunity.category, "region": opportunity.region})
    counts = Counter(row["supplier_id"] for row in retained_awards)
    eligible = {supplier for supplier, count in counts.items() if count >= 2}
    retained_awards = [row for row in retained_awards if row["supplier_id"] in eligible]
    all_labels = read_csv(cohort_dir / "cohort_labels.csv")
    earliest_outcome: dict[str, date] = {}
    for row in all_labels:
        if row.get("award_date"):
            d = date.fromisoformat(row["award_date"][:10])
            earliest_outcome[row["opportunity_id"]] = min(earliest_outcome.get(row["opportunity_id"], d), d)
    candidates, candidate_audit = [], []
    for old in read_csv(cohort_dir / "cohort_test_opportunities_compiled.csv"):
        ocid = old["opportunity_id"]
        feature = first_text(ocid)
        opportunity, _ = transform_release(feature[0]) if feature else (None, [])
        published = publication(feature[0]) if feature else None
        status = "verified" if opportunity and published and cutoff <= opportunity.publication_date <= end and publication_day(feature[0]) <= end else "outside_window_or_unavailable"
        if published and ocid in earliest_outcome and publication_day(feature[0]) > earliest_outcome[ocid]:
            status = "text_version_after_observed_award"
        candidate_audit.append({"ocid": ocid, "status": status,
                                "event_date": opportunity.publication_date if opportunity else "",
                                "text_published_at": feature[0].get("publishedDate", "") if feature else "",
                                "source_path": feature[1]["cache_path"] if feature else "",
                                "sha256": feature[1]["sha256"] if feature else ""})
        if status == "verified" and opportunity.text:
            candidates.append({**asdict(opportunity), "awarded_supplier_id": ""})
    ids = {row["opportunity_id"] for row in candidates}
    labels = [row for row in all_labels if row["supplier_id"] in eligible and row["opportunity_id"] in ids]
    award_fields = ["award_id", "supplier_id", "award_date", "title", "description", "technical_text", "category", "amount", "region", "object_type", "source_url", "currency"]
    opportunity_fields = ["opportunity_id", "publication_date", "title", "description", "technical_text", "category", "amount", "region", "object_type", "source_url", "awarded_supplier_id", "currency"]
    write_csv(output_dir / "historical_awards.csv", retained_awards, award_fields)
    write_csv(output_dir / "test_opportunities.csv", candidates, opportunity_fields)
    write_csv(output_dir / "labels.csv", labels, ["supplier_id", "opportunity_id", "label", "judgment_source", "award_date"])
    write_csv(output_dir / "award_availability.csv", award_audit, list(award_audit[0]))
    write_csv(output_dir / "opportunity_availability.csv", candidate_audit, list(candidate_audit[0]))
    attributes = attribute_report(retained_awards, candidates)
    report = {"protocol_version": "publication-audit-v3", "cutoff": cutoff.isoformat(), "end": end.isoformat(),
              "input_profiles": len({row["supplier_id"] for row in award_audit}), "retained_profiles_ge_2": len(eligible),
              "retained_historical_awards": len(retained_awards), "retained_candidates": len(candidates),
              "retained_positive_pairs": len(labels), "award_statuses": dict(Counter(row["status"] for row in award_audit)),
              "candidate_statuses": dict(Counter(row["status"] for row in candidate_audit)), "attributes": attributes,
              "evaluation_status": "development_cohort_previously_inspected_not_independent_test",
                "timestamp_basis": "release.publishedDate; metadata identifier timestamps used only for locating versions",
                "date_timezone": "America/Lima; same-day outcome order remains unresolved when award timestamps have day precision",
              "evidence_limit": "Publication metadata reported by OECE, not an independently archived observation of public access at that time.",
              "input_sha256": {str(p): sha256(p) for p in [audit_dir / "version_queue.csv", audit_dir / "version_results.csv",
                                cohort_dir / "cohort_historical_awards.csv", cohort_dir / "cohort_test_opportunities_compiled.csv", cohort_dir / "cohort_labels.csv"]},
              "source_sha256": {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("ocds.py")]},
              "output_sha256": {p.name: sha256(p) for p in sorted(output_dir.glob("*.csv"))}}
    (output_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def attribute_report(awards: list[dict], opportunities: list[dict]) -> dict:
    def counts(rows: list[dict]) -> dict:
        valid = lambda row: row.get("amount") is not None and math.isfinite(float(row["amount"])) and float(row["amount"]) > 0
        return {"records": len(rows), "positive_finite_amounts": sum(valid(row) for row in rows),
                "amount_and_currency": sum(valid(row) and bool(row.get("currency", "").strip()) for row in rows),
                "currencies": dict(Counter(row.get("currency", "") or "missing" for row in rows)),
                "generic_categories": sum(row.get("category", "").strip().lower() in {"services", "consultingservices", ""} for row in rows),
                "region_semantics": "buyer_address_not_verified_service_execution_location"}
    return {"historical_awards": counts(awards), "opportunities": counts(opportunities)}


def prepare_annotation(cohort_dir: Path, audit_dir: Path, output_dir: Path,
                       size: int = 50, seed: int = 42) -> dict:
    if size <= 0:
        raise ValueError("El tamaño de muestra debe ser positivo")
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = output_dir / "field_annotations.csv"
    if annotation_path.exists() and any(
        any(row.get(field, "").strip() for field in ("gold_value", "gold_state", "annotator_id", "reviewed_at", "evidence_fragment", "notes"))
        for row in read_csv(annotation_path)
    ):
        raise ValueError("La muestra ya contiene revisión humana. Conservarla y usar otro directorio para una nueva muestra")
    candidates = read_csv(cohort_dir / "test_opportunities.csv")
    availability = {row["ocid"]: row for row in read_csv(cohort_dir / "opportunity_availability.csv")}
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in candidates:
        positive = bool(row["amount"]) and math.isfinite(float(row["amount"])) and float(row["amount"]) > 0
        strata[(row["publication_date"][:7], positive)].append(row)
    for group in strata.values():
        group.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['opportunity_id']}".encode()).hexdigest())
    selected = []
    while len(selected) < min(size, len(candidates)):
        for key in sorted(strata):
            if strata[key] and len(selected) < size:
                selected.append(strata[key].pop(0))
    for row in selected:
        source = availability[row["opportunity_id"]]
        if sha256(Path(source["source_path"])) != source["sha256"]:
            raise ValueError(f"La fuente de anotación cambió: {row['opportunity_id']}")
    fields_paths = {"title": "tender.title", "description": "tender.description",
                    "technical_text": "tender.items[*].description/classification.description/additionalClassifications[*].description",
                    "category": "tender.items[0].classification.description/id or tender.mainProcurementCategory",
                    "amount": "tender.value.amount", "currency": "tender.value.currency",
                    "region": "parties[buyer].address.department/region/locality"}
    annotations = []
    for row in selected:
        source = availability[row["opportunity_id"]]
        for field, source_path in fields_paths.items():
            annotations.append({"opportunity_id": row["opportunity_id"], "field": field,
                                "prediction": row[field], "source_url": row["source_url"],
                                "release_path": source["source_path"], "release_sha256": source["sha256"],
                                "json_path": source_path, "gold_value": "", "gold_state": "",
                                "annotator_id": "", "reviewed_at": "", "evidence_fragment": "", "notes": ""})
    write_csv(output_dir / "sample.csv", selected, list(candidates[0]) if candidates else ["opportunity_id"])
    write_csv(output_dir / "field_annotations.csv", annotations, list(annotations[0]) if annotations else ["opportunity_id", "field", "gold_value"])
    packet = [{"opportunity_id": row["opportunity_id"], "release": json.loads(Path(availability[row["opportunity_id"]]["source_path"]).read_text())}
              for row in selected]
    (output_dir / "source_packets.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"seed": seed, "requested_opportunities": size, "selected_opportunities": len(selected),
                "field_judgments_required": len(annotations), "human_judgments_completed": 0,
                "selection": "round_robin_month_and_positive_amount_strata_with_sha256_order",
                "task": "field_extraction_from_OCDS_records_not_full_PDF_or_compatibility_judgment",
                "sampling_note": "Stratified diagnostic sample, not a proportional estimate of corpus prevalence.",
                "dataset_sha256": sha256(cohort_dir / "test_opportunities.csv"),
                "version_evidence_sha256": sha256(audit_dir / "version_results.csv"),
                "sample_sha256": sha256(output_dir / "sample.csv")}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def evaluate_annotation(annotation_path: Path, output_dir: Path) -> dict:
    """Calcula métricas solo cuando existen juicios humanos completos."""
    from decimal import Decimal, InvalidOperation
    rows = read_csv(annotation_path)
    if not rows:
        raise ValueError("La muestra de anotación está vacía")
    pairs = [(row["opportunity_id"], row["field"]) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Hay juicios duplicados de oportunidad y campo")
    allowed = {"observed", "absent", "ambiguous", "illegible"}
    if any(row.get("gold_state") not in allowed or not row.get("annotator_id", "").strip()
           or not row.get("reviewed_at", "").strip() or not row.get("evidence_fragment", "").strip() for row in rows):
        raise ValueError("Completar estado, responsable, fecha y evidencia de cada juicio humano")
    if any(row["gold_state"] == "observed" and not row.get("gold_value", "").strip()
           or row["gold_state"] == "absent" and row.get("gold_value", "").strip() for row in rows):
        raise ValueError("Un campo observado requiere valor y un campo ausente requiere valor vacío")

    def normalize(value: str, field: str) -> str:
        value = " ".join(value.split()).casefold()
        if field == "amount" and value:
            try:
                amount = Decimal(value)
                if not amount.is_finite():
                    raise ValueError("El monto anotado o extraído no es finito")
                return str(amount.normalize())
            except InvalidOperation as error:
                raise ValueError("Montos: usar punto decimal y omitir separadores de miles") from error
        return value

    results = {}
    for field in sorted({row["field"] for row in rows}):
        group = [row for row in rows if row["field"] == field]
        usable = [row for row in group if row["gold_state"] in {"observed", "absent"}]
        truth = [{field: normalize(row["gold_value"], field)} for row in usable]
        predictions = [{field: normalize(row["prediction"], field)} for row in usable]
        results[field] = {**extraction_metrics(truth, predictions, (field,))[field],
                          "evaluated": len(usable), "non_evaluable": len(group) - len(usable),
                          "predicted_nonempty": sum(bool(row["prediction"].strip()) for row in usable)}
    report = {"annotation_sha256": sha256(annotation_path), "judgments": len(rows),
              "opportunities": len({row["opportunity_id"] for row in rows}),
              "normalization": "casefold_and_collapsed_whitespace; amount_decimal_without_thousands_separator",
              "task": "exact_field_extraction_from_OCDS_records", "metrics_by_field": results}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "extraction_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def verify_experiment_inputs(report_path: Path, awards_path: Path,
                             opportunities_path: Path, labels_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("protocol_version") != "publication-audit-v3":
        raise ValueError("El informe no pertenece al protocolo de disponibilidad v3")
    for path in (awards_path, opportunities_path, labels_path):
        expected = report.get("output_sha256", {}).get(path.name)
        if not expected or sha256(path) != expected:
            raise ValueError(f"La entrada no coincide con la auditoría: {path}")
    return report
