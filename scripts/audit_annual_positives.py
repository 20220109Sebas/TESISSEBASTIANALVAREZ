"""Revela y audita desenlaces solo después de fijar las entradas anuales."""
from __future__ import annotations

import argparse
import csv
import json
import resource
import shutil
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from tesis_recommender.annual_evaluation import classify_positive, iter_snapshot_records
from tesis_recommender.contrast_candidates import candidate_queue
from tesis_recommender.io import read_awards
from tesis_recommender.profiles import build_profiles
from tesis_recommender.validation import read_csv, write_csv, sha256


def audit(args):
    if args.output.exists():
        raise ValueError("No sobrescribir una auditoría de etiquetas")
    plan, queue = candidate_queue(args.selection, args.preparation)
    plan_path = args.preparation / "frozen_plan_with_code.json"
    candidate_report = json.loads((args.cohort / "candidate_validation_report.json").read_text())
    if not candidate_report["retrieval_complete"] or not candidate_report["evaluation_ready"]:
        raise ValueError("Completar el universo de entradas antes de revelar etiquetas")
    for name, digest in candidate_report["output_sha256"].items():
        if sha256(args.cohort / name) != digest:
            raise ValueError("Se modificó el corpus anual")
    verification = json.loads((args.cache / "verification.json").read_text())
    for name, digest in verification["output_sha256"].items():
        if sha256(args.cache / name) != digest:
            raise ValueError("Se modificó el caché verificado")
    metadata = json.loads((args.cache / "metadata.json").read_text())
    for field in ("model_name", "model_revision", "profile_strategy", "opportunity_strategy"):
        if metadata[field] != plan[field]:
            raise ValueError("El caché no corresponde al modelo congelado")
    for item in metadata["inputs"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Cambió una entrada del caché")
    cutoff = date.fromisoformat(plan["cutoff"])
    profiles = build_profiles(read_awards(args.cohort / "historical_awards.csv"), cutoff)
    availability_rows = read_csv(args.cohort / "opportunity_availability.csv")
    availability = {r["ocid"]: r for r in availability_rows}
    expected_ids = {r["ocid"] for r in queue}
    if len(availability) != len(availability_rows) or set(availability) != expected_ids:
        raise ValueError("La disponibilidad no cubre la selección congelada")
    retained = {r["opportunity_id"] for r in read_csv(args.cohort / "test_opportunities.csv")}
    if retained != {ocid for ocid, r in availability.items() if r["status"] == "verified"}:
        raise ValueError("La auditoría no coincide con los candidatos")
    entries = json.loads((args.cache / "index.json").read_text())
    if {r["id"] for r in entries if r["kind"] == "profile"} != set(profiles) or {r["id"] for r in entries if r["kind"] == "opportunity"} != retained:
        raise ValueError("Las entradas del caché no coinciden con la población")
    corpus_report_path = args.corpus / "corpus_report.json"
    corpus = json.loads(corpus_report_path.read_text())
    labels_path = args.corpus / "historical_labels.csv"
    lineage_path = args.corpus / "lineage.csv"
    for p in (labels_path, lineage_path):
        if sha256(p) != corpus["output_sha256"][p.name]:
            raise ValueError("Cambió una fuente consolidada de desenlaces")
    lineage = {r["ocid"]: r for r in read_csv(lineage_path) if r["ocid"] in expected_ids}
    if set(lineage) != expected_ids:
        raise ValueError("Falta procedencia de los desenlaces")
    snapshot_ids = {r["selected_snapshot_id"] for r in lineage.values()}
    snapshots = {}
    for item in corpus["snapshots"]:
        path = Path(item["snapshot"])
        snapshot_id = path.parent.parent.name + "-" + path.parent.name
        if snapshot_id in snapshot_ids:
            snapshots[snapshot_id] = {"path": str(path), "sha256": item["sha256"]}
    if set(snapshots) != snapshot_ids:
        raise ValueError("Falta un snapshot de desenlaces")
    args.output.mkdir(parents=True)
    archive = args.output / "code_snapshot"
    archive.mkdir()
    for p in (Path(__file__), Path(__file__).resolve().parents[1] / "src/tesis_recommender/annual_evaluation.py"):
        shutil.copyfile(p, archive / p.name)
    # Este registro se guarda ANTES de abrir etiquetas o los awards consolidados.
    reveal_plan = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels_opened_at_recording": False, "plan_sha256": sha256(plan_path),
        "cohort_sha256": candidate_report["output_sha256"],
        "cache_sha256": verification["output_sha256"], "profiles": sorted(profiles),
        "candidate_count": len(retained), "candidate_selection_unchanged": True,
        "code_sha256": {p.name: sha256(p) for p in archive.iterdir()},
        "outcome_snapshot_sources": snapshots,
        "labels_sha256_before_read": sha256(labels_path), "lineage_sha256": sha256(lineage_path),
        "pair_rule": "Frozen observed award relation, exact supplier and OCID, known award date >= cutoff and >= first text publication day in Lima. Preserve same-day ambiguity. No upper award-date filter beyond outcomes present in the archived selected snapshots. No filtering by rank, no replacement of companies, no candidate removal based on outcomes.",
        "award_status_rule": "Preserve the frozen historical_award definition. Record native award status, do not silently impose a new status filter. Contradictory or non-active evidence must be reported.",
        "pair_deduplication": "One positive per supplier-OCID. Keep all matching award evidence separately."}
    (args.output / "reveal_plan.json").write_text(json.dumps(reveal_plan, ensure_ascii=False, indent=2))
    selected_labels = {}
    with labels_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["supplier_id"] not in profiles or row["opportunity_id"] not in expected_ids:
                continue
            key = (row["supplier_id"], row["opportunity_id"])
            if key in selected_labels or row["label"] != "1":
                raise ValueError("Etiqueta duplicada o distinta de un positivo observado")
            selected_labels[key] = row
    evidence = defaultdict(list)
    found = set()
    for snapshot_id, info in sorted(snapshots.items()):
        path = Path(info["path"])
        if sha256(path) != info["sha256"]:
            raise ValueError("Cambió un snapshot de evidencia")
        for record in iter_snapshot_records(path):
            ocid = record.get("ocid")
            if ocid not in lineage or lineage[ocid]["selected_snapshot_id"] != snapshot_id:
                continue
            if ocid in found:
                raise ValueError("Identidad repetida en la fuente seleccionada")
            found.add(ocid)
            compiled = record.get("compiledRelease") or {}
            if compiled.get("ocid") != ocid or compiled.get("date", "") != lineage[ocid]["compiled_release_date"]:
                raise ValueError("La versión consolidada no corresponde al lineage")
            for award in compiled.get("awards", []):
                if not award.get("date"):
                    continue
                for supplier in award.get("suppliers", []):
                    supplier_id = supplier.get("id") or supplier.get("name", "")
                    if supplier_id in profiles:
                        evidence[(supplier_id, ocid)].append({"supplier_id": supplier_id, "opportunity_id": ocid,
                            "award_id": award.get("id", ""), "award_date": award["date"],
                            "award_status": award.get("status", ""), "snapshot_id": snapshot_id,
                            "snapshot_path": str(path), "snapshot_sha256": info["sha256"],
                            "compiled_release_id": compiled.get("id", ""),
                            "compiled_release_date": compiled.get("date", "")})
        if sha256(path) != info["sha256"]:
            raise ValueError("Snapshot modificado durante la lectura")
        print(json.dumps({"phase": "outcome_source_audit", "snapshot": snapshot_id,
                          "candidate_records_checked": len(found), "observed_pairs": len(evidence)}), flush=True)
    if found != expected_ids or set(evidence) != set(selected_labels):
        raise ValueError("La evidencia original no reproduce el universo de etiquetas seleccionadas")
    audit_rows, positives = [], []
    for key, row in sorted(selected_labels.items()):
        matching = [r for r in evidence[key] if r["award_date"][:10] == row["award_date"][:10]]
        if not matching:
            raise ValueError("La fecha de la etiqueta no está respaldada por la fuente")
        status, ordering = classify_positive(row, set(profiles), availability, cutoff)
        source = availability[key[1]]
        audit_rows.append({**row, "audit_status": status, "temporal_order": ordering,
            "candidate_status": source["status"], "text_published_at": source["text_published_at"],
            "evidence_count": len(matching), "native_award_statuses": "|".join(sorted({r["award_status"] for r in matching})),
            "snapshot_id": matching[0]["snapshot_id"], "source_snapshot_sha256": matching[0]["snapshot_sha256"]})
        if status == "verified_positive":
            positives.append(row)
    label_fields = ["supplier_id", "opportunity_id", "label", "judgment_source", "award_date"]
    write_csv(args.output / "labels.csv", positives, label_fields)
    write_csv(args.output / "positive_pair_audit.csv", audit_rows, [*label_fields, "audit_status", "temporal_order",
        "candidate_status", "text_published_at", "evidence_count", "native_award_statuses", "snapshot_id", "source_snapshot_sha256"])
    evidence_rows = [r for key in sorted(evidence) for r in evidence[key]]
    write_csv(args.output / "award_evidence.csv", evidence_rows, ["supplier_id", "opportunity_id", "award_id", "award_date",
        "award_status", "snapshot_id", "snapshot_path", "snapshot_sha256", "compiled_release_id", "compiled_release_date"])
    counts = Counter(r["supplier_id"] for r in positives)
    write_csv(args.output / "profile_label_coverage.csv", [{"supplier_id": supplier,
        "history_count": len(profile.awards), "known_positives": counts[supplier],
        "evaluation_status": "evaluable" if counts[supplier] else "no_known_positives"} for supplier, profile in sorted(profiles.items())],
        ["supplier_id", "history_count", "known_positives", "evaluation_status"])
    # Las fuentes, los vectores y la selección no se escriben en esta operación.
    for name, digest in reveal_plan["cohort_sha256"].items():
        if sha256(args.cohort / name) != digest:
            raise ValueError("Cambió el corpus al vincular etiquetas")
    for name, digest in reveal_plan["cache_sha256"].items():
        if sha256(args.cache / name) != digest:
            raise ValueError("Cambió el caché al vincular etiquetas")
    if sha256(labels_path) != reveal_plan["labels_sha256_before_read"]:
        raise ValueError("Cambió la fuente de etiquetas")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_status": plan["evaluation_status"],
        "profiles": len(profiles), "candidates": len(retained), "planned_candidates": len(expected_ids),
        "observed_pairs_before_temporal_gate": len(selected_labels), "verified_positive_pairs": len(positives),
        "pair_statuses": dict(Counter(r["audit_status"] for r in audit_rows)),
        "native_award_statuses": dict(Counter(r["award_status"] or "missing" for r in evidence_rows)),
        "same_day_positive_pairs": sum(r["audit_status"] == "verified_positive" and r["temporal_order"] == "same_day_order_unresolved" for r in audit_rows),
        "retained_award_years": dict(Counter(r["award_date"][:4] for r in positives)),
        "profiles_with_known_positives": len(counts), "profiles_without_known_positives": len(profiles) - len(counts),
        "all_candidate_outcome_records_checked": len(found), "source_snapshots_checked": len(snapshots),
        "labels_read": True, "candidate_selection_unchanged": True, "profile_replacements": 0,
        "inputs_and_embeddings_unchanged": True, "outcomes_used_to_tune": False,
        "rss_peak_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "input_sha256": {str(p): sha256(p) for p in (plan_path, args.cohort / "candidate_validation_report.json", args.cache / "verification.json", labels_path, corpus_report_path, lineage_path)},
        "cohort_sha256": reveal_plan["cohort_sha256"], "cache_sha256": reveal_plan["cache_sha256"],
        "output_sha256": {p.name: sha256(p) for p in args.output.glob("*.csv")},
        "reveal_plan_sha256": sha256(args.output / "reveal_plan.json"), "code_sha256": reveal_plan["code_sha256"],
        "limits": "Only archived observed outcomes, not exhaustive compatibility judgments. Same-day order unresolved. Annual static pool, not a real-time backtest. Same inspected source/year, company-disjoint exploratory contrast."}
    (args.output / "label_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("selection", "preparation", "cohort", "cache", "corpus", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(audit(parser.parse_args()), ensure_ascii=False, indent=2))
