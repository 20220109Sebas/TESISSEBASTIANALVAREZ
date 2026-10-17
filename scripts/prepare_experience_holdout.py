"""Freeze new service opportunities before PDF access or extraction predictions."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.register_browser_experience_sources import dump, sha, verify_hashes, protected_outputs
from scripts.audit_experience_sources import before_initial
from tesis_recommender.annex_sources import document_role, safe_annex_url, normalize_title

ROOT = Path(__file__).resolve().parents[1]
SEED = "20261010-experience-holdout-v1"
ROLES = {"administrative_bases": 0, "terms_of_reference": 1,
         "technical_requirements": 2, "integrated_bases": 3}


def select_rows(rows, excluded, count=50):
    seen, eligible = set(), []
    for row in rows:
        ocid = row["opportunity_id"]
        if ocid in seen:
            raise ValueError("Duplicate opportunity")
        seen.add(ocid)
        domain = normalize_title(row.get("description", "") + " " + row.get("title", ""))
        if ocid in excluded or row["object_type"] != "services" or "consultoria de obra" in domain or "consultoria en obra" in domain:
            continue
        if not row["publication_date"].startswith("2025-"):
            continue
        eligible.append(ocid)
    ordered = sorted(eligible, key=lambda ocid: (hashlib.sha256((SEED + ":" + ocid).encode()).hexdigest(), ocid))
    if len(ordered) < count:
        raise ValueError("Insufficient candidate universe")
    return ordered[:count], len(eligible)


def build(output):
    if output.exists():
        raise ValueError("Choose a new plan directory")
    inputs = dict(protected_outputs())
    audit = ROOT / "artifacts/extraction_commitment_v1/review_packet.json"
    manifest = json.loads((audit.parent / "manifest.json").read_text())
    verify_hashes({str(audit): manifest["output_sha256"][audit.name]})
    excluded = {r["opportunity_id"] for r in json.loads(audit.read_text())}
    exclusion_files = [audit, ROOT / "artifacts/experience_source_inventory_v1/document_inventory.json",
                       ROOT / "artifacts/experience_substitute_plan_v1/pending_plan.json"]
    for path in exclusion_files:
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else data["pending_sources"]
        excluded.update(r["opportunity_id"] for r in rows)
        inputs[str(path)] = sha(path)
    # Reject downloaded duplicate bytes after access, even when OCIDs differ.
    known_pdf_hashes = set()
    access_files = sorted(p for p in (ROOT / "artifacts").glob("experience*/**/access_results.json")
                         if not p.relative_to(ROOT / "artifacts").parts[0].startswith("experience_holdout_"))
    for path in access_files:
        rows = json.loads(path.read_text())
        for row in rows:
            if row.get("status") == "downloaded_pdf":
                excluded.add(row["opportunity_id"])
                known_pdf_hashes.add(row["sha256"])
        inputs[str(path)] = sha(path)
    candidate = ROOT / "data/processed/contrast_candidates_v1/annual_v1/test_opportunities.csv"
    availability = candidate.parent / "opportunity_availability.csv"
    experiment = json.loads((ROOT / "artifacts/experiment_annual_v1/manifest.json").read_text())
    verify_hashes({str(candidate): experiment["input_sha256"][str(candidate.relative_to(ROOT))]})
    inputs.update({str(candidate): sha(candidate), str(availability): sha(availability)})
    with candidate.open(encoding="utf-8", newline="") as handle:
        ids, eligible_count = select_rows(list(csv.DictReader(handle)), excluded)
    with availability.open(encoding="utf-8", newline="") as handle:
        by_id = {r["ocid"]: r for r in csv.DictReader(handle) if r["ocid"] in ids}
    samples, inventory, queue = [], [], []
    seen_urls = set()
    for number, ocid in enumerate(ids, 1):
        meta = by_id[ocid]
        path = ROOT / meta["source_path"]
        verify_hashes({str(path): meta["sha256"]})
        inputs[str(path)] = meta["sha256"]
        package = json.loads(path.read_text())
        if len(package["releases"]) != 1:
            raise ValueError("Expected one initial release")
        release = package["releases"][0]
        if release["ocid"] != ocid or release["publishedDate"] != meta["text_published_at"]:
            raise ValueError("Source identity or date changed")
        docs = (release.get("tender") or {}).get("documents") or []
        options = []
        for index, doc in enumerate(docs):
            role = document_role(doc.get("title"), doc.get("documentType"))
            temporal = before_initial(doc.get("datePublished"), meta["text_published_at"])
            url = doc.get("url", "")
            selected_role = role in ROLES and temporal == "declared_at_or_before_initial" and safe_annex_url(url) and str(doc.get("format", "")).casefold() == "pdf"
            row = {"opportunity_id": ocid, "sample_number": number, "document_index": index,
                   "document_id": doc.get("id"), "title": doc.get("title"), "document_type": doc.get("documentType"),
                   "document_role": role, "format": doc.get("format"), "url": url,
                   "date_published_declared": doc.get("datePublished"),
                   "initial_version_published_at": meta["text_published_at"],
                   "initial_source_sha256": meta["sha256"], "initial_source_uri": package.get("uri"),
                   "initial_source_path": str(path), "temporal_metadata_status": temporal,
                   "access_pilot_eligible": selected_role,
                   "historical_content_proven": False, "ranking_eligible": False}
            inventory.append(row)
            if selected_role:
                options.append(row)
        ordered = sorted(options, key=lambda d: (ROLES[d["document_role"]], d["document_index"], d["url"]))
        chosen = next((row for row in ordered if row["url"] not in seen_urls), None)
        status = "eligible_pdf_selected" if chosen else "no_unique_eligible_initial_pdf"
        if chosen:
            queue.append(chosen)
            seen_urls.add(chosen["url"])
        samples.append({"sample_number": number, "opportunity_id": ocid,
                        "source_path": str(path), "source_sha256": meta["sha256"],
                        "status": status, "reference_status": "not_reviewed",
                        "prediction_generated": False})
    inputs[str(ROOT / "docs/protocolo_evaluacion_experiencia_v1.md")] = sha(ROOT / "docs/protocolo_evaluacion_experiencia_v1.md")
    verify_hashes(inputs)
    output.mkdir(parents=True)
    selection = {"seed": SEED, "selection_rule": "sha256(seed:ocid)_ascending_without_requirements_or_outcomes",
        "selected_opportunities": samples, "excluded_ocids": sorted(excluded),
        "known_development_pdf_sha256": sorted(known_pdf_hashes), "input_sha256": inputs,
        "scope_screen": "services_2025_exclude_explicit_consultoria_de_obra_in_title_description",
        "scope_requires_source_check": True, "universe": eligible_count,
        "reference_status": "not_reviewed", "predictions_generated": False,
        "no_silent_replacements": True, "ranking_inputs_modified": False}
    dump(output / "selection.json", selection)
    dump(output / "document_inventory.json", inventory)
    dump(output / "queue.json", queue)
    summary = {"selected_opportunities": len(samples), "eligible_universe": eligible_count,
               "excluded_development_ocids": len(excluded), "eligible_initial_pdf_sources": len(queue),
               "source_status": dict(Counter(s["status"] for s in samples)),
               "predictions_generated": False, "reference_created": False, "f1_estimable": False,
               "pdf_duplicates_check_pending_download": True}
    dump(output / "summary.json", summary)
    # Small access batches share the same selection and never replace failures.
    limits = {"documents": 5, "max_bytes_per_document": 10 * 1024 * 1024,
              "timeout_seconds": 30, "max_pages_per_document": 120, "workers": 1, "ocr": False}
    for batch, start in enumerate(range(0, len(queue), 5), 1):
        directory = output / "batches" / f"batch_{batch:02d}"
        directory.mkdir(parents=True)
        plan = {"purpose": "new_source_reference_access_before_extraction_predictions",
                "limits": limits, "selected": queue[start:start + 5], "input_sha256": inputs,
                "known_development_pdf_sha256": sorted(known_pdf_hashes),
                "ranking_eligible": False, "historical_content_proven": False}
        dump(directory / "access_plan.json", plan)
        dump(directory / "manifest.json", {"input_sha256": inputs,
             "output_sha256": {"access_plan.json": sha(directory / "access_plan.json")}})
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
         "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve())))
