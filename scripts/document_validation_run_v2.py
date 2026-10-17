"""Integrate immutable selection, PDF preparation and reference-first evaluation.

Version 2 records the changed bundled Python BEFORE selecting any new content.
It keeps the candidate, field comparator, sampling seed and source rules intact.
"""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.freeze_document_experience_candidate_v1 import validate_lock as validate_candidate
from scripts.freeze_document_source_design_v1 import validate_design as validate_sources
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes
from scripts.run_document_experience_evaluation_v1 import FILES as EVALUATION_FILES, validate_design
from scripts.run_document_experience_evaluation_v1 import load_sources
from tesis_recommender.document_sampling_v1 import SAMPLE_SIZE, SEED, choose_document, select_ids
from tesis_recommender.annex_sources import document_role

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/document_experience_candidate_lock_v1"
OLD_DESIGN = ROOT / "artifacts/document_experience_evaluation_design_lock_v1"
FILES = (*EVALUATION_FILES, "scripts/document_validation_run_v2.py",
         "tests/test_document_validation_run_v2.py", "docs/document_validation_run_v2.json")


def checked(directory, inputs):
    verify_artifact(directory, inputs)
    inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])


def write_artifact(output, files, inputs):
    if output.exists():
        raise ValueError("Use a new output directory; preserve all prior artifacts")
    verify_hashes(inputs)
    output.mkdir(parents=True)
    for name, value in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        dump(target, value)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return files["summary.json"]


def runtime_transition(old_design, new_runtime):
    """Verify old files; allow only the explicitly recorded bundled binary change."""
    manifest = json.loads((old_design / "manifest.json").read_text())
    verify_hashes({str(old_design / p): h for p, h in manifest["output_sha256"].items()})
    binary = new_runtime["python_executable"]
    changed = []
    for name, expected in manifest["input_sha256"].items():
        current = sha(name)
        if current != expected:
            if name != binary:
                raise ValueError("Unapproved historical code/data/runtime drift: " + name)
            changed.append({"path": name, "prior_sha256": expected, "current_sha256": current})
    return {"observed_changes": changed, "allowed_changed_paths": [binary],
            "before_new_sample_selection": True, "old_artifacts_modified": False,
            "interpretation": "new_environment_version_not_reproduction_of_old_runtime"}


def freeze(source_design, output):
    if output.exists():
        raise ValueError("Use a fresh integrated lock directory")
    validate_candidate(CANDIDATE)
    source = validate_sources(source_design, before_selection=True)
    inputs = dict(protected_outputs())
    checked(CANDIDATE, inputs)
    checked(source_design, inputs)
    transition = runtime_transition(OLD_DESIGN, source["source_runtime"])
    # Pin the historical record, not its obsolete runtime as a current input.
    inputs[str(OLD_DESIGN / "manifest.json")] = sha(OLD_DESIGN / "manifest.json")
    inputs.update({str(ROOT / p): sha(ROOT / p) for p in FILES})
    contract = json.loads((ROOT / FILES[-1]).read_text())
    if contract["sample_size"] != SAMPLE_SIZE or contract["seed"] != SEED:
        raise ValueError("Do not change sampling seed or sample size")
    code = {str(ROOT / p): sha(ROOT / p) for p in FILES}
    evaluation = {"version": "document_experience_evaluation_design_lock_v1", "code_sha256": code,
        "candidate_lock_path": str(CANDIDATE), "candidate_manifest_sha256": sha(CANDIDATE / "manifest.json"),
        "source_design_path": str(source_design), "source_design_manifest_sha256": sha(source_design / "manifest.json"),
        "comparison_adapter_locked": True, "reference_first_gates_locked": True,
        "new_sample_selected": False, "ready_to_select_new_evaluation_sample": False,
        "ready_to_compute_metrics": False,
        "interpretation": "immutable_preparation_checkpoint_see_integrated_run_lock_for_selection_permission"}
    run = {"version": "document_validation_integrated_run_v2", "code_sha256": code,
           "ready_to_select_new_evaluation_sample": True, "new_sample_selected": False,
           "reference_before_inference_required": True, "ready_to_compute_metrics": False,
           "source_design_path": str(source_design), "source_design_manifest_sha256": sha(source_design / "manifest.json"),
           "seed": SEED, "sample_size": SAMPLE_SIZE, "ranking_eligible": False}
    files = {"evaluation_design_lock.json": evaluation, "integrated_run_lock.json": run,
        "protocol.json": json.loads((ROOT / "docs/document_experience_evaluation_design_v1.json").read_text()),
        "runtime_transition.json": transition,
        "summary.json": {"stage": "integrated_protocol_fixed_before_selection", "ready_to_select_new_evaluation_sample": True,
            "source_rules_changed": False, "extractor_changed": False, "comparison_rules_changed": False,
            "runtime_changed_paths": len(transition["observed_changes"]), "new_sample_selected": False,
            "new_pdf_downloads": 0, "metrics_computed": False, "ranking_inputs_modified": False}}
    summary = write_artifact(output, files, inputs)
    # Archive code as additional registered outputs without rewriting any prior artifact.
    for rel in FILES:
        target = output / "code_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
                          if p.is_file() and p.name != "manifest.json"}})
    return summary


def validate_run(directory, *, before_selection=False):
    validate_design(directory)
    run = json.loads((directory / "integrated_run_lock.json").read_text())
    if (run["version"] != "document_validation_integrated_run_v2"
            or run["ready_to_select_new_evaluation_sample"] is not True or run["new_sample_selected"] is not False
            or run["ready_to_compute_metrics"] is not False or run["seed"] != SEED or run["sample_size"] != SAMPLE_SIZE):
        raise ValueError("Integrated run contract changed")
    validate_sources(Path(run["source_design_path"]), before_selection=before_selection)
    return run


def selected_metadata(rows, availability, excluded, load_release):
    ids, universe = select_ids(rows, set(excluded))
    selected = set(ids)
    by_id = {}
    for row in availability:
        if row["ocid"] in selected:
            if row["ocid"] in by_id:
                raise ValueError("Duplicate availability join key")
            by_id[row["ocid"]] = row
    if set(by_id) != selected:
        raise ValueError("Missing availability join; do not replace a selected case")
    samples, inventory, queue, seen_urls = [], [], [], set()
    for number, ocid in enumerate(ids, 1):
        meta = by_id[ocid]
        package, path = load_release(meta)
        if len(package["releases"]) != 1:
            raise ValueError("Exactly one pinned initial release required")
        release = package["releases"][0]
        if release["ocid"] != ocid or release["publishedDate"] != meta["text_published_at"]:
            raise ValueError("Initial release identity/date changed")
        docs = (release.get("tender") or {}).get("documents") or []
        chosen, status = choose_document(docs, meta["text_published_at"], seen_urls)
        base = {"sample_number": number, "opportunity_id": ocid,
                "source_path": str(path), "source_sha256": meta["sha256"],
                "initial_version_published_at": meta["text_published_at"], "status": status,
                "reference_status": "not_reviewed", "predictions_generated": False}
        samples.append(base)
        for index, doc in enumerate(docs):
            inventory.append({"sample_number": number, "opportunity_id": ocid, "document_index": index,
                "document_id": doc.get("id"), "title": doc.get("title"), "document_type": doc.get("documentType"),
                "document_role": document_role(doc.get("title"), doc.get("documentType")),
                "format": doc.get("format"), "url": doc.get("url"), "date_published_declared": doc.get("datePublished"),
                "selected": chosen is not None and chosen["document_index"] == index,
                "ranking_eligible": False, "historical_content_proven": False})
        if chosen is not None:
            doc = chosen["document"]
            queue.append({"sample_number": number, "opportunity_id": ocid, "document_index": chosen["document_index"],
                "document_id": doc.get("id"), "title": doc.get("title"), "document_type": doc.get("documentType"),
                "document_role": chosen["document_role"], "format": doc["format"], "url": doc["url"],
                "date_published_declared": doc["datePublished"], "initial_version_published_at": meta["text_published_at"],
                "initial_source_path": str(path), "initial_source_sha256": meta["sha256"],
                "ranking_eligible": False, "historical_content_proven": False})
            seen_urls.add(doc["url"])
    return samples, inventory, queue, universe


def initial_source_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT / "data/raw/oece/contrast_candidate_versions_v1"):
        raise ValueError("Initial source path outside the frozen candidate-version inventory")
    return path


def select(directory, output):
    if output.exists():
        raise ValueError("Preserve the fixed selection; choose a new reproduction directory")
    run = validate_run(directory, before_selection=True)
    source_design = Path(run["source_design_path"])
    inputs = {}
    checked(directory, inputs)
    registry = json.loads((source_design / "exclusion_registry.json").read_text())
    base = ROOT / "data/processed/contrast_candidates_v1/annual_v1"
    candidate, availability = base / "test_opportunities.csv", base / "opportunity_availability.csv"
    verify_hashes({str(p): inputs[str(p)] for p in (candidate, availability)})
    # Explicitly project metadata: no technical text or awarded supplier enters selection.
    with candidate.open(encoding="utf-8", newline="") as stream:
        rows = [{k: r[k] for k in ("opportunity_id", "object_type", "publication_date", "title", "description")}
                for r in csv.DictReader(stream)]
    with availability.open(encoding="utf-8", newline="") as stream:
        available = list(csv.DictReader(stream))
    def load(meta):
        path = initial_source_path(meta["source_path"])
        verify_hashes({str(path): meta["sha256"]})
        inputs[str(path)] = meta["sha256"]
        return json.loads(path.read_text()), path
    samples, inventory, queue, universe = selected_metadata(rows, available, registry["excluded_ocids"], load)
    selection = {"version": "document_validation_selection_v2", "seed": SEED, "selected_opportunities": samples,
        "excluded_ocids": registry["excluded_ocids"], "known_development_pdf_sha256": registry["known_pdf_sha256"],
        "source_design_manifest_sha256": run["source_design_manifest_sha256"],
        "integrated_design_manifest_sha256": sha(directory / "manifest.json"), "eligible_universe": universe,
        "no_silent_replacements": True, "reference_created": False, "predictions_generated": False,
        "metadata_only_selection": True, "ranking_inputs_modified": False}
    files = {"selection.json": selection, "document_inventory.json": inventory, "queue.json": queue,
        "summary.json": {"selected_opportunities": len(samples), "unique_ocids": len({r["opportunity_id"] for r in samples}),
            "eligible_universe": universe, "excluded_ocids": len(registry["excluded_ocids"]),
            "known_pdf_hashes": len(registry["known_pdf_sha256"]), "eligible_initial_pdf_sources": len(queue),
            "source_status": dict(Counter(s["status"] for s in samples)), "pdf_downloads_performed": False,
            "reference_created": False, "predictions_generated": False, "metrics_computed": False,
            "ranking_inputs_modified": False}}
    for batch, start in enumerate(range(0, len(queue), 5), 1):
        prefix = f"batches/batch_{batch:02d}/"
        plan = {"purpose": "new_source_reference_access_before_extraction_predictions", "selected": queue[start:start+5],
            "limits": {"documents": 5, "max_bytes_per_document": 10485760, "timeout_seconds": 30,
                       "max_pages_per_document": 120, "workers": 1, "ocr": False},
            "input_sha256": inputs, "known_development_pdf_sha256": registry["known_pdf_sha256"],
            "source_design_manifest_sha256": run["source_design_manifest_sha256"],
            "integrated_design_manifest_sha256": sha(directory / "manifest.json"),
            "ranking_eligible": False, "historical_content_proven": False}
        # Compute exactly the bytes dump() will write, without premature file writes.
        import hashlib
        plan_sha = hashlib.sha256((json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode()).hexdigest()
        files[prefix + "access_plan.json"] = plan
        files[prefix + "manifest.json"] = {"input_sha256": inputs, "output_sha256": {"access_plan.json": plan_sha}}
    return write_artifact(output, files, inputs)


def audit_selection(directory, selection_dir, output):
    """Independent eligibility/hash-order check without the selection helper."""
    run = validate_run(directory)
    inputs = {}
    checked(directory, inputs)
    checked(selection_dir, inputs)
    selection = json.loads((selection_dir / "selection.json").read_text())
    if (selection["integrated_design_manifest_sha256"] != sha(directory / "manifest.json")
            or selection["source_design_manifest_sha256"] != run["source_design_manifest_sha256"]):
        raise ValueError("Selection belongs to a different frozen protocol")
    registry = json.loads((Path(run["source_design_path"]) / "exclusion_registry.json").read_text())
    if selection["excluded_ocids"] != registry["excluded_ocids"] or selection["known_development_pdf_sha256"] != registry["known_pdf_sha256"]:
        raise ValueError("Selection dropped known-contamination exclusions")
    import hashlib
    import unicodedata
    excluded = set(registry["excluded_ocids"])
    eligible = []
    path = ROOT / "data/processed/contrast_candidates_v1/annual_v1/test_opportunities.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            domain = " ".join("".join(c for c in unicodedata.normalize("NFKD", row["title"] + " " + row["description"])
                                      if not unicodedata.combining(c)).casefold().split())
            if (row["opportunity_id"] not in excluded and row["object_type"] == "services"
                    and row["publication_date"].startswith("2025-") and "consultoria de obra" not in domain
                    and "consultoria en obra" not in domain):
                eligible.append(row["opportunity_id"])
    expected = sorted(eligible, key=lambda ocid: (hashlib.sha256((SEED + ":" + ocid).encode()).hexdigest(), ocid))[:50]
    cases = selection["selected_opportunities"]
    if [r["opportunity_id"] for r in cases] != expected or [r["sample_number"] for r in cases] != list(range(1,51)):
        raise ValueError("Sample differs from independently recomputed fixed order")
    queue = json.loads((selection_dir / "queue.json").read_text())
    if len({r["url"] for r in queue}) != len(queue):
        raise ValueError("Duplicate source URL in access queue")
    if {r["opportunity_id"] for r in queue} != {r["opportunity_id"] for r in cases if r["status"] == "eligible_initial_pdf_selected"}:
        raise ValueError("Queue and fifty-case selection disagree")
    # Independently ordered population plus exact metadata/source reconstruction:
    # check all queue fields and batch plans, not merely queue count/URL safety.
    base = path.parent
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [{k: r[k] for k in ("opportunity_id", "object_type", "publication_date", "title", "description")}
                for r in csv.DictReader(stream)]
    with (base / "opportunity_availability.csv").open(encoding="utf-8", newline="") as stream:
        available = list(csv.DictReader(stream))
    def load(meta):
        source_path = initial_source_path(meta["source_path"])
        verify_hashes({str(source_path): meta["sha256"]})
        return json.loads(source_path.read_text()), source_path
    recomputed, inventory, fixed_queue, universe = selected_metadata(rows, available, excluded, load)
    if recomputed != cases or fixed_queue != queue or inventory != json.loads((selection_dir / "document_inventory.json").read_text()):
        raise ValueError("Selection/document metadata changed after fixation")
    plans = sorted((selection_dir / "batches").glob("*/access_plan.json"))
    if len(plans) != (len(queue) + 4) // 5:
        raise ValueError("Incomplete fixed access batch plans")
    for n, plan_path in enumerate(plans):
        plan = json.loads(plan_path.read_text())
        if (plan["selected"] != queue[n*5:n*5+5]
                or plan["integrated_design_manifest_sha256"] != sha(directory / "manifest.json")
                or plan["source_design_manifest_sha256"] != run["source_design_manifest_sha256"]):
            raise ValueError("Access plan differs from fixed selection")
        verify_artifact(plan_path.parent, inputs)
    summary = {"stage": "selection_audited_before_document_access", "selected_cases": len(cases),
        "eligible_universe_independently_recomputed": len(eligible), "fixed_order_matches": True,
        "known_ocid_overlap": len(set(expected) & excluded), "unique_document_urls": len(queue),
        "metadata_only": True, "pdf_content_inspected": False, "reference_created": False,
        "predictions_generated": False, "metrics_computed": False, "ranking_inputs_modified": False}
    return write_artifact(output, {"summary.json": summary}, inputs)


def source_gate(directory, selection_dir, audit_dir, source_dir, output):
    """Link every source to the audited fixed sample before reference/inference."""
    run = validate_run(directory)
    inputs = {}
    for artifact in (directory, selection_dir, audit_dir):
        checked(artifact, inputs)
    audit_manifest = json.loads((audit_dir / "manifest.json").read_text())
    if audit_manifest["input_sha256"].get(str(selection_dir / "manifest.json")) != sha(selection_dir / "manifest.json"):
        raise ValueError("Audit does not cover this exact sample")
    if json.loads((audit_dir / "summary.json").read_text())["fixed_order_matches"] is not True:
        raise ValueError("Selection has not passed the independent audit")
    documents, ledger = load_sources(source_dir, inputs, directory)
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    if source_manifest["input_sha256"].get(str(selection_dir / "manifest.json")) != sha(selection_dir / "manifest.json"):
        raise ValueError("Source view does not depend on the fixed fifty-case selection")
    selected = json.loads((selection_dir / "selection.json").read_text())["selected_opportunities"]
    if [(r["sample_number"], r["opportunity_id"]) for r in ledger] != [(r["sample_number"], r["opportunity_id"]) for r in selected]:
        raise ValueError("Source denominator differs from the audited fifty-case sample")
    queue = {r["opportunity_id"]: r for r in json.loads((selection_dir / "queue.json").read_text())}
    for doc in documents:
        fixed = queue[doc["opportunity_id"]]
        if any(doc["source"].get(k) != v for k, v in fixed.items()):
            raise ValueError("Source bytes refer to a different document than selected")
    summary = {"stage": "source_population_ready_for_reference_review" if documents else "stopped_no_complete_source_documents",
        "selected_cases": len(ledger), "complete_unique_source_documents": len(documents),
        "physical_pages": sum(len(d["pages"]) for d in documents),
        "unresolved_ocr_pages": sum(p["selected_ocr_unresolved"] for d in documents for p in d["pages"]),
        "source_status_counts": dict(Counter(r["source_status"] for r in ledger)),
        "ready_for_reference_review": bool(documents), "ready_for_inference": False,
        "reference_created": False, "predictions_generated": False, "metrics_computed": False,
        "metrics": None, "no_source_is_not_negative": True, "ranking_inputs_modified": False}
    return write_artifact(output, {"summary.json": summary}, inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "select", "audit-selection", "source-gate"))
    for name in ("source-design", "design", "selection", "audit", "source", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    needed = {"freeze": ("source_design", "output"), "select": ("design", "output"),
              "audit-selection": ("design", "selection", "output"),
              "source-gate": ("design", "selection", "audit", "source", "output")}[args.command]
    if any(getattr(args, n) is None for n in needed):
        parser.error("Required: " + ", ".join(needed))
    function = {"freeze": freeze, "select": select, "audit-selection": audit_selection, "source-gate": source_gate}[args.command]
    print(json.dumps(function(*(getattr(args, n).resolve() for n in needed)), ensure_ascii=False))
