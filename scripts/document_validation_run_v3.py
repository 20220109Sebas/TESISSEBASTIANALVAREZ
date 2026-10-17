"""Metadata path revision of v2, before PDF access; same seed and 50 OCIDs.

Both pinned initial-release partitions are legitimate availability targets.
The rejected v2 integration and all original locks remain unmodified.
"""
import argparse
import csv
import hashlib
import json
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from scripts import document_validation_run_v2 as prior
from scripts.register_browser_experience_sources import dump, sha, verify_hashes
from scripts.audit_holdout_evaluation_readiness import verify_artifact
from tesis_recommender.document_sampling_v1 import SEED

ROOT = prior.ROOT
FILES = ("scripts/document_validation_run_v3.py", "tests/test_document_validation_run_v3.py",
         "docs/document_validation_run_v3.json")
PARTITIONS = (ROOT / "data/raw/oece/contrast_candidate_versions_v1", ROOT / "data/raw/oece/initial_releases")


def initial_source_path(relative):
    path = (ROOT / relative).resolve()
    if not any(path.is_relative_to(partition) for partition in PARTITIONS):
        raise ValueError("Source path outside the two pinned initial-release partitions")
    return path


def freeze(source_design, output):
    # v2 creates a fresh, not-yet-published artifact. Extend that new artifact's
    # registered code before returning it; never alter the rejected v2 lock.
    prior.freeze(source_design, output)
    manifest = json.loads((output / "manifest.json").read_text())
    code = {str(ROOT / p): sha(ROOT / p) for p in FILES}
    manifest["input_sha256"].update(code)
    for name in ("evaluation_design_lock.json", "integrated_run_lock.json"):
        lock = json.loads((output / name).read_text())
        lock["code_sha256"].update(code)
        lock["integration_revision"] = 3
        dump(output / name, lock)
    for rel in FILES:
        target = output / "code_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    summary = json.loads((output / "summary.json").read_text())
    summary.update(integration_revision=3, initial_release_partitions=2,
        rejected_v2_selection_preserved=True, sampling_population_or_order_changed=False,
        source_path_change_before_pdf_access=True)
    dump(output / "summary.json", summary)
    manifest["output_sha256"] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
                                  if p.is_file() and p.name != "manifest.json"}
    verify_hashes(manifest["input_sha256"])
    dump(output / "manifest.json", manifest)
    return summary


def validate_run(directory, *, before_selection=False):
    run = prior.validate_run(directory, before_selection=before_selection)
    if run.get("integration_revision") != 3:
        raise ValueError("Use the integrated v3 metadata-path revision")
    return run


def metadata(directory, inputs):
    run = validate_run(directory, before_selection=True)
    prior.checked(directory, inputs)
    registry = json.loads((Path(run["source_design_path"]) / "exclusion_registry.json").read_text())
    base = ROOT / "data/processed/contrast_candidates_v1/annual_v1"
    for path in (base / "test_opportunities.csv", base / "opportunity_availability.csv"):
        verify_hashes({str(path): inputs[str(path)]})
    with (base / "test_opportunities.csv").open(encoding="utf-8", newline="") as stream:
        rows = [{k: r[k] for k in ("opportunity_id", "object_type", "publication_date", "title", "description")}
                for r in csv.DictReader(stream)]
    with (base / "opportunity_availability.csv").open(encoding="utf-8", newline="") as stream:
        available = list(csv.DictReader(stream))
    def load(meta):
        path = initial_source_path(meta["source_path"])
        verify_hashes({str(path): meta["sha256"]})
        inputs[str(path)] = meta["sha256"]
        return json.loads(path.read_text()), path
    samples, inventory, queue, universe = prior.selected_metadata(rows, available, registry["excluded_ocids"], load)
    return run, registry, rows, samples, inventory, queue, universe


def select(directory, output):
    if output.exists():
        raise ValueError("Use a fresh selection directory; never replace fixed cases")
    inputs = {}
    run, registry, _, samples, inventory, queue, universe = metadata(directory, inputs)
    selection = {"version": "document_validation_selection_v3", "seed": SEED, "selected_opportunities": samples,
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
        plan_sha = hashlib.sha256((json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode()).hexdigest()
        files[prefix + "access_plan.json"] = plan
        files[prefix + "manifest.json"] = {"input_sha256": inputs, "output_sha256": {"access_plan.json": plan_sha}}
    return prior.write_artifact(output, files, inputs)


def audit_selection(directory, selection_dir, output):
    inputs = {}
    run, registry, rows, samples, inventory, queue, universe = metadata(directory, inputs)
    prior.checked(selection_dir, inputs)
    selection = json.loads((selection_dir / "selection.json").read_text())
    if (selection["integrated_design_manifest_sha256"] != sha(directory / "manifest.json")
            or selection["source_design_manifest_sha256"] != run["source_design_manifest_sha256"]
            or selection["excluded_ocids"] != registry["excluded_ocids"]
            or selection["known_development_pdf_sha256"] != registry["known_pdf_sha256"]):
        raise ValueError("Selection lost its frozen protocol or contamination registry")
    # Independent full-population eligibility/order calculation, not select_ids().
    eligible = []
    for row in rows:
        domain = " ".join("".join(c for c in unicodedata.normalize("NFKD", row["title"] + " " + row["description"])
                                  if not unicodedata.combining(c)).casefold().split())
        if (row["opportunity_id"] not in registry["excluded_ocids"] and row["object_type"] == "services"
                and row["publication_date"].startswith("2025-") and "consultoria de obra" not in domain
                and "consultoria en obra" not in domain):
            eligible.append(row["opportunity_id"])
    expected = sorted(eligible, key=lambda ocid: (hashlib.sha256((SEED + ":" + ocid).encode()).hexdigest(), ocid))[:50]
    if ([r["opportunity_id"] for r in selection["selected_opportunities"]] != expected
            or selection["selected_opportunities"] != samples or universe != len(eligible)):
        raise ValueError("Sample differs from independent eligibility/order calculation")
    if (json.loads((selection_dir / "document_inventory.json").read_text()) != inventory
            or json.loads((selection_dir / "queue.json").read_text()) != queue):
        raise ValueError("Document metadata differs from pinned initial releases")
    plans = sorted((selection_dir / "batches").glob("*/access_plan.json"))
    if len(plans) != (len(queue) + 4) // 5 or len({q["url"] for q in queue}) != len(queue):
        raise ValueError("Batch plan missing or duplicate URL")
    for n, path in enumerate(plans):
        verify_artifact(path.parent, inputs)
        plan = json.loads(path.read_text())
        if (plan["selected"] != queue[n*5:n*5+5] or plan["known_development_pdf_sha256"] != registry["known_pdf_sha256"]
                or plan["source_design_manifest_sha256"] != run["source_design_manifest_sha256"]
                or plan["integrated_design_manifest_sha256"] != sha(directory / "manifest.json")
                or plan["limits"] != {"documents": 5, "max_bytes_per_document": 10485760, "timeout_seconds": 30,
                                       "max_pages_per_document": 120, "workers": 1, "ocr": False}):
            raise ValueError("Access plan differs from fixed selection/design")
    summary = {"stage": "selection_audited_before_document_access", "selected_cases": len(samples),
        "eligible_universe_independently_recomputed": universe, "fixed_order_matches": True,
        "known_ocid_overlap": len(set(expected) & set(registry["excluded_ocids"])), "unique_document_urls": len(queue),
        "metadata_only": True, "pdf_content_inspected": False, "reference_created": False,
        "predictions_generated": False, "metrics_computed": False, "ranking_inputs_modified": False}
    return prior.write_artifact(output, {"summary.json": summary}, inputs)


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
    function = {"freeze": freeze, "select": select, "audit-selection": audit_selection, "source-gate": prior.source_gate}[args.command]
    print(json.dumps(function(*(getattr(args, n).resolve() for n in needed)), ensure_ascii=False))
