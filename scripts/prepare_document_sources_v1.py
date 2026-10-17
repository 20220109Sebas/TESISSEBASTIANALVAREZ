"""Generic bounded native/OCR source preparation, separate from inference.

Use the recorded Python environment for runtime/native/OCR/compose commands. This
module never invokes an experience detector or reads reference values.
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.register_browser_experience_sources import dump, sha, verify_hashes
from tesis_recommender.annex_sources import safe_annex_url
from tesis_recommender.document_source_preparation_v1 import (
    MAX_BYTES, check_ocr, combine_batches, compose_documents, prepare_native, text_sha,
)


def runtime_report():
    import pypdf
    package = Path(pypdf.__file__).resolve().parent
    product = subprocess.check_output(["/usr/bin/sw_vers", "-productVersion"], text=True).strip()
    build = subprocess.check_output(["/usr/bin/sw_vers", "-buildVersion"], text=True).strip()
    return {"python_executable": str(Path(sys.executable).resolve()), "python_sha256": sha(Path(sys.executable).resolve()),
        "python_version": sys.version, "pypdf_version": pypdf.__version__, "pypdf_path": str(package),
        "pypdf_code_sha256": {str(p): sha(p) for p in sorted(package.rglob("*.py"))},
        "macos_product_version": product, "macos_build": build,
        "vision_os_version_string": f"Version {product} (Build {build})"}


def load_native(directory, inputs):
    verify_artifact(directory, inputs)
    inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
    return json.loads((directory / "native_plan.json").read_text())


def prepare_batch(design_dir, access_dir, output):
    from scripts.freeze_document_source_design_v1 import validate_design
    if output.exists():
        raise ValueError("Use a fresh native-plan directory")
    design = validate_design(design_dir)
    inputs = {}
    verify_artifact(design_dir, inputs)
    inputs.update(json.loads((design_dir / "manifest.json").read_text())["input_sha256"])
    manifest = json.loads((access_dir / "manifest.json").read_text())
    verify_hashes({str(access_dir / p): h for p, h in manifest["output_sha256"].items()})
    inputs.update({str(access_dir / p): h for p, h in manifest["output_sha256"].items()})
    inputs[str(access_dir / "manifest.json")] = sha(access_dir / "manifest.json")
    plan_path = Path(manifest["access_plan_path"])
    verify_hashes({str(plan_path): manifest["access_plan_sha256"]})
    access_plan = json.loads(plan_path.read_text())
    if access_plan["source_design_manifest_sha256"] != sha(design_dir / "manifest.json"):
        raise ValueError("Access plan belongs to another source design")
    verify_hashes(access_plan["input_sha256"])
    inputs.update(access_plan["input_sha256"])
    inputs[str(plan_path)] = manifest["access_plan_sha256"]
    rows = json.loads((access_dir / "access_results.json").read_text())
    selected = access_plan["selected"]
    if (not 0 < len(selected) <= 5 or len({r["opportunity_id"] for r in selected}) != len(selected)
            or [r["opportunity_id"] for r in rows] != [r["opportunity_id"] for r in selected]):
        raise ValueError("Access ledger must retain the entire fixed batch, in order")
    exclusions = json.loads((design_dir / "exclusion_registry.json").read_text())
    if {r["opportunity_id"] for r in selected} & set(exclusions["excluded_ocids"]):
        raise ValueError("Previously observed opportunities cannot enter new source preparation")
    if (access_plan["limits"]["max_bytes_per_document"] != MAX_BYTES
            or access_plan["limits"]["max_pages_per_document"] != 120
            or access_plan["limits"]["timeout_seconds"] != 30):
        raise ValueError("Access limits differ from fixed design")
    known = set(design["known_pdf_sha256"])
    if set(access_plan["known_development_pdf_sha256"]) != known:
        raise ValueError("Access plan lost the frozen duplicate-PDF registry")
    for row, fixed in zip(rows, selected):
        if any(row.get(k) != v for k, v in fixed.items()) or not safe_annex_url(row.get("url")):
            raise ValueError("Fetched document differs from fixed source metadata")
    hashes = [r["sha256"] for r in rows if r["status"] == "downloaded_pdf"]
    documents = []
    for row in rows:
        base = {"opportunity_id": row["opportunity_id"], "source": row, "pages": [],
                "ranking_eligible": False, "historical_content_proven": False}
        if row["status"] != "downloaded_pdf":
            documents.append(dict(base, status="access_not_complete_pdf", access_status=row["status"]))
            continue
        path = Path(row["file_path"]).resolve()
        if not path.is_relative_to(access_dir.resolve()):
            raise ValueError("Downloaded source path escapes the fixed access directory")
        verify_hashes({str(path): row["sha256"]})
        inputs[str(path)] = row["sha256"]
        if path.stat().st_size != row["bytes"] or not 0 < row["bytes"] <= MAX_BYTES:
            raise ValueError("Source byte count differs from fixed access metadata")
        if row["sha256"] in known or hashes.count(row["sha256"]) > 1:
            documents.append(dict(base, status="duplicate_pdf_not_new_no_replacement"))
            continue
        try:
            process = subprocess.run([design["source_python"], str(Path(__file__).resolve()), "worker"],
                input=json.dumps(row), capture_output=True, text=True, check=True, timeout=90)
            document = json.loads(process.stdout)
            if document["opportunity_id"] != row["opportunity_id"] or document["source"] != row:
                raise ValueError("Native worker source differs")
            documents.append(document)
        except subprocess.TimeoutExpired:
            documents.append(dict(base, status="native_worker_timeout"))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            documents.append(dict(base, status="native_worker_failed"))
    verify_hashes(inputs)
    output.mkdir(parents=True)
    native = {"version": "document_native_plan_v1", "documents": documents,
        "design_dir": str(design_dir.resolve()), "design_manifest_sha256": sha(design_dir / "manifest.json"),
        "selected_pages": [{"opportunity_id": d["opportunity_id"], "source_pdf_sha256": d["source"]["sha256"],
            "page": p["page"]} for d in documents if d["status"] == "complete_native_view"
            for p in d["pages"] if p["selected_for_ocr"]],
        "selection_before_ocr_and_inference": True, "reference_values_used": False,
        "native_text_is_not_layout_fidelity_or_semantic_accuracy": True}
    dump(output / "native_plan.json", native)
    summary = {"sources_in_batch": len(rows), "complete_native_documents": sum(d["status"] == "complete_native_view" for d in documents),
               "planned_ocr_pages": len(native["selected_pages"]), "new_ocr_performed": False, "predictions_generated": False}
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def run_ocr(native_dir, output):
    from scripts.freeze_document_source_design_v1 import validate_design
    if output.exists():
        raise ValueError("Use a fresh OCR directory")
    inputs = {}
    native = load_native(native_dir, inputs)
    design_dir = Path(native["design_dir"])
    verify_hashes({str(design_dir / "manifest.json"): native["design_manifest_sha256"]})
    design = validate_design(design_dir)
    documents = {d["opportunity_id"]: d for d in native["documents"]}
    output.mkdir(parents=True)
    (output / "pages").mkdir()
    rows, errors, stopped = [], 0, False
    for planned in native["selected_pages"]:
        row = dict(planned, status="not_attempted_after_error_stop")
        if not stopped:
            destination = output / "pages" / f'{planned["source_pdf_sha256"]}_{planned["page"]:03d}.json'
            source = documents[planned["opportunity_id"]]["source"]
            try:
                subprocess.run([design["vision_binary"], source["file_path"], str(planned["page"]), str(destination.resolve())],
                               capture_output=True, text=True, check=True, timeout=45)
                raw = json.loads(destination.read_text())
                check_ocr(raw, planned["page"], design["source_runtime"]["vision_os_version_string"])
                row.update(status="ok" if raw["text"].strip() else "empty_ocr", text=raw["text"],
                    text_sha256=text_sha(raw["text"]), ocr_json_path=str(destination.resolve()), ocr_json_sha256=sha(destination))
                errors = 0
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, KeyError, OSError) as exc:
                row.update(status="ocr_error", error_type=type(exc).__name__)
                errors += 1
                stopped = errors >= 2
        rows.append(row)
        dump(output / "ocr_results.json", rows)
    dump(output / "ocr_results.json", rows)
    validate_design(design_dir)
    verify_hashes(inputs)
    summary = {"planned_pages": len(rows), "pages_attempted": sum(r["status"] != "not_attempted_after_error_stop" for r in rows),
        "pages_with_ocr_text": sum(r["status"] == "ok" for r in rows), "stopped_after_errors": stopped,
        "ocr_executed": any(r["status"] != "not_attempted_after_error_stop" for r in rows), "model_inference_performed": False}
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return summary


def compose(native_dir, ocr_dir, output):
    from scripts.freeze_document_source_design_v1 import validate_design
    if output.exists():
        raise ValueError("Use a fresh source-view directory")
    inputs = {}
    native = load_native(native_dir, inputs)
    design_dir = Path(native["design_dir"])
    verify_hashes({str(design_dir / "manifest.json"): native["design_manifest_sha256"]})
    validate_design(design_dir)
    verify_artifact(ocr_dir, inputs)
    inputs.update(json.loads((ocr_dir / "manifest.json").read_text())["input_sha256"])
    rows = json.loads((ocr_dir / "ocr_results.json").read_text())
    documents = compose_documents(native["documents"], rows)
    validate_design(design_dir)
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "source_documents.json", documents)
    dump(output / "coverage_ledger.json", [{"opportunity_id": d["opportunity_id"], "source_status": d["status"],
        "absence_of_requirements_proven": False, "ranking_eligible": False} for d in native["documents"]])
    summary = {"complete_source_documents": len(documents), "physical_pages": sum(len(d["pages"]) for d in documents),
        "unresolved_ocr_pages": sum(p["selected_ocr_unresolved"] for d in documents for p in d["pages"]),
        "all_source_states_retained": True, "predictions_generated": False, "reference_created": False,
        "source_design_manifest_sha256": sha(design_dir / "manifest.json")}
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def native_worker(source):
    import pypdf
    path = Path(source["file_path"])
    if path.stat().st_size != source["bytes"] or sha(path) != source["sha256"]:
        raise ValueError("Native worker source bytes differ")
    with path.open("rb") as stream:
        if b"%PDF-" not in stream.read(1024):
            raise ValueError("Not a PDF source")
    try:
        return prepare_native(pypdf.PdfReader(path), source)
    except (pypdf.errors.PdfReadError, ValueError, KeyError, TypeError, RecursionError):
        return {"opportunity_id": source["opportunity_id"], "source": source, "pages": [],
                "status": "native_parse_error", "ranking_eligible": False, "historical_content_proven": False}


def combine_views(design_dir, selection_dir, views, output):
    from scripts.freeze_document_source_design_v1 import validate_design
    if output.exists():
        raise ValueError("Use a fresh combined source-view directory")
    validate_design(design_dir)
    inputs = {}
    verify_artifact(design_dir, inputs)
    verify_artifact(selection_dir, inputs)
    selection = json.loads((selection_dir / "selection.json").read_text())
    if selection["source_design_manifest_sha256"] != sha(design_dir / "manifest.json"):
        raise ValueError("Selection belongs to another source design")
    if len(selection["selected_opportunities"]) != 50:
        raise ValueError("Keep the entire fixed fifty-case selection")
    batches = []
    for directory in views:
        verify_artifact(directory, inputs)
        inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
        if json.loads((directory / "summary.json").read_text())["source_design_manifest_sha256"] != sha(design_dir / "manifest.json"):
            raise ValueError("Source view belongs to another preparation design")
        batches.append({"source_documents": json.loads((directory / "source_documents.json").read_text()),
                        "coverage_ledger": json.loads((directory / "coverage_ledger.json").read_text())})
    result = combine_batches(selection["selected_opportunities"], batches)
    verify_hashes(inputs)
    output.mkdir(parents=True)
    for name in ("source_documents", "coverage_ledger", "duplicate_documents_preserved"):
        dump(output / (name + ".json"), result[name])
    summary = {"selected_cases": len(result["coverage_ledger"]), "complete_unique_documents": len(result["source_documents"]),
        "duplicate_documents_retained_separately": len(result["duplicate_documents_preserved"]),
        "silent_replacements_performed": False, "reference_created": False, "predictions_generated": False}
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("runtime", "worker", "native", "ocr", "compose", "combine"))
    for flag in ("design", "selection", "access", "native", "ocr", "output"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--view", type=Path, action="append")
    args = parser.parse_args()
    if args.command == "runtime":
        result = runtime_report()
    elif args.command == "worker":
        result = native_worker(json.loads(sys.stdin.read()))
    elif args.command == "native":
        result = prepare_batch(args.design.resolve(), args.access.resolve(), args.output.resolve())
    elif args.command == "ocr":
        result = run_ocr(args.native.resolve(), args.output.resolve())
    elif args.command == "compose":
        result = compose(args.native.resolve(), args.ocr.resolve(), args.output.resolve())
    else:
        result = combine_views(args.design.resolve(), args.selection.resolve(), [p.resolve() for p in args.view], args.output.resolve())
    print(json.dumps(result, ensure_ascii=False))
