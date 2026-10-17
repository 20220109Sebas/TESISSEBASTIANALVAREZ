"""Apply frozen rules to the provenance-preserving native/OCR text view."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.process_browser_experience_sources import (
    ROOT, dump, protected_outputs, verify_hashes, sha, check_code,
    validate_result, verify_variable, verify_record, verify_ref,
    extract_experience_segments, represent_experience_clause, extract_formulations,
)


def process(inspection, output):
    if output.exists():
        raise ValueError("Preserve existing output and use a new directory")
    frozen_path = ROOT / "artifacts/experience_substitute_plan_v1/pending_plan.json"
    frozen = json.loads(frozen_path.read_text())
    check_code(frozen)
    manifest_path = inspection / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_hashes(manifest["input_sha256"])
    verify_hashes({str(inspection / p): h for p, h in manifest["output_sha256"].items()})
    packet_path = inspection / "pdf_text.json"
    inputs = dict(manifest["input_sha256"])
    inputs.update(protected_outputs())
    inputs.update(frozen["frozen_rule_and_runner_sha256"])
    inputs.update({str(manifest_path): sha(manifest_path), str(packet_path): sha(packet_path),
                   str(frozen_path): sha(frozen_path)})
    documents, variables, formulations = [], [], []
    for doc in json.loads(packet_path.read_text())["documents"]:
        source, pages = doc["source"], doc["pages"]
        if not source["source_substitution"] or source["ranking_eligible"]:
            raise ValueError("Unexpected source role")
        if source["opportunity_id"] not in {s["opportunity_id"] for s in frozen["pending_sources"]}:
            raise ValueError("Source outside the fixed queue")
        inputs[source["file_path"]] = source["sha256"]
        for page in pages:
            for text_key, hash_key in (("text", "text_sha256"), ("native_text", "native_text_sha256")):
                if hashlib.sha256(page[text_key].encode()).hexdigest() != page[hash_key]:
                    raise ValueError("Text hash differs")
            if page["selected_ocr_unresolved"]:
                raise ValueError("Unresolved selected OCR page")
            if page["ocr_used"]:
                inputs[page["ocr_json_path"]] = page["ocr_json_sha256"]
                if page["extraction_method"] != "apple_vision_local_ocr":
                    raise ValueError("OCR attribution missing")
        result = extract_experience_segments(pages)
        validate_result(result, pages)
        documents.append({"opportunity_id": doc["opportunity_id"], "source": source,
                          "pages": pages, "result": result, "processed_pages": len(pages),
                          "historical_content_proven": False, "ranking_eligible": False,
                          "absence_of_requirements_proven": False})
        for key in ("qualification_clauses", "evaluation_factor_clauses"):
            for clause in result[key]:
                variable = represent_experience_clause(clause)
                verify_variable(variable, clause, pages)
                variables.append({"opportunity_id": doc["opportunity_id"], "source": source,
                                  "variable": variable})
        extra = extract_formulations(pages, result["mentions"])
        by_page = {p["page"]: p for p in pages}
        for row in extra["records"]:
            verify_record(row, by_page)
        for warning in extra["warnings"]:
            verify_ref(warning["evidence"], by_page)
        formulations.append({"opportunity_id": doc["opportunity_id"], "source": source, "result": extra})
    summary = {
        "documents": len(documents), "physical_pages": sum(len(d["pages"]) for d in documents),
        "ocr_pages_used": sum(p["ocr_used"] for d in documents for p in d["pages"]),
        "experience_mentions": sum(len(d["result"]["mentions"]) for d in documents),
        "qualification_clauses": sum(len(d["result"]["qualification_clauses"]) for d in documents),
        "evaluation_factor_clauses": sum(len(d["result"]["evaluation_factor_clauses"]) for d in documents),
        "conditional_variables": len(variables),
        "conditional_monetary_branches": sum(len(v["variable"]["branches"]) for v in variables),
        "additional_literal_candidates": sum(len(d["result"]["records"]) for d in formulations),
        "additional_candidate_kinds": dict(Counter(r["kind"] for d in formulations for r in d["result"]["records"])),
        "frozen_rule_files_verified": len(frozen["frozen_rule_and_runner_sha256"]),
        "protected_outputs_verified": len(protected_outputs()),
        "native_text_preserved": True, "literal_evidence_verified": True,
        "independent_semantic_accuracy_validated": False, "f1_estimable": False,
        "ranking_inputs_modified": False, "human_annotation": False,
        "historical_content_proven": False, "negative_labels_created": False,
    }
    verify_hashes(inputs)
    check_code(frozen)
    output.mkdir(parents=True)
    for name, value in (("document_evidence.json", documents), ("conditional_variables.json", variables),
                        ("formulation_candidates.json", formulations), ("summary.json", summary)):
        dump(output / name, value)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "code_sha256": {str(Path(__file__)): sha(__file__)},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(process(args.inspection.resolve(), args.output.resolve())))
