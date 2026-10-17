"""Apply unchanged experience rules to a separate public-SEACE substitute packet."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.register_browser_experience_sources import dump, protected_outputs, verify_hashes
from scripts.fetch_experience_documents import sha
from scripts.fetch_pending_experience_sources import check_code
from scripts.compare_experience_segments import validate_result
from scripts.represent_experience_conditions import verify_variable
from scripts.extend_experience_formulations import verify_record, verify_ref
from tesis_recommender.ocr_evidence import merge_page
from tesis_recommender.experience_segments import extract_experience_segments
from tesis_recommender.experience_conditions import represent_experience_clause
from tesis_recommender.experience_formulations import extract_formulations


def process(inspection, output):
    if output.exists():
        raise ValueError("Use a new extraction directory")
    frozen_path = ROOT / "artifacts/experience_substitute_plan_v1/pending_plan.json"
    frozen = json.loads(frozen_path.read_text())
    check_code(frozen)
    manifest_path = inspection / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = inspection / "pdf_text.json"
    verify_hashes({str(inspection / p): h for p, h in manifest["output_sha256"].items()})
    packet = json.loads(path.read_text())
    inputs = {str(path): sha(path), str(manifest_path): sha(manifest_path), str(frozen_path): sha(frozen_path)}
    inputs.update(frozen["frozen_rule_and_runner_sha256"])
    inputs.update(protected_outputs())
    documents, variables, formulations = [], [], []
    for document in packet["documents"]:
        source = document.get("source")
        if not source or not source.get("source_substitution") or source["ranking_eligible"]:
            raise ValueError("Unexpected source role or ranking eligibility")
        if source["opportunity_id"] not in {s["opportunity_id"] for s in frozen["pending_sources"]}:
            raise ValueError("Document is outside the fixed substitute queue")
        inputs[source["file_path"]] = source["sha256"]
        verify_hashes({source["file_path"]: source["sha256"]})
        pages = [merge_page(p) for p in document.get("pages", [])]
        result = extract_experience_segments(pages)
        validate_result(result, pages)
        documents.append({"opportunity_id": document["opportunity_id"], "source": source,
            "status": document["status"], "pages": pages, "result": result,
            "total_pages": document.get("total_pages"), "processed_pages": len(pages),
            "truncated_by_page_limit": document.get("truncated_by_page_limit"),
            "historical_content_proven": False, "ranking_eligible": False,
            "absence_of_requirements_proven": False})
        for key in ("qualification_clauses", "evaluation_factor_clauses"):
            for clause in result[key]:
                variable = represent_experience_clause(clause)
                verify_variable(variable, clause, pages)
                variables.append({"opportunity_id": document["opportunity_id"], "source": source, "variable": variable})
        extra = extract_formulations(pages, result["mentions"])
        by_page = {p["page"]: p for p in pages}
        for row in extra["records"]:
            verify_record(row, by_page)
        for warning in extra["warnings"]:
            verify_ref(warning["evidence"], by_page)
        formulations.append({"opportunity_id": document["opportunity_id"], "source": source, "result": extra})
    summary = {
        "new_complete_substitute_pdfs": len(documents),
        "pages_processed": sum(d["processed_pages"] for d in documents),
        "documents_with_usable_native_text": sum(any(p["text"].strip() for p in d["pages"]) for d in documents),
        "truncated_documents": sum(bool(d["truncated_by_page_limit"]) for d in documents),
        "experience_mentions": sum(len(d["result"]["mentions"]) for d in documents),
        "qualification_clauses": sum(len(d["result"]["qualification_clauses"]) for d in documents),
        "evaluation_factor_clauses": sum(len(d["result"]["evaluation_factor_clauses"]) for d in documents),
        "conditional_variables": len(variables),
        "conditional_monetary_branches": sum(len(v["variable"]["branches"]) for v in variables),
        "additional_literal_candidates": sum(len(d["result"]["records"]) for d in formulations),
        "additional_candidate_kinds": dict(Counter(r["kind"] for d in formulations for r in d["result"]["records"])),
        "frozen_rule_files_verified": len(frozen["frozen_rule_and_runner_sha256"]),
        "protected_annual_outputs_verified": len(protected_outputs()) - 4,
        "protected_extraction_outputs_verified": 4,
        "literal_evidence_verified": True, "independent_semantic_accuracy_validated": False,
        "f1_estimable": False, "new_ocr_performed": False, "new_embeddings_generated": False,
        "original_documents_replaced": False, "historical_content_proven": False,
        "ranking_inputs_modified": False, "human_annotation": False, "negative_labels_created": False}
    verify_hashes(inputs)
    check_code(frozen)
    output.mkdir(parents=True)
    for name, value in (("document_evidence.json", documents), ("conditional_variables.json", variables),
                        ("formulation_candidates.json", formulations), ("summary.json", summary)):
        dump(output / name, value)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "code_sha256": {str(Path(__file__)): sha(__file__),
                        str(ROOT / "scripts/register_browser_experience_sources.py"): sha(ROOT / "scripts/register_browser_experience_sources.py")},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(process(args.inspection.resolve(), args.output.resolve())))
