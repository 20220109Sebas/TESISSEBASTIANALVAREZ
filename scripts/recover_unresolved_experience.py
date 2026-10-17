"""Build and verify separate candidates from saved native/OCR evidence."""
import argparse
import json
import re
from decimal import Decimal
from pathlib import Path

from scripts.process_completed_experience_text import (
    ROOT, sha, dump, verify_hashes, protected_outputs, check_code, verify_ref, validate_result,
)
from tesis_recommender.experience_recovery_candidates import recover_candidates


def verify_candidate(row, pages):
    for key in ("evidence", "context", "window_evidence"):
        if row.get(key) is not None:
            verify_ref(row[key], pages)
    raw = row["evidence"]["evidence"]
    if row["kind"] == "billed_experience_multiple_of_procedure_amount":
        # Different token validation from the extraction regex, literal only.
        tokens = raw.upper().split()
        factors = {"UNA": 1, "DOS": 2, "TRES": 3}
        factor = next((factors[t] for t in tokens if t in factors), None)
        if factor != row["multiplier"] or row["absolute_amount"] is not None or row["currency"] is not None:
            raise ValueError("Relative value differs or money was inferred")
        if row["reference_basis"] != "procedure_amount_not_offered_value":
            raise ValueError("Reference basis changed")
    elif row["kind"] == "literal_money_in_unresolved_bidder_context":
        token = re.fullmatch(r"(S\s*/\.?|US\s*\$|USD|PEN)\s*(\d[\d.,]*)", raw, re.I)
        if not token:
            raise ValueError("Invalid monetary token")
        value = token.group(2)
        if re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d{2})?", value):
            value = value.replace(",", "")
        elif re.fullmatch(r"(?:\d+|\d{1,3}(?:\.\d{3})+),\d{2}", value):
            value = value.replace(".", "").replace(",", ".")
        else:
            raise ValueError("Ambiguous money")
        currency = "USD" if token.group(1).upper().startswith("US") else "PEN"
        if Decimal(value) <= 0 or Decimal(value) != Decimal(row["billed_amount"]) or currency != row["currency"]:
            raise ValueError("Literal value differs")
    else:
        raise ValueError("Unknown candidate kind")
    if row["window_evidence"]:
        token = re.search(r"\(?(\d{1,2})\)?\s+a[ñn]os\s+anteriores", row["window_evidence"]["evidence"], re.I)
        if not token or int(token.group(1)) != row["lookback_years"]:
            raise ValueError("Window differs")
    elif row["lookback_years"] is not None:
        raise ValueError("Window lacks evidence")
    for key in ("ranking_eligible", "independently_validated", "historical_content_proven", "absolute_amount_inferred"):
        if row[key] is not False:
            raise ValueError("Unexpected inference")


def build(input_dir, output):
    if output.exists():
        raise ValueError("Use a new output directory")
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = dict(manifest["input_sha256"])
    inputs.update(manifest["code_sha256"])
    inputs.update({str(input_dir / p): h for p, h in manifest["output_sha256"].items()})
    inputs[str(manifest_path)] = sha(manifest_path)
    inputs.update(protected_outputs())
    frozen = json.loads((ROOT / "artifacts/experience_substitute_plan_v1/pending_plan.json").read_text())
    check_code(frozen)
    verify_hashes(inputs)
    documents = []
    for doc in json.loads((input_dir / "document_evidence.json").read_text()):
        validate_result(doc["result"], doc["pages"])
        result = recover_candidates(doc["pages"], doc["result"])
        for row in result["records"]:
            verify_candidate(row, {p["page"]: p for p in doc["pages"]})
        documents.append({"opportunity_id": doc["opportunity_id"], "source": doc["source"], "result": result})
    summary = {"documents": len(documents),
        "new_separate_candidates": sum(len(d["result"]["records"]) for d in documents),
        "literal_evidence_verified": True, "independent_accuracy_validated": False,
        "f1_estimable": False, "frozen_rules_modified": False, "ranking_inputs_modified": False,
        "new_ocr_performed": False, "human_annotation": False,
        "protected_outputs_verified": len(protected_outputs()),
        "frozen_rule_files_verified": len(frozen["frozen_rule_and_runner_sha256"])}
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "candidates.json", documents)
    dump(output / "summary.json", summary)
    code = [Path(__file__), ROOT / "src/tesis_recommender/experience_recovery_candidates.py"]
    dump(output / "manifest.json", {"input_sha256": inputs,
        "code_sha256": {str(p): sha(p) for p in code},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.input.resolve(), args.output.resolve())))
