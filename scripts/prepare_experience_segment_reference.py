"""Save source-only development cases before running the new segment parser.

These cases were selected after prior inspection. They are automated regression
references, not blind annotations, expert judgements or a held-out F1 dataset.
The builder never obtains expected answers from a parser result.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("essalud_qualification", "2025-2543-904", 36, "bidder", "qualification", "explicit_clause_extracted", "1000000.00", "PEN", 15),
    ("essalud_corrupt_ocr", "2025-2543-904", 34, "bidder", "qualification", "mention_without_supported_explicit_threshold", None, None, None),
    ("essalud_annex_reference_in_body", "2025-2543-904", 37, None, "qualification", None, None, None, None),
    ("essalud_annex_form", "2025-2543-904", 68, "bidder", "annex_form", "annex_form_not_requirement", None, None, None),
    ("junin_qualification", "2025-10478-41", 31, "bidder", "qualification", "explicit_clause_extracted", "40000.00", "PEN", 10),
    ("junin_personnel", "2025-10478-41", 29, "key_personnel", "qualification", "personnel_mention_not_bidder_requirement", None, None, None),
    ("junin_evaluation_factor", "2025-10478-41", 33, "bidder", "evaluation_factor", "explicit_clause_extracted", "40000.00", "PEN", 10),
    ("bagua_blank_annex", "2025-88-43", 86, "bidder", "annex_form", "annex_form_not_requirement", None, None, None),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, input_dir):
    if output.exists():
        raise ValueError("Choose a new reference output")
    path = input_dir / "document_evidence.json"
    manifest = json.loads((input_dir / "manifest.json").read_text())
    if sha(path) != manifest["output_sha256"][path.name]:
        raise ValueError("Evidence packet changed")
    docs = json.loads(path.read_text())
    cases = []
    for case_id, suffix, page, subject, role, status, amount, currency, years in CASES:
        doc = next(d for d in docs if d["opportunity_id"].endswith(suffix))
        source = next(p for p in doc["pages"] if p["page"] == page)
        if hashlib.sha256(source["text"].encode()).hexdigest() != source["text_sha256"]:
            raise ValueError("Reference page hash mismatch")
        cases.append({"case_id": case_id, "opportunity_id": doc["opportunity_id"],
                      "source": doc["source"], "page": page, "source_page": source,
                      "expected": {"subject": subject, "section_role": role, "status": status,
                                   "billed_amount": amount, "currency": currency, "lookback_years": years}})
    payload = {"reference_type": "post_inspection_automated_development_regression",
               "human_annotations": False, "expert_validation": False, "blind_selection": False,
               "expected_values_from_extractor_outputs": False, "population_f1_estimable": False,
               "scope": "Eight known page cases in three documents. Other pages remain unlabelled.",
               "cases": cases}
    output.mkdir(parents=True)
    (output / "reference_cases.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"input_sha256": {str(path): sha(path)},
        "code_sha256": {str(Path(__file__)): sha(__file__)},
        "output_sha256": {"reference_cases.json": sha(output / "reference_cases.json")}}, indent=2) + "\n")
    return {"development_cases": len(cases), "documents": len({c["opportunity_id"] for c in cases}),
            "population_f1_estimable": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "artifacts/experience_document_evidence_v2")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/experience_segment_reference_v1")
    a = p.parse_args()
    print(json.dumps(prepare(a.output, a.input)))
