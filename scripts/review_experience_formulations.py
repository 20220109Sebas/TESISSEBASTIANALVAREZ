"""Record two source-reviewed formulation gaps without changing the frozen parser.

This is an automated post-inspection development review, not blind annotation.
Assertions are grounded in the rendered pages already inspected in source review.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    {"opportunity_id": "ocds-dgv273-seacev3-1181191", "page": 19,
     "pdf_sha256": "47a9f3b1c156baacaeac0e2c7c03b7d1d1cb43182ee6c62bd869c5b37ecf7a53",
     "text_sha256": "8e732b8395e993ef76b55e651f278ba15bd3c2840ab3e6185cdee825c234f836",
     "render_path": "tmp/pdfs/annex_review_v5/logistica-19.png",
     "render_sha256": "01105056169e90f890d81ca50c8a4550b69427b8404d27e18db3efa0c3f1faa6",
     "findings": [
         {"kind": "minimum_prior_service_count", "observed_value": 3,
          "source_text": "Experiencia no menor de tres (03) servicios logísticos"},
         {"kind": "billed_experience_fraction_of_offered_value", "observed_percentage": 20,
          "reference_basis": "offered_value_not_procurement_budget", "absolute_amount": None,
          "source_text": "El proveedor deberá acreditar como mínimo un monto facturado acumulado \nequivalente al 20% del valor ofertado en el servicio objeto del presente \nrequerimiento."}
     ], "parser_limit": "Supplier profile, service count and percentage expression are outside the absolute-billed-amount clause pattern."},
    {"opportunity_id": "ocds-dgv273-seacev3-1129720", "page": 38,
     "pdf_sha256": "e96b8cc4152e6ea6a38a5cd810c1b384791952dbdb1b3497925303adadb024a1",
     "text_sha256": "c3eb1354575dc611b396d07d22953a04785de1dcfa7042ccbd5de50bfaecb55d",
     "render_path": "tmp/pdfs/annex_review_v5/san-ignacio-38.png",
     "render_sha256": "c563202cd9c9b9478158b5b00024c1ec67975076203632539704b6546fa9988d",
     "findings": [
         {"kind": "explicit_amount_with_bracket_delimiter", "observed_numeric_literal": "221857.81",
          "currency_literal": "PEN", "promoted_to_validated_requirement": False,
          "source_text": "equivalente a [ S/ 221,857.81"},
         {"kind": "literal_lookback_window", "observed_years": 20,
          "source_text": "durante los \nveinte (20) años anteriores a la fecha de la presentación de ofertas"}
     ], "parser_limit": "The opening bracket prevents the money token match. A filled numeric literal in brackets is not proof that the field is blank or that all document conditions are resolved."}
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence_span(text, fragment):
    if text.count(fragment) != 1:
        raise ValueError("Reviewed literal missing or ambiguous")
    start = text.index(fragment)
    return {"char_start": start, "char_end": start + len(fragment), "evidence": fragment,
            "evidence_sha256": hashlib.sha256(fragment.encode()).hexdigest()}


def review(input_dir, output):
    if output.exists():
        raise ValueError("Choose a new formulation review directory")
    path = input_dir / "document_evidence.json"
    manifest = json.loads((input_dir / "manifest.json").read_text())
    if sha(path) != manifest["output_sha256"][path.name]:
        raise ValueError("Evidence packet changed")
    docs = {d["opportunity_id"]: d for d in json.loads(path.read_text())}
    inputs = {str(path): sha(path)}
    cases = []
    for recipe in CASES:
        d = docs[recipe["opportunity_id"]]
        p = next(p for p in d["pages"] if p["page"] == recipe["page"])
        render = ROOT / recipe["render_path"]
        if sha(d["source"]["file_path"]) != recipe["pdf_sha256"] or hashlib.sha256(p["text"].encode()).hexdigest() != recipe["text_sha256"] or sha(render) != recipe["render_sha256"]:
            raise ValueError("Reviewed source or render changed")
        inputs[d["source"]["file_path"]] = recipe["pdf_sha256"]
        inputs[str(render)] = recipe["render_sha256"]
        findings = [dict(f, page=recipe["page"], source_span=evidence_span(p["text"], f["source_text"])) for f in recipe["findings"]]
        cases.append({"opportunity_id": recipe["opportunity_id"], "source": d["source"],
                      "page": recipe["page"], "page_text_sha256": recipe["text_sha256"],
                      "render_path": str(render), "render_sha256": recipe["render_sha256"],
                      "findings": findings, "parser_limit": recipe["parser_limit"],
                      "ranking_eligible": False, "company_capacity_determined": False,
                      "legal_eligibility_determined": False, "historical_content_proven": False})
    payload = {"review_type": "post_inspection_development_source_review",
               "human_annotations": False, "expert_validation": False, "blind_reference": False,
               "f1_estimable": False, "parser_changed": False, "rankings_changed": False, "cases": cases}
    for p, expected in inputs.items():
        if sha(p) != expected:
            raise ValueError("Review input changed")
    output.mkdir(parents=True)
    (output / "review.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"input_sha256": inputs,
        "code_sha256": {str(Path(__file__)): sha(__file__)},
        "output_sha256": {"review.json": sha(output / "review.json")}}, indent=2) + "\n")
    return {"reviewed_development_cases": len(cases), "source_findings": sum(len(c["findings"]) for c in cases),
            "f1_estimable": False, "parser_changed": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "artifacts/experience_document_evidence_v5")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/experience_formulation_review_v1")
    a = p.parse_args()
    print(json.dumps(review(a.input, a.output)))
