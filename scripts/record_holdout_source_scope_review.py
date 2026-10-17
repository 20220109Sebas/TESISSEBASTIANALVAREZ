"""Record source-only scope findings before extraction predictions.

These two findings are source review observations, not a reference for F1.
"""
import argparse
import json
from pathlib import Path
from scripts.register_browser_experience_sources import dump, sha, verify_hashes


CASES = [
    {"opportunity_id": "ocds-dgv273-seacev3-1176572", "page": 3,
     "quote": "El Consultor será elegido según el método de Consultores Individuales",
     "image": "tmp/pdfs/holdout_source_review_v1/ui_3.png",
     "scope_status": "individual_consultant_not_company_profile",
     "finding": "La contratación exige experiencia profesional individual. No se interpreta como experiencia facturada de una empresa.",
     "reference_status": "pending_second_source_only_scope_review"},
    {"opportunity_id": "ocds-dgv273-seacev3-2025-200276-396", "page": 5,
     "quote": "Evaluación: Se evaluará en función al tiempo de experiencia del personal registrado.",
     "image": "tmp/pdfs/holdout_source_review_v1/oefa_5.png",
     "scope_status": "generic_bases_requires_associated_terms_of_reference",
     "finding": "Las bases cubren personas naturales y jurídicas. El factor visible depende del personal registrado. No se asigna un umbral empresarial sin los términos de referencia asociados.",
     "reference_status": "abstain_pending_associated_terms_of_reference"},
]


def build(inspection, output):
    if output.exists():
        raise ValueError("Use a new review directory")
    manifest_path = inspection / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = {str(inspection / p): h for p, h in manifest["output_sha256"].items()}
    inputs[str(manifest_path)] = sha(manifest_path)
    verify_hashes(inputs)
    docs = {d["opportunity_id"]: d for d in json.loads((inspection / "pdf_text.json").read_text())["documents"]}
    findings = []
    for case in CASES:
        doc = docs[case["opportunity_id"]]
        page = next(p for p in doc["pages"] if p["page"] == case["page"])
        text, quote = page["text"], case["quote"]
        if text.count(quote) != 1:
            raise ValueError("Quote missing or ambiguous")
        start = text.index(quote)
        image = Path(case["image"]).resolve()
        inputs[str(image)] = sha(image)
        inputs[str(Path(doc["source"]["file_path"]).resolve())] = doc["source"]["sha256"]
        findings.append({**case, "source_pdf_sha256": doc["source"]["sha256"],
            "source_pdf_path": str(Path(doc["source"]["file_path"]).resolve()),
            "evidence": {"char_start": start, "char_end": start + len(quote),
                         "literal": quote, "page_text_sha256": page["text_sha256"]},
            "source_image_path": str(image), "source_image_sha256": sha(image),
            "reviewer": "source_only_automatic_review",
            "predictions_seen": False, "reference_conditions": None,
            "is_human_or_expert_reference": False, "second_source_only_review_complete": False,
            "absence_of_experience_proven": False, "ranking_eligible": False})
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "scope_findings.json", findings)
    dump(output / "summary.json", {"source_scope_first_pass": len(findings),
        "reference_conditions_complete": 0, "predictions_generated": False,
        "negative_labels_created": False, "f1_estimable": False})
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
         "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.inspection.resolve(), args.output.resolve())
