"""Preserve two source-image decisions without interpreting extractor predictions."""
import argparse
import json
from pathlib import Path
from scripts.register_browser_experience_sources import dump, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    if output.exists():
        raise ValueError("Use a new observation directory")
    folder = ROOT / "artifacts/experience_holdout_image_text_v1"
    manifest = json.loads((folder / "manifest.json").read_text())
    inputs = {str(folder / p): h for p, h in manifest["output_sha256"].items()}
    inputs[str(folder / "manifest.json")] = sha(folder / "manifest.json")
    verify_hashes(inputs)
    docs = {d["opportunity_id"]: d for d in json.loads((folder / "pdf_text.json").read_text())["documents"]}
    decisions = [
        ("ocds-dgv273-seacev3-1104505", 62, "hospital_62.png", "no_legible_text_visible_only_scan_marks",
         "OCR returned empty text. Full-page image review found no legible textual content. This is not a negative label for the document."),
        ("ocds-dgv273-seacev3-1150662", 1, "nueva_requena_1.png", "first_pass_scope_consultoria_de_obra",
         "Cover explicitly identifies consultoria de obra for supervision. Preserve the selected case and require second source-only scope review before formal exclusion."),
    ]
    rows = []
    for ocid, number, filename, status, reason in decisions:
        doc = docs[ocid]
        page = next(p for p in doc["pages"] if p["page"] == number)
        image = ROOT / "tmp/pdfs/holdout_reference_v1" / filename
        inputs[str(image)] = sha(image)
        inputs[str(Path(doc["source"]["file_path"]).resolve())] = doc["source"]["sha256"]
        rows.append({"opportunity_id": ocid, "page": number, "observation": status, "reason": reason,
            "source_image_path": str(image), "source_image_sha256": sha(image),
            "source_pdf_sha256": doc["source"]["sha256"], "page_text_sha256": page["text_sha256"],
            "reviewer": "source_only_visual_review", "predictions_seen": False,
            "reference_complete": False, "second_source_only_review_complete": False,
            "negative_label_created": False, "ranking_eligible": False})
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "observations.json", rows)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
         "output_sha256": {"observations.json": sha(output / "observations.json")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve())
