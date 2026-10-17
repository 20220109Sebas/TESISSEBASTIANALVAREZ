"""Screen saved source pages and lock OCR on the three wholly image-only PDFs.

No extraction predictions or requirement detections are used in this selection.
"""
import argparse
import json
from pathlib import Path
from scripts.complete_mixed_pdf_text import raster_sizes, needs_ocr, dump, sha, verify_hashes, protected_outputs

ROOT = Path(__file__).resolve().parents[1]


def prepare(packet_dir, output):
    import pypdf
    if output.exists():
        raise ValueError("Use a new quality/OCR plan directory")
    manifest_path = packet_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = dict(manifest["input_sha256"])
    inputs.update({str(packet_dir / p): h for p, h in manifest["output_sha256"].items()})
    inputs[str(manifest_path)] = sha(manifest_path)
    inputs.update(protected_outputs())
    model_dir = ROOT / "artifacts/experience_holdout_models_lock_v1"
    model = json.loads((model_dir / "model_lock.json").read_text())
    inputs.update(model["original_code_sha256"])
    inputs[str(model_dir / "model_lock.json")] = sha(model_dir / "model_lock.json")
    binary = ROOT / "tmp/vision_ocr_build/vision-pdf-ocr"
    for path in (binary, ROOT / "scripts/vision_pdf_ocr.swift", ROOT / "scripts/complete_mixed_pdf_text.py", Path(__file__)):
        inputs[str(path)] = sha(path)
    verify_hashes(inputs)
    documents, quality = [], []
    for row in json.loads((packet_dir / "source_review_packet.json").read_text()):
        if row["source_status"] != "downloaded_unique_pdf":
            continue
        native_path = Path(row["source_text_path"])
        native = next(d for d in json.loads(native_path.read_text())["documents"] if d["opportunity_id"] == row["opportunity_id"])
        inputs[str(native_path)] = sha(native_path)
        source = dict(row["source"], file_path=str(Path(row["source"]["file_path"]).resolve()))
        inputs[source["file_path"]] = source["sha256"]
        verify_hashes({source["file_path"]: source["sha256"]})
        reader = pypdf.PdfReader(source["file_path"])
        if len(reader.pages) != len(native["pages"]) or len(reader.pages) > 120 or native.get("truncated_by_page_limit"):
            raise ValueError("Page count mismatch or exceeded limit")
        screen = []
        for page, saved in zip(reader.pages, native["pages"]):
            sizes = raster_sizes(page.get("/Resources"))
            screen.append({"page": saved["page"], "word_count": saved["word_count"],
                           "image_pixel_sizes": sizes, "large_raster_present": needs_ocr(sizes)})
        no_native = all(not p["text"].strip() for p in native["pages"])
        quality.append({"sample_number": row["sample_number"], "opportunity_id": row["opportunity_id"],
            "source": source, "total_pages": len(reader.pages), "page_screen": screen,
            "no_native_text": no_native,
            "large_raster_pages": [p["page"] for p in screen if p["large_raster_present"]],
            "full_text_recovery_proven": False, "prediction_generated": False})
        if no_native:
            documents.append({"opportunity_id": row["opportunity_id"], "source": source,
                "inspection_path": str(native_path), "total_pages": len(reader.pages),
                "selected_pages": [p["page"] for p in native["pages"]]})
    if len(quality) != 27 or len(documents) != 3:
        raise ValueError("The fixed source scope changed")
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "source_quality.json", quality)
    plan = {"purpose": "holdout_source_text_before_reference_and_predictions",
        "input_sha256": inputs, "documents": documents, "binary_path": str(binary),
        "pypdf_version": pypdf.__version__,
        "page_selection": "All physical pages of the three PDF with entirely empty native text. Mixed PDF screened separately and not declared complete.",
        "limits": {"documents": 3, "max_pages_per_document": 120, "workers": 1,
                   "timeout_per_page_seconds": 45, "stop_after_consecutive_errors": 2},
        "settings": {"dpi": 180, "vision_revision": 3, "recognition_level": "accurate",
                     "languages": ["es-ES", "en-US"], "language_correction": False},
        "predictions_generated": False, "external_ocr_service": False,
        "ranking_eligible": False, "historical_content_proven": False, "human_annotation": False}
    dump(output / "plan.json", plan)
    dump(output / "plan_lock.json", {"plan_sha256": sha(output / "plan.json")})
    summary = {"source_documents_screened": len(quality), "physical_pages": sum(r["total_pages"] for r in quality),
               "image_only_documents": len(documents), "ocr_pages_planned": sum(d["total_pages"] for d in documents),
               "mixed_documents_with_large_raster": sum(not r["no_native_text"] and bool(r["large_raster_pages"]) for r in quality),
               "predictions_generated": False, "reference_complete": False}
    dump(output / "quality_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.packet.resolve(), args.output.resolve())))
