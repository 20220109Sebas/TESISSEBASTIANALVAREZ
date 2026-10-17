"""Merge native and OCR source pages without invoking any extraction model."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.register_browser_experience_sources import dump, sha, verify_hashes, protected_outputs

ROOT = Path(__file__).resolve().parents[1]


def merge_source_page(native, overlay=None):
    if hashlib.sha256(native["text"].encode()).hexdigest() != native["text_sha256"]:
        raise ValueError("Native page changed")
    if overlay is None:
        return dict(native, native_text=native["text"], native_text_sha256=native["text_sha256"],
                    extraction_method="native_pdf_text", ocr_selected=False, ocr_used=False,
                    selected_ocr_unresolved=False, absence_of_requirements_proven=False)
    if overlay["page"] != native["page"] or overlay["native_text"] != native["text"] or overlay["native_text_sha256"] != native["text_sha256"]:
        raise ValueError("OCR overlay lost native lineage")
    if not overlay["ocr_selected"] or hashlib.sha256(overlay["text"].encode()).hexdigest() != overlay["text_sha256"]:
        raise ValueError("Wrong overlay or text hash")
    return dict(overlay)


def build(packet_dir, overlay_dirs, output):
    if output.exists():
        raise ValueError("Use a new source-view directory")
    inputs = dict(protected_outputs())
    def checked(directory):
        path = directory / "manifest.json"
        m = json.loads(path.read_text())
        verify_hashes(m["input_sha256"])
        checked_outputs = {str(directory / p): h for p, h in m["output_sha256"].items()}
        verify_hashes(checked_outputs)
        inputs.update(checked_outputs)
        inputs[str(path)] = sha(path)
        return m
    checked(packet_dir)
    packet = json.loads((packet_dir / "source_review_packet.json").read_text())
    overlays = {}
    for directory in overlay_dirs:
        checked(directory)
        for doc in json.loads((directory / "pdf_text.json").read_text())["documents"]:
            for page in doc["pages"]:
                if not page["ocr_selected"]:
                    continue
                key = (doc["opportunity_id"], page["page"])
                if key in overlays:
                    raise ValueError("Duplicate OCR page across plans")
                overlays[key] = (doc["source"]["sha256"], page)
    documents = []
    for row in packet:
        if row["source_status"] != "downloaded_unique_pdf":
            continue
        path = Path(row["source_text_path"])
        source_manifest = json.loads((path.parent / "manifest.json").read_text())
        verify_hashes({str(path): source_manifest["output_sha256"][path.name]})
        inputs[str(path)] = sha(path)
        native_doc = next(d for d in json.loads(path.read_text())["documents"] if d["opportunity_id"] == row["opportunity_id"])
        inputs[str(Path(native_doc["source"]["file_path"]).resolve())] = native_doc["source"]["sha256"]
        pages = []
        for page in native_doc["pages"]:
            overlay = overlays.pop((row["opportunity_id"], page["page"]), None)
            if overlay and overlay[0] != native_doc["source"]["sha256"]:
                raise ValueError("Source PDF differs between OCR and native")
            pages.append(merge_source_page(page, overlay[1] if overlay else None))
        documents.append(dict(native_doc, pages=pages, sample_number=row["sample_number"],
                              all_document_content_proven_recovered=False, predictions_generated=False,
                              ranking_eligible=False, historical_content_proven=False))
    if overlays:
        raise ValueError("Unmatched OCR pages")
    inputs.update(json.loads((ROOT / "artifacts/experience_holdout_models_lock_v1/model_lock.json").read_text())["original_code_sha256"])
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "source_documents.json", documents)
    summary = {"documents": len(documents), "physical_pages": sum(len(d["pages"]) for d in documents),
        "ocr_pages_selected": sum(p["ocr_selected"] for d in documents for p in d["pages"]),
        "ocr_pages_with_text": sum(p["ocr_used"] for d in documents for p in d["pages"]),
        "selected_ocr_unresolved": sum(p["selected_ocr_unresolved"] for d in documents for p in d["pages"]),
        "native_text_preserved": True, "all_content_recovery_proven": False,
        "predictions_generated": False, "reference_complete": False, "ranking_inputs_modified": False}
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--overlay", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.packet.resolve(), [p.resolve() for p in args.overlay], args.output.resolve())))
