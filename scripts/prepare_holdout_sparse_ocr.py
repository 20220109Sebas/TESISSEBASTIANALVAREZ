"""Supplement source reading for low-resolution scans missed by pixel screening.

This is a disclosed source-quality amendment before model predictions, not a
change to extraction rules. Never select by extracted requirements or labels.
"""
import argparse
import json
from pathlib import Path
from scripts.complete_mixed_pdf_text import load_plan, dump, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]


def sparse_scan(page):
    return (not page["large_raster_present"] and page["word_count"] <= 40
            and any(w >= 300 and h >= 300 for w, h in page["image_pixel_sizes"]))


def prepare(quality_dir, packet_dir, output):
    if output.exists():
        raise ValueError("Use a new OCR plan directory")
    prior = load_plan(quality_dir)
    path = quality_dir / "source_quality.json"
    manifest = json.loads((quality_dir / "manifest.json").read_text())
    verify_hashes({str(path): manifest["output_sha256"][path.name]})
    packet_path = packet_dir / "source_review_packet.json"
    packet = {r["opportunity_id"]: r for r in json.loads(packet_path.read_text())}
    inputs = dict(prior["input_sha256"])
    inputs.update({str(path): sha(path), str(packet_path): sha(packet_path), str(Path(__file__)): sha(__file__)})
    documents = []
    for row in json.loads(path.read_text()):
        pages = [p["page"] for p in row["page_screen"] if sparse_scan(p)]
        if row["no_native_text"] or not pages:
            continue
        documents.append({"opportunity_id": row["opportunity_id"], "source": row["source"],
                          "inspection_path": packet[row["opportunity_id"]]["source_text_path"],
                          "total_pages": row["total_pages"], "selected_pages": pages})
    verify_hashes(inputs)
    output.mkdir(parents=True)
    plan = dict(prior, documents=documents, input_sha256=inputs,
                purpose="disclosed_source_quality_amendment_before_predictions",
                page_selection="Native <=40 words and an image >=300x300 pixels, excluding prior >=600x600 screen and wholly image-only PDFs. Fixed for all 27 source PDFs, no labels or predictions.",
                limits=dict(prior["limits"], documents=len(documents)))
    dump(output / "plan.json", plan)
    dump(output / "plan_lock.json", {"plan_sha256": sha(output / "plan.json")})
    return {"documents": len(documents), "selected_pages": sum(len(d["selected_pages"]) for d in documents)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("quality", "packet", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.quality.resolve(), args.packet.resolve(), args.output.resolve())))
