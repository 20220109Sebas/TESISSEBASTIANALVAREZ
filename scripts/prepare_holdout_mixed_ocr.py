"""Lock all large-raster pages in the thirteen mixed holdout PDF before reference."""
import argparse
import json
from pathlib import Path
from scripts.complete_mixed_pdf_text import load_plan, dump, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]


def prepare(quality_dir, packet_dir, output):
    if output.exists():
        raise ValueError("Use a new OCR directory")
    prior = load_plan(quality_dir)
    quality_path = quality_dir / "source_quality.json"
    m = json.loads((quality_dir / "manifest.json").read_text())
    verify_hashes({str(quality_path): m["output_sha256"][quality_path.name]})
    packet_path = packet_dir / "source_review_packet.json"
    packet = {r["opportunity_id"]: r for r in json.loads(packet_path.read_text())}
    inputs = dict(prior["input_sha256"])
    inputs.update({str(quality_path): sha(quality_path), str(Path(__file__)): sha(__file__),
                   str(packet_path): sha(packet_path)})
    documents = []
    for row in json.loads(quality_path.read_text()):
        if row["no_native_text"] or not row["large_raster_pages"]:
            continue
        documents.append({"opportunity_id": row["opportunity_id"], "source": row["source"],
                          "inspection_path": packet[row["opportunity_id"]]["source_text_path"],
                          "total_pages": row["total_pages"], "selected_pages": row["large_raster_pages"]})
    if len(documents) != 13:
        raise ValueError("Fixed mixed document scope changed")
    verify_hashes(inputs)
    plan = dict(prior, purpose="holdout_mixed_source_text_before_reference_and_predictions",
                input_sha256=inputs, documents=documents,
                page_selection="All pages with image XObjects at least 600 by 600 pixels in mixed PDF; locked by saved source metadata, no predictions read.",
                limits=dict(prior["limits"], documents=13))
    output.mkdir(parents=True)
    dump(output / "plan.json", plan)
    dump(output / "plan_lock.json", {"plan_sha256": sha(output / "plan.json")})
    return {"documents": len(documents), "selected_pages": sum(len(d["selected_pages"]) for d in documents)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("quality", "packet", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.quality.resolve(), args.packet.resolve(), args.output.resolve())))
