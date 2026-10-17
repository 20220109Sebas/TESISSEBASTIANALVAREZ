"""Plan and run native local OCR only for the original pilot's all-empty PDF."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, binary):
    if output.exists():
        raise ValueError("Use a new OCR directory")
    inspection = ROOT / "artifacts/experience_pdf_inspection_v1"
    m = json.loads((inspection / "manifest.json").read_text())
    text = inspection / "pdf_text.json"
    if sha(text) != m["output_sha256"][text.name]:
        raise ValueError("Pilot page text changed")
    docs = json.loads(text.read_text())["documents"]
    selected = []
    for doc in docs:
        if doc["status"] == "text_extracted" and doc["pages"] and all(not p["text"].strip() for p in doc["pages"]):
            if sha(doc["source"]["file_path"]) != doc["source"]["sha256"]:
                raise ValueError("PDF changed")
            selected.append({"opportunity_id": doc["opportunity_id"], "source": doc["source"],
                             "pages": [p["page"] for p in doc["pages"]]})
    if len(selected) != 1 or sum(len(d["pages"]) for d in selected) > 80:
        raise ValueError("Directed OCR must stay bounded to one all-empty pilot PDF, at most 80 pages")
    output.mkdir(parents=True)
    plan = {"purpose": "local_directed_ocr_not_historical_pdf_proof_or_field_evaluation",
            "inspection_path": str(text), "inspection_sha256": sha(text),
            "binary_path": str(binary.resolve()), "binary_sha256": sha(binary),
            "swift_source_sha256": sha(ROOT / "scripts/vision_pdf_ocr.swift"),
            "runner_sha256": sha(Path(__file__)), "selected": selected,
            "limits": {"pages": 80, "workers": 1, "timeout_per_page_seconds": 45},
            "settings": {"dpi": 180, "vision_revision": 3, "recognition_level": "accurate",
                         "languages": ["es-ES", "en-US"], "language_correction": False},
            "external_ocr_service": False, "historical_content_proven": False,
            "ranking_eligible": False, "semantic_f1_evaluated": False}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "plan_lock.json").write_text(json.dumps({"plan_sha256": sha(output / "plan.json")}) + "\n")
    return {"prepared": True, "planned_pages": sum(len(d["pages"]) for d in selected)}


def run(output):
    plan_path = output / "plan.json"
    lock = json.loads((output / "plan_lock.json").read_text())
    if sha(plan_path) != lock["plan_sha256"]:
        raise ValueError("OCR plan changed")
    p = json.loads(plan_path.read_text())
    for path, expected in ((p["binary_path"], p["binary_sha256"]),
                           (ROOT / "scripts/vision_pdf_ocr.swift", p["swift_source_sha256"]),
                           (Path(__file__), p["runner_sha256"]),
                           (p["inspection_path"], p["inspection_sha256"])):
        if sha(path) != expected:
            raise ValueError("OCR code or input changed")
    pages_dir = output / "pages"
    if pages_dir.exists():
        raise ValueError("OCR already started; use a new plan to repeat")
    pages_dir.mkdir()
    started = time.monotonic()
    rows, consecutive_errors = [], 0
    for doc in p["selected"]:
        if sha(doc["source"]["file_path"]) != doc["source"]["sha256"]:
            raise ValueError("PDF changed")
        for number in doc["pages"]:
            path = pages_dir / f"page_{number:03d}.json"
            row = {"opportunity_id": doc["opportunity_id"], "page": number,
                   "source_pdf_sha256": doc["source"]["sha256"], "method": "apple_vision_local_ocr"}
            try:
                subprocess.run([p["binary_path"], str(Path(doc["source"]["file_path"]).resolve()), str(number), str(path.resolve())],
                               check=True, capture_output=True, text=True, timeout=p["limits"]["timeout_per_page_seconds"])
                result = json.loads(path.read_text())
                if result["page"] != number or result["revision"] != 3 or result["languageCorrection"]:
                    raise ValueError("OCR settings differ from plan")
                row.update(status="ok", text=result["text"], text_sha256=hashlib.sha256(result["text"].encode()).hexdigest(),
                           ocr_json_path=str(path), ocr_json_sha256=sha(path), line_count=len(result["lines"]),
                           low_confidence_lines=sum(line["confidence"] < .8 for line in result["lines"]),
                           engine=result["engine"], languages=result["languages"], os_version=result["osVersion"],
                           raster_pixels=[result["pixelWidth"], result["pixelHeight"]])
                consecutive_errors = 0
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
                row.update(status="ocr_error", error=str(exc), stderr=getattr(exc, "stderr", None))
                consecutive_errors += 1
            rows.append(row)
            (output / "ocr_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
            if number == 1 or number % 10 == 0 or number == doc["pages"][-1]:
                print(json.dumps({"pages_attempted": len(rows), "pages_ok": sum(r["status"] == "ok" for r in rows),
                                  "elapsed_seconds": round(time.monotonic()-started, 2)}), flush=True)
            if consecutive_errors >= 2:
                break
    count = sum(len(d["pages"]) for d in p["selected"])
    summary = {"planned_pages": count, "pages_attempted": len(rows), "pages_ok": sum(r["status"] == "ok" for r in rows),
               "pages_with_experience_word": [r["page"] for r in rows if r["status"] == "ok" and "experiencia" in r["text"].casefold()],
               "complete": len(rows) == count and all(r["status"] == "ok" for r in rows),
               "low_confidence_lines": sum(r.get("low_confidence_lines", 0) for r in rows),
               "elapsed_seconds": round(time.monotonic()-started, 3),
               "confidence_is_not_semantic_accuracy": True, "historical_content_proven": False,
               "ranking_eligible": False, "labels_read": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {"plan_sha256": sha(plan_path), "runner_sha256": sha(Path(__file__)),
                "swift_source_sha256": p["swift_source_sha256"], "binary_sha256": p["binary_sha256"],
                "output_sha256": {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*")) if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experience_targeted_ocr_v1")
    parser.add_argument("--binary", type=Path, default=ROOT / "tmp/vision_ocr_build/vision-pdf-ocr")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output) if args.run else prepare(args.output, args.binary)))
