"""Build a self-contained, read-only view of the frozen automatic audit."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(audit, output):
    manifest = json.loads((audit / "manifest.json").read_text())
    for name, expected in manifest["output_sha256"].items():
        path = audit / name
        if not path.exists():
            path = ROOT / name
        if sha(path) != expected:
            raise ValueError(f"Audit output changed: {name}")
    packet = json.loads((audit / "review_packet.json").read_text())
    summary = json.loads((audit / "summary.json").read_text())
    if len(packet) != 50 or any(len(row["fields"]) != 5 for row in packet):
        raise ValueError("Expected the frozen 50 opportunities and five fields")
    payload = {"summary": summary, "opportunities": packet,
               "manifest": manifest, "audit_manifest_sha256": sha(audit / "manifest.json")}
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    template = (ROOT / "scripts/extraction_review.html").read_text()
    if template.count("__AUDIT_DATA__") != 1:
        raise ValueError("Invalid data placeholder")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__AUDIT_DATA__", encoded), encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha(output),
                      "opportunities": len(packet), "fields": 250}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=ROOT / "artifacts/extraction_commitment_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "output/extraction_review_v1/index.html")
    args = parser.parse_args()
    build(args.audit, args.output)
