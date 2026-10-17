"""Check bounded response prefixes for same-process non-PDF requirements.

This is an access diagnostic, not a download or extraction pipeline. Archives
are never opened and successful prefixes are not counted as recovered sources.
"""
import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.fetch_experience_documents import AllowedRedirect, sha
from scripts.fetch_pending_experience_sources import check_code
from tesis_recommender.substitute_sources import PRIORITY, timestamp
from tesis_recommender.annex_sources import document_role, safe_annex_url


def candidates(package, original):
    records = [r for r in package.get("records", []) if r.get("ocid") == original["opportunity_id"]]
    if len(records) != 1 or records[0].get("compiledRelease", {}).get("ocid") != original["opportunity_id"]:
        raise ValueError("Record does not uniquely identify the original process")
    result = []
    for doc in (records[0]["compiledRelease"].get("tender") or {}).get("documents") or []:
        role = document_role(doc.get("title"), doc.get("documentType"))
        fmt = str(doc.get("format", "")).casefold()
        published = timestamp(doc.get("datePublished"))
        modified = timestamp(doc.get("dateModified")) if doc.get("dateModified") else None
        if (role not in PRIORITY or fmt not in {"docx", "zip", "rar"} or
                doc.get("url") == original["url"] or not safe_annex_url(doc.get("url")) or
                not doc.get("id") or published is None or
                (doc.get("dateModified") and modified is None)):
            continue
        cutoff = timestamp(original["initial_version_published_at"])
        if cutoff is None:
            raise ValueError("Missing initial publication cutoff")
        result.append({"opportunity_id": original["opportunity_id"],
            "sample_number": original["sample_number"], "document_id": doc["id"],
            "title": doc.get("title"), "format": fmt, "document_role": role,
            "url": doc["url"], "date_published_declared": doc["datePublished"],
            "date_modified_declared": doc.get("dateModified"),
            "declared_after_or_modified_after_initial": published > cutoff or (modified is not None and modified > cutoff),
            "ranking_eligible": False, "historical_content_proven": False})
    return sorted(result, key=lambda r: (r["declared_after_or_modified_after_initial"],
        PRIORITY[r["document_role"]], timestamp(r["date_published_declared"]), str(r["document_id"]), r["url"]))


def probe(output):
    if output.exists():
        raise ValueError("Choose a new access diagnostic directory")
    folder = ROOT / "artifacts/experience_substitute_records_v1"
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / "record_access.json"
    if sha(path) != manifest["output_sha256"][path.name]:
        raise ValueError("Current record inventory changed")
    plan_path = ROOT / "artifacts/experience_substitute_plan_v1/pending_plan.json"
    plan = json.loads(plan_path.read_text())
    check_code(plan)
    pdf_processes = {r["opportunity_id"] for r in plan["pending_sources"]}
    inputs = {str(path): sha(path), str(folder / "manifest.json"): sha(folder / "manifest.json"), str(plan_path): sha(plan_path)}
    selected = []
    for access in json.loads(path.read_text()):
        if access["status"] != "record_saved" or access["opportunity_id"] in pdf_processes:
            continue
        source_path = Path(access["file_path"])
        if sha(source_path) != access["sha256"]:
            raise ValueError("Current record bytes changed")
        inputs[str(source_path)] = sha(source_path)
        available = candidates(json.loads(source_path.read_text()), access["original_document"])
        if available:
            selected.append(available[0])
    if len(selected) > 26 or len({r["url"] for r in selected}) != len(selected):
        raise ValueError("Unexpected substitute diagnostic scope")
    output.mkdir(parents=True)
    opener = urllib.request.build_opener(AllowedRedirect())
    rows = []
    for item in selected:
        row = dict(item, method="GET_prefix_only", prefix_limit_bytes=4096,
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            complete_document_recovered=False, rules_applied=False)
        try:
            req = urllib.request.Request(item["url"], headers={"User-Agent": "Thesis-document-audit/1.0", "Accept": "*/*"})
            with opener.open(req, timeout=20) as response:
                prefix = response.read(4096)
                row.update(status="response_prefix_available", http_status=response.status,
                    final_url=response.url, content_type=response.headers.get("Content-Type"),
                    prefix_bytes=len(prefix), prefix_sha256=hashlib.sha256(prefix).hexdigest())
        except urllib.error.HTTPError as exc:
            row.update(status="http_error", http_status=exc.code, error=str(exc))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            row.update(status="access_error", error=str(exc))
        row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        rows.append(row)
        (output / "access_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"sample": item["sample_number"], "format": item["format"], "status": row["status"], "HTTP": row.get("http_status")}), flush=True)
    check_code(plan)
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError("Input changed during diagnostic")
    summary = {"sources_probed": len(rows), "formats": dict(Counter(r["format"] for r in rows)),
        "http_statuses": dict(Counter(str(r.get("http_status")) for r in rows)),
        "complete_documents_recovered": 0, "rules_applied": False, "automatic_retries": False,
        "ranking_inputs_modified": False, "historical_content_proven": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"input_sha256": inputs,
        "code_sha256": {str(Path(__file__)): sha(__file__)},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.output.resolve())))
