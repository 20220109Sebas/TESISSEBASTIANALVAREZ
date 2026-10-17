"""Record HEAD availability of failed fixed sources without retrying PDF downloads."""
import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_experience_documents import AllowedRedirect, sha
from scripts.fetch_pending_experience_sources import check_code


def probe(plan_dir, output):
    if output.exists():
        raise ValueError("Choose a new probe directory")
    path = plan_dir / "pending_plan.json"
    m = json.loads((plan_dir / "manifest.json").read_text())
    if sha(path) != m["output_sha256"][path.name]:
        raise ValueError("Pending plan changed")
    plan = json.loads(path.read_text())
    check_code(plan)
    opener = urllib.request.build_opener(AllowedRedirect())
    rows = []
    output.mkdir(parents=True)
    for item in plan["pending_sources"]:
        row = {"opportunity_id": item["opportunity_id"], "sample_number": item["sample_number"],
               "url": item["url"], "method": "HEAD", "pdf_download_attempt": False,
               "started_at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            request = urllib.request.Request(item["url"], method="HEAD", headers={"User-Agent": "Thesis-document-audit/1.0", "Accept": "application/pdf"})
            with opener.open(request, timeout=15) as response:
                row.update(http_status=response.status, final_url=response.url,
                           content_type=response.headers.get("Content-Type"),
                           content_length=response.headers.get("Content-Length"))
        except urllib.error.HTTPError as exc:
            row.update(http_status=exc.code, error=str(exc), content_type=exc.headers.get("Content-Type"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            row.update(http_status=None, error=str(exc))
        row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        rows.append(row)
        (output / "head_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"sample": row["sample_number"], "HEAD": row["http_status"]}), flush=True)
    check_code(plan)
    summary = {"sources_probed": len(rows), "head_statuses": dict(Counter(str(r["http_status"]) for r in rows)),
               "pdf_bytes_downloaded": 0, "rules_unchanged": True,
               "head_does_not_prove_pdf_available_or_historical_content": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"input_sha256": {str(path): sha(path)},
        "code_sha256": {str(Path(__file__)): sha(__file__)},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(probe(a.plan.resolve(), a.output.resolve())))
