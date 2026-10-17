"""Fetch only the five public annex URLs fixed by the access plan, with bounded size."""
import argparse
import hashlib
import json
import socket
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AllowedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname != "prod1.seace.gob.pe" or parsed.username or parsed.password:
            raise ValueError("Redirect outside the authorized public host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(plan_path, output):
    if output.exists():
        raise ValueError("Use a new fetch directory")
    plan = json.loads(plan_path.read_text())
    inventory = json.loads((plan_path.parent / "manifest.json").read_text())
    if sha(plan_path) != inventory["output_sha256"][plan_path.name]:
        raise ValueError("Access plan changed")
    for path, expected in plan["input_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Planned input changed")
    if len(plan["selected"]) > 5 or plan["limits"]["max_bytes_per_document"] > 10 * 1024 * 1024:
        raise ValueError("Pilot exceeds the bounded scope")
    output.mkdir(parents=True)
    opener = urllib.request.build_opener(AllowedRedirect())
    rows = []
    for number, item in enumerate(plan["selected"], 1):
        row = dict(item, retrieval_started_at_utc=datetime.now(timezone.utc).isoformat(),
                   historical_content_status="not_proven_bytes_retrieved_now", status="pending")
        url = item["url"]
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "prod1.seace.gob.pe" or parsed.username or parsed.password:
            raise ValueError("Unexpected document host")
        name = hashlib.sha256(url.encode()).hexdigest()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Thesis-document-audit/1.0", "Accept": "application/pdf"})
            with opener.open(request, timeout=plan["limits"]["timeout_seconds"]) as response:
                row.update(http_status=response.status, final_url=response.url,
                           content_type=response.headers.get("Content-Type", ""),
                           etag=response.headers.get("ETag"), last_modified=response.headers.get("Last-Modified"))
                prefix = response.read(1024)
                if b"%PDF-" not in prefix:
                    row.update(status="not_pdf", observed_prefix_sha256=hashlib.sha256(prefix).hexdigest())
                else:
                    path = output / (name + ".pdf")
                    total, oversized = 0, False
                    with path.open("wb") as handle:
                        block = prefix
                        while block:
                            if total + len(block) > plan["limits"]["max_bytes_per_document"]:
                                oversized = True
                                break
                            handle.write(block); total += len(block)
                            block = response.read(65536)
                    row.update(status="too_large_partial_file" if oversized else "downloaded_pdf",
                               file_path=str(path), bytes=total, sha256=sha(path))
        except urllib.error.HTTPError as exc:
            row.update(status="http_error", http_status=exc.code, error=str(exc))
        except (urllib.error.URLError, TimeoutError, socket.timeout, ssl.SSLError, ValueError) as exc:
            row.update(status="access_error", error=str(exc))
        row["retrieval_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        rows.append(row)
        print(json.dumps({"document": number, "status": row["status"], "http_status": row.get("http_status")}), flush=True)
        (output / "access_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    manifest = {"access_plan_path": str(plan_path), "access_plan_sha256": sha(plan_path),
                "fetch_script_sha256": sha(Path(__file__)), "documents_attempted": len(rows),
                "downloaded_pdfs": sum(r["status"] == "downloaded_pdf" for r in rows),
                "historical_content_proven": False, "labels_read": False, "ranking_inputs_modified": False,
                "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=ROOT / "artifacts/experience_source_inventory_v1/access_plan.json")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/experience_document_access_v1")
    a = p.parse_args()
    print(json.dumps(fetch(a.plan, a.output)))
