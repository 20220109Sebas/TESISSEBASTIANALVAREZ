"""Continúa una cola existente con ocho conexiones HTTP reutilizables como máximo."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

from tesis_recommender.contrast_candidates import candidate_queue
from tesis_recommender.validation import load_release, read_csv, sha256, write_csv


def recover(args):
    _, frozen = candidate_queue(args.selection, args.preparation)
    expected = {row["url"]: row for row in frozen}
    queue = read_csv(args.queue)
    if len({row["url"] for row in queue}) != len(queue):
        raise ValueError("URLs repetidas")
    for row in queue:
        if row["url"] not in expected or any(row.get(key) != expected[row["url"]].get(key)
                for key in ("ocid", "role", "cache_key", "version_hint", "reference_date")):
            raise ValueError("La cola no corresponde al plan congelado")
    original_manifest = json.loads((args.fetch / "fetch_manifest.json").read_text())
    prior_path = args.fetch / "candidate_version_results.csv"
    if sha256(prior_path) != original_manifest["results_sha256"]:
        raise ValueError("El acumulado no coincide con su manifiesto")
    prior = read_csv(prior_path)
    known = {row["url"]: row for row in read_csv(args.results)} if args.results.exists() else {}
    args.audit.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), args.audit / Path(__file__).name)
    shutil.copyfile(args.queue, args.audit / "queue.csv")
    if args.results.exists():
        shutil.copyfile(args.results, args.audit / "initial_checkpoint.csv")
    fields = ["ocid", "role", "url", "version_hint", "reference_date", "cache_key", "status",
              "cache_path", "sha256", "release_published_at", "error"]
    args.cache.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with httpx.Client(headers={"User-Agent": "tesis-recommender/0.1"}, follow_redirects=True,
                      limits=httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers),
                      timeout=httpx.Timeout(45, connect=15)) as client:
        def fetch(row):
            path = args.cache / (row["cache_key"] + ".json")
            try:
                previous = known.get(row["url"])
                if previous and previous["status"] == "ok":
                    path = Path(previous["cache_path"])
                    if not path.exists() or sha256(path) != previous["sha256"]:
                        raise ValueError("Cambió una fuente del checkpoint")
                if not path.exists():
                    for attempt in range(3):
                        try:
                            response = client.get(row["url"])
                            response.raise_for_status()
                            temporary = path.with_suffix(".persistent.part")
                            temporary.write_bytes(response.content)
                            load_release(temporary, row["ocid"])
                            temporary.replace(path)
                            break
                        except (httpx.HTTPError, ValueError) as exc:
                            if attempt == 2:
                                raise exc
                            time.sleep(attempt + 1)
                release = load_release(path, row["ocid"])
                if release.get("id") != urlparse(row["url"]).path.rsplit("/", 1)[-1]:
                    raise ValueError("La respuesta corresponde a otra versión")
                return {**row, "status": "ok", "cache_path": str(path), "sha256": sha256(path),
                        "release_published_at": release.get("publishedDate", ""), "error": ""}
            except (httpx.HTTPError, ValueError, OSError) as error:
                return {**row, "status": "failed", "cache_path": str(path), "sha256": "",
                        "release_published_at": "", "error": str(error)}

        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(fetch, row) for row in queue]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 25 == 0:
                    write_csv(args.results, sorted(results, key=lambda row: row["url"]), fields)
                if len(results) % 500 == 0:
                    print(json.dumps({"queue_processed": len(results), "ok": sum(r["status"] == "ok" for r in results),
                                      "failed": sum(r["status"] != "ok" for r in results)}), flush=True)
        write_csv(args.results, sorted(results, key=lambda row: row["url"]), fields)
    aggregate = {row["url"]: row for row in prior}
    for row in results:
        if row["url"] in aggregate and aggregate[row["url"]]["status"] == "ok" and aggregate[row["url"]]["sha256"] != row["sha256"]:
            raise ValueError("Cambió una fuente ya conservada")
        aggregate[row["url"]] = row
    rows = [row for _, row in sorted(aggregate.items())]
    write_csv(prior_path, rows, fields)
    ok = sum(row["status"] == "ok" for row in rows)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "candidate_versions_planned": len(frozen),
              "versions_attempted": len(rows), "ok": ok, "failed": len(rows) - ok,
              "pending_versions": len(frozen) - ok, "retrieval_complete": ok == len(frozen),
              "results_sha256": sha256(prior_path), "plan_sha256": sha256(args.preparation / "frozen_plan_with_code.json"),
              "queue_sha256": sha256(args.preparation / "version_queue.csv"), "labels_read": False,
              "transport": "httpx_reused_connections", "httpx_version": httpx.__version__, "workers": args.workers,
              "transport_source_sha256": sha256(Path(__file__)), "checkpoint_sha256": sha256(args.results),
              "checkpoint": str(args.results), "elapsed_seconds": time.monotonic() - started}
    (args.audit / "manifest.json").write_text(json.dumps(report, indent=2))
    (args.fetch / "fetch_manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("selection", "preparation", "queue", "results", "cache", "fetch", "audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("Entre uno y ocho trabajadores")
    recover(args)
