"""Audita y codifica checkpoints de una recuperación anual que ya está en marcha.

No descarga ni abre etiquetas. Cada etapa conserva una copia estable del índice
de recuperación. Solo un proceso de este script puede escribir en el almacén.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from tesis_recommender.contrast_candidates import audit_candidate_batch, candidate_queue
from tesis_recommender.contrast_embeddings import encode_audited_batch
from tesis_recommender.validation import read_csv, sha256, write_csv


def stable_rows(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    before = path.read_bytes()
    if not before or not before.endswith(b"\n") or before != path.read_bytes():
        return None
    try:
        rows = list(csv.DictReader(io.StringIO(before.decode("utf-8-sig"))))
    except (UnicodeDecodeError, csv.Error):
        return None
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        return None
    return rows


def aggregate_checkpoint(prior: list[dict], incoming: list[dict], expected: dict) -> list[dict]:
    aggregate = {}
    for rows in (prior, incoming):
        if len({r.get("url") for r in rows}) != len(rows):
            raise ValueError("Un checkpoint tiene URLs repetidas")
        for row in rows:
            reference = expected.get(row.get("url"))
            if reference is None or any(row.get(key) != reference.get(key) for key in
                                        ("ocid", "role", "cache_key", "version_hint", "reference_date")):
                raise ValueError("El checkpoint no corresponde a la cola congelada")
            if row.get("status") not in {"ok", "failed"}:
                raise ValueError("Estado desconocido en checkpoint")
            if row["status"] == "ok" and (len(row.get("sha256", "")) != 64 or not row.get("cache_path")):
                raise ValueError("Fuente sin hash o ruta")
            previous = aggregate.get(row["url"])
            if previous and previous["status"] == "ok":
                if row["status"] != "ok" or row["sha256"] != previous["sha256"]:
                    raise ValueError("Cambió una fuente ya recuperada")
            aggregate[row["url"]] = row
    return [row for _, row in sorted(aggregate.items())]


def main(args):
    _, queue = candidate_queue(args.selection, args.preparation)
    expected = {row["url"]: row for row in queue}
    baseline_path = args.fetch / "candidate_version_results.csv"
    manifest = json.loads((args.fetch / "fetch_manifest.json").read_text())
    if sha256(baseline_path) != manifest["results_sha256"]:
        raise ValueError("La recuperación de entrada no coincide con su manifiesto")
    prior = read_csv(baseline_path)
    prior = aggregate_checkpoint(prior, [], expected)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "progress.json").exists():
        raise ValueError("Elegir otra carpeta para una continuación, sin sobrescribir etapas")
    archive = args.output / "code_snapshot"
    archive.mkdir()
    shutil.copyfile(Path(__file__), archive / Path(__file__).name)
    encoded_ok = sum(row["status"] == "ok" for row in prior)
    last_status = 0
    stage = 0
    while True:
        incoming = stable_rows(args.checkpoint) or []
        rows = aggregate_checkpoint(prior, incoming, expected)
        ok = sum(row["status"] == "ok" for row in rows)
        complete = ok == len(expected)
        if complete or ok - encoded_ok >= args.interval:
            stage += 1
            stage_dir = args.output / "stages" / f"stage_{stage:04d}"
            snapshot = stage_dir / "fetch_snapshot"
            snapshot.mkdir(parents=True)
            result_path = snapshot / "candidate_version_results.csv"
            write_csv(result_path, rows, list(prior[0]))
            snapshot_manifest = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_versions_planned": len(expected), "ok": ok,
                "failed": sum(row["status"] == "failed" for row in rows),
                "pending_versions": len(expected) - ok, "retrieval_complete": complete,
                "plan_sha256": sha256(args.preparation / "frozen_plan_with_code.json"),
                "results_sha256": sha256(result_path), "labels_read": False,
                "origin": "immutable_snapshot_of_original_recovery_and_inflight_checkpoint",
                "checkpoint_path": str(args.checkpoint),
            }
            (snapshot / "fetch_manifest.json").write_text(json.dumps(snapshot_manifest, indent=2))
            cohort = args.final_cohort if complete else stage_dir / "cohort"
            cache = args.final_cache if complete else stage_dir / "embeddings"
            report = audit_candidate_batch(args.selection, args.preparation, args.history, snapshot, cohort)
            status = {"phase": "encoding", "stage": stage, "recovered": ok,
                      "retained_candidates": report["retained_candidates"], "pending": len(expected) - ok,
                      "cohort": str(cohort), "cache": str(cache), "labels_read": False}
            (args.output / "progress.json").write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
            metadata = encode_audited_batch(cohort, args.preparation, args.store, cache,
                                           args.batch_size, args.amendment)
            encoded_ok = ok
            status.update({"phase": "complete" if complete else "waiting_for_next_checkpoint",
                           "profiles": metadata["profiles"], "opportunities": metadata["opportunities"],
                           "vector_count": metadata["vector_count"], "evaluation_ready": metadata["evaluation_ready"]})
            (args.output / "progress.json").write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
            gc.collect()
            if complete:
                return
        elif time.monotonic() - last_status >= 45:
            print(json.dumps({"phase": "waiting_for_next_checkpoint", "recovered": ok,
                              "pending": len(expected) - ok, "new_recovered_since_encoding": ok - encoded_ok}), flush=True)
            last_status = time.monotonic()
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("selection", "preparation", "history", "fetch", "checkpoint", "store", "amendment",
                 "output", "final-cohort", "final-cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--interval", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.interval <= 0 or args.batch_size <= 0:
        raise ValueError("Los tamaños de bloque deben ser positivos")
    main(args)
