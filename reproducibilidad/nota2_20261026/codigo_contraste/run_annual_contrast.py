"""Ejecuta los cuatro rankings congelados, un perfil a la vez y sin recodificar."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import resource
import shutil
import time
from datetime import date, datetime, timezone
from pathlib import Path

from tesis_recommender.annual_evaluation import MemmapEmbedder
from tesis_recommender.experiment_analysis import query_metrics
from tesis_recommender.hybrid import HybridRanker
from tesis_recommender.io import read_awards, read_opportunities, read_relevance_labels
from tesis_recommender.pipeline import ABLATIONS
from tesis_recommender.profiles import build_profiles, assert_no_temporal_leakage
from tesis_recommender.validation import sha256


def run(args):
    import numpy as np
    if args.output.exists():
        raise ValueError("No sobrescribir una ejecución de contraste")
    plan_path = args.preparation / "frozen_plan_with_code.json"
    plan = json.loads(plan_path.read_text())
    if ABLATIONS != plan["models"]:
        raise ValueError("Los pesos no coinciden con el plan congelado")
    source_root = Path(__file__).resolve().parents[1] / "src/tesis_recommender"
    ranking_modules = ["hybrid.py", "bm25.py", "schemas.py", "profiles.py", "text.py", "metrics.py", "pipeline.py", "io.py", "experiment_analysis.py"]
    for name in ranking_modules:
        if sha256(source_root / name) != plan["source_sha256"][name]:
            raise ValueError(f"Cambió un módulo congelado: {name}")
    metadata_path = args.cache / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    amendment_path = Path(metadata["embedding_amendment"]["path"])
    if sha256(amendment_path) != metadata["embedding_amendment"]["sha256"]:
        raise ValueError("Cambió la enmienda de embeddings")
    amendment = json.loads(amendment_path.read_text())
    if sha256(source_root / "semantic.py") != amendment["updated_source_sha256"]["semantic.py"] or amendment["ranking_weights_unchanged"] != plan["models"]:
        raise ValueError("Cambio semántico no autorizado por la enmienda")
    report_path = args.labels / "label_validation_report.json"
    report = json.loads(report_path.read_text())
    if not report["candidate_selection_unchanged"] or report["profile_replacements"]:
        raise ValueError("La auditoría modificó la selección")
    for name, digest in report["output_sha256"].items():
        if sha256(args.labels / name) != digest:
            raise ValueError("Cambió una salida de la auditoría de etiquetas")
    for name, digest in report["cohort_sha256"].items():
        if sha256(args.cohort / name) != digest:
            raise ValueError("Cambió una entrada anual")
    for name, digest in report["cache_sha256"].items():
        if sha256(args.cache / name) != digest:
            raise ValueError("Cambió el caché auditado")
    if report["input_sha256"][str(plan_path)] != sha256(plan_path):
        raise ValueError("Cambió el plan después de auditar etiquetas")
    cutoff = date.fromisoformat(plan["cutoff"])
    profiles = build_profiles(read_awards(args.cohort / "historical_awards.csv"), cutoff)
    assert_no_temporal_leakage(profiles)
    opportunities = read_opportunities(args.cohort / "test_opportunities.csv")
    candidate_ids = {o.opportunity_id for o in opportunities}
    if len(candidate_ids) != len(opportunities) or any(o.awarded_supplier_id or not o.text or o.object_type != "services" or not cutoff <= o.publication_date <= date.fromisoformat(plan["candidate_end"]) for o in opportunities):
        raise ValueError("El corpus cambió o incorpora desenlaces como entrada")
    relevant = read_relevance_labels(args.labels / "labels.csv")
    if set(relevant) - set(profiles) or any(ids - candidate_ids for ids in relevant.values()):
        raise ValueError("Hay etiquetas fuera de la población fijada")
    if len(profiles) != report["profiles"] or len(opportunities) != report["candidates"] or sum(map(len, relevant.values())) != report["verified_positive_pairs"]:
        raise ValueError("Las poblaciones no coinciden con la auditoría")
    for field in ("model_name", "model_revision", "profile_strategy", "opportunity_strategy"):
        if metadata[field] != plan[field]:
            raise ValueError("El modelo no coincide con el plan")
    started = time.monotonic()
    args.output.mkdir(parents=True)
    archive = args.output / "code_snapshot"
    archive.mkdir()
    sources = [Path(__file__), *(source_root / name for name in [*ranking_modules, "semantic.py", "annual_evaluation.py", "validation.py"])]
    for source in sources:
        shutil.copyfile(source, archive / source.name)
    input_paths = [plan_path, report_path, args.labels / "labels.csv", args.cohort / "historical_awards.csv",
                   args.cohort / "test_opportunities.csv", args.cohort / "candidate_validation_report.json",
                   metadata_path, args.cache / "index.json", args.cache / "embeddings.npy", args.cache / "verification.json", amendment_path]
    input_hashes = {str(p): sha256(p) for p in input_paths}
    manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "evaluation_status": plan["evaluation_status"], "input_sha256": input_hashes,
        "source_sha256": {p.name: sha256(p) for p in archive.iterdir()}, "models": plan["models"],
        "model_name": metadata["model_name"], "model_revision": metadata["model_revision"],
        "execution": "Local CPU ranking, lazy mmap vectors converted to Python floats per document. Frozen HybridRanker and BM25, deterministic score/OCID sort. One profile and one variant in memory. Clear component cache after each profile.",
        "embedding_model_loaded": False, "embeddings_regenerated": False, "ranking_code_changed": False,
        "weights_tuned": False, "profile_replacements": 0, "outcome_candidate_filter": False,
        "metrics": plan["metrics"], "labels_used_only_for_metrics": True}
    (args.output / "execution_plan.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    ranker = HybridRanker(opportunities, MemmapEmbedder(args.cache))
    runs = {name: {"metrics": {}, "top_k": {}} for name in plan["models"]}
    per_query_metrics = {name: [] for name in runs}
    ranking_hashes = {}
    rank_dir = args.output / "rankings"
    rank_dir.mkdir()
    component_dir = args.output / "components"
    component_dir.mkdir()
    (args.output / "candidate_order.json").write_text(json.dumps([o.opportunity_id for o in opportunities]))
    component_hashes = {}
    for number, (supplier, profile) in enumerate(sorted(profiles.items()), 1):
        ranker._component_cache.clear()
        components = ranker.component_scores(profile)
        component_path = component_dir / f"{supplier}.npz"
        np.savez_compressed(component_path, **{name: np.asarray(values, dtype=np.float64) for name, values in components.items()})
        component_hashes[str(component_path.relative_to(args.output))] = sha256(component_path)
        for name, weights in plan["models"].items():
            detailed = ranker.rank(profile, weights, top_k=None)
            if len(detailed) != len(opportunities) or {r.opportunity_id for r in detailed} != candidate_ids:
                raise ValueError("Una variante modificó la colección candidata")
            if any(not math.isfinite(r.score) for r in detailed):
                raise ValueError("Ranking con puntaje no finito")
            previous = None
            path = rank_dir / f"{supplier}__{name}.csv.gz"
            with path.open("xb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                    with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                        writer = csv.writer(text)
                        writer.writerow(["supplier_id", "model", "rank", "opportunity_id", "score"])
                        for position, item in enumerate(detailed, 1):
                            order = (-item.score, item.opportunity_id)
                            if previous is not None and order < previous:
                                raise ValueError("Ranking fuera de orden")
                            previous = order
                            writer.writerow([supplier, name, position, item.opportunity_id, repr(item.score)])
            ranking_hashes[str(path.relative_to(args.output))] = sha256(path)
            top = [{"opportunity_id": r.opportunity_id, "score": round(r.score, 6),
                    "source_url": r.source_url, "explanation": r.explanation} for r in detailed[:10]]
            runs[name]["top_k"][supplier] = top
            if relevant.get(supplier):
                per_query_metrics[name].append(query_metrics([r["opportunity_id"] for r in top], relevant[supplier]))
            del detailed
        ranker._component_cache.clear()
        del components
        checkpoint = {"profiles_completed": number, "profiles_planned": len(profiles),
            "candidates_per_ranking": len(opportunities), "variants_per_profile": len(runs),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        (args.output / "progress.json").write_text(json.dumps(checkpoint, indent=2))
        print(json.dumps(checkpoint), flush=True)
    for name in runs:
        values = per_query_metrics[name]
        runs[name]["metrics"] = {metric: round(sum(r[metric] for r in values) / len(values), 6) if values else None for metric in plan["metrics"]}
    result = {"model": metadata["model_name"], "cutoff": plan["cutoff"],
        "protocol_version": "frozen-annual-company-contrast-v1", "evaluation_status": plan["evaluation_status"],
        "evaluation_note": "Exploratory static annual retrieval with incomplete observed-positive labels. Company-disjoint, same inspected source/year, not blind temporal generalization.",
        "evaluation_population": {"profiles": len(profiles), "candidates": len(opportunities),
            "known_positive_pairs": sum(map(len, relevant.values())), "profiles_with_known_positives": len(relevant),
            "profiles_without_known_positives": len(profiles) - len(relevant)},
        "missing_attribute_policy": "Fixed weights; unavailable values contribute zero, not evidence of incompatibility; amount requires identical identified currency.",
        "runs": runs}
    (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    for p in input_paths:
        if sha256(p) != input_hashes[str(p)]:
            raise ValueError("Una entrada cambió durante la ejecución")
    manifest.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_population": result["evaluation_population"], "ranking_sha256": ranking_hashes,
        "component_sha256": component_hashes, "candidate_order_sha256": sha256(args.output / "candidate_order.json"),
        "ranking_files": len(ranking_hashes), "ranking_rows": len(profiles) * len(opportunities) * len(runs),
        "results_sha256": sha256(args.output / "results.json"), "inputs_unchanged": True,
        "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds": round(time.monotonic() - started, 3)})
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"status": "complete", "population": result["evaluation_population"],
            "metrics": {name: run["metrics"] for name, run in runs.items()},
            "peak_rss_native_units": manifest["peak_rss_native_units"], "elapsed_seconds": manifest["elapsed_seconds"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("preparation", "cohort", "cache", "labels", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
