"""Verifica rankings completos, explicaciones y métricas sin ajustar el modelo."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import resource
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

from tesis_recommender.annual_evaluation import MemmapEmbedder, classify_positive
from tesis_recommender.hybrid import HybridRanker
from tesis_recommender.io import read_awards, read_opportunities, read_relevance_labels
from tesis_recommender.profiles import build_profiles
from tesis_recommender.validation import read_csv, sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(args):
    import numpy as np

    require(not args.output.exists() and not args.output.with_suffix(".code.py").exists(), "No sobrescribir una verificación")
    manifest = json.loads((args.experiment / "manifest.json").read_text())
    require(manifest["status"] == "complete", "La ejecución no terminó")
    for path, digest in manifest["input_sha256"].items():
        require(sha256(Path(path)) == digest, "Una entrada cambió")
    for name, digest in manifest["source_sha256"].items():
        require(sha256(args.experiment / "code_snapshot" / name) == digest, "Código archivado modificado")
    require(sha256(args.experiment / "results.json") == manifest["results_sha256"], "Resultados modificados")
    require(sha256(args.experiment / "candidate_order.json") == manifest["candidate_order_sha256"], "Orden candidato modificado")
    payload = json.loads((args.experiment / "results.json").read_text())
    label_report = json.loads((args.labels / "label_validation_report.json").read_text())
    for name, digest in label_report["output_sha256"].items():
        require(sha256(args.labels / name) == digest, "Salida de auditoría modificada")
    reveal = json.loads((args.labels / "reveal_plan.json").read_text())
    require(sha256(args.labels / "reveal_plan.json") == label_report["reveal_plan_sha256"], "Plan de revelación modificado")
    for source in reveal["outcome_snapshot_sources"].values():
        require(sha256(Path(source["path"])) == source["sha256"], "Snapshot de desenlaces modificado")
    opportunities = read_opportunities(args.cohort / "test_opportunities.csv")
    order = json.loads((args.experiment / "candidate_order.json").read_text())
    require(order == [o.opportunity_id for o in opportunities], "Orden distinto al del corpus")
    candidate_set, positions = set(order), {ocid: i for i, ocid in enumerate(order)}
    require(len(candidate_set) == len(order), "Candidatos duplicados")
    by_id = {o.opportunity_id: o for o in opportunities}
    cutoff = date.fromisoformat(payload["cutoff"])
    profiles = build_profiles(read_awards(args.cohort / "historical_awards.csv"), cutoff)
    truth = read_relevance_labels(args.labels / "labels.csv")
    availability = {r["ocid"]: r for r in read_csv(args.cohort / "opportunity_availability.csv")}
    audit = read_csv(args.labels / "positive_pair_audit.csv")
    expected_pairs = set()
    for row in audit:
        status, temporal = classify_positive(row, set(profiles), availability, cutoff)
        require(status == row["audit_status"] and temporal == row["temporal_order"], "Clasificación temporal no reproducida")
        if status == "verified_positive":
            expected_pairs.add((row["supplier_id"], row["opportunity_id"]))
    require(expected_pairs == {(p, ocid) for p, ids in truth.items() for ocid in ids}, "Etiquetas distintas a los pares auditados")
    for name, run in payload["runs"].items():
        require(set(run["top_k"]) == set(profiles), "Perfiles diferentes por variante")
    require(set(payload["runs"]) == set(manifest["models"]), "Variantes diferentes al plan")
    totals = {model: {metric: 0.0 for metric in manifest["metrics"]} for model in manifest["models"]}
    rows_checked = explanations_checked = 0
    expected_rank_paths, expected_component_paths = set(), set()
    for supplier in sorted(profiles):
        component_relative = f"components/{supplier}.npz"
        expected_component_paths.add(component_relative)
        path = args.experiment / component_relative
        require(sha256(path) == manifest["component_sha256"][component_relative], "Señales modificadas")
        with np.load(path, allow_pickle=False) as loaded:
            components = {name: loaded[name].tolist() for name in loaded.files}
        require(set(components) == {"bm25", "semantic", "category", "region", "amount"}, "Señales inesperadas")
        require(all(len(values) == len(order) and all(math.isfinite(v) and 0 <= v <= 1 for v in values) for values in components.values()), "Señal inválida")
        for model, weights in manifest["models"].items():
            relative = f"rankings/{supplier}__{model}.csv.gz"
            expected_rank_paths.add(relative)
            path = args.experiment / relative
            require(sha256(path) == manifest["ranking_sha256"][relative], "Ranking modificado")
            seen, top, previous = set(), [], None
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                for position, row in enumerate(csv.DictReader(handle), 1):
                    require(row["supplier_id"] == supplier and row["model"] == model and int(row["rank"]) == position, "Identidad o posición de ranking inválida")
                    ocid, score = row["opportunity_id"], float(row["score"])
                    require(ocid in candidate_set and ocid not in seen and math.isfinite(score), "OCID repetido o puntaje no finito")
                    seen.add(ocid)
                    index = positions[ocid]
                    calculated = sum(weights.get(name, 0.0) * values[index] for name, values in components.items())
                    require(score == calculated, "Puntaje no reproduce exactamente sus componentes")
                    current = (-score, ocid)
                    require(previous is None or previous <= current, "Ranking fuera de orden")
                    previous = current
                    rows_checked += 1
                    if position <= 10:
                        top.append(ocid)
                        archived = payload["runs"][model]["top_k"][supplier][position - 1]
                        require(archived["opportunity_id"] == ocid and archived["score"] == round(score, 6) and archived["source_url"] == by_id[ocid].source_url, "Top-k no reproduce el ranking completo")
                        for name, values in components.items():
                            require(archived["explanation"][name + "_score"] == round(values[index], 6), "Explicación de señal incorrecta")
                            require(archived["explanation"][name + "_contribution"] == round(weights.get(name, 0.0) * values[index], 6), "Contribución explicada incorrecta")
                        explanations_checked += 1
            require(seen == candidate_set, "Variante con candidatos faltantes")
            if truth.get(supplier):
                for k in (5, 10):
                    hits = len(set(top[:k]) & truth[supplier])
                    p, r = hits / k, hits / len(truth[supplier])
                    f = 2 * p * r / (p + r) if p + r else 0.0
                    for metric, value in (("precision", p), ("recall", r), ("f1", f)):
                        totals[model][f"{metric}@{k}"] += value
    require(expected_rank_paths == set(manifest["ranking_sha256"]), "Archivos de rankings adicionales o faltantes")
    require(expected_component_paths == set(manifest["component_sha256"]), "Componentes adicionales o faltantes")
    metrics = {model: {m: round(v / len(truth), 6) if truth else None for m, v in totals[model].items()} for model in totals}
    for model in metrics:
        require(metrics[model] == payload["runs"][model]["metrics"], "Las métricas no se reproducen")
    # Replay determinista de componentes, sin elegir casos por sus resultados.
    sampled_profiles = sorted(profiles, key=lambda p: hashlib.sha256(f"42:{p}".encode()).hexdigest())[:3]
    ranker = HybridRanker(opportunities, MemmapEmbedder(args.cache))
    for supplier in sampled_profiles:
        actual = ranker.component_scores(profiles[supplier])
        with np.load(args.experiment / f"components/{supplier}.npz", allow_pickle=False) as saved:
            require(all(actual[name] == saved[name].tolist() for name in actual), "Replay de componentes diferente")
        ranker._component_cache.clear()
    require(rows_checked == len(profiles) * len(order) * len(manifest["models"]) == manifest["ranking_rows"], "Cantidad incorrecta de pares evaluados")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "verified": True,
        "profiles": len(profiles), "candidates_per_variant": len(order), "variants": len(manifest["models"]),
        "positive_pairs": len(expected_pairs), "profiles_with_known_positives": len(truth),
        "profiles_without_known_positives": len(profiles) - len(truth), "full_ranking_rows_verified": rows_checked,
        "top_k_explanations_verified": explanations_checked, "exact_weighted_scores_verified": True,
        "same_candidate_and_profile_sets_all_variants": True, "metrics_independently_recomputed": metrics,
        "component_replay_profiles_selected_by_fixed_hash_seed_42": sampled_profiles,
        "component_replay_exact": True, "weights_changed": False, "embeddings_regenerated": False,
        "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "manifest_sha256": sha256(args.experiment / "manifest.json"),
        "results_sha256": sha256(args.experiment / "results.json"),
        "label_audit_sha256": sha256(args.labels / "label_validation_report.json"),
        "verification_code_sha256": sha256(Path(__file__)),
        "evaluation_status": payload["evaluation_status"], "human_annotation_complete": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), args.output.with_suffix(".code.py"))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("experiment", "cohort", "cache", "labels", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(verify(parser.parse_args()), ensure_ascii=False, indent=2))
