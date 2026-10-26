"""Tablas por perfil y remuestreo pareado de métricas del curso."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from .validation import read_csv, write_csv, sha256


def query_metrics(ranking: list[str], positives: set[str]) -> dict[str, float]:
    if not positives:
        raise ValueError("Una consulta sin positivos conocidos no tiene Recall evaluable")
    result = {}
    for k in (5, 10):
        hits = len(set(ranking[:k]) & positives)
        precision, recall = hits / k, hits / len(positives)
        result[f"precision@{k}"] = precision
        result[f"recall@{k}"] = recall
        result[f"f1@{k}"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return result


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def paired_interval(differences: list[float], indices: list[list[int]]) -> dict:
    if not differences or not indices:
        raise ValueError("El remuestreo requiere consultas y réplicas")
    means = [sum(differences[i] for i in sample) / len(sample) for sample in indices]
    return {"mean": round(sum(differences) / len(differences), 6),
            "ci95_low": round(percentile(means, 0.025), 6),
            "ci95_high": round(percentile(means, 0.975), 6)}


def analyze_experiment(results_path: Path, labels_path: Path, awards_path: Path,
                       output_dir: Path, resamples: int = 5000, seed: int = 42) -> dict:
    if resamples < 100:
        raise ValueError("Usar al menos 100 réplicas de remuestreo")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    runs = payload["runs"]
    if not runs:
        raise ValueError("No hay variantes para comparar")
    truth: dict[str, set[str]] = {}
    for row in read_csv(labels_path):
        if float(row["label"]) > 0:
            truth.setdefault(row["supplier_id"], set()).add(row["opportunity_id"])
    profiles = set(next(iter(runs.values()))["top_k"])
    if any(set(run["top_k"]) != profiles for run in runs.values()):
        raise ValueError("Las variantes deben evaluar los mismos perfiles")
    if set(truth) - profiles:
        raise ValueError("Hay etiquetas para perfiles ausentes en los resultados")
    evaluable = sorted(p for p in profiles if truth.get(p))
    if not evaluable:
        raise ValueError("No hay consultas con positivos conocidos")
    histories = Counter(row["supplier_id"] for row in read_csv(awards_path))
    per_model = {}
    query_rows = []
    for name, run in runs.items():
        per_model[name] = {}
        for profile in sorted(profiles):
            ranking = [item["opportunity_id"] for item in run["top_k"][profile]][:10]
            if len(ranking) != len(set(ranking)):
                raise ValueError("Hay oportunidades duplicadas en el top-k")
            if len(ranking) < 10:
                raise ValueError("El análisis requiere conservar al menos diez posiciones")
            metrics = query_metrics(ranking, truth[profile]) if truth.get(profile) else {}
            if metrics:
                per_model[name][profile] = metrics
            query_rows.append({"model": name, "supplier_id": profile, "history_count": histories[profile],
                               "history_group": "2-3" if histories[profile] <= 3 else "4-7" if histories[profile] <= 7 else "8+",
                               "known_positives": len(truth.get(profile, set())),
                               "evaluation_status": "evaluable" if metrics else "no_known_positives",
                               **metrics, "top10": " | ".join(ranking)})
        for metric in query_metrics([], {"placeholder"}):
            mean = sum(per_model[name][p][metric] for p in evaluable) / len(evaluable)
            if abs(mean - run["metrics"][metric]) > 0.000001:
                raise ValueError(f"Las etiquetas no reproducen la métrica archivada: {name}/{metric}")
    generator = random.Random(seed)
    indices = [[generator.randrange(len(evaluable)) for _ in evaluable] for _ in range(resamples)]
    metrics = list(query_metrics([], {"placeholder"}))
    summary = []
    for name in runs:
        for metric in metrics:
            values = [per_model[name][p][metric] for p in evaluable]
            summary.append({"model": name, "metric": metric, **paired_interval(values, indices)})
    comparisons = []
    for left, right in [("labse", "bm25"), ("bm25_labse", "bm25"), ("hybrid_full", "bm25_labse")]:
        if left not in runs or right not in runs:
            continue
        for metric in metrics:
            differences = [per_model[left][p][metric] - per_model[right][p][metric] for p in evaluable]
            comparisons.append({"left": left, "right": right, "metric": metric,
                                **paired_interval(differences, indices)})
    error_rows = []
    if {"hybrid_full", "bm25_labse"} <= set(runs):
        for profile in evaluable:
            difference = per_model["hybrid_full"][profile]["recall@10"] - per_model["bm25_labse"][profile]["recall@10"]
            hybrid = {item["opportunity_id"] for item in runs["hybrid_full"]["top_k"][profile][:10]}
            base = {item["opportunity_id"] for item in runs["bm25_labse"]["top_k"][profile][:10]}
            error_rows.append({"supplier_id": profile, "known_positives": len(truth[profile]),
                               "history_count": histories[profile], "delta_recall10": round(difference, 6),
                               "gained_observed_positives": " | ".join(sorted((hybrid - base) & truth[profile])),
                               "lost_observed_positives": " | ".join(sorted((base - hybrid) & truth[profile])),
                               "human_interpretation": ""})
    error_rows.sort(key=lambda row: (row["delta_recall10"], row["supplier_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_profile_metrics.csv", query_rows,
              ["model", "supplier_id", "history_count", "history_group", "known_positives", "evaluation_status", *metrics, "top10"])
    write_csv(output_dir / "summary_intervals.csv", summary, ["model", "metric", "mean", "ci95_low", "ci95_high"])
    write_csv(output_dir / "paired_comparisons.csv", comparisons, ["left", "right", "metric", "mean", "ci95_low", "ci95_high"])
    write_csv(output_dir / "error_review.csv", error_rows,
              ["supplier_id", "known_positives", "history_count", "delta_recall10", "gained_observed_positives", "lost_observed_positives", "human_interpretation"])
    report = {"profiles": len(profiles), "profiles_evaluable": len(evaluable),
              "profiles_without_known_positives": len(profiles) - len(evaluable),
              "resamples": resamples, "seed": seed, "method": "paired_profile_bootstrap_percentile_95",
              "evaluation_status": payload.get("evaluation_status", "development_not_independent"),
              "interpretation_limit": "Exploratory intervals conditional on this inspected candidate pool and incomplete historical labels; not a confirmatory generalization test.",
              "inputs_sha256": {str(path): sha256(path) for path in (results_path, labels_path, awards_path)},
              "summary": summary, "comparisons": comparisons}
    (output_dir / "analysis_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
