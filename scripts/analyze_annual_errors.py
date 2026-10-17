"""Diagnóstico posterior del contraste congelado, sin entrenar ni ajustar pesos."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

MODELS = ("bm25", "labse", "bm25_labse", "hybrid_full")
SIGNALS = ("bm25", "semantic", "category", "region", "amount")
CASE_PROFILE = "PE-RUC-20604252122"
CASE_OPPORTUNITY = "ocds-dgv273-seacev3-1150972"


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def check_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f"La fuente no coincide con su hash: {path}")


def metrics(top, positives, k):
    if not positives:
        raise ValueError("No calcular Recall sin positivos conocidos")
    hits = len(set(top[:k]) & positives)
    precision, recall = hits / k, hits / len(positives)
    return {"hits": hits, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def rank_components(components, weights, order):
    # Usar la misma suma de floats Python del ranking archivado. Una suma
    # vectorizada sucesiva puede diferir en el último bit en Python 3.12.
    scores = np.fromiter((sum(weights.get(signal, 0.0) * float(components[signal][i])
                              for signal in SIGNALS) for i in range(len(order))),
                         dtype=np.float64, count=len(order))
    if not np.isfinite(scores).all():
        raise ValueError("Puntajes no finitos")
    ranking = sorted(range(len(order)), key=lambda i: (-float(scores[i]), order[i]))
    return scores, ranking


def transition(left_top, right_top, positives, k):
    left, right = set(left_top[:k]) & positives, set(right_top[:k]) & positives
    return {"gained": sorted(left - right), "lost": sorted(right - left),
            "left_hits": len(left), "right_hits": len(right)}


def diagnose(args):
    if args.output.exists():
        raise ValueError("Usar un directorio nuevo para el diagnóstico")
    manifest_path = args.experiment / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete" or set(manifest["models"]) != set(MODELS):
        raise ValueError("Se requiere la ejecución terminada de las cuatro variantes")
    inputs = {str(manifest_path): digest(manifest_path)}

    def checked(path, expected):
        check_hash(path, expected)
        inputs[str(path)] = expected

    results_path = args.experiment / "results.json"
    checked(results_path, manifest["results_sha256"])
    for path in (args.cohort / "test_opportunities.csv", args.cohort / "historical_awards.csv", args.labels):
        expected_hash = manifest["input_sha256"].get(str(path))
        if not expected_hash:
            raise ValueError(f"Entrada no registrada en el manifiesto: {path}")
        checked(path, expected_hash)
    order_path = args.experiment / "candidate_order.json"
    checked(order_path, manifest["candidate_order_sha256"])
    order = read_json(order_path)
    if len(order) != len(set(order)):
        raise ValueError("Candidatos duplicados")
    candidate_index = {ocid: i for i, ocid in enumerate(order)}
    results = read_json(results_path)
    profiles = sorted(results["runs"]["bm25"]["top_k"])
    if any(sorted(results["runs"][m]["top_k"]) != profiles for m in MODELS):
        raise ValueError("Poblaciones diferentes entre modelos")

    positives, label_rows = {}, {}
    for row in read_rows(args.labels):
        if float(row["label"]) <= 0:
            continue
        sid, ocid = row["supplier_id"], row["opportunity_id"]
        if sid not in profiles or ocid not in candidate_index:
            raise ValueError("Etiqueta fuera de las entradas fijadas")
        if (sid, ocid) in label_rows:
            raise ValueError("Par positivo duplicado")
        positives.setdefault(sid, set()).add(ocid)
        label_rows[sid, ocid] = row
    evaluable = sorted(positives)
    if not evaluable:
        raise ValueError("No hay positivos observados")
    wanted = set().union(*positives.values())
    for sid in evaluable:
        for model in MODELS:
            wanted.update(x["opportunity_id"] for x in results["runs"][model]["top_k"][sid])
    opportunity_rows, coverage, currencies = {}, Counter(), Counter()
    observed_ids = set()
    for row in read_rows(args.cohort / "test_opportunities.csv"):
        ocid = row["opportunity_id"]
        if ocid in observed_ids:
            raise ValueError("OCID duplicado en el corpus")
        observed_ids.add(ocid)
        if ocid in wanted:
            opportunity_rows[ocid] = row
        amount = row.get("amount", "")
        value = float(amount) if amount else None
        amount_state = "amount_missing" if value is None else "amount_nonfinite" if not math.isfinite(value) else "amount_zero" if value == 0 else "amount_negative" if value < 0 else "amount_positive"
        coverage[amount_state] += 1
        coverage["positive_amount_identified_currency"] += bool(value is not None and math.isfinite(value) and value > 0 and row.get("currency", "").strip())
        coverage["region_nonempty"] += bool(row.get("region", "").strip())
        coverage["generic_or_missing_category"] += row.get("category", "").strip().casefold() in {"", "services", "consultingservices"}
        currencies[row.get("currency", "") or "missing"] += 1
    if observed_ids != set(order):
        raise ValueError("El orden y el corpus tienen candidatos distintos")
    history = {}
    for row in read_rows(args.cohort / "historical_awards.csv"):
        history.setdefault(row["supplier_id"], []).append(row)

    pair_details, per_profile, diagnostics = [], [], []
    transition_cases = {f"{left}_vs_{right}@{k}": [] for left, right in (
        ("bm25_labse", "bm25"), ("bm25_labse", "labse"), ("hybrid_full", "bm25_labse")) for k in (5, 10)}
    case = None
    validated_rows = 0
    for sid in evaluable:
        component_path = args.experiment / "components" / f"{sid}.npz"
        checked(component_path, manifest["component_sha256"][str(component_path.relative_to(args.experiment))])
        with np.load(component_path, allow_pickle=False) as archive:
            components = {key: archive[key] for key in SIGNALS}
        if any(a.shape != (len(order),) or not np.isfinite(a).all() for a in components.values()):
            raise ValueError("Componentes inválidos")
        ranks, scores, tops = {}, {}, {}
        for model in MODELS:
            path = args.experiment / "rankings" / f"{sid}__{model}.csv.gz"
            checked(path, manifest["ranking_sha256"][str(path.relative_to(args.experiment))])
            vector, ranking = rank_components(components, manifest["models"][model], order)
            ranks[model] = {order[i]: pos for pos, i in enumerate(ranking, 1) if order[i] in positives[sid]}
            scores[model] = {ocid: float(vector[candidate_index[ocid]]) for ocid in positives[sid]}
            tops[model] = [order[i] for i in ranking[:10]]
            if tops[model] != [r["opportunity_id"] for r in results["runs"][model]["top_k"][sid]]:
                raise ValueError("El top 10 no reproduce la salida congelada")
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                count = 0
                for pos, row in enumerate(csv.DictReader(handle), 1):
                    idx = ranking[pos - 1]
                    if row["supplier_id"] != sid or row["model"] != model or int(row["rank"]) != pos or row["opportunity_id"] != order[idx] or float(row["score"]) != float(vector[idx]):
                        raise ValueError(f"El ranking no reproduce señales y pesos: {path}/{pos}")
                    count += 1
            if count != len(order):
                raise ValueError("Ranking incompleto")
            validated_rows += count
            per_profile.append({"supplier_id": sid, "model": model, "history_count": len(history.get(sid, [])), "known_positives": len(positives[sid]),
                                "metrics": {str(k): metrics(tops[model], positives[sid], k) for k in (5, 10)}})
        for ocid in sorted(positives[sid]):
            pair_details.append({"supplier_id": sid, "opportunity_id": ocid, "label": label_rows[sid, ocid],
                                 "ranks": {m: ranks[m][ocid] for m in MODELS}, "scores": {m: scores[m][ocid] for m in MODELS},
                                 "signals": {s: float(components[s][candidate_index[ocid]]) for s in SIGNALS}, "opportunity": opportunity_rows[ocid]})
        for left, right in (("bm25_labse", "bm25"), ("bm25_labse", "labse"), ("hybrid_full", "bm25_labse")):
            for k in (5, 10):
                transition_cases[f"{left}_vs_{right}@{k}"].append({"supplier_id": sid, **transition(tops[left], tops[right], positives[sid], k)})

        full = manifest["models"]["hybrid_full"]
        variants = {"frozen_full": dict(full), **{f"without_{s}": {**full, s: 0.0} for s in ("category", "region", "amount")},
                    "without_attributes": {**full, "category": 0.0, "region": 0.0, "amount": 0.0}}
        case_ablations = {}
        for name, weights in variants.items():
            vector, ranking = rank_components(components, weights, order)
            diagnostic_ranks = {order[i]: pos for pos, i in enumerate(ranking, 1) if order[i] in positives[sid]}
            diagnostics.append({"supplier_id": sid, "diagnostic": name, "weights": weights,
                                "metrics": {str(k): metrics([order[i] for i in ranking[:10]], positives[sid], k) for k in (5, 10)},
                                "positive_ranks": diagnostic_ranks})
            if sid == CASE_PROFILE:
                case_ablations[name] = diagnostic_ranks.get(CASE_OPPORTUNITY)
        if sid == CASE_PROFILE:
            case_ids = set(tops["bm25_labse"]) | set(tops["hybrid_full"]) | {CASE_OPPORTUNITY}
            evidence = []
            for ocid in sorted(case_ids):
                idx = candidate_index[ocid]
                signal_values = {s: float(components[s][idx]) for s in SIGNALS}
                contributions = {s: full.get(s, 0.0) * signal_values[s] for s in SIGNALS}
                evidence.append({"opportunity_id": ocid, "observed_positive": ocid in positives[sid],
                                 "opportunity": opportunity_rows[ocid], "signals": signal_values, "full_contributions": contributions,
                                 "full_text_contribution": contributions["bm25"] + contributions["semantic"],
                                 "full_attribute_contribution": sum(contributions[s] for s in ("category", "region", "amount")),
                                 "lexical_semantic_rank": (tops["bm25_labse"].index(ocid) + 1) if ocid in tops["bm25_labse"] else "outside_top10",
                                 "full_rank": (tops["hybrid_full"].index(ocid) + 1) if ocid in tops["hybrid_full"] else ranks["hybrid_full"].get(ocid, "outside_top10")})
            case = {"supplier_id": sid, "target": CASE_OPPORTUNITY, "history": history[sid], "target_ranks": {m: ranks[m][CASE_OPPORTUNITY] for m in MODELS},
                    "target_diagnostic_ranks": case_ablations, "evidence": evidence,
                    "selection": "posthoc_case_requested_after_observed_top10_loss_not_representative_or_confirmatory"}

    model_summary = {}
    for model in MODELS:
        rows = [r for r in per_profile if r["model"] == model]
        model_summary[model] = {}
        for k in (5, 10):
            calculated = {metric: sum(r["metrics"][str(k)][metric] for r in rows) / len(rows) for metric in ("precision", "recall", "f1")}
            for metric, value in calculated.items():
                if abs(value - results["runs"][model]["metrics"][f"{metric}@{k}"]) > 0.000001:
                    raise ValueError("No se reproduce una métrica archivada")
            model_summary[model][str(k)] = {"macro": calculated, "observed_positive_pairs_recovered": sum(r["metrics"][str(k)]["hits"] for r in rows)}
    transition_summary = {key: {"gained_pairs": sum(len(r["gained"]) for r in rows), "lost_pairs": sum(len(r["lost"]) for r in rows),
                               "profiles_with_more_hits": sum(r["left_hits"] > r["right_hits"] for r in rows),
                               "profiles_with_fewer_hits": sum(r["left_hits"] < r["right_hits"] for r in rows),
                               "profiles_with_same_hits": sum(r["left_hits"] == r["right_hits"] for r in rows)} for key, rows in transition_cases.items()}
    complementary = [{"supplier_id": r["supplier_id"], "opportunity_id": r["opportunity_id"], "ranks": r["ranks"]} for r in pair_details
                     if r["ranks"]["bm25_labse"] <= 10 < r["ranks"]["bm25"] and r["ranks"]["labse"] > 10]
    summary = {"profiles": len(profiles), "profiles_with_known_positives": len(evaluable), "profiles_without_known_positives": len(profiles) - len(evaluable),
               "known_positive_pairs": len(pair_details), "candidates": len(order), "revalidated_ranking_rows": validated_rows,
               "model_summary": model_summary, "transitions": transition_summary, "complementary_top10_cases": complementary,
               "corpus_field_coverage": dict(coverage), "currencies": dict(currencies),
               "interpretation": "Posthoc mechanical diagnosis; unlabelled pairs are not confirmed negatives. Diagnostic removals are not validated replacement models or weight tuning."}
    for path, expected_hash in inputs.items():
        check_hash(path, expected_hash)
    args.output.mkdir(parents=True)
    outputs = {"summary.json": summary, "positive_rankings.json": pair_details, "per_profile.json": per_profile,
               "transitions.json": transition_cases, "component_diagnostics.json": diagnostics, "case_evidence.json": case}
    for name, payload in outputs.items():
        (args.output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diagnostic_manifest = {"analysis_type": "posthoc_mechanical_error_analysis_not_confirmatory", "models_modified": False, "weights_tuned": False,
                           "human_judgments_created": False, "embeddings_regenerated": False, "inputs_unchanged": True,
                           "inputs_sha256": inputs, "script_sha256": digest(Path(__file__)),
                           "outputs_sha256": {name: digest(args.output / name) for name in outputs}}
    (args.output / "manifest.json").write_text(json.dumps(diagnostic_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("artifacts/experiment_annual_v1"))
    parser.add_argument("--cohort", type=Path, default=Path("data/processed/contrast_candidates_v1/annual_v1"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/annual_positive_audit_v2/labels.csv"))
    parser.add_argument("--output", type=Path, required=True)
    diagnose(parser.parse_args())
