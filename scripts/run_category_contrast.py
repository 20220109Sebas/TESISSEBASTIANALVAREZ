"""Congela y ejecuta una comparación de desarrollo cambiando solo categoría."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import random
import resource
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from tesis_recommender.category_ranking import CategoryCodeSignal, SIGNALS, VARIANTS, rank_components
from tesis_recommender.experiment_analysis import paired_interval, query_metrics
from tesis_recommender.validation import sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def checked(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Archivo modificado: {path}")


def unique_index(records, field):
    index = {}
    for record in records:
        identity = record.get(field)
        if not identity or identity in index:
            raise ValueError(f"Identidad ausente o duplicada: {field}")
        index[identity] = record
    return index


def load_features(plan):
    cohort, categories = Path(plan["cohort"]), Path(plan["categories"])
    order = read_json(Path(plan["original"]) / "candidate_order.json")
    csv_opportunities = unique_index(read_rows(cohort / "test_opportunities.csv"), "opportunity_id")
    csv_history = unique_index(read_rows(cohort / "historical_awards.csv"), "award_id")
    with (categories / "opportunity_categories.jsonl").open(encoding="utf-8") as handle:
        opportunities = unique_index([json.loads(line) for line in handle], "ocid")
    with (categories / "history_categories.jsonl").open(encoding="utf-8") as handle:
        history = unique_index([json.loads(line) for line in handle], "award_id")
    if set(order) != set(opportunities) or set(order) != set(csv_opportunities) or len(order) != len(set(order)):
        raise ValueError("Las oportunidades auxiliares no corresponden a la población congelada")
    if set(history) != set(csv_history):
        raise ValueError("Los antecedentes auxiliares no corresponden a la población congelada")
    cutoff = datetime.fromisoformat(plan["cutoff"]).date()
    end = datetime.fromisoformat(plan["candidate_end"]).date()
    for ocid, record in opportunities.items():
        if record["legacy_category"] != csv_opportunities[ocid]["category"]:
            raise ValueError("Cambió la categoría original de una oportunidad")
        published = datetime.fromisoformat(record["release_published_at"].replace("Z", "+00:00"))
        if published.tzinfo is None or not cutoff <= published.astimezone(ZoneInfo("America/Lima")).date() <= end:
            raise ValueError("Categoría de oportunidad publicada fuera de la ventana")
    for aid, record in history.items():
        reference = csv_history[aid]
        if (record["supplier_id"] != reference["supplier_id"] or record["legacy_category"] != reference["category"]
                or record["ocid"] != aid.split(":", 1)[0]):
            raise ValueError("Antecedente vinculado a otro proveedor, proceso o categoría")
        published = datetime.fromisoformat(record["release_published_at"].replace("Z", "+00:00"))
        if published.tzinfo is None or published.astimezone(ZoneInfo("America/Lima")).date() >= cutoff:
            raise ValueError("Categoría histórica no disponible antes del corte")
    return order, [opportunities[o] for o in order], list(history.values()), csv_opportunities


def prepare(args):
    if args.output.exists():
        raise ValueError("Usar una carpeta nueva para conservar el contraste original")
    original_manifest = args.original / "manifest.json"
    original = read_json(original_manifest)
    if original.get("status") != "complete":
        raise ValueError("El contraste original no está terminado")
    category_manifest = args.categories / "manifest.json"
    category = read_json(category_manifest)
    if category["labels_read"] or category["rankings_run"]:
        raise ValueError("La representación auxiliar debe preceder a esta comparación")
    paths = {str(original_manifest): sha256(original_manifest), str(category_manifest): sha256(category_manifest),
             str(args.original / "results.json"): original["results_sha256"],
             str(args.original / "candidate_order.json"): original["candidate_order_sha256"]}
    original_plan_path = Path(next(p for p in original["input_sha256"] if p.endswith("frozen_plan_with_code.json")))
    paths[str(original_plan_path)] = original["input_sha256"][str(original_plan_path)]
    for name in ("test_opportunities.csv", "historical_awards.csv"):
        path = args.cohort / name
        expected = original["input_sha256"][str(path)]
        if category["input_sha256"].get(str(path)) != expected:
            raise ValueError("La revisión de categorías corresponde a otros datos")
        paths[str(path)] = expected
    if original["input_sha256"].get(str(args.labels)) is None:
        raise ValueError("Etiquetas no registradas en el contraste original")
    # Solo se calcula el hash. No se leen etiquetas durante la preparación.
    paths[str(args.labels)] = original["input_sha256"][str(args.labels)]
    for name in ("opportunity_categories.jsonl", "history_categories.jsonl"):
        paths[str(args.categories / name)] = category["output_sha256"][name]
    for name, value in original["component_sha256"].items():
        paths[str(args.original / name)] = value
    for name, value in original["ranking_sha256"].items():
        if name.endswith("__hybrid_full.csv.gz"):
            paths[str(args.original / name)] = value
    for path, expected in paths.items():
        checked(Path(path), expected)
    original_plan = read_json(original_plan_path)
    plan = {"kind": "category_representation_development_comparison_v1",
            "created_at_lima": datetime.now(ZoneInfo("America/Lima")).isoformat(),
            "evaluation_status": "posthoc_development_on_inspected_annual_pool_not_independent_or_confirmatory",
            "original": str(args.original), "categories": str(args.categories), "cohort": str(args.cohort),
            "labels": str(args.labels), "cutoff": original_plan["cutoff"], "candidate_end": original_plan["candidate_end"],
            "variants": VARIANTS, "weights": original["models"]["hybrid_full"],
            "changed_signal": "category_only", "other_signals": "reuse_original_float64_components_exactly",
            "primary_metric": "recall@10", "primary_reference": "hybrid_full",
            "primary_metric_rationale": "Recover observed opportunities with equal weight per company; all six protocol metrics remain reported.",
            "metrics": list(query_metrics([], {"placeholder"})), "bootstrap_resamples": 5000, "bootstrap_seed": 42,
            "missing_policy": "zero_contribution_fixed_weights_no_imputation_no_prefix_derivation",
            "code_validation": "published_scheme_characters_length_not_external_catalog_membership",
            "item_scope_limit": "First or any tender item describes a historical process, not certified item-level supplier capacity.",
            "automatic_model_selection": False, "weights_tuning": False, "labels_read_during_preparation": False,
            "prior_results_already_inspected": True, "input_sha256": paths}
    order, opportunities, history, _ = load_features(plan)
    profiles = sorted({r["supplier_id"] for r in history})
    originals = read_json(args.original / "results.json")
    if profiles != sorted(originals["runs"]["hybrid_full"]["top_k"]):
        raise ValueError("Los proveedores auxiliares difieren de la corrida original")
    for configuration in VARIANTS.values():
        CategoryCodeSignal(history, opportunities, configuration["scheme"], configuration["first_item_only"])
    plan["population"] = {"profiles": len(profiles), "opportunities": len(order), "historical_awards": len(history)}
    args.output.mkdir(parents=True)
    archive = args.output / "code_snapshot"
    archive.mkdir()
    root = Path(__file__).resolve().parents[1]
    code_paths = [Path(__file__), Path(__file__).with_name("verify_category_contrast.py"), *(root / "src/tesis_recommender" / name for name in
                  ("category_ranking.py", "category_review.py", "experiment_analysis.py", "validation.py"))]
    for path in code_paths:
        shutil.copyfile(path, archive / path.name)
    plan["code_sha256"] = {str(path): sha256(path) for path in code_paths}
    plan["archived_code_sha256"] = {p.name: sha256(p) for p in archive.iterdir()}
    save_json(args.output / "plan.json", plan)
    save_json(args.output / "plan_lock.json", {"plan_sha256": sha256(args.output / "plan.json"), "status": "prepared_before_new_rankings"})
    return plan


def save_ranking(path, supplier, variant, scores, ranking, order):
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["supplier_id", "model", "rank", "opportunity_id", "score"])
                writer.writerows((supplier, variant, pos, order[i], repr(float(scores[i]))) for pos, i in enumerate(ranking, 1))


def verify_reference(path, supplier, scores, ranking, order):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        count = 0
        for count, row in enumerate(reader, 1):
            if count > len(order):
                raise ValueError("Ranking original con filas adicionales")
            i = ranking[count - 1]
            if (row["supplier_id"] != supplier or row["model"] != "hybrid_full" or int(row["rank"]) != count
                    or row["opportunity_id"] != order[i] or float(row["score"]) != float(scores[i])):
                raise ValueError("Las señales no reproducen exactamente el híbrido original")
        if count != len(order):
            raise ValueError("Ranking original incompleto")


def run(plan_path):
    output = plan_path.parent
    lock = read_json(output / "plan_lock.json")
    checked(plan_path, lock["plan_sha256"])
    plan = read_json(plan_path)
    if (output / "manifest.json").exists() or (output / "results.json").exists() or (output / "rankings").exists():
        raise ValueError("No sobrescribir una ejecución preparada o terminada")
    if plan["variants"] != VARIANTS or plan["changed_signal"] != "category_only":
        raise ValueError("La especificación de variantes cambió")
    for path, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        checked(Path(path), expected)
    for name, expected in plan["archived_code_sha256"].items():
        checked(output / "code_snapshot" / name, expected)
    order, opportunities, history, csv_opportunities = load_features(plan)
    profiles = sorted({r["supplier_id"] for r in history})
    signals = {name: CategoryCodeSignal(history, opportunities, cfg["scheme"], cfg["first_item_only"])
               for name, cfg in plan["variants"].items()}
    original_dir = Path(plan["original"])
    original = read_json(original_dir / "results.json")
    truth = defaultdict(set)
    seen = set()
    candidate_ids = set(order)
    for row in read_rows(Path(plan["labels"])):
        identity = row["supplier_id"], row["opportunity_id"]
        if identity in seen or row["supplier_id"] not in profiles or row["opportunity_id"] not in candidate_ids or float(row["label"]) != 1:
            raise ValueError("Etiqueta duplicada, no positiva o fuera de la población congelada")
        seen.add(identity)
        truth[row["supplier_id"]].add(row["opportunity_id"])
    evaluable = sorted(truth)
    if not evaluable:
        raise ValueError("No hay positivos observados para evaluar")
    metrics = plan["metrics"]
    per_profile = {m: {} for m in original["runs"]}
    for model, result in original["runs"].items():
        if sorted(result["top_k"]) != profiles:
            raise ValueError("Una referencia usa otros proveedores")
        for sid in evaluable:
            per_profile[model][sid] = query_metrics([x["opportunity_id"] for x in result["top_k"][sid]], truth[sid])
        for metric in metrics:
            mean = sum(per_profile[model][sid][metric] for sid in evaluable) / len(evaluable)
            if abs(mean - result["metrics"][metric]) > 1e-6:
                raise ValueError("Las etiquetas no reproducen las métricas de referencia")
    started = time.monotonic()
    (output / "rankings").mkdir()
    (output / "category_components").mkdir()
    manifest = {"status": "running", "plan_sha256": lock["plan_sha256"],
                "evaluation_status": plan["evaluation_status"], "weights_tuned": False, "default_ranker_changed": False,
                "embedding_model_loaded": False, "embeddings_regenerated": False, "ranking_sha256": {}, "category_component_sha256": {}}
    save_json(output / "manifest.json", manifest)
    runs = {name: {"metrics": {}, "top_k": {}} for name in signals}
    per_profile.update({name: {} for name in signals})
    positive_positions, coverage, transitions = [], {}, []
    for name, signal in signals.items():
        coverage[name] = {"candidates_with_codes": sum(bool(x) for x in signal.candidate_keys),
                          "profiles_with_codes": sum(bool(x) for x in signal.profile_keys.values()),
                          "scheme": signal.scheme, "scope": "first_item" if signal.first_item_only else "any_item"}
    reference_rows_checked = 0
    for number, sid in enumerate(profiles, 1):
        with np.load(original_dir / "components" / f"{sid}.npz", allow_pickle=False) as arrays:
            components = {s: arrays[s] for s in SIGNALS}
        scores, ranking = rank_components(components, plan["weights"], order)
        verify_reference(original_dir / "rankings" / f"{sid}__hybrid_full.csv.gz", sid, scores, ranking, order)
        reference_rows_checked += len(order)
        replacements = {}
        for name, signal in signals.items():
            category = signal.scores(sid)
            replacements[name] = category
            scores, ranking = rank_components(components, plan["weights"], order, category)
            path = output / "rankings" / f"{sid}__{name}.csv.gz"
            save_ranking(path, sid, name, scores, ranking, order)
            manifest["ranking_sha256"][str(path.relative_to(output))] = sha256(path)
            top = []
            for i in ranking[:10]:
                contributions = {s: plan["weights"].get(s, 0.) * float(category[i] if s == "category" else components[s][i]) for s in SIGNALS}
                if sum(contributions.values()) != float(scores[i]):
                    raise ValueError("La explicación no reproduce el score")
                explanation = signal.explanation(sid, i)
                explanation.update({"component_scores": {s: float(category[i] if s == "category" else components[s][i]) for s in SIGNALS},
                                    "contributions": contributions, "region_semantics": "buyer_address_not_verified_execution_location",
                                    "missing_policy": plan["missing_policy"], "dominant_signal": max(contributions, key=contributions.get)})
                top.append({"opportunity_id": order[i], "score": float(scores[i]), "source_url": csv_opportunities[order[i]]["source_url"],
                            "initial_source_uri": opportunities[i].get("source_uri", ""), "explanation": explanation})
            runs[name]["top_k"][sid] = top
            if sid in truth:
                new_top = [x["opportunity_id"] for x in top]
                per_profile[name][sid] = query_metrics(new_top, truth[sid])
                positions = {order[i]: pos for pos, i in enumerate(ranking, 1) if order[i] in truth[sid]}
                for ocid in sorted(truth[sid]):
                    i = order.index(ocid)
                    positive_positions.append({"supplier_id": sid, "opportunity_id": ocid, "model": name, "rank": positions[ocid],
                                               "category_score": float(category[i]), "category_explanation": signal.explanation(sid, i)})
                for reference in ("hybrid_full", "bm25_labse", "bm25"):
                    base = [x["opportunity_id"] for x in original["runs"][reference]["top_k"][sid]]
                    for k in (5, 10):
                        gains = (set(new_top[:k]) - set(base[:k])) & truth[sid]
                        losses = (set(base[:k]) - set(new_top[:k])) & truth[sid]
                        transitions.append({"supplier_id": sid, "model": name, "reference": reference, "k": k,
                                            "gained_observed_positives": sorted(gains), "lost_observed_positives": sorted(losses)})
        path = output / "category_components" / f"{sid}.npz"
        np.savez_compressed(path, **replacements)
        manifest["category_component_sha256"][str(path.relative_to(output))] = sha256(path)
        progress = {"profiles_completed": number, "profiles_planned": len(profiles), "elapsed_seconds": round(time.monotonic() - started, 3)}
        save_json(output / "progress.json", progress)
        if number == 1 or number % 10 == 0 or number == len(profiles):
            print(json.dumps(progress), flush=True)
    for name in runs:
        runs[name]["metrics"] = {metric: round(sum(per_profile[name][sid][metric] for sid in evaluable) / len(evaluable), 6) for metric in metrics}
    generator = random.Random(plan["bootstrap_seed"])
    indices = [[generator.randrange(len(evaluable)) for _ in evaluable] for _ in range(plan["bootstrap_resamples"])]
    comparisons = []
    contrasts = [(name, ref) for name in runs for ref in ("hybrid_full", "bm25_labse", "bm25")]
    contrasts += [("hybrid_cubso_any", "hybrid_cubso_first"), ("hybrid_unspsc_any", "hybrid_unspsc_first")]
    for left, right in contrasts:
        for metric in metrics:
            differences = [per_profile[left][sid][metric] - per_profile[right][sid][metric] for sid in evaluable]
            comparisons.append({"left": left, "right": right, "metric": metric, **paired_interval(differences, indices)})
    result = {"evaluation_status": plan["evaluation_status"], "population": {**plan["population"],
               "known_positive_pairs": len(seen), "profiles_with_known_positives": len(evaluable), "profiles_without_known_positives": len(profiles) - len(evaluable)},
              "primary_metric": plan["primary_metric"], "primary_reference": plan["primary_reference"],
              "runs": runs, "reference_metrics": {name: original["runs"][name]["metrics"] for name in original["runs"]},
              "coverage": coverage, "comparisons": comparisons,
              "interpretation_limit": "Exploratory paired intervals on an inspected pool with incomplete observed-positive labels. No automatic winner or hypothesis confirmation."}
    save_json(output / "results.json", result)
    save_json(output / "per_profile_metrics.json", per_profile)
    save_json(output / "positive_positions.json", positive_positions)
    save_json(output / "transitions.json", transitions)
    for path, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        checked(Path(path), expected)
    manifest.update({"status": "complete", "inputs_unchanged": True, "reference_rows_reproduced_exactly": reference_rows_checked,
                     "new_ranking_rows": len(order) * len(profiles) * len(runs), "top10_explanations": len(profiles) * len(runs) * 10,
                     "output_sha256": {name: sha256(output / name) for name in ("results.json", "per_profile_metrics.json", "positive_positions.json", "transitions.json")},
                     "elapsed_seconds": round(time.monotonic() - started, 3), "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    save_json(output / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "metrics": {name: value["metrics"] for name, value in runs.items()},
                      "peak_rss_native_units": manifest["peak_rss_native_units"], "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=Path("artifacts/experiment_annual_v1"))
    parser.add_argument("--categories", type=Path, default=Path("artifacts/annual_category_review_v1"))
    parser.add_argument("--cohort", type=Path, default=Path("data/processed/contrast_candidates_v1/annual_v1"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/annual_positive_audit_v2/labels.csv"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.plan:
        run(args.plan)
    else:
        if args.output is None:
            parser.error("Indicar --output para preparar un plan o --plan para ejecutar uno congelado")
        prepare(args)
        print("Plan conservado antes de ejecutar las nuevas variantes", flush=True)
        if not args.freeze_only:
            run(args.output / "plan.json")
