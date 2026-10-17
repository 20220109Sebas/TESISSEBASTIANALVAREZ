"""Prepare, then separately execute one maximum-over-awards development contrast."""
from __future__ import annotations

import os
for _thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_variable] = "1"

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
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from tesis_recommender.experiment_analysis import paired_interval, query_metrics
from tesis_recommender.io import read_awards, read_opportunities
from tesis_recommender.profiles import build_profiles, assert_no_temporal_leakage
from tesis_recommender.semantic_aggregation import (MODELS, SIGNALS, SourceVectors, OrderedVectors,
    award_text, checked, maximum_similarity, rank_components, text_hash)
from tesis_recommender.validation import sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def provenance(path, identity):
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row[identity]
            if key in result:
                raise ValueError("Duplicate provenance identity")
            result[key] = {k: row[k] for k in (identity, "ocid", "source_path", "source_sha256", "source_uri", "release_id", "release_published_at")}
            if "supplier_id" in row:
                result[key]["supplier_id"] = row["supplier_id"]
    return result


def load_population(plan):
    cohort, cache = Path(plan["cohort"]), Path(plan["cache"])
    awards = read_awards(cohort / "historical_awards.csv")
    if len({a.award_id for a in awards}) != len(awards):
        raise ValueError("Duplicate award identity")
    profiles = build_profiles(awards, date.fromisoformat(plan["cutoff"]))
    assert_no_temporal_leakage(profiles)
    opportunities = read_opportunities(cohort / "test_opportunities.csv")
    by_id = {o.opportunity_id: o for o in opportunities}
    order = read_json(Path(plan["original"]) / "candidate_order.json")
    if len(by_id) != len(opportunities) or len(order) != len(set(order)) or set(order) != set(by_id):
        raise ValueError("Candidate population changed")
    index = read_json(cache / "index.json")
    entries = {}
    for entry in index:
        identity = entry["kind"], entry["id"]
        if identity in entries:
            raise ValueError("Duplicate embedding identity")
        entries[identity] = entry
    expected = {("profile", sid): p.text for sid, p in profiles.items()}
    expected.update({("opportunity", oid): by_id[oid].text for oid in order})
    if set(entries) != set(expected) or any(entries[k]["text_sha256"] != text_hash(v) for k, v in expected.items()):
        raise ValueError("Embedding index does not represent this population")
    metadata = read_json(cache / "metadata.json")
    matrix = np.load(cache / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if matrix.shape != (metadata["vector_count"], metadata["embedding_dimension"]) or matrix.dtype != np.float32:
        raise ValueError("Invalid candidate matrix")
    vectors = OrderedVectors(matrix, [entries["opportunity", oid]["row"] for oid in order])
    historical = provenance(Path(plan["provenance"]) / "history_categories.jsonl", "award_id")
    candidates = provenance(Path(plan["provenance"]) / "opportunity_categories.jsonl", "ocid")
    if set(historical) != {a.award_id for a in awards} or set(candidates) != set(order):
        raise ValueError("Provenance population differs")
    for award in awards:
        if historical[award.award_id]["supplier_id"] != award.supplier_id:
            raise ValueError("Historical evidence belongs to another supplier")
        stamp = datetime.fromisoformat(historical[award.award_id]["release_published_at"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.astimezone(ZoneInfo("America/Lima")).date() >= date.fromisoformat(plan["cutoff"]):
            raise ValueError("Historical evidence unavailable before cutoff")
    for oid, source in candidates.items():
        stamp = datetime.fromisoformat(source["release_published_at"].replace("Z", "+00:00"))
        if (stamp.tzinfo is None or not date.fromisoformat(plan["cutoff"]) <= stamp.astimezone(ZoneInfo("America/Lima")).date() <= date.fromisoformat(plan["candidate_end"])
                or by_id[oid].awarded_supplier_id or by_id[oid].object_type != "services"):
            raise ValueError("Candidate evidence outside window or containing outcomes")
    return profiles, order, by_id, vectors, historical, candidates


def prepare(args):
    if args.output.exists():
        raise ValueError("Use a new output directory")
    original = read_json(args.original / "manifest.json")
    if original["status"] != "complete":
        raise ValueError("Original run is incomplete")
    paths = {str(args.original / "manifest.json"): sha256(args.original / "manifest.json")}
    # Follow the original manifest's expected hashes, including the temporal plan.
    paths.update(original["input_sha256"])
    paths[str(args.original / "results.json")] = original["results_sha256"]
    paths[str(args.original / "candidate_order.json")] = original["candidate_order_sha256"]
    for key in ("component_sha256", "ranking_sha256"):
        paths.update({str(args.original / name): digest for name, digest in original[key].items()})
    for path in (args.cohort / "historical_awards.csv", args.cohort / "test_opportunities.csv", args.labels,
                 args.cache / "metadata.json", args.cache / "index.json", args.cache / "embeddings.npy"):
        if str(path) not in paths:
            raise ValueError(f"Input absent from original manifest: {path}")
    for path, expected in paths.items():
        checked(path, expected)
    metadata = read_json(args.cache / "metadata.json")
    snapshot = args.cache / "source_store_index.json"
    paths[str(snapshot)] = metadata["source_store_index_snapshot_sha256"]
    checked(snapshot, paths[str(snapshot)])
    source_index = read_json(snapshot)
    for name, info in source_index["blocks"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe source path")
        paths[str(args.store / name)] = info["sha256"]
    evidence_manifest = read_json(args.provenance / "manifest.json")
    paths[str(args.provenance / "manifest.json")] = sha256(args.provenance / "manifest.json")
    for name in ("history_categories.jsonl", "opportunity_categories.jsonl"):
        paths[str(args.provenance / name)] = evidence_manifest["output_sha256"][name]
    for name in ("historical_awards.csv", "test_opportunities.csv"):
        path = str(args.cohort / name)
        if evidence_manifest["input_sha256"][path] != paths[path]:
            raise ValueError("Provenance from another cohort")
    frozen_path = next(p for p in original["input_sha256"] if p.endswith("frozen_plan_with_code.json"))
    frozen = read_json(frozen_path)
    plan = {"kind": "semantic_max_award_development_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "original": str(args.original), "cohort": str(args.cohort), "cache": str(args.cache), "store": str(args.store),
        "provenance": str(args.provenance), "labels": str(args.labels), "cutoff": frozen["cutoff"],
        "candidate_end": frozen["candidate_end"], "models": {m + "_max_award": original["models"][m] for m in MODELS},
        "references": {m + "_max_award": m for m in MODELS}, "changed_signal": "semantic_only",
        "aggregation": "max_float64_dot_over_individual_award_document_embeddings_then_original_profile_minmax",
        "cosine_convention": "dot_of_cached_approximately_L2_unit_float32_vectors_in_float64_as_original_no_renormalization",
        "unit_norm_absolute_tolerance": 1e-5,
        "award_order_and_tie": "ascending_award_id_first_exact_maximum", "candidate_tie": "ascending_opportunity_id",
        "category": "original_name_exact_match_unchanged", "block_size": 2048, "blas_threads": 1,
        "primary_contrast": ["hybrid_full_max_award", "hybrid_full", "recall@10"],
        "metrics": list(query_metrics([], {"placeholder"})), "bootstrap_resamples": 5000, "bootstrap_seed": 42,
        "evaluation_status": "posthoc_development_inspected_pool_not_independent_or_confirmatory",
        "weights_tuned": False, "default_changed": False, "labels_parsed_during_preparation": False,
        "limits": ["historical_positive_labels_incomplete", "maximum_can_favor_long_diverse_histories",
                   "document_chunk_aggregation_unchanged", "no_automatic_model_selection"], "input_sha256": paths}
    profiles, order, _, _, history, candidates = load_population(plan)
    for record in [*history.values(), *candidates.values()]:
        paths[record["source_path"]] = record["source_sha256"]
    for path, digest in paths.items():
        checked(path, digest)
    plan["population"] = {"profiles": len(profiles), "opportunities": len(order), "historical_awards": len(history)}
    args.output.mkdir(parents=True)
    archive = args.output / "code_snapshot"
    archive.mkdir()
    root = Path(__file__).resolve().parents[1]
    code = [Path(__file__), Path(__file__).with_name("verify_semantic_aggregation_contrast.py")]
    code += [root / "src/tesis_recommender" / name for name in
             ("semantic_aggregation.py", "io.py", "profiles.py", "schemas.py", "experiment_analysis.py", "validation.py", "ocds.py", "metrics.py")]
    plan["code_sha256"] = {str(p): sha256(p) for p in code}
    for path in code:
        shutil.copyfile(path, archive / path.name)
    plan["archived_code_sha256"] = {p.name: sha256(p) for p in archive.iterdir()}
    save_json(args.output / "plan.json", plan)
    save_json(args.output / "plan_lock.json", {"plan_sha256": sha256(args.output / "plan.json"), "status": "prepared_before_new_rankings"})
    return plan


def ranking_file(path, sid, model, scores, ranking, order, verify=False):
    if verify:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            count = 0
            for count, row in enumerate(csv.DictReader(handle), 1):
                if count > len(order):
                    raise ValueError("Extra ranking row")
                i = ranking[count - 1]
                if (row["supplier_id"], row["model"], int(row["rank"]), row["opportunity_id"], float(row["score"])) != (sid, model, count, order[i], scores[i]):
                    raise ValueError("Original ranking not exactly reproduced")
            if count != len(order):
                raise ValueError("Incomplete ranking")
        return
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["supplier_id", "model", "rank", "opportunity_id", "score"])
                writer.writerows((sid, model, pos, order[i], repr(scores[i])) for pos, i in enumerate(ranking, 1))


def check_plan(plan_path):
    output = plan_path.parent
    checked(plan_path, read_json(output / "plan_lock.json")["plan_sha256"])
    plan = read_json(plan_path)
    for path, digest in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        checked(path, digest)
    for name, digest in plan["archived_code_sha256"].items():
        checked(output / "code_snapshot" / name, digest)
    return plan


def run(plan_path):
    plan = check_plan(plan_path)
    output = plan_path.parent
    if any((output / name).exists() for name in ("manifest.json", "rankings", "semantic_components", "results.json")):
        raise ValueError("Do not overwrite a run")
    profiles, order, opportunities, vectors, historical, candidates = load_population(plan)
    original_dir = Path(plan["original"])
    original = read_json(original_dir / "results.json")
    if any(set(r["top_k"]) != set(profiles) for r in original["runs"].values()):
        raise ValueError("Original profile population differs")
    truth, seen = defaultdict(set), set()
    with Path(plan["labels"]).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            identity = row["supplier_id"], row["opportunity_id"]
            if identity in seen or identity[0] not in profiles or identity[1] not in opportunities or float(row["label"]) != 1:
                raise ValueError("Invalid positive label")
            seen.add(identity)
            truth[identity[0]].add(identity[1])
    evaluable = sorted(truth)
    if not evaluable:
        raise ValueError("No evaluable profiles")
    per_profile = {m: {sid: query_metrics([x["opportunity_id"] for x in r["top_k"][sid]], truth[sid]) for sid in evaluable}
                   for m, r in original["runs"].items()}
    for model, values in per_profile.items():
        if any(abs(sum(v[m] for v in values.values()) / len(evaluable) - original["runs"][model]["metrics"][m]) > 1e-6 for m in plan["metrics"]):
            raise ValueError("Original metrics do not reproduce")
    store = SourceVectors(plan["cache"], plan["store"])
    (output / "rankings").mkdir()
    (output / "semantic_components").mkdir()
    manifest = {"status": "running", "plan_sha256": sha256(plan_path), "ranking_sha256": {}, "semantic_sha256": {},
                "numpy_version": np.__version__, "blas_threads": 1, "weights_tuned": False, "embeddings_regenerated": False,
                "default_changed": False, "evaluation_status": plan["evaluation_status"]}
    save_json(output / "manifest.json", manifest)
    runs = {m: {"metrics": {}, "top_k": {}} for m in plan["models"]}
    per_profile.update({m: {} for m in runs})
    positive_positions, transitions, histories = [], [], []
    started = time.monotonic()
    with threadpool_limits(limits=1, user_api="blas"):
        manifest["threadpools"] = threadpool_info()
        for number, sid in enumerate(sorted(profiles), 1):
            awards = sorted(profiles[sid].awards, key=lambda a: a.award_id)
            resolved = [store.get(award_text(a)) for a in awards]
            raw, semantic, support = maximum_similarity([v for v, _ in resolved], vectors, plan["block_size"])
            path = output / "semantic_components" / f"{sid}.npz"
            np.savez_compressed(path, raw_max=raw, semantic=semantic, support=support)
            manifest["semantic_sha256"][str(path.relative_to(output))] = sha256(path)
            with np.load(original_dir / "components" / f"{sid}.npz", allow_pickle=False) as loaded:
                components = {s: loaded[s] for s in SIGNALS}
            histories.append({"supplier_id": sid, "history_count": len(awards), "unique_history_texts": len({text_hash(award_text(a)) for a in awards}),
                              "known_positives": len(truth.get(sid, set())), "evaluation_status": "evaluable" if sid in truth else "no_known_positives"})

            def explanation(i, weights):
                j = int(support[i])
                scores = {s: float(semantic[i] if s == "semantic" else components[s][i]) for s in SIGNALS}
                return {"aggregation": plan["aggregation"], "raw_max_cosine": float(raw[i]),
                        "component_scores": scores, "contributions": {s: weights.get(s, 0.) * scores[s] for s in SIGNALS},
                        "supporting_award": {**historical[awards[j].award_id], **resolved[j][1]},
                        "candidate_source": {**candidates[order[i]], "text_sha256": text_hash(opportunities[order[i]].text)},
                        "history_count": len(awards), "support_limit": "closest_historical_process_not_certified_capacity",
                        "missing_policy": "zero_fixed_weights_not_evidence_of_incompatibility"}

            for model, weights in plan["models"].items():
                reference = plan["references"][model]
                base_scores, base_rank = rank_components(components, weights, order)
                ranking_file(original_dir / "rankings" / f"{sid}__{reference}.csv.gz", sid, reference, base_scores, base_rank, order, True)
                scores, ranking = rank_components(components, weights, order, semantic)
                path = output / "rankings" / f"{sid}__{model}.csv.gz"
                ranking_file(path, sid, model, scores, ranking, order)
                manifest["ranking_sha256"][str(path.relative_to(output))] = sha256(path)
                runs[model]["top_k"][sid] = [{"opportunity_id": order[i], "score": scores[i], "source_url": opportunities[order[i]].source_url,
                                             "explanation": explanation(i, weights)} for i in ranking[:10]]
                if sid in truth:
                    new_top = [order[i] for i in ranking[:10]]
                    per_profile[model][sid] = query_metrics(new_top, truth[sid])
                    base_positions = {order[i]: pos for pos, i in enumerate(base_rank, 1)}
                    for pos, i in enumerate(ranking, 1):
                        if order[i] in truth[sid]:
                            positive_positions.append({"supplier_id": sid, "model": model, "reference": reference, "opportunity_id": order[i],
                                                       "rank": pos, "reference_rank": base_positions[order[i]], "explanation": explanation(i, weights)})
                    for k in (5, 10):
                        base_top = {order[i] for i in base_rank[:k]}
                        transitions.append({"supplier_id": sid, "model": model, "reference": reference, "k": k,
                            "gained_observed_positives": sorted((set(new_top[:k]) - base_top) & truth[sid]),
                            "lost_observed_positives": sorted((base_top - set(new_top[:k])) & truth[sid])})
            progress = {"profiles_completed": number, "profiles_planned": len(profiles), "elapsed_seconds": round(time.monotonic() - started, 3)}
            save_json(output / "progress.json", progress)
            if number == 1 or number % 10 == 0 or number == len(profiles):
                print(json.dumps(progress), flush=True)
    generator = random.Random(plan["bootstrap_seed"])
    indices = [[generator.randrange(len(evaluable)) for _ in evaluable] for _ in range(plan["bootstrap_resamples"])]
    comparisons = []
    for model in runs:
        runs[model]["metrics"] = {m: round(sum(v[m] for v in per_profile[model].values()) / len(evaluable), 6) for m in plan["metrics"]}
        reference = plan["references"][model]
        for metric in plan["metrics"]:
            differences = [per_profile[model][sid][metric] - per_profile[reference][sid][metric] for sid in evaluable]
            comparisons.append({"left": model, "right": reference, "metric": metric, **paired_interval(differences, indices)})
    result = {"evaluation_status": plan["evaluation_status"], "population": {**plan["population"], "known_positive_pairs": len(seen),
        "profiles_with_known_positives": len(evaluable), "profiles_without_known_positives": len(profiles) - len(evaluable)},
        "primary_contrast": plan["primary_contrast"], "runs": runs, "comparisons": comparisons,
        "reference_metrics": {m: r["metrics"] for m, r in original["runs"].items()}, "limits": plan["limits"]}
    products = {"results.json": result, "per_profile_metrics.json": per_profile, "positive_positions.json": positive_positions,
                "transitions.json": transitions, "history_diagnostics.json": histories}
    for name, value in products.items():
        save_json(output / name, value)
    check_plan(plan_path)
    manifest.update({"status": "complete", "inputs_unchanged": True, "reference_rows_reproduced_exactly": len(order) * len(profiles) * len(runs),
        "new_ranking_rows": len(order) * len(profiles) * len(runs), "top10_explanations": len(profiles) * len(runs) * 10,
        "output_sha256": {name: sha256(output / name) for name in products}, "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    save_json(output / "manifest.json", manifest)
    return {"status": "complete", "metrics": {m: r["metrics"] for m, r in runs.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=Path("artifacts/experiment_annual_v1"))
    parser.add_argument("--cohort", type=Path, default=Path("data/processed/contrast_candidates_v1/annual_v1"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/contrast_embeddings_v1/annual_v1"))
    parser.add_argument("--store", type=Path, default=Path("artifacts/contrast_document_vectors_v1"))
    parser.add_argument("--provenance", type=Path, default=Path("artifacts/annual_category_review_v1"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/annual_positive_audit_v2/labels.csv"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(run(args.plan), ensure_ascii=False))
    else:
        if not args.output:
            parser.error("Use --output to freeze only; --plan to execute a previously frozen plan")
        prepare(args)
        print("Plan frozen; execution requires a separate --plan invocation.")
