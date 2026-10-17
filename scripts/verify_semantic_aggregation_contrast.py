"""Independent maximum, score/order, explanation and metric checks for this contrast.

Does not import the new ranker, aggregation function or execution script.
"""
from __future__ import annotations

import os
for _thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

SIGNALS = ("bm25", "semantic", "category", "region", "amount")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(top, positives):
    answer = {}
    for k in (5, 10):
        hits = len(set(top[:k]) & positives)
        p, r = hits / k, hits / len(positives)
        answer.update({f"precision@{k}": p, f"recall@{k}": r, f"f1@{k}": 2 * p * r / (p + r) if p + r else 0.})
    return answer


def verify(output):
    output = Path(output)
    require(digest(output / "plan.json") == read_json(output / "plan_lock.json")["plan_sha256"], "Plan changed")
    plan, manifest = read_json(output / "plan.json"), read_json(output / "manifest.json")
    require(manifest["status"] == "complete" and manifest["plan_sha256"] == digest(output / "plan.json"), "Run not complete or plan changed")
    for path, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        require(digest(path) == expected, f"Input/code changed: {path}")
    for name, expected in plan["archived_code_sha256"].items():
        require(digest(output / "code_snapshot" / name) == expected, f"Archived code changed: {name}")
    for group in ("ranking_sha256", "semantic_sha256", "output_sha256"):
        for name, expected in manifest[group].items():
            require(digest(output / name) == expected, f"Output changed: {name}")
    cohort, cache, original = Path(plan["cohort"]), Path(plan["cache"]), Path(plan["original"])
    order = read_json(original / "candidate_order.json")
    require(len(order) == len(set(order)), "Duplicate candidate")
    positions = {oid: i for i, oid in enumerate(order)}
    history = {}
    with (cohort / "historical_awards.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            history.setdefault(row["supplier_id"], []).append(row)
    for awards in history.values():
        awards.sort(key=lambda row: row["award_id"])
    with (cohort / "test_opportunities.csv").open(encoding="utf-8-sig", newline="") as handle:
        opportunity_rows = {row["opportunity_id"]: row for row in csv.DictReader(handle)}
    provenance = {}
    for filename, key in (("history_categories.jsonl", "award_id"), ("opportunity_categories.jsonl", "ocid")):
        with (Path(plan["provenance"]) / filename).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                provenance[row[key]] = row
    matrix = np.load(cache / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    index = {(row["kind"], row["id"]): row for row in read_json(cache / "index.json")}
    candidate_rows = np.asarray([index["opportunity", oid]["row"] for oid in order])
    sources = read_json(cache / "source_store_index.json")
    blocks = {}
    truth = {}
    with Path(plan["labels"]).open(encoding="utf-8-sig", newline="") as handle:
        identities = set()
        for row in csv.DictReader(handle):
            identity = row["supplier_id"], row["opportunity_id"]
            require(identity not in identities and identity[0] in history and identity[1] in positions and float(row["label"]) == 1., "Invalid label")
            identities.add(identity)
            truth.setdefault(identity[0], set()).add(identity[1])
    results = read_json(output / "results.json")
    require(set(results["runs"]) == set(plan["models"]), "Variant set mismatch")
    require(all(set(run["top_k"]) == set(history) for run in results["runs"].values()), "Profile set mismatch")
    saved_per_profile = read_json(output / "per_profile_metrics.json")
    original_results = read_json(original / "results.json")
    positive_list = read_json(output / "positive_positions.json")
    positive_records = {(r["supplier_id"], r["model"], r["opportunity_id"]): r for r in positive_list}
    require(len(positive_list) == len(positive_records), "Duplicate positive position")
    expected_positive_keys, verified_rows, explanation_count = set(), 0, 0
    per_profile = {model: {} for model in results["runs"]}
    computed_transitions = []
    with threadpool_limits(limits=1, user_api="blas"):
        for sid, awards in sorted(history.items()):
            vectors, source_entries = [], []
            for award in awards:
                text = " ".join(award[k] for k in ("title", "description", "technical_text") if award[k]).strip()
                text_digest = hashlib.sha256(text.encode()).hexdigest()
                entry = sources["texts"][text_digest]
                name = entry["block"]
                if name not in blocks:
                    blocks[name] = np.load(Path(plan["store"]) / name, mmap_mode="r", allow_pickle=False)
                vectors.append(np.asarray(blocks[name][entry["row"]], dtype=np.float64))
                source_entries.append((text_digest, entry))
            with np.load(output / "semantic_components" / f"{sid}.npz", allow_pickle=False) as values:
                raw, normalized, support = values["raw_max"], values["semantic"], values["support"]
            require(raw.shape == normalized.shape == support.shape == (len(order),), "Invalid new component shape")
            require(np.isfinite(raw).all() and np.isfinite(normalized).all(), "Nonfinite new component")
            require(np.issubdtype(support.dtype, np.integer) and np.all((support >= 0) & (support < len(awards))), "Invalid support")
            # Independent candidate blocks and award-wise reduction, not production matrix multiplication.
            for start in range(0, len(order), 1024):
                candidates = np.asarray(matrix[candidate_rows[start:start + 1024]], dtype=np.float64)
                all_scores = np.asarray([np.einsum("ij,j->i", candidates, v, optimize=False) for v in vectors])
                independent = all_scores.max(axis=0)
                end = start + len(candidates)
                require(np.allclose(independent, raw[start:end], rtol=0., atol=2e-12), "Maximum cosine mismatch")
                chosen = all_scores[support[start:end], np.arange(len(candidates))]
                require(np.allclose(chosen, independent, rtol=0., atol=2e-12), "Supporting award not a maximum")
                # Identical document vectors must always select the lowest award_id.
                for j in range(len(awards)):
                    earlier = [k for k in range(j) if np.array_equal(vectors[k], vectors[j])]
                    require(not earlier or not np.any(support[start:end] == j), "Duplicate-vector tie did not select first award")
            low, high = float(raw.min()), float(raw.max())
            expected_normalized = np.full(len(raw), float(high > 0)) if math.isclose(low, high) else (raw - low) / (high - low)
            require(np.array_equal(normalized, expected_normalized), "Minmax mismatch")
            with np.load(original / "components" / f"{sid}.npz", allow_pickle=False) as values:
                original_components = {s: values[s] for s in SIGNALS}

            def verify_explanation(explanation, i, weights):
                nonlocal explanation_count
                selected = int(support[i])
                aid = awards[selected]["award_id"]
                require(explanation["raw_max_cosine"] == float(raw[i]), "Explanation raw cosine mismatch")
                require(explanation["supporting_award"]["award_id"] == aid, "Wrong explanation support")
                for label, key in (("supporting_award", aid), ("candidate_source", order[i])):
                    for field in ("source_path", "source_sha256", "source_uri", "release_id", "release_published_at", "ocid"):
                        require(explanation[label][field] == provenance[key][field], "Wrong source evidence")
                source_digest, entry = source_entries[selected]
                require(explanation["supporting_award"]["text_sha256"] == source_digest, "Wrong historical text hash")
                for field in ("block", "row"):
                    require(explanation["supporting_award"][field] == entry[field], "Wrong source vector")
                require(explanation["supporting_award"]["block_sha256"] == sources["blocks"][entry["block"]]["sha256"], "Wrong block hash")
                candidate = opportunity_rows[order[i]]
                text = " ".join(candidate[k] for k in ("title", "description", "technical_text") if candidate[k]).strip()
                require(explanation["candidate_source"]["text_sha256"] == hashlib.sha256(text.encode()).hexdigest(), "Wrong candidate text hash")
                for s in SIGNALS:
                    expected = float(normalized[i] if s == "semantic" else original_components[s][i])
                    require(explanation["component_scores"][s] == expected, "Explanation component mismatch")
                    require(explanation["contributions"][s] == weights.get(s, 0.) * expected, "Explanation contribution mismatch")
                explanation_count += 1

            for model, weights in plan["models"].items():
                scores = [sum(weights.get(s, 0.) * float(normalized[i] if s == "semantic" else original_components[s][i]) for s in SIGNALS) for i in range(len(order))]
                ranked = sorted(range(len(order)), key=lambda i: (-scores[i], order[i]))
                rank_positions = {order[i]: pos for pos, i in enumerate(ranked, 1)}
                with gzip.open(output / "rankings" / f"{sid}__{model}.csv.gz", "rt", encoding="utf-8", newline="") as handle:
                    count = 0
                    for count, row in enumerate(csv.DictReader(handle), 1):
                        require(count <= len(order), "Extra ranking row")
                        i = ranked[count - 1]
                        require((row["supplier_id"], row["model"], int(row["rank"]), row["opportunity_id"], float(row["score"])) == (sid, model, count, order[i], scores[i]), "Ranking score/order mismatch")
                    require(count == len(order), "Incomplete ranking")
                    verified_rows += count
                top = results["runs"][model]["top_k"][sid]
                require(len(top) == 10, "Wrong top length")
                for row, i in zip(top, ranked[:10]):
                    require(row["opportunity_id"] == order[i] and row["score"] == scores[i], "Top differs from full ranking")
                    verify_explanation(row["explanation"], i, weights)
                if sid in truth:
                    per_profile[model][sid] = metrics([order[i] for i in ranked[:10]], truth[sid])
                    reference = plan["references"][model]
                    base_scores = [sum(weights.get(s, 0.) * float(original_components[s][i]) for s in SIGNALS) for i in range(len(order))]
                    base_ranked = sorted(range(len(order)), key=lambda i: (-base_scores[i], order[i]))
                    base_positions = {order[i]: pos for pos, i in enumerate(base_ranked, 1)}
                    for oid in truth[sid]:
                        key = sid, model, oid
                        expected_positive_keys.add(key)
                        record = positive_records[key]
                        require(record["rank"] == rank_positions[oid] and record["reference_rank"] == base_positions[oid], "Positive position mismatch")
                        verify_explanation(record["explanation"], positions[oid], weights)
                    for k in (5, 10):
                        new_top, old_top = {order[i] for i in ranked[:k]}, {order[i] for i in base_ranked[:k]}
                        computed_transitions.append({"supplier_id": sid, "model": model, "reference": reference, "k": k,
                            "gained_observed_positives": sorted((new_top - old_top) & truth[sid]),
                            "lost_observed_positives": sorted((old_top - new_top) & truth[sid])})
            print(json.dumps({"supplier_id": sid, "ranking_rows_verified": verified_rows}), flush=True)
    require(set(positive_records) == expected_positive_keys, "Unexpected positive record")
    canonical = lambda rows: sorted(json.dumps(r, sort_keys=True) for r in rows)
    require(canonical(computed_transitions) == canonical(read_json(output / "transitions.json")), "Transition mismatch")
    for model, values in per_profile.items():
        require(saved_per_profile[model] == values, "Per profile metrics mismatch")
        means = {metric: round(sum(v[metric] for v in values.values()) / len(values), 6) for metric in next(iter(values.values()))}
        require(results["runs"][model]["metrics"] == means, "Aggregate metrics mismatch")
    require(results["population"]["profiles_without_known_positives"] == len(history) - len(truth), "Unknown profiles counted incorrectly")
    # Independently regenerate paired bootstrap indices and percentile interpolation.
    import random
    rng = random.Random(plan["bootstrap_seed"])
    evaluable = sorted(truth)
    draws = [[rng.randrange(len(evaluable)) for _ in evaluable] for _ in range(plan["bootstrap_resamples"])]
    expected_comparisons = {(left, right, metric) for left, right in plan["references"].items() for metric in plan["metrics"]}
    actual_comparisons = [(row["left"], row["right"], row["metric"]) for row in results["comparisons"]]
    require(len(actual_comparisons) == len(expected_comparisons) and set(actual_comparisons) == expected_comparisons, "Incomplete or duplicate comparisons")
    for comparison in results["comparisons"]:
        left, right, metric = comparison["left"], comparison["right"], comparison["metric"]
        deltas = [per_profile[left][sid][metric] - metrics([r["opportunity_id"] for r in original_results["runs"][right]["top_k"][sid]], truth[sid])[metric] for sid in evaluable]
        means = sorted(sum(deltas[i] for i in draw) / len(draw) for draw in draws)
        expected = {"mean": round(sum(deltas) / len(deltas), 6)}
        for key, p in (("ci95_low", .025), ("ci95_high", .975)):
            at = (len(means) - 1) * p
            lower = int(at)
            expected[key] = round(means[lower] + (means[min(lower + 1, len(means) - 1)] - means[lower]) * (at - lower), 6)
        require(all(comparison[k] == v for k, v in expected.items()), "Bootstrap mismatch")
    return {"status": "passed", "ranking_rows": verified_rows, "explanations": explanation_count,
            "profiles": len(history), "profiles_evaluable": len(truth), "positive_positions": len(positive_records),
            "independent_cosine_atol": 2e-12, "scores_python_sum_exact": True, "bootstrap_verified": True,
            "plan_sha256": digest(output / "plan.json"), "manifest_sha256": digest(output / "manifest.json"),
            "verifier_sha256": digest(Path(__file__)), "scientific_independent_validation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    target = args.run / "verification.json"
    if target.exists():
        raise ValueError("Do not overwrite verification")
    report = verify(args.run)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
