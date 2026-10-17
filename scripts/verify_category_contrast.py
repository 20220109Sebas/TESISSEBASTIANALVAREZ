"""Verifica la comparación por códigos sin usar su ranker para la referencia."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tesis_recommender.experiment_analysis import query_metrics
from tesis_recommender.validation import sha256

SIGNALS = ("bm25", "semantic", "category", "region", "amount")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Archivo modificado: {path}")


def independently_published_codes(record, scheme, first_only):
    length = {"CUBSO": 16, "UNSPSC": 8}[scheme]
    return {entry["code"] for entry in record["classifications"]
            if entry["scheme"] == scheme and isinstance(entry["code"], str)
            and entry["code"].isascii() and entry["code"].isdecimal() and len(entry["code"]) == length
            and (not first_only or entry["item_index"] == 0)}


def verify(directory):
    manifest = read_json(directory / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("La comparación no está terminada")
    check(directory / "plan.json", manifest["plan_sha256"])
    plan = read_json(directory / "plan.json")
    check(directory / "plan.json", read_json(directory / "plan_lock.json")["plan_sha256"])
    for path, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        check(Path(path), expected)
    for name, expected in plan["archived_code_sha256"].items():
        check(directory / "code_snapshot" / name, expected)
    for section in ("output_sha256", "ranking_sha256", "category_component_sha256"):
        for name, expected in manifest[section].items():
            check(directory / name, expected)
    result = read_json(directory / "results.json")
    originals = read_json(Path(plan["original"]) / "results.json")
    order = read_json(Path(plan["original"]) / "candidate_order.json")
    index = {ocid: i for i, ocid in enumerate(order)}
    category_dir = Path(plan["categories"])
    with (category_dir / "opportunity_categories.jsonl").open(encoding="utf-8") as handle:
        opportunities = {r["ocid"]: r for r in map(json.loads, handle)}
    with (category_dir / "history_categories.jsonl").open(encoding="utf-8") as handle:
        history = list(map(json.loads, handle))
    profiles = sorted({r["supplier_id"] for r in history})
    truth = defaultdict(set)
    with Path(plan["labels"]).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            truth[row["supplier_id"]].add(row["opportunity_id"])
    archived_positions = {(r["supplier_id"], r["opportunity_id"], r["model"]): r for r in read_json(directory / "positive_positions.json")}
    checked_rows = explanations = positives_checked = 0
    values = defaultdict(list)
    for sid in profiles:
        with np.load(Path(plan["original"]) / "components" / f"{sid}.npz", allow_pickle=False) as loaded:
            components = {name: loaded[name] for name in SIGNALS}
        with np.load(directory / "category_components" / f"{sid}.npz", allow_pickle=False) as loaded:
            stored = {name: loaded[name] for name in plan["variants"]}
        for variant, cfg in plan["variants"].items():
            profile_codes = set().union(*(independently_published_codes(r, cfg["scheme"], cfg["first_item_only"])
                                         for r in history if r["supplier_id"] == sid))
            candidate_codes = [independently_published_codes(opportunities[o], cfg["scheme"], cfg["first_item_only"]) for o in order]
            category = np.array([float(bool(keys & profile_codes)) for keys in candidate_codes])
            if not np.array_equal(stored[variant], category):
                raise ValueError("La categoría almacenada difiere de los códigos publicados")
            seen, top, previous = set(), [], None
            path = directory / "rankings" / f"{sid}__{variant}.csv.gz"
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                for position, row in enumerate(csv.DictReader(handle), 1):
                    ocid = row["opportunity_id"]
                    if ocid not in index or ocid in seen or int(row["rank"]) != position or row["supplier_id"] != sid or row["model"] != variant:
                        raise ValueError("Identidad, cobertura u orden de posiciones incorrectos")
                    seen.add(ocid)
                    i = index[ocid]
                    expected = sum(plan["weights"].get(name, 0.) * float(category[i] if name == "category" else components[name][i]) for name in SIGNALS)
                    key = (-expected, ocid)
                    if float(row["score"]) != expected or previous is not None and key < previous:
                        raise ValueError("Score o desempate no reproducible")
                    previous = key
                    checked_rows += 1
                    if position <= 10:
                        top.append(ocid)
                        item = result["runs"][variant]["top_k"][sid][position - 1]
                        explanation = item["explanation"]
                        shared = candidate_codes[i] & profile_codes
                        status = "unavailable" if not candidate_codes[i] or not profile_codes else "observed_match" if shared else "observed_no_exact_match"
                        expected_scores = {name: float(category[i] if name == "category" else components[name][i]) for name in SIGNALS}
                        expected_contributions = {name: plan["weights"].get(name, 0.) * expected_scores[name] for name in SIGNALS}
                        if (item["opportunity_id"] != ocid or item["score"] != expected
                                or explanation["component_scores"] != expected_scores or explanation["contributions"] != expected_contributions
                                or explanation["category_status"] != status or explanation["shared_category_codes"] != sorted(shared)):
                            raise ValueError("Explicación inconsistente con códigos o componentes")
                        explanations += 1
                    if ocid in truth.get(sid, set()):
                        observed = archived_positions[sid, ocid, variant]
                        if observed["rank"] != position or observed["category_score"] != float(category[i]):
                            raise ValueError("Posición positiva archivada incorrecta")
                        positives_checked += 1
            if len(seen) != len(order):
                raise ValueError("Ranking incompleto")
            if sid in truth:
                values[variant].append(query_metrics(top, truth[sid]))
    for name in plan["variants"]:
        for metric in plan["metrics"]:
            mean = round(sum(row[metric] for row in values[name]) / len(values[name]), 6)
            if mean != result["runs"][name]["metrics"][metric]:
                raise ValueError("Las métricas no se reproducen desde los rankings")
    if result["reference_metrics"] != {name: run["metrics"] for name, run in originals["runs"].items()}:
        raise ValueError("Las métricas de referencia cambiaron")
    report = {"status": "passed", "new_ranking_rows_checked": checked_rows, "top10_explanations_checked": explanations,
              "positive_positions_checked": positives_checked, "category_rule_independently_recomputed": True,
              "metrics_recomputed_from_full_rankings": True, "source_and_code_hashes_verified": True,
              "verifier_sha256": sha256(Path(__file__)), "manifest_sha256": sha256(directory / "manifest.json"),
              "interpretation": "Software verification, not independent scientific validation of relevance."}
    target = directory / "verification.json"
    if target.exists() and read_json(target) != report:
        raise ValueError("La nueva verificación difiere de la archivada")
    if not target.exists():
        target.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    verify(parser.parse_args().experiment)
