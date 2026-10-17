"""Auditoría sin etiquetas de las categorías de las entradas anuales congeladas.

No ejecuta rankings, ajusta pesos ni incorpora datos de versiones posteriores.
Las coincidencias son descriptivas y no métricas de relevancia o elegibilidad.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from tesis_recommender.category_review import (code_keys, extract_categories, informative_label,
                                              normalize_label, primary_labels)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def features_from_source(path, ocid):
    package = json_read(path)
    releases = package.get("releases", [])
    if len(releases) != 1 or releases[0].get("ocid") != ocid:
        raise ValueError(f"Fuente con identidad o versión inesperada: {path}")
    release = releases[0]
    features = extract_categories(release.get("tender") or {})
    features.update({"ocid": ocid, "release_id": release.get("id"),
                     "release_published_at": release.get("publishedDate"), "source_uri": package.get("uri", "")})
    return features


def coverage(records):
    counts, schemes = Counter(), Counter()
    for record in records:
        counts["records"] += 1
        counts["multiple_items"] += record["item_count"] > 1
        counts["generic_or_missing_first_category"] += not informative_label(record["legacy_category"])
        first_label = normalize_label(record["legacy_category"])
        counts["additional_distinct_primary_labels"] += bool(primary_labels(record) - {first_label})
        for scheme in ("CUBSO", "UNSPSC"):
            keys = code_keys(record, scheme)
            counts[f"with_valid_{scheme.lower()}"] += bool(keys)
            counts[f"multiple_distinct_{scheme.lower()}"] += len(keys) > 1
        counts["no_valid_supported_codes"] += not any(e["valid_supported_code"] for e in record["classifications"])
        for entry in record["classifications"]:
            counts["classification_entries"] += 1
            schemes[entry["scheme"] or "missing"] += 1
            counts["invalid_or_unsupported_entries"] += not entry["valid_supported_code"]
        for index in range(record["item_count"]):
            cubso = {e["code"] for e in record["classifications"] if e["item_index"] == index
                     and e["valid_supported_code"] and e["scheme"] == "CUBSO"}
            unspsc = {e["code"] for e in record["classifications"] if e["item_index"] == index
                      and e["valid_supported_code"] and e["scheme"] == "UNSPSC"}
            if cubso and unspsc:
                counts["items_with_both_schemes"] += 1
                counts["items_with_matching_cubso_prefix_unspsc"] += any(c[:8] in unspsc for c in cubso)
    return {**dict(sorted(counts.items())), "schemes": dict(sorted(schemes.items()))}


def inventory(records):
    by_code, by_label = {}, defaultdict(set)
    for record in records:
        for entry in record["classifications"]:
            if not entry["valid_supported_code"]:
                continue
            key = entry["scheme"], entry["code"]
            item = by_code.setdefault(key, {"descriptions": Counter(), "example_sources": {}, "occurrences": 0})
            label = entry["description"]
            item["descriptions"][label] += 1
            item["occurrences"] += 1
            item["example_sources"].setdefault(label, {"ocid": record["ocid"], "source_path": record["source_path"],
                                                       "field": entry["field"]})
            if informative_label(label):
                by_label[entry["scheme"], normalize_label(label)].add(entry["code"])
    catalog = [{"scheme": scheme, "code": code, "occurrences": item["occurrences"],
                "descriptions": dict(sorted(item["descriptions"].items())),
                "normalized_description_count": len({normalize_label(x) for x in item["descriptions"] if informative_label(x)}),
                "example_sources": item["example_sources"]} for (scheme, code), item in sorted(by_code.items())]
    aliases = [{"scheme": scheme, "normalized_description": label, "codes": sorted(codes)}
               for (scheme, label), codes in sorted(by_label.items()) if len(codes) > 1]
    return catalog, aliases


def compare_profile_matches(history, opportunities):
    profiles = defaultdict(list)
    for record in history:
        profiles[record["supplier_id"]].append(record)
    result = []
    for sid, records in sorted(profiles.items()):
        legacy = {r["legacy_category"].strip().lower() for r in records if informative_label(r["legacy_category"])}
        normalized = {normalize_label(r["legacy_category"]) for r in records if informative_label(r["legacy_category"])}
        all_labels = set().union(*(primary_labels(r) for r in records))
        codes = {s: set().union(*(code_keys(r, s) for r in records)) for s in ("CUBSO", "UNSPSC")}
        counts = Counter()
        for candidate in opportunities:
            original = informative_label(candidate["legacy_category"]) and candidate["legacy_category"].strip().lower() in legacy
            norm = informative_label(candidate["legacy_category"]) and normalize_label(candidate["legacy_category"]) in normalized
            checks = {"legacy_first_label": original, "normalized_first_label": norm,
                      "normalized_any_item_label": bool(primary_labels(candidate) & all_labels),
                      **{f"exact_{s.lower()}_any_item": bool(code_keys(candidate, s) & codes[s]) for s in codes}}
            for name, matched in checks.items():
                counts[name] += bool(matched)
                counts[f"{name}_added_vs_legacy"] += bool(matched and not original)
                counts[f"{name}_lost_vs_legacy"] += bool(original and not matched)
        result.append({"supplier_id": sid, "historical_awards": len(records), **dict(sorted(counts.items()))})
    return result


def audit(args):
    if args.output.exists():
        raise ValueError("Usar un directorio nuevo sin sobrescribir una auditoría")
    inputs = {}

    def checked(path, expected=None):
        value = digest(path)
        if expected and value != expected:
            raise ValueError(f"Hash modificado: {path}")
        if str(path) in inputs and inputs[str(path)] != value:
            raise ValueError(f"Fuente cambió durante la lectura: {path}")
        inputs[str(path)] = value
        return value

    experiment_path = args.experiment / "manifest.json"
    checked(experiment_path)
    experiment = json_read(experiment_path)
    if experiment.get("status") != "complete":
        raise ValueError("Se requiere el contraste terminado")
    checked(args.experiment / "results.json", experiment["results_sha256"])
    candidate_report_path = args.cohort / "candidate_validation_report.json"
    checked(candidate_report_path, experiment["input_sha256"][str(candidate_report_path)])
    candidate_report = json_read(candidate_report_path)
    history_report_path = args.history / "history_validation_report.json"
    checked(history_report_path, candidate_report["input_sha256"][str(history_report_path)])
    history_report = json_read(history_report_path)
    for name in ("test_opportunities.csv", "historical_awards.csv"):
        checked(args.cohort / name, experiment["input_sha256"][str(args.cohort / name)])
    checked(args.cohort / "opportunity_availability.csv", candidate_report["output_sha256"]["opportunity_availability.csv"])
    checked(args.history / "award_availability.csv", history_report["output_sha256"]["award_availability.csv"])
    opportunities_csv = {r["opportunity_id"]: r for r in rows(args.cohort / "test_opportunities.csv")}
    history_csv = {r["award_id"]: r for r in rows(args.cohort / "historical_awards.csv")}
    opportunities, history = [], []
    for kind, refs in (("opportunities", rows(args.cohort / "opportunity_availability.csv")),
                       ("history", rows(args.history / "award_availability.csv"))):
        for ref in refs:
            identity = ref["ocid"] if kind == "opportunities" else ref["award_id"]
            table = opportunities_csv if kind == "opportunities" else history_csv
            if identity not in table:
                continue
            if ref["status"] != "verified":
                raise ValueError("Un registro retenido carece de fuente verificada")
            path = Path(ref["source_path"] if kind == "opportunities" else ref["text_source"])
            expected = ref["sha256"] if kind == "opportunities" else ref["text_sha256"]
            evidence = candidate_report if kind == "opportunities" else history_report
            if evidence["evidence_sha256"].get(str(path)) != expected:
                raise ValueError("Hash de fuente no registrado en el informe previo")
            checked(path, expected)
            record = features_from_source(path, ref["ocid"])
            if record["legacy_category"] != table[identity]["category"]:
                raise ValueError("La categoría archivada difiere de la fuente inicial")
            record.update({"source_path": str(path), "source_sha256": expected})
            if kind == "history":
                record.update({"award_id": identity, "supplier_id": ref["supplier_id"]})
            (opportunities if kind == "opportunities" else history).append(record)
        print(json.dumps({"phase": "source_audit", "population": kind,
                          "records": len(opportunities if kind == "opportunities" else history)}), flush=True)
    if len(opportunities) != len(opportunities_csv) or len(history) != len(history_csv):
        raise ValueError("No se recuperaron todas las fuentes de las entradas congeladas")
    catalog, aliases = inventory(history + opportunities)
    labels = defaultdict(Counter)
    for record in history + opportunities:
        if informative_label(record["legacy_category"]):
            labels[normalize_label(record["legacy_category"])][record["legacy_category"]] += 1
    normalization_groups = [{"normalized_label": key, "raw_labels": dict(sorted(values.items()))}
                            for key, values in sorted(labels.items()) if len(values) > 1]
    matches = compare_profile_matches(history, opportunities)
    sid, ocid = "PE-RUC-20604252122", "ocds-dgv273-seacev3-1150972"
    target = next((r for r in opportunities if r["ocid"] == ocid), None)
    target_history = [r for r in history if r["supplier_id"] == sid]
    case = {"selection": "previously_inspected_error_case_not_a_validation_sample", "target": target,
            "historical_evidence": target_history,
            "shared_codes": {s: sorted(code_keys(target, s) & set().union(*(code_keys(r, s) for r in target_history)))
                             for s in ("CUBSO", "UNSPSC")} if target and target_history else {}}
    summary = {"status": "complete", "labels_read": False, "rankings_run": False, "weights_tuned": False,
               "production_ranker_modified": False, "opportunities": coverage(opportunities), "history": coverage(history),
               "profiles": len(matches), "catalog_codes": len(catalog), "first_label_normalization_groups": len(normalization_groups),
               "codes_with_multiple_normalized_descriptions": sum(r["normalized_description_count"] > 1 for r in catalog),
               "normalized_labels_with_multiple_codes_within_scheme": len(aliases),
               "interpretation": "Category overlap counts are not relevance metrics, legal compatibility or evidence of improved ranking.",
               "any_item_scope_limit": "Tender-level history does not prove the supplier was awarded every listed item."}
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise ValueError(f"La entrada cambió durante la auditoría: {path}")
    args.output.mkdir(parents=True)
    documents = {"summary.json": summary, "catalog_inventory.json": catalog, "label_code_ambiguities.json": aliases,
                 "normalization_groups.json": normalization_groups, "profile_match_counts.json": matches, "case_evidence.json": case}
    for name, value in documents.items():
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    for name, records in (("opportunity_categories.jsonl", opportunities), ("history_categories.jsonl", history)):
        with (args.output / name).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    module_path = Path(__file__).resolve().parents[1] / "src/tesis_recommender/category_review.py"
    manifest = {"input_sha256": dict(sorted(inputs.items())), "input_hashes_unchanged": True,
                "code_sha256": {"scripts/audit_annual_categories.py": digest(__file__),
                                "src/tesis_recommender/category_review.py": digest(module_path)},
                "output_sha256": {name: digest(args.output / name) for name in sorted([*documents, "opportunity_categories.jsonl", "history_categories.jsonl"])},
                "labels_read": False, "rankings_run": False, "automatic_diagnostic_not_human_annotation": True}
    (args.output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=Path("artifacts/experiment_annual_v1"))
    parser.add_argument("--cohort", type=Path, default=Path("data/processed/contrast_candidates_v1/annual_v1"))
    parser.add_argument("--history", type=Path, default=Path("data/processed/contrast_history_validated_v1"))
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args())
