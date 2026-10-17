from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _allocate(counts: Counter[tuple[str, ...]], target: int) -> dict[tuple[str, ...], int]:
    total = sum(counts.values())
    raw = {key: target * value / total for key, value in counts.items()}
    quotas = {key: min(counts[key], math.floor(value)) for key, value in raw.items()}
    remaining = target - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - math.floor(raw[key])), key))
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] < counts[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break
    return quotas


def _history_band(count: int) -> str:
    return "2-3" if count <= 3 else "4-7" if count <= 7 else "8+"


def _mode(values: list[str], fallback: str = "SIN_DATO") -> str:
    normalized = [value.strip().upper() for value in values if value.strip()]
    return Counter(normalized).most_common(1)[0][0] if normalized else fallback


def _ocid_from_url(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def build_experimental_cohort(
    corpus_dir: str | Path,
    output_dir: str | Path,
    profile_target: int = 50,
    candidate_target: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    corpus_dir, output_dir = Path(corpus_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    awards = _read(corpus_dir / "historical_awards.csv")
    labels = _read(corpus_dir / "historical_labels.csv")
    test_opportunities = _read(corpus_dir / "test_opportunities.csv")
    lineage = {row["ocid"]: row for row in _read(corpus_dir / "lineage.csv")}

    awards_by_supplier: dict[str, list[dict[str, str]]] = defaultdict(list)
    labels_by_supplier: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in awards:
        awards_by_supplier[row["supplier_id"]].append(row)
    for row in labels:
        labels_by_supplier[row["supplier_id"]].append(row)

    eligible: list[dict[str, object]] = []
    for supplier_id, supplier_awards in awards_by_supplier.items():
        positives = labels_by_supplier.get(supplier_id, [])
        if len(supplier_awards) < 2 or not positives:
            continue
        amounts = [float(row["amount"]) for row in supplier_awards if row["amount"]]
        eligible.append({
            "supplier_id": supplier_id,
            "history_count": len(supplier_awards),
            "positive_2025_count": len({row["opportunity_id"] for row in positives}),
            "primary_region": _mode([row["region"] for row in supplier_awards]),
            "primary_category": _mode([row["category"] for row in supplier_awards]),
            "median_amount": statistics.median(amounts) if amounts else "",
            "history_band": _history_band(len(supplier_awards)),
        })
    if len(eligible) < profile_target:
        raise ValueError(f"Solo existen {len(eligible)} perfiles elegibles para una meta de {profile_target}")

    top_regions = {region for region, _ in Counter(str(row["primary_region"]) for row in eligible).most_common(8)}
    strata: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in eligible:
        row["region_group"] = row["primary_region"] if row["primary_region"] in top_regions else "OTRAS_REGIONES"
        strata[(str(row["region_group"]), str(row["history_band"]))].append(row)
    quotas = _allocate(Counter({key: len(rows) for key, rows in strata.items()}), profile_target)
    selected_profiles: list[dict[str, object]] = []
    for key, rows in strata.items():
        rows.sort(key=lambda row: _stable_key(str(row["supplier_id"]), seed))
        selected_profiles.extend(rows[:quotas.get(key, 0)])
    selected_profiles.sort(key=lambda row: str(row["supplier_id"]))
    selected_ids = {str(row["supplier_id"]) for row in selected_profiles}

    cohort_awards = [row for row in awards if row["supplier_id"] in selected_ids]
    cohort_labels = [row for row in labels if row["supplier_id"] in selected_ids]
    positive_ids = {row["opportunity_id"] for row in cohort_labels}
    test_by_id = {row["opportunity_id"]: row for row in test_opportunities}
    missing_positive_ids = positive_ids - set(test_by_id)
    if missing_positive_ids:
        raise ValueError(f"Etiquetas sin oportunidad de prueba: {len(missing_positive_ids)}")
    positive_candidates = [test_by_id[opportunity_id] for opportunity_id in sorted(positive_ids)]
    if len(positive_candidates) > candidate_target:
        raise ValueError("La meta de candidatos es menor que el conjunto de positivos obligatorio")

    remaining = [row for row in test_opportunities if row["opportunity_id"] not in positive_ids]
    candidate_regions = {region for region, _ in Counter(row["region"].strip().upper() for row in remaining).most_common(8)}
    candidate_strata: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in remaining:
        month = int(row["publication_date"][5:7])
        quarter = f"Q{(month - 1) // 3 + 1}"
        region = row["region"].strip().upper()
        region_group = region if region in candidate_regions else "OTRAS_REGIONES"
        candidate_strata[(region_group, quarter)].append(row)
    needed = candidate_target - len(positive_candidates)
    candidate_quotas = _allocate(Counter({key: len(rows) for key, rows in candidate_strata.items()}), needed)
    sampled_candidates: list[dict[str, str]] = []
    for key, rows in candidate_strata.items():
        rows.sort(key=lambda row: _stable_key(row["opportunity_id"], seed))
        sampled_candidates.extend(rows[:candidate_quotas.get(key, 0)])
    cohort_candidates = positive_candidates + sampled_candidates
    cohort_candidates.sort(key=lambda row: row["opportunity_id"])

    roles_by_ocid: dict[str, set[str]] = defaultdict(set)
    for row in cohort_awards:
        roles_by_ocid[_ocid_from_url(row["source_url"])].add("historical_profile")
    for row in cohort_candidates:
        roles_by_ocid[row["opportunity_id"]].add("test_candidate")
    queue = []
    for ocid, roles in sorted(roles_by_ocid.items()):
        trace = lineage.get(ocid)
        if not trace or not trace.get("initial_release_url"):
            raise ValueError(f"OCID sin primera entrega trazable: {ocid}")
        initial_release_url = trace["initial_release_url"].replace(
            "https://contratacionesabiertas.osce.gob.pe/",
            "https://contratacionesabiertas.oece.gob.pe/",
        )
        cache_key = hashlib.sha256(initial_release_url.encode()).hexdigest()
        queue.append({
            "ocid": ocid, "role": "|".join(sorted(roles)), "initial_release_url": initial_release_url,
            "initial_release_date": trace["initial_release_date"], "cache_key": cache_key,
        })

    profile_fields = ["supplier_id", "history_count", "positive_2025_count", "primary_region", "region_group", "primary_category", "median_amount", "history_band"]
    _write(output_dir / "cohort_profiles.csv", selected_profiles, profile_fields)
    _write(output_dir / "cohort_historical_awards.csv", cohort_awards, list(cohort_awards[0]))
    _write(output_dir / "cohort_labels.csv", cohort_labels, list(cohort_labels[0]))
    _write(output_dir / "cohort_test_opportunities_compiled.csv", cohort_candidates, list(cohort_candidates[0]))
    _write(output_dir / "initial_release_queue.csv", queue, ["ocid", "role", "initial_release_url", "initial_release_date", "cache_key"])

    selected_band_counts = Counter(str(row["history_band"]) for row in selected_profiles)
    selected_region_counts = Counter(str(row["region_group"]) for row in selected_profiles)
    report: dict[str, object] = {
        "seed": seed, "profile_target": profile_target, "candidate_target": candidate_target,
        "eligibility_rule": "historical_awards_before_2025 >= 2 and observed_positive_in_2025 >= 1",
        "eligible_profiles": len(eligible), "selected_profiles": len(selected_profiles),
        "selected_historical_awards": len(cohort_awards), "selected_positive_labels": len(cohort_labels),
        "unique_positive_opportunities": len(positive_ids), "candidate_opportunities": len(cohort_candidates),
        "initial_releases_required": len(queue), "top_region_groups": sorted(top_regions),
        "selected_history_bands": dict(selected_band_counts), "selected_region_groups": dict(selected_region_counts),
        "sampling": "proportional largest-remainder by region group/history band; stable SHA-256 order",
    }
    (output_dir / "cohort_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
