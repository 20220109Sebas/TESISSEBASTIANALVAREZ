"""Evidencia automática de cambios de ranking, separada del juicio humano."""
from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from .hybrid import HybridRanker
from .io import read_awards, read_opportunities, read_relevance_labels
from .profiles import build_profiles
from .semantic import PrecomputedEmbedder
from .validation import read_csv, write_csv, sha256


REVIEW_FIELDS = ["human_judgement", "reason", "evidence", "reviewer", "reviewed_at"]


def build_error_dossier(experiment_dir: Path, cohort_dir: Path, cache_dir: Path,
                        output_dir: Path) -> dict:
    review_path = output_dir / "human_reviews.csv"
    if review_path.exists() and any(any(row.get(f, "").strip() for f in REVIEW_FIELDS)
                                   for row in read_csv(review_path)):
        raise ValueError("Hay revisiones humanas. Usar otra carpeta para no sobrescribirlas")
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    archived = json.loads((experiment_dir / "results.json").read_text(encoding="utf-8"))
    paths = [cohort_dir / "historical_awards.csv", cohort_dir / "test_opportunities.csv",
             cohort_dir / "labels.csv", cache_dir / "metadata.json", cache_dir / "index.json",
             cache_dir / "embeddings.npy"]
    for path in paths:
        if manifest["input_sha256"].get(str(path)) != sha256(path):
            raise ValueError(f"La entrada no corresponde al experimento archivado: {path}")
    awards = read_awards(paths[0])
    opportunities = read_opportunities(paths[1])
    profiles = build_profiles(awards, date.fromisoformat(manifest["cutoff"]))
    truth = read_relevance_labels(paths[2])
    ranker = HybridRanker(opportunities, PrecomputedEmbedder(cache_dir))
    names = ("bm25_labse", "hybrid_full")
    full = {}
    for name in names:
        full[name] = {}
        if set(archived["runs"][name]["top_k"]) != set(profiles):
            raise ValueError("Los perfiles no corresponden a la corrida archivada")
        for supplier, profile in sorted(profiles.items()):
            ranking = ranker.rank(profile, manifest["ablations"][name], top_k=None)
            expected = archived["runs"][name]["top_k"][supplier]
            actual = [(r.opportunity_id, round(r.score, 6)) for r in ranking[:len(expected)]]
            if actual != [(r["opportunity_id"], r["score"]) for r in expected]:
                raise ValueError(f"No se reproduce el top-k archivado: {name}/{supplier}")
            full[name][supplier] = ranking
    # Diagnóstico mecánico. Quitar una señal y reescalar los pesos restantes
    # conserva el orden del score sin esa señal. No ajusta pesos al desenlace.
    diagnostic_weights = {}
    diagnostic_ranks = {}
    for removed in ("category", "region", "amount"):
        weights = {key: (0.0 if key == removed else value)
                   for key, value in manifest["ablations"]["hybrid_full"].items()}
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("El diagnóstico requiere al menos una señal restante")
        weights = {key: value / total for key, value in weights.items()}
        diagnostic_weights[removed] = weights
        diagnostic_ranks[removed] = {
            supplier: {r.opportunity_id: i + 1 for i, r in enumerate(ranker.rank(profile, weights, top_k=None))}
            for supplier, profile in sorted(profiles.items())}
    by_id = {o.opportunity_id: o for o in opportunities}
    cases, review_rows, case_rows = [], [], []
    changed_profiles = set()
    for supplier, profile in sorted(profiles.items()):
        left, right = (full[n][supplier] for n in names)
        left_map = {r.opportunity_id: (i + 1, r) for i, r in enumerate(left)}
        right_map = {r.opportunity_id: (i + 1, r) for i, r in enumerate(right)}
        left10 = {r.opportunity_id for r in left[:10]}
        right10 = {r.opportunity_id for r in right[:10]}
        changed = left10 ^ right10
        if changed:
            changed_profiles.add(supplier)
        for opportunity_id in sorted(changed):
            o = by_id[opportunity_id]
            left_rank, left_item = left_map[opportunity_id]
            right_rank, right_item = right_map[opportunity_id]
            observed = opportunity_id in truth.get(supplier, set())
            change = "entered_top10" if opportunity_id in right10 else "left_top10"
            case_id = f"{supplier}:{opportunity_id}:{change}"
            row = {"case_id": case_id, "supplier_id": supplier, "opportunity_id": opportunity_id,
                   "change_type": change, "label_status": "observed_positive" if observed else "unlabelled_not_negative",
                   "lexical_semantic_rank": left_rank, "hybrid_rank": right_rank,
                   "lexical_semantic_score": round(left_item.score, 6), "hybrid_score": round(right_item.score, 6),
                   **{f"{s}_contribution": right_item.explanation[f"{s}_contribution"]
                      for s in ("bm25", "semantic", "category", "region", "amount")},
                   "category_status": right_item.explanation["category_status"],
                   "region_status": right_item.explanation["region_status"],
                   "amount_status": right_item.explanation["amount_status"], "source_url": o.source_url}
            row.update({f"rank_without_{s}": diagnostic_ranks[s][supplier][opportunity_id]
                        for s in ("category", "region", "amount")})
            case_rows.append(row)
            cases.append({**row, "opportunity": {"title": o.title, "description": o.description,
                          "technical_text": o.technical_text, "category": o.category, "region": o.region,
                          "amount": o.amount, "currency": o.currency, "publication_date": o.publication_date.isoformat()},
                          "profile": {"history_count": len(profile.awards), "categories": sorted(profile.categories),
                                      "buyer_regions": sorted(profile.regions),
                                      "median_amount_same_currency": profile.median_amount_in(o.currency),
                                      "history": [{"award_id": a.award_id, "date": a.award_date.isoformat(),
                                                   "description": a.description, "technical_text": a.technical_text,
                                                   "source_url": a.source_url} for a in profile.awards]},
                          "lexical_semantic_explanation": left_item.explanation,
                          "hybrid_explanation": right_item.explanation})
            review_rows.append({"case_id": case_id, "supplier_id": supplier, "opportunity_id": opportunity_id,
                                "change_type": change, **{field: "" for field in REVIEW_FIELDS}})
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "supplier_id", "opportunity_id", "change_type", "label_status",
              "lexical_semantic_rank", "hybrid_rank", "lexical_semantic_score", "hybrid_score",
              "bm25_contribution", "semantic_contribution", "category_contribution", "region_contribution",
              "amount_contribution", "category_status", "region_status", "amount_status", "source_url"]
    fields.extend(f"rank_without_{s}" for s in ("category", "region", "amount"))
    write_csv(output_dir / "rank_changes.csv", case_rows, fields)
    write_csv(review_path, review_rows, ["case_id", "supplier_id", "opportunity_id", "change_type", *REVIEW_FIELDS])
    (output_dir / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = Counter(f"{r['change_type']}:{r['label_status']}" for r in case_rows)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "profiles": len(profiles),
              "profiles_with_top10_changes": len(changed_profiles), "cases": len(cases), "counts": dict(counts),
              "archived_top10_reproduced": True, "human_reviews_completed": 0,
              "diagnostic_weights": diagnostic_weights,
              "diagnostic_note": "Signal removal is a fixed-score rank sensitivity check. It is not a new tuned model or a human compatibility judgement.",
              "evaluation_status": archived.get("evaluation_status", "development_not_independent"),
              "inputs_sha256": {str(p): sha256(p) for p in [*paths, experiment_dir / "manifest.json", experiment_dir / "results.json"]},
              "interpretation_limit": "A rank change is an automatic observation, not a judgement of compatibility. Unlabelled opportunities are not negatives. Scores use different weight allocations and must not be directly interpreted as improvement."}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
