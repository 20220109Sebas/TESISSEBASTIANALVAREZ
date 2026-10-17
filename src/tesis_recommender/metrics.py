from __future__ import annotations

import math


def evaluate_rankings(
    rankings: dict[str, list[str]], relevant: dict[str, set[str]], ks: tuple[int, ...] = (5, 10)
) -> dict[str, float]:
    """Macro-promedios por perfil; Precision@k usa denominador k aun con listas cortas.

    Los positivos históricos son evidencia observada, no juicios exhaustivos de
    compatibilidad. F1@k se calcula por perfil antes de promediar.
    """
    if any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in ks):
        raise ValueError("Los valores de k deben ser enteros positivos")
    if any(len(items) != len(set(items)) for items in rankings.values()):
        raise ValueError("Un ranking no puede contener oportunidades duplicadas")
    query_ids = [query_id for query_id in rankings if relevant.get(query_id)]
    if not query_ids:
        return {**{f"{metric}@{k}": 0.0 for k in ks for metric in ("precision", "recall", "f1", "hitrate")}, "mrr": 0.0, "coverage": 0.0}
    output: dict[str, float] = {}
    for k in ks:
        precisions, recalls, f1s, hits = [], [], [], []
        for query_id in query_ids:
            retrieved = set(rankings[query_id][:k])
            truth = relevant[query_id]
            found = len(retrieved & truth)
            precision, recall = found / k, found / len(truth)
            precisions.append(precision)
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
            recalls.append(found / len(truth))
            hits.append(float(found > 0))
        output[f"recall@{k}"] = sum(recalls) / len(recalls)
        output[f"precision@{k}"] = sum(precisions) / len(precisions)
        output[f"f1@{k}"] = sum(f1s) / len(f1s)
        output[f"hitrate@{k}"] = sum(hits) / len(hits)
    reciprocal_ranks = []
    for query_id in query_ids:
        truth = relevant[query_id]
        reciprocal_ranks.append(next((1 / rank for rank, item in enumerate(rankings[query_id], 1) if item in truth), 0.0))
    output["mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    output["coverage"] = len(query_ids) / len(relevant) if relevant else 0.0
    return {name: round(value, 6) for name, value in output.items()}


def extraction_metrics(
    truth: list[dict[str, str]], predicted: list[dict[str, str]], fields: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Exact-match por campo para una muestra anotada y alineada por fila."""
    if len(truth) != len(predicted):
        raise ValueError("Verdad y predicción deben tener la misma cantidad de filas")
    result: dict[str, dict[str, float]] = {}
    for field in fields:
        tp = fp = fn = 0
        for expected, observed in zip(truth, predicted):
            gold = (expected.get(field) or "").strip().lower()
            guess = (observed.get(field) or "").strip().lower()
            tp += int(bool(gold) and guess == gold)
            fp += int(bool(guess) and guess != gold)
            fn += int(bool(gold) and guess != gold)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[field] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "support": tp + fn}
    return result


def ndcg_at_k(ranking: list[str], graded_relevance: dict[str, int], k: int = 5) -> float:
    """Métrica opcional cuando existan juicios ordinales 0/1/2."""
    def dcg(items: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(position + 2) for position, grade in enumerate(items))

    observed = [graded_relevance.get(item, 0) for item in ranking[:k]]
    ideal = sorted(graded_relevance.values(), reverse=True)[:k]
    denominator = dcg(ideal)
    return round(dcg(observed) / denominator, 6) if denominator else 0.0
