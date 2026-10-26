"""Recalcula las tablas del artículo con la biblioteca estándar de Python."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent
METRICS = [f"{name}@{k}" for k in (5, 10) for name in ("precision", "recall", "f1")]

def read_csv(name):
    with (BASE / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))

def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)

def interval(values, samples):
    means = [sum(values[i] for i in sample) / len(sample) for sample in samples]
    return {"mean": round(sum(values) / len(values), 6),
            "ci95_low": round(percentile(means, 0.025), 6),
            "ci95_high": round(percentile(means, 0.975), 6)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("La carpeta de salida debe ser nueva")
    integrity = json.loads((BASE / "sha256.json").read_text(encoding="utf-8"))
    for name, expected_hash in integrity.items():
        actual = hashlib.sha256((BASE / name).read_bytes()).hexdigest()
        if actual != expected_hash:
            raise ValueError(f"Integridad incorrecta: {name}")
    per_model = {}
    for row in read_csv("recuperacion_por_perfil.csv"):
        model, profile = row["modelo"], row["perfil"]
        values = {}
        positives = int(row["positivos"])
        for k in (5, 10):
            hits = int(row[f"aciertos_{k}"])
            if not 0 <= hits <= min(k, positives):
                raise ValueError("Aciertos fuera del rango admisible")
            precision, recall = hits / k, hits / positives
            values.update({f"precision@{k}": precision, f"recall@{k}": recall,
                           f"f1@{k}": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
        if profile in per_model.setdefault(model, {}):
            raise ValueError("Perfil duplicado")
        per_model[model][profile] = values
    profiles = sorted(per_model["bm25"])
    if len(profiles) != 13 or any(sorted(v) != profiles for v in per_model.values()):
        raise ValueError("Los modelos requieren los mismos 13 perfiles")
    means = {m: {k: round(sum(v[p][k] for p in profiles) / len(profiles), 6) for k in METRICS}
             for m, v in per_model.items()}
    expected = json.loads((BASE / "metricas_esperadas.json").read_text(encoding="utf-8"))
    if means != expected:
        raise ValueError("Las métricas recalculadas difieren de las archivadas")
    rng = random.Random(42)
    samples = [[rng.randrange(len(profiles)) for _ in profiles] for _ in range(5000)]
    contrasts = []
    records = read_csv("contrastes_originales.csv") + json.loads((BASE / "contrastes_agregacion.json").read_text(encoding="utf-8"))
    for record in records:
        left, right, metric = record["left"], record["right"], record["metric"]
        result = interval([per_model[left][p][metric] - per_model[right][p][metric] for p in profiles], samples)
        if any(abs(result[k] - float(record[k])) > 1e-6 for k in result):
            raise ValueError(f"Contraste distinto: {left}/{right}/{metric}")
        contrasts.append({"left": left, "right": right, "metric": metric, **result})
    documentary = []
    for row in read_csv("extraccion_documental.csv"):
        tp, fp, fn = (int(row[k]) for k in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        documentary.append({**row, "precision": precision, "recall": recall, "f1": f1})
    gains = {}
    for left, right in [("bm25_labse", "bm25"), ("hybrid_full", "bm25_labse")]:
        differences = [per_model[left][p]["f1@10"] - per_model[right][p]["f1@10"] for p in profiles]
        gains[f"{left}_vs_{right}"] = {"mejor": sum(d > 1e-12 for d in differences),
                                              "igual": sum(abs(d) <= 1e-12 for d in differences),
                                              "peor": sum(d < -1e-12 for d in differences)}
    args.output.mkdir(parents=True)
    report = {"integridad": "correcta", "tablas_recuperacion": means,
              "contrastes": contrasts, "tabla_documental": documentary, "comparacion_por_perfil": gains}
    (args.output / "resultados_recalculados.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for filename, models in [("tabla_1.csv", list(per_model)[:4]), ("tabla_2.csv", list(per_model)[4:])]:
        with (args.output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["modelo", *METRICS])
            writer.writeheader()
            writer.writerows({"modelo": m, **means[m]} for m in models)
    with (args.output / "tabla_3.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(documentary[0]))
        writer.writeheader()
        writer.writerows(documentary)
    print("Integridad correcta. Tablas 1, 2 y 3 e intervalos pareados reproducidos.")
    print(f"Salidas: {args.output}")

if __name__ == "__main__":
    main()
