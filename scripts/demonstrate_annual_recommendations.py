"""Replay one audited profile from frozen signals, without loading LaBSE or labels."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from datetime import date, datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ("bm25", "semantic", "category", "region", "amount")
MODEL_NAMES = ("bm25", "labse", "bm25_labse", "hybrid_full")
NAMES = {"bm25": "BM25", "labse": "LaBSE", "bm25_labse": "BM25 + LaBSE",
         "hybrid_full": "Híbrido con atributos"}
SIGNAL_NAMES = {"bm25": "Léxica", "semantic": "Semántica", "category": "Categoría",
                "region": "Región", "amount": "Monto"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked(path, expected, evidence):
    if sha(path) != expected:
        raise ValueError(f"Archivo modificado: {path}")
    evidence[str(path)] = expected
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def default_supplier(suppliers):
    return min(suppliers, key=lambda s: (hashlib.sha256(("42:" + s).encode()).hexdigest(), s))


def replay_scores(order, components, weights):
    if not order or len(set(order)) != len(order):
        raise ValueError("Colección vacía o identificadores duplicados")
    if tuple(components) != SIGNALS or set(weights) != set(SIGNALS):
        raise ValueError("Señales distintas del contrato original")
    if any(not math.isfinite(w) or w < 0 for w in weights.values()) or not math.isclose(sum(weights.values()), 1, abs_tol=1e-6):
        raise ValueError("Pesos inválidos")
    for name, values in components.items():
        if len(values) != len(order) or any(not math.isfinite(float(v)) for v in values):
            raise ValueError(f"Componentes inválidos: {name}")
    scores = [sum(weights[name] * float(components[name][i]) for name in SIGNALS)
              for i in range(len(order))]
    ranked = sorted(range(len(order)), key=lambda i: (-scores[i], order[i]))
    return scores, ranked


def verify_ranking(path, supplier, model, order, scores, ranked):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        count = 0
        for count, row in enumerate(csv.DictReader(handle), 1):
            if count > len(ranked):
                raise ValueError("Ranking con filas adicionales")
            i = ranked[count - 1]
            if (row["supplier_id"] != supplier or row["model"] != model
                    or int(row["rank"]) != count or row["opportunity_id"] != order[i]
                    or float(row["score"]) != scores[i]):
                raise ValueError(f"Replay distinto del ranking archivado, fila {count}")
        if count != len(order):
            raise ValueError("Ranking incompleto")
    return count


def source_details(path, expected, ocid, evidence):
    checked(path, expected, evidence)
    package = read_json(path)
    releases = package.get("releases", [])
    if len(releases) != 1 or releases[0].get("ocid") != ocid:
        raise ValueError("Identidad de fuente no verificable")
    release = releases[0]
    uri = package.get("uri", "")
    if not uri.startswith("https://contratacionesabiertas.oece.gob.pe/api/v1/release/"):
        raise ValueError("Falta URI de versión específica")
    return {"ocid": ocid, "release_id": release["id"], "published_at": release["publishedDate"],
            "uri": uri, "path": str(path), "sha256": expected}


def present(packet):
    p = packet["profile"]
    lines = ["# Demostración reproducible del recomendador", "",
             f"Perfil: **{p['supplier_id']}**. Corte histórico: {packet['cutoff']}.", "",
             f"Antecedentes auditados: {len(p['history'])}. Colección: {packet['candidate_count']} oportunidades de 2025.", "",
             "Esta demostración recalcula los cuatro rankings a partir de señales congeladas y comprueba su igualdad exacta con los archivos verificados. No vuelve a generar embeddings, no carga el Transformer y no lee etiquetas para recomendar.", "",
             "Las oportunidades son históricas. No se afirma que estén abiertas actualmente. El puntaje sirve para ordenar y no es una probabilidad de adjudicación.", "", "## Historial del perfil", ""]
    for item in p["history"]:
        lines += [f"- {item['award_date']}: {item['description']}",
                  f"  Fuente del texto: [{item['ocid']}]({item['text_source']['uri']})."]
    for model, run in packet["runs"].items():
        lines += ["", f"## {NAMES[model]}", "",
                  f"Se comprobaron exactamente {run['rows_verified']} posiciones. Pesos originales conservados.", ""]
        for item in run["recommendations"]:
            lines += [f"### Puesto {item['rank']}. {item['opportunity']['description']}", "",
                      f"Identificador: {item['opportunity_id']}. Puntaje: {item['score']:.6f}.", "",
                      f"Publicado por el proceso: {item['opportunity']['publication_date']}. Versión auditada: {item['source']['published_at']}.", "",
                      "| Señal | Valor normalizado | Peso | Aporte |", "|---|---:|---:|---:|"]
            for name in SIGNALS:
                lines.append(f"| {SIGNAL_NAMES[name]} | {item['signals'][name]:.6f} | {run['weights'][name]:.3f} | {item['contributions'][name]:.6f} |")
            lines += ["", f"Mayor aporte: {SIGNAL_NAMES[item['dominant_signal']]}. La explicación describe contribuciones numéricas, no acredita causalidad ni cumplimiento legal.", ""]
            for warning in item["warnings"]:
                lines.append(f"- {warning}")
            lines += ["", f"[Abrir la versión exacta de la fuente]({item['source']['uri']})", "",
                      f"Archivo: {item['source']['path']}", "", f"SHA-256: {item['source']['sha256']}", ""]
    lines += ["## Límites de la demostración", "",
              "Región es la dirección de la entidad, no un lugar de ejecución confirmado. Un monto cero o una moneda no comparable no aporta evidencia monetaria. Un aporte estructurado cero no demuestra incompatibilidad. Experiencia no está implementada como campo de extracción. No se selecciona un modelo ganador ni se modifica el predeterminado.", "",
              "La elección automática del perfil usa SHA-256 de 42:identificador, sin consultar sus positivos ni métricas. Se puede elegir explícitamente cualquiera de los 49 perfiles. El replay verifica composición y orden sobre el caché existente, no una nueva codificación ni generalización.", ""]
    return "\n".join(lines)


def build(experiment, supplier, top_k, output):
    if top_k not in (5, 10):
        raise ValueError("La demostración usa top 5 o top 10")
    if output.exists():
        raise ValueError("Usa una salida nueva, no se sobrescriben demostraciones")
    evidence = {}
    manifest_path = experiment / "manifest.json"
    verification_path = experiment / "verification.json"
    verification = read_json(verification_path)
    checked(manifest_path, verification["manifest_sha256"], evidence)
    if not verification.get("verified") or not verification.get("exact_weighted_scores_verified"):
        raise ValueError("Experimento sin verificación final")
    manifest = read_json(manifest_path)
    if manifest["status"] != "complete" or tuple(manifest["models"]) != MODEL_NAMES:
        raise ValueError("Experimento incompleto o modelos no originales")
    # Bind the verifier report to its archived code and result hashes as well.
    checked(experiment / "verification.code.py", verification["verification_code_sha256"], evidence)
    checked(experiment / "results.json", manifest["results_sha256"], evidence)
    evidence[str(verification_path)] = sha(verification_path)
    cohort = ROOT / "data/processed/contrast_candidates_v1/annual_v1"
    for name in ("historical_awards.csv", "test_opportunities.csv", "candidate_validation_report.json"):
        path = cohort / name
        checked(path, manifest["input_sha256"][str(path.relative_to(ROOT))], evidence)
    report = read_json(cohort / "candidate_validation_report.json")
    checked(cohort / "opportunity_availability.csv", report["output_sha256"]["opportunity_availability.csv"], evidence)
    history_dir = ROOT / "data/processed/contrast_history_validated_v1"
    history_report_path = history_dir / "history_validation_report.json"
    checked(history_report_path, report["input_sha256"][str(history_report_path.relative_to(ROOT))], evidence)
    history_report = read_json(history_report_path)
    checked(history_dir / "award_availability.csv", history_report["output_sha256"]["award_availability.csv"], evidence)
    results = read_json(experiment / "results.json")
    cutoff = results["cutoff"]
    awards = read_csv(cohort / "historical_awards.csv")
    suppliers = sorted({r["supplier_id"] for r in awards})
    supplier = supplier or default_supplier(suppliers)
    if supplier not in suppliers:
        raise ValueError("Perfil no pertenece a los 49 perfiles auditados")
    selected = sorted((r for r in awards if r["supplier_id"] == supplier), key=lambda r: r["award_id"])
    if len(selected) < 2 or any(r["award_date"] >= cutoff for r in selected):
        raise ValueError("Historial insuficiente o posterior al corte")
    opportunities = read_csv(cohort / "test_opportunities.csv")
    if any(r.get("awarded_supplier_id") or r["object_type"] != "services" for r in opportunities):
        raise ValueError("Las entradas contienen desenlaces o no son servicios")
    by_id = {r["opportunity_id"]: r for r in opportunities}
    checked(experiment / "candidate_order.json", manifest["candidate_order_sha256"], evidence)
    order = read_json(experiment / "candidate_order.json")
    if len(by_id) != len(opportunities) or len(order) != len(opportunities) or set(order) != set(by_id):
        raise ValueError("La colección no coincide con el orden congelado")
    available = {r["ocid"]: r for r in read_csv(cohort / "opportunity_availability.csv")}
    history_evidence = {r["award_id"]: r for r in read_csv(history_dir / "award_availability.csv")}
    history = []
    cutoff_time = datetime.combine(date.fromisoformat(cutoff), datetime.min.time(), ZoneInfo("America/Lima"))
    for row in selected:
        a = history_evidence[row["award_id"]]
        if a["supplier_id"] != supplier or a["status"] != "verified":
            raise ValueError("Historial no auditado")
        item = {k: row[k] for k in ("award_id", "award_date", "description", "category", "region", "amount", "currency")}
        item["ocid"] = a["ocid"]
        for role in ("text", "award"):
            source = source_details(ROOT / a[f"{role}_source"], a[f"{role}_sha256"], a["ocid"], evidence)
            if source["published_at"] != a[f"{role}_published_at"] or datetime.fromisoformat(source["published_at"]) >= cutoff_time:
                raise ValueError("Fuente histórica fuera del corte")
            item[f"{role}_source"] = source
        history.append(item)
    import numpy as np
    component_path = experiment / "components" / f"{supplier}.npz"
    checked(component_path, manifest["component_sha256"][f"components/{supplier}.npz"], evidence)
    with np.load(component_path, allow_pickle=False) as archive:
        components = {name: archive[name] for name in archive.files}
    packet = {"kind": "frozen_signal_replay_not_new_encoding_or_new_experiment", "cutoff": cutoff,
              "candidate_count": len(order), "top_k": top_k,
              "profile_selection": "explicit_supplier" if supplier != default_supplier(suppliers) else "fixed_hash_seed_42_or_explicit_same_supplier",
              "profile": {"supplier_id": supplier, "history": history},
              "model": {"name": manifest["model_name"], "revision": manifest["model_revision"]},
              "labels_read": False, "embedding_model_loaded": False, "default_model_changed": False, "runs": {}}
    total = 0
    for model, weights in manifest["models"].items():
        scores, ranked = replay_scores(order, components, weights)
        ranking_path = experiment / "rankings" / f"{supplier}__{model}.csv.gz"
        checked(ranking_path, manifest["ranking_sha256"][f"rankings/{supplier}__{model}.csv.gz"], evidence)
        count = verify_ranking(ranking_path, supplier, model, order, scores, ranked)
        total += count
        recommended = []
        archived = results["runs"][model]["top_k"][supplier]
        for rank, i in enumerate(ranked[:top_k], 1):
            oid = order[i]
            row, availability = by_id[oid], available[oid]
            if availability["status"] != "verified":
                raise ValueError("Candidato no auditado")
            source = source_details(ROOT / availability["source_path"], availability["sha256"], oid, evidence)
            if source["published_at"] != availability["text_published_at"]:
                raise ValueError("Fecha de fuente distinta de la auditada")
            signals = {name: float(components[name][i]) for name in SIGNALS}
            contributions = {name: signals[name] * weights[name] for name in SIGNALS}
            if archived[rank - 1]["opportunity_id"] != oid or archived[rank - 1]["score"] != round(scores[i], 6):
                raise ValueError("Top explicado no coincide con el replay")
            for name in SIGNALS:
                exp = archived[rank - 1]["explanation"]
                if exp[f"{name}_score"] != round(signals[name], 6) or exp[f"{name}_contribution"] != round(contributions[name], 6):
                    raise ValueError("Explicación archivada distinta")
            warnings = ["Colección histórica estática. No acredita vigencia actual ni elegibilidad legal.",
                        "Región representa la dirección de la entidad, no el lugar de ejecución verificado.",
                        "Experiencia no implementada como campo de extracción."]
            amount_status = archived[rank - 1]["explanation"]["amount_status"]
            if amount_status != "observed_same_currency":
                warnings.append("Monto no disponible o no comparable. No se completa con información posterior.")
            if row["amount"] and float(row["amount"]) == 0:
                warnings.append("Monto inicial publicado como cero.")
            recommended.append({"rank": rank, "opportunity_id": oid, "score": scores[i], "signals": signals,
                                "contributions": contributions, "dominant_signal": max(contributions, key=contributions.get),
                                "amount_status": amount_status,
                                "opportunity": {k: row[k] for k in ("description", "category", "region", "publication_date", "amount", "currency")},
                                "source": source, "warnings": warnings})
        packet["runs"][model] = {"weights": weights, "rows_verified": count, "recommendations": recommended}
    for path, expected in evidence.items():
        if sha(path) != expected:
            raise ValueError("Una entrada cambió durante la demostración")
    output.mkdir(parents=True)
    (output / "recommendations.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "LEEME_DEMOSTRACION.md").write_text(present(packet), encoding="utf-8")
    demo_manifest = {"purpose": packet["kind"], "input_sha256": evidence,
                     "script_sha256": sha(Path(__file__)), "supplier_id": supplier,
                     "top_k": top_k, "labels_read": False, "full_ranking_rows_replayed": total,
                     "weights_unchanged": True, "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}
    (output / "manifest.json").write_text(json.dumps(demo_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "supplier_id": supplier, "history_count": len(history),
            "rows_verified": total, "recommendations": 4 * top_k, "labels_read": False,
            "embedding_model_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "artifacts/experiment_annual_v1")
    parser.add_argument("--supplier-id")
    parser.add_argument("--top-k", type=int, choices=(5, 10), default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "output/seminario_demo_v1")
    args = parser.parse_args()
    print(json.dumps(build(args.experiment, args.supplier_id, args.top_k, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
