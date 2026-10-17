"""Automatic contract-to-source evaluation of the five fields in the research sheet.

Independent reference rules, not imported from the production extractor. No
human judgments, supplier outcomes, model selection or retrieval labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

FIELDS = ("object", "category", "region", "amount", "experience")
FIELD_NAMES = {"object": "Objeto del servicio", "category": "Categoría",
               "region": "Región de la entidad", "amount": "Monto publicado",
               "experience": "Experiencia identificable"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm(value, field):
    if value is None or str(value).strip() == "":
        return None
    if field == "amount":
        try:
            number = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError("Monto no numérico") from exc
        if not number.is_finite():
            raise ValueError("Monto no finito")
        return number
    return " ".join(str(value).split()).casefold()


def source_reference(release, field):
    tender = release.get("tender") or {}
    items = tender.get("items") or []
    if field == "object":
        return tender.get("description"), "tender.description"
    if field == "category":
        classification = (items[0].get("classification") or {}) if items else {}
        for key in ("description", "id"):
            if classification.get(key):
                return classification[key], "tender.items[0].classification." + key
        return tender.get("mainProcurementCategory"), "tender.mainProcurementCategory"
    if field == "amount":
        return (tender.get("value") or {}).get("amount"), "tender.value.amount"
    if field == "region":
        references = [release.get("buyer") or {}, tender.get("procuringEntity") or {}]
        ids = {r["id"] for r in references if r.get("id")}
        parties = [(i, p) for i, p in enumerate(release.get("parties") or []) if p.get("id") in ids]
        if not parties:
            parties = [(i, p) for i, p in enumerate(release.get("parties") or [])
                       if set(p.get("roles") or []) & {"buyer", "procuringEntity"}]
        found = []
        for i, party in parties:
            for key in ("department", "region", "locality"):
                value = (party.get("address") or {}).get(key)
                if value:
                    found.append((value, f"parties[{i}].address.{key}"))
                    break
        if len({norm(value, field) for value, _ in found}) > 1:
            raise ValueError("Direcciones de entidades contradictorias")
        return found[0] if found else (None, "parties[buyer/procuringEntity].address")
    if field == "experience":
        # Inspect the available textual scope, never an unvisited PDF or later release.
        pieces = [("tender.title", tender.get("title", "")),
                  ("tender.description", tender.get("description", ""))]
        for i, item in enumerate(items):
            pieces.append((f"tender.items[{i}].description", item.get("description", "")))
            pieces.append((f"tender.items[{i}].classification.description",
                           (item.get("classification") or {}).get("description", "")))
            pieces.extend((f"tender.items[{i}].additionalClassifications[{j}].description", c.get("description", ""))
                          for j, c in enumerate(item.get("additionalClassifications") or []))
        mentions = [{"path": p, "text": v} for p, v in pieces if "experiencia" in str(v).casefold()]
        return mentions, "available_text_scope_not_full_technical_documents"
    raise ValueError("Campo desconocido")


def score_field(rows):
    usable = [r for r in rows if r["reference_state"] in {"observed", "absent"}]
    tp = fp = fn = tn = 0
    for row in usable:
        expected, predicted = norm(row["reference_value"], row["field"]), norm(row["prediction"], row["field"])
        tp += int(expected is not None and expected == predicted)
        fp += int(predicted is not None and expected != predicted)
        fn += int(expected is not None and expected != predicted)
        tn += int(expected is None and predicted is None)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    # No observed reference support does not demonstrate extraction effectiveness.
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn and 2 * tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "support": tp + fn,
            "precision": precision, "recall": recall, "f1": f1,
            "evaluated": len(usable), "non_evaluable": len(rows) - len(usable),
            "agreement": sum(r["status"] == "match" for r in usable) / len(usable) if usable else None,
            "predicted_nonempty": sum(norm(r["prediction"], r["field"]) is not None for r in rows),
            "reference_state_counts": dict(Counter(r["reference_state"] for r in rows)),
            "metric_reference": "automatic_source_contract_not_human_gold"}


def audit(cohort, experiment, output, size=50, seed=42):
    if output.exists():
        raise ValueError("No sobrescribir una auditoría. Usar una salida nueva")
    if size <= 0:
        raise ValueError("Tamaño de muestra inválido")
    inputs = {}

    def checked(path, expected=None):
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError(f"Hash modificado: {path}")
        inputs[str(path)] = value
        return value

    manifest = json.loads((experiment / "manifest.json").read_text())
    checked(experiment / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Se requiere contraste original terminado")
    candidate_path = cohort / "test_opportunities.csv"
    report_path = cohort / "candidate_validation_report.json"
    checked(candidate_path, manifest["input_sha256"][str(candidate_path)])
    checked(report_path, manifest["input_sha256"][str(report_path)])
    report = json.loads(report_path.read_text())
    availability_path = cohort / "opportunity_availability.csv"
    checked(availability_path, report["output_sha256"][availability_path.name])
    candidates = read_rows(candidate_path)
    if len({r["opportunity_id"] for r in candidates}) != len(candidates):
        raise ValueError("Candidatos duplicados")
    availability_rows = read_rows(availability_path)
    if len({r["ocid"] for r in availability_rows}) != len(availability_rows):
        raise ValueError("Disponibilidad duplicada")
    availability = {r["ocid"]: r for r in availability_rows}
    selected = sorted(candidates, key=lambda r: (hashlib.sha256(f"{seed}:{r['opportunity_id']}".encode()).hexdigest(), r["opportunity_id"]))[:size]
    if len(selected) != size:
        raise ValueError("Corpus insuficiente para la muestra comprometida")
    details, documents, findings = [], [], []
    for number, candidate in enumerate(selected, 1):
        ocid = candidate["opportunity_id"]
        source = availability[ocid]
        if source["status"] != "verified":
            raise ValueError("Fuente no retenida")
        path = Path(source["source_path"])
        checked(path, source["sha256"])
        package = json.loads(path.read_text(encoding="utf-8"))
        releases = package.get("releases", [])
        if len(releases) != 1 or releases[0].get("ocid") != ocid:
            raise ValueError("La versión inicial no tiene identidad única")
        release, local = releases[0], []
        tender = release.get("tender") or {}
        if release.get("publishedDate") != source["text_published_at"]:
            raise ValueError("Fecha de fuente discordante")
        if candidate.get("awarded_supplier_id"):
            raise ValueError("Candidato incorpora adjudicatario posterior")
        for field in FIELDS:
            prediction = candidate.get({"object": "description"}.get(field, field), "")
            value, pointer = source_reference(release, field)
            state = "observed" if field != "experience" and norm(value, field) is not None else "absent"
            note = ""
            if field == "experience":
                # Current frozen production schema does not extract this field.
                prediction = ""
                note = "El extractor congelado no implementa experiencia. No se infiere desde adjudicaciones ni se completa con anexos."
                state = "requires_semantic_reference" if value else "not_identified_in_available_text"
                note += " Hay menciones que requieren interpretar el requisito." if value else " No aparece la palabra experiencia en el ámbito textual revisado. Esto no prueba ausencia del requisito en anexos."
            match = field != "experience" and norm(value, field) == norm(prediction, field)
            item = {"sample_number": number, "opportunity_id": ocid, "field": field,
                    "field_name": FIELD_NAMES[field], "prediction": prediction,
                    "reference_value": value, "reference_state": state,
                    "json_path": pointer, "status": "match" if match else "not_evaluable" if field == "experience" else "mismatch",
                    "note": note, "currency": candidate.get("currency", "") if field == "amount" else ""}
            details.append(item)
            local.append(item)
        if norm((tender.get("value") or {}).get("amount"), "amount") == Decimal(0):
            findings.append({"opportunity_id": ocid, "kind": "zero_amount", "note": "Cero consta en la primera publicación. Se conserva y no se sustituye con un monto posterior."})
        description = str(tender.get("description") or "")
        for i, item in enumerate(tender.get("items") or []):
            text = str(item.get("description") or "")
            if text and len(text) < len(description) and description.startswith(text):
                findings.append({"opportunity_id": ocid, "kind": "possible_source_truncation", "item_index": i,
                                 "note": "La descripción del ítem es un prefijo más corto del objeto. Es una alerta de fuente, no un truncamiento demostrado del extractor."})
        buyer_ids = {(release.get("buyer") or {}).get("id"), (tender.get("procuringEntity") or {}).get("id")}
        parties = [p for p in release.get("parties") or [] if p.get("id") in buyer_ids or set(p.get("roles") or []) & {"buyer", "procuringEntity"}]
        documents.append({"sample_number": number, "opportunity_id": ocid, "publication_date": candidate["publication_date"],
                          "source_path": str(path), "source_sha256": source["sha256"], "release_id": release.get("id"),
                          "source_uri": package.get("uri", ""), "source_published_at": release.get("publishedDate"),
                          "source_excerpt": {"tender": {k: tender[k] for k in ("id", "title", "description", "mainProcurementCategory", "value", "items", "documents") if k in tender}, "buyer": release.get("buyer"), "parties": parties},
                          "fields": local, "findings": [f for f in findings if f["opportunity_id"] == ocid]})
    metrics = {field: score_field([r for r in details if r["field"] == field]) for field in FIELDS}
    summary = {"task": "automatic_source_concordance_not_human_annotation_or_expert_validation",
               "permission_basis": "Tesista confirmó en esta conversación aprobación docente para evaluación automática. La ficha se conserva por solicitud del tesista. No se verificó acta externa.",
               "sample_size": size, "seed": seed, "population": len(candidates), "selection": "first_50_sorted_by_sha256_utf8_seed_colon_ocid_tie_ocid_no_strata_no_outcomes",
               "fields": list(FIELDS), "judgments": len(details), "metrics_by_field": metrics,
               "status_counts": dict(Counter(r["status"] for r in details)), "findings_counts": dict(Counter(f["kind"] for f in findings)),
               "processable_sources": len(documents), "sample_source_processing_rate": len(documents) / size,
               "processable_rate_scope": "retained_candidate_population_not_all_downloaded_documents",
               "ocr_evaluated": False, "pdf_extraction_evaluated": False, "human_judgments_completed": 0,
               "labels_read": False, "ranking_modified": False,
               "limitations": ["Objeto significa tender.description, no el código de convocatoria guardado en title.",
                               "Región significa dirección de entidad, no lugar verificado de ejecución.",
                               "Montos cero son valores observados y no presupuestos utilizables.",
                               "Experiencia no implementada en el extractor congelado. F1 no estimable sin referencia utilizable.",
                               "Concordancia automática verifica reglas y transformación, no veracidad del registro, interpretación legal ni extracción completa de anexos."]}
    for p, expected in inputs.items():
        if digest(p) != expected:
            raise ValueError("Una entrada cambió durante la auditoría")
    output.mkdir(parents=True)
    for name, value in (("summary.json", summary), ("row_results.json", details), ("review_packet.json", documents), ("source_findings.json", findings)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_manifest = {"input_sha256": inputs, "script_sha256": digest(Path(__file__)),
                      "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
                      "selection_sha256": hashlib.sha256(json.dumps([r["opportunity_id"] for r in selected]).encode()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(audit_manifest, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("data/processed/contrast_candidates_v1/annual_v1"))
    parser.add_argument("--experiment", type=Path, default=Path("artifacts/experiment_annual_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(audit(args.cohort, args.experiment, args.output, args.size, args.seed), ensure_ascii=False, indent=2))
