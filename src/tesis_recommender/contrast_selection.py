"""Selección provisional sin consultar etiquetas ni exigir positivos futuros."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .cohort import _allocate, _history_band
from .validation import read_csv, write_csv, sha256, version_hint, normalized_url


def _key(identifier: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def select_contrast_cohort(corpus_dir: Path, excluded_profiles_path: Path, output_dir: Path,
                           profiles_target: int = 50, seed: int = 20261008,
                           cutoff: date = date(2025, 1, 1), end: date = date(2025, 12, 31)) -> dict:
    if profiles_target <= 0 or end < cutoff:
        raise ValueError("Revisar tamaño de perfiles y ventana temporal")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("La selección se conserva sin sobrescribir. Usar una carpeta nueva")
    excluded = {r["supplier_id"] for r in read_csv(excluded_profiles_path)}
    award_path = corpus_dir / "historical_awards.csv"
    candidate_path = corpus_dir / "test_opportunities.csv"
    # Las proyecciones no usan textos o atributos de versiones consolidadas
    # posteriores al corte. La publicación nativa se auditará antes de modelar.
    history = defaultdict(dict)
    for row in read_csv(award_path):
        event = date.fromisoformat(row["award_date"][:10])
        if row["supplier_id"] in excluded or event >= cutoff or row.get("object_type", "").lower() != "services":
            continue
        projection = {f: row[f] for f in ("supplier_id", "award_id", "award_date", "source_url")}
        identity = (row["supplier_id"], row["award_id"])
        old = history[row["supplier_id"]].get(identity)
        if old is not None and old != projection:
            raise ValueError("Identidad histórica duplicada con evidencia distinta")
        history[row["supplier_id"]][identity] = projection
    strata = defaultdict(list)
    for supplier, items in sorted(history.items()):
        if len(items) >= 2:
            strata[(_history_band(len(items)),)].append(supplier)
    counts = Counter({band: len(ids) for band, ids in strata.items()})
    if sum(counts.values()) < profiles_target:
        raise ValueError("No hay suficientes perfiles históricos provisionales")
    quotas = _allocate(counts, profiles_target)
    selected = sorted(supplier for band, ids in sorted(strata.items())
                      for supplier in sorted(ids, key=lambda s: _key(s, seed))[:quotas[band]])
    profiles = [{"supplier_id": s, "history_count_provisional": len(history[s]),
                 "history_band": _history_band(len(history[s]))} for s in selected]
    history_rows = [r for s in selected for _, r in sorted(history[s].items())]
    candidates = {}
    for row in read_csv(candidate_path):
        event = date.fromisoformat(row["publication_date"][:10])
        if not cutoff <= event <= end or row.get("object_type", "").lower() != "services":
            continue
        projection = {f: row[f] for f in ("opportunity_id", "publication_date", "object_type", "source_url")}
        old = candidates.get(row["opportunity_id"])
        if old is not None and old != projection:
            raise ValueError("Oportunidad duplicada con datos de selección distintos")
        candidates[row["opportunity_id"]] = projection
    if not candidates:
        raise ValueError("No hay oportunidades en la ventana")
    candidate_rows = [r for _, r in sorted(candidates.items())]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "selected_profiles.csv", profiles,
              ["supplier_id", "history_count_provisional", "history_band"])
    write_csv(output_dir / "profile_history_ids.csv", history_rows,
              ["supplier_id", "award_id", "award_date", "source_url"])
    write_csv(output_dir / "candidate_ids.csv", candidate_rows,
              ["opportunity_id", "publication_date", "object_type", "source_url"])
    selected_paths = [output_dir / name for name in ("selected_profiles.csv", "profile_history_ids.csv", "candidate_ids.csv")]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
              "cutoff": cutoff.isoformat(), "candidate_end": end.isoformat(),
              "profiles": len(selected), "historical_awards_provisional": len(history_rows),
              "candidates_provisional": len(candidate_rows), "excluded_development_profiles": len(excluded),
              "eligible_profiles_provisional": sum(counts.values()),
              "profile_strata_population": {k[0]: v for k, v in sorted(counts.items())},
              "profile_strata_selected": dict(Counter(r["history_band"] for r in profiles)),
              "candidate_rule": "all services in the event-date window, no positive inclusion or outcome filtering",
              "selection_uses_future_labels": False, "labels_read": False,
              "native_publication_audit_pending": True,
              "evaluation_status": "provisional_new_company_contrast_same_inspected_source_year_not_blind_temporal_test",
              "next_gate": "Audit native publication and recover pre-cutoff history and initial opportunity texts before creating model inputs. Exclusions must follow publication and scope only, not require future positives. Do not replace companies to increase observed positives.",
              "inputs_sha256": {str(p): sha256(p) for p in (award_path, candidate_path, excluded_profiles_path)},
              "selection_sha256": {p.name: sha256(p) for p in selected_paths},
              "source_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("cohort.py"))}}
    (output_dir / "selection_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def freeze_contrast_code(preparation_dir: Path) -> dict:
    """Archiva código recuperable, no solo sus hashes, sin revelar etiquetas."""
    original_path = preparation_dir / "frozen_evaluation_plan.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    preparation = json.loads((preparation_dir / "preparation_manifest.json").read_text(encoding="utf-8"))
    if sha256(original_path) != preparation["plan_sha256"] or sha256(preparation_dir / "version_queue.csv") != preparation["queue_sha256"]:
        raise ValueError("Se modificó el plan o la cola congelada")
    sources = sorted(Path(__file__).parent.glob("*.py"))
    changed = [p.name for p in sources if original["source_sha256"].get(p.name) != sha256(p)]
    ranking_modules = {"hybrid.py", "bm25.py", "semantic.py", "schemas.py", "profiles.py", "text.py", "embedding_cache.py", "metrics.py", "pipeline.py", "io.py"}
    if set(changed) & ranking_modules:
        raise ValueError("Cambió un módulo del modelo. Revisar y congelar un nuevo plan explícito")
    archive = preparation_dir / "code_snapshot"
    if archive.exists():
        raise ValueError("El código archivado no se sobrescribe")
    archive.mkdir()
    for path in sources:
        shutil.copyfile(path, archive / path.name)
    project_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if project_file.exists():
        shutil.copyfile(project_file, archive / project_file.name)
    final = {**original, "code_archived_at_utc": datetime.now(timezone.utc).isoformat(),
             "original_plan_sha256": sha256(original_path),
             "non_ranking_modules_changed_since_original_freeze": changed,
             "source_sha256": {p.name: sha256(archive / p.name) for p in sources},
             "code_archive_sha256": {p.name: sha256(p) for p in sorted(archive.iterdir())},
             "code_archive": str(archive), "labels_read": False}
    (preparation_dir / "frozen_plan_with_code.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"archived_files": len(list(archive.iterdir())), "ranking_modules_unchanged": True,
            "labels_read": False, "changed_non_ranking_modules": changed}


def prepare_contrast_sources(selection_dir: Path, corpus_dir: Path, raw_root: Path,
                             model_metadata_path: Path, output_dir: Path) -> dict:
    """Localiza versiones sin abrir etiquetas y congela las decisiones actuales.

    La cola no certifica disponibilidad. El hint del identificador se verifica
    después contra publishedDate dentro de cada respuesta de la API.
    """
    from .pipeline import ABLATIONS

    selection_path = selection_dir / "selection_manifest.json"
    manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    for name, expected in manifest["selection_sha256"].items():
        if sha256(selection_dir / name) != expected:
            raise ValueError(f"La selección congelada fue modificada: {name}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("La preparación no se sobrescribe. Usar una carpeta nueva")
    cutoff = date.fromisoformat(manifest["cutoff"])
    roles = defaultdict(set)
    for row in read_csv(selection_dir / "profile_history_ids.csv"):
        roles[row["source_url"].rstrip("/").rsplit("/", 1)[-1]].add("historical_profile")
    for row in read_csv(selection_dir / "candidate_ids.csv"):
        roles[row["opportunity_id"]].add("test_candidate")
    snapshots = defaultdict(set)
    for row in read_csv(corpus_dir / "lineage.csv"):
        if row["ocid"] in roles:
            snapshots[row["selected_snapshot_id"]].add(row["ocid"])
    queue, sources, missing_refs, found = {}, [], [], set()
    for snapshot_id, ids in sorted(snapshots.items()):
        year, month = snapshot_id.split("-")
        paths = sorted((raw_root / year / month).glob("*.json"))
        if len(paths) != 1:
            raise ValueError(f"Snapshot no identificado de forma única: {snapshot_id}")
        path = paths[0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            ocid = record.get("ocid")
            if ocid not in ids:
                continue
            found.add(ocid)
            refs = record.get("releases", [])
            tender = [r for r in refs if "tender" in r.get("tag", []) and r.get("url")]
            known = [r for r in tender if version_hint(r) is not None]
            selected = [min(known, key=lambda r: (version_hint(r), r["url"]))] if known else []
            selected += [r for r in tender if version_hint(r) is None]
            if not tender:
                missing_refs.append({"ocid": ocid, "issue": "no_tender_reference"})
            if "historical_profile" in roles[ocid]:
                award_refs = [r for r in refs if r.get("url") and any(t.startswith("award") for t in r.get("tag", []))
                              and (version_hint(r) is None or version_hint(r).astimezone(ZoneInfo("America/Lima")).date() < cutoff)]
                selected += award_refs
                if not award_refs:
                    missing_refs.append({"ocid": ocid, "issue": "no_pre_cutoff_award_reference_hint"})
            for ref in selected:
                url = normalized_url(ref["url"])
                if url in queue and queue[url]["ocid"] != ocid:
                    raise ValueError("Una URL de versión corresponde a dos procesos distintos")
                queue[url] = {"ocid": ocid, "role": "|".join(sorted(roles[ocid])), "url": url,
                              "version_hint": version_hint(ref).isoformat() if version_hint(ref) else "",
                              "reference_date": ref.get("date", ""),
                              "cache_key": hashlib.sha256(url.encode()).hexdigest()}
        sources.append({"path": str(path), "sha256": sha256(path)})
    if set(roles) - found:
        raise ValueError("Hay procesos seleccionados que no se localizaron en sus snapshots")
    rows = sorted(queue.values(), key=lambda r: (r["ocid"], r["version_hint"], r["url"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_fields = ["ocid", "role", "url", "version_hint", "reference_date", "cache_key"]
    write_csv(output_dir / "version_queue.csv", rows, queue_fields)
    historic = [r for r in rows if "historical_profile" in r["role"].split("|")]
    write_csv(output_dir / "historical_version_queue.csv", historic, queue_fields)
    write_csv(output_dir / "reference_issues.csv", missing_refs, ["ocid", "issue"])
    metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    plan = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_manifest_sha256": sha256(selection_path), "selection_sha256": manifest["selection_sha256"],
            "evaluation_status": manifest["evaluation_status"], "cutoff": manifest["cutoff"],
            "candidate_end": manifest["candidate_end"], "seed": manifest["seed"],
            "models": ABLATIONS, "model_name": metadata["model_name"], "model_revision": metadata["model_revision"],
            "profile_strategy": metadata["profile_strategy"], "opportunity_strategy": metadata["opportunity_strategy"],
            "metrics": [f"{m}@{k}" for k in (5, 10) for m in ("precision", "recall", "f1")],
            "macro_population": "Only profiles with known positives after the frozen candidate selection. Report every profile and the count without known positives. Never replace profiles because of their outcomes.",
            "candidate_pool": "Full selected annual service window. No candidate injection based on winners. Date and publication audit may exclude unusable sources without consulting future labels.",
            "positive_definition": "Observed later award to the selected company. A known outcome preceding first verified text availability is not a forecast-positive label. Other pairs remain unlabelled, not negatives.",
            "publication_gate": "Require native release.publishedDate and pre-cutoff evidence for history. Candidate texts must be in the selected event-date and native-publication window. Hints and event dates never replace native publication.",
            "analysis": "Paired per-profile bootstrap percentile 95%, 5000 resamples, seed 42. Exploratory same-source/company-disjoint contrast, not a blind temporal generalization test.",
            "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    (output_dir / "frozen_evaluation_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"created_at_utc": plan["created_at_utc"], "processes": len(roles),
              "versions_to_inspect": len(rows), "historical_versions_to_inspect": len(historic),
              "reference_issues": len(missing_refs), "labels_read": False,
              "publication_verified": False, "sources": sources,
              "queue_sha256": sha256(output_dir / "version_queue.csv"),
              "plan_sha256": sha256(output_dir / "frozen_evaluation_plan.json"),
              "inputs_sha256": {str(p): sha256(p) for p in (selection_path, corpus_dir / "lineage.csv", model_metadata_path)}}
    (output_dir / "preparation_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
