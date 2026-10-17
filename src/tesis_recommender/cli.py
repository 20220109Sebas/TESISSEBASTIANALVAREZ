from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .io import read_awards, read_opportunities, read_relevance_labels
from .embedding_cache import build_embedding_cache
from .corpus import build_corpus
from .cohort import build_experimental_cohort
from .initial_releases import download_initial_releases, materialize_temporal_cohort
from .ocds import export_canonical_csv
from .oece_client import catalog_range, download_json_snapshot, download_range, list_snapshots
from .pipeline import run_experiment, save_experiment
from .semantic import DeterministicTestEmbedder, LaBSEEmbedder, PrecomputedEmbedder
from .validation import prepare_version_audit, fetch_versions, materialize_validated_cohort, prepare_annotation, evaluate_annotation, verify_experiment_inputs
from .experiment_analysis import analyze_experiment
from .error_dossier import build_error_dossier
from .contrast_selection import select_contrast_cohort, prepare_contrast_sources, freeze_contrast_code
from .contrast_history import fetch_contrast_history, materialize_contrast_history
from .contrast_candidates import fetch_candidate_batch, audit_candidate_batch
from .contrast_embeddings import encode_audited_batch, record_embedding_amendment


def _execute(
    awards_path: Path, opportunities_path: Path, embedder_name: str, output_dir: Path,
    labels_path: Path | None = None, embedding_cache: Path | None = None,
    validation_report: Path | None = None,
) -> None:
    audit = None
    if validation_report:
        if labels_path is None:
            raise ValueError("La ejecución auditada requiere etiquetas explícitas")
        audit = verify_experiment_inputs(validation_report, awards_path, opportunities_path, labels_path)
    embedder = (
        PrecomputedEmbedder(embedding_cache)
        if embedding_cache
        else LaBSEEmbedder()
        if embedder_name == "labse"
        else DeterministicTestEmbedder()
    )
    relevant = read_relevance_labels(labels_path) if labels_path else None
    result = run_experiment(
        read_awards(awards_path), read_opportunities(opportunities_path), embedder,
        date.fromisoformat(audit["cutoff"]) if audit else date(2025, 1, 1), relevant_override=relevant,
    )
    if audit:
        result["protocol_version"] = audit["protocol_version"]
        result["evaluation_status"] = audit["evaluation_status"]
        result["availability_audit"] = {
            "report_path": str(validation_report), "timestamp_basis": audit["timestamp_basis"],
            "date_timezone": audit["date_timezone"], "evidence_limit": audit["evidence_limit"],
        }
    inputs = [awards_path, opportunities_path] + ([labels_path] if labels_path else [])
    if embedding_cache:
        inputs.extend([embedding_cache / "metadata.json", embedding_cache / "index.json", embedding_cache / "embeddings.npy"])
    if validation_report:
        inputs.append(validation_report)
    save_experiment(result, output_dir, inputs)
    print(json.dumps({name: run["metrics"] for name, run in result["runs"].items()}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="POC de ranking híbrido para servicios públicos")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Valida el pipeline con datos sintéticos")
    demo.add_argument("--data-dir", type=Path, default=Path("data/sample"))
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    run = subparsers.add_parser("run", help="Ejecuta el experimento sobre el snapshot procesado")
    run.add_argument("--awards", type=Path, required=True)
    run.add_argument("--opportunities", type=Path, required=True)
    run.add_argument("--embedder", choices=("labse", "test"), default="labse")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--labels", type=Path, help="Etiquetas proveedor-oportunidad externas")
    run.add_argument("--embedding-cache", type=Path, help="Caché LaBSE creado con nlp-cache")
    run.add_argument("--validation-report", type=Path, help="Verifica hashes y registra la auditoría de disponibilidad v3")
    nlp_cache = subparsers.add_parser("nlp-cache", help="Genera un caché reproducible de embeddings LaBSE")
    nlp_cache.add_argument("--awards", type=Path, required=True)
    nlp_cache.add_argument("--opportunities", type=Path, required=True)
    nlp_cache.add_argument("--output-dir", type=Path, required=True)
    nlp_cache.add_argument("--batch-size", type=int, default=16)
    ocds_list = subparsers.add_parser("oece-list", help="Lista snapshots masivos publicados por OECE")
    ocds_list.add_argument("--source", choices=("seace_v3", "seace_v2"), default="seace_v3")
    ocds_list.add_argument("--year", type=int)
    ocds_download = subparsers.add_parser("oece-download", help="Descarga y verifica un snapshot JSON mensual")
    ocds_download.add_argument("--source", choices=("seace_v3", "seace_v2"), default="seace_v3")
    ocds_download.add_argument("--year", type=int, required=True)
    ocds_download.add_argument("--month", type=int, required=True)
    ocds_download.add_argument("--output-dir", type=Path, required=True)
    ocds_transform = subparsers.add_parser("oece-transform", help="Convierte un snapshot OCDS al esquema canónico")
    ocds_transform.add_argument("--snapshot", type=Path, required=True)
    ocds_transform.add_argument("--output-dir", type=Path, required=True)
    catalog = subparsers.add_parser("oece-catalog", help="Cataloga disponibilidad y volumen remoto por rango")
    catalog.add_argument("--source", choices=("seace_v3", "seace_v2"), default="seace_v3")
    catalog.add_argument("--start-year", type=int, required=True)
    catalog.add_argument("--end-year", type=int, required=True)
    catalog.add_argument("--output", type=Path, required=True)
    batch = subparsers.add_parser("oece-download-range", help="Descarga reanudable y verifica un rango completo")
    batch.add_argument("--source", choices=("seace_v3", "seace_v2"), default="seace_v3")
    batch.add_argument("--start-year", type=int, required=True)
    batch.add_argument("--end-year", type=int, required=True)
    batch.add_argument("--root-dir", type=Path, required=True)
    batch.add_argument("--workers", type=int, default=4)
    corpus = subparsers.add_parser("oece-build-corpus", help="Consolida, filtra y deduplica la ventana OCDS")
    corpus.add_argument("--raw-root", type=Path, required=True)
    corpus.add_argument("--output-dir", type=Path, required=True)
    cohort = subparsers.add_parser("oece-build-cohort", help="Selecciona una cohorte experimental estratificada")
    cohort.add_argument("--corpus-dir", type=Path, required=True)
    cohort.add_argument("--output-dir", type=Path, required=True)
    cohort.add_argument("--profiles", type=int, default=50)
    cohort.add_argument("--candidates", type=int, default=1000)
    cohort.add_argument("--seed", type=int, default=42)
    releases = subparsers.add_parser("oece-fetch-initial", help="Descarga y valida primeras entregas de una cola")
    releases.add_argument("--queue", type=Path, required=True)
    releases.add_argument("--cache-dir", type=Path, required=True)
    releases.add_argument("--output", type=Path, required=True)
    releases.add_argument("--workers", type=int, default=8)
    releases.add_argument("--retries", type=int, default=3)
    materialize = subparsers.add_parser("oece-materialize-cohort", help="Reconstruye perfiles y candidatos con las primeras entregas")
    materialize.add_argument("--cohort-dir", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    audit = subparsers.add_parser("oece-audit-versions", help="Audita publicación histórica y prepara la recuperación de versiones")
    audit.add_argument("--raw-root", type=Path, required=True)
    audit.add_argument("--corpus-dir", type=Path, required=True)
    audit.add_argument("--cohort-dir", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    versions = subparsers.add_parser("oece-fetch-versions", help="Recupera versiones referenciadas para verificar disponibilidad")
    versions.add_argument("--audit-dir", type=Path, required=True)
    versions.add_argument("--cohort-dir", type=Path, required=True)
    versions.add_argument("--cache-dir", type=Path, required=True)
    versions.add_argument("--workers", type=int, default=4)
    validated = subparsers.add_parser("oece-validate-cohort", help="Reconstruye un piloto con evidencia de publicación y moneda")
    validated.add_argument("--cohort-dir", type=Path, required=True)
    validated.add_argument("--audit-dir", type=Path, required=True)
    validated.add_argument("--output-dir", type=Path, required=True)
    annotation = subparsers.add_parser("annotation-prepare", help="Prepara fuentes y campos para anotación humana")
    annotation.add_argument("--cohort-dir", type=Path, required=True)
    annotation.add_argument("--audit-dir", type=Path, required=True)
    annotation.add_argument("--output-dir", type=Path, required=True)
    annotation.add_argument("--size", type=int, default=50)
    annotation.add_argument("--seed", type=int, default=42)
    annotation_eval = subparsers.add_parser("annotation-evaluate", help="Evalúa campos después de completar los juicios humanos")
    annotation_eval.add_argument("--annotations", type=Path, required=True)
    annotation_eval.add_argument("--output-dir", type=Path, required=True)
    analysis = subparsers.add_parser("analyze", help="Genera métricas por perfil, intervalos pareados y casos para revisión")
    analysis.add_argument("--results", type=Path, required=True)
    analysis.add_argument("--labels", type=Path, required=True)
    analysis.add_argument("--awards", type=Path, required=True)
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--resamples", type=int, default=5000)
    analysis.add_argument("--seed", type=int, default=42)
    dossier = subparsers.add_parser("error-dossier", help="Reconstruye cambios de ranking con contexto y revisión humana vacía")
    dossier.add_argument("--experiment-dir", type=Path, required=True)
    dossier.add_argument("--cohort-dir", type=Path, required=True)
    dossier.add_argument("--embedding-cache", type=Path, required=True)
    dossier.add_argument("--output-dir", type=Path, required=True)
    contrast = subparsers.add_parser("contrast-select", help="Preselecciona empresas nuevas sin etiquetas futuras y todo el pool anual")
    contrast.add_argument("--corpus-dir", type=Path, required=True)
    contrast.add_argument("--exclude-profiles", type=Path, required=True)
    contrast.add_argument("--output-dir", type=Path, required=True)
    contrast.add_argument("--profiles", type=int, default=50)
    contrast.add_argument("--seed", type=int, default=20261008)
    contrast_sources = subparsers.add_parser("contrast-prepare", help="Prepara versiones y congela el plan antes de revelar etiquetas")
    contrast_sources.add_argument("--selection-dir", type=Path, required=True)
    contrast_sources.add_argument("--corpus-dir", type=Path, required=True)
    contrast_sources.add_argument("--raw-root", type=Path, required=True)
    contrast_sources.add_argument("--model-metadata", type=Path, required=True)
    contrast_sources.add_argument("--output-dir", type=Path, required=True)
    contrast_freeze = subparsers.add_parser("contrast-freeze", help="Archiva el código sin cambiar las decisiones del modelo")
    contrast_freeze.add_argument("--preparation-dir", type=Path, required=True)
    history_fetch = subparsers.add_parser("contrast-fetch-history", help="Recupera la cola histórica de la selección sin etiquetas futuras")
    history_fetch.add_argument("--selection-dir", type=Path, required=True)
    history_fetch.add_argument("--preparation-dir", type=Path, required=True)
    history_fetch.add_argument("--cache-dir", type=Path, required=True)
    history_fetch.add_argument("--output-dir", type=Path, required=True)
    history_fetch.add_argument("--cache-index", type=Path, action="append", default=[])
    history_fetch.add_argument("--workers", type=int, default=4)
    history_validate = subparsers.add_parser("contrast-validate-history", help="Reconstruye antecedentes publicados antes del corte sin abrir etiquetas")
    history_validate.add_argument("--selection-dir", type=Path, required=True)
    history_validate.add_argument("--preparation-dir", type=Path, required=True)
    history_validate.add_argument("--fetch-dir", type=Path, required=True)
    history_validate.add_argument("--output-dir", type=Path, required=True)
    candidates_fetch = subparsers.add_parser("contrast-fetch-candidates", help="Recupera un lote computacional sin reducir la selección anual")
    candidates_fetch.add_argument("--selection-dir", type=Path, required=True)
    candidates_fetch.add_argument("--preparation-dir", type=Path, required=True)
    candidates_fetch.add_argument("--cache-dir", type=Path, required=True)
    candidates_fetch.add_argument("--output-dir", type=Path, required=True)
    candidates_fetch.add_argument("--cache-index", type=Path, action="append", default=[])
    candidates_fetch.add_argument("--max-new", type=int, default=4096)
    candidates_fetch.add_argument("--workers", type=int, default=4)
    candidates_audit = subparsers.add_parser("contrast-audit-candidates", help="Audita fuentes recuperadas y registra explícitamente los candidatos pendientes")
    candidates_audit.add_argument("--selection-dir", type=Path, required=True)
    candidates_audit.add_argument("--preparation-dir", type=Path, required=True)
    candidates_audit.add_argument("--history-dir", type=Path, required=True)
    candidates_audit.add_argument("--fetch-dir", type=Path, required=True)
    candidates_audit.add_argument("--output-dir", type=Path, required=True)
    contrast_encode = subparsers.add_parser("contrast-nlp-cache", help="Genera embeddings de fuentes auditadas y reutiliza documentos ya codificados")
    contrast_encode.add_argument("--cohort-dir", type=Path, required=True)
    contrast_encode.add_argument("--preparation-dir", type=Path, required=True)
    contrast_encode.add_argument("--store-dir", type=Path, required=True)
    contrast_encode.add_argument("--output-dir", type=Path, required=True)
    contrast_encode.add_argument("--batch-size", type=int, default=8)
    contrast_encode.add_argument("--embedding-amendment", type=Path)
    amendment = subparsers.add_parser("contrast-amend-embedding", help="Registra la corrección de fragmentación antes de consultar etiquetas")
    amendment.add_argument("--preparation-dir", type=Path, required=True)
    amendment.add_argument("--diagnosis", type=Path, required=True)
    amendment.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "demo":
        _execute(args.data_dir / "historical_awards.csv", args.data_dir / "test_opportunities.csv", "test", args.output_dir)
    elif args.command == "run":
        _execute(args.awards, args.opportunities, args.embedder, args.output_dir, args.labels, args.embedding_cache, args.validation_report)
    elif args.command == "nlp-cache":
        awards = read_awards(args.awards)
        opportunities = read_opportunities(args.opportunities)
        result = build_embedding_cache(
            awards,
            opportunities,
            args.output_dir,
            LaBSEEmbedder(batch_size=args.batch_size),
            input_paths=[args.awards, args.opportunities],
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "oece-list":
        print(json.dumps(list_snapshots(args.source, args.year), ensure_ascii=False, indent=2))
    elif args.command == "oece-download":
        print(json.dumps(download_json_snapshot(args.source, args.year, args.month, args.output_dir), ensure_ascii=False, indent=2))
    elif args.command == "oece-transform":
        print(json.dumps(export_canonical_csv(args.snapshot, args.output_dir), ensure_ascii=False, indent=2))
    elif args.command == "oece-catalog":
        result = catalog_range(args.source, range(args.start_year, args.end_year + 1))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("snapshot_count", "known_total_json_zip_bytes")}, indent=2))
    elif args.command == "oece-download-range":
        result = download_range(args.source, range(args.start_year, args.end_year + 1), args.root_dir, args.workers)
        print(json.dumps({"completed": result["completed"], "failed": result["failed"], "errors": result["errors"]}, ensure_ascii=False, indent=2))
    elif args.command == "oece-build-corpus":
        result = build_corpus(args.raw_root, args.output_dir)
        summary_keys = ["snapshot_count", "raw_records", "unique_service_ocids", "duplicate_versions_removed", "historical_opportunities", "test_opportunities", "historical_awards_before_cutoff", "test_positive_labels", "unique_historical_suppliers", "profiles_ge_2_awards", "profiles_ge_3_awards", "exclusions"]
        print(json.dumps({key: result[key] for key in summary_keys}, ensure_ascii=False, indent=2))
    elif args.command == "oece-build-cohort":
        result = build_experimental_cohort(args.corpus_dir, args.output_dir, args.profiles, args.candidates, args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "oece-fetch-initial":
        result = download_initial_releases(args.queue, args.cache_dir, args.output, args.workers, args.retries)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "oece-materialize-cohort":
        result = materialize_temporal_cohort(args.cohort_dir, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "oece-audit-versions":
        result = prepare_version_audit(args.raw_root, args.corpus_dir, args.cohort_dir, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "oece-fetch-versions":
        result = fetch_versions(args.audit_dir / "version_queue.csv", args.cache_dir,
                                args.audit_dir / "version_results.csv", args.cohort_dir / "initial_release_results.csv", args.workers)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "oece-validate-cohort":
        result = materialize_validated_cohort(args.cohort_dir, args.audit_dir, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "annotation-prepare":
        result = prepare_annotation(args.cohort_dir, args.audit_dir, args.output_dir, args.size, args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "annotation-evaluate":
        result = evaluate_annotation(args.annotations, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "analyze":
        result = analyze_experiment(args.results, args.labels, args.awards, args.output_dir, args.resamples, args.seed)
        print(json.dumps({key: result[key] for key in ("profiles", "profiles_evaluable", "profiles_without_known_positives", "resamples", "comparisons")}, ensure_ascii=False, indent=2))
    elif args.command == "error-dossier":
        result = build_error_dossier(args.experiment_dir, args.cohort_dir, args.embedding_cache, args.output_dir)
        print(json.dumps({key: result[key] for key in ("profiles", "profiles_with_top10_changes", "cases", "counts", "archived_top10_reproduced")}, ensure_ascii=False, indent=2))
    elif args.command == "contrast-select":
        result = select_contrast_cohort(args.corpus_dir, args.exclude_profiles, args.output_dir, args.profiles, args.seed)
        print(json.dumps({key: result[key] for key in ("profiles", "historical_awards_provisional", "candidates_provisional", "labels_read", "native_publication_audit_pending")}, ensure_ascii=False, indent=2))
    elif args.command == "contrast-prepare":
        result = prepare_contrast_sources(args.selection_dir, args.corpus_dir, args.raw_root, args.model_metadata, args.output_dir)
        print(json.dumps({key: result[key] for key in ("processes", "versions_to_inspect", "historical_versions_to_inspect", "reference_issues", "labels_read", "publication_verified")}, ensure_ascii=False, indent=2))
    elif args.command == "contrast-freeze":
        print(json.dumps(freeze_contrast_code(args.preparation_dir), ensure_ascii=False, indent=2))
    elif args.command == "contrast-fetch-history":
        result = fetch_contrast_history(args.selection_dir, args.preparation_dir, args.cache_dir, args.output_dir, args.cache_index, args.workers)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "contrast-validate-history":
        result = materialize_contrast_history(args.selection_dir, args.preparation_dir, args.fetch_dir, args.output_dir)
        print(json.dumps({k: result[k] for k in ("input_profiles", "eligible_profiles_ge_2", "excluded_profiles", "retained_historical_awards", "award_statuses", "version_statuses", "labels_read")}, ensure_ascii=False, indent=2))
    elif args.command == "contrast-fetch-candidates":
        result = fetch_candidate_batch(args.selection_dir, args.preparation_dir, args.cache_dir, args.output_dir, args.cache_index, args.max_new, args.workers)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "contrast-audit-candidates":
        result = audit_candidate_batch(args.selection_dir, args.preparation_dir, args.history_dir, args.fetch_dir, args.output_dir)
        print(json.dumps({k: result[k] for k in ("retained_candidates", "candidate_statuses", "text_quality", "evaluation_ready", "labels_read")}, ensure_ascii=False, indent=2))
    elif args.command == "contrast-nlp-cache":
        result = encode_audited_batch(args.cohort_dir, args.preparation_dir, args.store_dir, args.output_dir, args.batch_size, args.embedding_amendment)
        print(json.dumps({k: result[k] for k in ("profiles", "opportunities", "vector_count", "model_revision", "chunking_audit", "evaluation_ready")}, ensure_ascii=False, indent=2))
    elif args.command == "contrast-amend-embedding":
        result = record_embedding_amendment(args.preparation_dir, args.diagnosis, args.output_dir)
        print(json.dumps({k: result[k] for k in ("chunk_partition_version", "model_revision", "labels_read", "results_used_to_tune_change")}, indent=2))
