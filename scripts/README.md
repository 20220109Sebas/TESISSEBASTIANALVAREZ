# Guía de scripts

La interfaz principal para instalar y probar el recomendador es `tesis-recommender`. Los scripts amplían el recorrido con experimentos, auditorías y preparación documental. Se ejecutan desde la raíz, preferentemente como módulos:

```bash
python -m scripts.nombre_del_script --help
```

Los nombres versionados conservan las etapas del estudio. No es necesario ejecutar todos los scripts para utilizar el recomendador. Algunos verificadores leen salidas de una ejecución concreta y requieren sus manifiestos originales, no incluidos en el repositorio.

| Grupo | Archivos de entrada al recorrido | Finalidad |
|---|---|---|
| Corpus anual | `materialize_annual_pool.py`, `audit_annual_positives.py` | Preparar la colección anual y comprobar positivos |
| Contraste de ranking | `run_annual_contrast.py`, `verify_annual_contrast.py` | Ejecutar variantes y contrastar integridad |
| Demostración archivada | `demonstrate_annual_recommendations.py` | Recalcular desde señales guardadas, sin volver a ejecutar LaBSE |
| Categorías | `audit_annual_categories.py`, `run_category_contrast.py` | Auditar cobertura y variantes de categoría |
| Fuentes documentales | `fetch_experience_documents.py`, `prepare_document_sources_v1.py` | Obtener PDF y preparar evidencia verificable |
| Muestreo documental | `freeze_document_source_design_v1.py`, `document_validation_run_v3.py` | Fijar controles y seleccionar fuentes de validación |
| Requisitos de experiencia | `extract_experience_pilot.py`, `freeze_document_experience_candidate_v1.py` | Extraer requisitos y fijar una versión del candidato |
| Evaluación documental | `run_document_experience_evaluation_v1.py` | Separar referencia, inferencia y comparación |
| Lecturas de fuentes | `record_holdout_*.py`, `verify_holdout_*.py` | Registrar y comprobar lecturas y conciliación |
| OCR | `vision_pdf_ocr.swift`, `complete_mixed_pdf_text.py` | Recuperar texto de páginas sin texto utilizable |
| Revisión de extracción | `build_extraction_review.py`, `check_extraction_review.cjs` | Preparar una vista local y comprobar su funcionamiento |

Consulta los parámetros de cada archivo y la [guía de ejecución](../docs/ejecucion.md) antes de utilizarlo. No cambies hashes ni omitas controles para hacer pasar un verificador histórico sobre datos diferentes.
