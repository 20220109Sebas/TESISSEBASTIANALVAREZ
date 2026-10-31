# Módulos adicionales de implementación

La implementación incluye el candidato de extracción selectiva `document_experience_candidate_v2`, la evaluación de referencias `document_experience_evaluation_v2` y la comparación `document_experience_selective_evaluation_v1`. El candidato conserva las abstenciones y la evaluación mantiene los campos conocidos omitidos en el denominador.

`review_context_v2` resume antecedentes, categoría y montos comparables. `human_relevance_v1` calcula métricas únicamente cuando se dispone del conjunto completo de valoraciones; los valores faltantes no se convierten en negativos.

`rank_fusion_v1` y `reranking_v1` contienen alternativas experimentales de fusión y reordenamiento. No son las cuatro configuraciones originales ni cambian las tablas del artículo. Las pruebas asociadas utilizan ejemplos sintéticos. El suplemento `reproducibilidad/nota2_20261026/` conserva la reproducción de los resultados preliminares originales.

Para comprobar estos módulos desde la raíz:

```bash
python -m pytest tests/test_document_experience_selective_v2.py tests/test_human_relevance_v1.py tests/test_rank_fusion_v1.py tests/test_reranking_v1.py tests/test_review_context_v2.py
```
