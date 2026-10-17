# Arquitectura de la implementación

El sistema se ejecuta localmente en Python. Conserva las fuentes en archivos, las tablas canónicas en CSV, los vectores en una caché numérica y los resultados y sus manifiestos en JSON. No requiere una base de datos para ejecutar el experimento.

## Recorrido de los datos

```text
Fuentes OCDS de OECE
        |
        v
Integración de publicaciones, limpieza y comprobación temporal
        |
        +--> Adjudicaciones históricas --> Perfil por empresa
        |
        +--> Licitaciones candidatas --> Corpus de comparación
                                           |
Perfil y corpus --> BM25 + LaBSE + categoría/región/monto
                                           |
                                           v
                           Puntaje ponderado y ranking explicado
                                           |
                                           v
                         Evaluación con etiquetas independientes
```

`awards`, `tender` y `parties` ya existen en las fuentes OCDS. La implementación no inventa adjudicaciones: extrae campos, verifica su evidencia y los une en el esquema canónico. `awards` en el código Python es una lista de objetos `Award`; un perfil agrupa varios antecedentes de una empresa, no los datos de una única licitación.

## Módulos principales

| Responsabilidad | Módulos de `src/tesis_recommender/` |
|---|---|
| Descarga y lectura de fuentes | `oece_client.py`, `ocds.py` |
| Corpus, cohorte y disponibilidad temporal | `corpus.py`, `cohort.py`, `initial_releases.py`, `validation.py` |
| Esquemas, CSV y perfiles | `schemas.py`, `io.py`, `profiles.py` |
| Texto y BM25 | `text.py`, `bm25.py` |
| LaBSE, fragmentación y caché | `semantic.py`, `embedding_cache.py` |
| Señales y combinación ponderada | `hybrid.py`, `pipeline.py` |
| Métricas y análisis | `metrics.py`, `annual_evaluation.py`, `experiment_analysis.py` |
| Extracción documental de experiencia | `experience.py`, `experience_segments.py`, `experience_conditions.py`, `document_experience_candidate_v1.py` |
| Fuentes y evaluación documental | `document_sampling_v1.py`, `document_source_preparation_v1.py`, `document_experience_evaluation_v1.py` |
| Interfaz de ejecución | `cli.py`, `__main__.py` |

## Comparación textual y atributos

BM25 calcula coincidencia de términos considerando su frecuencia, su rareza en el corpus y la longitud del documento. La señal semántica compara vectores obtenidos con LaBSE mediante similitud coseno. Los textos extensos se fragmentan respetando el tokenizador; sus representaciones se agregan y normalizan.

Los puntajes textuales se normalizan entre cero y uno para cada empresa y conjunto de candidatos antes de aplicar los pesos. No son porcentajes de palabras coincidentes ni probabilidades.

Categoría y región se comparan por coincidencia con el historial. La región corresponde a la entidad compradora, no necesariamente al lugar de ejecución del servicio. El monto se compara con la mediana histórica de la empresa en la misma moneda, utilizando solo importes positivos y finitos. Un dato ausente o no comparable aporta cero sin acreditar incompatibilidad.

## Separación de evaluación

Las etiquetas indican pares empresa–licitación adjudicados observados y se usan después del ranking. No entran en la representación del perfil ni en sus señales de compatibilidad. Una licitación sin positivo conocido no se considera automáticamente incompatible.

La evaluación documental compara requisitos extraídos con una referencia previamente fijada, y registra por separado cobertura y alcance de las fuentes. Los resultados automáticos y las revisiones de fuentes no se presentan como validación experta independiente.
