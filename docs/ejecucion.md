# Ejecución y reproducción

Ejecuta los comandos desde la raíz del repositorio, con el entorno virtual activado. Utiliza una carpeta de salida nueva para cada ejecución: los verificadores documentales evitan sobrescribir evidencia.

## 1. Instalación

```bash
python -m pip install -e ".[dev]"
tesis-recommender --help
```

Los componentes adicionales se instalan según la tarea:

```bash
python -m pip install -e ".[semantic]"  # Modelo LaBSE
python -m pip install -e ".[sources]"   # Recuperación concurrente de versiones
python -m pip install -e ".[pdf]"       # Lectura de PDF en los scripts documentales
python -m pip install -e ".[notebook]"  # Ejecución programática de notebooks
```

La extracción mediante Poppler requiere `pdftotext` y `pdftoppm` instalados en el sistema. El OCR de `scripts/vision_pdf_ocr.swift` utiliza Vision y requiere macOS y Swift. La revisión visual automatizada opcional utiliza Node.js, Playwright y Chrome; no es un requisito del recomendador ni de la demostración.

## 2. Demostración sin fuentes externas

```bash
tesis-recommender demo --data-dir data/sample --output-dir artifacts/demo
```

Genera `results.json` y `manifest.json`. La representación semántica determinista de prueba verifica el recorrido de ejecución, pero no permite evaluar el rendimiento de LaBSE.

## 3. Obtener y transformar fuentes

La interfaz permite listar archivos, descargarlos con verificación de integridad y transformarlos. Los siguientes comandos muestran un recorrido acotado con un mes:

```bash
tesis-recommender oece-list --source seace_v3 --year 2025
tesis-recommender oece-download \
  --source seace_v3 --year 2025 --month 1 --output-dir data/raw/2025-01
tesis-recommender oece-transform \
  --snapshot data/raw/2025-01/2025-01_seace_v3.json \
  --output-dir data/processed/2025-01
```

Comprueba la ruta `json_path` que devuelve la descarga: el nombre del archivo dentro del ZIP depende de la fuente. Ajusta `--snapshot` si la ruta informada es distinta. Los archivos pueden ser grandes; comprueba el espacio disponible antes de descargar un rango completo.

La descarga no entrega perfiles empresariales listos para recomendar. La transformación selecciona servicios, integra publicaciones y prepara campos. La disponibilidad de un antecedente y del texto de una oportunidad debe verificarse por separado para evitar información posterior al corte.

Para la ventana completa consulta:

```bash
tesis-recommender oece-download-range --help
tesis-recommender oece-build-corpus --help
tesis-recommender oece-build-cohort --help
tesis-recommender oece-audit-versions --help
tesis-recommender oece-validate-cohort --help
```

El muestreo acotado de `oece-build-cohort` no debe confundirse con la colección anual completa de los scripts `materialize_annual_pool.py` y `run_annual_contrast.py`. Estos últimos requieren las auditorías y las tablas de disponibilidad correspondientes.

## 4. Ranking con LaBSE

Las entradas mínimas son las tablas canónicas de [este diccionario](data_dictionary.md). Prepara la ventana histórica antes del 1 de enero de 2025 y las oportunidades de evaluación de 2025.

```bash
tesis-recommender run \
  --awards data/processed/historical_awards.csv \
  --opportunities data/processed/test_opportunities.csv \
  --labels data/processed/labels.csv \
  --embedder labse \
  --output-dir artifacts/experimento_001
```

`labels.csv` contiene `supplier_id`, `opportunity_id` y `label`; los valores positivos identifican pares relevantes conocidos. Sin `--labels`, la interfaz conserva compatibilidad con la columna `awarded_supplier_id` de las oportunidades. Para la evaluación real se utilizan etiquetas externas y múltiples adjudicatarios cuando corresponde.

Para reutilizar vectores:

```bash
tesis-recommender nlp-cache \
  --awards data/processed/historical_awards.csv \
  --opportunities data/processed/test_opportunities.csv \
  --output-dir artifacts/vectores_001
```

Después, añade `--embedding-cache artifacts/vectores_001` al comando `run`. Cuando exista una auditoría de disponibilidad, `--validation-report` comprueba los hashes de las entradas contra esa auditoría. Los CSV y la caché deben corresponder a los mismos datos.

## 5. Extracción documental

La lógica de extracción y comparación está en el paquete; los scripts conservan el procedimiento utilizado en el estudio. El orden es: fijar diseño y exclusiones, seleccionar las licitaciones, obtener las fuentes, comprobar identidad y cobertura, preparar la referencia, fijarla y solo entonces ejecutar la inferencia y comparar.

Las fuentes insuficientes no se convierten en casos sin requisitos. Los controles de alcance, las lecturas de documentos y la referencia deben completarse antes de presentar nuevas métricas.

Los JSON y Markdown versionados de `docs/` describen contratos y decisiones de diseño, no acreditan por sí solos que una evaluación esté terminada. Las salidas de ejecución son las que registran su estado.

## Qué se puede reproducir con esta distribución

- La instalación, la demostración sintética y las pruebas autocontenidas funcionan con los archivos del repositorio.
- Las funciones y los comandos de datos reales requieren descargar o suministrar las fuentes y las tablas indicadas.
- Las verificaciones históricas exactas requieren los mismos archivos y manifiestos originales. Esta distribución de código no incluye ese archivo de evidencia: cambios de rutas o metadatos del entorno no deben solventarse editando sus hashes.
- Los scripts de registro de lecturas y de congelamiento conservan sus dependencias entre etapas. No son comandos iniciales para una carpeta sin datos.

La copia publicada incorpora ajustes de portabilidad y presentación. No sustituye los snapshots de código vinculados mediante hash a ejecuciones previas ni modifica sus resultados.
