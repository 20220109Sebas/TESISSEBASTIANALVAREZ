# Datos

`sample/` contiene una muestra **sintética** de adjudicaciones y licitaciones, utilizada únicamente para comprobar el sistema.

Las ejecuciones reales separan tres niveles:

- `data/raw/`: archivos originales OCDS y sus metadatos de descarga.
- `data/processed/`: tablas canónicas y controles de disponibilidad temporal.
- `artifacts/`: resultados, vectores, auditorías y manifiestos de cada ejecución.

Estas carpetas se generan localmente y se excluyen de Git. El corpus masivo y los PDF no se distribuyen como código fuente.

Las entradas principales son `historical_awards.csv`, `test_opportunities.csv` y, para evaluar, `labels.csv`. Consulta los campos y su procedencia en el [diccionario de datos](../docs/data_dictionary.md).

Los identificadores de una licitación y de sus publicaciones permiten integrar cambios sin contar cada versión como una licitación distinta. Para construir perfiles se utilizan adjudicaciones verificadas anteriores al corte; las adjudicaciones posteriores utilizadas como etiquetas no se incorporan al historial del perfil.
