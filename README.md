# Recomendador de licitaciones públicas de servicios

Implementación de tesis de **Sebastian Alvarez Dominguez**. El proyecto construye perfiles empresariales con adjudicaciones históricas y ordena licitaciones según su compatibilidad textual y sus atributos. Cada recomendación conserva el aporte de las señales utilizadas.

## Inicio rápido

Requiere Python 3.10 o superior. La demostración funciona con los CSV sintéticos incluidos, sin descargar el corpus histórico ni el modelo LaBSE.

```bash
git clone https://github.com/20220109Sebas/TESISSEBASTIANALVAREZ.git
cd TESISSEBASTIANALVAREZ
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
tesis-recommender demo --data-dir data/sample --output-dir artifacts/demo
python -m pytest
```

En Windows, activa el entorno con `.venv\Scripts\activate`. También puedes ejecutar la interfaz con `python -m tesis_recommender`.

La demo utiliza una representación semántica **determinista de prueba**, no LaBSE. Sirve para comprobar la instalación, el recorrido de datos y la generación de resultados; sus métricas no son resultados de la tesis.

## Qué implementa

1. Descarga y lectura de registros OCDS de OECE, integración de publicaciones de una misma licitación y verificación de disponibilidad temporal.
2. Preparación del corpus y agrupación de adjudicaciones anteriores al corte para formar perfiles empresariales.
3. Comparación léxica con BM25 y representación semántica con el modelo preentrenado LaBSE.
4. Ranking híbrido con categoría, región de la entidad compradora y monto comparable por moneda, sin convertir estos atributos en embeddings.
5. Evaluación de recuperación, explicación de puntajes y conservación de manifiestos para verificar los resultados.
6. Extracción documental de requisitos de experiencia, preparación de fuentes y controles de referencia para su evaluación separada.

El puntaje híbrido suma los aportes ponderados: `0,35 × BM25 + 0,50 × LaBSE + 0,08 × categoría + 0,04 × región + 0,03 × monto`. Es un criterio de ordenamiento, no una probabilidad de ganar. Los pesos están definidos en la implementación; `config/default.json` documenta la configuración de referencia y no se carga automáticamente como argumento de la interfaz.

## Ejecutar con LaBSE y datos reales

Instala el componente semántico y prepara los CSV descritos en el [diccionario de datos](docs/data_dictionary.md):

```bash
python -m pip install -e ".[semantic]"
tesis-recommender run \
  --awards data/processed/historical_awards.csv \
  --opportunities data/processed/test_opportunities.csv \
  --embedder labse \
  --output-dir artifacts/experimento_001
```

La primera ejecución requiere conexión para descargar `sentence-transformers/LaBSE`. El modelo se utiliza preentrenado; este proyecto no lo entrena desde cero. Las etiquetas de evaluación se proporcionan por separado con `--labels`, y no se utilizan para generar el ranking.

Las fuentes masivas, los PDF y los resultados históricos no están incorporados al repositorio. Se incluye el código para obtener y transformar las fuentes, junto con una muestra ejecutable. Consulta [ejecución y datos](docs/ejecucion.md) para conocer los requisitos y los límites de reproducción.

## Organización

| Carpeta | Contenido |
|---|---|
| `src/tesis_recommender/` | Implementación del sistema y de su interfaz de comandos |
| `scripts/` | Ejecución, auditorías y preparación documental; [guía](scripts/README.md) |
| `tests/` | Pruebas autocontenidas y verificadores de ejecuciones históricas |
| `data/sample/` | Datos sintéticos para la demostración |
| `config/` | Parámetros de referencia del estudio |
| `notebooks/` | Recorridos de demostración y exploración de calidad |
| `docs/` | Arquitectura, ejecución, diccionario y protocolos |
| `documentos/` | Ficha de investigación y artículo en PDF, con enlaces a sus originales |

## Verificación y alcance

`python -m pytest` ejecuta las pruebas autocontenidas. Los verificadores que necesitan el corpus, los PDF o las salidas archivadas se conservan, pero requieren `--run-integration` y sus entradas originales. No forman parte de la comprobación rápida de una descarga nueva. Detalles en [pruebas](tests/README.md).

El contraste de ranking y la evaluación documental son componentes distintos. La existencia de código de extracción no significa que la nueva evaluación documental esté cerrada. Los protocolos conservan sus versiones y las métricas solo deben interpretarse junto con su muestra, referencia y evidencia.

Más información: [arquitectura](docs/arquitectura.md), [ejecución](docs/ejecucion.md) y [documentos académicos](documentos/README.md).
