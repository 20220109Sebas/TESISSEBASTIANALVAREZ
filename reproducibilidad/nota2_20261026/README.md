# Reproducción de los resultados preliminares del artículo

Este suplemento corresponde al corte del 26 de octubre de 2026. Permite recalcular las tres tablas y los intervalos pareados publicados, sin instalar bibliotecas externas ni descargar el corpus o LaBSE. Requiere Python 3.10 o superior.

Desde la raíz del repositorio:

```bash
python reproducibilidad/nota2_20261026/verificar_resultados.py --output salida_resultados_nota2
```

La carpeta de salida debe ser nueva. El programa comprueba los hashes SHA-256 antes de calcular. Produce `tabla_1.csv`, `tabla_2.csv`, `tabla_3.csv` y `resultados_recalculados.json`. Cada tabla debe coincidir con el artículo al redondear a seis decimales. Los intervalos utilizan 5000 remuestreos pareados por perfil y semilla 42.

## Archivos

| Archivo | Contenido |
|---|---|
| `recuperacion_por_perfil.csv` | Positivos conocidos y aciertos en k=5 y k=10 para 13 perfiles y siete configuraciones |
| `extraccion_documental.csv` | TP, FP y FN por rol y campo, conservando los casos omitidos |
| `contrastes_originales.csv` | Diferencias e intervalos del contraste original |
| `contrastes_agregacion.json` | Diferencias e intervalos del ensayo de agregación por máximo |
| `ejecucion.json` | Tamaños de las colecciones, versión del modelo y reglas de cálculo |
| `procedencia.json` | Hashes de las salidas originales de las que se extrajo este suplemento |
| `codigo_contraste/`, `codigo_agregacion/` | Copias exactas del código conservado por las ejecuciones originales |
| `sha256.json` | Control de integridad de este paquete |

Los identificadores P01–P13 sustituyen a los identificadores empresariales y conservan su orden para reproducir el remuestreo. La precisión, Recall y F1 se calculan por perfil y luego se promedian; F1 no se calcula a partir de las medias de precisión y Recall. Los otros 36 perfiles reciben rankings, pero no tienen positivos conocidos para este cálculo.

La tabla documental mide concordancia con una referencia automática fijada antes de ejecutar el extractor. Sus 17 documentos evaluables proceden de una selección separada de 50 licitaciones. No mezcla sus resultados con las métricas del ranking.

## Alcance de la reproducción

Este comando recalcula resultados a partir de los conteos archivados. No vuelve a generar embeddings ni los 196 rankings completos. Para repetir esa ejecución se requieren los CSV canónicos, las auditorías de disponibilidad, las etiquetas y la caché de vectores descritos en la sección 4.6 del artículo. Los CSV masivos, los PDF y los vectores no forman parte de este suplemento. La [guía de ejecución](../../docs/ejecucion.md) describe la instalación, obtención de fuentes y ejecución con datos reales.

La demostración de `data/sample` utiliza datos sintéticos y un codificador de prueba. Sus métricas no sustituyen los resultados históricos de este suplemento. Las copias de código incluidas conservan los hashes de la corrida; los cambios de portabilidad del paquete publicado se mantienen separados.
