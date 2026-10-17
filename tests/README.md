# Pruebas

## Comprobación de una descarga nueva

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Esta suite comprueba el código con datos creados dentro de las pruebas: lectura de CSV, cortes temporales, perfiles, señales de ranking, extracción y reglas de validación. No requiere descargar el modelo LaBSE. La integración continua ejecuta esta suite y la demostración sintética.

## Verificadores de ejecuciones históricas

Los módulos enumerados en `integration_requirements.json` comprueban archivos reales de ejecuciones archivadas, PDF, lecturas o manifiestos congelados. Se conservan como parte del código, pero no se recopilan por defecto porque sus entradas no se distribuyen con el repositorio. En el módulo mixto de evaluación documental se conservan las pruebas autocontenidas y se omiten explícitamente solo las seis pruebas de flujo que requieren esos archivos.

Solo después de restaurar las entradas y salidas de la ejecución correspondiente:

```bash
python -m pytest --run-integration
```

No es suficiente descargar datos recientes: los hashes y las rutas deben corresponder a la ejecución verificada. Algunos módulos mezclan comprobaciones de funciones con comprobaciones de archivos; se clasifican completos como integración para evitar que una descarga nueva dependa de esas fuentes.

Una suite de software aprobada no demuestra superioridad del ranking ni sustituye la revisión documental. Las referencias automáticas, las lecturas de fuentes y las anotaciones independientes deben mantenerse diferenciadas.
