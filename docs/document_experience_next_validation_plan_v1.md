# Preparación de la próxima evaluación documental

14 de octubre de 2026. Este documento es un plan de preparación, no el protocolo final congelado ni una nueva evaluación ejecutada. La prueba anterior ya fue observada y ahora sirve para desarrollo; no puedo reutilizarla para afirmar una mejora independiente.

## 1. Cerrar o declarar los límites antes de elegir la muestra

Antes de seleccionar nuevos documentos debo fijar el comportamiento ante importes malformados, discrepancias entre cifras y letras, texto OCR incompleto y periodos no soportados. Puedo conservar abstenciones explícitas; no debo completarlas con la referencia conocida.

Debo fijar también el alcance por rol. Los requisitos del proveedor en TDR, los requisitos de calificación y los factores de evaluación no son automáticamente intercambiables. No mezclo experiencia del personal con experiencia empresarial.

## 2. Fijar la unidad lógica y los duplicados

Debo aprobar una política general **antes** de observar la nueva muestra. La consolidación particular del caso 15 anterior no autoriza una regla automática universal.

La futura política debe especificar una clave por licitación, rol, sujeto, ítem y rama; cómo relacionar menciones repetidas; qué hacer con valores incompatibles; y cuándo detener el cálculo. No debe escoger la mención que mejor coincida con la referencia ni deduplicar solo por importe. Mientras no se fije, las versiones de desarrollo conservan todas las ocurrencias.

## 3. Congelar versión y diseño

Debo registrar hashes de código y dependencias relevantes, extracción nativa/OCR, límites de acceso y lectura, regla y semilla de selección, tamaño de muestra, alcance, esquema de referencia, normalización, comparador y tratamiento de desconocidos. Se debe precisar qué automatizo y qué revisión requiere intervención humana.

No he seleccionado nuevos documentos en este paso ni he declarado la versión de desarrollo como modelo final. La decisión sobre evaluación con revisión humana o por el mismo agente debe quedar explícita. Repetir lecturas por el mismo agente no crea independencia humana o experta.

## 4. Separar documentos observados y prueba nueva

El registro de exclusiones integra la auditoría API inicial, el inventario previo, sustitutos, la selección antigua completa y los registros de acceso. Excluye por OCID incluso los casos inaccesibles o insuficientes; no solo los 27 PDF leídos. Conserva hashes conocidos de PDF para detectar los mismos bytes bajo otro OCID después de descargarlos.

Antes de seleccionar debo actualizar este registro e incluir cualquier otro documento que se haya inspeccionado después. El registro actual es una instantánea trazable de los archivos enumerados, no una garantía de que no existan observaciones fuera de ellos.

La futura selección se hará sin consultar requisitos, anotaciones o predicciones. Los fallos de acceso, duplicados de PDF, documentos fuera de alcance e insuficientes permanecerán en el denominador y se reportarán por separado, sin reemplazos silenciosos. No equivalen a ausencia de requisitos ni son negativos automáticos.

Si utilizo la misma colección anual de 2025, puedo obtener contenido documental no inspeccionado, pero no una prueba de generalización a otro año, fuente o población. El inventario de licitaciones ya participó en el experimento del ranking.

## 5. Referencia, inferencia y reporte

Después de congelar la versión, accederé de forma acotada a los documentos. La referencia debe basarse en las fuentes, con alcance, roles, ramas, ítems, desconocidos y procedencia explícitos, y quedar bloqueada antes de inferir. Cualquier dependencia del revisor y cualquier consulta a notas anteriores debe declararse.

La inferencia recibirá solo las entradas documentales acordadas, nunca valores de referencia. Persistiré todas las emisiones, evidencias, errores y abstenciones. Una ambigüedad de unidad lógica no prevista detendrá la evaluación; no se resolverá escogiendo valores por conveniencia.

Reportaré cobertura de acceso, alcance evaluable, precisión, recuperación y F1 por campo con sus denominadores, y las limitaciones. Solo atribuiré una mejora a una comparación compatible y ejecutada con su protocolo fijado. No mezclaré estas métricas con los tests del software ni con las métricas del ranking.

Los PDF recuperados en 2026 siguen sin demostrar identidad histórica con 2025. La extracción documental de desarrollo permanece fuera de los rankings, pesos y embeddings congelados.

## Preparación reproducible

`scripts/prepare_document_experience_validation_readiness_v1.py` verifica la línea de desarrollo anterior y las listas históricas, y genera `exclusion_registry.json`, `summary.json` y `manifest.json` en una carpeta nueva. No selecciona una muestra, no abre contenido nuevo y no descarga PDF. Su estado de preparación queda expresamente en falso hasta superar los cinco pasos anteriores.
