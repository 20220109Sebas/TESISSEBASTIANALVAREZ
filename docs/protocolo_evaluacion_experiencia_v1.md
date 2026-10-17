# Protocolo propuesto de evaluación separada de experiencia

Estado: preparado, todavía no ejecutado. La aprobación docente de revisión automática se mantiene. No se requieren anotaciones humanas ficticias ni se presenta la revisión de reviewer como juicio experto.

## 1. Fijar lo que se evalúa

La unidad es una condición explícita de experiencia del postor en una oportunidad de servicios. Se distinguirán calificación, factor de evaluación, formulario y personal. Los campos serán tipo de requisito, importe o multiplicador, moneda cuando corresponda, base de referencia, ventana temporal y condición de aplicación, incluido el caso MYPE. El presupuesto del proceso no se tratará como experiencia empresarial.

Las cláusulas aceptadas por las reglas congeladas y los candidatos de la extensión se reportarán por separado. Para evaluar a la extensión se contará cada candidato emitido, incluidos los erróneos. La etiqueta "requiere revisión" no lo elimina del denominador. Las abstenciones y los casos ilegibles se informarán expresamente.

## 2. Separar desarrollo y evaluación

Los documentos ya inspeccionados son de desarrollo, incluidos los 19 originales y los seis sustitutos. No se reutilizarán para afirmar exactitud independiente de las mejoras. La muestra nueva deberá excluir sus OCID, procesos repetidos y archivos con el mismo hash. La meta es seleccionar 50 oportunidades de servicios con un procedimiento determinista, semilla y listado guardados antes de consultar sus requisitos o las predicciones.

La selección partirá del inventario guardado, no de documentos elegidos porque tengan cláusulas fáciles. La falta de descarga se conservará como falta de cobertura. No se reemplazarán silenciosamente documentos inaccesibles por otros favorables. Si no se recupera una muestra suficiente, se informarán el tamaño real y sus límites. Consultoría de obra y obras se excluirán del resultado formal de servicios.

Antes de inferir sobre la muestra se guardarán los hashes del código, fuentes, versiones OCR, parámetros y plan de comparación. Cualquier cambio posterior dará lugar a una nueva versión de desarrollo y no a recalcular la misma prueba como si siguiera intacta.

## 3. Construir una referencia automática separada

Un proceso de revisión recibirá únicamente las fuentes y el diccionario de anotación, sin resultados del extractor ni sus puntuaciones. Cada condición de referencia deberá incluir página, fragmento literal, método de lectura, rol y valores. Cuando la fuente sea una imagen, el revisor contrastará la página renderizada y no usará únicamente el mismo texto OCR que produjo la predicción.

La referencia se guardará y bloqueará antes de unirla con las predicciones. Un segundo pase, también sin acceso a predicciones, deberá revisar las condiciones y abstenciones. Se conservarán desacuerdos, correcciones y motivos. Una fuente ilegible o un importe ambiguo no se resolverá inventando un valor. Los requisitos relativos se mantendrán relativos.

Esta ruta reduce la dependencia directa del extractor, pero una referencia creada automáticamente puede compartir sesgos o errores con él. El reporte la identificará como referencia automática, no gold standard humano ni validación experta. Si no se comprueba una referencia suficientemente fiable, el resultado será concordancia y cobertura, sin afirmar exactitud semántica independiente.

## 4. Comparar con criterios fijados

Se emparejarán condiciones por proceso, rol y fragmento fuente, sin contar varias veces una cláusula que cruza páginas. Las ramas general y MYPE se compararán por separado cuando sean distinguibles. Una cifra coincidente con rol o condición equivocados será un error de extracción.

Se calcularán Precision, Recall y F1 por campo únicamente cuando exista referencia evaluable y soporte positivo. Los campos sin soporte quedarán como no estimables. También se informarán cobertura documental, cobertura textual, abstenciones, condiciones relativas no resueltas y proporción de páginas con OCR. La muestra accesible no representará automáticamente todos los procesos del inventario.

Se compararán las reglas congeladas y la extensión sobre las mismas fuentes y referencia. No se reutilizará la concordancia anterior de cuatro campos de la API como evaluación de experiencia documental. Ninguna de estas métricas será una medida de adjudicación ni sustituirá las métricas del ranking.

## 5. Interpretar y decidir

Antes de adoptar la extensión deberá revisarse si recupera condiciones útiles sin aumentar errores de sujeto, sección, moneda, ventana o aplicabilidad. El análisis separará errores de acceso, OCR, segmentación, formulación y representación condicional. Una mejora en documentos de desarrollo no bastará para incorporarla al experimento histórico.

Aunque la extracción mejore, se mantendrá fuera del ranking si no se acredita disponibilidad del contenido antes de cada oportunidad y evidencia comparable de capacidades empresariales. Los resultados y límites se añadirán al artículo en la entrega correspondiente. Este protocolo todavía no proporciona valores de exactitud ni cierra el objetivo de evaluación de experiencia.
