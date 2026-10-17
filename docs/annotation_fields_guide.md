# Guía de revisión de campos de oportunidades

Esta guía describe la revisión de campos extraídos del registro OCDS. Los pasos siguientes corresponden a una anotación humana independiente. Las salidas automáticas se guardan aparte y no se presentan como juicios humanos.

El comando `annotation-prepare` genera la muestra y la plantilla de revisión. Una muestra de 50 oportunidades produce 350 juicios, siete por oportunidad. La ruta `artifacts/annotation_fields_v3` que aparece a continuación es un ejemplo de salida local; sus datos históricos no se distribuyen en el repositorio. El responsable de la anotación registra su evidencia e identidad real.

## Archivos

- sample.csv: oportunidades seleccionadas y sus valores procesados.
- field_annotations.csv: tabla que se completa durante la revisión.
- source_packets.json: paquetes originales correspondientes a la muestra.
- manifest.json: selección, semilla y hashes.

La columna release_path permite abrir el JSON local. La columna json_path indica dónde buscar. El hash identifica la versión exacta. El enlace al proceso ayuda a rastrear su procedencia, aunque la API actual puede mostrar versiones posteriores.

## Cómo completar cada fila

Leer la fuente antes de aceptar la predicción. Registrar gold_value con el valor comprobado, gold_state con el estado, annotator_id con el identificador del responsable real, reviewed_at con fecha ISO, y evidence_fragment con el fragmento del JSON que sustenta el juicio. Usar notes para incidencias.

Los estados admitidos son observed, absent, ambiguous e illegible. Un dato ausente lleva gold_value vacío. Los campos ambiguos o ilegibles se reportan aparte y no se convierten en errores o aciertos de extracción. El número cero cuenta como valor observado cuando la fuente lo contiene, aunque no sea un monto útil para el ranking.

## Regla por campo

| Campo | Evidencia y regla |
|---|---|
| title | Valor de tender.title |
| description | Valor de tender.description |
| technical_text | Por cada elemento de tender.items, concatenar su descripción, la descripción de classification y las descripciones de additionalClassifications, en ese orden. Conservar repeticiones de la fuente |
| category | Descripción de la clasificación del primer ítem. Si falta, su identificador. Si ambos faltan, mainProcurementCategory |
| amount | Valor numérico de tender.value.amount. Usar punto decimal y omitir separadores de miles |
| currency | Código de tender.value.currency. No deducirlo del país |
| region | Dirección de la entidad identificada como buyer o procuringEntity. Prioridad department, region y locality. No inferir lugar de ejecución |

Si la categoría es services o consultingservices, su extracción puede ser correcta. El ranking la trata como genérica y no le atribuye afinidad específica.

La coincidencia textual ignora mayúsculas y espacios repetidos. Los montos se normalizan como números decimales. No se consideran equivalentes dos valores monetarios en distintas monedas.

## Evaluación

```bash
.venv/bin/python -m tesis_recommender annotation-evaluate \
  --annotations artifacts/annotation_fields_v3/field_annotations.csv \
  --output-dir artifacts/extraction_evaluation_v3
```

El programa exige responsable, fecha, estado y evidencia en todas las filas. Una predicción incorrecta de un campo observado cuenta como falso positivo y falso negativo. La ausencia correctamente identificada no añade un verdadero positivo. Se reportan Precision, Recall, F1, soporte y casos no evaluables por campo.

La muestra se estratificó por mes y disponibilidad de monto positivo. Sus métricas son diagnósticas y no estiman proporcionalmente el comportamiento del corpus completo.

El generador protege una muestra que ya contenga juicios humanos. Para otra muestra se usa un directorio distinto. No se afirma doble anotación hasta que participe y quede registrada una segunda persona.
