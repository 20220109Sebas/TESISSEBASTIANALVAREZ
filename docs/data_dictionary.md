# Diccionario de datos canónico

## `historical_awards.csv`

| Campo | Tipo | Requerido | Uso |
|---|---|---:|---|
| `award_id` | texto | sí | Unicidad y trazabilidad |
| `supplier_id` | texto | sí | Construcción del perfil |
| `award_date` | fecha ISO | sí | Corte temporal |
| `title` | texto | sí | Señales léxica/semántica |
| `description`, `technical_text` | texto | no | Enriquecimiento NLP |
| `category`, `region` | texto | no | Señales estructuradas |
| `amount` | decimal | no | Comparación de escala solo para importes positivos y finitos |
| `currency` | código de moneda | no | Comparación limitada a la misma moneda identificada, sin asumir PEN |
| `object_type` | texto | sí | Filtro `services` |
| `source_url` | URL | recomendado | Auditoría |

## `test_opportunities.csv`

Sustituye `award_id`, `supplier_id`, `award_date` por `opportunity_id` y `publication_date`. Conserva `currency`. `awarded_supplier_id` es un campo de compatibilidad con el esquema anterior. En v3 queda vacío y los pares positivos se leen exclusivamente de labels.csv, que admite varias empresas por oportunidad.

El mapeo por campo está en [guía de anotación](annotation_fields_guide.md). Los montos históricos proceden de awards.value en la versión que acredita la adjudicación. Los montos de oportunidades proceden de tender.value en la versión textual seleccionada. La región es la dirección de la entidad compradora.

## Disponibilidad y evidencia v3

award_availability.csv registra por antecedente el OCID, la fecha del evento, text_published_at, award_published_at, rutas de las fuentes y estado de inclusión. opportunity_availability.csv registra la fecha del proceso, text_published_at, fuente, hash y estado de cada candidato.

La publicación se comprueba en release.publishedDate. No se sustituye por la fecha del evento ni por package.publishedDate. Los límites diarios se interpretan en America/Lima. Los metadatos son evidencia informada por la fuente, no una observación independiente de acceso público en la fecha histórica.

validation_report.json reúne exclusiones, cobertura útil de atributos, hashes de entradas y salidas y código de transformación. El comando run con validation-report comprueba los hashes antes del experimento.
