# Política de ocurrencias, unidades y conflictos — desarrollo v1

14 de octubre de 2026. Política para una capa nueva de desarrollo. No modifica la consolidación particular autorizada del caso 15 ni la evaluación congelada. La diseñé después de observar la muestra antigua; no es una política preespecificada para aquel experimento. Debe incorporarse al protocolo final antes de elegir una prueba nueva.

## Identidad y conservación

Una ocurrencia es una aparición del requisito en una ubicación del documento. Su identidad incorpora OCID, hash del PDF e identificador local de predicción. Conservo todas las ocurrencias y su evidencia original, aunque se proyecten a una sola unidad.

La clave candidata de unidad contiene OCID, hash del PDF, sujeto, rol, ítem y rama. Importe, moneda y periodo no forman la identidad. No mezclo requisitos de calificación, factores de evaluación, personal o perfil del proveedor. No asigno ítems desde valores esperados.

La clave candidata no prueba por sí sola identidad semántica: también se comprueban texto, valores y alcance. Múltiples PDF de una misma licitación, o los mismos bytes bajo distintos OCID, requieren resolver identidad de fuente; no selecciono automáticamente una versión.

## Consolidación técnica permitida

Solo proyecto varias apariciones a una unidad si cumplen **todas** estas condiciones:

1. Misma clave candidata.
2. Texto completo de cláusula idéntico tras colapsar espacios, tabulaciones y saltos de línea. Conservo mayúsculas, signos, tildes, letras y cifras; no utilizo similitud semántica ni comparación solo por importe.
3. Los tres campos son idénticos, incluidos los nulos, y coinciden el estado de lectura monetaria y las advertencias.
4. Existe el mismo ítem explícito con evidencia, o, si el ítem no está identificado, coinciden las ubicaciones exactas de los encabezados de rol y sujeto. Dos ítems desconocidos no se presumen la misma unidad al cambiar de apartado.
5. No existe una ambigüedad de fuente de las indicadas arriba.

El resultado conserva todos los identificadores de las apariciones, el motivo de consolidación y sus referencias. Una fila única conserva sus nulos; no se convierte en requisito correcto o completo por ser única. Los registros idénticos aportados dos veces como entrada se rechazan como duplicación de entrada, no se presentan como dos evidencias documentales.

## Detención por conflicto o identidad no resuelta

Si hay valores incompatibles, diferencias conocido/nulo, texto distinto bajo la misma clave, advertencias incompatibles o alcance de ítem desconocido entre apartados distintos, conservo las apariciones y marco la unidad pendiente. No elijo el máximo, el mínimo, el primero ni el que coincida con una referencia. Tampoco completo un campo con el de otra aparición.

La unidad pendiente no emite un campo lógico escogido. Cualquier pendiente bloquea la proyección para comparación; no se elimina de la muestra para poder calcular F1. Esto se distingue de una abstención explícita de campo en una unidad identificada.

Para una prueba futura, cualquier decisión de identidad basada en fuente debe quedar fijada antes de inferir y sin escoger valores por el desempeño del extractor. Si aparece una ambigüedad no prevista después, debo detener el cálculo y registrar la enmienda; no presentar ese resultado como totalmente precomprometido.

## Límites

Esta es una política conservadora de identidad técnica, no una interpretación legal o semántica experta. El alcance por ítem y las condiciones no representadas pueden necesitar revisión adicional. La igualdad textual no demuestra elegibilidad de una empresa. Los importes desconocidos, discrepancias OCR, cifras frente a letras y periodos no soportados conservan sus abstenciones.

La capa no lee etiquetas, referencia de campos ni autorizaciones posteriores de la evaluación antigua. No genera métricas, no ejecuta OCR y no modifica pesos, vectores o rankings. Sus resultados sobre documentos observados son desarrollo, no validación independiente.
