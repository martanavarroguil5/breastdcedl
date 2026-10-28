# Experimentos

Selección de los ensayos que explican las principales decisiones del proyecto. Cada carpeta reúne la pregunta, el protocolo, los resultados, la decisión, figuras y evidencia agregada con procedencia verificable. No es un volcado de todas las ejecuciones.

## Catálogo

| Carpeta | Experimento | Decisión | Qué aporta |
|---|---|---|---|
| [01_realce_temporal](01_realce_temporal/README.md) | Realce temporal sobre el multimodal | Descartado | No demostró una mejora útil sobre el multimodal histórico. |
| [02_film_y_sam](02_film_y_sam/README.md) | FiLM, SAM y dependencia de la clínica | Descartados | Sin mejora concluyente; SAM apenas utilizó la rama visual. |
| [03_cnn_cinetica_multiescala](03_cnn_cinetica_multiescala/README.md) | CNN cinética multiescala | Ajuste insuficiente | La arquitectura fue un antecedente; el ajuste no alcanzó el objetivo. |
| [04_aprendizaje_multitarea](04_aprendizaje_multitarea/README.md) | Aprendizaje multitarea | Descartado | Aprender HR, HER2 y volumen no mejoró la predicción de pCR. |
| [05_regularizacion_y_presupuesto](05_regularizacion_y_presupuesto/README.md) | Regularización y presupuesto de entrenamiento | Presupuesto lineal adoptado | 12 épocas fijas y menor dispersión, sin mejora concluyente de AUC. |
| [06_fusion_tardia](06_fusion_tardia/README.md) | Fusión tardía de imagen y clínica | Adoptada | Base combinada actual; aporte sobre clínica completa aún inconcluyente. |
| [07_offset_clinico](07_offset_clinico/README.md) | Aprendizaje del residuo clínico | Descartado | Ninguna variante superó el criterio de adopción frente a la fusión. |

## Cómo interpretar los resultados

- **OOF:** cada predicción procede de un modelo entrenado sin esa paciente. La selección histórica de modelos sobre OOF todavía puede introducir optimismo.
- **AUC media de folds y AUC agrupada:** no son intercambiables. Tampoco son comparables sin más los modelos solo imagen y los que reciben clínica.
- **DE e IC95 %:** la DE describe dispersión entre folds; el IC95 % de la diferencia se obtiene por bootstrap pareado. Una diferencia cuyo intervalo incluye cero no confirma mejora ni equivalencia.
- **FP y FN:** el positivo es pCR. Los errores dependen del umbral y la calibración especificados en cada ficha.
- **Estado:** «descartado» significa que no se adoptó por sus resultados; no implica que fallara técnicamente ni que toda la familia de métodos sea inferior.
- **Test:** este catálogo usa resultados de desarrollo. El historial contiene test de versiones anteriores; no se reutiliza como confirmación de los paquetes actuales. El test de estos últimos sigue pendiente.

No se publica una clasificación conjunta de AUC entre protocolos distintos. La base vigente es [imagen_v2](../reports/modelo_elegido/imagen_v2/) y [fusion_clinica_v1](../reports/modelo_elegido/fusion_clinica_v1/).

## Contenido de cada carpeta

`README.md` explica el ensayo; `resultados.json` contiene las cifras completas y su correspondencia con los campos de origen; `metricas.csv` permite consultar la tabla; `figuras/` contiene gráficos nuevos; `fuentes/` conserva informes agregados y `procedencia.json` identifica repositorio, revisión y SHA-256.

Los primeros tres bloques se recuperaron del historial del repositorio anterior. Los cuatro posteriores documentan los experimentos que dieron lugar a la base actual, a partir de sus resúmenes ya migrados. Las fuentes se copian sin alterar sus cifras; las denominaciones de las fichas nuevas siguen las del proyecto actual. Los extractos de texto se identifican como tales.

Para regenerar las figuras desde los JSON, sin entrenar ni cargar imágenes o pesos:

```bash
.venv/bin/python experiments/generar_figuras.py
```

La figura histórica de curvas del bloque 05 se conserva como fuente y no se regenera con este comando. Las ejecuciones completas permanecen en `runs/` y en el archivo del proyecto anterior; no se duplican pesos ni datos de pacientes en este catálogo.

## Documentos y código de referencia

- [Lecciones y decisiones](LECCIONES.md): síntesis histórica, con el contexto de cada versión.
- [Protocolo de fusión](PROTOCOLO_FUSION.md) y [protocolo de offset](PROTOCOLO_OFFSET.md).
- [Pendientes](PENDIENTES.md).
- [Piloto de offset](offset.py) y [análisis](analisis/): código experimental existente.

Estos ensayos no cambian los paquetes congelados ni la selección de modelos de la aplicación.
