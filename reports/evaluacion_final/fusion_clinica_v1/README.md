# Evaluación final del modelo combinado

**29/10/2026 · Test cerrado · Política Youden congelada.** Evaluación única de `fusion_clinica_v1` y comparación con su CNN `imagen_v2`, a partir de la misma ejecución de inferencia.

## Protocolo aplicado

- 176 pacientes de test: 53 con pCR y 123 sin pCR. Separación por paciente respecto a train comprobada.
- Cinco redes ya entrenadas, cuatro reflexiones por corte, media de probabilidades por paciente. Sin reentrenamiento.
- Fusión con las regresiones logísticas guardadas. Las 174 pacientes con clínica completa usan umbral **0,258094**; dos sin subtipo usan edad + volumen + imagen, con umbral **0,286764**. No se imputan variables.
- CNN sola con su calibración y Youden guardados. Se reutiliza la columna de imagen del test combinado; no se repite la inferencia.
- Métrica principal: ROC-AUC por paciente del combinado. IC95 % mediante 5.000 bootstrap estratificados por clase, semilla 20261029. El contraste con la CNN remuestrea las mismas pacientes.
- Ninguna búsqueda de umbral, calibración, arquitectura o pesos en test. El [protocolo y hashes](protocolo.json) se guardaron antes de ejecutar la evaluación.

## Resultados

| Métrica | Imagen + clínica | CNN sola |
|---|---:|---:|
| ROC-AUC | 0.7026 | 0.5519 |
| Average precision | 0.4924 | 0.3464 |
| Brier | 0.1895 | 0.2093 |
| BCE | 0.5586 | 0.6089 |
| IC95 % de AUC | [0.6191; 0.7796] | [0.4599; 0.6458] |
| Sensibilidad | 71.7 % | 49.1 % |
| Especificidad | 56.1 % | 57.7 % |
| Precisión de pCR | 41.3 % | 33.3 % |
| Exactitud | 60.8 % | 55.1 % |
| F1 pCR | 0.5241 | 0.3969 |
| Macro-F1 | 0.5954 | 0.5197 |
| TP | 38 | 26 |
| TN | 69 | 71 |
| FP | 54 | 52 |
| FN | 15 | 27 |

Positivo significa **pCR**, no presencia de cáncer. El combinado detecta 38 de las 53 pCR, pero también predice pCR en 54 de las 123 pacientes sin respuesta completa. Su precisión positiva es 41,3 %.

La diferencia pareada de AUC combinado − CNN es **+0,1506**, IC95 % **[+0,0394; +0,2606]**. Esta comparación no mide el aporte de la imagen sobre clínica sola; ese contraste no se ejecutó en este test.

## Cohortes

| Cohorte | Pacientes | pCR | AUC combinado | AUC CNN | FP combinado | FN combinado |
|---|---:|---:|---:|---:|---:|---:|
| duke | 42 | 9 | 0.5421 | 0.6633 | 21 | 3 |
| spy1 | 35 | 12 | 0.7790 | 0.3587 | 6 | 4 |
| spy2 | 99 | 32 | 0.7523 | 0.5513 | 27 | 8 |

El resultado combinado es más débil en Duke (0,542) que en I-SPY1 (0,779) e I-SPY2 (0,752). Son subgrupos pequeños y estas diferencias son descriptivas; no se usa su resultado para sustituir el modelo o cambiar la política por cohorte.

## Lectura y cierre

La AUC del combinado en test, 0,703, se aproxima a la media de CV anidada, 0,714. Son estimaciones en conjuntos y esquemas distintos; no se interpreta su diferencia como una prueba estadística de cambio. La discriminación sigue siendo moderada y los falsos positivos son frecuentes. La CNN sola tiene IC95 % que incluye 0,5.

El test público se consultó con versiones anteriores del proyecto: este cierre no equivale a una validación externa inédita. El intervalo bootstrap condiciona a los modelos fijos y no incorpora la búsqueda histórica. La validación privada sigue pendiente.

Se mantienen los modelos y la política previa. El test no se convierte en otra ronda de optimización. Las gráficas de calibración son descriptivas, con intervalos fijos de probabilidad; no se ajusta una nueva calibración.

## Evidencia y panel

- [Protocolo previo](protocolo.json): fecha de cierre, criterio, semillas, umbrales y hashes de los artefactos existentes.
- [Informe agregado completo](resumen.json): métricas, curvas, cohortes y huella de las predicciones.
- [Salida original del evaluador](../../modelo_elegido/fusion_clinica_v1/evaluacion_test.json).
- [Registro de la CNN](../../modelo_elegido/imagen_v2/evaluacion_test.json), derivado de la misma ejecución.
- [Recurso del panel](../../../app/recursos/evaluacion_test.json), sin identificadores ni predicciones individuales.

Inicio e Imagen + clínica muestran **Test final** por defecto. La CV se conserva en otra vista; CNN de imagen también permite consultar su test. Se puede descargar el informe agregado desde el panel. El selector alternativo de umbral permanece únicamente en desarrollo.

Reexportar las métricas y curvas desde las predicciones ya guardadas:

```bash
.venv/bin/python scripts/exportar_test.py
```

Este comando no carga imágenes ni modelos y no entrena. Valida modelos, metadatos, umbrales y la huella de las predicciones antes de exportar. Los CSV individuales quedan localmente fuera de Git. Los manifiestos `paquete.json` conservan el estado que tenían al crear los paquetes; el estado de evaluación actual lo determinan los informes y marcas de test.
