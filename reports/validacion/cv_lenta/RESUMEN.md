# Resumen de la validación cruzada de RedDCE

AUC por paciente (media de probabilidades de cortes y de semillas). El exterior de cada fold no se usó para entrenar ni para parar.

| Brazo | Media AUC ± DE (5 folds) | AUC agrupada | AUC dentro de cohorte | AUC HR / HER2 aux | Época elegida | Pico sel. | Brecha elegida / final | Tendencia 1–14 | Semillas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal_aux0 | 0.5494 ± 0.0214 | 0.5217 | 0.5215 | 0.502 / 0.482 | 12 | 7 | 0.001 / 0.079 | 14/15 | 3 |
| ponderada_aux0 | 0.5505 ± 0.0384 | 0.5328 | 0.5378 | 0.511 / 0.509 | 12 | 10 | 0.015 / 0.071 | 14/15 | 3 |

Referencia solo cohorte: 0.5490 ± 0.0265. Referencias históricas (otro protocolo): V007 0,5766; control 27 0,5563.

| Comparación (Δ media AUC por fold) | Δ | IC95 % pareado | P(Δ≤0) |
|---|---:|---|---:|
| ponderada_aux0 − normal_aux0 | +0.0011 | [-0.0296; +0.0305] | 0.473 |
| normal_aux0 − control cv_control | -0.0308 | [-0.0597; -0.0024] | 0.982 |
| ponderada_aux0 − control cv_control | -0.0309 | [-0.0604; +0.0000] | 0.974 |
| normal_aux0 − solo cohorte | +0.0004 | [-0.0378; +0.0391] | 0.472 |
| ponderada_aux0 − solo cohorte | +0.0015 | [-0.0380; +0.0414] | 0.459 |
