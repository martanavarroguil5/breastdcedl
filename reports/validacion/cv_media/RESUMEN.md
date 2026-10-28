# Resumen de la validación cruzada de RedDCE

AUC por paciente (media de probabilidades de cortes y de semillas). El exterior de cada fold no se usó para entrenar ni para parar.

| Brazo | Media AUC ± DE (5 folds) | AUC agrupada | AUC dentro de cohorte | AUC HR / HER2 aux | Época elegida | Pico sel. | Brecha elegida / final | Tendencia 1–14 | Semillas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal_aux0 | 0.5842 ± 0.0355 | 0.5671 | 0.5576 | 0.501 / 0.503 | 8 | 6 | 0.064 / 0.144 | 7/15 | 3 |
| ponderada_aux0 | 0.5734 ± 0.0269 | 0.5685 | 0.5368 | 0.481 / 0.518 | 12 | 11 | 0.067 / 0.162 | 4/15 | 3 |

Referencia solo cohorte: 0.5490 ± 0.0265. Referencias históricas (otro protocolo): V007 0,5766; control 27 0,5563.

| Comparación (Δ media AUC por fold) | Δ | IC95 % pareado | P(Δ≤0) |
|---|---:|---|---:|
| ponderada_aux0 − normal_aux0 | -0.0108 | [-0.0305; +0.0079] | 0.851 |
| normal_aux0 − control cv_control | +0.0040 | [-0.0145; +0.0220] | 0.341 |
| ponderada_aux0 − control cv_control | -0.0080 | [-0.0254; +0.0088] | 0.820 |
| normal_aux0 − solo cohorte | +0.0352 | [-0.0017; +0.0714] | 0.030 |
| ponderada_aux0 − solo cohorte | +0.0244 | [-0.0136; +0.0632] | 0.096 |
