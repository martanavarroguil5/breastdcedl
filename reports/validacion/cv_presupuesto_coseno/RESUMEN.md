# Resumen de la validación cruzada de RedDCE

AUC por paciente (media de probabilidades de cortes y de semillas). El exterior de cada fold no se usó para entrenar ni para parar.

| Brazo | Media AUC ± DE (5 folds) | AUC agrupada | AUC dentro de cohorte | AUC HR / HER2 aux | Época elegida | Pico sel. | Brecha elegida / final | Tendencia 1–14 | Semillas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal_aux0 | 0.5718 ± 0.0147 | 0.5616 | 0.5388 | 0.519 / 0.528 | 12 | 7 | 0.039 / 0.039 | 11/15 | 3 |
| ponderada_aux0 | 0.5802 ± 0.0281 | 0.5714 | 0.5604 | 0.519 / 0.548 | 12 | 6 | 0.070 / 0.070 | 10/15 | 3 |

Referencia solo cohorte: 0.5490 ± 0.0265. Referencias históricas (otro protocolo): V007 0,5766; control 27 0,5563.

| Comparación (Δ media AUC por fold) | Δ | IC95 % pareado | P(Δ≤0) |
|---|---:|---|---:|
| ponderada_aux0 − normal_aux0 | +0.0085 | [-0.0049; +0.0229] | 0.114 |
| normal_aux0 − control cv_media | -0.0124 | [-0.0286; +0.0044] | 0.926 |
| ponderada_aux0 − control cv_media | +0.0068 | [-0.0107; +0.0251] | 0.233 |
| normal_aux0 − solo cohorte | +0.0228 | [-0.0155; +0.0601] | 0.108 |
| ponderada_aux0 − solo cohorte | +0.0312 | [-0.0047; +0.0687] | 0.041 |
