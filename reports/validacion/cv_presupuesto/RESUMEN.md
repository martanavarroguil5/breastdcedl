# Resumen de la validación cruzada de RedDCE

AUC por paciente (media de probabilidades de cortes y de semillas). El exterior de cada fold no se usó para entrenar ni para parar.

| Brazo | Media AUC ± DE (5 folds) | AUC agrupada | AUC dentro de cohorte | AUC HR / HER2 aux | Época elegida | Pico sel. | Brecha elegida / final | Tendencia 1–14 | Semillas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal_aux0 | 0.5718 ± 0.0125 | 0.5611 | 0.5413 | 0.512 / 0.509 | 12 | 7 | 0.083 / 0.083 | 10/15 | 3 |
| ponderada_aux0 | 0.5814 ± 0.0183 | 0.5722 | 0.5669 | 0.520 / 0.530 | 12 | 7 | 0.094 / 0.094 | 8/15 | 3 |

Referencia solo cohorte: 0.5490 ± 0.0265. Referencias históricas (otro protocolo): V007 0,5766; control 27 0,5563.

| Comparación (Δ media AUC por fold) | Δ | IC95 % pareado | P(Δ≤0) |
|---|---:|---|---:|
| ponderada_aux0 − normal_aux0 | +0.0096 | [-0.0020; +0.0223] | 0.050 |
| normal_aux0 − control cv_media | -0.0124 | [-0.0295; +0.0045] | 0.922 |
| ponderada_aux0 − control cv_media | +0.0080 | [-0.0108; +0.0275] | 0.206 |
| normal_aux0 − solo cohorte | +0.0228 | [-0.0139; +0.0599] | 0.115 |
| ponderada_aux0 − solo cohorte | +0.0324 | [-0.0045; +0.0693] | 0.043 |
