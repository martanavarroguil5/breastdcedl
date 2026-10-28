# Resumen de la validación cruzada de RedDCE

AUC por paciente (media de probabilidades de cortes y de semillas). El exterior de cada fold no se usó para entrenar ni para parar.

| Brazo | Media AUC ± DE (5 folds) | AUC agrupada | AUC dentro de cohorte | AUC HR / HER2 aux | Época mediana | Semillas |
|---|---:|---:|---:|---:|---:|---:|
| normal_aux0 | 0.5802 ± 0.0247 | 0.5708 | 0.5665 | 0.509 / 0.525 | 5 | 3 |
| normal_aux1 | 0.5622 ± 0.0583 | 0.5483 | 0.5320 | 0.534 / 0.493 | 10 | 3 |
| ponderada_aux0 | 0.5814 ± 0.0292 | 0.5778 | 0.5609 | 0.525 / 0.494 | 4 | 3 |
| ponderada_aux1 | 0.5671 ± 0.0379 | 0.5636 | 0.5299 | 0.526 / 0.470 | 8 | 3 |

Referencia solo cohorte: 0.5490 ± 0.0265. Referencias históricas (otro protocolo): V007 0,5766; control 27 0,5563.

| Comparación (Δ media AUC por fold) | Δ | IC95 % pareado | P(Δ≤0) |
|---|---:|---|---:|
| normal_aux1 − normal_aux0 | -0.0180 | [-0.0415; +0.0047] | 0.936 |
| ponderada_aux1 − ponderada_aux0 | -0.0143 | [-0.0362; +0.0080] | 0.908 |
| ponderada_aux0 − normal_aux0 | +0.0012 | [-0.0119; +0.0143] | 0.427 |
| ponderada_aux1 − normal_aux1 | +0.0049 | [-0.0121; +0.0212] | 0.285 |
| normal_aux0 − solo cohorte | +0.0312 | [-0.0072; +0.0709] | 0.053 |
| normal_aux1 − solo cohorte | +0.0132 | [-0.0254; +0.0545] | 0.245 |
| ponderada_aux0 − solo cohorte | +0.0324 | [-0.0044; +0.0709] | 0.042 |
| ponderada_aux1 − solo cohorte | +0.0180 | [-0.0210; +0.0607] | 0.179 |
