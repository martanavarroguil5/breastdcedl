# Fusión tardía imagen + clínica: CV anidada

AUC por paciente en las 1.097 pacientes de train; ningún coeficiente vio su fold.

| Brazo | Media AUC ± DE (5 folds) | AUC agrupada | AP | Brier |
|---|---:|---:|---:|---:|
| imagen | 0.5814 ± 0.0183 | 0.5627 | 0.3396 | 0.2066 |
| subtipo | 0.6794 ± 0.0314 | 0.6706 | 0.4287 | 0.1882 |
| subtipo+imagen | 0.6991 ± 0.0308 | 0.6889 | 0.4672 | 0.1876 |
| completa | 0.7078 ± 0.0328 | 0.7062 | 0.4959 | 0.1842 |
| completa+imagen | 0.7135 ± 0.0326 | 0.7109 | 0.5055 | 0.1829 |
| completa ponderada | 0.7076 ± 0.0337 | 0.7073 | 0.4978 | 0.2171 |
| completa+imagen ponderada | 0.7121 ± 0.0318 | 0.7115 | 0.5067 | 0.2162 |
| completa+imagen×subtipo | 0.7043 ± 0.0263 | 0.7013 | 0.4994 | 0.1842 |
| completa+imagen ingenuo | 0.7126 ± 0.0322 | 0.7090 | 0.5071 | 0.1832 |

Todas las combinaciones (pérdida normal):

| Submodelo | Media AUC | AP |
|---|---:|---:|
| imagen | 0.5814 | 0.3396 |
| subtipo | 0.6794 | 0.4287 |
| subtipo+imagen | 0.6991 | 0.4672 |
| edad | 0.5425 | 0.3068 |
| edad+imagen | 0.5620 | 0.3244 |
| volumen | 0.5481 | 0.3099 |
| volumen+imagen | 0.5711 | 0.3325 |
| subtipo+edad | 0.6935 | 0.4619 |
| subtipo+edad+imagen | 0.6994 | 0.4734 |
| subtipo+volumen | 0.6996 | 0.4767 |
| subtipo+volumen+imagen | 0.7065 | 0.4880 |
| edad+volumen | 0.5633 | 0.3474 |
| edad+volumen+imagen | 0.5836 | 0.3604 |
| subtipo+edad+volumen | 0.7078 | 0.4959 |
| subtipo+edad+volumen+imagen | 0.7135 | 0.5055 |

| Comparación (Δ media AUC por fold) | Δ | IC95 % pareado | P(Δ≤0) |
|---|---:|---|---:|
| principal: completa+imagen − completa | +0.0057 | [-0.0011; +0.0126] | 0.054 |
| subtipo+imagen − subtipo | +0.0201 | [+0.0076; +0.0325] | 0.000 |
| completa+imagen − imagen | +0.1321 | [+0.0845; +0.1808] | 0.000 |
| completa+imagen ponderada − normal | -0.0014 | [-0.0033; +0.0004] | 0.938 |
| completa ponderada − normal | -0.0002 | [-0.0019; +0.0015] | 0.592 |
| exploratoria: completa+imagen×subtipo − completa+imagen | -0.0079 | [-0.0164; -0.0001] | 0.976 |
| diagnóstico: ingenuo − anidado | -0.0009 | [-0.0057; +0.0035] | 0.653 |

Coeficientes estandarizados de completa+imagen (media y rango en los 5 folds):

- HR: -0.639 [-0.670; -0.590]
- HER2: +0.446 [+0.423; +0.476]
- edad: -0.258 [-0.302; -0.143]
- log_volumen: -0.321 [-0.355; -0.258]
- imagen: +0.133 [+0.078; +0.199]

| Subtipo | Imagen | Completa | Completa+imagen |
|---|---:|---:|---:|
| HER2+ | 0.600 | 0.643 | 0.654 |
| HR+/HER2− | 0.549 | 0.555 | 0.565 |
| TN | 0.550 | 0.602 | 0.616 |
