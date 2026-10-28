# Piloto offset: CNN entrenada sobre la clínica

Protocolo: `experiments/PROTOCOLO_OFFSET.md`. OOF de train, 5 folds × 3 semillas.

| Modelo | AUC media ± DE | AUC agrupada | AP | Brier |
|---|---|---|---|---|
| cnn_actual | 0.581 ± 0.018 | 0.572 | 0.343 | 0.243 |
| clinica_offset | 0.708 ± 0.033 | 0.706 | 0.496 | 0.184 |
| clinica_fusion | 0.708 ± 0.033 | 0.706 | 0.496 | 0.184 |
| fusion_v1 | 0.714 ± 0.033 | 0.711 | 0.505 | 0.183 |
| offset | 0.711 ± 0.032 | 0.711 | 0.505 | 0.183 |
| offset_subtipo | 0.708 ± 0.031 | 0.707 | 0.506 | 0.184 |
| offset|imagen | 0.559 ± 0.026 | 0.555 | 0.322 | 0.244 |
| offset_subtipo|imagen | 0.580 ± 0.041 | 0.578 | 0.355 | 0.244 |

| Comparación | Δ AUC | IC95 % | P(Δ ≤ 0) |
|---|---|---|---|
| offset − fusion_v1 | -0.0022 | [-0.0090; +0.0048] | 0.730 |
| offset − clinica_offset | +0.0036 | [-0.0083; +0.0151] | 0.287 |
| offset_subtipo − fusion_v1 | -0.0052 | [-0.0112; +0.0008] | 0.958 |
| offset_subtipo − clinica_offset | +0.0005 | [-0.0077; +0.0087] | 0.447 |
| fusion_v1 − clinica_fusion | +0.0057 | [-0.0011; +0.0126] | 0.054 |
| clinica_offset − clinica_fusion | +0.0000 | [+0.0000; +0.0000] | 1.000 |
| offset_subtipo − offset | -0.0030 | [-0.0109; +0.0051] | 0.769 |

| Grupo | n | Clínica | Fusión v1 | Offset | Offset subtipo |
|---|---|---|---|---|---|
| subtipo HER2+ | 287 | 0.643 | 0.654 | 0.658 | 0.656 |
| subtipo HR+/HER2− | 453 | 0.555 | 0.565 | 0.574 | 0.560 |
| subtipo TN | 354 | 0.602 | 0.616 | 0.621 | 0.601 |
| subtipo desconocido | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| cohorte duke | 209 | 0.743 | 0.735 | 0.730 | 0.736 |
| cohorte spy1 | 104 | 0.781 | 0.783 | 0.752 | 0.781 |
| cohorte spy2 | 784 | 0.692 | 0.695 | 0.697 | 0.690 |

## Criterio de adopción

- `offset`: Δ frente a la fusión -0.0022 (≥ +0,010), IC frente a su clínica [-0.0083; +0.0151] (excluye 0) → no cumple
- `offset_subtipo`: Δ frente a la fusión -0.0052 (≥ +0,010), IC frente a su clínica [-0.0077; +0.0087] (excluye 0) → no cumple

**Adoptado:** ninguno (sigue fusion_clinica_v1)
