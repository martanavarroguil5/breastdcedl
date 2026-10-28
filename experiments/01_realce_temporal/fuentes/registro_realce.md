## Realce temporal: fracaso del objetivo de mejora (10/10/2026)

**Decisión: experimento cerrado y descartado; nos quedamos con
`v002_multimodal_clinico_20261007`.** No se adopta ninguno de los dos brazos.
«Fracaso» significa que el ensayo no demostró una mejora útil sobre la base
multimodal, no que el entrenamiento fallara ni que toda regularización sea inferior.

Código ejecutado: `c0f23167fd465d7929e868629a7e9d693c566541`.
Se completaron 20 ejecuciones (dos variantes × semillas 42/2026 × cinco folds),
293 épocas en total, en 61 minutos y 10 segundos. BCE ponderada por corte,
máximo 46 épocas, lote 64 y parada temprana. Solo se entrenaron las variantes
nuevas; los modelos previos se reutilizaron para comparar.

`enhancement_clinical` añadía a PRE/EARLY/LATE tres diferencias firmadas:
EARLY−PRE, LATE−PRE y LATE−EARLY. `enhancement_regularized_clinical`
añadía además dropout 0,45, Dropout2d 0,10 y weight decay 0,003, frente a
dropout 0,35 y weight decay 0,001. La rama clínica usaba edad, volumen tumoral,
HR y HER2; imputación, escala e inicialización se ajustaban con train de cada fold.

Comparación interna sobre las mismas 1.097 pacientes de desarrollo (322 pCR
y 775 no pCR), promediando las dos semillas por corte. La versión conservada
agrega cortes mediante mediana; los mejores candidatos nuevos usaron media.

| Modelo | AUC OOF | AUC media de folds | AP | Brier crudo | FP a 0,5 crudo | FN a 0,5 crudo |
|---|---:|---:|---:|---:|---:|---:|
| Multimodal v002 (conservado) | 0.724568 | 0.724311 | 0.531646 | 0.210852 | 272 | 95 |
| Realce temporal | 0.726087 | 0.728979 | 0.536726 | 0.213788 | 275 | 97 |
| Realce con mayor regularización | 0.721992 | 0.724584 | 0.524162 | 0.208895 | 261 | 104 |

El realce simple mejora la estimación de AUC OOF en solo +0,001519, con peor
Brier crudo y tres FP y dos FN adicionales a ese umbral. El brazo regularizado
reduce once FP a cambio de nueve FN adicionales y una AUC OOF menor (−0,002577).
La AUC media de folds del regularizado sube ligeramente: no se afirma que
empeore todas las métricas.

El análisis posterior pareado (2.000 bootstrap estratificados por clase,
semilla 20261010) dio IC95% de ΔAUC OOF [−0,009415, +0,011369] para realce
y [−0,012555, +0,007150] para realce regularizado. Ambos incluyen cero.
Estos intervalos condicionan a los modelos y a sus predicciones OOF: no corrigen
el optimismo de selección de checkpoints/candidatos ni equivalen a validación externa.

Con la calibración y el umbral Youden propios de cada modelo, ajustados sobre
esas mismas OOF, el realce simple produjo 211 FP y 116 FN, frente a 187 FP y
128 FN del multimodal. Son resultados aparentes con umbrales distintos;
la mayor sensibilidad se paga con más falsos positivos. Tampoco respaldan
sustituir la base actual. No se reabrió el test reservado.

Evidencia conservada:

- `resultados/04_entrenamiento/realce_temporal_20261009/comparacion/seleccion.json`
- `resultados/04_entrenamiento/realce_temporal_20261009/comparacion/oof_pacientes.csv`
- Configuraciones, resúmenes, checkpoints y curvas de los 20 runs en esa carpeta.
- `seguimiento/estado.json`, log de entrenamiento y comprobación de pesos anteriores.
- Referencia conservada: `modelos/versiones/v002_multimodal_clinico_20261007/`.

Se retiran del entrenador activo ambos perfiles de realce, sus seis canales,
Dropout2d y el lanzador del experimento. También se retiran las otras alternativas
de entrenamiento, incluida la BCE por paciente multimodal. Sus informes y
predicciones anteriores siguen disponibles; su código se recupera del commit
correspondiente, sin reinterpretar los comandos antiguos en el entrenador actual.
