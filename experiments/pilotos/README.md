# Pilotos experimentales

| Módulo | Propósito | Estado |
|---|---|---|
| [offset_clinico.py](offset_clinico.py) | CNN que aprende el residuo de un predictor clínico fijo | Descartado; se conserva para reproducibilidad |

La [ficha del experimento](../07_offset_clinico/README.md) reúne resultados y figuras. El [protocolo](../protocolos/offset_clinico.md) describe el diseño fijado antes del ensayo.

Los comandos `python -m training offset-cv` y `python -m training offset-evaluar` siguen siendo sus entradas explícitas. El primero entrena redes nuevas; no hace falta ejecutarlo para consultar el experimento o usar los modelos actuales.
