# Análisis de desarrollo

| Archivo | Qué analiza |
|---|---|
| [aporte_imagen.py](aporte_imagen.py) | Aporte de la puntuación visual sobre las variables clínicas |
| [comparacion_clinica.py](comparacion_clinica.py) | Comparación exploratoria de combinaciones clínicas e imagen |
| [rasgos_cineticos.py](rasgos_cineticos.py) | Rasgos cinéticos extraídos de las fases DCE |
| [curvas_epocas.py](curvas_epocas.py) | Curvas de pérdida y AUC por época |
| [evaluar_imagen.py](evaluar_imagen.py) | Informe de desarrollo de la CNN de imagen |

Son análisis históricos de train. Algunos ajustan logísticas o escriben informes al ejecutarse; no forman parte del arranque de la aplicación. Se ejecutan desde la raíz del proyecto como módulos, por ejemplo `python -m experiments.analisis.comparacion_clinica`.

Para actualizar los recursos habituales del panel, consultar los [lanzadores de scripts](../../scripts/README.md). El test final usa el informe ya guardado y tiene un exportador independiente.
