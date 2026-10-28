# Modelos elegidos

| Paquete | Uso | AUC media de los cinco folds |
|---|---|---:|
| `imagen_v2` | CNN de imagen DCE, conjunto de cinco semillas | 0,581 ± 0,018 |
| `fusion_clinica_v1` | Fusión tardía de imagen y datos clínicos | 0,714 ± 0,033 |

La CNN se denomina `RedDCE` en el código. Sus pesos ya están entrenados.
La fusión reutiliza los cinco pesos de imagen y añade regresiones logísticas
para los distintos patrones de datos clínicos disponibles. Sin clínica usa
el paquete de imagen. La calibración, los umbrales y las métricas se conservan.

La reorganización cambia nombres y rutas, no los valores de los pesos.
El paquete de imagen valida los SHA-256 de sus cinco archivos; la fusión
valida la huella del manifiesto de imagen con los nombres actuales.
El test de ambos paquetes se cerró el **29/10/2026**, con los pesos y
umbrales existentes. AUC: imagen 0,552 [0,460; 0,646]; fusión 0,703
[0,619; 0,780]. [Informe final](../evaluacion_final/fusion_clinica_v1/README.md).
Las marcas `evaluacion_test.json` registran el cierre; los manifiestos originales
se mantienen intactos para conservar sus huellas.

Verificación: `.venv/bin/python scripts/verificar_modelos.py`.
