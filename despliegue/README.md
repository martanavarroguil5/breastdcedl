---
title: RedDCE pCR
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-4.0
short_description: Predicción de pCR con resonancia DCE y datos clínicos
---

# RedDCE · pCR

Panel docente que estima la respuesta patológica completa (pCR) en cáncer de
mama a partir de resonancia DCE y datos clínicos, con una CNN propia y una
fusión tardía con edad, volumen tumoral, HR y HER2. Ejecuta inferencia en CPU
con pesos congelados; no entrena.

**Uso educativo. No es un dispositivo médico** y no debe usarse para
diagnóstico ni para decidir un tratamiento.

## Datos

Los casos de ejemplo son 36 pacientes de train, con imágenes desidentificadas
derivadas de BreastDCEDL (licencia CC BY-NC 4.0, incluida en
`dataset/breastdcedl/LICENSE`). Esta copia conserva las imágenes sin modificar;
las tablas se han reducido a esas pacientes y a las columnas que usa el panel.

Las métricas de test del panel son agregadas: no se incluyen imágenes, etiquetas
ni predicciones individuales de test.
