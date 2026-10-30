---
title: RedDCE pCR
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: cc-by-nc-4.0
short_description: Predicción de pCR con resonancia DCE y datos clínicos
---

# RedDCE · pCR

Panel docente que estima la respuesta patológica completa (pCR) en cáncer de
mama a partir de resonancia DCE y datos clínicos, con una CNN propia y una
fusión tardía con edad, volumen tumoral, HR y HER2. La predicción se calcula en
el navegador con ONNX Runtime Web, a partir de los pesos congelados: las
imágenes no salen del equipo de quien usa la página.

**Uso educativo. No es un dispositivo médico** y no tiene validez clínica.

## Datos

Los casos de ejemplo son 36 pacientes de train, con imágenes desidentificadas
derivadas de BreastDCEDL (licencia CC BY-NC 4.0, en `ejemplos_LICENSE`). Las
imágenes se conservan sin modificar.

Las métricas de test del panel son agregadas: no se incluyen imágenes, etiquetas
ni predicciones individuales de test.
