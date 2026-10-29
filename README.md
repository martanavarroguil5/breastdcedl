# Predicción de pCR con resonancia DCE

Proyecto BreastDCEDL para estimar la respuesta patológica completa (pCR)
tras quimioterapia neoadyuvante. Incluye una CNN propia de imagen y una
fusión tardía con edad, volumen tumoral, HR y HER2, junto con la aplicación,
el entrenamiento y los resultados de desarrollo.

Los modelos elegidos ya están entrenados. No es necesario volver a entrenar
para usar la aplicación o realizar predicciones. Este repositorio tiene
historial nuevo y una única rama, `main`.

## Estructura

```text
app/                         Panel HTML y servidor de inferencia
  templates/                 Plantillas HTML, controles y visores
  static/                    CSS, JavaScript y librerías de gráficas
  vistas/                    Contenido y datos de cada página
  requirements.txt           Dependencias de la aplicación
dataset/                     Distribución local de BreastDCEDL (fuera de Git)
experiments/                 Catálogo de ensayos con sus métricas y figuras
  protocolos/                Diseños previos de los experimentos
  pilotos/                   Implementaciones de pilotos experimentales
  analisis/                  Análisis de desarrollo
  figuras/                   Figuras compartidas de las lecciones
reports/
  modelo_elegido/
    imagen_v2/               Cinco pesos, calibración, umbral e historias
    fusion_clinica_v1/        Fusión clínica y referencia a la CNN
  validacion/                Resúmenes y gráficas de desarrollo
  evaluacion_final/          Protocolo e informe del test cerrado
runs/                        Ejecuciones completas y caché (fuera de Git)
scripts/                     Entradas de ejecución, exportación y verificación
src/
  training/                  Red, entrenamiento, CV, fusión e inferencia
  utils/                     Datos, métricas, rutas y recursos de la web
tests/                       Pruebas de modelos, datos y aplicación
.gitignore
pyproject.toml
README.md
requirements.txt
```

Los archivos Python usan nombres descriptivos en `snake_case`; se mantienen
los nombres convencionales de Python y del proyecto (`README.md`, `__init__.py`,
`__main__.py`, `requirements.txt`). Los documentos comunes de experimentos
usan minúsculas. Los paquetes entrenados y sus fuentes históricas conservan
los nombres versionados para mantener la trazabilidad y sus hashes.

El [índice de scripts](scripts/README.md) identifica qué hace cada lanzador.
Los datos locales, la caché, las ejecuciones y el entorno virtual quedan fuera
de Git; el código, los informes y los paquetes finales se versionan.

## Modelos

`RedDCE` recibe PRE, EARLY y LATE, deriva tres diferencias firmadas y usa
tres vistas centrales con un tronco compartido. Tiene 593.716 parámetros.
Cada paciente se representa por la media de las probabilidades de sus cortes;
la inferencia promedia cinco redes y cuatro reflexiones.

| Paquete | Configuración elegida | AUC media de CV | AUC test [IC95 %] |
|---|---|---:|---:|
| `imagen_v2` | Receta `presupuesto`, 12 épocas, pérdida ponderada, sin tareas auxiliares | 0,581 ± 0,018 | 0,552 [0,460; 0,646] |
| `fusion_clinica_v1` | Regresión logística sobre la imagen y la clínica disponible | 0,714 ± 0,033 | 0,703 [0,619; 0,780] |

La imagen mejora la AUC frente a la clínica sola en +0,006, con IC95 %
[−0,001; +0,013] en desarrollo; esta diferencia no confirma una mejora.
El test público se cerró el **29/10/2026**, con Youden congelado, sin
reentrenamiento ni ajustes sobre test. El combinado obtuvo 38 TP, 69 TN,
54 FP y 15 FN en 176 pacientes. [Informe completo](reports/evaluacion_final/fusion_clinica_v1/README.md).
Los protocolos y sus limitaciones están en
[catálogo de experimentos](experiments/README.md) y en
[las lecciones del proyecto](experiments/lecciones.md).

## Instalación

Python 3.12. En este equipo el entorno `.venv` está preparado. Para instalar
en otro equipo, ejecutar desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r app/requirements.txt
python -m pip install --no-deps -e .
```

Colocar la distribución de datos en `dataset/breastdcedl/`, respetando
[dataset/README.md](dataset/README.md). Los datos, la caché y las ejecuciones
están disponibles localmente; Git conserva el código, los informes y los
pesos finales. Para instalar solo entrenamiento, usar `requirements.txt`.

## Uso de los pesos existentes

```bash
.venv/bin/python scripts/verificar_modelos.py
.venv/bin/python scripts/servir_panel.py
```

La aplicación escucha en `http://127.0.0.1:8501`. Si se ejecuta por SSH, reenviar
ese puerto desde el editor. La inferencia utiliza CPU. El navegador recibe HTML,
CSS y JavaScript; Flask y Waitress sirven las páginas y ejecutan la inferencia
con los pesos existentes. Las gráficas usan Vega y los visores conservan sus
controles interactivos. No requiere un proceso de compilación frontend.

[Arquitectura y ejecución del panel](app/README.md).

```bash
.venv/bin/python -m training predecir \
  --pre a_PRE.png --early a_EARLY.png --late a_LATE.png

.venv/bin/python -m training predecir \
  --modelo reports/modelo_elegido/fusion_clinica_v1 \
  --pre a_PRE.png --early a_EARLY.png --late a_LATE.png \
  --edad 50 --volumen 20 --hr 1 --her2 0
```

Las variables clínicas son opcionales: se selecciona el submodelo del patrón
disponible y, si faltan todas, se utiliza solo imagen.

## Exploración en la interfaz

Inicio e Imagen + clínica muestran por defecto **Test final**. La validación
cruzada permanece en una vista independiente. La página CNN de imagen
también incluye su resultado de test, calculado con la misma inferencia.
Hay matrices de confusión, ROC, PR, calibración observada y desglose por
cohorte; el umbral de test permanece fijo.

La página de imagen permite consultar pérdidas y AUC por época, seleccionar
cada fold y comparar sus resultados. La página combinada incluye un recorrido
interactivo de la fusión con sus coeficientes reales, matrices de confusión
según umbral y datos disponibles, curvas PR, calibración y diagnóstico de la
brecha entre ajuste y validación de la etapa logística.

Para actualizar esos recursos con las ejecuciones de train ya guardadas:

```bash
.venv/bin/python scripts/exportar_curvas_cnn.py
.venv/bin/python scripts/exportar_diagnostico_fusion.py
```

El primer comando lee las historias de la CNN. El segundo reajusta las
logísticas de los folds para calcular el diagnóstico; no cambia los paquetes
finales ni entrena la CNN. Ambos utilizan únicamente resultados de train.

## Entrenamiento y reproducción

Solo ejecutar estos comandos si se quiere hacer una nueva ejecución.
Las carpetas de paquetes existentes no se sobrescriben.

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m training cache
.venv/bin/python -m training humo

.venv/bin/python -m training cv --nombre cv_nueva --receta presupuesto \
  --perdidas ponderada --lambdas 0 --semillas 42 2026 7 --folds 0 1 2 3 4

.venv/bin/python -m training final --cv cv_nueva --perdida ponderada \
  --lambda-aux 0 --nombre imagen_nueva
```

`cv` reanuda las redes pendientes de una ejecución; las ya terminadas no se
repiten. Las recetas `control`, `lenta` y `media`, así como las tareas
auxiliares, se conservan para reproducir las comparaciones documentadas.

La fusión vigente usa la CV exterior `cv_presupuesto` y la CV interna
`cv_interna_fusion`, ambas disponibles en `runs/`:

```bash
.venv/bin/python -m training fusion-interna
.venv/bin/python -m training fusion-evaluar
.venv/bin/python -m training fusion-final --nombre fusion_nueva --imagen imagen_v2
```

Los comandos `offset-cv` y `offset-evaluar` reproducen el piloto descartado
de `experiments/pilotos/offset_clinico.py`. `recursos-app` regenera los resúmenes de la web
desde las ejecuciones de train. `scripts/entrenar.py` expone la misma CLI.

## Evaluación

La separación es por paciente. El fold exterior no selecciona checkpoints;
el test no decide arquitectura, calibración ni umbral. La evaluación de test
es una acción final explícita con `evaluar-test --confirmo-modelo-congelado`.
El test actual ya está evaluado. Las marcas de evaluación impiden repetirlo
accidentalmente. El test público también se consultó con versiones históricas;
no equivale a validación externa inédita. La validación privada sigue pendiente.

Para regenerar el informe y el recurso del panel desde las predicciones
guardadas, sin cargar imágenes ni ejecutar de nuevo los modelos:

```bash
.venv/bin/python scripts/exportar_test.py
```

Las predicciones individuales se conservan localmente fuera de Git. Los
informes agregados y el protocolo fijado antes de evaluar sí se versionan.
