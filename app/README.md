# Panel web RedDCE · pCR

La interfaz se sirve como HTML, CSS y JavaScript. Conserva las cinco páginas,
los resultados, los selectores y los visores de la versión anterior. Flask
recibe las peticiones y Waitress sirve la aplicación; la inferencia se hace
en CPU con los mismos pesos congelados. No hay entrenamiento desde la web.

## Ejecutar

Desde la raíz del repositorio, con Python 3.12:

```bash
.venv/bin/python -m pip install -r app/requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/servir_panel.py
```

Abrir `http://127.0.0.1:8501`. Para otro puerto, usar `--port 8502`. En una
conexión SSH, reenviar el puerto al equipo local. El servidor escucha en
loopback por defecto; no se publica automáticamente en Internet.

## Archivos

| Ruta | Responsabilidad |
|---|---|
| `servidor.py` | Rutas HTTP, recursos permitidos y límites de las peticiones |
| `templates/base.html` | Documento HTML, cabecera y navegación |
| `templates/control.html` | Formularios, selectores, rangos y carga de archivos |
| `templates/visores/` | Visor DCE, recorrido neuronal y arquitectura 3D |
| `static/css/` | Estilo original y controles adaptables a móvil |
| `static/js/panel.js` | Actualización de formularios y gráficas sin recargar la página |
| `static/vendor/` | Vega, Vega-Lite y Vega-Embed, con versiones, hashes y licencias |
| `presentacion.py` | Construcción de fragmentos HTML y validación de controles por petición |
| `componentes.py` | Tarjetas, tablas, perfiles de pCR y especificaciones de las gráficas |
| `vistas/` | Contenido y selección de datos de las cinco páginas |
| `vista_test.py` | Informe común del test ya cerrado |
| `estado.py` | Carga de los pesos y acceso coordinado a la inferencia |
| `motor.py` | Lectura de PNG, probabilidades y activaciones reales |
| `recursos/` | Logo y resúmenes agregados de desarrollo, datos y test |

Las páginas aceptan GET y POST. Las peticiones del navegador con
`X-Panel-Fragmento: 1` reciben solo el contenido actualizado; sin JavaScript,
los formularios permiten actualizar el documento completo. Las gráficas y
los visores interactivos necesitan JavaScript. No hace falta Node ni compilar
el frontend para ejecutar el panel.

Los PNG subidos permanecen en memoria durante la petición, sin guardarse en
disco. Se aceptan hasta 32 cortes completos (96 PNG), con un límite de 48 MB.
La página valida las tres fases y el formato de 256 × 256. El navegador
conserva los archivos seleccionados mientras se ajusta la clínica; al cambiar
a ejemplos o salir de la página se descartan. Los controles y resultados se
mantienen separados entre peticiones y no hay una sesión clínica global.

Las probabilidades se leen por cinco perfiles de pCR (muy baja, baja,
intermedia, alta y muy alta), con cortes en los percentiles 10, 30, 70 y 90
de las predicciones fuera de fold de train. Cada perfil guarda su pCR
observada, su IC95 %, la composición por subtipo y la distribución de las
probabilidades. `recursos/datos.json` reúne los mapas medios de realce, la
cinética del contraste, la clínica agregada y las particiones de los folds.
Ambos se regeneran sin entrenar ni leer test, en el equipo que conserva
`runs/`:

```bash
.venv/bin/python scripts/exportar_perfiles_y_datos.py
```

Los ejemplos y la portada requieren el dataset local en
`dataset/breastdcedl/`. Los modelos están en `reports/modelo_elegido/`; las
vistas de evaluación leen los agregados de `recursos/`. No se recalcula el test.
Las fuentes tipográficas se descargan de Google Fonts y la arquitectura 3D
usa Three.js; las librerías de las gráficas se sirven desde este repositorio.

## Verificación

```bash
.venv/bin/python -m unittest discover -s tests
```

Las pruebas del panel comprueban las cinco rutas, cambios de fold y época,
umbrales de desarrollo, separación del test, descarga del informe, carga de
PNG, variables clínicas, aislamiento entre clientes y equivalencia con la
inferencia de los paquetes guardados.
