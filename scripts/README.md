# Lanzadores del proyecto

Ejecutar desde la raíz del repositorio con `.venv/bin/python scripts/<archivo>.py`.

| Archivo | Función | Datos que utiliza |
|---|---|---|
| [servir_panel.py](servir_panel.py) | Inicia el panel HTML y su servidor de inferencia | Pesos congelados, recursos agregados y PNG de entrada |
| [entrenar.py](entrenar.py) | Entrada a la CLI de entrenamiento, inferencia y evaluación; consultar `--help` | Depende del subcomando explícito |
| [verificar_modelos.py](verificar_modelos.py) | Comprueba los paquetes y la inferencia de los pesos existentes | Paquetes finales y comprobaciones de inferencia |
| [exportar_curvas_cnn.py](exportar_curvas_cnn.py) | Actualiza las curvas de la CNN en el panel | Historiales guardados de train |
| [exportar_diagnostico_fusion.py](exportar_diagnostico_fusion.py) | Actualiza curvas y diagnóstico de ajuste de la fusión | Reajusta logísticas de los folds de train; no cambia el paquete final |
| [exportar_test.py](exportar_test.py) | Reexporta el test ya cerrado al panel | Predicciones de test guardadas; sin repetir inferencia |
| [generar_figuras_experimentos.py](generar_figuras_experimentos.py) | Regenera las figuras del catálogo de experimentos | Métricas JSON agregadas; sin entrenar |
| [preparar_space.py](preparar_space.py) | Genera `despliegue/space` para Hugging Face y bloquea cualquier dato individual de test | Código versionado, pesos y 36 pacientes de ejemplo de train; no publica nada |

La lógica reutilizable de modelos, datos e inferencia permanece en `src/`; estos archivos son entradas de ejecución. Los nombres `exportar_*` indican que generan informes o recursos, y `verificar_*` identifica comprobaciones.
