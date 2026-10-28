"""Rutas del proyecto independientes del directorio de ejecución."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "dataset" / "breastdcedl"
CACHE = RAIZ / "runs" / "cache"
MODELOS = RAIZ / "reports" / "modelo_elegido"
