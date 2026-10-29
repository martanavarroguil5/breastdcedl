"""Exporta los perfiles de pCR y los resúmenes de datos del panel desde train, sin entrenar ni leer test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.recursos_app import exportar_datos, exportar_perfiles

if __name__ == "__main__":
    try:
        for ruta in [*exportar_perfiles(), exportar_datos()]:
            print(ruta)
    except FileNotFoundError as error:
        raise SystemExit(str(error))
