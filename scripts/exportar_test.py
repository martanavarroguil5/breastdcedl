"""Exporta el test ya realizado al panel, sin inferencia ni entrenamiento."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.evaluacion_test import exportar

if __name__ == "__main__":
    print(exportar())
