"""Comprueba pesos, referencia de fusión e inferencia con entradas sintéticas."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from training.fusion import PaqueteFusion
from utils.paths import MODELOS

def main():
    paquete = PaqueteFusion(MODELOS / "fusion_clinica_v1", "cpu")
    imagen = np.random.default_rng(42).integers(0, 256, (1, 3, 256, 256), dtype=np.uint8)
    p = float(paquete.imagen.probabilidad_cortes(imagen)[0])
    assert np.isfinite(p)
    for clinica in ({}, {"edad": 50}, {"edad": 50, "volumen": 20, "HR": 1, "HER2": 0}):
        assert np.isfinite(paquete.decidir(p, clinica)["probabilidad_pcr"])
    print(json.dumps({"estado": "correcto", "redes": len(paquete.imagen.modelos),
                      "imagen": paquete.imagen.info["nombre"], "fusion": paquete.info["nombre"]},
                     ensure_ascii=False))

if __name__ == "__main__":
    main()
