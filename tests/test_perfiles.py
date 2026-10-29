"""Los perfiles de pCR y los resúmenes de datos del panel deben ser agregados, coherentes y solo de train."""
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / "src")]
from utils import recursos_app  # noqa: E402

RECURSOS = RAIZ / "app" / "recursos"


class TestPerfiles(unittest.TestCase):
    def test_cinco_perfiles_con_cuotas_y_distribucion(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0.05, 0.85, 1000)
        y = (rng.uniform(size=1000) < p).astype(int)
        sub = rng.choice(["HR+/HER2−", "HER2+", "Triple negativo"], 1000)
        info = recursos_app.perfiles(y, p, sub)
        filas = info["perfiles"]
        self.assertEqual([f["perfil"] for f in filas], list(recursos_app.PERFILES))
        self.assertEqual([f["n"] for f in filas], [100, 200, 400, 200, 100])
        self.assertEqual(sum(f["pcr"] for f in filas), int(y.sum()))
        self.assertEqual(sum(sum(f["subtipos"].values()) for f in filas), 1000)
        self.assertEqual(sum(h["pcr"] + h["no_pcr"] for h in info["histograma"]), 1000)
        self.assertTrue(np.all(np.diff(info["percentiles"]) >= 0))
        self.assertEqual(len(info["percentiles"]), 101)
        for f in filas:
            self.assertAlmostEqual(f["razon"], f["tasa_pcr"] / info["prevalencia"])
            self.assertLessEqual(f["ic95"][0], f["tasa_pcr"])
            self.assertGreaterEqual(f["ic95"][1], f["tasa_pcr"])
        # Con señal real, la pCR observada crece de un perfil al siguiente.
        self.assertLess(filas[0]["tasa_pcr"], filas[-1]["tasa_pcr"])

    def test_recursos_del_panel_usan_perfiles_y_no_filas_individuales(self):
        imagen = json.loads((RECURSOS / "modelo_imagen.json").read_text(encoding="utf-8"))
        fusion = json.loads((RECURSOS / "modelo_fusion.json").read_text(encoding="utf-8"))
        self.assertNotIn("bandas", imagen)
        self.assertNotIn("bandas", fusion)
        for info in [imagen["perfiles"], *fusion["perfiles"].values()]:
            self.assertEqual(len(info["perfiles"]), 5)
            self.assertEqual(len(info["cortes"]), 4)
            self.assertEqual(sum(f["n"] for f in info["perfiles"]), info["n"])
        texto = (RECURSOS / "datos.json").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"ACRIN-6698-\d|ISPY[12][-_]\d|Breast_MRI_\d", texto))
        datos = json.loads(texto)
        self.assertEqual(datos["clinica"]["n"], imagen["n_pacientes"])
        self.assertEqual(len(datos["particiones"]), 5)
        for particion in datos["particiones"]:
            self.assertEqual(sum(particion[rol]["n"] for rol in ("fit", "seleccion", "exterior")), imagen["n_pacientes"])

    def test_lectura_situa_la_probabilidad_en_su_perfil(self):
        from app.componentes import lectura, perfil_de, percentil_de
        from app.presentacion import Pagina

        info = json.loads((RECURSOS / "modelo_fusion.json").read_text(encoding="utf-8"))["perfiles"]["subtipo+edad+volumen+imagen"]
        alto = info["cortes"][-1] + 0.01
        self.assertEqual(perfil_de(alto, info), "muy_alta")
        self.assertEqual(perfil_de(info["cortes"][0] - 0.01, info), "muy_baja")
        self.assertGreaterEqual(percentil_de(alto, info), 90)
        pagina = Pagina()
        html = pagina.renderizar(lambda _: lectura("Prueba", alto, info, 0.3, "pie"))
        self.assertIn("P5 · Muy alta", html)
        self.assertEqual(html.count('<div class="seg"'), 5)
        self.assertIn('class="escalera"', html)


if __name__ == "__main__":
    unittest.main()
