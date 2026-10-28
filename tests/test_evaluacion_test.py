"""Contratos del informe final: política congelada y remuestreo pareado."""
import unittest

import numpy as np

from utils.evaluacion_test import clasificacion, intervalos_auc, resumen_modelo


class TestInformeFinal(unittest.TestCase):
    def test_aplica_umbrales_por_paciente_incluida_igualdad(self):
        m = clasificacion([1, 1, 0, 0], [.3, .2, .3, .8], [.3, .4, .4, .7])
        self.assertEqual([m[k] for k in ["TP", "FN", "TN", "FP"]], [1, 1, 1, 1])
        self.assertEqual(m["sensibilidad"], .5)
        self.assertNotIn("umbral", m)

    def test_rechaza_umbral_no_finito_o_desalineado(self):
        for umbral in ([.1], [np.nan, .2]):
            with self.subTest(umbral=umbral), self.assertRaises(ValueError):
                clasificacion([0, 1], [.2, .8], umbral)

    def test_diferencia_pareada_cero_para_predicciones_identicas(self):
        y = [0, 0, 1, 1, 0, 1]
        p = [.1, .5, .7, .4, .9, .8]
        r = intervalos_auc(y, p, p, n=100)
        self.assertEqual(r["fusion"], r["imagen"])
        self.assertEqual(r["diferencia"]["delta"], 0)
        self.assertEqual(r["diferencia"]["ic95"], [0, 0])
        self.assertEqual(r, intervalos_auc(y, p, p, n=100))

    def test_curvas_y_calibracion_incluyen_probabilidades_extremas(self):
        r = resumen_modelo([0, 1, 0, 1], [0, .8, .2, 1], .5, "Prueba", [1, 1])
        self.assertEqual(r["auc"], 1)
        self.assertEqual(r["clasificacion"]["TP"], 2)
        self.assertEqual(sum(g["n"] for g in r["calibracion"]), 4)
        self.assertEqual(r["calibracion"][0]["desde"], 0)
        self.assertEqual(r["calibracion"][-1]["hasta"], 1)
        self.assertEqual(r["roc"][0], [0, 0])
        self.assertEqual(r["roc"][-1], [1, 1])


if __name__ == "__main__":
    unittest.main()
