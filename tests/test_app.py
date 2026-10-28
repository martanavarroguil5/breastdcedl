"""Ejecuta cada página de la web sin navegador (requiere app/recursos y los paquetes)."""
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
APP = RAIZ / "app"
LISTO = (APP / "recursos" / "modelo_imagen.json").exists() and (RAIZ / "reports" / "modelo_elegido" / "imagen_v2").exists()


@unittest.skipUnless(LISTO, "faltan recursos de la web o el paquete de imagen")
class TestPaginas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for ruta in (str(RAIZ / "src"), str(APP)):
            if ruta not in sys.path:
                sys.path.insert(0, ruta)

    def ejecutar(self, pagina):
        from streamlit.testing.v1 import AppTest
        # El inicio solo funciona dentro de la navegación (enlaces a otras páginas).
        ruta = APP / "inicio.py" if pagina == "resumen.py" else APP / "paginas" / pagina
        at = AppTest.from_file(str(ruta), default_timeout=120)
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def test_paginas_cargan(self):
        for pagina in ("resumen.py", "red_imagen.py", "combinado.py", "metodologia.py", "prediccion.py"):
            with self.subTest(pagina=pagina):
                self.ejecutar(pagina)

    def test_prediccion_con_paciente_de_ejemplo(self):
        at = self.ejecutar("prediccion.py")
        at.segmented_control[0].set_value("Paciente de ejemplo").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        textos = " ".join(m.value for m in at.markdown)
        self.assertIn("Solo imagen · CNN", textos)
        self.assertIn("submodelo subtipo+edad+volumen+imagen", textos)
        self.assertEqual(len(at.get("iframe")), 2)  # visor y red

    def test_curvas_por_fold_y_epoca(self):
        at = self.ejecutar("red_imagen.py")
        selector = at.segmented_control(key="cnn_curvas_ambito")
        self.assertEqual(len(selector.options), 6)
        for fold in ("Fold 1", "Fold 5"):
            at.segmented_control(key="cnn_curvas_ambito").set_value(fold).run()
            at.select_slider(key=f"cnn_epoca_{fold}").set_value(1).run()
            self.assertFalse(at.exception, [e.value for e in at.exception])
            self.assertTrue(any("resume 3 redes" in c.value for c in at.caption))
            self.assertTrue(any("Época 1 · entrenamiento" in m.value for m in at.markdown))
        at.segmented_control(key="cnn_fold").set_value("Fold 3").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertTrue(any("Fold 3 · AUC" in m.value for m in at.markdown))

    def test_fusion_con_umbral_y_patron_clinico(self):
        import json
        from componentes import num

        at = self.ejecutar("combinado.py")
        at.segmented_control(key="fusion_evaluacion").set_value("Validación cruzada").run()
        self.assertEqual(len(at.get("iframe")), 1)
        at.selectbox[0].set_value("subtipo+imagen").run()
        next(c for c in at.segmented_control if "Máximo macro-F1" in c.options).set_value("Máximo macro-F1").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        recurso = json.loads((APP / "recursos/modelo_fusion.json").read_text(encoding="utf-8"))
        metrica = recurso["umbrales"]["subtipo+imagen"]["macro_f1"]
        self.assertTrue(any(f"Umbral {num(metrica['umbral'])}" in c.value for c in at.caption))
        self.assertEqual(len(recurso["diagnostico_generalizacion"]), 5)
        self.assertTrue(recurso["pr"]["completa+imagen"])
        self.assertTrue(recurso["calibracion"]["completa+imagen"])


    def test_resultados_finales_separados_de_validacion(self):
        import json
        from componentes import num, pct

        recurso = json.loads((APP / "recursos/evaluacion_test.json").read_text())
        at = self.ejecutar("combinado.py")
        self.assertEqual(at.segmented_control(key="fusion_evaluacion").value, "Test final")
        self.assertFalse(any("Máximo macro-F1" in c.options for c in at.segmented_control))
        m = recurso["modelos"]["fusion"]
        textos = " ".join(x.value for x in at.markdown)
        self.assertIn(num(m["auc"]), textos)
        self.assertIn(pct(m["clasificacion"]["sensibilidad"], 1), textos)
        self.assertIn("resultado en test", textos)
        self.assertNotIn("test pendiente", textos.lower())
        at.segmented_control(key="fusion_evaluacion").set_value("Validación cruzada").run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("iframe")), 1)
        at.segmented_control(key="fusion_evaluacion").set_value("Test final").run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("iframe")), 0)
        at = self.ejecutar("red_imagen.py")
        at.segmented_control(key="cnn_evaluacion").set_value("Test final").run()
        self.assertFalse(at.exception)
        self.assertIn(num(recurso["modelos"]["imagen"]["auc"]), " ".join(x.value for x in at.markdown))
        at = self.ejecutar("resumen.py")
        self.assertEqual(at.segmented_control(key="resumen_evaluacion").value, "Test final")
        at.segmented_control(key="resumen_evaluacion").set_value("Validación cruzada").run()
        self.assertFalse(at.exception)

if __name__ == "__main__":
    unittest.main()
