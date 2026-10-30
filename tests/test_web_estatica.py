"""La web estática debe componer lo mismo que Flask y no llevar ningún dato individual de test."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / "src")]


class TestWebEstatica(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import web_estatica
        from app.servidor import crear_app
        cls.web = web_estatica
        cls.app = crear_app()
        cls.test_id = pd.read_csv(RAIZ / "dataset/breastdcedl/metadata/patients.csv").query("split == 'test'")["pid"].iloc[0]

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / "datos").mkdir()
        (self.tmp / "ejemplos/dataset/train").mkdir(parents=True)
        (self.tmp / "ejemplos/dataset/train/a_PRE.png").write_bytes(b"png")
        (self.tmp / "datos/prediccion.json").write_text(json.dumps(
            {"ejemplos": [{"cortes": [{"rutas": ["dataset/train/a_PRE.png"]}]}]}))

    def assertBloquea(self, texto):
        with self.assertRaisesRegex(ValueError, texto):
            self.web.comprobar(self.tmp)

    def test_carpeta_limpia_pasa(self):
        self.web.comprobar(self.tmp)

    def test_bloquea_predicciones_de_test(self):
        (self.tmp / "predicciones_test.csv").write_text("p\n")
        self.assertBloquea("predicciones_test")

    def test_bloquea_identificador_de_test(self):
        (self.tmp / "datos/variante.json").write_text(json.dumps({"id": self.test_id}))
        self.assertBloquea("identificador de test")

    def test_bloquea_imagen_que_no_es_de_ejemplo(self):
        (self.tmp / "ejemplos/dataset/train/b_PRE.png").write_bytes(b"png")
        self.assertBloquea("no coinciden")

    def test_clave_sigue_al_control_que_depende_de_otro(self):
        campos = ["cnn_curvas_ambito", "cnn_epoca_{cnn_curvas_ambito}"]
        self.assertEqual(self.web.clave(campos, {"cnn_curvas_ambito": "Fold 2", "cnn_epoca_Fold 2": "3"}), "Fold 2|3")
        self.assertEqual(self.web.clave(campos, {"cnn_curvas_ambito": "Fold 2", "cnn_epoca_Fold 1": "3"}), "Fold 2|")

    def test_composicion_coincide_con_la_vista(self):
        with self.app.test_request_context("/"):
            almacen = self.web.Almacen(self.tmp / "v")
            manifiesto = self.web.exportar_manifiesto("combinado", almacen)
            self.assertGreater(self.web.verificar_composicion({"combinado": manifiesto}, almacen, n=6), 0)

    def test_pagina_usa_rutas_relativas_y_el_modo_estatico(self):
        html = self.app.test_client().get("/prediccion").get_data(as_text=True)
        salida = self.web._html_estatico(html + '<a href="combinado"></a>', "prediccion")
        self.assertIn('window.PAGINA_ESTATICA = "prediccion"', salida)
        self.assertIn('src="static/js/inferencia.js"', salida)
        self.assertIn('href="red_imagen.html"', salida)
        self.assertNotRegex(salida, r'href="(prediccion|red_imagen|combinado|metodologia)["#]')
        self.assertLess(salida.index("estatico.js"), salida.index("static/js/panel.js"))
        self.assertNotRegex(salida.split("<main")[0], r'(src|href)="/(static|recursos|prediccion)')

    def test_panel_con_servidor_sigue_pidiendo_fragmentos(self):
        panel = (RAIZ / "app/static/js/panel.js").read_text(encoding="utf-8")
        self.assertIn("window.panelEstatico ? await window.panelEstatico(datos, control) : await pedirFragmento", panel)


if __name__ == "__main__":
    unittest.main()
