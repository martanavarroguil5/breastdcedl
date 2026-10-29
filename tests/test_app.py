"""Comprueba el contrato HTTP, las interacciones y la inferencia del panel HTML."""
import ast
import base64
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import sys
import unittest

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / 'src')]

class Documento(HTMLParser):
    def __init__(self, contenido):
        super().__init__()
        self.graficas, self.descargas, self.marcos = [], [], []
        self.datos = None
        self.feed(contenido)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'script' and a.get('class') == 'grafica-datos': self.datos = ''
        if tag == 'a' and 'download' in a: self.descargas.append(a['href'])
        if tag == 'iframe': self.marcos.append(a['srcdoc'])

    def handle_data(self, data):
        if self.datos is not None: self.datos += data

    def handle_endtag(self, tag):
        if tag == 'script' and self.datos is not None:
            self.graficas.append(json.loads(self.datos)); self.datos = None

class TestPanelHTML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from app.servidor import crear_app
        torch.set_num_threads(2)
        cls.app = crear_app()
        cls.app.config['TESTING'] = True
        cls.recursos = RAIZ / 'app/recursos'

    def setUp(self):
        self.cliente = self.app.test_client()

    def pagina(self, ruta, datos=None):
        r = self.cliente.get(ruta) if datos is None else self.cliente.post(ruta, data=datos)
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True)[:500])
        return r.get_data(as_text=True)

    def test_cinco_paginas_y_recursos(self):
        for ruta in ['/', '/prediccion', '/red_imagen', '/combinado', '/metodologia']:
            with self.subTest(ruta=ruta):
                texto = self.pagina(ruta)
                self.assertIn('RedDCE · pCR', texto)
                self.assertIn('<html lang="es">', texto)
                self.assertNotIn('streamlit', texto.lower())
                Documento(texto)
                if ruta == '/prediccion': self.assertIn('step="any"', texto)
        for ruta in ['/salud', '/static/css/panel.css', '/static/js/panel.js', '/static/vendor/vega.min.js', '/recursos/logo.svg']:
            with self.cliente.get(ruta) as respuesta:
                self.assertEqual(respuesta.status_code, 200)
        for ruta in ['/reports/modelo_elegido/imagen_v2/paquete.json', '/recursos/estructura_fusion.html', '/dataset', '/desconocida']:
            self.assertEqual(self.cliente.get(ruta).status_code, 404)

    def test_test_final_no_cambia_politica_y_descarga_exacta(self):
        from app.componentes import num, pct
        t = json.loads((self.recursos/'evaluacion_test.json').read_text())
        for modelo, ruta, campos in [('fusion','/combinado',{}), ('imagen','/red_imagen',{'cnn_evaluacion':'Test final'})]:
            texto = self.pagina(ruta, campos)
            self.assertIn(num(t['modelos'][modelo]['auc']), texto)
            self.assertIn(pct(t['modelos'][modelo]['clasificacion']['sensibilidad'],1), texto)
            self.assertNotIn('Máximo macro-F1', texto)
            documento = Documento(texto)
            self.assertEqual(len(documento.graficas), 4)
            self.assertEqual(json.loads(base64.b64decode(documento.descargas[0].split(',')[1])), t)

    def test_curvas_fold_epoca_y_resaltado(self):
        from app.componentes import num
        imagen=json.loads((self.recursos/'modelo_imagen.json').read_text())
        for fold in (1,5):
            texto=self.pagina('/red_imagen', {'cnn_curvas_ambito':f'Fold {fold}',f'cnn_epoca_Fold {fold}':'1','cnn_fold':'Fold 3'})
            c=imagen['curvas_por_fold'][str(fold-1)][0]
            self.assertIn('Época 1 · entrenamiento',texto)
            self.assertIn(num(c['fit_bce_obj'],4),texto)
            self.assertIn('resume 3 redes',texto)
            self.assertIn('Fold 3 · AUC',texto)
            self.assertEqual(len(Documento(texto).marcos),2)

    def test_fusion_patron_y_umbral(self):
        from app.componentes import num
        datos={'fusion_evaluacion':'Validación cruzada','datos_disponibles_del_modelo_combinado':'subtipo+imagen','umbral_de_decision':'Máximo macro-F1'}
        texto=self.pagina('/combinado',datos)
        recurso=json.loads((self.recursos/'modelo_fusion.json').read_text())
        self.assertIn('Umbral '+num(recurso['umbrales']['subtipo+imagen']['macro_f1']['umbral']),texto)
        # Estructura animada de la fusión y calculadora paso a paso.
        self.assertEqual(len(Documento(texto).marcos),2)
        self.assertIn('Brecha AUC',texto)

    def test_inicio_separa_test_y_cv(self):
        self.assertIn('Imagen + clínica · test',self.pagina('/'))
        self.assertNotIn('Imagen + clínica · test',self.pagina('/',{'resumen_evaluacion':'Validación cruzada'}))

    @staticmethod
    def png(valor=120, tamaño=(256,256)):
        from PIL import Image
        salida=io.BytesIO(); Image.new('L',tamaño,valor).save(salida,'PNG'); salida.seek(0)
        return salida

    def test_subida_png_equivale_a_inferencia_y_clinica(self):
        import numpy as np
        from app import estado, motor
        from app.componentes import num
        pixeles=np.stack([np.full((256,256),v,dtype=np.uint8) for v in [50,130,100]])[None]
        pred=estado.prediccion(pixeles,('corte',))
        esperada=motor.calibrar(estado.paquete_imagen(),pred.p_cruda)
        datos={'origen':'Subir PNG','archivos':[(self.png(v),f'corte_{fase}.png') for fase,v in zip(['PRE','EARLY','LATE'],[50,130,100])],
               'edad_conocida':'1','edad_anos':'50','volumen_conocido':'1','volumen_tumoral_cm3':'20','hr_subida':'Positivo','her2_subida':'Negativo'}
        texto=self.pagina('/prediccion',datos)
        self.assertIn(num(100*esperada,1)+'<small>%</small>',texto)
        d=estado.paquete_fusion().decidir(pred.p_cruda,{'edad':50,'volumen':20,'HR':1.,'HER2':0.})
        self.assertTrue(num(100*d['probabilidad_pcr'],1)+'<small>%</small>' in texto, 'La lectura combinada debe coincidir con los pesos guardados')
        self.assertIn('submodelo subtipo+edad+volumen+imagen',texto)
        self.assertEqual(len(Documento(texto).marcos),2)

    def test_datos_incompletos_duplicados_y_clinica_invalida(self):
        texto=self.pagina('/prediccion',{'origen':'Subir PNG','archivos':[(self.png(),'corte_PRE.png')]})
        self.assertIn('faltan fases',texto)
        r=self.cliente.post('/prediccion',data={'origen':'Subir PNG','archivos':[(self.png(),'corte_PRE.png'),(self.png(),'corte_PRE.png')]})
        self.assertEqual(r.status_code,400)
        for valor in ('nan','inf','200'):
            r=self.cliente.post('/prediccion',data={'origen':'Subir PNG','edad_conocida':'1','edad_anos':valor})
            self.assertEqual(r.status_code,400)
        r=self.cliente.post('/combinado',headers={'Origin':'https://otro.example'})
        self.assertEqual(r.status_code,403)

    def test_fragmentos_y_aislamiento_entre_clientes(self):
        r=self.cliente.post('/combinado',data={'fusion_evaluacion':'Validación cruzada'},headers={'X-Panel-Fragmento':'1'})
        self.assertEqual(r.status_code,200)
        self.assertNotIn('<!doctype',r.get_data(as_text=True))
        with self.app.test_client() as otra:
            self.assertIn('resultado en test',otra.get('/combinado').get_data(as_text=True))
        self.assertEqual(r.headers['Cache-Control'],'no-store')

    def test_html_escapa_nombres_en_visores(self):
        from app.presentacion import Pagina
        from app.componentes import visor
        with self.app.test_request_context('/'):
            p=Pagina()
            contenido=p.renderizar(lambda pagina: visor([], '</script><script>alert(1)</script>'))
            doc=Documento(str(contenido))
            self.assertNotIn('</script><script>alert(1)',doc.marcos[0])
            self.assertIn('\\u003c/script>',doc.marcos[0])

    def test_sin_dependencia_de_la_interfaz_anterior(self):
        for p in (RAIZ/'app').rglob('*.py'):
            for nodo in ast.walk(ast.parse(p.read_text())):
                if isinstance(nodo,ast.Import): self.assertFalse(any(n.name.startswith('streamlit') for n in nodo.names))
                if isinstance(nodo,ast.ImportFrom): self.assertFalse((nodo.module or '').startswith('streamlit'))
        self.assertNotIn('streamlit',sys.modules)

    def test_estructura_de_la_fusion_y_orden_de_la_cnn(self):
        for datos in (None, {'fusion_evaluacion': 'Validación cruzada'}):
            with self.subTest(datos=datos):
                texto = self.pagina('/combinado', datos)
                self.assertIn('Cómo se combinan imagen y clínica', texto)
                self.assertTrue(any('"ejemplos"' in m and '"coeficientes"' in m for m in Documento(texto).marcos))
        cnn = self.pagina('/red_imagen')
        self.assertLess(cnn.index('Recorrido de un corte por la red'), cnn.index('Pérdida de la CNN'))
        self.assertLess(cnn.index('Arquitectura en bloques'), cnn.index('Pérdida de la CNN'))
        self.assertRegex(cnn, r'<details id="detalle_[^"]+"><summary>Validación cruzada en 5 folds')

if __name__=='__main__': unittest.main()
