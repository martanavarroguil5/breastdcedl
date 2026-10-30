"""Componentes HTML y controles de formulario aislados por petición HTTP."""
from contextvars import ContextVar
from dataclasses import dataclass, field
import base64
import hashlib
import html
import io
import json
import math
import re
import unicodedata
from flask import render_template
from markupsafe import Markup

_actual = ContextVar('pagina_actual')

class FinPagina(Exception):
    """Final normal de una vista que ya ha emitido su contenido."""

@dataclass
class Bloque:
    inicio: str = ''
    fin: str = ''
    hijos: list = field(default_factory=list)

    def __str__(self):
        return self.inicio + ''.join(map(str, self.hijos)) + self.fin

class Contenedor:
    def __init__(self, pagina, bloque):
        self.pagina, self.bloque = pagina, bloque

    def __enter__(self):
        self.pagina.pila.append(self.bloque)
        return self

    def __exit__(self, *exc):
        self.pagina.pila.pop()

    def __getattr__(self, nombre):
        def dentro(*args, **kwargs):
            with self:
                return getattr(self.pagina, nombre)(*args, **kwargs)
        return dentro

def identificador(texto):
    limpio = unicodedata.normalize('NFKD', str(texto)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', '_', limpio).strip('_')

class Pagina:
    def __init__(self, valores=None, archivos=None):
        self.valores = valores or {}
        self.subidos = archivos
        self.raiz = Bloque()
        self.pila = [self.raiz]
        self.contador = 0

    def renderizar(self, mostrar):
        token = _actual.set(self)
        try:
            try:
                mostrar(self)
            except FinPagina:
                pass
            return Markup(str(self.raiz))
        finally:
            _actual.reset(token)

    def agregar(self, contenido):
        self.pila[-1].hijos.append(contenido)

    def terminar(self):
        raise FinPagina()

    def columnas(self, anchos, gap='medium'):
        anchos = [1] * anchos if isinstance(anchos, int) else anchos
        pesos = ' '.join(f'minmax(0,{float(x)}fr)' for x in anchos)
        bloque = Bloque(f'<div class="columnas {gap}" style="--columnas:{pesos}">', '</div>')
        self.agregar(bloque)
        columnas = []
        for _ in anchos:
            hijo = Bloque('<div class="columna">', '</div>')
            bloque.hijos.append(hijo)
            columnas.append(Contenedor(self, hijo))
        return columnas

    def grupo(self, border=False):
        bloque = Bloque(f'<div class="grupo{" borde" if border else ""}">', '</div>')
        self.agregar(bloque)
        return Contenedor(self, bloque)

    def desplegable(self, titulo, icon=None):
        ident = identificador(titulo)
        bloque = Bloque(f'<div class="desplegable"><details id="detalle_{ident}"><summary>{html.escape(titulo)}</summary><div class="detalle">', '</div></details></div>')
        self.agregar(bloque)
        return Contenedor(self, bloque)

    def control(self, tipo, etiqueta, key=None, **datos):
        nombre = key or identificador(etiqueta)
        ident = 'campo_' + hashlib.sha256(nombre.encode()).hexdigest()[:12]
        self.agregar(render_template('control.html', tipo=tipo, etiqueta=etiqueta, nombre=nombre, ident=ident, **datos))

    def _elegir(self, etiqueta, opciones, defecto, key):
        recibido = self.valores.get(key or identificador(etiqueta))
        return next((x for x in opciones if str(x) == recibido), defecto)

    def segmentos(self, etiqueta, options, default=None, key=None, label_visibility=None):
        opciones = list(options)
        valor = self._elegir(etiqueta, opciones, default if default is not None else opciones[0], key)
        self.control('segmentos', etiqueta, key, opciones=[(str(x), str(x)) for x in opciones], valor=str(valor), ocultar=label_visibility == 'collapsed')
        return valor

    def selector(self, etiqueta, options, index=0, format_func=str, key=None):
        opciones = list(options)
        if not opciones:
            raise ValueError('No hay ejemplos disponibles en el conjunto de entrenamiento.')
        valor = self._elegir(etiqueta, opciones, opciones[min(index, len(opciones) - 1)], key)
        self.control('selector', etiqueta, key, opciones=[(str(x), str(format_func(x))) for x in opciones], valor=str(valor))
        return valor

    def rango(self, etiqueta, options, value=None, format_func=str, key=None):
        opciones = list(options)
        valor = self._elegir(etiqueta, opciones, value if value is not None else opciones[0], key)
        self.control('rango', etiqueta, key, opciones=[(str(x), str(format_func(x))) for x in opciones], indice=opciones.index(valor), valor=str(valor))
        return valor

    def interruptor(self, etiqueta, value=False, key=None):
        nombre = key or identificador(etiqueta)
        valor = self.valores.get(nombre, '1' if value else '0') == '1'
        self.control('interruptor', etiqueta, key, valor=valor)
        return valor

    def numero(self, etiqueta, min_value, max_value, value, step=1, disabled=False, help=None, key=None):
        valor = float(self.valores.get(key or identificador(etiqueta), value))
        if disabled:
            valor = value if math.isfinite(value) else min_value
        elif not math.isfinite(valor) or not min_value <= valor <= max_value:
            raise ValueError(f'{etiqueta}: introduce un valor entre {min_value:g} y {max_value:g}.')
        self.control('numero', etiqueta, key, valor=valor, minimo=min_value, maximo=max_value, paso=step, desactivado=disabled, ayuda=help)
        return valor

    def archivos(self, etiqueta, type=None, accept_multiple_files=True):
        enviados = self.subidos.getlist('archivos') if self.subidos is not None else []
        enviados = [f for f in enviados if f.filename]
        self.control('archivos', etiqueta, 'archivos', cantidad=len(enviados))
        if len(enviados) > 96:
            raise ValueError('El máximo es 32 cortes (96 archivos PNG) por predicción.')
        archivos, nombres = [], set()
        for f in enviados:
            nombre = f.filename.replace('\\', '/').rsplit('/', 1)[-1]
            if nombre.casefold() in nombres:
                raise ValueError('Hay nombres de archivo duplicados. Revisa las fases de cada corte.')
            nombres.add(nombre.casefold())
            archivo = io.BytesIO(f.read())
            archivo.name = nombre
            archivos.append(archivo)
        return archivos

    def texto_auxiliar(self, texto):
        self.agregar(f'<p class="texto-auxiliar">{html.escape(str(texto))}</p>')

    def texto(self, texto, unsafe_allow_html=False):
        valor = str(texto) if unsafe_allow_html else html.escape(str(texto))
        if valor.lstrip().startswith('<'):
            self.agregar(valor)
            return
        valor = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', valor)
        valor = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', valor)
        if valor.startswith('### '):
            self.agregar('<h3>' + valor[4:] + '</h3>')
        elif valor.startswith('- '):
            self.agregar('<ul class="lista">' + ''.join('<li>' + x[2:] + '</li>' for x in valor.splitlines() if x.startswith('- ')) + '</ul>')
        else:
            self.agregar('<p>' + valor + '</p>')

    def advertencia(self, texto):
        self.agregar(f'<div class="nota atencion" role="status">{html.escape(str(texto))}</div>')

    def error(self, texto):
        self.agregar(f'<div class="nota atencion" role="alert">{html.escape(str(texto))}</div>')

    def informacion(self, texto):
        self.agregar(f'<div class="nota info">{html.escape(str(texto))}</div>')

    def codigo(self, texto, language=None):
        self.agregar('<pre><code>' + html.escape(texto) + '</code></pre>')

    def descarga(self, etiqueta, datos, file_name, mime, key=None):
        contenido = datos.encode() if isinstance(datos, str) else datos
        enlace = f'data:{mime};base64,' + base64.b64encode(contenido).decode()
        self.agregar(f'<a class="boton" href="{html.escape(enlace, quote=True)}" download="{html.escape(file_name, quote=True)}">↓ {html.escape(etiqueta)}</a>')

def emitir_html(contenido):
    _actual.get().agregar(contenido)

def emitir_marco(codigo, height):
    pagina = _actual.get()
    pagina.contador += 1
    titulo = 'Visualización interactiva ' + str(pagina.contador)
    emitir_html(f'<iframe class="visualizacion" title="{titulo}" height="{height}" sandbox="allow-scripts allow-same-origin" loading="lazy" srcdoc="{html.escape(codigo, quote=True)}"></iframe>')

def especificacion(chart):
    salida = chart.to_dict()
    salida['width'] = 'container'
    salida['autosize'] = {'type': 'fit', 'contains': 'padding', 'resize': True}
    return salida

def emitir_grafica(chart):
    grafica = especificacion(chart)
    datos = json.dumps(grafica, ensure_ascii=False).replace('<', '\\u003c')
    alto = grafica.get('height', 300)
    alto = alto if isinstance(alto, (int, float)) else 300
    emitir_html(f'<div class="grafica" role="figure" style="min-height:{alto}px"><script type="application/json" class="grafica-datos">' + datos + '</script></div>')
