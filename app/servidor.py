"""Servidor HTTP del panel HTML. Las peticiones solo consultan o predicen."""
from importlib import import_module
from pathlib import Path
from urllib.parse import urlsplit
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from .presentacion import Pagina

NAVEGACION = [('resumen', 'Inicio'), ('prediccion', 'Predecir'), ('red_imagen', 'CNN de imagen'),
             ('combinado', 'Imagen + clínica'), ('metodologia', 'Metodología')]
RAIZ = Path(__file__).resolve().parent

def crear_app():
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config.update(MAX_CONTENT_LENGTH=48 * 1024 * 1024, MAX_FORM_PARTS=180, MAX_FORM_MEMORY_SIZE=512 * 1024)

    @app.before_request
    def comprobar_origen():
        if request.method == 'POST' and request.headers.get('Origin'):
            if urlsplit(request.headers['Origin']).netloc != request.host:
                abort(403)

    @app.after_request
    def cabeceras(respuesta):
        respuesta.headers['X-Content-Type-Options'] = 'nosniff'
        respuesta.headers['Referrer-Policy'] = 'same-origin'
        if request.path != '/salud' and not request.path.startswith(('/static/', '/recursos/')):
            respuesta.headers['Cache-Control'] = 'no-store'
        return respuesta

    @app.get('/salud')
    def salud():
        return jsonify(estado='correcto', interfaz='html')

    @app.get('/recursos/<nombre>')
    def recurso(nombre):
        if nombre not in {'icono.svg', 'logo.svg', 'evaluacion_test.json'}:
            abort(404)
        return send_from_directory(RAIZ / 'recursos', nombre)

    @app.route('/', methods=['GET', 'POST'])
    @app.route('/<nombre>', methods=['GET', 'POST'])
    def pagina(nombre='resumen'):
        if nombre not in dict(NAVEGACION):
            abort(404)
        vista = import_module('.vistas.' + nombre, __package__)
        valores = request.form if request.method == 'POST' else request.args
        contenido = Pagina(valores, request.files).renderizar(vista.mostrar)
        if request.headers.get('X-Panel-Fragmento') == '1':
            return str(contenido)
        return render_template('base.html', titulo=dict(NAVEGACION)[nombre], activa=nombre,
                               navegacion=NAVEGACION, contenido=contenido)

    @app.errorhandler(ValueError)
    def datos_invalidos(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(413)
    def demasiado_grande(error):
        return jsonify(error='Los archivos superan el límite de 48 MB por petición.'), 413

    @app.errorhandler(500)
    def error_servidor(error):
        return jsonify(error='No se ha podido completar la operación. Revisa el registro del servidor.'), 500

    return app
