"""Web estática del panel: páginas ya calculadas y modelo ONNX para predecir en el navegador.

Lo que no depende de una imagen nueva se genera aquí con las vistas reales. Predecir y el
recorrido de la red se calculan en el navegador con el modelo exportado desde los pesos congelados.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import re
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from utils import datos
from utils.paths import RAIZ
from utils.space import PROHIBIDO, SIN_TEXTO, identificadores_de_test

from . import componentes, estado, motor
from .presentacion import Pagina, especificacion
from .servidor import NAVEGACION, crear_app

DESTINO = RAIZ / "despliegue" / "web"
ORO = RAIZ / "despliegue" / "web_oro.json"
ORT_VERSION = "1.30.0"
ORT_INTEGRIDAD = "sha512-q0y+JrrtukXSzsBWEMccVfqX25LRmosXHF+CaRJmg8pZClzcV7svNc4rKY3jL02Vb7QmRMDs1SigqR4CXAfKYQ=="
ORT_ARCHIVOS = ("ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm")
HTML_DE = {"resumen": "index.html", **{r: f"{r}.html" for r, _ in NAVEGACION[1:]}}
SENTINELA = "__MARCA__"


# ---------------------------------------------------------------- modelo ONNX

class Conjunto(torch.nn.Module):
    """Cinco redes y cuatro reflexiones, igual que predecir(): píxeles 0–255 [N,3,256,256] → p [redes,N]."""

    def __init__(self, redes):
        super().__init__()
        self.redes = torch.nn.ModuleList(redes)

    def forward(self, pixeles):
        x = pixeles / 255
        variantes = [x, x.flip(-1), x.flip(-2), x.flip(-2).flip(-1)]
        return torch.stack([torch.stack([torch.sigmoid(red(v)[:, 0]) for v in variantes]).mean(0)
                            for red in self.redes])


class Trayecto(torch.nn.Module):
    """Una red con sus mapas intermedios, en el mismo orden que motor.recorrido()."""

    def __init__(self, red):
        super().__init__()
        self.red = red

    def forward(self, pixeles):
        red = self.red
        vistas = red.vistas(pixeles / 255)[0]
        h = red.canales(vistas)
        mapas = [h]
        for bloque in red.backbone:
            h = bloque(h)
            mapas.append(h)
        z = torch.cat((h.mean((2, 3)), h.amax((2, 3))), 1).reshape(-1)
        oculta = torch.relu(red.tronco[1](z))
        return (vistas, *mapas, z, oculta, red._pcr(oculta[None])[0])


def _exportar(modelo, ejemplo, ruta: Path, entrada: str, salidas: list[str], dinamico: bool) -> None:
    ejes = {entrada: {0: "n"}, salidas[0]: {1: "n"}} if dinamico else None
    torch.onnx.export(modelo.eval(), (ejemplo,), ruta, input_names=[entrada], output_names=salidas,
                      dynamic_axes=ejes, opset_version=18, dynamo=True, external_data=False, verbose=False)


def exportar_onnx(destino: Path) -> dict:
    """Escribe los dos modelos y comprueba que coinciden con PyTorch antes de darlos por buenos."""
    import onnxruntime as ort

    destino.mkdir(parents=True, exist_ok=True)
    paquete = estado.paquete_imagen()
    pixeles, _ = estado.cortes_ejemplo(estado.ejemplos().index[0])
    x = torch.from_numpy(pixeles[:2]).float()
    _exportar(Conjunto(paquete.modelos), x, destino / "reddce.onnx", "pixeles", ["p"], True)
    salidas = ["vistas", "canales", "b1", "b2", "b3", "b4", "z", "oculta", "logit"]
    _exportar(Trayecto(paquete.modelos[0]), x[:1], destino / "recorrido.onnx", "pixeles", salidas, False)
    sesion = ort.InferenceSession(destino / "reddce.onnx", providers=["CPUExecutionProvider"])
    obtenido = sesion.run(None, {"pixeles": pixeles.astype(np.float32)})[0]
    esperado = motor.predecir_cortes(paquete, pixeles, list(range(len(pixeles)))).p_cortes
    if np.abs(obtenido - esperado).max() > 1e-5:
        raise RuntimeError("El modelo ONNX no coincide con PyTorch")
    return {n: hashlib.sha256((destino / n).read_bytes()).hexdigest() for n in ("reddce.onnx", "recorrido.onnx")}


def pesos_recorrido() -> dict:
    """Pesos que motor.recorrido() lee directamente de la red: conectividad, capa densa y salida."""
    red = estado.paquete_imagen().modelos[0]
    codificar = lambda t: {"forma": list(t.shape), "b64": base64.b64encode(
        t.detach().numpy().astype(np.float32).tobytes()).decode()}
    return {"conectividad": [codificar(motor._conectividad(b).detach()) for b in red.backbone],
            "lineal": codificar(red.tronco[1].weight), "cabeza": codificar(red.cabeza_pcr.weight)}


# ---------------------------------------------------------------- páginas ya calculadas

@dataclass
class Grupo:
    """Controles que afectan a los mismos bloques; cada combinación se guarda una vez."""
    campos: list[str]
    combinaciones: list[dict]
    ligero: bool = False                                   # no recalcula los visores de la red


@dataclass
class Rama:
    valores: dict = field(default_factory=dict)
    grupos: list = field(default_factory=list)
    dinamico: dict | None = None


def _producto(campo: str, opciones: list[str]) -> list[dict]:
    return [{campo: o} for o in opciones]


def ramas(pagina: str) -> tuple[str | None, dict[str, Rama]]:
    img = componentes.recurso("modelo_imagen")
    fus = componentes.recurso("modelo_fusion")
    if pagina == "resumen":
        mapa = Grupo(["inicio_mapa"], _producto("inicio_mapa", ["Realce temprano · EARLY − PRE", "Fase tardía · LATE − EARLY"]))
        return "resumen_evaluacion", {v: Rama({"resumen_evaluacion": v}, [mapa]) for v in ("Test final", "Validación cruzada")}
    if pagina == "red_imagen":
        historias = {"Media de los 5 folds": img["curvas"], **{f"Fold {int(k) + 1}": v for k, v in img["curvas_por_fold"].items() if v}}
        curvas = [{"cnn_curvas_ambito": a, **({} if e is None else {f"cnn_epoca_{a}": str(e)})}
                  for a, h in historias.items() for e in [None, *[c["epoca"] for c in h]]]
        folds = [{"cnn_folds_metrica": m, "cnn_fold": f}
                 for m in ("Pérdida de validación", "Pérdida de entrenamiento", "AUC de validación")
                 for f in ("Todos", *[f"Fold {i + 1}" for i in range(len(img["metricas"]["auc_por_fold"]))])]
        cortes = [{"paciente_de_ejemplo_train": p, **({} if k is None else {"corte": str(k)})}
                  for p in estado.ejemplos().index for k in [None, *range(int(estado.ejemplos().loc[p, "cortes"]))]]
        desarrollo = Rama({"cnn_evaluacion": "Desarrollo y arquitectura"}, [
            Grupo(["paciente_de_ejemplo_train", "corte"], cortes, ligero=True),
            Grupo(["cnn_curvas_ambito", "cnn_epoca_{cnn_curvas_ambito}"], curvas),
            Grupo(["cnn_folds_metrica", "cnn_fold"], folds),
            Grupo(["umbral"], _producto("umbral", ["Youden", "Máximo macro-F1"]))],
            dinamico={"campos": ["paciente_de_ejemplo_train", "corte"], "red": 620, "bloques": 560})
        return "cnn_evaluacion", {"Desarrollo y arquitectura": desarrollo,
                                  "Test final": Rama({"cnn_evaluacion": "Test final"})}
    if pagina == "combinado":
        umbrales = [k for k in fus["umbrales"] if k.endswith("+imagen")]
        errores = [{"datos_disponibles_del_modelo_combinado": p, "umbral_de_decision": r}
                   for p in umbrales for r in ("Youden", "Máximo macro-F1")]
        cv = Rama({"fusion_evaluacion": "Validación cruzada"}, [
            Grupo(["fusion_cruce"], _producto("fusion_cruce", ["Subtipo × volumen", "Subtipo × edad"])),
            Grupo(["datos_disponibles_del_modelo_combinado", "umbral_de_decision"], errores)])
        return "fusion_evaluacion", {"Test final": Rama({"fusion_evaluacion": "Test final"}), "Validación cruzada": cv}
    return None, {"": Rama()}


def bloques(pagina: str, valores: dict, ligero: bool = False) -> list[str]:
    """Bloques de primer nivel de una vista; ligero sustituye los visores de la red por un marcador."""
    vista = import_module(f"app.vistas.{pagina}")
    salvar = {}
    if ligero:
        marcador = lambda *a, **k: componentes.emitir_html("<iframe></iframe>")
        for nombre in ("red_neuronal", "bloques_3d"):
            salvar[nombre] = getattr(vista, nombre)
            setattr(vista, nombre, marcador)
        recorrido, bloques_red = estado.recorrido, estado.bloques
        estado.recorrido = estado.bloques = lambda *a: None
    try:
        hoja = Pagina(valores)
        hoja.renderizar(vista.mostrar)
        return [str(h) for h in hoja.raiz.hijos]
    finally:
        if ligero:
            for nombre, funcion in salvar.items():
                setattr(vista, nombre, funcion)
            estado.recorrido, estado.bloques = recorrido, bloques_red


def clave(campos: list[str], valores: dict) -> str:
    return "|".join(str(valores.get(c.format_map(valores), "")) for c in campos)


class Almacen:
    """Guarda cada contenido una sola vez, con su huella como nombre."""

    def __init__(self, carpeta: Path):
        self.carpeta = carpeta
        carpeta.mkdir(parents=True, exist_ok=True)

    def guardar(self, contenido) -> str:
        texto = json.dumps(contenido, ensure_ascii=False, separators=(",", ":"))
        nombre = hashlib.sha256(texto.encode()).hexdigest()[:16]
        (self.carpeta / f"{nombre}.json").write_text(texto, encoding="utf-8")
        return nombre


def exportar_rama(pagina: str, rama: Rama, almacen: Almacen) -> dict:
    base = bloques(pagina, rama.valores)
    salida = {"bloques": almacen.guardar(base), "grupos": [], "dinamico": None}
    ocupados: set[int] = set()
    for grupo in rama.grupos:
        variantes = [(c, bloques(pagina, {**rama.valores, **c}, grupo.ligero)) for c in grupo.combinaciones]
        if any(len(v) != len(base) for _, v in variantes):
            raise RuntimeError(f"{pagina}: {grupo.campos} cambia la estructura de la página")
        visores = {i for i, b in enumerate(base) if grupo.ligero and "<iframe" in b}
        indices = sorted({i for _, v in variantes for i, (a, b) in enumerate(zip(base, v)) if a != b} - visores)
        if ocupados & set(indices):
            raise RuntimeError(f"{pagina}: {grupo.campos} comparte bloques con otro grupo")
        ocupados |= set(indices)
        claves = {clave(grupo.campos, c): almacen.guardar({i: v[i] for i in indices}) for c, v in variantes}
        salida["grupos"].append({"campos": grupo.campos, "claves": claves})
    if rama.dinamico:
        visores = [i for i, b in enumerate(base) if "<iframe" in b]
        salida["dinamico"] = {**rama.dinamico, "indices": visores[:2]}
    return salida


def exportar_manifiesto(pagina: str, almacen: Almacen) -> dict:
    campo, opciones = ramas(pagina)
    return {"campo": campo, "ramas": {v: exportar_rama(pagina, r, almacen) for v, r in opciones.items()}}


def componer(pagina: str, valores: dict, manifiesto: dict, almacen: Almacen) -> list[str]:
    """Lo mismo que hace estatico.js en el navegador, para comprobarlo contra la vista real."""
    leer = lambda n: json.loads((almacen.carpeta / f"{n}.json").read_text(encoding="utf-8"))
    rama = manifiesto["ramas"].get(valores.get(manifiesto["campo"], ""), next(iter(manifiesto["ramas"].values())))
    salida = leer(rama["bloques"])
    for grupo in rama["grupos"]:
        k = clave(grupo["campos"], valores)
        partes = k.split("|")
        partes[-1] = ""
        for i, html in leer(grupo["claves"].get(k, grupo["claves"].get("|".join(partes), next(iter(grupo["claves"].values()))))).items():
            salida[int(i)] = html
    return salida


def verificar_composicion(paginas: dict, almacen: Almacen, n: int = 40) -> int:
    """Estados al azar: la página compuesta debe ser idéntica a la que genera Flask."""
    azar = random.Random(7)
    comprobados = 0
    for pagina, manifiesto in paginas.items():
        campo, opciones = ramas(pagina)
        for _ in range(n):
            rama = azar.choice(list(opciones.values()))
            valores = dict(rama.valores)
            for grupo in rama.grupos:
                valores.update(azar.choice(grupo.combinaciones))
            ligero = any(g.ligero for g in rama.grupos)
            esperado, obtenido = bloques(pagina, valores, ligero), componer(pagina, valores, manifiesto, almacen)
            for i, (a, b) in enumerate(zip(esperado, obtenido)):
                if a != b and not (ligero and "<iframe" in a):
                    raise RuntimeError(f"{pagina}: el bloque {i} difiere con {valores}")
            comprobados += 1
    return comprobados


# ---------------------------------------------------------------- Predecir

def _sin_histograma(info: dict) -> dict:
    return {k: v for k, v in info.items() if k != "histograma"}


def _plantilla_rango() -> str:
    hoja = Pagina({})
    hoja.renderizar(lambda p: p.rango("Corte que se propaga", options=["__VALOR__"], format_func=lambda _: "__TEXTO__"))
    return str(hoja.raiz.hijos[0])


def _distribucion(info: dict) -> dict:
    grafica = componentes.distribucion_grafica(info, SENTINELA, marca=0.3, texto_marca=SENTINELA)
    return especificacion(grafica)


def datos_prediccion(huellas: dict, plantillas: dict) -> dict:
    paquete, fusion = estado.paquete_imagen(), estado.paquete_fusion()
    img, fus = componentes.recurso("modelo_imagen"), componentes.recurso("modelo_fusion")
    ej = estado.ejemplos()
    indice = datos.cargar_indice("train")
    ejemplos = []
    for pid, f in ej.iterrows():
        filas = indice[indice["patient_id"] == pid]
        ejemplos.append({
            "id": pid, "cohorte": componentes.COHORTES.get(f["cohorte"], f["cohorte"]), "pcr": int(f["pCR"]),
            "edad": None if np.isnan(f["edad"]) else float(f["edad"]),
            "volumen": None if np.isnan(f["log_volumen"]) else float(np.expm1(f["log_volumen"])),
            "HR": None if np.isnan(f["HR"]) else float(f["HR"]), "HER2": None if np.isnan(f["HER2"]) else float(f["HER2"]),
            "cortes": [{"nombre": f"z{int(z):03d}", "rutas": [str(r[c]) for c in datos.COLUMNAS_FASE]}
                       for z, (_, r) in zip(filas["slice_index"], filas.iterrows())]})
    submodelos = {k: {c: v[c] for c in ("columnas", "media", "escala", "coeficientes", "intercepto", "interaccion")}
                  | {"umbral": v["politica"]["umbral"]} for k, v in fusion.info["submodelos"].items()}
    perfiles = {**fus["perfiles"], "imagen": img["perfiles"]}
    return {
        "modelo": {"nombre": paquete.info["nombre"], "sha256": huellas["reddce.onnx"], "sha256_recorrido": huellas["recorrido.onnx"]},
        "platt": paquete.info["politica"]["calibracion_platt"], "umbral": paquete.info["politica"]["umbral"],
        "submodelos": submodelos, "perfiles": {k: _sin_histograma(v) for k, v in perfiles.items()},
        "distribucion": {k: _distribucion(v) for k, v in perfiles.items()},
        "ejemplos": ejemplos, "plantillas": plantillas, "capas": estado.capas_red(),
        "config": paquete.info["config_modelo"], "sentinela": SENTINELA,
        "colores": {"perfiles": componentes.PERFILES, "nombres": componentes.NOMBRE_PERFIL, "tinta": componentes.TINTA},
        "magma": (motor.magma(np.arange(256) / 256)).tolist(), "pesos": pesos_recorrido()}


def plantillas_prediccion() -> dict:
    """Trozos de Predecir: los que no cambian salen de la vista real con un ejemplo y sin archivos."""
    ejemplo = bloques("prediccion", {})
    subida = bloques("prediccion", {"origen": "Subir PNG"})
    indice = lambda lista, texto: next(i for i, b in enumerate(lista) if texto in b)
    i_clinica = indice(ejemplo, "Datos clínicos")
    i_lectura = indice(ejemplo, ">Lectura<")
    i_columnas = indice(ejemplo, 'class="lectura"')
    i_red = indice(ejemplo, "Recorrido del corte por la red")
    i_origen = indice(ejemplo, 'name="origen"')
    i_final = indice(ejemplo, "Cómo interpretarlo")
    columnas = ejemplo[i_columnas]
    return {
        "cabecera": "".join(ejemplo[:i_origen]), "origen_ejemplo": ejemplo[i_origen], "origen_subida": subida[i_origen],
        "fuente_ejemplo": ejemplo[i_origen + 1], "aviso_ejemplo": ejemplo[i_origen + 2],
        "fuente_subida": subida[i_origen + 1], "vacio": next(b for b in subida if "Carga imágenes o elige" in b),
        "marco_visor": ejemplo[i_origen + 3], "seccion_clinica": ejemplo[i_clinica],
        "clinica": "".join(ejemplo[i_clinica + 1:i_lectura]), "seccion_lectura": ejemplo[i_lectura],
        "columnas_inicio": columnas[:columnas.index('<div class="columna">')],
        "seccion_red": ejemplo[i_red], "marco_red": next(b for b in ejemplo[i_red:] if "<iframe" in b),
        "rango": _plantilla_rango(), "final": "".join(ejemplo[i_final:]), "pie": subida[-1],
        "red": componentes._RED.replace("__FUENTES__", componentes.FUENTES),
        "bloques": componentes._BLOQUES.replace("__FUENTES__", componentes.FUENTES),
        "visor": componentes._VISOR.replace("__FUENTES__", componentes.FUENTES)}


def exportar_oro(destino: Path) -> None:
    """Resultados de Python para contrastar el navegador; no se publica."""
    paquete, fusion = estado.paquete_imagen(), estado.paquete_fusion()
    img, fus = componentes.recurso("modelo_imagen"), componentes.recurso("modelo_fusion")
    clinicas = [{"edad": 50.0, "volumen": 20.0, "HR": 1.0, "HER2": 0.0}, {"edad": None, "volumen": 5.0, "HR": None, "HER2": None},
                {"edad": 38.0, "volumen": None, "HR": 0.0, "HER2": 1.0}, {"edad": None, "volumen": None, "HR": 0.0, "HER2": 0.0}]
    casos = []
    for pid in estado.ejemplos().index:
        pixeles, nombres = estado.cortes_ejemplo(pid)
        pred = motor.predecir_cortes(paquete, pixeles, nombres)
        p = motor.calibrar(paquete, pred.p_cruda)
        decisiones = [fusion.decidir(pred.p_cruda, c) for c in clinicas]
        hoja = Pagina({})
        hoja.renderizar(lambda _: componentes.lectura("Solo imagen · CNN", p, img["perfiles"], paquete.info["politica"]["umbral"], "pie"))
        casos.append({"id": pid, "p_cortes": pred.p_cortes.tolist(), "p": p, "lectura": str(hoja.raiz),
                      "decisiones": [{"p": d["probabilidad_pcr"], "submodelo": d["submodelo"], "umbral": d["umbral"]} for d in decisiones]})
    pid = estado.ejemplos().index[3]
    pixeles, _ = estado.cortes_ejemplo(pid)
    k = len(pixeles) // 2
    recorrido = motor.recorrido(paquete.modelos[0], pixeles[k])
    resumen = {"p": recorrido["p"], "logit": recorrido["logit"], "aristas": recorrido["aristas"],
               "capas": [[c["id"], len(c["nodos"]), c["nodos"][0].get("v"), c["nodos"][0]["tit"]] for c in recorrido["capas"]]}
    destino.write_text(json.dumps({"clinicas": clinicas, "casos": casos, "recorrido": {"paciente": pid, "corte": k, **resumen}}, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- carpeta publicable

def _html_estatico(texto: str, pagina: str, huella: str = "") -> str:
    texto = re.sub(r'(href|src)="/(static|recursos)/', r'\1="\2/', texto)
    texto = re.sub(r'href="/?(prediccion|red_imagen|combinado|metodologia)(#[^"]*)?"', lambda m: f'href="{m[1]}.html{m[2] or ""}"', texto)
    texto = re.sub(r'href="/(#[^"]*)?"', lambda m: f'href="index.html{m[1] or ""}"', texto)
    texto = re.sub(r'action="[^"]*"', f'action="{HTML_DE[pagina]}"', texto, count=1)
    if pagina == "prediccion":
        nombre = estado.paquete_imagen().info["nombre"]
        linea = (f'<p class="texto-auxiliar" id="info-inferencia">Caso inicial ya calculado con estos mismos pesos. Al elegir otra '
                 f'paciente o subir PNG, la inferencia se ejecuta en este navegador · WebAssembly (CPU) · sin gradientes, modo '
                 f'evaluación · modelo {nombre} · SHA-256 {huella[:12]}</p>')
        texto = texto.replace('<p class="texto-auxiliar">Resultado real', linea + '<p class="texto-auxiliar">Resultado real', 1)
    guiones = f'<script>window.PAGINA_ESTATICA = "{pagina}";</script>\n  <script defer src="static/js/estatico.js"></script>\n'
    if pagina in ("prediccion", "red_imagen"):
        guiones += '  <script defer src="static/js/inferencia.js"></script>\n'
    return texto.replace('  <script defer src="static/vendor/vega.min.js">', "  " + guiones + '  <script defer src="static/vendor/vega.min.js">')


def descargar_ort(destino: Path) -> None:
    """Copia el motor de ONNX de un paquete fijado y comprobado; así la web no depende de otra CDN."""
    url = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{ORT_VERSION}.tgz"
    contenido = urllib.request.urlopen(url, timeout=120).read()
    huella = "sha512-" + base64.b64encode(hashlib.sha512(contenido).digest()).decode()
    if huella != ORT_INTEGRIDAD:
        raise RuntimeError("El paquete de ONNX Runtime no coincide con la huella fijada")
    destino.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(contenido)) as paquete:
        for nombre in ORT_ARCHIVOS:
            (destino / nombre).write_bytes(paquete.extractfile(f"package/dist/{nombre}").read())
        (destino / "package.json").write_bytes(paquete.extractfile("package/package.json").read())


def comprobar(destino: Path) -> None:
    """Detiene la preparación si queda algo individual de test o una imagen que no sea de los ejemplos."""
    ids = identificadores_de_test()
    errores = []
    for ruta in sorted(destino.rglob("*")):
        relativa = ruta.relative_to(destino).as_posix()
        if PROHIBIDO.search(relativa):
            errores.append(f"archivo prohibido: {relativa}")
        elif ruta.is_file() and ruta.suffix not in SIN_TEXTO | {".wasm", ".onnx"} and ids.search(ruta.read_bytes()):
            errores.append(f"identificador de test: {relativa}")
    esperadas = {r for e in json.loads((destino / "datos/prediccion.json").read_text(encoding="utf-8"))["ejemplos"]
                 for c in e["cortes"] for r in c["rutas"]}
    presentes = {p.relative_to(destino / "ejemplos").as_posix() for p in (destino / "ejemplos").rglob("*") if p.is_file()}
    if esperadas != presentes:
        errores.append(f"imágenes: {len(esperadas ^ presentes)} no coinciden con los ejemplos")
    if errores:
        raise ValueError("La carpeta de la web no es publicable:\n" + "\n".join(errores))


def preparar(destino: Path = DESTINO) -> dict:
    if destino.exists():
        shutil.rmtree(destino)
    app = crear_app()
    with app.test_request_context("/"):
        huellas = exportar_onnx(destino / "modelo")
        (destino / "datos").mkdir(parents=True)
        almacen = Almacen(destino / "datos/v")
        paginas = {p: exportar_manifiesto(p, almacen) for p in ("resumen", "red_imagen", "combinado")}
        estados = verificar_composicion(paginas, almacen)
        for pagina, manifiesto in paginas.items():
            (destino / f"datos/{pagina}.json").write_text(json.dumps(manifiesto, ensure_ascii=False), encoding="utf-8")
        informacion = datos_prediccion(huellas, plantillas_prediccion())
        (destino / "datos/prediccion.json").write_text(json.dumps(informacion, ensure_ascii=False), encoding="utf-8")
        exportar_oro(ORO)
    cliente = app.test_client()
    for pagina, archivo in HTML_DE.items():
        ruta = "/" if pagina == "resumen" else f"/{pagina}"
        (destino / archivo).write_text(_html_estatico(cliente.get(ruta).get_data(as_text=True), pagina, huellas["reddce.onnx"]), encoding="utf-8")
    app_dir = RAIZ / "app"
    shutil.copytree(app_dir / "static", destino / "static", ignore=shutil.ignore_patterns("__pycache__"))
    (destino / "recursos").mkdir()
    for nombre in ("icono.svg", "logo.svg", "evaluacion_test.json"):
        shutil.copy2(app_dir / "recursos" / nombre, destino / "recursos" / nombre)
    descargar_ort(destino / "static/vendor/ort")
    shutil.copy2(datos.DATOS / "LICENSE", destino / "ejemplos_LICENSE")
    for e in informacion["ejemplos"]:
        for c in e["cortes"]:
            for r in c["rutas"]:
                final = destino / "ejemplos" / r
                final.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(datos.DATOS / r, final)
    shutil.copy2(RAIZ / "despliegue/README_web.md", destino / "README.md")
    comprobar(destino)
    return {"estados": estados, "huellas": huellas}
