"""Piezas visuales compartidas: estilo «sala de lectura», cabeceras, lecturas, red, visor y gráficas."""
from __future__ import annotations

import html
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from .presentacion import emitir_html, emitir_marco, emitir_grafica

RECURSOS = Path(__file__).resolve().parent / "recursos"

# Paleta de gráficas validada (CVD y contraste) sobre el panel oscuro #0e1217.
COBRE, ACERO, OCRE, GRIS = "#c46a46", "#3790b8", "#a8843a", "#8d949d"
TINTA, TINTA2, TENUE, LINEA, PANEL = "#ece8df", "#a9afb7", "#6b7480", "#1d252f", "#0d1116"
BANDAS = {"baja": "#4a3027", "intermedia": "#8a4e33", "alta": "#c97753"}
NOMBRE_BANDA = {"baja": "Probabilidad baja", "intermedia": "Zona intermedia", "alta": "Probabilidad alta"}
COHORTES = {"spy2": "I-SPY2", "duke": "Duke", "spy1": "I-SPY1"}
FUENTES = ("https://fonts.googleapis.com/css2?family=Geist:wght@300..700&family=Geist+Mono:wght@400..600"
           "&family=Instrument+Serif:ital@0;1&display=swap")


def recurso(nombre: str) -> dict | None:
    ruta = RECURSOS / f"{nombre}.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None




def cabecera(ojo: str, titulo: str, texto: str, meta: list[tuple[str, str]] | None = None) -> None:
    """titulo admite <em> para la palabra destacada; meta son pares (etiqueta, valor)."""
    bloque = ""
    if meta:
        bloque = '<div class="meta">' + "".join(f"<div>{k}<b>{v}</b></div>" for k, v in meta) + "</div>"
    emitir_html(f'<div class="cabecera"><div class="ojo">{ojo}</div><div class="titular">{titulo}</div>'
                f'<p class="entradilla">{texto}</p>{bloque}</div>')


def seccion(numero: str, titulo: str, texto: str | None = None) -> None:
    emitir_html(f'<div class="seccion"><span class="num">{int(numero):02d} /</span><span class="h2">{titulo}</span>'
                f'<span class="regla"></span></div>')
    if texto:
        emitir_html(f'<div class="seccion-texto">{texto}</div>')


def tarjetas(items: list[dict]) -> None:
    """items: t (título), v (valor), d (detalle), ayuda (hover), barra (0–1), barra_txt, acento."""
    partes = []
    for it in items:
        ayuda = f'<div class="ayuda">{it["ayuda"]}</div>' if it.get("ayuda") else ""
        barra = ""
        if it.get("barra") is not None:
            barra = (f'<div class="barra"><i style="width:{100 * it["barra"]:.1f}%"></i>'
                     f'<span>{it.get("barra_txt", "")}</span></div>')
        clase = "kpi acento" if it.get("acento") else "kpi"
        partes.append(f'<div class="{clase}"><div class="t">{it["t"]}</div><div class="v">{it["v"]}</div>'
                      f'<div class="d">{it.get("d", "")}</div>{barra}{ayuda}</div>')
    emitir_html(f'<div class="rejilla">{"".join(partes)}</div>')


def aviso(texto: str, tipo: str = "info") -> None:
    emitir_html(f'<div class="nota {tipo}">{texto}</div>')


def pasos(items: list[tuple[str, str]]) -> None:
    emitir_html(f'<div class="hitos" style="--n:{len(items)}">' + "".join(
        f'<div class="hito"><div class="n">{i:02d}</div><b>{t}</b><span>{d}</span></div>'
        for i, (t, d) in enumerate(items, 1)) + "</div>")


def tabla(df: pd.DataFrame) -> None:
    emitir_html('<div class="tabla-scroll">' + df.to_html(index=False, classes="limpia", border=0, escape=True) + '</div>')


def destinos(items: list[tuple[str, str, str]]) -> None:
    """items: (ruta de la página, título, descripción)."""
    emitir_html('<div class="destinos">' + "".join(
        f'<a class="destino" href="{ruta}" target="_self"><span class="num">{i:02d}</span><span class="fl">↗</span>'
        f'<span class="h">{t}</span><span class="d">{d}</span></a>'
        for i, (ruta, t, d) in enumerate(items, 1)) + "</div>")


def pct(x: float, dec: int = 0) -> str:
    return f"{100 * x:.{dec}f} %".replace(".", ",")


def num(x: float, dec: int = 3) -> str:
    return f"{x:.{dec}f}".replace(".", ",")


def banda_de(p: float, info: dict) -> str:
    c1, c2 = info["cortes"]
    return "baja" if p < c1 else ("intermedia" if p < c2 else "alta")


def chip_banda(banda: str) -> str:
    color = "#0b0e12" if banda == "alta" else TINTA
    return f'<span class="chip" style="background:{BANDAS[banda]};color:{color}">{NOMBRE_BANDA[banda]}</span>'


# ---------------------------------------------------------------- lectura de una probabilidad

def lectura(titulo: str, p: float, info: dict, umbral: float, pie: str, extra: str = "") -> None:
    """Cifra grande, banda observada en la CV y regla con las tres bandas, el umbral y la aguja."""
    filas = info["bandas"]
    hi = max(0.6, min(1.0, filas[-1]["p_max"] + 0.05))
    x = lambda v: 100 * min(max(v, 0.0), hi) / hi
    limites = [0.0, *info["cortes"], hi]
    banda = banda_de(p, info)
    fila = next(f for f in filas if f["banda"] == banda)
    segmentos = []
    for f, a, b in zip(filas, limites[:-1], limites[1:]):
        ic = f["ic95"]
        tip = (f"<b>{NOMBRE_BANDA[f['banda']]}</b> · {num(a, 2)}–{num(b, 2)}<br>En la validación cruzada: "
               f"{f['pcr']} pCR de {f['n']} pacientes ({pct(f['tasa_pcr'])}; IC95 % {pct(ic[0])}–{pct(ic[1])})")
        segmentos.append(f'<div class="seg" style="left:{x(a):.2f}%;width:{x(b) - x(a):.2f}%;background:{BANDAS[f["banda"]]}">'
                         f'<div class="tip">{tip}</div></div>')
    marcas = "".join(
        f'<div class="tick{" m" if round(v * 100) % 10 == 0 else ""}" style="left:{x(v):.2f}%">'
        + (f"<span>{num(v, 1)}</span>" if round(v * 100) % 10 == 0 else "") + "</div>"
        for v in np.arange(0, hi + 1e-9, 0.05))
    clase = "pCR" if p >= umbral else "no pCR"
    emitir_html(
        f'<div class="lectura"><div class="cab"><span class="t">{titulo}</span><span>umbral {num(umbral, 3)} → {clase}</span></div>'
        f'<div class="cifra">{num(100 * p, 1)}<small>%</small></div>'
        f'<div class="que">{chip_banda(banda)} En la validación cruzada, <b>{pct(fila["tasa_pcr"])}</b> de las pacientes '
        f'de esta banda alcanzó pCR (IC95 % {pct(fila["ic95"][0])}–{pct(fila["ic95"][1])}).{extra}</div>'
        f'<div class="regla-p">{"".join(segmentos)}{marcas}'
        f'<div class="umbral {"izq" if p >= umbral else "der"}" style="left:{x(umbral):.2f}%"><span>UMBRAL</span></div>'
        f'<div class="aguja" style="left:{x(p):.2f}%"></div></div>'
        f'<div class="pie-l">{pie}</div></div>')


def lectura_vacia(titulo: str, texto: str) -> None:
    emitir_html(f'<div class="lectura vacia"><div class="cab"><span class="t">{titulo}</span></div>'
                f'<div class="cifra gris">—</div><div class="que">{texto}</div></div>')


# ---------------------------------------------------------------- portada

def portada(img: dict, titulo: str, texto: str, enlaces: list[tuple[str, str, bool]]) -> None:
    botones = "".join(f'<a class="{"prim" if prim else ""}" href="{ruta}" target="_self">{t} <span class="fl">→</span></a>'
                      for ruta, t, prim in enlaces)
    cohorte = COHORTES.get(img["cohorte"], img["cohorte"])
    fases = "".join(f'<div><img src="{img[f]}" alt="{f}"><span>{f}</span></div>' for f in ("PRE", "EARLY", "LATE"))
    emitir_html(f"""
<div class="portada">
  <div>
    <div class="ojo">Trabajo académico · Cáncer de mama · DCE-MRI</div>
    <div class="titular" style="font-size:clamp(2.4rem,4.2vw,3.7rem);line-height:1.04">{titulo}</div>
    <p class="entradilla">{texto}</p>
    <div class="cta">{botones}</div>
  </div>
  <div>
    <div class="visor-p">
      <img src="{img['EARLY']}" alt="Fase EARLY"><img class="r" src="{img['REALCE']}" alt="Mapa de realce">
      <div class="scan"></div><div class="mira"></div>
      <div class="esq tl">{cohorte} · {html.escape(img['paciente'])}<br>corte {img['corte']}</div>
      <div class="esq tr">DCE-MRI · T1<br><span class="o">EARLY + realce</span></div>
      <div class="esq bl">256 × 256 · 8 bit</div>
      <div class="esq br">paciente de train</div>
    </div>
    <div class="fases-p">{fases}</div>
  </div>
</div>""")


# ---------------------------------------------------------------- red neuronal animada

_RED = (Path(__file__).parent / "templates/visores/red.html").read_text()


def red_neuronal(datos: dict, alto: int = 600) -> None:
    """Red con conexiones: mapas de activación reales de un corte y pesos efectivos entre capas."""
    codigo = (_RED.replace("__FUENTES__", FUENTES).replace("__ALTO__", str(alto))
              .replace("__DATOS__", json.dumps(datos, ensure_ascii=False).replace("<", "\\u003c")))
    emitir_marco(codigo, height=alto + 4)


# ---------------------------------------------------------------- arquitectura en bloques 3D

_BLOQUES = (Path(__file__).parent / "templates/visores/bloques.html").read_text()


def bloques_3d(capas: list[dict], alto: int = 560) -> None:
    """Arquitectura en bloques 3D (three.js): mapas reales como texturas, conos de convolución y hover."""
    codigo = (_BLOQUES.replace("__FUENTES__", FUENTES).replace("__ALTO__", str(alto))
              .replace("__DATOS__", json.dumps(capas, ensure_ascii=False).replace("<", "\\u003c")))
    emitir_marco(codigo, height=alto + 4)


# ---------------------------------------------------------------- visor de cortes

_VISOR = (Path(__file__).parent / "templates/visores/cortes.html").read_text()


def visor(cortes: list[dict], titulo: str, alto: int = 500, inicial: int = 0) -> None:
    """Visor tipo estación de lectura: fases, mapa de realce, ventanas de la red y p por corte."""
    codigo = (_VISOR.replace("__FUENTES__", FUENTES).replace("__ALTO__", str(alto)).replace("__LADO__", str(alto))
              .replace("__CORTES__", json.dumps(cortes).replace("<", "\\u003c")).replace("__META__", json.dumps({"titulo": titulo}).replace("<", "\\u003c"))
              .replace("__INICIAL__", str(inicial)))
    emitir_marco(codigo, height=alto + 4)


# ---------------------------------------------------------------- gráficas (Altair, con hover)

def _base(grafica: alt.Chart, alto: int = 300) -> alt.Chart:
    return (grafica.properties(height=alto, background="transparent", padding={"left": 4, "right": 8, "top": 6, "bottom": 4})
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor="#8d949d", titleColor="#8d949d", gridColor="#161d25", domainColor="#2a3440",
                            tickColor="#2a3440", labelFont="Geist Mono", titleFont="Geist", labelFontSize=10.5,
                            titleFontSize=11.5, titleFontWeight=400, titlePadding=10, labelPadding=6)
            .configure_legend(labelColor=TINTA2, orient="top", labelFont="Geist", title=None, symbolStrokeWidth=3,
                              labelFontSize=11.5, symbolSize=120, symbolOpacity=1)
            .configure_title(font="Instrument Serif", fontSize=21, color=TINTA, anchor="start", fontWeight=400, offset=14)
            .configure_text(font="Geist Mono"))


def grafica(chart: alt.Chart) -> None:
    emitir_grafica(chart)


def curvas_epoca(curvas: list[dict], clave_fit: str, clave_sel: str, titulo: str, eje: str,
                 dominio=None, marca_x: int | None = None) -> alt.Chart:
    filas = []
    for c in curvas:
        for clave, serie in ((clave_fit, "Entrenamiento"), (clave_sel, "Validación")):
            filas.append({"época": c["epoca"], "serie": serie, "valor": c[clave],
                          "p25": c[clave + "_p25"], "p75": c[clave + "_p75"], "redes": c["n"]})
    df = pd.DataFrame(filas)
    color = alt.Color("serie:N", scale=alt.Scale(domain=["Entrenamiento", "Validación"], range=[ACERO, COBRE]))
    x = alt.X("época:Q", axis=alt.Axis(tickMinStep=1, values=list(range(1, 13)), grid=False), scale=alt.Scale(domain=[1, 12]))
    y = alt.Y("valor:Q", title=eje, scale=alt.Scale(domain=dominio, zero=False) if dominio else alt.Scale(zero=False))
    banda = alt.Chart(df).mark_area(opacity=0.16).encode(x=x, y="p25:Q", y2="p75:Q", color=color)
    linea = alt.Chart(df).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    sel = alt.selection_point(fields=["época"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    puntos = alt.Chart(df).mark_circle(size=80, stroke=PANEL, strokeWidth=2).encode(
        x=x, y=y, color=color, opacity=alt.condition(sel, alt.value(1), alt.value(0.0)),
        tooltip=[alt.Tooltip("época:Q"), alt.Tooltip("serie:N"), alt.Tooltip("valor:Q", format=".4f"),
                 alt.Tooltip("p25:Q", title="percentil 25", format=".4f"),
                 alt.Tooltip("p75:Q", title="percentil 75", format=".4f"), alt.Tooltip("redes:Q")]
    ).add_params(sel)
    regla = alt.Chart(df).mark_rule(color=TENUE, strokeDash=[2, 3]).encode(x=x).transform_filter(sel)
    capas = [banda, linea, regla, puntos]
    if marca_x:
        capas.append(alt.Chart(pd.DataFrame({"x": [marca_x]})).mark_rule(color=TINTA2, strokeWidth=1, strokeDash=[1, 3])
                     .encode(x="x:Q"))
    return _base(alt.layer(*capas).properties(title=titulo))


def curvas_comparadas(control: list[dict], actual: list[dict]) -> alt.Chart:
    filas = []
    for nombre, curvas in (("Receta antigua (control)", control), ("imagen_v2", actual)):
        for c in curvas:
            if c["epoca"] > 13:
                continue
            filas.append({"época": c["epoca"], "receta": nombre, "brecha": c["fit_auc"] - c["sel_auc"],
                          "AUC entrenamiento": c["fit_auc"], "AUC validación": c["sel_auc"]})
    df = pd.DataFrame(filas)
    sel = alt.selection_point(fields=["época"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    color = alt.Color("receta:N", scale=alt.Scale(domain=["Receta antigua (control)", "imagen_v2"],
                                                  range=[GRIS, COBRE]))
    base = alt.Chart(df).encode(x=alt.X("época:Q", axis=alt.Axis(tickMinStep=1, grid=False)), color=color)
    linea = base.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36, filled=True)).encode(
        y=alt.Y("brecha:Q", title="AUC entrenamiento − AUC validación"))
    puntos = base.mark_circle(size=100, stroke=PANEL, strokeWidth=2).encode(
        y="brecha:Q", opacity=alt.condition(sel, alt.value(1), alt.value(0)),
        tooltip=["receta:N", "época:Q", alt.Tooltip("AUC entrenamiento:Q", format=".3f"),
                 alt.Tooltip("AUC validación:Q", format=".3f"), alt.Tooltip("brecha:Q", format=".3f")]
    ).add_params(sel)
    cero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=TENUE, strokeDash=[2, 3]).encode(y="y:Q")
    return _base(alt.layer(cero, linea, puntos).properties(title="Brecha de memorización por época"), 280)


def roc(curvas: dict[str, list], colores: dict[str, str], titulo: str = "Curva ROC") -> alt.Chart:
    filas = [{"modelo": n, "1 − especificidad": x, "sensibilidad": y} for n, pts in curvas.items() for x, y in pts]
    df = pd.DataFrame(filas)
    color = alt.Color("modelo:N", scale=alt.Scale(domain=list(colores), range=list(colores.values())))
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=TENUE, strokeDash=[2, 3]).encode(x="x:Q", y="y:Q")
    linea = alt.Chart(df).mark_line(strokeWidth=2, interpolate="step-after").encode(
        x=alt.X("1 − especificidad:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("sensibilidad:Q", scale=alt.Scale(domain=[0, 1])), color=color,
        tooltip=["modelo:N", alt.Tooltip("1 − especificidad:Q", format=".2f"), alt.Tooltip("sensibilidad:Q", format=".2f")])
    return _base(alt.layer(diag, linea).properties(title=titulo), 360)


def pr(puntos: list, prevalencia: float) -> alt.Chart:
    df = pd.DataFrame(puntos, columns=["sensibilidad", "precisión"])
    area = alt.Chart(df).mark_area(color=COBRE, opacity=0.08).encode(x="sensibilidad:Q", y="precisión:Q")
    linea = alt.Chart(df).mark_line(strokeWidth=2, color=COBRE).encode(
        x=alt.X("sensibilidad:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("precisión:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("sensibilidad:Q", format=".2f"), alt.Tooltip("precisión:Q", format=".2f")])
    azar = alt.Chart(pd.DataFrame({"y": [prevalencia], "t": [f"azar = prevalencia ({prevalencia:.2f})"]})).mark_rule(
        color=TENUE, strokeDash=[2, 3]).encode(y="y:Q", tooltip=["t:N"])
    return _base(alt.layer(area, azar, linea).properties(title="Precisión–sensibilidad"), 360)


def matriz(r: dict) -> alt.Chart:
    filas = []
    for real, n_real in (("no pCR", r["TN"] + r["FP"]), ("pCR", r["TP"] + r["FN"])):
        for pred in ("no pCR", "pCR"):
            k = {("no pCR", "no pCR"): "TN", ("no pCR", "pCR"): "FP", ("pCR", "no pCR"): "FN", ("pCR", "pCR"): "TP"}[(real, pred)]
            nombre = {"TN": "Verdaderos negativos", "FP": "Falsos positivos", "FN": "Falsos negativos", "TP": "Verdaderos positivos"}[k]
            filas.append({"real": real, "predicción": pred, "n": r[k], "fila": r[k] / n_real, "tipo": nombre,
                          "texto": f"{r[k]}  ·  {round(100 * r[k] / n_real)} %"})
    df = pd.DataFrame(filas)
    base = alt.Chart(df).encode(x=alt.X("predicción:N", sort=["no pCR", "pCR"], title="Predicción", axis=alt.Axis(labelAngle=0)),
                                y=alt.Y("real:N", sort=["no pCR", "pCR"], title="Realidad"))
    celdas = base.mark_rect(stroke=PANEL, strokeWidth=4).encode(
        color=alt.Color("fila:Q", scale=alt.Scale(domain=[0, 1], range=["#141b23", COBRE]), legend=None),
        tooltip=["tipo:N", "n:Q", alt.Tooltip("fila:Q", title="% de la fila", format=".1%")])
    texto = base.mark_text(fontSize=15, fontWeight=500).encode(text="texto:N", color=alt.value(TINTA))
    return _base(alt.layer(celdas, texto), 260)


def calibracion(puntos: list[dict], intervalos_fijos: bool = False) -> alt.Chart:
    df = pd.DataFrame(puntos)
    grupo = "intervalo" if intervalos_fijos else "decil"
    lim = max(0.6, df[["predicha", "observada"]].max().max() + 0.05)
    diag = alt.Chart(pd.DataFrame({"x": [0, lim], "y": [0, lim]})).mark_line(color=TENUE, strokeDash=[2, 3]).encode(x="x:Q", y="y:Q")
    linea = alt.Chart(df).mark_line(color=COBRE, strokeWidth=2, point=alt.OverlayMarkDef(size=80, filled=True, stroke=PANEL, strokeWidth=2)).encode(
        x=alt.X("predicha:Q", title=f"Probabilidad predicha (media del {grupo})", scale=alt.Scale(domain=[0, lim])),
        y=alt.Y("observada:Q", title=f"pCR observada en el {grupo}", scale=alt.Scale(domain=[0, lim])),
        tooltip=[alt.Tooltip("predicha:Q", format=".3f"), alt.Tooltip("observada:Q", format=".3f"), "n:Q"])
    titulo = "Calibración · intervalos fijos de 0,1" if intervalos_fijos else "Calibración por deciles"
    return _base(alt.layer(diag, linea).properties(title=titulo), 320)


def bandas_grafica(info: dict, titulo: str) -> alt.Chart:
    df = pd.DataFrame(info["bandas"])
    df["nombre"] = df["banda"].map(NOMBRE_BANDA)
    df["lo"], df["hi"] = df["ic95"].str[0], df["ic95"].str[1]
    df["rango"] = df.apply(lambda f: f"{f.p_min:.2f}–{f.p_max:.2f}", axis=1)
    df["texto"] = df["tasa_pcr"].map(lambda v: f"{100 * v:.0f} %")
    orden = [NOMBRE_BANDA[b] for b in ("baja", "intermedia", "alta")]
    x = alt.X("nombre:N", sort=orden, title=None, axis=alt.Axis(labelAngle=0, labelFont="Geist", labelFontSize=11.5))
    barras = alt.Chart(df).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2, size=64).encode(
        x=x, y=alt.Y("tasa_pcr:Q", title="pCR observada", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.8])),
        color=alt.Color("banda:N", scale=alt.Scale(domain=list(BANDAS), range=list(BANDAS.values())), legend=None),
        tooltip=[alt.Tooltip("nombre:N", title="banda"), alt.Tooltip("rango:N", title="probabilidad predicha"),
                 alt.Tooltip("n:Q", title="pacientes"), alt.Tooltip("pcr:Q", title="con pCR"),
                 alt.Tooltip("tasa_pcr:Q", title="pCR observada", format=".1%"),
                 alt.Tooltip("lo:Q", title="IC95 % inferior", format=".1%"),
                 alt.Tooltip("hi:Q", title="IC95 % superior", format=".1%")])
    error = alt.Chart(df).mark_rule(color=TINTA2, strokeWidth=1.2).encode(x=x, y="lo:Q", y2="hi:Q")
    texto = alt.Chart(df).mark_text(dy=-10, dx=22, fontSize=11, color=TINTA2, align="left").encode(x=x, y="hi:Q", text="texto:N")
    prev = alt.Chart(pd.DataFrame({"y": [info["prevalencia"]]})).mark_rule(color=TENUE, strokeDash=[3, 3]).encode(y="y:Q")
    return _base(alt.layer(barras, error, texto, prev).properties(title=titulo), 300)


def mancuernas(df: pd.DataFrame) -> alt.Chart:
    """df: fila (texto), sin (AUC sin imagen), con (AUC con imagen). Una línea por combinación."""
    largo = df.melt(id_vars="fila", value_vars=["sin", "con"], var_name="imagen", value_name="AUC")
    largo["imagen"] = largo["imagen"].map({"sin": "Sin imagen", "con": "Con imagen"})
    y = alt.Y("fila:N", title=None, sort=df["fila"].tolist(), axis=alt.Axis(labelFont="Geist", labelFontSize=11.5, labelLimit=260))
    regla = alt.Chart(df).mark_rule(color="#3a4552", strokeWidth=2).encode(y=y, x=alt.X("sin:Q", title="AUC media de los 5 folds",
                                                                                        scale=alt.Scale(zero=False)), x2="con:Q")
    puntos = alt.Chart(largo).mark_circle(size=130, opacity=1, stroke=PANEL, strokeWidth=2).encode(
        y=y, x="AUC:Q",
        color=alt.Color("imagen:N", scale=alt.Scale(domain=["Sin imagen", "Con imagen"], range=[GRIS, COBRE])),
        tooltip=["fila:N", "imagen:N", alt.Tooltip("AUC:Q", format=".3f")])
    return _base(alt.layer(regla, puntos), 34 * len(df) + 40)


def pie() -> None:
    emitir_html('<div class="pie"><span>RED DCE · BREASTDCEDL · CNN 2D PROPIA ENTRENADA DESDE CERO</span>'
                '<span>USO EDUCATIVO · NO ES UN DISPOSITIVO MÉDICO NI SUSTITUYE EL JUICIO CLÍNICO</span></div>',
                )


def escapar(texto: str) -> str:
    return html.escape(str(texto))


def estructura_fusion(params: dict) -> None:
    """Recorrido interactivo de la fusión con los parámetros reales del paquete."""
    plantilla = (RECURSOS / "estructura_fusion.html").read_text(encoding="utf-8")
    codigo = plantilla.replace("__FUENTES__", FUENTES).replace(
        "__PARAMETROS__", json.dumps(params, ensure_ascii=True).replace("<", "\\u003c"))
    emitir_marco(codigo, height=1280)
