
from .. import estado
import numpy as np

from ..componentes import (aviso, cinetica_grafica, destinos, emitir_html, grafica, mapa_realce, num, pasos, pct, pie,
                           portada, recurso, seccion, subtipos_grafica, tarjetas)


def mostrar(pagina):
    img = recurso("modelo_imagen")
    fus = recurso("modelo_fusion")
    m = img["metricas"]
    evaluacion_test = recurso("evaluacion_test")
    datos = recurso("datos")

    portada(estado.portada(), "Predicción de la respuesta patológica completa en <em>cáncer de mama</em>",
            "Modelo que estima, a partir de la resonancia magnética con contraste (DCE-MRI) previa a la quimioterapia "
            "neoadyuvante, la probabilidad de <b>respuesta patológica completa (pCR)</b>: ausencia de tumor invasivo "
            "residual en la pieza quirúrgica. La red neuronal convolucional se ha diseñado y entrenado desde cero y "
            "puede combinarse con cuatro variables clínicas.",
            [("prediccion", "Realizar una predicción", True), ("red_imagen", "Arquitectura del modelo", False)])

    aviso("<b>Uso exclusivamente educativo.</b> Una probabilidad no es una certeza: una estimación alta no garantiza "
          "pCR ni permite evitar la cirugía, y una baja no la descarta.", "atencion")

    seccion("1", "¿Qué es la pCR?",
            "La <b>respuesta patológica completa (pCR)</b> es la variable que predice todo el proyecto. Es un resultado "
            "que solo se conoce después de operar; el modelo intenta anticiparlo meses antes, con la resonancia previa "
            "al tratamiento.")
    pasos([
        ("Resonancia DCE previa", "PRE, EARLY y LATE antes de empezar. Es la única imagen que recibe el modelo"),
        ("Quimioterapia neoadyuvante", "Varios meses de tratamiento antes de operar, para reducir el tumor"),
        ("Cirugía", "Se extirpa el lecho tumoral y se analizan los ganglios axilares"),
        ("Anatomía patológica", "El patólogo busca tumor invasivo residual en la pieza: ahí se decide pCR = 1 o 0"),
    ])
    c1, c2 = pagina.columnas([1, 1.15], gap="large")
    with c1:
        clin = datos["clinica"] if datos else None
        miles = lambda n: f"{n:,}".replace(",", ".")
        cifra = lambda n: f'<em>{miles(n)} de {miles(clin["n"])}</em>' if clin else ""
        emitir_html(
            '<div class="desenlaces">'
            f'<div class="desenlace" style="--c:var(--realce)"><div class="t"><span>pCR = 1 · respuesta completa</span>'
            f'{cifra(clin["n_pcr"]) if clin else ""}</div>'
            "<b>No queda tumor invasivo</b><span>Ni en la mama ni en los ganglios analizados (definición habitual "
            "ypT0/is ypN0; puede persistir componente in situ). Se asocia a mejor supervivencia libre de enfermedad, "
            "sobre todo en triple negativo y HER2+.</span></div>"
            f'<div class="desenlace" style="--c:#6b7480"><div class="t"><span>pCR = 0 · enfermedad residual</span>'
            f'{cifra(clin["n"] - clin["n_pcr"]) if clin else ""}</div>'
            "<b>Queda tumor tras el tratamiento</b><span>La quimioterapia redujo o no el tumor, pero quedan células "
            "invasivas en la pieza quirúrgica. Es el caso más frecuente en las pacientes de desarrollo.</span></div>"
            "</div>")
    with c2:
        if datos:
            grafica(subtipos_grafica(datos["clinica"]["por_subtipo"], datos["clinica"]["prevalencia"]))
            pagina.texto_auxiliar(f"{datos['clinica']['n']:,} pacientes de desarrollo.".replace(",", ".") + " La pCR depende mucho del subtipo: "
                                  "por eso HR y HER2 son las variables clínicas que más aportan al modelo combinado.")
    prevalencia = img["n_pcr"] / img["n_pacientes"]
    aviso("<b>Qué calcula el modelo.</b> Una <b>probabilidad de pCR</b> entre 0 y 1 para cada paciente, no un diagnóstico. "
          f"Como solo el {pct(prevalencia)} alcanza pCR, decir siempre «no pCR» acertaría el {pct(1 - prevalencia)} sin "
          "aprender nada: por eso se mide con "
          "la AUC y la probabilidad se lee por <b>perfiles</b>, comparándola con lo que les pasó a pacientes parecidas.", "clave")

    # Escala de las barras: AUC entre el azar (0,5) y la perfección (1).
    escala = lambda auc: (auc - 0.5) / 0.5
    seccion("2", "Resultados principales")
    ambito = "Validación cruzada"
    if evaluacion_test:
        ambito = pagina.segmentos("Resultados principales", ["Test final", "Validación cruzada"],
                                      default="Test final", key="resumen_evaluacion") or "Test final"
    if ambito == "Test final":
        t = evaluacion_test
        combinado, cnn = t["modelos"]["fusion"], t["modelos"]["imagen"]
        tarjetas([
            {"t": "Imagen + clínica · test", "v": num(combinado["auc"]), "acento": True,
             "d": f"AUC · IC95 % {num(combinado['auc_ic95'][0])}–{num(combinado['auc_ic95'][1])}"},
            {"t": "CNN de imagen · test", "v": num(cnn["auc"]),
             "d": f"AUC · IC95 % {num(cnn['auc_ic95'][0])}–{num(cnn['auc_ic95'][1])}"},
            {"t": "Pacientes de test", "v": str(t["n_pacientes"]),
             "d": f"{t['n_pcr']} con pCR · {pct(t['prevalencia'], 1)}"},
            {"t": "Evaluación completada", "v": "Youden", "d": "Umbrales fijados antes del test · " + "/".join(reversed(t["fecha"].split("-")))},
        ])
        pagina.texto_auxiliar("El detalle de errores, curvas y resultados por cohorte está en Imagen + clínica → Test final. "
                   "Los intervalos de test son IC95 %, no desviaciones entre folds.")
    else:
        if fus:
            r = fus["resumen"]["brazos"][fus["resumen"]["alias"]["completa+imagen"]]
            fusion = {"t": "Imagen + clínica", "v": num(r["auc_media_folds"]), "acento": True,
                      "d": f"AUC media ± {num(r['auc_de_folds'])} · CV anidada", "barra": escala(r["auc_media_folds"]),
                      "barra_txt": "0,5 → 1", "ayuda": "Regresión logística sobre la puntuación de la CNN, edad, volumen "
                      "tumoral, HR y HER2. Ningún coeficiente vio el fold que evalúa."}
        else:
            fusion = {"t": "Imagen + clínica", "v": "…", "d": "Evaluación anidada en marcha",
                      "ayuda": "Se está midiendo sin fugas cuánto aporta la imagen sobre los datos clínicos."}
        tarjetas([
            {"t": "CNN de imagen", "v": num(m["auc_media_folds"]), "d": f"AUC media ± {num(m['auc_de_folds'])} en 5 folds",
             "barra": escala(m["auc_media_folds"]), "barra_txt": "0,5 → 1",
             "ayuda": "Media de la AUC por paciente de los 5 folds exteriores. Juntando las 1.097 pacientes: "
                      f"{num(m['auc_agrupada'])} (IC95 % {num(m['auc_ic95'][0])}–{num(m['auc_ic95'][1])})."},
            fusion,
            {"t": "Pacientes de desarrollo", "v": "1.097", "d": f"{img['n_pcr']} con pCR ({round(100 * m['prevalencia'])} %)",
             "barra": m["prevalencia"], "barra_txt": "con pCR",
             "ayuda": "Train: tres cohortes (I-SPY2, Duke, I-SPY1), ≈10 cortes por paciente. La separación es siempre por paciente."},
            {"t": "Test reservado", "v": "176", "d": "Evaluado · vista Test final" if evaluacion_test else "Reservado; aún no evaluado",
             "ayuda": "Test del paquete congelado. Sus métricas se muestran separadas de las de desarrollo."},
        ])

    if datos:
        im = datos["imagenes"]
        seccion("3", "Las imágenes: qué separa a pCR de no pCR",
                f"Media de los recortes de las {im['n_pcr'] + im['n_no_pcr']:,} pacientes de desarrollo".replace(",", ".")
                + ", agrupadas según su respuesta. Cada celda resume 8 × 8 píxeles; el tumor queda hacia el centro "
                "porque los recortes se centran en él. Pasa el ratón por el mapa para ver el valor de cada zona.")
        modo = pagina.segmentos("Mapa de las imágenes", ["Realce temprano · EARLY − PRE", "Fase tardía · LATE − EARLY"],
                                default="Realce temprano · EARLY − PRE", key="inicio_mapa") or "Realce temprano · EARLY − PRE"
        clave = "temprano" if modo.startswith("Realce") else "lavado"
        a, b = ("EARLY", "PRE") if clave == "temprano" else ("LATE", "EARLY")
        tope = max(float(np.max(np.array(im["medias"][g][a]) - np.array(im["medias"][g][b]))) for g in ("pcr", "no_pcr"))
        c1, c2, c3 = pagina.columnas(3, gap="medium")
        with c1:
            grafica(mapa_realce(im["medias"]["no_pcr"], im["lado"], clave, f"Sin pCR · {im['n_no_pcr']}", tope))
        with c2:
            grafica(mapa_realce(im["medias"]["pcr"], im["lado"], clave, f"Con pCR · {im['n_pcr']}", tope))
        with c3:
            grafica(mapa_realce(im["medias"], im["lado"], clave, "Diferencia · pCR − sin pCR"))
        c1, c2 = pagina.columnas([1.2, 1], gap="large")
        with c1:
            grafica(cinetica_grafica(im["cinetica"]))
        with c2:
            aviso("<b>Cómo se lee.</b> El realce temprano mide cuánto capta el tejido al llegar el contraste; la fase tardía, "
                  "si sigue captando (positivo) o lo pierde (lavado, negativo). En el mapa de diferencia, el naranja marca "
                  "zonas donde las pacientes con pCR captan más y el azul, donde captan menos. Las diferencias medias son "
                  "de unas pocas centésimas: a "
                  "simple vista, los dos grupos son casi iguales.<br><br>La curva resume la región central "
                  f"({im['region_central'][0]}–{im['region_central'][1]} px): intensidad antes del contraste, en la "
                  "fase temprana y en la tardía. La sombra es el rango intercuartílico entre pacientes y se solapa "
                  "casi por completo. Por eso la CNN tiene que buscar patrones finos de textura y cinética, y su señal "
                  "es débil.", "info")

    seccion("4", "Flujo de predicción")
    pasos([
        ("Tres fases", "PRE, EARLY y LATE del mismo corte, antes del tratamiento"),
        ("CNN propia", f"{img['parametros']:,} parámetros desde cero; 5 redes en conjunto".replace(",", ".")),
        ("Probabilidad", "Media de los cortes de la paciente, calibrada con la validación cruzada"),
        ("Clínica (opcional)", "Edad, volumen tumoral, HR y HER2 en una regresión logística"),
        ("Perfil", "Uno de cinco perfiles, de muy baja a muy alta, con la pCR observada en cada uno"),
    ])

    seccion("5", "Interpretación de los resultados")
    c1, c2 = pagina.columnas(2)
    with c1:
        aviso('<div class="error"><div class="mini"><i></i><i class="on"></i><i></i><i></i></div><div><b>Falso positivo.</b> '
              "El modelo estima pCR alta, pero queda tumor residual. Si se usase para relajar el tratamiento, se "
              "infratrataría a la paciente: por eso nunca debe cambiar la cirugía prevista.</div></div>", "clave")
    with c2:
        aviso('<div class="error"><div class="mini"><i></i><i></i><i class="on"></i><i></i></div><div><b>Falso negativo.</b> '
              "Estima pCR baja, pero la paciente sí responde por completo. Podría llevar a intensificar sin necesidad "
              "o a una preocupación injustificada.</div></div>", "info")
    rango = ""
    if fus:
        extremos = fus["perfiles"]["subtipo+edad+volumen+imagen"]["perfiles"]
        rango = (f": con imagen y clínica, del {pct(extremos[0]['tasa_pcr'])} en el perfil muy bajo al "
                 f"{pct(extremos[-1]['tasa_pcr'])} en el muy alto")
    aviso("Los <b>perfiles</b> reparten a las pacientes de la validación cruzada en cinco grupos según su probabilidad: "
          "el 10 % más bajo, el 20 % siguiente, el 40 % central, el 20 % siguiente y el 10 % más alto. En cada perfil se "
          f"cuenta cuántas tuvieron pCR de verdad{rango}. Así la predicción se lee con datos observados y no como un sí "
          "o un no.", "clave")

    seccion("6", "Contenido")
    destinos([
        ("prediccion", "Predicción individual", "Carga de las tres fases o paciente de ejemplo, visor de cortes y perfil de probabilidad."),
        ("red_imagen", "CNN de imagen", "Validación cruzada en 5 folds, curvas de pérdida, arquitectura y discriminación."),
        ("combinado", "Imagen + clínica", "Fusión tardía: test final y validación cruzada en vistas separadas."),
        ("metodologia", "Datos y metodología", "Cohortes, separación por paciente, reglas de evaluación y limitaciones."),
    ])
    pie()
