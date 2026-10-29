import altair as alt
import pandas as pd

from .. import estado
from ..componentes import (ACERO, COBRE, COHORTES, GRIS, OCRE, PANEL, TINTA2, _base, aviso, bloques_3d, cabecera,
                         calibracion, composicion_grafica, curvas_comparadas, curvas_epoca, distribucion_grafica, grafica,
                         matriz, num, pct, perfiles_grafica, pie, pr, recurso, red_neuronal, roc, seccion, tabla, tarjetas)
from ..vista_test import mostrar_test


def mostrar(pagina):
    evaluacion_test = recurso("evaluacion_test")
    if evaluacion_test:
        vista = pagina.segmentos("Resultados de la CNN", ["Desarrollo y arquitectura", "Test final"],
                                     default="Desarrollo y arquitectura", key="cnn_evaluacion")
        if vista == "Test final":
            mostrar_test(pagina, evaluacion_test, "imagen")
            pie()
            pagina.terminar()

    img = recurso("modelo_imagen")
    m = img["metricas"]
    curvas = img["curvas"]
    fin = curvas[-1]
    sel_min = min(curvas, key=lambda c: c["sel_bce_obj"])
    # Meseta: primera época a menos de 0,001 de la mejor AUC media de validación.
    auc_pico = next(c for c in curvas if c["sel_auc"] >= max(x["sel_auc"] for x in curvas) - 0.001)

    cabecera("Modelo 1 · solo imagen", "CNN de imagen: <em>RedDCE</em>",
             "Una CNN 2D propia, sin pesos preentrenados ni datos clínicos de entrada. Su capacidad de discriminación "
             "es modesta y se evalúa por paciente; las curvas permiten examinar la brecha entre entrenamiento y validación.",
             meta=[("parámetros", f"{img['parametros']:,}".replace(",", ".")), ("redes", "5 × semilla"),
                   ("épocas", "12"), ("entrada", "3 fases · 256²"), ("pacientes", "1.097")])

    tarjetas([
        {"t": "AUC media (5 folds)", "v": num(m["auc_media_folds"]), "d": f"± {num(m['auc_de_folds'])} entre folds", "acento": True,
         "ayuda": "Cada paciente se predice con redes que no la vieron. Por fold: "
                  + ", ".join(num(a) for a in m["auc_por_fold"]) + "."},
        {"t": "IC95 % de la AUC", "v": f"{num(m['auc_ic95'][0], 2)}–{num(m['auc_ic95'][1], 2)}", "d": "No incluye el 0,5 del azar",
         "ayuda": "Bootstrap estratificado por clase sobre las 1.097 predicciones fuera de fold."},
        {"t": "Precisión media (AP)", "v": num(m["ap"]), "d": f"Azar = {num(m['prevalencia'])}",
         "ayuda": "Área bajo la curva precisión–sensibilidad. Solo con la cohorte: " + num(m["ap_cohorte"]) + "."},
        {"t": "Brecha de memorización", "v": num(fin["fit_auc"] - fin["sel_auc"], 2), "d": "AUC entrenamiento − validación, época 12",
         "ayuda": "Con la receta antigua llegaba a 0,28 y la validación caía al azar."},
    ])

    # ---------------------------------------------------------------- validación cruzada
    datos = recurso("datos")
    seccion("1", "Validación cruzada en 5 folds",
            "Las 1.097 pacientes se reparten en 5 folds estratificados por pCR. En cada ronda se entrenan redes con "
            "cuatro folds y se predice el quinto, que nunca vieron. Así, cada paciente recibe una predicción "
            "<b>fuera de fold</b> y la AUC no premia la memorización.")
    if datos and datos.get("particiones"):
        nombres_rol = {"fit": "Entrenamiento (fit)", "seleccion": "Selección · 15 %", "exterior": "Evaluación · fold exterior"}
        filas_rol = pd.DataFrame([
            {"Ronda": f"Ronda {p['fold'] + 1}", "Rol": nombres_rol[rol], "orden": i, "Pacientes": p[rol]["n"],
             "Con pCR": p[rol]["pcr"], "pCR": p[rol]["pcr"] / p[rol]["n"]}
            for p in datos["particiones"] for i, rol in enumerate(("fit", "seleccion", "exterior"))])
        barras_rol = alt.Chart(filas_rol).mark_bar(size=26, stroke=PANEL, strokeWidth=2).encode(
            y=alt.Y("Ronda:N", title=None, axis=alt.Axis(labelFont="Geist", labelFontSize=11.5)),
            x=alt.X("Pacientes:Q", stack="zero", title="Pacientes de desarrollo"),
            color=alt.Color("Rol:N", scale=alt.Scale(domain=list(nombres_rol.values()), range=["#3a4552", ACERO, COBRE]),
                            legend=alt.Legend(title=None)),
            order=alt.Order("orden:Q"),
            tooltip=["Ronda:N", "Rol:N", "Pacientes:Q", "Con pCR:Q", alt.Tooltip("pCR:Q", title="proporción pCR", format=".1%")])
        grafica(_base(barras_rol.properties(title="Qué hace cada paciente en cada ronda"), 250))
        pagina.texto_auxiliar("Entrenamiento ajusta los pesos; selección (15 % del resto) solo vigila las curvas de pérdida; "
                              "evaluación es el fold exterior, que da la AUC. Todos los cortes de una paciente van juntos y la "
                              "proporción de pCR es casi idéntica en todos los folds.")

    fuentes_folds = {k: v for k, v in img.get("curvas_por_fold", {}).items() if v}
    if fuentes_folds:
        metricas_fold = {"Pérdida de validación": ("sel_bce_obj", "Pérdida BCE ponderada"),
                         "Pérdida de entrenamiento": ("fit_bce_obj", "Pérdida BCE ponderada"),
                         "AUC de validación": ("sel_auc", "AUC")}
        metrica_fold = pagina.segmentos("Curva de los 5 folds", list(metricas_fold), default="Pérdida de validación",
                                        key="cnn_folds_metrica") or "Pérdida de validación"
        clave_fold, eje_fold = metricas_fold[metrica_fold]
        filas_curva = pd.DataFrame([
            {"Fold": f"Fold {int(k) + 1}", "época": c["epoca"], "valor": c[clave_fold],
             "AUC fold exterior": m["auc_por_fold"][int(k)]}
            for k, historia in sorted(fuentes_folds.items(), key=lambda par: int(par[0])) for c in historia])
        resalte = alt.selection_point(fields=["Fold"], bind="legend")
        lineas = alt.Chart(filas_curva).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=28, filled=True)).encode(
            x=alt.X("época:Q", axis=alt.Axis(tickMinStep=1, values=list(range(1, 13)), grid=False), scale=alt.Scale(domain=[1, 12])),
            y=alt.Y("valor:Q", title=eje_fold, scale=alt.Scale(zero=False)),
            color=alt.Color("Fold:N", scale=alt.Scale(range=[COBRE, ACERO, OCRE, GRIS, "#7f9c6b"]),
                            legend=alt.Legend(title=None)),
            opacity=alt.condition(resalte, alt.value(1), alt.value(0.12)),
            tooltip=["Fold:N", "época:Q", alt.Tooltip("valor:Q", title=metrica_fold.lower(), format=".4f"),
                     alt.Tooltip("AUC fold exterior:Q", format=".3f")]).add_params(resalte)
        grafica(_base(lineas.properties(title=f"{metrica_fold} · los 5 folds a la vez"), 300))
        pagina.texto_auxiliar("Cada línea es la media de las redes de un fold (una por semilla). Haz clic en la leyenda para "
                              "aislar un fold. Que las cinco curvas sigan la misma forma indica que la receta de "
                              "entrenamiento se comporta igual con particiones distintas de pacientes.")

    pagina.texto("### AUC de cada fold exterior")
    pagina.texto_auxiliar("En cada fold se evalúa un grupo distinto de pacientes, excluido del entrenamiento de sus redes. "
               "Los folds son particiones independientes, no épocas consecutivas.")
    valores_fold = m["auc_por_fold"]
    opciones_fold = ["Todos"] + [f"Fold {i + 1}" for i in range(len(valores_fold))]
    fold_elegido = pagina.segmentos("Resaltar un fold de la CNN", opciones_fold, default="Todos", key="cnn_fold") or "Todos"
    filas_fold = pd.DataFrame([
        {"Fold": f"Fold {i + 1}", "AUC": valor, "Azar": 0.5,
         "Diferencia con la media": valor - m["auc_media_folds"],
         "Resaltado": fold_elegido == "Todos" or fold_elegido == f"Fold {i + 1}"}
        for i, valor in enumerate(valores_fold)
    ])
    base_fold = alt.Chart(filas_fold).encode(
        x=alt.X("Fold:O", sort=opciones_fold[1:], title="Fold exterior · paciente fuera del entrenamiento"))
    trazos_fold = base_fold.mark_rule(strokeWidth=3).encode(
        y=alt.Y("Azar:Q", title="AUC · eje ampliado (0,45–0,65)", scale=alt.Scale(domain=[0.45, 0.65], zero=False)),
        y2="AUC:Q", color=alt.condition(alt.datum.Resaltado, alt.value(COBRE), alt.value("#3c4652")))
    puntos_fold = base_fold.mark_circle(size=180, stroke="#0d1116", strokeWidth=2).encode(
        y="AUC:Q", color=alt.condition(alt.datum.Resaltado, alt.value(COBRE), alt.value(GRIS)),
        tooltip=["Fold:O", alt.Tooltip("AUC:Q", format=".4f"),
                 alt.Tooltip("Diferencia con la media:Q", format="+.4f")])
    texto_fold = base_fold.mark_text(dy=-18, fontSize=14, color=TINTA2).encode(
        y="AUC:Q", text=alt.Text("AUC:Q", format=".3f"))
    referencias_fold = pd.DataFrame([
        {"AUC": m["auc_media_folds"], "Referencia": "Media de los 5 folds"},
        {"AUC": 0.5, "Referencia": "Azar"},
    ])
    lineas_fold = alt.Chart(referencias_fold).mark_rule(strokeDash=[5, 4], strokeWidth=1).encode(
        y="AUC:Q", color=alt.Color("Referencia:N", scale=alt.Scale(
            domain=["Media de los 5 folds", "Azar"], range=[COBRE, GRIS]), legend=alt.Legend(title=None)),
        tooltip=["Referencia:N", alt.Tooltip("AUC:Q", format=".3f")])
    grafica(_base(alt.layer(trazos_fold, lineas_fold, puntos_fold, texto_fold).properties(
        title="Rendimiento de la CNN en cada partición de validación"), 320))
    if fold_elegido != "Todos":
        valor_fold = valores_fold[opciones_fold.index(fold_elegido) - 1]
        tarjetas([
            {"t": fold_elegido + " · AUC", "v": num(valor_fold), "d": "Predicciones fuera de fold", "acento": True},
            {"t": "Diferencia con la media", "v": f"{valor_fold - m['auc_media_folds']:+.3f}".replace(".", ","),
             "d": "Media de referencia: " + num(m["auc_media_folds"])},
            {"t": "Diferencia con el azar", "v": f"{valor_fold - 0.5:+.3f}".replace(".", ","), "d": "Azar = 0,500"},
        ])
    else:
        tarjetas([
            {"t": "Media de los folds", "v": num(m["auc_media_folds"]), "d": "5 particiones de pacientes", "acento": True},
            {"t": "Variación entre folds", "v": num(m["auc_de_folds"]), "d": "Desviación estándar; no es un IC"},
            {"t": "Rango observado", "v": num(min(valores_fold)) + "–" + num(max(valores_fold)),
             "d": "Separación máximo − mínimo: " + num(max(valores_fold) - min(valores_fold))},
        ])
    pagina.texto_auxiliar("Los cinco valores son superiores a 0,5, con una señal débil y variación pequeña entre particiones. "
               "Sin intervalos por fold, eso no demuestra significación en cada uno ni ausencia de sobreajuste.")


    # ---------------------------------------------------------------- entrenamiento
    pagina.texto('<div id="perdida-cnn"></div>', unsafe_allow_html=True)
    seccion("2", "Pérdida de la CNN · época a época",
            "Entrenamiento y validación en la misma gráfica. Una pérdida menor indica menos error; "
            "si entrenamiento baja y validación sube, aumenta la brecha de generalización.")
    fuentes_curvas = {"Media de los 5 folds": curvas}
    for fold_id, historia in sorted(img.get("curvas_por_fold", {}).items(), key=lambda par: int(par[0])):
        if historia:
            fuentes_curvas[f"Fold {int(fold_id) + 1}"] = historia
    if len(fuentes_curvas) > 1:
        ambito_curvas = pagina.segmentos("Curvas de entrenamiento de la CNN", list(fuentes_curvas),
                                            default="Media de los 5 folds", key="cnn_curvas_ambito") or "Media de los 5 folds"
    else:
        ambito_curvas = "Media de los 5 folds"
    curvas_vista = fuentes_curvas[ambito_curvas]
    epocas_vista = [c["epoca"] for c in curvas_vista]
    epoca_elegida = pagina.rango("Época de entrenamiento", options=epocas_vista, value=epocas_vista[-1],
                                    key=f"cnn_epoca_{ambito_curvas}")
    fila_epoca = next(c for c in curvas_vista if c["epoca"] == epoca_elegida)
    minimo_vista = min(curvas_vista, key=lambda c: c["sel_bce_obj"])
    final_vista = curvas_vista[-1]
    grafica(curvas_epoca(curvas_vista, "fit_bce_obj", "sel_bce_obj",
                         f"Pérdida BCE ponderada · {ambito_curvas.lower()}", "Pérdida BCE ponderada",
                         marca_x=epoca_elegida))
    tarjetas([
        {"t": f"Época {epoca_elegida} · entrenamiento", "v": num(fila_epoca["fit_bce_obj"], 4),
         "d": "BCE ponderada, evaluada sin aumentos"},
        {"t": f"Época {epoca_elegida} · validación", "v": num(fila_epoca["sel_bce_obj"], 4),
         "d": "Pacientes de selección no usadas para entrenar", "acento": True},
        {"t": "Brecha de pérdida", "v": f"{fila_epoca['sel_bce_obj'] - fila_epoca['fit_bce_obj']:+.4f}".replace(".", ","),
         "d": "Validación − entrenamiento en esta época"},
    ])
    pagina.texto_auxiliar(f"Cada punto resume {fila_epoca['n']} redes en la época seleccionada. La sombra es el rango "
               "intercuartílico entre redes, no un intervalo de confianza. «Validación» es el conjunto interno de "
               "selección (15 %); no es el fold exterior utilizado para calcular la AUC OOF.")
    pagina.texto_auxiliar(f"Mínimo de pérdida de validación: época {minimo_vista['epoca']} ({num(minimo_vista['sel_bce_obj'], 4)}). "
               f"Última época: {final_vista['epoca']} ({num(final_vista['sel_bce_obj'], 4)}). "
               "Una subida tras el mínimo es una señal de sobreajuste; se interpreta junto a la magnitud de la brecha y la AUC.")
    if len(fuentes_curvas) == 1:
        pagina.texto_auxiliar("Curva media de 5 folds × 3 semillas. El desglose de pérdida por fold no está disponible en esta versión.")
    with pagina.desplegable("AUC de entrenamiento y validación por época", icon=":material/show_chart:"):
        grafica(curvas_epoca(curvas_vista, "fit_auc", "sel_auc", f"AUC por época · {ambito_curvas.lower()}", "AUC",
                             marca_x=epoca_elegida))
    with pagina.desplegable("Valores de pérdida en cada época", icon=":material/table_rows:"):
        tabla(pd.DataFrame([{"Época": c["epoca"], "Entrenamiento": num(c["fit_bce_obj"], 4),
                             "Validación": num(c["sel_bce_obj"], 4),
                             "Brecha": num(c["sel_bce_obj"] - c["fit_bce_obj"], 4), "Redes": c["n"]}
                            for c in curvas_vista]))

    with pagina.desplegable("¿Cómo era antes? Comparación con la receta antigua", icon=":material/compare_arrows:"):
        grafica(curvas_comparadas(img["curvas_control"], curvas))
        pagina.texto_auxiliar("Con todos los cortes en cada época y una tasa de aprendizaje más alta, la red memorizaba desde la "
                   "época 2–3: la brecha crecía hasta 0,28. La receta actual usa 3 cortes al azar por paciente y época "
                   "y un presupuesto fijo de 12 épocas.")

    # ---------------------------------------------------------------- arquitectura
    seccion("3", "Recorrido de un corte por la red",
            "Un corte real se propaga por la red de izquierda a derecha. Cada miniatura es el mapa de activación de un "
            "filtro; las líneas representan los pesos efectivos entre los filtros con mayor contribución a la salida. "
            "El selector de vista muestra el mismo filtro sobre el recorte de 256, 128 o 64 px.")
    ej = estado.ejemplos()
    c1, c2 = pagina.columnas([3, 2])
    pid = c1.selector("Paciente de ejemplo (train)", ej.index, index=3, format_func=lambda p: (
        f"{p} · {COHORTES.get(ej.loc[p, 'cohorte'], ej.loc[p, 'cohorte'])} · {'pCR' if ej.loc[p, 'pCR'] else 'no pCR'}"))
    pixeles, nombres = estado.cortes_ejemplo(pid)
    k = c2.rango("Corte", options=list(range(len(nombres))), value=len(nombres) // 2,
                         format_func=lambda i: nombres[i]) if len(nombres) > 1 else 0
    red_neuronal(estado.recorrido(pixeles[k]), alto=620)

    seccion("4", "Arquitectura en bloques",
            "La misma red como volúmenes: la altura de cada bloque es proporcional a la resolución espacial y su "
            "profundidad, al número de canales. Dentro de cada bloque se muestran mapas reales del corte seleccionado "
            "y los conos indican el campo receptivo de la convolución entre etapas.")
    bloques_3d(estado.bloques(pixeles[k]))
    with pagina.desplegable("Tabla de capas", icon=":material/table_rows:"):
        tabla(pd.DataFrame([{"Etapa": c["nombre"], "Salida": c["forma"], "Qué hace": c["detalle"],
                             "Parámetros": c["params"] or "—"} for c in estado.capas_red()]))
        pagina.texto_auxiliar("Las tres vistas pasan por el mismo tronco convolucional (pesos compartidos). La cabeza auxiliar del "
                   "código no se usa en este modelo (λ = 0). Entrenamiento: AdamW, LR lineal hasta 0 en la época 12, EMA "
                   "de pesos, BCE ponderada y aumentos geométricos idénticos en las tres fases. En la vista de arriba, el "
                   "peso efectivo entre filtros es la suma de |pesos| de las dos convoluciones de cada bloque, escalada por "
                   "su BatchNorm y por la activación del filtro de origen.")

    # ---------------------------------------------------------------- discriminación
    seccion("5", "Capacidad de discriminación",
            "Predicciones fuera de fold de las 1.097 pacientes de desarrollo. Estas gráficas no incluyen test.")
    c1, c2 = pagina.columnas(2, gap="large")
    with c1:
        grafica(roc({"CNN de imagen": img["roc"], "Solo cohorte": img["roc_cohorte"]},
                    {"CNN de imagen": COBRE, "Solo cohorte": GRIS}))
    with c2:
        grafica(pr(img["pr"], m["prevalencia"]))

    grupos = {**{f"Cohorte {COHORTES.get(c, c)}": a for c, a in m["auc_por_cohorte"].items()},
              **{f"Subtipo {s}": a for s, a in m["auc_por_subtipo"].items()}}
    with pagina.desplegable("AUC por cohorte y por subtipo", icon=":material/groups:"):
        tabla(pd.DataFrame([{"Grupo": g, "AUC": num(a)} for g, a in grupos.items()]))
        pagina.texto_auxiliar("La imagen discrimina algo mejor en HER2+ y casi nada en HR+/HER2−.")

    # ---------------------------------------------------------------- umbral y perfiles
    perfiles = img["perfiles"]
    seccion("6", "Umbral y perfiles de pCR",
            "Un sí/no depende de un umbral y oculta la incertidumbre. Los perfiles ofrecen una lectura más honesta: qué "
            "proporción de pacientes con una probabilidad parecida alcanzó pCR.")
    c1, c2 = pagina.columnas([1, 1.25], gap="large")
    with c1:
        regla = pagina.segmentos("Umbral", ["Youden", "Máximo macro-F1"], default="Youden")
        r = img["umbrales"]["youden" if regla != "Máximo macro-F1" else "macro_f1"]
        grafica(matriz(r))
        pagina.texto_auxiliar(f"Umbral {num(r['umbral'])} sobre la probabilidad calibrada · sensibilidad {pct(r['sensibilidad'], 1)} · "
                   f"especificidad {pct(r['especificidad'], 1)} · VPP {pct(r['precision_pcr'], 1)}.")
    with c2:
        grafica(perfiles_grafica(perfiles, "pCR observada en cada perfil · solo imagen"))
        pagina.texto_auxiliar("Perfiles por percentiles de las OOF: 10 % · 20 % · 40 % · 20 % · 10 % de las pacientes. La línea "
                              "discontinua es la media; el texto indica cuántas veces la supera cada perfil.")
    grafica(distribucion_grafica(perfiles, "Distribución de las probabilidades de la CNN"))
    pagina.texto_auxiliar("Todas las pacientes caen entre ≈0,13 y ≈0,46 y las dos clases se solapan casi por completo: con "
                          "solo imagen, el perfil muy alto llega al " + pct(perfiles["perfiles"][-1]["tasa_pcr"]) +
                          " de pCR frente al " + pct(perfiles["perfiles"][0]["tasa_pcr"]) + " del muy bajo. La señal existe, "
                          "pero es débil; la clínica la amplía (ver Imagen + clínica).")
    with pagina.desplegable("Subtipos dentro de cada perfil", icon=":material/groups:"):
        grafica(composicion_grafica(perfiles, "Composición por subtipo de cada perfil · solo imagen"))
        pagina.texto_auxiliar("La CNN no recibe el subtipo. Que la mezcla de subtipos apenas cambie entre perfiles muestra "
                              "que la imagen sola no reconoce bien el subtipo, que es el factor clínico más fuerte.")

    with pagina.desplegable("Calibración", icon=":material/tune:"):
        grafica(calibracion(img["calibracion"]))
        pagina.texto_auxiliar("Cada punto agrupa un 10 % de pacientes. Cerca de la diagonal = probabilidades honestas. Que todos "
                   "queden entre ≈0,2 y ≈0,35 muestra que el modelo nunca está muy seguro.")

    seccion("7", "Conclusión")
    aviso("Con imagen sola y una CNN desde cero se observa una <b>señal pequeña en desarrollo</b>: el intervalo "
          "reportado de la AUC excluye el azar y la variación entre folds es baja. Las curvas permiten examinar el "
          "sobreajuste, pero no descartarlo. "
          + ("La evaluación realizada está disponible en la vista Test final." if evaluacion_test
             else "La confirmación final en test sigue pendiente."), "info")
    pie()
