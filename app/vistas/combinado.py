import altair as alt
import pandas as pd

from ..componentes import (ACERO, COBRE, GRIS, TINTA2, _base, aviso, cabecera, calibracion, composicion_grafica,
                         distribucion_grafica, grafica, mapa_clinico, matriz, num, pct, perfiles_grafica, pie, pr, recurso,
                         roc, seccion, tabla, tarjetas, estructura_fusion)
from ..vista_test import mostrar_test


def mostrar(pagina):
    evaluacion_test = recurso("evaluacion_test")
    if evaluacion_test:
        vista = pagina.segmentos("Resultados del combinado", ["Test final", "Validación cruzada"],
                                     default="Test final", key="fusion_evaluacion") or "Test final"
        if vista == "Test final":
            mostrar_test(pagina, evaluacion_test)
            pie()
            pagina.terminar()

    fus = recurso("modelo_fusion")
    if not fus:
        pagina.informacion("Faltan los recursos de evaluación del modelo combinado.")
        pagina.terminar()
    r = fus["resumen"]
    m = r["brazos"][r["alias"]["completa+imagen"]]
    config = fus.get("configuracion", {})
    sub = config.get("submodelo")
    perfiles = fus["perfiles"]["subtipo+edad+volumen+imagen"]
    prevalencia = perfiles["prevalencia"]
    datos = recurso("datos")

    cabecera("Modelo 2 · imagen + clínica", "Evaluación del <em>modelo combinado</em>",
             "Regresión logística regularizada sobre la puntuación de imagen, edad, volumen, HR y HER2. "
             "Resultados de desarrollo por paciente en validación cruzada anidada.",
             meta=[("variables", "5"), ("evaluación", "5 folds anidados"),
                   ("regularización", "L2 · C = 1"), ("test", "evaluado · vista separada" if evaluacion_test else "pendiente")])
    tarjetas([
        {"t": "Discriminación · AUC", "v": num(m["auc_media_folds"]), "d": f"± {num(m['auc_de_folds'])} entre folds",
         "acento": True, "ayuda": "AUC media en validación anidada. 0,5 = azar; 1 = separación perfecta."},
        {"t": "Precisión media · AP", "v": num(m["ap"]), "d": f"Referencia sin señal: {num(prevalencia)}"},
        {"t": "Error probabilístico · Brier", "v": num(m["brier"]), "d": "Menor es mejor; 0 es perfecto"},
        {"t": "Pérdida · BCE", "v": num(m["bce"]), "d": "Fuera de fold; menor es mejor"},
    ])


    if sub:
        estructura_fusion(sub)

    seccion("1", "Discriminación y estabilidad", "Cada paciente se evalúa fuera del ajuste de su fold.")
    c1, c2 = pagina.columnas(2, gap="large")
    with c1:
        grafica(roc({"Combinado": fus["roc"]["completa+imagen"]}, {"Combinado": COBRE}, "ROC del combinado · OOF anidada"))
    with c2:
        df = pd.DataFrame({"Fold": [str(i + 1) for i in range(len(m["auc_por_fold"]))], "AUC": m["auc_por_fold"]})
        barras = alt.Chart(df).mark_bar(color=COBRE, size=32).encode(
            x=alt.X("Fold:O", title="Fold exterior"), y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=["Fold:O", alt.Tooltip("AUC:Q", format=".3f")])
        etiquetas = barras.mark_text(dy=-9, color=TINTA2).encode(text=alt.Text("AUC:Q", format=".3f"))
        azar = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color=GRIS, strokeDash=[3, 3]).encode(y="y:Q")
        grafica(_base(alt.layer(barras, etiquetas, azar).properties(title="AUC en cada fold"), 360))
    if fus.get("pr", {}).get("completa+imagen"):
        grafica(pr(fus["pr"]["completa+imagen"], prevalencia))
    else:
        pagina.texto_auxiliar("Curva precisión–sensibilidad pendiente de exportar desde las predicciones OOF originales; AP guardada arriba.")
    pagina.texto_auxiliar("Discriminación moderada. La DE describe variación entre folds; no es un intervalo de confianza ni descarta sobreajuste.")
    with pagina.desplegable("Rendimiento por cohorte y subtipo", icon=":material/groups:"):
        filas = [{"Grupo": "Cohorte " + k, "AUC del combinado": num(v)} for k, v in m["auc_dentro_cohorte"].items()]
        filas += [{"Grupo": "Subtipo " + k, "AUC del combinado": num(v["completa+imagen"])} for k, v in r["auc_por_subtipo"].items()]
        tabla(pd.DataFrame(filas))

    seccion("2", "Qué aporta cada dato",
            "Importancia medida sin fugas: la misma validación anidada repetida quitando una variable cada vez. "
            "Cuanto más cae la AUC al quitarla, más aporta esa variable además de las demás.")
    brazos = r["brazos"]
    completa = brazos[r["alias"]["completa+imagen"]]["auc_media_folds"]
    variables = {"Subtipo (HR y HER2)": "subtipo", "Volumen tumoral": "volumen", "Edad": "edad", "Imagen (CNN)": "imagen"}
    todas = ["subtipo", "edad", "volumen", "imagen"]
    filas_imp = []
    for nombre, v in variables.items():
        sin = "+".join(x for x in todas if x != v) + "|normal"
        filas_imp.append({"Variable": nombre, "Caída de AUC al quitarla": completa - brazos[sin]["auc_media_folds"],
                          "AUC sin ella": brazos[sin]["auc_media_folds"], "AUC sola": brazos[f"{v}|normal"]["auc_media_folds"]})
    imp = pd.DataFrame(filas_imp)
    imp["texto"] = imp.apply(lambda f: f"−{f['Caída de AUC al quitarla']:.3f} · sola {f['AUC sola']:.2f}".replace(".", ","), axis=1)
    c1, c2 = pagina.columnas([1, 1], gap="large")
    with c1:
        y_imp = alt.Y("Variable:N", sort=list(variables), title=None, axis=alt.Axis(labelFont="Geist", labelFontSize=11.5))
        barras_imp = alt.Chart(imp).mark_bar(size=26, cornerRadiusTopRight=2, cornerRadiusBottomRight=2).encode(
            y=y_imp, x=alt.X("Caída de AUC al quitarla:Q", title="Caída de la AUC al quitar la variable",
                             scale=alt.Scale(domain=[0, 0.22])),
            color=alt.condition(alt.datum.Variable == "Imagen (CNN)", alt.value(COBRE), alt.value(ACERO)),
            tooltip=["Variable:N", alt.Tooltip("Caída de AUC al quitarla:Q", format=".4f"),
                     alt.Tooltip("AUC sin ella:Q", format=".3f"), alt.Tooltip("AUC sola:Q", format=".3f")])
        etiquetas_imp = barras_imp.mark_text(align="left", dx=6, fontSize=11, color=TINTA2).encode(text="texto:N")
        grafica(_base(alt.layer(barras_imp, etiquetas_imp).properties(title="Importancia de cada variable"), 250))
        pagina.texto_auxiliar(f"AUC del modelo completo: {num(completa)}. «Sola» es la AUC de una logística con esa única "
                              "variable. El subtipo domina; imagen, volumen y edad añaden poco cada uno, pero juntos suman.")
    with c2:
        if datos:
            clin = datos["clinica"]
            cruce = pagina.segmentos("Cruce clínico", ["Subtipo × volumen", "Subtipo × edad"], default="Subtipo × volumen",
                                     key="fusion_cruce") or "Subtipo × volumen"
            if cruce == "Subtipo × volumen":
                grafica(mapa_clinico(clin["subtipo_volumen"], "volumen_grupo", ["Pequeño", "Mediano", "Grande"],
                                     "pCR observada · subtipo y volumen", "Volumen tumoral (tercil)"))
                pagina.texto_auxiliar("Terciles de volumen en train: pequeño < " + num(clin["cortes_volumen"][0], 1) +
                                      " cm³ ≤ mediano < " + num(clin["cortes_volumen"][1], 1) + " cm³ ≤ grande.")
            else:
                grafica(mapa_clinico(clin["subtipo_edad"], "edad_grupo", ["< 40", "40–49", "50–59", "≥ 60"],
                                     "pCR observada · subtipo y edad", "Edad (años)"))
                pagina.texto_auxiliar("Grupos de edad al diagnóstico. Cada celda indica la pCR observada y el número de pacientes.")
    aviso("<b>Lectura.</b> Dentro de cada subtipo, los tumores pequeños y las pacientes jóvenes alcanzan pCR algo más a "
          "menudo. La imagen aporta algo distinto de estas variables: por eso la fusión mejora a la clínica sola, aunque "
          "la mejora sea pequeña.", "info")

    seccion("3", "Errores de clasificación", "Selecciona el umbral: cambia el equilibrio entre falsos positivos y falsos negativos.")
    umbrales = {k: v for k, v in fus.get("umbrales", {}).items() if k.endswith("+imagen")}
    if umbrales:
        patron = pagina.selector("Datos disponibles del modelo combinado", list(umbrales),
                             index=list(umbrales).index("subtipo+edad+volumen+imagen"))
        regla = pagina.segmentos("Umbral de decisión", ["Youden", "Máximo macro-F1"], default="Youden")
        met = umbrales[patron]["macro_f1" if regla == "Máximo macro-F1" else "youden"]
        c1, c2 = pagina.columnas(2, gap="large")
        with c1:
            grafica(matriz(met))
            pagina.texto_auxiliar(f"Umbral {num(met['umbral'])} · {met['TP'] + met['TN'] + met['FP'] + met['FN']} pacientes.")
        with c2:
            tabla(pd.DataFrame([{"Métrica": nombre, "Valor": pct(met[clave], 1)} for clave, nombre in
                 (("sensibilidad", "Sensibilidad"), ("especificidad", "Especificidad"),
                  ("precision_pcr", "Valor predictivo positivo"), ("exactitud", "Exactitud"),
                  ("f1_pcr", "F1 pCR"), ("macro_f1", "Macro-F1"))]))
        pagina.texto_auxiliar("Umbrales elegidos con estas mismas OOF: las métricas de clasificación pueden ser optimistas. "
                   + ("La vista Test final aplica únicamente el Youden congelado." if evaluacion_test else "Test pendiente."))

    seccion("4", "Perfiles de pCR",
            "En lugar de un sí o un no, cada paciente se sitúa en uno de cinco perfiles según su probabilidad. Los cortes "
            "son percentiles de las predicciones fuera de fold: el 10 % más bajo, el 20 % siguiente, el 40 % central, el "
            "20 % siguiente y el 10 % más alto.")
    extremos = perfiles["perfiles"]
    tarjetas([
        {"t": "Perfil muy alto", "v": pct(extremos[-1]["tasa_pcr"]), "acento": True,
         "d": f"pCR observada · {num(extremos[-1]['razon'], 1)} × la media · {extremos[-1]['n']} pacientes"},
        {"t": "Perfil muy bajo", "v": pct(extremos[0]["tasa_pcr"]),
         "d": f"pCR observada · {num(extremos[0]['razon'], 1)} × la media · {extremos[0]['n']} pacientes"},
        {"t": "Separación", "v": f"{extremos[-1]['tasa_pcr'] / extremos[0]['tasa_pcr']:.1f} ×".replace(".", ","),
         "d": "pCR del perfil muy alto frente al muy bajo"},
        {"t": "Media de desarrollo", "v": pct(prevalencia), "d": f"{perfiles['n']:,} pacientes, validación anidada".replace(",", ".")},
    ])
    c1, c2 = pagina.columnas([1, 1], gap="large")
    with c1:
        grafica(perfiles_grafica(perfiles, "pCR observada en cada perfil"))
    with c2:
        grafica(composicion_grafica(perfiles, "Subtipos dentro de cada perfil"))
    grafica(distribucion_grafica(perfiles, "Distribución de las probabilidades del combinado"))
    pagina.texto_auxiliar("Barras apiladas: pacientes con y sin pCR en cada tramo de probabilidad, sobre las zonas de los cinco "
                          "perfiles. Ningún tramo es puro: incluso en el perfil muy alto, " +
                          pct(1 - extremos[-1]["tasa_pcr"]) + " no alcanzó pCR. Los perfiles reparten la incertidumbre; "
                          "no la eliminan. Cortes fijados solo con train, sin mirar el test.")

    seccion("5", "Calidad de las probabilidades")
    c1, c2 = pagina.columnas(2, gap="large")
    with c1:
        if fus.get("calibracion", {}).get("completa+imagen"):
            grafica(calibracion(fus["calibracion"]["completa+imagen"]))
    with c2:
        ref_brier = prevalencia * (1 - prevalencia)
        df = pd.DataFrame([{"Predicción": "Prevalencia constante", "Brier": ref_brier},
                           {"Predicción": "Combinado OOF", "Brier": m["brier"]}])
        barras = alt.Chart(df).mark_bar(color=COBRE, size=40).encode(
            x=alt.X("Predicción:N", axis=alt.Axis(labelAngle=0), title=None),
            y=alt.Y("Brier:Q", title="Brier · menor es mejor", scale=alt.Scale(domain=[0, 0.25])),
            tooltip=["Predicción:N", alt.Tooltip("Brier:Q", format=".3f")])
        grafica(_base(alt.layer(barras, barras.mark_text(dy=-9, color=TINTA2).encode(text=alt.Text("Brier:Q", format=".3f")))
                     .properties(title="Error de las probabilidades"), 320))
    pagina.texto_auxiliar("La prevalencia constante es una referencia descriptiva calculada en las OOF. "
               "El Brier combina calibración y discriminación; por sí solo no demuestra buena calibración.")
    if not fus.get("calibracion", {}).get("completa+imagen"):
        pagina.texto_auxiliar("Calibración por deciles pendiente: faltan las probabilidades individuales OOF en esta copia.")

    seccion("6", "Complejidad y parámetros de la fusión",
            "Cinco coeficientes y un intercepto en el submodelo completo; la CNN se mantiene congelada.")
    if sub:
        tarjetas([
            {"t": "Parámetros logísticos", "v": str(len(sub["coeficientes"]) + 1), "d": "5 coeficientes + intercepto"},
            {"t": "Pacientes del ajuste final", "v": str(sub["n_ajuste"]), "d": f"{sub['pcr_ajuste']} con pCR"},
            {"t": "Penalización", "v": "L2", "d": "C = 1 · sin pesos de clase"},
            {"t": "Interacciones", "v": "0", "d": "Variables estandarizadas"},
        ])
    coef = pd.DataFrame(fus["coeficientes_por_fold"]).melt(id_vars="fold", var_name="variable", value_name="coeficiente")
    nombres = {"HR": "HR positivo", "HER2": "HER2 positivo", "edad": "Edad", "log_volumen": "Volumen (log)", "imagen": "Puntuación de imagen"}
    coef["variable"] = coef["variable"].map(nombres)
    base = alt.Chart(coef).encode(y=alt.Y("variable:N", title=None, sort=list(nombres.values())))
    rango = base.mark_rule(color=COBRE, strokeWidth=5, opacity=0.3).encode(x="min(coeficiente):Q", x2="max(coeficiente):Q")
    puntos = base.mark_point(filled=True, color=COBRE, size=85).encode(
        x=alt.X("coeficiente:Q", title="Coeficiente estandarizado · cada punto es un fold"),
        tooltip=["variable:N", "fold:Q", alt.Tooltip("coeficiente:Q", format="+.3f")])
    cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=GRIS, strokeDash=[3, 3]).encode(x="x:Q")
    grafica(_base(alt.layer(cero, rango, puntos), 280))
    pagina.texto_auxiliar("L2 reduce la magnitud de los coeficientes; no garantiza ausencia de sobreajuste. "
               "La estabilidad entre folds es un diagnóstico complementario. Los 6 parámetros corresponden a la etapa logística, no a la CNN.")
    if sub:
        with pagina.desplegable("Parámetros del paquete final", icon=":material/table_rows:"):
            tabla(pd.DataFrame([{"Variable": nombres.get(k, k), "Coeficiente": num(v),
                                 "Media": num(sub["media"][i]), "Escala": num(sub["escala"][i])}
                                for i, (k, v) in enumerate(sub["coeficientes"].items())]))
            pagina.texto_auxiliar(f"Intercepto {num(sub['intercepto'])}. Volumen transformado como log(1 + volumen). "
                       "Cada patrón de datos ausentes tiene su propia logística.")

    seccion("7", "¿Hay sobreajuste de la fusión?",
            "Diagnóstico propio: rendimiento en el ajuste logístico frente al fold exterior reservado.")
    diag = fus.get("diagnostico_generalizacion", r.get("diagnostico_generalizacion", []))
    if diag:
        filas = [{"Fold": str(d["fold"] + 1), "Conjunto": titulo,
                  "AUC": d[clave]["auc"], "BCE": d[clave]["bce"]}
                 for d in diag for clave, titulo in (("ajuste", "Ajuste"), ("validacion", "Validación exterior"))]
        for metrica in ("AUC", "BCE"):
            grafica(_base(alt.Chart(pd.DataFrame(filas)).mark_line(point=True).encode(
                x=alt.X("Fold:O"), y=alt.Y(f"{metrica}:Q", scale=alt.Scale(zero=False)),
                color="Conjunto:N", tooltip=["Fold:O", "Conjunto:N", alt.Tooltip(f"{metrica}:Q", format=".3f")])
                .properties(title=f"{metrica} · ajuste y validación de la fusión"), 260))
        tabla(pd.DataFrame([{"Fold": d["fold"] + 1, "Brecha AUC (ajuste − validación)": num(d["brecha_auc"]),
                             "Brecha BCE (validación − ajuste)": num(d["brecha_bce"])} for d in diag]))
        pagina.texto_auxiliar("Brechas positivas indican optimismo del ajuste. Las puntuaciones de imagen del ajuste son cross-fitted. "
                   "Este diagnóstico mide la etapa logística, no la memorización de la CNN; no hay un umbral universal que descarte sobreajuste.")
    else:
        aviso("<b>No determinado.</b> La copia local no contiene las predicciones internas necesarias para medir "
              "la brecha de ajuste–validación. La AUC anidada mide generalización, pero no cuantifica esa brecha.", "atencion")
        with pagina.desplegable("Cómo completar el diagnóstico", icon=":material/terminal:"):
            pagina.codigo("python scripts/exportar_diagnostico_fusion.py", language="bash")
            pagina.texto_auxiliar("Ejecutar desde la raíz del proyecto en el equipo que conserva runs/cv_interna_fusion y "
                       "runs/cv_presupuesto. Reajusta solo las logísticas de los folds con el mismo C = 1; "
                       "no entrena CNN, no cambia el paquete final y no lee test.")

    seccion("8", "Conclusión")
    aviso("<b>Discriminación moderada en validación anidada.</b> El combinado obtiene AUC 0,714 y AP 0,505; "
          "sus probabilidades mejoran el Brier de la referencia constante. Esto no basta para afirmar "
          "buena calibración ni ausencia de sobreajuste. "
          + ("<b>El test realizado se muestra en su vista independiente.</b>" if evaluacion_test
             else "<b>Evaluación final en test pendiente.</b>"), "clave")
    pie()
