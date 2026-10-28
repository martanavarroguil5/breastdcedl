"""Vista de la evaluación cerrada: resultados agregados y política fija."""
import json

import pandas as pd
import streamlit as st

from componentes import (ACERO, COBRE, COHORTES, aviso, cabecera, calibracion, grafica,
                         matriz, num, pct, pr, roc, seccion, tabla, tarjetas)


def mostrar_test(informe: dict, modelo: str = "fusion") -> None:
    m = informe["modelos"][modelo]
    c = m["clasificacion"]
    fecha = "/".join(reversed(informe["fecha"].split("-")))
    cabecera("Evaluación final · test público", f"{m['nombre']}: <em>resultado en test</em>",
             "Predicciones por paciente con los pesos, la calibración y los umbrales fijados antes de evaluar. "
             "No se han ajustado parámetros con estos resultados.",
             meta=[("estado", "evaluado"), ("fecha", fecha), ("pacientes", str(informe['n_pacientes'])),
                   ("con pCR", str(informe['n_pcr'])), ("política", "Youden congelado")])
    tarjetas([
        {"t": "AUC · test", "v": num(m["auc"]), "acento": True,
         "d": f"IC95 % {num(m['auc_ic95'][0])}–{num(m['auc_ic95'][1])}",
         "ayuda": "Intervalo por bootstrap estratificado por clase, 5.000 remuestreos. No es desviación entre folds."},
        {"t": "Sensibilidad", "v": pct(c["sensibilidad"], 1),
         "d": f"{c['TP']} de {c['TP'] + c['FN']} respuestas completas detectadas"},
        {"t": "Especificidad", "v": pct(c["especificidad"], 1),
         "d": f"{c['TN']} de {c['TN'] + c['FP']} casos sin pCR identificados"},
        {"t": "Precisión de pCR", "v": pct(c["precision_pcr"], 1),
         "d": f"{c['TP']} respuestas reales entre {c['TP'] + c['FP']} predicciones positivas"},
    ])
    seccion("1", "Errores con la política congelada",
            "Positivo significa respuesta patológica completa (pCR). El umbral se eligió en desarrollo y se conserva aquí.")
    izquierda, derecha = st.columns(2, gap="large")
    with izquierda:
        grafica(matriz(c))
    with derecha:
        tarjetas([
            {"t": "Falsos positivos", "v": str(c["FP"]), "d": "Se predijo pCR, pero no la hubo", "acento": True},
            {"t": "Falsos negativos", "v": str(c["FN"]), "d": "Hubo pCR, pero no se detectó"},
        ])
        tabla(pd.DataFrame([
            {"Métrica": "Exactitud", "Test": pct(c["exactitud"], 1)},
            {"Métrica": "F1 de pCR", "Test": num(c["f1_pcr"])},
            {"Métrica": "Macro-F1", "Test": num(c["macro_f1"])},
        ]))
    if modelo == "fusion":
        with st.expander("Datos disponibles y umbrales aplicados"):
            tabla(pd.DataFrame([{"Submodelo": f["submodelo"], "Pacientes": f["n"], "Umbral guardado": num(f["umbral"], 6)}
                                for f in informe["umbrales_aplicados"]]))
            st.caption("Se utiliza el patrón de variables disponibles, sin imputación y sin buscar un umbral en test.")
    seccion("2", "Discriminación y probabilidades")
    izquierda, derecha = st.columns(2, gap="large")
    with izquierda:
        grafica(roc({v["nombre"]: v["roc"] for v in informe["modelos"].values()},
                    {"Imagen + clínica": COBRE, "CNN de imagen": ACERO}, "ROC · mismas pacientes de test"))
    with derecha:
        grafica(pr(m["pr"], informe["prevalencia"]))
    tarjetas([
        {"t": "Precisión media · AP", "v": num(m["ap"]), "d": f"Referencia de prevalencia: {num(informe['prevalencia'])}"},
        {"t": "Brier", "v": num(m["brier"]), "d": "Error de las probabilidades; menor es mejor"},
        {"t": "Pérdida · BCE", "v": num(m["bce"]), "d": "Calculada en test, sin recalibración"},
    ])
    with st.expander("Calibración observada en test"):
        grafica(calibracion(m["calibracion"], intervalos_fijos=True))
        st.caption("Intervalos fijos de probabilidad de anchura 0,1; se omiten los vacíos. El tamaño de cada grupo "
                   "aparece al pasar el cursor. Es un diagnóstico descriptivo; no se ha reajustado la calibración.")
    seccion("3", "Comparación en el mismo test")
    tabla(pd.DataFrame([
        {"Modelo": v["nombre"], "AUC": num(v["auc"]), "IC95 % AUC": f"{num(v['auc_ic95'][0])}–{num(v['auc_ic95'][1])}",
         "AP": num(v["ap"]), "Brier": num(v["brier"]),
         "FP": v["clasificacion"]["FP"], "FN": v["clasificacion"]["FN"]}
        for v in informe["modelos"].values()
    ]))
    d = informe["diferencia_auc"]
    st.caption(f"Combinado − CNN: ΔAUC {d['delta']:+.3f}; IC95 % [{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}]. "
               "Bootstrap pareado sobre las mismas pacientes. Los errores usan la política propia guardada de cada modelo.")
    aviso("Esta comparación mide el combinado frente a la CNN sola. No demuestra que la imagen mejore a la clínica "
          "sola: ese contraste no forma parte de esta evaluación de test.", "info")
    seccion("4", "Resultados por cohorte", "Subgrupos pequeños: las diferencias son descriptivas y no se usan para cambiar el modelo.")
    tabla(pd.DataFrame([
        {"Cohorte": COHORTES.get(f["cohorte"], f["cohorte"]), "Pacientes": f["n"], "pCR": f["pcr"],
         "AUC combinado": num(f["auc_fusion"]) if f["auc_fusion"] is not None else "—",
         "AUC CNN": num(f["auc_imagen"]) if f["auc_imagen"] is not None else "—",
         "FP combinado": f["clasificacion_fusion"]["FP"], "FN combinado": f["clasificacion_fusion"]["FN"]}
        for f in informe["cohortes"]
    ]))
    seccion("5", "Alcance del cierre")
    aviso(informe["limite"] + " La validación privada sigue pendiente. Los pesos, coeficientes, calibración y "
          "umbrales se mantienen como estaban antes de abrir este test.", "atencion")
    st.caption("La validación cruzada se conserva en la otra vista. Su AUC media entre folds y su DE no son "
               "la misma estimación que la AUC de este test y su IC95 %.")
    st.download_button("Descargar informe de test · JSON", json.dumps(informe, ensure_ascii=False, indent=2),
                       file_name="evaluacion_test.json", mime="application/json", key=f"descarga_test_{modelo}")
