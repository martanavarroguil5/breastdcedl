import numpy as np
import pandas as pd

from .. import estado
from .. import motor
from ..componentes import (COHORTES, aviso, cabecera, distribucion_grafica, escapar, grafica, lectura, lectura_vacia, num,
                         pct, pie, recurso, red_neuronal, seccion, visor)


def mostrar(pagina):
    cabecera("Predicción", "Predicción individual",
             "Carga las tres fases de uno o varios cortes de la misma paciente o selecciona un caso de ejemplo. Los datos "
             "clínicos son opcionales; si se introducen, se muestra también la estimación combinada.")

    img_info = recurso("modelo_imagen")
    fus_info = recurso("modelo_fusion")
    paquete = estado.paquete_imagen()
    fusion = estado.paquete_fusion()

    # ---------------------------------------------------------------- 1. imágenes
    seccion("1", "Imágenes", "PNG en gris de 256 × 256 por fase. El nombre debe terminar en <code>_PRE</code>, "
            "<code>_EARLY</code> o <code>_LATE</code>; los cortes se agrupan por el resto del nombre.")
    origen = pagina.segmentos("Origen", ["Paciente de ejemplo", "Subir PNG"], default="Paciente de ejemplo",
                                  label_visibility="collapsed")
    pixeles, nombres, clinica_ejemplo, etiqueta_real, titulo_visor = None, [], {}, None, "subida"
    if origen == "Subir PNG":
        archivos = pagina.archivos("Fases PRE, EARLY y LATE", type="png", accept_multiple_files=True)
        if archivos:
            grupos, errores = motor.agrupar_subidas(archivos)
            for e in errores:
                pagina.advertencia(e)
            if grupos:
                try:
                    pixeles, nombres = motor.leer_cortes(grupos), list(grupos)
                except (ValueError, OSError) as exc:
                    pagina.error(f"Formato no válido: {exc}")
    else:
        ej = estado.ejemplos()
        pid = pagina.selector("Paciente de ejemplo (train)", ej.index, format_func=lambda p: (
            f"{p} · {COHORTES.get(ej.loc[p, 'cohorte'], ej.loc[p, 'cohorte'])} · {ej.loc[p, 'cortes']} cortes"))
        pixeles, nombres = estado.cortes_ejemplo(pid)
        f = ej.loc[pid]
        clinica_ejemplo = {"edad": f["edad"], "volumen": float(np.expm1(f["log_volumen"])), "HR": f["HR"], "HER2": f["HER2"]}
        etiqueta_real = int(f["pCR"])
        titulo_visor = f"{COHORTES.get(f['cohorte'], f['cohorte'])} · {escapar(pid)}"
        aviso("Estas pacientes son de <b>entrenamiento</b>: el modelo final las vio, así que su predicción es optimista "
              "y no sirve para evaluar. Solo ilustran el funcionamiento.", "atencion")

    pred = None
    if pixeles is not None:
        pred = estado.prediccion(pixeles, tuple(nombres))
        p_cortes = tuple(motor.calibrar(paquete, p) for p in pred.por_corte)
        visor(estado.visor(pixeles, tuple(nombres), p_cortes), titulo_visor, alto=500,
              inicial=int(np.argsort(p_cortes)[len(p_cortes) // 2]))
        pagina.texto_auxiliar(f"{len(nombres)} corte(s) válidos. Las ventanas discontinuas son las tres vistas centrales que recibe "
                   "la red (256, 128 y 64 px). La probabilidad de cada corte ya está calibrada.")

    # ---------------------------------------------------------------- 2. clínica
    seccion("2", "Datos clínicos <em>(opcional)</em>",
            "Deja en «Desconocido» lo que no tengas: el modelo usa el submodelo entrenado con exactamente los datos "
            "disponibles, sin inventar los que faltan.")


    def _opcion(valor):
        return "Desconocido" if valor is None or pd.isna(valor) else ("Positivo" if valor == 1 else "Negativo")


    with pagina.grupo(border=True):
        c1, c2, c3, c4 = pagina.columnas([1, 1, 1.45, 1.45], gap="medium")
        edad_conocida = c1.interruptor("Edad conocida", value=bool(clinica_ejemplo) and not pd.isna(clinica_ejemplo.get("edad")))
        edad = c1.numero("Edad (años)", 18.0, 100.0, float(clinica_ejemplo.get("edad") or 50.0), 1.0,
                               disabled=not edad_conocida)
        vol_conocido = c2.interruptor("Volumen conocido", value=bool(clinica_ejemplo) and not pd.isna(clinica_ejemplo.get("volumen")))
        volumen = c2.numero("Volumen tumoral (cm³)", 0.0, 1000.0, float(clinica_ejemplo.get("volumen") or 10.0), 0.5,
                                  disabled=not vol_conocido, help="Mismas unidades que tum_vol en patients.csv (≈ cm³).")
        opciones = ["Desconocido", "Positivo", "Negativo"]
        hr = c3.segmentos("Receptores hormonales (HR)", opciones,
                                  default=_opcion(clinica_ejemplo.get("HR")), key=f"hr_{titulo_visor}") or "Desconocido"
        her2 = c4.segmentos("HER2", opciones, default=_opcion(clinica_ejemplo.get("HER2")),
                                    key=f"her2_{titulo_visor}") or "Desconocido"
    a_num = {"Desconocido": None, "Positivo": 1.0, "Negativo": 0.0}
    clinica = {"edad": edad if edad_conocida else None, "volumen": volumen if vol_conocido else None,
               "HR": a_num[hr], "HER2": a_num[her2]}
    hay_clinica = any(v is not None for v in clinica.values())

    # ---------------------------------------------------------------- 3. lectura
    seccion("3", "Lectura")
    if pred is None:
        aviso("Carga imágenes o elige una paciente de ejemplo para ver la predicción.", "info")
        pie()
        pagina.terminar()

    p_img = motor.calibrar(paquete, pred.p_cruda)
    umbral_img = paquete.info["politica"]["umbral"]
    dispersion = float(np.std([motor.calibrar(paquete, p) for p in pred.por_red]))
    # La distribución se muestra para la estimación más completa disponible.
    referencia = (p_img, img_info["perfiles"], "solo imagen")
    col_img, col_fus = pagina.columnas(2, gap="medium")
    with col_img:
        lectura("Solo imagen · CNN", p_img, img_info["perfiles"], umbral_img,
                f"{len(nombres)} corte(s) × 5 redes × 4 reflexiones · desacuerdo entre redes ± {num(dispersion, 3)}")
    with col_fus:
        if fusion is None or fus_info is None:
            lectura_vacia("Imagen + clínica", "El modelo combinado aún no está congelado.")
        elif not hay_clinica:
            lectura_vacia("Imagen + clínica", "Añade al menos un dato clínico para ver la estimación combinada.")
        else:
            d = fusion.decidir(pred.p_cruda, clinica)
            sub = d["submodelo"]
            info_p = fus_info["perfiles"].get(sub, img_info["perfiles"])
            if sub in fus_info["perfiles"]:
                referencia = (d["probabilidad_pcr"], info_p, "imagen + clínica")
            cambio = 100 * (d["probabilidad_pcr"] - p_img)
            lectura("Imagen + clínica", d["probabilidad_pcr"], info_p, d["umbral"],
                    f"submodelo {escapar(sub)} · según los datos disponibles",
                    extra=f" Con la clínica, la estimación cambia <b>{cambio:+.1f} puntos</b>.".replace(".", ",", 1))

    if etiqueta_real is not None:
        pagina.texto_auxiliar(f"Resultado real de esta paciente de entrenamiento: {'pCR' if etiqueta_real else 'no pCR'}.")

    p_ref, info_ref, nombre_ref = referencia
    grafica(distribucion_grafica(info_ref, f"Dónde cae entre las pacientes de desarrollo · {nombre_ref}", marca=p_ref,
                                 texto_marca=f"esta paciente · {num(100 * p_ref, 1)} %"))
    n_ref = f"{info_ref['n']:,}".replace(",", ".")
    pagina.texto_auxiliar(f"Probabilidades fuera de fold de las {n_ref} pacientes de desarrollo con el mismo modelo "
                          "y los mismos datos disponibles. La línea blanca es esta estimación; las zonas de color, los cinco "
                          "perfiles. Pasa el ratón por las barras para ver cuántas pacientes de cada tramo alcanzaron pCR.")

    # ---------------------------------------------------------------- 4. dentro de la red
    seccion("4", "Recorrido del corte por la red",
            "El corte seleccionado se propaga por una de las cinco redes. Cada miniatura es un mapa de activación real; "
            "las líneas representan los pesos efectivos entre los filtros con mayor contribución a la salida.")
    k = pagina.rango("Corte que se propaga", options=list(range(len(nombres))), format_func=lambda i: (
        f"{nombres[i]} · p {pct(p_cortes[i], 1)}"), value=int(np.argsort(p_cortes)[len(p_cortes) // 2])) \
        if len(nombres) > 1 else 0
    red_neuronal(estado.recorrido(pixeles[k]))

    aviso("<b>Cómo interpretarlo.</b> El perfil indica en qué grupo de las pacientes de la validación cruzada cae la "
          "estimación y qué proporción de ese grupo alcanzó pCR (pasa el ratón por la regla o por la escalera de perfiles). "
          "El perfil intermedio reúne al 40 % central y es el de mayor incertidumbre; los extremos, al 10 % en el que el "
          "modelo está más seguro. Ningún resultado debe modificar el tratamiento ni la cirugía previstos.", "clave")
    pie()
