"""Resúmenes ligeros para la web: métricas, curvas, perfiles y datos, calculados solo con train.

Escribe ``app/recursos/*.json`` (versionados) para que la web no dependa de
``runs/`` (ignorado en Git). Nunca lee test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from utils import datos, metricas

RECURSOS = datos.RAIZ / "app" / "recursos"
RUNS = datos.RAIZ / "runs"


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margen = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centro - margen), float(centro + margen)]


PERFILES = ("muy_baja", "baja", "intermedia", "alta", "muy_alta")
# Percentiles de corte: 10 % · 20 % · 40 % · 20 % · 10 % de las pacientes. Los extremos son estrechos
# para que reúnan solo a las pacientes en las que el modelo está más seguro.
CUANTILES_PERFILES = (0.10, 0.30, 0.70, 0.90)
SUBTIPOS = ("HR+/HER2−", "HER2+", "Triple negativo")


def subtipo(df: pd.DataFrame) -> np.ndarray:
    return np.select([df["HR"].isna() | df["HER2"].isna(), df["HER2"] == 1, df["HR"] == 1],
                     ["desconocido", "HER2+", "HR+/HER2−"], "Triple negativo")


def perfiles(y, p, subtipos=None, n_bins: int = 36) -> dict:
    """Cinco perfiles de probabilidad de pCR (percentiles de las OOF) con la pCR observada en cada uno.

    También guarda la distribución de las OOF (histograma por clase y percentiles) para situar
    a una paciente nueva entre las de desarrollo. Solo agregados; ninguna fila individual.
    """
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    cortes = [float(c) for c in np.quantile(p, CUANTILES_PERFILES)]
    grupo = np.searchsorted(cortes, p, side="right")
    prevalencia = float(y.mean())
    filas = []
    for i, nombre in enumerate(PERFILES):
        m = grupo == i
        k, n = int(y[m].sum()), int(m.sum())
        fila = {"perfil": nombre, "n": n, "pcr": k, "tasa_pcr": k / n if n else float("nan"),
                "ic95": _wilson(k, n), "razon": (k / n) / prevalencia if n else float("nan"),
                "p_min": float(p[m].min()), "p_max": float(p[m].max()), "p_media": float(p[m].mean())}
        if subtipos is not None:
            fila["subtipos"] = {s: int((np.asarray(subtipos)[m] == s).sum()) for s in SUBTIPOS}
        filas.append(fila)
    bordes = np.linspace(np.floor(p.min() * 20) / 20, np.ceil(p.max() * 20) / 20, n_bins + 1)
    cajon = np.clip(np.searchsorted(bordes, p, side="right") - 1, 0, n_bins - 1)
    histograma = [{"desde": float(bordes[b]), "hasta": float(bordes[b + 1]),
                   "pcr": int(((cajon == b) & (y == 1)).sum()), "no_pcr": int(((cajon == b) & (y == 0)).sum())}
                  for b in range(n_bins)]
    return {"cortes": cortes, "regla": "percentiles 10, 30, 70 y 90 de las probabilidades OOF (solo train)",
            "prevalencia": prevalencia, "n": int(len(y)), "perfiles": filas, "histograma": histograma,
            "percentiles": [float(v) for v in np.quantile(p, np.linspace(0, 1, 101))]}


def _curva(x, y, n=200):
    """Submuestrea una curva a ~n puntos conservando extremos."""
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).round().astype(int))
    return [[float(x[i]), float(y[i])] for i in idx]


def _historias(cv: str, brazo: str) -> pd.DataFrame:
    return pd.concat(pd.read_csv(f).assign(red=str(f.parent), fold=int(f.parents[2].name.removeprefix("fold")))
                     for f in sorted((RUNS / cv).glob(f"fold*/{brazo}/semilla*/historia.csv")))


def _por_epoca(h: pd.DataFrame, columnas: list[str], minimo: int = 5) -> list[dict]:
    filas = []
    for epoca, g in h.groupby("epoca"):
        if len(g) < minimo:
            continue
        fila = {"epoca": int(epoca), "n": int(len(g))}
        for c in columnas:
            fila[c] = float(g[c].mean())
            fila[c + "_p25"], fila[c + "_p75"] = (float(v) for v in g[c].quantile([0.25, 0.75]))
        filas.append(fila)
    return filas


COLUMNAS_CURVAS_CNN = ["bce_pcr_fit_aumentada", "fit_bce_obj", "sel_bce_obj", "fit_auc", "sel_auc"]


def _curvas_por_fold(h: pd.DataFrame) -> dict:
    """Media e IQR entre semillas, separadas por fold y época (no mezcla folds)."""
    return {str(int(fold)): _por_epoca(g, COLUMNAS_CURVAS_CNN, minimo=1)
            for fold, g in h.groupby("fold", sort=True)}


def exportar_curvas_cnn() -> Path:
    """Actualiza únicamente las curvas guardadas; no entrena ni lee OOF o test."""
    destino = RECURSOS / "modelo_imagen.json"
    info = json.loads(destino.read_text(encoding="utf-8"))
    cv, brazo = info["cv"], info["brazo"]
    carpeta = RUNS / cv
    if not list(carpeta.glob(f"fold*/{brazo}/semilla*/historia.csv")):
        raise FileNotFoundError(f"Faltan históricos: {carpeta}/fold*/{brazo}/semilla*/historia.csv. "
                                "Ejecutar en el equipo que conserva el entrenamiento.")
    h = _historias(cv, brazo)
    info["curvas"] = _por_epoca(h, COLUMNAS_CURVAS_CNN)
    info["curvas_por_fold"] = _curvas_por_fold(h)
    destino.write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    return destino


def modelo_imagen(cv: str = "cv_presupuesto", brazo: str = "ponderada_aux0",
                  paquete: str = "imagen_v2", control: str = "cv_control") -> dict:
    info = json.loads((datos.RAIZ / "reports" / "modelo_elegido" / paquete / "paquete.json").read_text(encoding="utf-8"))
    politica = info["politica"]
    oof = pd.read_csv(RUNS / cv / f"oof_{brazo}.csv")
    y, p = oof["pCR"].to_numpy(), oof["p"].to_numpy()
    pc = metricas.aplicar_platt(politica["calibracion_platt"], p)
    media, de, folds = metricas.auc_media_folds(oof, "p")
    rng = np.random.default_rng(20261027)
    boot = [metricas.auc(y[i], p[i]) for i in (
        np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum()) for c in (0, 1)])
        for _ in range(2000))]
    fpr, tpr, _ = roc_curve(y, p)
    fpr_c, tpr_c, _ = roc_curve(y, oof["p_cohorte"])
    prec, rec, _ = precision_recall_curve(y, p)
    bordes = np.quantile(pc, np.linspace(0, 1, 11))
    decil = np.clip(np.searchsorted(bordes, pc, side="right") - 1, 0, 9)
    sub = subtipo(oof)
    h = _historias(cv, brazo)
    hc = _historias(control, brazo)
    return {
        "paquete": paquete, "cv": cv, "brazo": brazo, "epocas": info["epocas"], "semillas": info["semillas"],
        "parametros": info["parametros_entrenables"], "n_pacientes": int(len(oof)), "n_pcr": int(y.sum()),
        "metricas": {
            "auc_media_folds": media, "auc_de_folds": de, "auc_por_fold": folds,
            "auc_agrupada": metricas.auc(y, p), "auc_ic95": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
            "ap": float(average_precision_score(y, p)), "prevalencia": float(y.mean()),
            "auc_cohorte_media_folds": metricas.auc_media_folds(oof, "p_cohorte")[0],
            "ap_cohorte": float(average_precision_score(y, oof["p_cohorte"])),
            "auc_por_cohorte": {c: metricas.auc(g["pCR"], g["p"]) for c, g in oof.groupby("cohorte")},
            "auc_por_subtipo": {s: metricas.auc(g["pCR"], g["p"]) for s, g in oof.assign(s=sub).groupby("s")
                                if s != "desconocido"},
        },
        "curvas": _por_epoca(h, COLUMNAS_CURVAS_CNN),
        "curvas_por_fold": _curvas_por_fold(h),
        "curvas_control": _por_epoca(hc, ["fit_auc", "sel_auc"]),
        "roc": _curva(fpr, tpr), "roc_cohorte": _curva(fpr_c, tpr_c),
        "pr": _curva(rec[::-1], prec[::-1]),
        "calibracion": [{"predicha": float(pc[decil == d].mean()), "observada": float(y[decil == d].mean()),
                         "n": int((decil == d).sum())} for d in range(10)],
        "umbrales": {"youden": politica["oof_con_umbral"], "macro_f1": politica["oof_con_umbral_macro_f1"]},
        "calibracion_platt": politica["calibracion_platt"],
        "perfiles": perfiles(y, pc, sub),
    }


def modelo_fusion(interna: str = "cv_interna_fusion", paquete: str = "fusion_clinica_v1") -> dict | None:
    """Resumen de la CV anidada de la fusión; None si aún no existe."""
    evaluacion = RUNS / interna / "evaluacion"
    if not (evaluacion / "resumen.json").exists():
        return None
    resumen = json.loads((evaluacion / "resumen.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(evaluacion / "oof_fusion.csv")
    alias = resumen["alias"]
    curvas, perfiles_sub, curvas_pr, calibraciones = {}, {}, {}, {}
    for nombre in ("imagen", "subtipo", "completa", "completa+imagen"):
        t = oof.dropna(subset=[alias[nombre]])
        fpr, tpr, _ = roc_curve(t["pCR"], t[alias[nombre]])
        curvas[nombre] = _curva(fpr, tpr)
        prec, rec, _ = precision_recall_curve(t["pCR"], t[alias[nombre]])
        curvas_pr[nombre] = _curva(rec[::-1], prec[::-1])
        grupos = pd.qcut(t[alias[nombre]], 10, duplicates="drop")
        calibraciones[nombre] = [
            {"predicha": float(g[alias[nombre]].mean()), "observada": float(g["pCR"].mean()), "n": int(len(g))}
            for _, g in t.groupby(grupos, observed=True)
        ]
    for clave in resumen["brazos"]:
        if clave.endswith("|normal") and clave.split("|")[0].endswith("imagen"):
            t = oof.dropna(subset=[clave])
            perfiles_sub[clave.split("|")[0]] = perfiles(t["pCR"], t[clave], subtipo(t))
    coef = pd.read_csv(evaluacion / "coeficientes_completa_imagen.csv")
    info_path = datos.RAIZ / "reports" / "modelo_elegido" / paquete / "paquete.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    umbrales = {nombre: {"youden": sub["politica"]["oof_con_umbral"],
                         "macro_f1": sub["politica"]["oof_con_umbral_macro_f1"]}
                for nombre, sub in info.get("submodelos", {}).items()}
    return {"resumen": resumen, "roc": curvas, "perfiles": perfiles_sub,
            "pr": curvas_pr, "calibracion": calibraciones, "umbrales": umbrales,
            "diagnostico_generalizacion": resumen.get("diagnostico_generalizacion", []),
            "configuracion": {"penalizacion": "L2", "C": 1.0, "max_iter": 2000,
                              "class_weight": None, "submodelo": info.get("submodelos", {}).get("subtipo+edad+volumen+imagen")},
            "coeficientes_por_fold": coef.to_dict(orient="records"),
            "paquete_disponible": (datos.RAIZ / "reports" / "modelo_elegido" / paquete / "paquete.json").exists()}


def escribir() -> list[Path]:
    RECURSOS.mkdir(parents=True, exist_ok=True)
    rutas = [RECURSOS / "modelo_imagen.json"]
    rutas[0].write_text(json.dumps(modelo_imagen(), indent=1, ensure_ascii=False), encoding="utf-8")
    fusion = modelo_fusion()
    if fusion is not None:
        rutas.append(RECURSOS / "modelo_fusion.json")
        rutas[1].write_text(json.dumps(fusion, indent=1, ensure_ascii=False), encoding="utf-8")
    return rutas


def exportar_perfiles() -> list[Path]:
    """Recalcula solo los perfiles de pCR desde las OOF guardadas; no entrena ni lee test."""
    destino_img, destino_fus = RECURSOS / "modelo_imagen.json", RECURSOS / "modelo_fusion.json"
    img = json.loads(destino_img.read_text(encoding="utf-8"))
    oof_img = RUNS / img["cv"] / f"oof_{img['brazo']}.csv"
    oof_fus = RUNS / "cv_interna_fusion" / "evaluacion" / "oof_fusion.csv"
    for ruta in (oof_img, oof_fus):
        if not ruta.exists():
            raise FileNotFoundError(f"Faltan predicciones OOF: {ruta}. Ejecutar en el equipo que conserva runs/.")
    oof = pd.read_csv(oof_img)
    pc = metricas.aplicar_platt(img["calibracion_platt"], oof["p"].to_numpy())
    img.pop("bandas", None)
    img["perfiles"] = perfiles(oof["pCR"].to_numpy(), pc, subtipo(oof))
    destino_img.write_text(json.dumps(img, indent=1, ensure_ascii=False), encoding="utf-8")
    rutas = [destino_img]
    if destino_fus.exists():
        fus = json.loads(destino_fus.read_text(encoding="utf-8"))
        oof = pd.read_csv(oof_fus)
        fus.pop("bandas", None)
        fus["perfiles"] = {}
        for clave in fus["resumen"]["brazos"]:
            if clave.endswith("|normal") and clave.split("|")[0].endswith("imagen"):
                t = oof.dropna(subset=[clave])
                fus["perfiles"][clave.split("|")[0]] = perfiles(t["pCR"], t[clave], subtipo(t))
        destino_fus.write_text(json.dumps(fus, indent=1, ensure_ascii=False), encoding="utf-8")
        rutas.append(destino_fus)
    return rutas


def _medias_por_paciente(indice: pd.DataFrame, lado: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Media de los cortes de cada paciente por fase, reducida a lado × lado, y su región central."""
    bloque = datos.LADO // lado
    pacientes, mapas = [], []
    for pid, g in indice.groupby("patient_id", sort=True):
        fases = np.stack([np.stack([datos.leer_png(datos.DATOS / r) for r in f[list(datos.COLUMNAS_FASE)]])
                          for _, f in g.iterrows()]).astype(np.float32) / 255
        centro = fases[:, :, 96:160, 96:160].mean(axis=(0, 2, 3))
        mapas.append(fases.reshape(len(g), 3, lado, bloque, lado, bloque).mean(axis=(0, 3, 5)))
        pacientes.append({"patient_id": pid, "pCR": int(g["pCR"].iloc[0]),
                          **{f"centro_{f}": float(v) for f, v in zip(datos.FASES, centro)}})
    return pd.DataFrame(pacientes), np.stack(mapas)


def _tasas(t: pd.DataFrame, filas: str, columnas: str) -> list[dict]:
    return [{filas: str(a), columnas: str(b), "n": int(len(g)), "pcr": int(g["pCR"].sum()),
             "tasa_pcr": float(g["pCR"].mean())}
            for (a, b), g in t.groupby([filas, columnas], observed=True)]


def exportar_datos(lado: int = 32) -> Path:
    """Resúmenes agregados de imágenes, clínica y particiones de train para las páginas del panel."""
    from training.fusion import tabla_clinica

    info = json.loads((RECURSOS / "modelo_imagen.json").read_text(encoding="utf-8"))
    indice = datos.cargar_indice("train")
    pacientes, mapas = _medias_por_paciente(indice, lado)
    y = pacientes["pCR"].to_numpy()
    redondea = lambda a: np.round(a, 4).tolist()
    imagenes = {"lado": lado, "n_pcr": int(y.sum()), "n_no_pcr": int((y == 0).sum()), "region_central": [96, 160],
                "medias": {clase: {f: redondea(mapas[y == c, k].mean(axis=0)) for k, f in enumerate(datos.FASES)}
                           for clase, c in (("pcr", 1), ("no_pcr", 0))},
                "cinetica": [{"clase": clase, "fase": f, "media": float(g[f"centro_{f}"].mean()),
                              "p25": float(g[f"centro_{f}"].quantile(0.25)),
                              "p75": float(g[f"centro_{f}"].quantile(0.75))}
                             for clase, g in (("pCR", pacientes[y == 1]), ("no pCR", pacientes[y == 0]))
                             for f in datos.FASES]}

    clin = tabla_clinica("train")
    clin["subtipo"] = subtipo(clin)
    clin["volumen"] = np.expm1(clin["log_volumen"])
    t = clin[clin["subtipo"] != "desconocido"].copy()
    t["edad_grupo"] = pd.cut(t["edad"], [0, 40, 50, 60, 120], right=False,
                             labels=["< 40", "40–49", "50–59", "≥ 60"])
    cortes_vol = [float(v) for v in t["volumen"].quantile([1 / 3, 2 / 3])]
    t["volumen_grupo"] = pd.cut(t["volumen"], [-np.inf, *cortes_vol, np.inf],
                                labels=["Pequeño", "Mediano", "Grande"])
    clinica = {
        "n": int(len(clin)), "n_pcr": int(clin["pCR"].sum()), "prevalencia": float(clin["pCR"].mean()),
        "por_subtipo": [{"subtipo": s, "n": int(len(g)), "pcr": int(g["pCR"].sum()),
                         "tasa_pcr": float(g["pCR"].mean()), "ic95": _wilson(int(g["pCR"].sum()), len(g))}
                        for s, g in t.groupby("subtipo") if s in SUBTIPOS],
        "por_cohorte": [{"cohorte": c, "n": int(len(g)), "pcr": int(g["pCR"].sum()), "tasa_pcr": float(g["pCR"].mean())}
                        for c, g in clin.groupby("cohorte")],
        "subtipo_edad": _tasas(t, "subtipo", "edad_grupo"),
        "subtipo_volumen": _tasas(t, "subtipo", "volumen_grupo"),
        "cortes_volumen": cortes_vol,
        "desconocidos": int((clin["subtipo"] == "desconocido").sum()),
    }

    particiones = []
    for ruta in sorted((RUNS / info["cv"]).glob("roles_fold*.csv")):
        roles = pd.read_csv(ruta)
        fold = int(ruta.stem.removeprefix("roles_fold"))
        particiones.append({"fold": fold, **{rol: {"n": int(len(g)), "pcr": int(g["pCR"].sum())}
                                             for rol, g in roles.groupby("rol")}})
    destino = RECURSOS / "datos.json"
    destino.write_text(json.dumps({"imagenes": imagenes, "clinica": clinica, "particiones": particiones},
                                  ensure_ascii=False), encoding="utf-8")
    return destino
