"""Informe final desde predicciones guardadas: sin inferencia ni ajustes en test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from utils import datos, metricas


def _sha(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _guardar(ruta: Path, objeto: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(objeto, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def clasificacion(y, p, umbrales) -> dict:
    """Aplica el umbral de cada paciente; no busca un umbral global en test."""
    y, p, u = np.asarray(y), np.asarray(p, dtype=float), np.asarray(umbrales, dtype=float)
    if p.shape != y.shape or u.shape not in ((), p.shape) or not np.isfinite(p).all() or not np.isfinite(u).all():
        raise ValueError("Predicciones y umbrales deben ser finitos y tener dimensiones compatibles")
    resultado = metricas.binarias(y, p - u, 0.0)
    resultado.pop("umbral")
    return resultado


def intervalos_auc(y, fusion, imagen, n=5000, semilla=20261029) -> dict:
    """Bootstrap pareado por paciente, estratificado por clase; sin reentrenar."""
    y, fusion, imagen = np.asarray(y), np.asarray(fusion), np.asarray(imagen)
    if not (len(y) == len(fusion) == len(imagen)) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Se requieren las mismas pacientes y ambas clases")
    rng = np.random.default_rng(semilla)
    estratos = [np.flatnonzero(y == c) for c in (0, 1)]
    boot = np.empty((n, 2))
    for k in range(n):
        idx = np.concatenate([rng.choice(g, len(g)) for g in estratos])
        boot[k] = metricas.auc(y[idx], fusion[idx]), metricas.auc(y[idx], imagen[idx])
    ic = lambda valores: [float(v) for v in np.quantile(valores, [.025, .975])]
    return {"fusion": ic(boot[:, 0]), "imagen": ic(boot[:, 1]),
            "diferencia": {"delta": metricas.auc(y, fusion) - metricas.auc(y, imagen),
                           "ic95": ic(boot[:, 0] - boot[:, 1]), "remuestreos": n, "semilla": semilla}}


def resumen_modelo(y, p, umbrales, nombre: str, ic95: list[float]) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    # Intervalos fijados antes de evaluar. La agrupación solo describe calibración.
    grupo = np.clip(np.floor(p * 10).astype(int), 0, 9)
    calibracion = [{"predicha": float(p[grupo == k].mean()), "observada": float(y[grupo == k].mean()),
                    "n": int((grupo == k).sum()), "desde": k / 10, "hasta": (k + 1) / 10}
                   for k in range(10) if (grupo == k).any()]
    return {"nombre": nombre, **metricas.resumen_probabilidades(y, p), "auc_ic95": ic95,
            "clasificacion": clasificacion(y, p, umbrales),
            "roc": [[float(x), float(v)] for x, v in zip(fpr, tpr)],
            "pr": [[float(x), float(v)] for x, v in zip(recall[::-1], precision[::-1])],
            "calibracion": calibracion}


def exportar(raiz: Path = datos.RAIZ) -> Path:
    """Reexportable desde el CSV existente; nunca abre PNG ni carga pesos de red."""
    raiz = Path(raiz)
    paquete = raiz / "reports/modelo_elegido/fusion_clinica_v1"
    imagen = raiz / "reports/modelo_elegido/imagen_v2"
    salida = raiz / "reports/evaluacion_final/fusion_clinica_v1"
    protocolo = json.loads((salida / "protocolo.json").read_text(encoding="utf-8"))
    for rel, h in protocolo["sha256_modelos"].items():
        # La instantánea previa incluía el índice documental, que se actualiza al cerrar test.
        if rel == "reports/modelo_elegido/README.md":
            continue
        if _sha(raiz / rel) != h:
            raise ValueError(f"El artefacto congelado ha cambiado: {rel}")
    metadata = raiz / "dataset/breastdcedl/metadata"
    for rel, h in protocolo["sha256_metadata"].items():
        if _sha(metadata / rel) != h:
            raise ValueError(f"Los metadatos han cambiado: {rel}")
    predicciones = paquete / "predicciones_test.csv"
    huella = _sha(predicciones)
    previo = salida / "resumen.json"
    if previo.exists() and json.loads(previo.read_text())["sha256_predicciones"] != huella:
        raise ValueError("Las predicciones difieren de las de la evaluación cerrada")
    t = pd.read_csv(predicciones).set_index("patient_id").sort_index()
    pacientes = pd.read_csv(metadata / "patients.csv")
    test = pacientes[pacientes["split"] == "test"].set_index("pid").sort_index()
    if not t.index.is_unique or set(t.index) != set(test.index):
        raise ValueError("El informe debe cubrir una vez a cada paciente de test")
    if not np.array_equal(t.pCR.to_numpy(), test.pCR.to_numpy()) or not t.cohorte.equals(test.dataset.rename("cohorte")):
        raise ValueError("Las etiquetas o cohortes no coinciden con test")
    if set(t.index) & set(pacientes.loc[pacientes.split == "train", "pid"]):
        raise ValueError("Hay pacientes compartidas entre train y test")
    if not np.isfinite(t[["p", "p_imagen", "umbral"]].to_numpy()).all() or not t[["p", "p_imagen", "umbral"]].ge(0).all().all() or not t[["p", "p_imagen", "umbral"]].le(1).all().all():
        raise ValueError("Probabilidades o umbrales fuera de rango")
    info = json.loads((imagen / "paquete.json").read_text())
    for submodelo, g in t.groupby("submodelo"):
        esperado = (protocolo["umbral_cnn"] if submodelo == "imagen sola (paquete de imagen)"
                    else protocolo["umbrales_fusion"][submodelo])
        if not np.allclose(g.umbral, esperado, rtol=0, atol=1e-15):
            raise ValueError("El umbral no coincide con la política congelada")
    p_imagen = metricas.aplicar_platt(info["politica"]["calibracion_platt"], t.p_imagen)
    config = protocolo["incertidumbre"]
    intervalos = intervalos_auc(t.pCR, t.p, p_imagen, config["remuestreos"], config["semilla"])
    modelos = {
        "fusion": resumen_modelo(t.pCR, t.p, t.umbral, "Imagen + clínica", intervalos["fusion"]),
        "imagen": resumen_modelo(t.pCR, p_imagen, protocolo["umbral_cnn"], "CNN de imagen", intervalos["imagen"]),
    }
    original = json.loads((paquete / "evaluacion_test.json").read_text())
    if not np.isclose(original["auc"], modelos["fusion"]["auc"], rtol=0, atol=1e-12):
        raise ValueError("La AUC no coincide con la evaluación original")
    if any(original[k] != modelos["fusion"]["clasificacion"][k] for k in ("TP", "TN", "FP", "FN")):
        raise ValueError("La matriz de confusión no coincide con la evaluación original")
    cohortes = []
    for cohorte, g in t.groupby("cohorte"):
        p_cnn = metricas.aplicar_platt(info["politica"]["calibracion_platt"], g.p_imagen)
        fila = {"cohorte": cohorte, "n": len(g), "pcr": int(g.pCR.sum()),
                "clasificacion_fusion": clasificacion(g.pCR, g.p, g.umbral)}
        for clave, p in (("auc_fusion", g.p), ("auc_imagen", p_cnn)):
            valor = metricas.auc(g.pCR, p)
            fila[clave] = valor if np.isfinite(valor) else None
        cohortes.append(fila)
    umbrales = [{"submodelo": clave, "n": len(g), "umbral": float(g.umbral.iloc[0])}
               for clave, g in t.groupby("submodelo")]
    resumen = {"estado": "evaluado", "paquete": protocolo["paquete"], "fecha": protocolo["fecha_cierre"][:10],
               "n_pacientes": len(t), "n_pcr": int(t.pCR.sum()), "prevalencia": float(t.pCR.mean()),
               "politica": protocolo["politica"], "limite": protocolo["limite"],
               "sha256_predicciones": huella, "sha256_protocolo": _sha(salida / "protocolo.json"),
               "modelos": modelos, "diferencia_auc": intervalos["diferencia"],
               "cohortes": cohortes, "umbrales_aplicados": umbrales,
               "fuente": "reports/evaluacion_final/fusion_clinica_v1/resumen.json"}
    marca_imagen = imagen / "evaluacion_test.json"
    if marca_imagen.exists() and json.loads(marca_imagen.read_text()).get("sha256_predicciones_fuente") != huella:
        raise ValueError("La CNN ya tiene otra evaluación de test: no se sobrescribe")
    _guardar(salida / "resumen.json", resumen)
    _guardar(raiz / "app/recursos/evaluacion_test.json", resumen)
    # La misma inferencia ya evaluó la CNN: se registra sin una segunda ejecución.
    _guardar(marca_imagen, {"paquete": "imagen_v2", "pacientes": len(t), "pcr": int(t.pCR.sum()),
                           "auc": modelos["imagen"]["auc"], "auc_ic95": intervalos["imagen"],
                           "probabilidades_calibradas": {k: modelos["imagen"][k] for k in ("auc", "ap", "brier", "bce")},
                           "umbral_congelado": {"umbral": protocolo["umbral_cnn"], **modelos["imagen"]["clasificacion"]},
                           "fuente": "Predicciones de imagen de la única ejecución del paquete combinado; sin repetir inferencia.",
                           "sha256_predicciones_fuente": huella,
                           "auc_por_cohorte": {g["cohorte"]: g["auc_imagen"] for g in cohortes}})
    return salida / "resumen.json"
