"""Análisis exploratorio: cuánto aporta la imagen (OOF de imagen_v2) sobre la clínica.

Solo train. Logística por fold, preprocesado ajustado en los folds de ajuste.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from utils import metricas

oof = pd.read_csv("runs/cv_presupuesto/oof_ponderada_aux0.csv")
pac = pd.read_csv("dataset/breastdcedl/metadata/patients.csv").rename(columns={"pid": "patient_id"})
d = oof[["patient_id", "p", "pCR", "fold", "cohorte"]].merge(
    pac[["patient_id", "age", "tum_vol", "HR", "HER2", "HR_HER2_STATUS"]], on="patient_id", validate="1:1")
d["logvol"] = np.log1p(d.tum_vol)
d["img"] = np.log(d.p / (1 - d.p))
CLIN = ["age", "logvol", "HR", "HER2"]


def oof_logistica(cols):
    salida = np.zeros(len(d))
    for k in range(5):
        tr, te = d.fold != k, d.fold == k
        X = d[cols].copy()
        for c in cols:
            if X[c].isna().any() or c in CLIN:
                X[c + "_na"] = X[c].isna().astype(float)
        med = X[tr].median()
        X = X.fillna(med)
        mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
        X = (X - mu) / sd
        m = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], d.pCR[tr])
        salida[te.to_numpy()] = m.predict_proba(X[te])[:, 1]
    return salida


brazos = {"imagen": d.p.to_numpy(),
          "clinica": oof_logistica(CLIN),
          "clinica+imagen": oof_logistica(CLIN + ["img"]),
          "subtipo": oof_logistica(["HR", "HER2"]),
          "subtipo+imagen": oof_logistica(["HR", "HER2", "img"]),
          "volumen": oof_logistica(["logvol"]),
          "volumen+imagen": oof_logistica(["logvol", "img"])}
for k, v in brazos.items():
    d[k] = v
res = {"auc": {}}
for k in brazos:
    folds = [roc_auc_score(g.pCR, g[k]) for _, g in d.groupby("fold")]
    res["auc"][k] = {"media_folds": round(float(np.mean(folds)), 4), "de": round(float(np.std(folds, ddof=1)), 4),
                     "agrupada": round(roc_auc_score(d.pCR, d[k]), 4),
                     "ap": round(average_precision_score(d.pCR, d[k]), 4)}
res["delta"] = {f"{a} − {b}": metricas.bootstrap_delta(d, a, b) for a, b in
                (("clinica+imagen", "clinica"), ("subtipo+imagen", "subtipo"), ("volumen+imagen", "volumen"),
                 ("clinica", "imagen"))}
res["imagen_dentro_subtipo"] = {s: {"n": len(g), "pcr": int(g.pCR.sum()), "auc_imagen": round(roc_auc_score(g.pCR, g.p), 4),
                                    "auc_volumen": round(roc_auc_score(g.pCR[g.logvol.notna()], -g.logvol.dropna()), 4)}
                                for s, g in d.dropna(subset=["HR_HER2_STATUS"]).groupby("HR_HER2_STATUS")}
res["imagen_dentro_cohorte"] = {s: round(roc_auc_score(g.pCR, g.p), 4) for s, g in d.groupby("cohorte")}
res["clinica_dentro_cohorte"] = {s: round(roc_auc_score(g.pCR, g.clinica), 4) for s, g in d.groupby("cohorte")}
res["que_codifica_la_imagen"] = {
    "spearman_img_logvol": round(spearmanr(d.img, d.logvol, nan_policy="omit")[0], 3),
    "spearman_img_age": round(spearmanr(d.img, d.age, nan_policy="omit")[0], 3),
    "auc_img_para_HR": round(roc_auc_score(d.HR.dropna(), d.img[d.HR.notna()]), 3),
    "auc_img_para_HER2": round(roc_auc_score(d.HER2.dropna(), d.img[d.HER2.notna()]), 3),
    "spearman_img_residuo_clinico": round(spearmanr(d.img, d.pCR - d.clinica)[0], 3)}
print(json.dumps(res, indent=1, default=float))
