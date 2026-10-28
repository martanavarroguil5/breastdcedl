"""¿Dónde se pierde el aporte de la imagen sobre la clínica? Solo OOF de train (cv_presupuesto)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from training import fusion

RAIZ = Path("runs/cv_presupuesto")
SAL = Path("runs/analisis_aporte_imagen")
SAL.mkdir(parents=True, exist_ok=True)
CLIN = ["edad", "log_volumen", "HR", "HER2"]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

# ---------------------------------------------------------------- cortes OOF por semilla
filas = []
for f in sorted(RAIZ.glob("fold*/ponderada_aux0/semilla*/predicciones.csv")):
    d = pd.read_csv(f)
    d = d[d["rol"] == "exterior"]
    d["semilla"] = f.parent.name
    filas.append(d[["sample_id", "patient_id", "p", "semilla"]])
cortes = pd.concat(filas)
cortes["z"] = logit(cortes["p"])
clin = fusion.tabla_clinica("train")
y = clin["pCR"]

# Agregaciones por paciente (media de las 3 semillas por corte primero)
c = cortes.groupby(["patient_id", "sample_id"])["z"].mean().reset_index()
g = c.groupby("patient_id")["z"]
agreg = pd.DataFrame({
    "media_logit": g.mean(), "mediana": g.median(), "max": g.max(), "min": g.min(),
    "top3": g.apply(lambda s: s.nlargest(3).mean()), "de_cortes": g.std(),
})
agreg["media_prob"] = logit(cortes.groupby(["patient_id", "sample_id"])["p"].mean().groupby("patient_id").mean())
tab = clin.join(agreg, how="inner")
completa = tab.dropna(subset=CLIN)
res = {"n_pacientes": len(tab), "n_completa": len(completa)}


def dauc_cv(t, base, extra, reps=20):
    """ΔAUC media (repetida 5-fold) de añadir `extra` a la logística con `base`."""
    out = []
    for r in range(reps):
        pa, pb = np.zeros(len(t)), np.zeros(len(t))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(t, t["pCR"]):
            for cols, dest in ((base, pa), (base + extra, pb)):
                X = t[cols].to_numpy(float)
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                m = LogisticRegression(C=1.0, max_iter=2000).fit((X[tr] - mu) / sd, t["pCR"].iloc[tr])
                dest[te] = m.decision_function((X[te] - mu) / sd)
        out.append(roc_auc_score(t["pCR"], pb) - roc_auc_score(t["pCR"], pa))
    out = np.array(out)
    return {"dauc": float(out.mean()), "de": float(out.std()), "frac_pos": float((out > 0).mean())}


def lr_test(t, base, extra):
    def ll(cols):
        X = t[cols].to_numpy(float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        m = LogisticRegression(C=1e6, max_iter=5000).fit(X, t["pCR"])
        p = m.predict_proba(X)[:, 1]
        return float(np.sum(t["pCR"] * np.log(p) + (1 - t["pCR"]) * np.log(1 - p)))
    d = 2 * (ll(base + extra) - ll(base))
    return {"chi2": d, "p": float(stats.chi2.sf(d, len(extra)))}


# ---------------------------------------------------------------- A. redundancia con la clínica
img = tab["media_prob"]
red = {k: float(stats.spearmanr(tab[k], img, nan_policy="omit")[0]) for k in CLIN}
red["auc_img_predice_HR"] = float(roc_auc_score(tab.dropna(subset=["HR"])["HR"], tab.dropna(subset=["HR"])["media_prob"]))
red["auc_img_predice_HER2"] = float(roc_auc_score(tab.dropna(subset=["HER2"])["HER2"], tab.dropna(subset=["HER2"])["media_prob"]))
# R² de la puntuación de imagen explicada por la clínica
X = completa[CLIN].to_numpy(float)
X1 = np.column_stack([np.ones(len(X)), X])
b = np.linalg.lstsq(X1, completa["media_prob"], rcond=None)[0]
resid = completa["media_prob"] - X1 @ b
red["r2_imagen_por_clinica"] = float(1 - resid.var() / completa["media_prob"].var())
red["auc_imagen"] = float(roc_auc_score(completa["pCR"], completa["media_prob"]))
red["auc_residuo_imagen"] = float(roc_auc_score(completa["pCR"], resid))
res["A_redundancia"] = red

# ---------------------------------------------------------------- B. por subtipo
sub = completa.assign(subtipo=np.select(
    [completa.HER2 == 1, (completa.HR == 0) & (completa.HER2 == 0)], ["HER2+", "TN"], "HR+/HER2−"))
B = {}
for s, t in sub.groupby("subtipo"):
    B[s] = {"n": len(t), "pcr": float(t.pCR.mean()), "auc_imagen": float(roc_auc_score(t.pCR, t.media_prob)),
            "auc_clinica_sin_subtipo": float(roc_auc_score(t.pCR, -t.edad.fillna(0)) if t.pCR.nunique() > 1 else np.nan),
            **{f"aporte_{k}": v for k, v in dauc_cv(t, ["edad", "log_volumen"], ["media_prob"], reps=10).items()}}
res["B_por_subtipo"] = B

# ---------------------------------------------------------------- C. agregación de cortes
C = {}
for k in ["media_prob", "media_logit", "mediana", "max", "top3", "min"]:
    C[k] = {"auc": float(roc_auc_score(completa.pCR, completa[k])), **dauc_cv(completa, CLIN, [k]),
            **lr_test(completa, CLIN, [k])}
C["media+de_cortes"] = {**dauc_cv(completa, CLIN, ["media_prob", "de_cortes"]),
                        **lr_test(completa, CLIN, ["media_prob", "de_cortes"])}
C["de_cortes_sola_sobre_media"] = lr_test(completa, CLIN + ["media_prob"], ["de_cortes"])
res["C_agregacion"] = C

# ---------------------------------------------------------------- D. ruido entre semillas
ps = cortes.groupby(["patient_id", "semilla"])["p"].mean().unstack()
ps = ps.loc[completa.index]
corr = ps.corr(method="spearman").to_numpy()
rho = float(corr[np.triu_indices(3, 1)].mean())
D = {"rho_entre_semillas": rho,
     "fiabilidad_3_semillas_spearman_brown": 3 * rho / (1 + 2 * rho),
     "fiabilidad_15_semillas_spearman_brown": 15 * rho / (1 + 14 * rho)}
for n in (1, 2, 3):
    vals = []
    for combo in [list(ps.columns[:n])] if n == 3 else [[c_] for c_ in ps.columns] if n == 1 else \
            [[a, b_] for i, a in enumerate(ps.columns) for b_ in ps.columns[i + 1:]]:
        t = completa.assign(semx=logit(ps[combo].mean(axis=1)))
        vals.append((roc_auc_score(t.pCR, t.semx), dauc_cv(t, CLIN, ["semx"], reps=10)["dauc"]))
    D[f"{n}_semillas"] = {"auc": float(np.mean([v[0] for v in vals])), "dauc_sobre_clinica": float(np.mean([v[1] for v in vals]))}
res["D_semillas"] = D

# ---------------------------------------------------------------- E. variación dentro de paciente (ICC)
cc = c[c.patient_id.isin(completa.index)]
entre = cc.groupby("patient_id")["z"].mean().var()
dentro = cc.groupby("patient_id")["z"].var().mean()
res["E_cortes"] = {"icc_cortes": float(entre / (entre + dentro)),
                   "cortes_por_paciente": float(cc.groupby("patient_id").size().mean())}

# ---------------------------------------------------------------- F. por terciles de riesgo clínico
pc = np.zeros(len(completa))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(completa, completa.pCR):
    X = completa[CLIN].to_numpy(float)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    pc[te] = LogisticRegression(C=1.0, max_iter=2000).fit((X[tr] - mu) / sd, completa.pCR.iloc[tr]).decision_function((X[te] - mu) / sd)
ter = pd.qcut(pc, 3, labels=["bajo", "medio", "alto"])
res["F_terciles_clinicos"] = {str(k): {"n": int(len(t)), "pcr": float(t.pCR.mean()),
                                        "auc_imagen": float(roc_auc_score(t.pCR, t.media_prob))}
                              for k, t in completa.groupby(ter, observed=True)}
res["auc_clinica_cv"] = float(roc_auc_score(completa.pCR, pc))

(SAL / "resumen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=2, ensure_ascii=False))
