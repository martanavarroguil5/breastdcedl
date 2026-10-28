"""Diagnóstico: rasgos cinéticos simples del centro (zona tumoral) frente a la CNN. Solo train."""
import json
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from experiments.analisis.aporte import completa, dauc_cv, lr_test, CLIN
s = pd.read_csv("dataset/breastdcedl/metadata/samples.csv"); s = s[s.split == "train"]
filas = []
for _, f in s.iterrows():
    a = lambda k: np.asarray(Image.open("dataset/breastdcedl/" + f[k]), dtype=np.float32)[64:192, 64:192] / 255
    pre, ea, la = a("path_pre"), a("path_early"), a("path_late")
    E = ea - pre
    m = E > np.quantile(E, 0.9)
    ser = E[m] / np.maximum(la[m] - pre[m], 1e-3)
    filas.append({"patient_id": f.patient_id, "realce_top10": E[m].mean(), "realce_rel": (E[m] / np.maximum(pre[m], .02)).mean(),
                  "frac_lavado": float(((la - ea)[m] < -0.01).mean()), "ser_med": float(np.median(np.clip(ser, 0, 5))),
                  "area_realce": float((E > 0.1).mean()), "late_vs_early": float(((la - ea)[m]).mean())})
k = pd.DataFrame(filas).groupby("patient_id").mean()
t = completa.join(k, how="inner")
rasgos = list(k.columns)
out = {}
for r in rasgos:
    auc = roc_auc_score(t.pCR, t[r]); out[r] = {"auc": float(max(auc, 1 - auc)), **dauc_cv(t, CLIN, [r], reps=10)}
out["todos_sobre_clinica"] = {**dauc_cv(t, CLIN, rasgos, reps=10), **lr_test(t, CLIN, rasgos)}
out["todos_sobre_clinica+cnn"] = {**dauc_cv(t, CLIN + ["media_prob"], rasgos, reps=10), **lr_test(t, CLIN + ["media_prob"], rasgos)}
out["cnn_sobre_clinica+todos"] = {**dauc_cv(t, CLIN + rasgos, ["media_prob"], reps=10), **lr_test(t, CLIN + rasgos, ["media_prob"])}
out["rho_cnn_rasgos"] = {r: float(t[[r, "media_prob"]].corr("spearman").iloc[0, 1]) for r in rasgos}
json.dump(out, open("runs/analisis_aporte_imagen/cinetica.json", "w"), indent=2)
print(json.dumps(out, indent=2))
