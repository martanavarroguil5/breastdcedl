"""Evaluación gráfica del modelo de imagen (imagen_v2) solo con train.

Usa la CV cv_presupuesto, brazo ponderada_aux0 (la receta del paquete), y la CV
del control (cv_control) como ejemplo de sobreentrenamiento. Nunca lee test.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from utils import metricas

RAIZ = Path("runs")
SALIDA = Path("runs/evaluacion_imagen")
SALIDA.mkdir(exist_ok=True)
BRAZO = "ponderada_aux0"
AZUL, NARANJA, AQUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
TINTA, TINTA2, FONDO = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
                     "axes.edgecolor": "#c9c8c2", "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
                     "ytick.color": TINTA2, "text.color": TINTA, "axes.grid": True, "grid.color": "#ecebe7",
                     "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "lines.linewidth": 2})


def historias(cv):
    filas = []
    for f in sorted((RAIZ / cv).glob(f"fold*/{BRAZO}/semilla*/historia.csv")):
        h = pd.read_csv(f)
        h["red"] = f"{f.parts[-4]}/{f.parts[-2]}"
        filas.append(h)
    return pd.concat(filas, ignore_index=True)


def media_por_epoca(h, col, minimo=5):
    g = h.groupby("epoca")[col]
    t = pd.DataFrame({"media": g.mean(), "p25": g.quantile(0.25), "p75": g.quantile(0.75), "n": g.size()})
    return t[t.n >= minimo]


def banda(ax, h, col, color, etiqueta, individuales=True):
    if individuales:
        for _, r in h.groupby("red"):
            ax.plot(r.epoca, r[col], color=color, alpha=0.12, lw=0.8)
    t = media_por_epoca(h, col)
    ax.fill_between(t.index, t.p25, t.p75, color=color, alpha=0.18, lw=0)
    ax.plot(t.index, t.media, color=color, label=etiqueta)
    return t


# ---------------------------------------------------------------- curvas
pres, ctrl = historias("cv_presupuesto"), historias("cv_control")
resumen = {}

fig, ejes = plt.subplots(1, 4, figsize=(19, 4.6))
ax = ejes[0]
banda(ax, pres, "bce_pcr_fit_aumentada", GRIS, "lotes de entrenamiento (aumentados)")
ax.set(title="Pérdida de entrenamiento", xlabel="época", ylabel="BCE ponderada")
ax.legend(frameon=False, fontsize=9)
ax = ejes[1]
t_fit = banda(ax, pres, "fit_bce_obj", AZUL, "fit (150 pacientes, sin aumentos)", False)
t_sel = banda(ax, pres, "sel_bce_obj", NARANJA, "selección = validación (131 pacientes)", False)
e_min = int(t_sel.media.idxmin())
ax.axvline(e_min, color=NARANJA, lw=1, ls="--")
ax.annotate(f"mínimo validación: época {e_min}", (e_min, t_sel.media.min()), xytext=(e_min + 0.4, 1.0),
            fontsize=9, color=TINTA2)
ax.set(title="Pérdida fit frente a validación (zoom)", xlabel="época", ylabel="BCE ponderada",
       ylim=(0.93, 1.01))
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax = ejes[2]
banda(ax, pres, "fit_auc", AZUL, "fit")
banda(ax, pres, "sel_auc", NARANJA, "selección = validación")
ax.axhline(0.5, color=GRIS, lw=1, ls=":")
ax.set(title="AUC por paciente", xlabel="época", ylabel="AUC", ylim=(0.35, 0.85))
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax = ejes[3]
for h, color, nombre in ((ctrl, GRIS, "control (receta antigua)"), (pres, AZUL, "presupuesto (imagen_v2)")):
    h = h.assign(brecha=h.fit_auc - h.sel_auc)
    banda(ax, h, "brecha", color, nombre, False)
ax.axhline(0, color=GRIS, lw=1, ls=":")
ax.set(title="Brecha AUC fit − validación (memorización)", xlabel="época", ylabel="Δ AUC")
ax.legend(frameon=False, fontsize=9)
for a in ejes:
    a.set_xticks(range(1, 14))
    a.set_xlim(0.5, 13.5)
fig.suptitle("imagen_v2 · CV de 5 folds × 3 semillas (15 redes) · media e intercuartil", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(SALIDA / "1_curvas_perdida_auc.png", dpi=150)
plt.close(fig)

# control frente a presupuesto: qué aspecto tiene el sobreentrenamiento
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, h, titulo in ((ejes[0], ctrl, "Control (antes): sobreentrena desde la época ~3"),
                      (ejes[1], pres, "Presupuesto (imagen_v2): 12 épocas")):
    banda(ax, h, "fit_auc", AZUL, "fit")
    banda(ax, h, "sel_auc", NARANJA, "selección")
    ax.axhline(0.5, color=GRIS, lw=1, ls=":")
    ax.set(title=titulo, xlabel="época", ylim=(0.35, 1.0), xlim=(0.5, 13.5), xticks=range(1, 14))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
ejes[0].set_ylabel("AUC por paciente")
fig.tight_layout()
fig.savefig(SALIDA / "2_control_vs_presupuesto.png", dpi=150)
plt.close(fig)

# diagnóstico numérico por red
diag = []
for red, r in pres.groupby("red"):
    r = r.sort_values("epoca")
    pend = lambda c: float(np.polyfit(r.epoca[r.epoca >= 8], r[c][r.epoca >= 8], 1)[0])
    diag.append({"red": red,
                 "min_sel_bce_obj": int(r.epoca[r.sel_bce_obj.idxmin()]),
                 "min_sel_bce": int(r.epoca[r.sel_bce.idxmin()]),
                 "max_sel_auc": int(r.epoca[r.sel_auc.idxmax()]),
                 "max_sel_auc_suavizada": int(r.epoca[r.sel_auc_suavizada.idxmax()]),
                 "pend_sel_bce_obj_8_12": pend("sel_bce_obj"), "pend_fit_bce_obj_8_12": pend("fit_bce_obj"),
                 "pend_sel_auc_8_12": pend("sel_auc"),
                 "brecha_auc_12": float(r.fit_auc.iloc[-1] - r.sel_auc.iloc[-1]),
                 "sel_auc_12": float(r.sel_auc.iloc[-1]), "sel_auc_max": float(r.sel_auc.max())})
diag = pd.DataFrame(diag)
diag.to_csv(SALIDA / "diagnostico_por_red.csv", index=False)
cdiag = []
for red, r in ctrl.groupby("red"):
    cdiag.append({"max_sel_auc": int(r.epoca[r.sel_auc.idxmax()]), "min_sel_bce": int(r.epoca[r.sel_bce.idxmin()]),
                  "epocas": len(r), "brecha_final": float(r.fit_auc.iloc[-1] - r.sel_auc.iloc[-1])})
cdiag = pd.DataFrame(cdiag)
resumen["curvas_presupuesto_media"] = {
    c: media_por_epoca(pres, c).media.round(4).to_dict() for c in
    ("bce_pcr_fit_aumentada", "fit_bce_obj", "sel_bce_obj", "fit_bce", "sel_bce", "fit_auc", "sel_auc")}
resumen["diagnostico_presupuesto"] = {
    "mediana": diag.drop(columns="red").median().round(4).to_dict(),
    "redes_con_pendiente_sel_bce_obj_positiva_8_12": int((diag.pend_sel_bce_obj_8_12 > 0).sum()),
    "redes_con_pendiente_sel_auc_negativa_8_12": int((diag.pend_sel_auc_8_12 < 0).sum()),
    "redes_pico_auc_en_epoca_ge_8": int((diag.max_sel_auc >= 8).sum()),
    "conteo_epoca_pico_auc": diag.max_sel_auc.value_counts().sort_index().to_dict(),
    "conteo_epoca_min_bce_obj": diag.min_sel_bce_obj.value_counts().sort_index().to_dict(),
    "n_redes": len(diag)}
resumen["diagnostico_control"] = {"mediana": cdiag.median().round(4).to_dict(), "n_redes": len(cdiag)}

# AUC en la época final por rol: fit completo, selección y fold exterior
oof = pd.read_csv(RAIZ / "cv_presupuesto" / f"oof_{BRAZO}.csv")
etiquetas = oof.set_index("patient_id").pCR
roles = []
for f in sorted((RAIZ / "cv_presupuesto").glob(f"fold*/{BRAZO}/semilla*/predicciones.csv")):
    p = pd.read_csv(f).groupby(["patient_id", "rol"], as_index=False).p.mean()
    p["pCR"] = p.patient_id.map(etiquetas)
    for rol, g in p.groupby("rol"):
        roles.append({"red": f"{f.parts[-4]}/{f.parts[-2]}", "rol": rol, "n": len(g),
                      "auc": roc_auc_score(g.pCR, g.p)})
roles = pd.DataFrame(roles)
resumen["auc_epoca12_por_rol"] = roles.groupby("rol").auc.describe()[["count", "mean", "std", "min", "max"]] \
    .round(4).to_dict(orient="index")

# ---------------------------------------------------------------- ROC y PR (OOF por paciente)
y, p, pc = oof.pCR.to_numpy(), oof.p.to_numpy(), oof.p_cohorte.to_numpy()
auc_folds = {int(k): roc_auc_score(g.pCR, g.p) for k, g in oof.groupby("fold")}
ap_folds = {int(k): average_precision_score(g.pCR, g.p) for k, g in oof.groupby("fold")}
auc_coh_folds = {int(k): roc_auc_score(g.pCR, g.p_cohorte) for k, g in oof.groupby("fold")}
prev = float(y.mean())
fig, ejes = plt.subplots(1, 2, figsize=(12, 5.2))
ax = ejes[0]
for k, g in oof.groupby("fold"):
    fpr, tpr, _ = roc_curve(g.pCR, g.p)
    ax.plot(fpr, tpr, color=AZUL, alpha=0.25, lw=1)
fpr, tpr, _ = roc_curve(y, pc)
ax.plot(fpr, tpr, color=GRIS, ls="--", lw=1.5, label=f"solo cohorte (AUC {roc_auc_score(y, pc):.3f})")
fpr, tpr, _ = roc_curve(y, p)
ax.plot(fpr, tpr, color=AZUL, label=f"modelo, 1.097 pacientes OOF (AUC {roc_auc_score(y, p):.3f})")
ax.plot([0, 1], [0, 1], color=GRIS, lw=1, ls=":")
ax.plot([], [], color=AZUL, alpha=0.25, lw=1,
        label=f"cada fold (media {np.mean(list(auc_folds.values())):.3f} ± {np.std(list(auc_folds.values()), ddof=1):.3f})")
ax.set(title="Curva ROC (fuera de fold, por paciente)", xlabel="1 − especificidad", ylabel="sensibilidad",
       xlim=(0, 1), ylim=(0, 1), aspect="equal")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax = ejes[1]
for k, g in oof.groupby("fold"):
    pr, rc, _ = precision_recall_curve(g.pCR, g.p)
    ax.plot(rc, pr, color=AZUL, alpha=0.25, lw=1)
pr, rc, _ = precision_recall_curve(y, pc)
ax.plot(rc, pr, color=GRIS, ls="--", lw=1.5, label=f"solo cohorte (AP {average_precision_score(y, pc):.3f})")
pr, rc, _ = precision_recall_curve(y, p)
ax.plot(rc, pr, color=AZUL, label=f"modelo OOF (AP {average_precision_score(y, p):.3f})")
ax.axhline(prev, color=GRIS, lw=1, ls=":", label=f"azar = prevalencia ({prev:.3f})")
ax.plot([], [], color=AZUL, alpha=0.25, lw=1, label="cada fold")
ax.set(title="Curva precisión–sensibilidad (PR)", xlabel="sensibilidad (recall)", ylabel="precisión (VPP)",
       xlim=(0, 1), ylim=(0, 1), aspect="equal")
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(SALIDA / "3_roc_pr.png", dpi=150)
plt.close(fig)
resumen["roc_pr"] = {"auc_agrupada": roc_auc_score(y, p), "auc_folds": auc_folds,
                     "auc_media_folds": float(np.mean(list(auc_folds.values()))),
                     "auc_de_folds": float(np.std(list(auc_folds.values()), ddof=1)),
                     "auc_cohorte_agrupada": roc_auc_score(y, pc), "auc_cohorte_folds": auc_coh_folds,
                     "ap_agrupada": average_precision_score(y, p), "ap_folds": ap_folds,
                     "ap_cohorte": average_precision_score(y, pc), "prevalencia": prev}
rng = np.random.default_rng(20261027)
boot = []
for _ in range(2000):
    i = np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum()) for c in (0, 1)])
    boot.append(roc_auc_score(y[i], p[i]))
resumen["roc_pr"]["auc_agrupada_ic95"] = np.percentile(boot, [2.5, 97.5]).round(4).tolist()

# ---------------------------------------------------------------- matrices de confusión
paquete = json.loads(Path("reports/modelo_elegido/imagen_v2/paquete.json").read_text())
pol = paquete["politica"]
pcal = metricas.aplicar_platt(pol["calibracion_platt"], p)
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (nombre, u) in zip(ejes, (("Youden (umbral del paquete)", pol["umbral"]),
                                  ("máximo macro-F1 (alternativo)", pol["umbral_macro_f1_alternativo"]))):
    yp = (pcal >= u).astype(int)
    m = np.array([[((y == a) & (yp == b)).sum() for b in (0, 1)] for a in (0, 1)])
    ax.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1.15)
    for a in (0, 1):
        for b in (0, 1):
            frac = m[a, b] / m[a].sum()
            ax.text(b, a, f"{m[a, b]}\n{frac:.0%} de la fila", ha="center", va="center", fontsize=12,
                    color="white" if frac > 0.6 else TINTA)
    ax.set_xticks([0, 1], ["pred. no pCR", "pred. pCR"])
    ax.set_yticks([0, 1], ["real no pCR\n(775)", "real pCR\n(322)"])
    ax.grid(False)
    r = metricas.binarias(y, pcal, u)
    ax.set_title(f"{nombre}: p ≥ {u:.3f}\nsens. {r['sensibilidad']:.1%} · espec. {r['especificidad']:.1%} · "
                 f"VPP {r['precision_pcr']:.1%} · macro-F1 {r['macro_f1']:.3f}", fontsize=10)
    resumen[f"confusion_{u:.4f}"] = {k: (float(v) if not isinstance(v, (int, np.integer)) else int(v))
                                    for k, v in r.items()}
fig.suptitle("Matrices de confusión OOF (1.097 pacientes de train, probabilidades calibradas)", x=0.01,
             ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(SALIDA / "4_matrices_confusion.png", dpi=150)
plt.close(fig)

# calibración
fig, ax = plt.subplots(figsize=(5.2, 5))
bordes = np.quantile(pcal, np.linspace(0, 1, 11))
grupo = np.clip(np.searchsorted(bordes, pcal, side="right") - 1, 0, 9)
xs = [pcal[grupo == g].mean() for g in range(10)]
ys = [y[grupo == g].mean() for g in range(10)]
ax.plot([0, 1], [0, 1], color=GRIS, lw=1, ls=":")
ax.plot(xs, ys, "o-", color=AZUL, ms=6)
ax.set(title="Calibración OOF (deciles)", xlabel="probabilidad predicha (Platt)", ylabel="pCR observada",
       xlim=(0, 0.6), ylim=(0, 0.6), aspect="equal")
fig.tight_layout()
fig.savefig(SALIDA / "5_calibracion.png", dpi=150)
plt.close(fig)
resumen["calibracion_rango_predicho"] = [float(pcal.min()), float(pcal.max())]

(SALIDA / "resumen.json").write_text(json.dumps(resumen, indent=1, default=float))
print(json.dumps(resumen, indent=1, default=float))
