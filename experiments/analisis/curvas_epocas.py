"""Curvas clásicas época a época: entrenamiento frente a validación, lado a lado."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SALIDA = Path("runs/evaluacion_imagen")
AZUL, NARANJA, GRIS, TINTA2, FONDO = "#2a78d6", "#eb6834", "#8a8984", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
                     "axes.edgecolor": "#c9c8c2", "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
                     "ytick.color": TINTA2, "axes.grid": True, "grid.color": "#ecebe7",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "lines.linewidth": 2.2})

h = pd.concat(pd.read_csv(f) for f in Path("runs/cv_presupuesto").glob("fold*/ponderada_aux0/semilla*/historia.csv"))
m = h.groupby("epoca")[["fit_bce_obj", "sel_bce_obj", "fit_auc", "sel_auc"]].mean()
m.round(4).to_csv(SALIDA / "epocas_media.csv")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, (cf, cs, titulo, ylab) in zip((a1, a2), (("fit_bce_obj", "sel_bce_obj", "Pérdida por época", "pérdida (BCE ponderada)"),
                                                 ("fit_auc", "sel_auc", "AUC por época", "AUC por paciente"))):
    ax.plot(m.index, m[cf], "o-", color=AZUL, ms=5, label="entrenamiento")
    ax.plot(m.index, m[cs], "o-", color=NARANJA, ms=5, label="validación")
    ax.axvspan(5 if cf == "fit_bce_obj" else 9, 12.4, color=GRIS, alpha=0.10, lw=0)
    ax.set(title=titulo, xlabel="época", ylabel=ylab, xticks=range(1, 13), xlim=(0.6, 12.4))
    ax.legend(frameon=False, loc="lower left" if cf == "fit_bce_obj" else "upper left")
a1.annotate("mínimo de validación\n(época 5)", (5, m.sel_bce_obj[5]), xytext=(5.6, 0.988), fontsize=10,
            color=TINTA2, arrowprops=dict(arrowstyle="->", color=TINTA2))
a2.axhline(0.5, color=GRIS, lw=1, ls=":")
a2.annotate("validación deja de\nmejorar (época 9)", (9, m.sel_auc[9]), xytext=(9.2, 0.515), fontsize=10,
            color=TINTA2, arrowprops=dict(arrowstyle="->", color=TINTA2))
fig.suptitle("imagen_v2 · media de 15 redes de la CV · zona gris: la validación ya no mejora (pérdida desde la 5, AUC desde la 9)", x=0.01,
             ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(SALIDA / "0_epocas_entrenamiento_validacion.png", dpi=150)
print(m.round(4).to_string())
