"""Carpeta de despliegue del panel: código versionado, pesos y pacientes de ejemplo de train."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .datos import COLUMNAS_FASE
from .paths import DATOS, RAIZ

PLANTILLAS = RAIZ / "despliegue"
DESTINO = PLANTILLAS / "space"
CODIGO = ("app", "src/training", "src/utils", "scripts/servir_panel.py", "requirements.txt")
PAQUETES = ("reports/modelo_elegido/*/paquete.json", "reports/modelo_elegido/imagen_v2/pesos/*.pt")
# Las únicas columnas de patients.csv que lee el panel.
COLUMNAS_PACIENTE = ["pid", "pCR", "tum_vol", "age", "HR", "HER2", "dataset", "split"]
PROHIBIDO = re.compile(r"predicciones_test|(^|/)dataset/test/|(^|/)runs/|(^|/)\.git/|(^|/)\.env")
SIN_TEXTO = {".png", ".pt"}


def archivos_versionados() -> list[Path]:
    """Rutas relativas que Git no ignora: lo excluido por .gitignore nunca llega al Space."""
    salida = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
                             "--", *CODIGO, *PAQUETES], cwd=RAIZ, check=True, capture_output=True)
    rutas = [Path(r) for r in salida.stdout.decode().split("\0") if r]
    return [r for r in rutas if (RAIZ / r).is_file()]


def filtrar_metadatos(pacientes, raiz: Path = DATOS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """samples.csv y patients.csv reducidos a las pacientes pedidas, siempre de train."""
    pedidas = set(pacientes)
    samples = pd.read_csv(raiz / "metadata/samples.csv")
    patients = pd.read_csv(raiz / "metadata/patients.csv")
    samples = samples[(samples["split"] == "train") & samples["patient_id"].isin(pedidas)]
    patients = patients.loc[(patients["split"] == "train") & patients["pid"].isin(pedidas), COLUMNAS_PACIENTE]
    if set(samples["patient_id"]) != pedidas or set(patients["pid"]) != pedidas:
        raise ValueError("Todas las pacientes de ejemplo deben existir en train")
    return samples, patients


def comprobar(destino: Path, raiz: Path = DATOS) -> None:
    """Detiene la preparación si queda algún dato individual de test o un archivo que no debe subir."""
    pacientes = pd.read_csv(raiz / "metadata/patients.csv")
    otras = sorted(pacientes.loc[pacientes["split"] != "train", "pid"], key=len, reverse=True)
    ids = re.compile(rb"(?<![\w-])(?:" + "|".join(re.escape(p) for p in otras).encode() + rb")(?![\w-])")
    errores = []
    for ruta in sorted(destino.rglob("*")):
        relativa = ruta.relative_to(destino).as_posix()
        if PROHIBIDO.search(relativa):
            errores.append(f"archivo prohibido: {relativa}")
        elif ruta.is_file() and ruta.suffix not in SIN_TEXTO and otras and ids.search(ruta.read_bytes()):
            errores.append(f"identificador de test: {relativa}")
    copia = destino / "dataset/breastdcedl"
    samples = pd.read_csv(copia / "metadata/samples.csv")
    patients = pd.read_csv(copia / "metadata/patients.csv")
    if not (samples["split"] == "train").all() or not (patients["split"] == "train").all():
        errores.append("metadatos: hay filas fuera de train")
    if not samples[list(COLUMNAS_FASE)].stack().str.startswith("dataset/train/").all():
        errores.append("metadatos: hay rutas fuera de dataset/train/")
    esperadas = set(samples[list(COLUMNAS_FASE)].stack())
    presentes = {p.relative_to(copia).as_posix() for p in (copia / "dataset").rglob("*") if p.is_file()}
    if esperadas != presentes:
        errores.append(f"imágenes: {len(presentes ^ esperadas)} no coinciden con samples.csv")
    if errores:
        raise ValueError("La carpeta del Space no es publicable:\n" + "\n".join(errores))


def preparar(pacientes, destino: Path = DESTINO, raiz: Path = DATOS) -> Path:
    """Regenera la carpeta desde cero y la comprueba antes de darla por buena."""
    samples, patients = filtrar_metadatos(pacientes, raiz)
    if destino.exists():
        shutil.rmtree(destino)
    copias = {RAIZ / r: destino / r for r in archivos_versionados()}
    copias |= {PLANTILLAS / n: destino / n for n in ("Dockerfile", "README.md")}
    copia = destino / "dataset/breastdcedl"
    copias[raiz / "LICENSE"] = copia / "LICENSE"
    copias |= {raiz / r: copia / r for r in samples[list(COLUMNAS_FASE)].stack()}
    for origen, final in copias.items():
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origen, final)
    (copia / "metadata").mkdir(parents=True, exist_ok=True)
    samples.to_csv(copia / "metadata/samples.csv", index=False)
    patients.to_csv(copia / "metadata/patients.csv", index=False)
    comprobar(destino, raiz)
    return destino
