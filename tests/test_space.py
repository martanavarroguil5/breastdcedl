"""La carpeta del Space solo puede llevar pacientes de ejemplo de train y ningún dato individual de test."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / "src")]
from utils import space  # noqa: E402

FASES = ("pre", "early", "late")


class TestSpace(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.raiz = self.tmp / "origen"
        (self.raiz / "metadata").mkdir(parents=True)
        filas = [("T1", "train"), ("T2", "train"), ("X1", "test")]
        pd.DataFrame([{"sample_id": f"{p}_z", "patient_id": p, "split": s, "slice_index": 1, "pCR": 0, "fold": 0,
                       **{f"path_{f}": f"dataset/{s}/{p}_{f}.png" for f in FASES}} for p, s in filas]
                     ).to_csv(self.raiz / "metadata/samples.csv", index=False)
        pd.DataFrame([{"pid": p, "split": s, "pCR": 0, "tum_vol": 1.5, "age": 50, "HR": 1, "HER2": 0, "dataset": "spy1",
                       "race_white": 1} for p, s in filas]).to_csv(self.raiz / "metadata/patients.csv", index=False)
        self.destino = self.tmp / "space"
        copia = self.destino / "dataset/breastdcedl"
        (copia / "metadata").mkdir(parents=True)
        (copia / "dataset/train").mkdir(parents=True)
        samples, patients = space.filtrar_metadatos(["T1"], self.raiz)
        samples.to_csv(copia / "metadata/samples.csv", index=False)
        patients.to_csv(copia / "metadata/patients.csv", index=False)
        for f in FASES:
            (copia / f"dataset/train/T1_{f}.png").write_bytes(b"png")
        self.copia = copia

    def test_filtra_pacientes_y_columnas(self):
        samples, patients = space.filtrar_metadatos(["T1"], self.raiz)
        self.assertEqual(list(samples["patient_id"]), ["T1"])
        self.assertEqual(list(patients.columns), space.COLUMNAS_PACIENTE)

    def test_no_admite_pacientes_de_test(self):
        with self.assertRaises(ValueError):
            space.filtrar_metadatos(["T1", "X1"], self.raiz)

    def test_carpeta_limpia_pasa(self):
        space.comprobar(self.destino, self.raiz)

    def assertBloquea(self, texto):
        with self.assertRaisesRegex(ValueError, texto):
            space.comprobar(self.destino, self.raiz)

    def test_bloquea_predicciones_de_test(self):
        ruta = self.destino / "reports/fusion/predicciones_test.csv"
        ruta.parent.mkdir(parents=True)
        ruta.write_text("p\n")
        self.assertBloquea("predicciones_test")

    def test_bloquea_imagenes_de_test(self):
        ruta = self.copia / "dataset/test/X1_pre.png"
        ruta.parent.mkdir()
        ruta.write_bytes(b"png")
        self.assertBloquea("dataset/test")

    def test_bloquea_imagen_que_no_esta_en_samples(self):
        (self.copia / "dataset/train/T2_pre.png").write_bytes(b"png")
        self.assertBloquea("no coinciden")

    def test_bloquea_identificador_de_test_en_un_texto(self):
        (self.destino / "notas.json").write_text('{"paciente": "X1"}')
        self.assertBloquea("identificador de test: notas.json")

    def test_no_confunde_identificadores_parecidos(self):
        (self.destino / "notas.json").write_text('{"paciente": "X10", "otra": "AX1"}')
        space.comprobar(self.destino, self.raiz)

    def test_bloquea_filas_de_test_en_metadatos(self):
        ruta = self.copia / "metadata/samples.csv"
        samples = pd.read_csv(ruta)
        samples.loc[0, "split"] = "test"
        samples.to_csv(ruta, index=False)
        self.assertBloquea("fuera de train")


if __name__ == "__main__":
    unittest.main()
