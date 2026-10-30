"""Genera despliegue/space para Hugging Face; no entrena, no evalúa test y no publica nada."""
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / 'src')]

def main():
    from app import motor
    from utils.space import preparar
    destino = preparar(motor.ejemplos().index)
    archivos = [p for p in destino.rglob('*') if p.is_file()]
    print(f'{destino}: {len(archivos)} archivos, {sum(p.stat().st_size for p in archivos) / 1e6:.1f} MB')

if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error))
