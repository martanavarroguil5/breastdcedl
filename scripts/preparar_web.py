"""Genera despliegue/web, la versión estática del panel; no entrena, no evalúa test y no publica nada."""
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / 'src')]

def main():
    import torch
    torch.set_num_threads(4)
    from app.web_estatica import preparar
    resultado = preparar()
    destino = RAIZ / 'despliegue/web'
    archivos = [p for p in destino.rglob('*') if p.is_file()]
    print(f'{destino}: {len(archivos)} archivos, {sum(p.stat().st_size for p in archivos) / 1e6:.1f} MB')
    print(f"Composición idéntica a Flask en {resultado['estados']} estados; ONNX {resultado['huellas']['reddce.onnx'][:12]}")

if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error))
