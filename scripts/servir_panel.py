"""Inicia el panel HTML con los pesos existentes; no entrena ni evalúa test."""
import argparse
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / 'src')]

def main():
    parser = argparse.ArgumentParser(description='Panel web RedDCE · pCR')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()
    import torch
    from waitress import serve
    from app.servidor import crear_app
    torch.set_num_threads(4)
    print(f'Panel disponible en http://{args.host}:{args.port}', flush=True)
    serve(crear_app(), host=args.host, port=args.port, threads=4, max_request_body_size=48 * 1024 * 1024)

if __name__ == '__main__':
    main()
