# Imagen del Space: inferencia en CPU con los pesos congelados; no entrena ni evalúa.
FROM python:3.12-slim

RUN useradd -m -u 1000 usuario
WORKDIR /code

COPY requirements.txt requirements.txt
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r app/requirements.txt

COPY --chown=usuario . .
USER usuario
ENV MPLCONFIGDIR=/tmp/matplotlib

CMD ["python", "scripts/servir_panel.py", "--host", "0.0.0.0", "--port", "7860", "--hilos", "2"]
