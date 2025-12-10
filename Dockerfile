FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

CMD ["sh", "-lc", "streamlit run app.py --server.address=0.0.0.0 --server.port ${PORT:-7860} --server.headless true --server.enableCORS false --server.enableXsrfProtection false"]






