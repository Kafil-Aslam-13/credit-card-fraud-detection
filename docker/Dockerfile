FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements-lock.txt setup.py ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir kaggle

COPY src ./src
COPY app ./app
COPY configs ./configs
COPY start.sh ./start.sh

RUN chmod +x start.sh

EXPOSE 8501

CMD ["./start.sh"]