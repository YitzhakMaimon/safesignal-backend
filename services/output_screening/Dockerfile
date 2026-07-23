FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./

EXPOSE 8005

# --workers 1: a second worker process would double memory (each worker loads
# its own aioboto3/anthropic clients) on a 1GB t3.micro. --limit-concurrency
# caps in-flight requests so a burst can't queue unbounded work and OOM the
# single worker.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005", "--workers", "1", "--limit-concurrency", "50"]
