FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY input_screening.py output_screening.py main.py ./

# The fine-tuned HeBERT weights (~419MB) are deliberately NOT baked into this
# image -- that's real bloat for a large binary asset that changes on its own
# retraining cadence, independent of code changes. Mount it at runtime instead:
#   docker run -v /path/to/hebert_distress_model:/app/hebert_distress_model:ro ...
# If not mounted, input_screening.py falls back to the untrained public
# "avichr/heBERT_sentiment_analysis" model (same fallback the monolith had).

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
