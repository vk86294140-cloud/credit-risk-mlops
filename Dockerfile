FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md .
COPY src ./src
COPY pipelines ./pipelines
COPY static ./static
RUN pip install --no-cache-dir -e .

# Train a model at build time so the image ships ready to serve.
RUN python -m credit_risk.train

EXPOSE 8000
CMD ["uvicorn", "credit_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
