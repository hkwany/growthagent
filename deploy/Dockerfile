# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-optional.txt

COPY growthagent ./growthagent
COPY examples ./examples

ENV LLM_PROVIDER=ollama \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

EXPOSE 8000
CMD ["uvicorn", "growthagent.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
