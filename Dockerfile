FROM python:3.12-slim

LABEL org.opencontainers.image.title="Federated RAG"
LABEL org.opencontainers.image.description="Biomedical federated RAG system with dual-corpus air-gap deployment"
LABEL org.opencontainers.image.version="0.5.0"

# System dependencies for scispaCy, docling, sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY papers/ ./papers/
COPY projects/ ./projects/
COPY config/ ./config/
COPY phase4_demo.py .
COPY phase4_viz.py .
COPY phase4_benchmark.py .
COPY phase4_benchmark_batch2.py .
COPY phase3_demo.py .

# Create directories for runtime data
RUN mkdir -p /app/logs /app/projects/default/extractions \
    /app/projects/default/embeddings /app/projects/default/query_cache \
    /app/projects/default/chroma_data /app/projects/default/cache

ENV PYTHONUNBUFFERED=1
ENV PROJECT_DIR=/app/projects/default
ENV SECURITY_AUDIT_LOG=/app/logs/security_audit.log

# Default to DeepSeek API; override with LLM_PROVIDER=ollama in compose
ENV LLM_PROVIDER=deepseek

EXPOSE 8501

# The entrypoint is the demo script; override for different modes
CMD ["python", "phase4_demo.py"]
