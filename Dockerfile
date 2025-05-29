# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install Rust and required build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . $HOME/.cargo/env

# Copy only requirements files first to leverage Docker cache
COPY requirements.txt .
COPY lightrag/api/requirements.txt ./lightrag/api/

# Install dependencies
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --user --no-cache-dir -r requirements.txt
RUN pip install --user --no-cache-dir -r lightrag/api/requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY ./lightrag ./lightrag
COPY setup.py .
# COPY ./monitoring ./monitoring
# COPY ./monitoring/query_api.py ./monitoring/query_api.py

COPY example_pg_test.py ./example_pg_test.py
COPY gemini_test.py ./gemini_test.py
COPY insert_data.py ./insert_data.py

RUN pip install .
# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create necessary directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs

# Docker data directories
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs

# Expose the default port
EXPOSE 9621 8888 2727 9090 3000

CMD ["uvicorn", "query_api:app", "--host", "0.0.0.0", "--port", "2727"]