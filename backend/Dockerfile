# ============================================================
#  Backend image for Railway (or any container host)
# ============================================================
FROM python:3.11-slim

# System deps kept minimal; slim image already has what asyncpg needs.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    TRANSFORMERS_OFFLINE=0

WORKDIR /app

# 1) Install CPU-only PyTorch first (avoids pulling the multi-GB CUDA build),
#    then the rest of the requirements.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install -r requirements.txt

# 2) Copy the application code.
COPY . .

# 3) Pre-download the embedding + reranker models into the image so the first
#    request doesn't pay the download cost (and works on read-only/offline hosts).
RUN python -c "from sentence_transformers import SentenceTransformer; \
import os; SentenceTransformer(os.getenv('EMBEDDING_MODEL', \
'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))"
RUN python -c "from sentence_transformers import CrossEncoder; \
import os; CrossEncoder(os.getenv('RERANK_MODEL', \
'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'))"

# Railway/most PaaS inject $PORT. Default to 8000 locally.
ENV PORT=8000
EXPOSE 8000

# Shell form so $PORT is expanded at runtime.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
