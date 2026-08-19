# ============================================================
#  Backend image for Railway (or any container host)
# ============================================================
FROM python:3.11-slim

# Deliberately single-stage. A builder stage is worth it when the final image
# would otherwise carry compilers and caches; here the bulk is PyTorch and two
# pre-baked models, which have to be in the runtime image regardless, and the
# wheels need no toolchain. The complexity would buy almost nothing.

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    TRANSFORMERS_OFFLINE=0

WORKDIR /app

# 1) CPU-only PyTorch first (the default index would pull the multi-GB CUDA
#    build), then the rest of the requirements.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install -r requirements.txt

# 2) Application code.
COPY . .

# 3) Pre-download the embedding and reranker models so the first request does
#    not pay the download, and so the container works on an offline or
#    read-only host.
RUN python -c "from sentence_transformers import SentenceTransformer; \
import os; SentenceTransformer(os.getenv('EMBEDDING_MODEL', \
'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))"
RUN python -c "from sentence_transformers import CrossEncoder; \
import os; CrossEncoder(os.getenv('RERANK_MODEL', \
'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'))"

# 4) Drop root. A container escape should not start from uid 0, and nothing
#    here needs write access outside the model cache. Done after the installs
#    so the application directory stays owned by root and read-only to the app.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app/.hf_cache
USER appuser

# Railway and most PaaS inject $PORT; 8000 is the local default.
ENV PORT=8000
EXPOSE 8000

# Readiness, not liveness: a container that cannot reach its database should be
# reported unhealthy rather than left in rotation. `start-period` covers the
# first boot, where migrations run and the models load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/api/ready', timeout=4).status == 200 else 1)"

# Shell form so $PORT is expanded at runtime.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
