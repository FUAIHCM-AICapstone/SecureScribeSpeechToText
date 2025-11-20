FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV ENV=development

# Set working directory
WORKDIR /app

# Create non-root user
RUN addgroup --system appuser && \
    adduser --system --ingroup appuser appuser

# Install build deps including cmake for KenLM and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ make cmake git \
    libboost-program-options-dev libboost-system-dev libboost-thread-dev libboost-test-dev \
    libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Build and install KenLM from source
RUN git clone https://github.com/kpu/kenlm.git /tmp/kenlm \
    && cd /tmp/kenlm \
    && mkdir -p build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DBUILD_TESTING=OFF \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd / \
    && rm -rf /tmp/kenlm

# Install uv + deps (PyTorch CPU-only first, then others)
COPY requirements.txt .
RUN pip install --no-cache-dir uv \
    && uv pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu --system --no-cache \
    && uv pip install -r requirements.txt --system --no-cache
# Copy application code
COPY app/ ./app/
COPY start.sh ./start.sh
COPY ./datasets ./datasets
COPY 6gram_lm_corpus.binary ./data/6gram_lm_corpus.binary
COPY ./checkpoints_56_90h.ckpt ./checkpoints_56_90h.ckpt

RUN chmod +x start.sh && sed -i 's/\r$//' start.sh && chown appuser:appuser start.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

CMD ["/bin/bash", "-c", "exec ./start.sh"]
