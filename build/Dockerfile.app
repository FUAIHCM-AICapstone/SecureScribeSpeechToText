ARG RUNTIME_IMAGE=runtime:latest
FROM ${RUNTIME_IMAGE}

# Set environment variables
ENV ENV=production

# Ensure we're running as root for setup
USER root

# Copy application code
COPY app/ ./app/
COPY start.sh ./start.sh
COPY ./datasets ./datasets

# Create data directory and download model files
RUN mkdir -p ./data && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -L "https://huggingface.co/hynt/EfficientConformerVietnamese/resolve/main/6gram_lm_corpus.binary?download=true" -o ./data/6gram_lm_corpus.binary && \
    curl -L "https://drive.google.com/uc?id=11Dwa_1NV3QaHZcH6fGI9zawJjnuc_zyK&export=download" -o ./checkpoints_56_90h.ckpt && \
    apt-get remove -y curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Fix permissions and make start.sh executable
RUN chmod +x start.sh && sed -i 's/\r$//' start.sh && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

CMD ["/bin/bash", "-c", "exec ./start.sh"]
