ARG RUNTIME_IMAGE=runtime:latest
FROM ${RUNTIME_IMAGE}

# Set environment variables
ENV ENV=production

# Copy application code
COPY app/ ./app/
COPY start.sh ./start.sh

# Download model files
RUN mkdir -p ./data && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -L "https://huggingface.co/hynt/EfficientConformerVietnamese/resolve/main/6gram_lm_corpus.binary?download=true" -o ./data/6gram_lm_corpus.binary && \
    curl -L "https://drive.google.com/uc?id=11Dwa_1NV3QaHZcH6fGI9zawJjnuc_zyK&export=download" -o ./checkpoints_56_90h.ckpt && \
    apt-get remove -y curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN chmod +x start.sh && sed -i 's/\r$//' start.sh && chown appuser:appuser start.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

CMD ["/bin/bash", "-c", "exec ./start.sh"]
