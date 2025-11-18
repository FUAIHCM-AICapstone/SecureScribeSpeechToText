ARG RUNTIME_IMAGE=runtime:latest
FROM ${RUNTIME_IMAGE}

# Set environment variables
ENV ENV=production

# Copy application code
COPY app/ ./app/
COPY start.sh ./start.sh
COPY 6gram_lm_corpus.binary ./data/6gram_lm_corpus.binary
COPY ./checkpoints_56_90h.ckpt ./checkpoints_56_90h.ckpt

RUN chmod +x start.sh && sed -i 's/\r$//' start.sh && chown appuser:appuser start.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

CMD ["/bin/bash", "-c", "exec ./
CMD ["/bin/bash", "-c", "exec ./start.sh"]
