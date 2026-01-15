# Application Dockerfile - Uses runtime image and adds app code
ARG RUNTIME_IMAGE
FROM ${RUNTIME_IMAGE}

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY start.sh ./start.sh

RUN chmod +x start.sh && sed -i 's/\r$//' start.sh && chown appuser:appuser start.sh

# Switch to non-root user (inherited from runtime)
USER appuser

# Expose port
EXPOSE 8000

CMD ["/bin/bash", "-c", "exec ./start.sh"]