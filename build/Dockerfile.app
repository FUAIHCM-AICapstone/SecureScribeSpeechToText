# Application Dockerfile - Uses runtime image and adds app code
ARG RUNTIME_IMAGE=python:3.11-slim-bullseye
FROM ${RUNTIME_IMAGE}

# Switch to root to copy files and set permissions
USER root

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY start.sh ./start.sh

# Fix line endings and set permissions
RUN dos2unix start.sh || sed -i 's/\r$//' start.sh && \
    chmod +x start.sh && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

CMD ["/bin/bash", "-c", "exec ./start.sh"]