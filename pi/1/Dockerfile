FROM python:3.11-slim

# Unbuffered Python output for Docker logs
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
# RUN adduser --disabled-password mesh
# USER mesh
RUN mkdir -p /home/mesh/bbs
WORKDIR /home/mesh/bbs

# Install Python dependencies
COPY --chown=mesh:mesh requirements.txt ./
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy local app code
COPY --chown=mesh:mesh *.py ./

# Define config volume
VOLUME /home/mesh/bbs/config
WORKDIR /home/mesh/bbs/config

ENTRYPOINT [ "python3", "/home/mesh/bbs/server.py" ]