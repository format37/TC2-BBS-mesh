FROM python:3.11-slim

# Unbuffered Python output for Docker logs
ENV PYTHONUNBUFFERED=1

# Install timezone data (required for TZ environment variable)
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*

# Setup working directory (running as root for serial port access)
RUN mkdir -p /home/mesh/bbs
WORKDIR /home/mesh/bbs

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy local app code
COPY *.py ./

# Define config volume
VOLUME /home/mesh/bbs/config
WORKDIR /home/mesh/bbs/config

ENTRYPOINT [ "python3", "/home/mesh/bbs/server.py" ]
