#!/bin/bash
# TC2-BBS Meshtastic Connection Checker
# This script shows available serial ports and helps configure the BBS

echo "=== Meshtastic Connection Checker ==="
echo ""

# Find serial devices
echo "Available serial ports:"
PORTS=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null)

if [ -z "$PORTS" ]; then
    echo "  [!] No serial devices found!"
    echo "  Make sure your Meshtastic device is connected via USB."
    exit 1
fi

for port in $PORTS; do
    echo "  - $port"
done

echo ""

# Check current config
CONFIG_FILE="data/config.ini"
if [ -f "$CONFIG_FILE" ]; then
    CURRENT_PORT=$(grep "^port" "$CONFIG_FILE" | cut -d'=' -f2 | tr -d ' ')
    echo "Current configured port: $CURRENT_PORT"

    if [ -e "$CURRENT_PORT" ]; then
        echo "  [OK] Port exists"
    else
        echo "  [!] Port does NOT exist - update config!"
    fi
else
    echo "[!] Config file not found: $CONFIG_FILE"
fi

echo ""

# Show docker-compose device mapping
COMPOSE_FILE="docker-compose.yaml"
if [ -f "$COMPOSE_FILE" ]; then
    MAPPED_PORT=$(grep "/dev/tty" "$COMPOSE_FILE" | head -1 | sed 's/.*- //' | cut -d':' -f1)
    echo "Docker device mapping: $MAPPED_PORT"

    if [ -e "$MAPPED_PORT" ]; then
        echo "  [OK] Mapped device exists"
    else
        echo "  [!] Mapped device does NOT exist - update docker-compose.yaml!"
    fi
fi

echo ""

# Suggest fix if needed
FIRST_PORT=$(echo "$PORTS" | head -1)
if [ -n "$CURRENT_PORT" ] && [ "$CURRENT_PORT" != "$FIRST_PORT" ]; then
    echo "=== Suggested fix ==="
    echo "Update port to: $FIRST_PORT"
    echo ""
    echo "Run these commands:"
    echo "  sed -i 's|port = .*|port = $FIRST_PORT|' data/config.ini"
    echo "  sed -i 's|/dev/tty[A-Z]*[0-9]*:/dev/tty[A-Z]*[0-9]*|$FIRST_PORT:$FIRST_PORT|g' docker-compose.yaml"
    echo "  docker compose down && docker compose up -d"
fi

echo ""
echo "=== Container status ==="
docker ps --filter "name=tc2-bbs-mesh" --format "{{.Names}}: {{.Status}}" 2>/dev/null || echo "Docker not running or container not found"
