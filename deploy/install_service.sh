#!/bin/bash
# =============================================================================
#  Install "grape gripper" product mode as a systemd service.
#
#  After install:
#      - the board powers on -> the service starts automatically
#      - no terminal, no keyboard, no display needed
#      - say  "jia pu tao"  (grasp the grape) to start working
#
#  Commands:
#      sudo bash deploy/install_service.sh        # install + enable + start
#      systemctl status grape-gripper             # check
#      journalctl -u grape-gripper -f             # live log
#      sudo systemctl stop grape-gripper          # stop
#      sudo bash deploy/uninstall_service.sh      # remove
#
#  ASCII only on purpose (locale safety).
# =============================================================================
set -eu

SERVICE_NAME="grape-gripper"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
UNIT="/etc/systemd/system/${SERVICE_NAME}.service"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: run with sudo"
    exit 1
fi

if [ ! -f "$ROOT/apps/product_mode.py" ]; then
    echo "ERROR: apps/product_mode.py not found under $ROOT"
    exit 1
fi

# Resolve the user whose home holds the ASR model cache.
# If you normally run as a normal user (e.g. daiweikai) the model lives in
# that user's ~/.cache; running as root would miss it. So we set HOME
# explicitly in the unit instead of relying on root's home.
RUN_USER="${SUDO_USER:-root}"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
if [ -z "$RUN_HOME" ]; then RUN_HOME="/root"; fi

echo "Installing ${SERVICE_NAME}.service"
echo "  project : $ROOT"
echo "  user    : $RUN_USER"
echo "  HOME    : $RUN_HOME"
echo "  unit    : $UNIT"
echo

cat > "$UNIT" <<EOF
[Unit]
Description=Grape Gripper - voice triggered robotic grasping
After=sound.target
Wants=sound.target

[Service]
Type=simple
User=root
WorkingDirectory=$ROOT
Environment=HOME=$RUN_HOME
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONIOENCODING=utf-8
ExecStart=/usr/bin/python3 $ROOT/apps/product_mode.py
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

sleep 2
echo
echo "=============================================================="
systemctl --no-pager --full status "$SERVICE_NAME" || true
echo "=============================================================="
echo
echo "Done. Useful commands:"
echo "  journalctl -u $SERVICE_NAME -f      # live log"
echo "  sudo systemctl stop $SERVICE_NAME"
echo "  sudo systemctl disable $SERVICE_NAME"