#!/bin/bash
# =============================================================================
#  Remove the grape-gripper systemd service.
#
#      sudo bash deploy/uninstall_service.sh
#
#  ASCII only on purpose (locale safety).
# =============================================================================
set -eu

SERVICE_NAME="grape-gripper"
UNIT="/etc/systemd/system/${SERVICE_NAME}.service"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: run with sudo"
    exit 1
fi

if systemctl list-unit-files | grep -q "^${SERVICE_NAME}.service"; then
    echo "Stopping and disabling ${SERVICE_NAME} ..."
    systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    systemctl disable "$SERVICE_NAME" 2>/dev/null || true
fi

if [ -f "$UNIT" ]; then
    rm -f "$UNIT"
    echo "Removed $UNIT"
else
    echo "Unit file not found: $UNIT"
fi

systemctl daemon-reload
echo
echo "Done. The service will no longer start on boot."
echo "You can still run it manually:"
echo "  cd ~/grape-gripper && sudo python3 apps/product_mode.py"