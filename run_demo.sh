#!/bin/bash
# =============================================================================
#  Grape Gripper - single entry point
#
#  Run it directly on the board (no PC / no SSH needed):
#
#      cd ~/grape-gripper && ./run_demo.sh
#
#  sudo is required because the servo uses GPIO and both force sensors
#  use serial ports. This project has passwordless sudo configured, so the
#  script escalates automatically without prompting.
#
#  NOTE: this file is intentionally ASCII-only. Non-ASCII comments in a
#  shell script break on systems whose locale is not UTF-8.
#
#  Auto-start on boot: see the notes at the bottom of this file.
# =============================================================================
set -u

cd "$(dirname "$0")" || exit 1

echo "=============================================================="
echo " Grape Gripper - starting"
echo "=============================================================="
echo " workdir: $(pwd)"
echo

# ---- sanity: are we in the right place? ----
if [ ! -f "apps/demo.py" ]; then
    echo "ERROR: apps/demo.py not found - wrong directory?"
    exit 1
fi

if [ ! -f "config.yaml" ]; then
    echo "ERROR: config.yaml not found"
    exit 1
fi

# ---- kill leftovers: they hold the serial ports / microphone ----
for pat in voice_grasp.py grasp_with_verify.py vision_grasp.py arecord; do
    if pgrep -f "$pat" > /dev/null 2>&1; then
        echo "WARN: killing leftover process ($pat) ..."
        pkill -f "$pat" 2>/dev/null
        sleep 0.5
    fi
done

# ---- escalate and launch ----
if [ "$(id -u)" -eq 0 ]; then
    exec python3 apps/demo.py "$@"
else
    # -E keeps environment (the ASR needs a correct HOME)
    # -n non-interactive: passwordless sudo passes; otherwise we bail out
    if sudo -n true 2>/dev/null; then
        exec sudo -E python3 apps/demo.py "$@"
    else
        echo "WARN: passwordless sudo is not available. Please run:"
        echo "        sudo python3 apps/demo.py"
        echo
        echo "      or (voice only, no root needed):"
        echo "        python3 apps/demo.py --run 4"
        exit 1
    fi
fi

# =============================================================================
#  Auto-start on boot (optional)
# =============================================================================
#
#  To pop up the demo menu right after the board boots:
#
#      mkdir -p ~/.config/autostart
#      cat > ~/.config/autostart/grape-demo.desktop <<'EOF'
#      [Desktop Entry]
#      Type=Application
#      Name=Grape Gripper
#      Exec=x-terminal-emulator -e "bash -c 'cd ~/grape-gripper && ./run_demo.sh; exec bash'"
#      Terminal=false
#      X-GNOME-Autostart-enabled=true
#      EOF
#
#  To disable it again:
#      rm ~/.config/autostart/grape-demo.desktop
#
#  Desktop shortcut only (no autostart):
#      copy the same .desktop file to ~/Desktop/ and chmod +x it
# =============================================================================