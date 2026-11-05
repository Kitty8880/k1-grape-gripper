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
# 过滤 cam-test 的 syslog 噪声
#   cam-test 是厂商二进制，每次启动都往 syslog 打十几行
#   （TcpListener / CameraTuning / Socket 等），
#   不是 stdout，所以 K1Camera 里的 capture_output=True 拦不住。
#   NPU 监控每 5 秒抓一次帧 → journal 被淹没、真正日志看不清。
#
#   ⚠️ LogFilterPatterns 的正则是**整串匹配**（隐式锚定 ^...$），
#      必须写 .*关键字.* ；第一版漏了 .* ，实测完全无效。
#   需要 systemd 253+（本机 255 ✅）
LogFilterPatterns=!~".*(CameraTuning|TcpListener|Socket: |VioHal).*"
# 退出码 6（SIGABRT）算正常
#   停服务时 NPU 推理线程在 SpaceMIT ExecutionProvider 里，
#   进程拆解阶段 EP 断言失败 → 退出码 6/ABRT。
#   应用侧已尝试"先暂停再退出"，但 EP 的解构顺序不受控。
#   这只是退出路径的问题，不影响运行和断电重启；
#   声明 6 为成功，避免 systemctl stop 后状态变成 failed。
SuccessExitStatus=6
# 给应用 8 秒自己退（它内部 2 秒后强退），超时才 SIGKILL
TimeoutStopSec=8

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