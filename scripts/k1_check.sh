#!/bin/bash
# ============================================================================
# K1 MusePi Pro 硬件资源诊断脚本
# 用途：一次收齐 PWM / GPIO / I2C / 摄像头 的全部信息，用于确定接线方案
# 用法：bash k1_check.sh
# 说明：只读操作，不改任何配置。部分命令需要 i2c-tools，没装会自动跳过。
# ============================================================================

echo ""
echo "========================================================================"
echo " 1. 系统信息"
echo "========================================================================"
uname -a
echo "---"
cat /etc/os-release 2>/dev/null | head -4
echo "---"
echo -n "CPU 型号: "; grep -m1 "model name\|uarch\|isa" /proc/cpuinfo 2>/dev/null || true
echo -n "核心数:   "; nproc
echo -n "内存:     "; free -h 2>/dev/null | awk '/^Mem:/{print $2}'
echo -n "Python:   "; python3 -V 2>&1

echo ""
echo "========================================================================"
echo " 2. PWM 资源 ★ 决定舵机怎么接"
echo "========================================================================"
for chip in /sys/class/pwm/pwmchip*; do
    [ -d "$chip" ] || continue
    name=$(basename "$chip")
    npwm=$(cat "$chip/npwm" 2>/dev/null || echo "读不到")
    echo ""
    echo "== $name : npwm=$npwm =="
    if [ -d "$chip/device" ]; then
        echo -n "   硬件地址: "
        readlink -f "$chip/device" 2>/dev/null | sed 's|.*/||' || echo "?"
    fi
    ls -d "$chip"/pwm* 2>/dev/null | while read -r p; do
        echo "   已导出: $(basename "$p")  (period=$(cat "$p/period" 2>/dev/null) duty=$(cat "$p/duty_cycle" 2>/dev/null) enable=$(cat "$p/enable" 2>/dev/null))"
    done
    echo -n "   当前是否有通道已导出: "
    ls -d "$chip"/pwm* >/dev/null 2>&1 && echo "是" || echo "否（需要 echo N > export）"
done

echo ""
echo "--- 当前 PWM 引脚复用状态（如果内核暴露了 debugfs）---"
if [ -r /sys/kernel/debug/pinctrl ]; then
    for d in /sys/kernel/debug/pinctrl/*/; do
        [ -d "$d" ] || continue
        echo "  pinctrl: $(basename "$d")"
    done
else
    echo "  /sys/kernel/debug/pinctrl 不可读（需要 sudo 或未挂载 debugfs）"
fi
echo "--- 设备树里的 PWM 节点 ---"
ls /proc/device-tree/ 2>/dev/null | grep -i pwm || echo "  读不到 /proc/device-tree"

echo ""
echo "========================================================================"
echo " 3. GPIO 资源"
echo "========================================================================"
if [ -d /sys/class/gpio ]; then
    echo "老式 /sys/class/gpio 存在，已导出的引脚："
    ls /sys/class/gpio/ 2>/dev/null | grep -v "^export\|^unexport" | sed 's/^/  /' || echo "  （无）"
else
    echo "无 /sys/class/gpio 目录（内核可能只用 libgpiod 接口）"
fi
echo "--- gpiochip 列表 ---"
for c in /sys/class/gpio/gpiochip*; do
    [ -d "$c" ] || continue
    echo "  $(basename "$c"): base=$(cat "$c/base" 2>/dev/null) ngpio=$(cat "$c/ngpio" 2>/dev/null) label=$(cat "$c/label" 2>/dev/null)"
done
[ -d /sys/class/gpio/gpiochip0 ] || echo "  （没有 gpiochipN 目录，需要用 libgpiod）"
command -v gpiodetect >/dev/null 2>&1 && gpiodetect || echo "  gpiodetect 未安装 → 运行: sudo apt install -y gpiod"

echo ""
echo "========================================================================"
echo " 4. I2C 总线 ★ 决定压力传感器 ADC 怎么接"
echo "========================================================================"
command -v i2cdetect >/dev/null 2>&1 || {
    echo "i2cdetect 未安装。请先运行： sudo apt install -y i2c-tools"
}
for bus in /dev/i2c-*; do
    [ -e "$bus" ] || continue
    n=$(basename "$bus" | sed 's/i2c-//')
    echo ""
    echo "== $bus =="
    if command -v i2cdetect >/dev/null 2>&1; then
        i2cdetect -y -r "$n" 2>/dev/null | sed 's/^/   /' || echo "   （读取失败，可能被占用或需要权限）"
    fi
done
echo ""
echo "--- 提示 ---"
echo "OV5647 摄像头的 I2C 地址是 0x36，在上面某条总线上应该能看到。"
echo "ADS1115 压力传感器 ADC 默认地址 0x48（ADDR 接地），接上后也应出现在这里。"

echo ""
echo "========================================================================"
echo " 5. SPI 资源"
echo "========================================================================"
ls /dev/spidev* 2>/dev/null || echo "无 /dev/spidev* → SPI 未启用（用 MCP3008 的话需要先开 SPI；ADS1115 走 I2C 不受影响）"

echo ""
echo "========================================================================"
echo " 6. 摄像头 ★ 确认 OV5647 有没有被识别"
echo "========================================================================"
echo "--- /dev/video* 列表 ---"
ls /dev/video* 2>/dev/null | tr '\n' ' '; echo ""
echo ""
echo "--- 哪些 video 节点是真实采集设备（能列出格式的才是）---"
command -v v4l2-ctl >/dev/null 2>&1 || echo "v4l2-ctl 未安装 → sudo apt install -y v4l-utils"
for dev in /dev/video*; do
    [ -e "$dev" ] || continue
    # 只探测可能真实的采集节点，跳过编解码器
    case "$dev" in
        /dev/video-dec*|/dev/video-enc*) continue ;;
    esac
    caps=$(v4l2-ctl -d "$dev" --all 2>/dev/null | grep -m1 "Device Caps\|Driver name" || true)
    if [ -n "$caps" ]; then
        echo "  $dev: $caps"
    fi
done

echo ""
echo "--- 摄像头 sensor 是否注册 ---"
if [ -d /sys/bus/i2c/drivers ]; then
    found=0
    for drv in /sys/bus/i2c/drivers/*ov5647* /sys/bus/i2c/drivers/*ovti*; do
        [ -e "$drv" ] || continue
        found=1
        echo "  发现驱动: $(basename "$drv")"
        ls "$drv" 2>/dev/null | grep -E "^[0-9]+-" | sed 's/^/    设备: /' || true
    done
    [ "$found" = "0" ] && echo "  没有找到 ov5647 驱动目录"
fi
echo "--- media 设备拓扑（能看到 sensor 名字就说明识别了）---"
command -v media-ctl >/dev/null 2>&1 && media-ctl -p -d /dev/media0 2>/dev/null | head -40 || echo "  media-ctl 未安装（在 v4l-utils 包里）"

echo ""
echo "========================================================================"
echo " 7. 虚拟摄像头（vivi）—— 无需硬件即可测试视觉代码"
echo "========================================================================"
for dev in /dev/video50 /dev/video51; do
    [ -e "$dev" ] || continue
    echo "== $dev =="
    v4l2-ctl -d "$dev" --list-formats-ext 2>/dev/null | head -20 | sed 's/^/  /'
done

echo ""
echo "========================================================================"
echo " 8. 关键结论速查（把这一整段输出发给我）"
echo "========================================================================"
echo "PWM chip 数: $(ls -d /sys/class/pwm/pwmchip* 2>/dev/null | wc -l)"
for chip in /sys/class/pwm/pwmchip*; do
    [ -d "$chip" ] || continue
    echo "  $(basename "$chip"): $(cat "$chip/npwm" 2>/dev/null) 路"
done
echo "I2C 总线数: $(ls /dev/i2c-* 2>/dev/null | wc -l)"
echo "SPI:        $(ls /dev/spidev* 2>/dev/null | wc -l) 个节点"
echo "video 节点: $(ls /dev/video* 2>/dev/null | wc -l) 个"
echo ""
echo "====== 诊断完成，请把以上全部输出复制给我 ======"
