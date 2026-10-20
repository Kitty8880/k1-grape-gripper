# -*- coding: utf-8 -*-
"""
压力传感器采集：ADC 抽象 + 滤波 + 标定。

设计要点
--------
1. **ADC 抽象**：ADS1115(I2C) / MCP3008(SPI) / K1 片内 SARADC / 模拟，
   上层代码只看到 ``read_force_n()``，换硬件不动控制逻辑。
2. **零依赖**：I2C/SPI 直接操作 /dev 节点（ioctl），不依赖 Adafruit/smbus。
   RISC-V 平台上这些第三方库经常有可用性问题，直接走内核接口最稳。
3. **滤波在驱动层做**：控制环拿到的是已经平滑过的力，避免把 ADC 噪声
   放大成舵机抖动（舵机抖动 → 力波动 → 更抖，正反馈发散）。
"""

from __future__ import annotations

import math
import os
import struct
import time
from collections import deque
from typing import Optional

# ---------------------------------------------------------------------------
# 可选的内核接口依赖。缺失时只影响对应的驱动，模拟模式永远可用。
# ---------------------------------------------------------------------------

try:
    import fcntl
except ImportError:  # pragma: no cover - 非 Linux
    fcntl = None  # type: ignore

# Linux i2c-dev.h 中的常量
I2C_SLAVE = 0x0703


class SensorError(Exception):
    """传感器读写失败。"""


# ---------------------------------------------------------------------------
# 通用滤波
# ---------------------------------------------------------------------------

class MovingAverage:
    """滑动平均。窗口内样本不足时按实际数量平均（不做零填充）。"""

    def __init__(self, window: int):
        if window < 1:
            raise ValueError("window 必须 >= 1")
        self.window = window
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._buf.append(value)
        return sum(self._buf) / len(self._buf)

    @property
    def value(self) -> Optional[float]:
        if not self._buf:
            return None
        return sum(self._buf) / len(self._buf)

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.window

    def reset(self) -> None:
        self._buf.clear()


class MedianFilter:
    """中值滤波：干掉单个采样点上的尖峰（电磁干扰、接触瞬间的毛刺）。"""

    def __init__(self, window: int):
        if window < 1:
            raise ValueError("window 必须 >= 1")
        # 保持奇数窗口，中值才有明确定义
        self.window = window if window % 2 == 1 else window + 1
        self._buf: deque[float] = deque(maxlen=self.window)

    def update(self, value: float) -> float:
        self._buf.append(value)
        ordered = sorted(self._buf)
        return ordered[len(ordered) // 2]

    def reset(self) -> None:
        self._buf.clear()


# ---------------------------------------------------------------------------
# ADC 驱动
# ---------------------------------------------------------------------------

class AdcDriver:
    """ADC 抽象基类。"""

    name = "base"
    #: 该驱动的原始读数上限（用于判断是否饱和）
    raw_max = 32767.0

    def read_raw(self, channel: int) -> float:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return self.name


#: 模拟夹爪可以夹的各种物体的力学参数预设。
#: contact_at 是"碰到物体时，夹爪走了多少行程"（0=全开, 1=全闭），
#: 由物体直径 / 夹爪最大开口决定。
#: stiffness 是超过接触点后，每多走 1.0 单位行程产生多少牛顿。
OBJECT_PRESETS = {
    # 巨峰葡萄：直径约 22mm，夹爪开口 40mm → 22/40 ≈ 0.45
    # 破裂力 1~3N；实测"轻碰住"约 0.3N，0.8N 已经有明显压痕
    "grape": {"contact_at": 0.45, "stiffness": 12.0, "break_force_n": 2.0},
    # 海绵：直径 40mm，很软
    "sponge": {"contact_at": 0.05, "stiffness": 2.5, "break_force_n": None},
    # 小番茄：直径约 30mm，比葡萄硬
    "tomato": {"contact_at": 0.25, "stiffness": 18.0, "break_force_n": 6.0},
    # 空夹（没有物体）
    "none": {"contact_at": 1.01, "stiffness": 0.0, "break_force_n": None},
}


class SimulatedAdc(AdcDriver):
    """
    模拟 ADC：按"夹爪行程 → 接触 → 力上升"的物理模型产生读数。

    建模方式（很重要，决定了模拟调参有没有意义）：

        闭合行程 s ∈ [0, 1]，0 = 完全张开，1 = 完全闭合到机械极限。
        物体在 s = contact_at 处被碰到（= 物体直径 / 夹爪最大开口）。
        s < contact_at   → 力 = 0
        s > contact_at   → 力 = stiffness * (s - contact_at) * 非线性修正

    这样"力的上升斜率"就对应了真实情况：硬的物体（stiffness 大）力涨得快，
    软物体涨得慢。用这个模型调出来的 ramp_rate / kp / ki 搬到真机上才靠谱。

    ★ 注意：闭合行程 s 必须由**夹爪机械行程**（angle_open~angle_close）算出，
      不能用舵机电气行程（0~180°）。否则接触点会整体偏移，模拟结果失效。
      这一点由 SimulatedServo 负责。
    """

    name = "simulated"

    #: 模拟的力 → ADC 原始码换算系数（假设 10N 对应约 26000 码）
    RAW_PER_NEWTON = 2600.0

    def __init__(self, noise: float = 40.0, seed: int = 12345,
                 preset: str = "grape",
                 contact_at: Optional[float] = None,
                 stiffness: Optional[float] = None,
                 asymmetry: float = 0.08):
        import random

        self._rng = random.Random(seed)
        self._noise = noise
        self._closure = 0.0

        params = OBJECT_PRESETS.get(preset, OBJECT_PRESETS["grape"])
        self.preset = preset if preset in OBJECT_PRESETS else "grape"
        self._contact_at = float(params["contact_at"] if contact_at is None else contact_at)
        self._stiffness = float(params["stiffness"] if stiffness is None else stiffness)
        self.break_force_n = params.get("break_force_n")
        # 两个传感器的不对称程度（装歪 / 葡萄没对准中线的仿真）
        self._asymmetry = max(0.0, min(0.9, asymmetry))

    def set_closure(self, closure: float) -> None:
        """由舵机驱动调用，告知当前夹爪**机械**行程完成度 0~1。"""
        self._closure = max(0.0, min(1.0, closure))

    def force_model_n(self, closure: Optional[float] = None) -> float:
        """当前闭合量下产生的力（不含噪声），便于自检和绘图。"""
        s = self._closure if closure is None else closure
        if s <= self._contact_at:
            return 0.0
        excess = s - self._contact_at
        # 略微的非线性：越压越硬（仿真实 FSR + 硅胶垫的组合刚度）
        return self._stiffness * excess * (1.0 + 1.6 * excess)

    def read_raw(self, channel: int) -> float:
        force = self.force_model_n()
        # 两侧不等：通道 1 受力更大或更小，由 asymmetry 决定方向
        if channel == 1:
            force *= 1.0 + self._asymmetry
        else:
            force *= 1.0 - self._asymmetry
        raw = force * self.RAW_PER_NEWTON + self._rng.gauss(0.0, self._noise)
        return max(0.0, raw)


class Ads1115Adc(AdcDriver):
    """
    ADS1115 16bit I2C ADC，直接走 /dev/i2c-N + ioctl，不依赖第三方库。

    接线（单端测量，ADDR 接地 → 地址 0x48）：
        VDD  → 3.3V
        GND  → GND
        SDA  → K1 的 I2C SDA
        SCL  → K1 的 I2C SCL
        ADDR → GND
        A0   → 左侧 FSR 分压中点
        A1   → 右侧 FSR 分压中点

    FSR 分压电路（每个传感器一份）：
        3.3V ──[ FSR ]──┬── A0
                        │
                     [ 10kΩ ]
                        │
                       GND
        压力越大 → FSR 阻值越小 → A0 电压越高 → 读数越大
    """

    name = "ads1115"

    # 寄存器指针
    _REG_CONVERSION = 0x00
    _REG_CONFIG = 0x01

    # 单端输入的多路选择器配置（bits 14:12）
    _MUX_SINGLE = {0: 0b100, 1: 0b101, 2: 0b110, 3: 0b111}

    # PGA 配置（bits 11:9）→ 满量程电压
    _PGA = {
        1: (0b000, 4.096),
        2: (0b001, 2.048),
        4: (0b010, 1.024),
        8: (0b011, 0.512),
        16: (0b100, 0.256),
    }

    # 采样率配置（bits 7:5）
    _DR = {8: 0b000, 16: 0b001, 32: 0b010, 64: 0b011,
           128: 0b100, 250: 0b101, 475: 0b110, 860: 0b111}

    def __init__(self, i2c_bus: int = 1, address: int = 0x48,
                 gain: int = 1, sample_rate: int = 128):
        if fcntl is None:
            raise SensorError("当前系统不支持 fcntl/ioctl，无法使用 I2C")

        self.address = address
        self.gain = gain
        if gain not in self._PGA:
            raise SensorError(f"ADS1115 不支持的 gain={gain}，可选 {list(self._PGA)}")
        if sample_rate not in self._DR:
            raise SensorError(
                f"ADS1115 不支持的 sample_rate={sample_rate}，可选 {list(self._DR)}"
            )
        self._pga_bits, self.full_scale_v = self._PGA[gain]
        self._dr_bits = self._DR[sample_rate]
        # LSB 大小：满量程 / 2^15（单端时正半轴）
        self.lsb_v = self.full_scale_v / 32768.0
        self.raw_max = 32767.0

        self._path = f"/dev/i2c-{i2c_bus}"
        try:
            self._fd = os.open(self._path, os.O_RDWR)
        except FileNotFoundError as exc:
            raise SensorError(
                f"找不到 {self._path}。请确认 I2C 已启用：ls /dev/i2c*"
            ) from exc
        except PermissionError as exc:
            raise SensorError(
                f"打开 {self._path} 权限不足。把用户加入 i2c 组："
                "sudo usermod -aG i2c $USER 然后重新登录"
            ) from exc

        try:
            fcntl.ioctl(self._fd, I2C_SLAVE, self.address)
        except OSError as exc:
            self.close()
            raise SensorError(
                f"无法与地址 0x{self.address:02X} 通信。用 i2cdetect 确认地址。"
            ) from exc

    def _write_register(self, reg: int, value: int) -> None:
        os.write(self._fd, struct.pack(">BH", reg, value & 0xFFFF))

    def _read_register(self, reg: int) -> int:
        os.write(self._fd, struct.pack(">B", reg))
        data = os.read(self._fd, 2)
        if len(data) != 2:
            raise SensorError("ADS1115 读取返回长度异常")
        return struct.unpack(">H", data)[0]

    def read_raw(self, channel: int) -> float:
        """读取指定通道的单端电压对应的原始码值（有符号，0~32767）。"""
        if channel not in self._MUX_SINGLE:
            raise SensorError(f"ADS1115 通道只能是 0-3，收到 {channel}")

        # 配置字：OS=1(启动单次转换) | MUX | PGA | MODE=1(单次) | DR | 比较器关闭
        config = 0x8000
        config |= self._MUX_SINGLE[channel] << 12
        config |= self._pga_bits << 9
        config |= 0b1 << 8          # MODE = 单次转换
        config |= self._dr_bits << 5
        config |= 0b11              # 比较器禁用
        self._write_register(self._REG_CONFIG, config)

        # 等转换完成：读取 config 的 bit15 (OS)，1 表示空闲/完成
        # 超时 = 1/采样率 的两倍，最少 20ms
        deadline = time.monotonic() + max(0.02, 2.0 / max(1, self._current_dr()))
        while time.monotonic() < deadline:
            if self._read_register(self._REG_CONFIG) & 0x8000:
                break
            time.sleep(0.001)
        else:
            raise SensorError("ADS1115 转换超时")

        raw = self._read_register(self._REG_CONVERSION)
        if raw > 32767:
            raw -= 65536
        return float(max(0, raw))

    def _current_dr(self) -> int:
        for sps, bits in self._DR.items():
            if bits == self._dr_bits:
                return sps
        return 128

    def read_voltage(self, channel: int) -> float:
        return self.read_raw(channel) * self.lsb_v

    def close(self) -> None:
        fd = getattr(self, "_fd", None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            self._fd = None

    def describe(self) -> str:
        return (f"ADS1115 @ {self._path} addr=0x{self.address:02X} "
                f"gain={self.gain} ({self.full_scale_v}V)")


class Mcp3008Adc(AdcDriver):
    """
    MCP3008 10bit SPI ADC。

    接线：VDD/VREF → 3.3V，AGND/DGND → GND，CH0/CH1 → FSR 分压中点，
    CLK/DIN/DOUT/CS → K1 的 SPI 引脚。
    """

    name = "mcp3008"
    raw_max = 1023.0

    def __init__(self, spi_bus: int = 0, spi_device: int = 0, vref: float = 3.3):
        self.vref = vref
        self.lsb_v = vref / 1024.0
        self._path = f"/dev/spidev{spi_bus}.{spi_device}"
        try:
            self._fd = os.open(self._path, os.O_RDWR)
        except FileNotFoundError as exc:
            raise SensorError(
                f"找不到 {self._path}。请确认 SPI 已启用：ls /dev/spidev*"
            ) from exc
        except PermissionError as exc:
            raise SensorError(f"打开 {self._path} 权限不足，检查 spi 组权限") from exc

        # SPI 模式配置（MODE 0, 1MHz）
        if fcntl is not None:
            try:
                SPI_IOC_WR_MODE = 0x40016B01
                SPI_IOC_WR_MAX_SPEED_HZ = 0x40046B04
                fcntl.ioctl(self._fd, SPI_IOC_WR_MODE, struct.pack("B", 0))
                fcntl.ioctl(self._fd, SPI_IOC_WR_MAX_SPEED_HZ,
                            struct.pack("I", 1_000_000))
            except OSError:
                # 配置失败不致命，用内核默认参数继续
                pass

    def read_raw(self, channel: int) -> float:
        if not 0 <= channel <= 7:
            raise SensorError(f"MCP3008 通道只能是 0-7，收到 {channel}")
        # 标准 MCP3008 单端读取帧：起始位 + 单端标志 + 通道号 + 填充
        tx = bytes([0x01, (0x08 | channel) << 4, 0x00])
        os.write(self._fd, tx)
        rx = os.read(self._fd, 3)
        if len(rx) != 3:
            raise SensorError("MCP3008 读取返回长度异常")
        value = ((rx[1] & 0x03) << 8) | rx[2]
        return float(value)

    def read_voltage(self, channel: int) -> float:
        return self.read_raw(channel) * self.lsb_v

    def close(self) -> None:
        fd = getattr(self, "_fd", None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            self._fd = None

    def describe(self) -> str:
        return f"MCP3008 @ {self._path} vref={self.vref}V"


class K1SarAdc(AdcDriver):
    """
    K1 片内 SARADC，走 Linux IIO 子系统。

    设备路径形如 /sys/bus/iio/devices/iio:deviceN/in_voltageN_raw。
    注意：K1 的 SARADC 引脚在 MUSE Pi Pro 的 40Pin 上是否引出需要实测确认，
    没引出就用外置 ADS1115。
    """

    name = "k1_saradc"

    def __init__(self, device: int = 0, vref: float = 1.8, bits: int = 12, scale: float = 1.0):
        self.vref = vref
        self.raw_max = float((1 << bits) - 1)
        self.scale = scale
        self._base = f"/sys/bus/iio/devices/iio:device{device}"
        if not os.path.isdir(self._base):
            raise SensorError(
                f"找不到 {self._base}。用 'ls /sys/bus/iio/devices/' 查看实际设备名，"
                "并把 device 参数改成对应编号。"
            )

    def read_raw(self, channel: int) -> float:
        # 有的驱动把通道写作 in_voltage0_raw，有的写作 in_voltage_0_raw
        candidates = [
            os.path.join(self._base, f"in_voltage{channel}_raw"),
            os.path.join(self._base, f"in_voltage_{channel}_raw"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="ascii") as fh:
                        value = float(fh.read().strip())
                    return max(0.0, value * self.scale)
                except (OSError, ValueError) as exc:
                    raise SensorError(f"读取 {path} 失败：{exc}") from exc
        raise SensorError(
            f"通道 {channel} 不存在，候选路径：{candidates}"
        )

    def describe(self) -> str:
        return f"K1 SARADC @ {self._base}"


def create_adc(cfg) -> AdcDriver:
    """根据配置创建 ADC 驱动实例。"""
    adc_type = cfg.get_str("sensor.adc", "simulated").lower()

    if cfg.simulate or adc_type == "simulated":
        sim = cfg.get("sensor.simulated", {}) or {}
        return SimulatedAdc(
            preset=sim.get("preset", "grape"),
            contact_at=sim.get("contact_at"),
            stiffness=sim.get("stiffness"),
            asymmetry=float(sim.get("asymmetry", 0.08)),
        )

    if adc_type == "ads1115":
        return Ads1115Adc(
            i2c_bus=cfg.get_int("sensor.ads1115.i2c_bus", 1),
            address=int(cfg.get("sensor.ads1115.address", 0x48)),
            gain=cfg.get_int("sensor.ads1115.gain", 1),
            sample_rate=cfg.get_int("sensor.ads1115.sample_rate", 128),
        )

    if adc_type == "mcp3008":
        return Mcp3008Adc(
            spi_bus=cfg.get_int("sensor.mcp3008.spi_bus", 0),
            spi_device=cfg.get_int("sensor.mcp3008.spi_device", 0),
            vref=cfg.get_float("sensor.mcp3008.vref", 3.3),
        )

    if adc_type == "k1_saradc":
        return K1SarAdc()

    raise SensorError(f"未知的 sensor.adc 类型：{adc_type!r}")


# ---------------------------------------------------------------------------
# 力传感器：封装单个 FSR（标定 + 滤波）
# ---------------------------------------------------------------------------

class ForceSensor:
    """
    单个压力传感器。

    标定模型（两段式，兼顾简单与可用）：
        1. 减去零点：        raw' = max(0, raw - raw_zero)
        2. 幂律映射：        F = force_max_n * (raw' / raw_span) ** gamma

    gamma = 1.0 时就是线性映射。真实 FSR 的 gamma 通常在 0.7~1.3 之间，
    用 apps/calibrate_sensor.py 拟合得到。

    **重要**：这个映射给出的"牛顿"是相对力，用于闭环控制足够；
    若要标注绝对力值，需要用标准砝码逐点标定。
    """

    def __init__(self, name: str, driver: AdcDriver, channel: int, calib: dict,
                 filter_window: int = 7, median_window: int = 5):
        self.name = name
        self.driver = driver
        self.channel = channel

        self.raw_zero = float(calib.get("raw_zero", 0.0))
        self.raw_span = float(calib.get("raw_span", 1000.0))
        self.force_max_n = float(calib.get("force_max_n", 10.0))
        self.gamma = float(calib.get("gamma", 1.0))
        if self.raw_span <= 0:
            raise SensorError(f"{name}: raw_span 必须大于 0")

        self._median = MedianFilter(median_window) if median_window > 1 else None
        self._average = MovingAverage(max(1, filter_window))

        self.raw_value: float = 0.0
        self.force_n: float = 0.0
        self.saturated: bool = False

    def update(self) -> float:
        """采样一次并更新滤波后的力值，返回牛顿数。"""
        raw = self.driver.read_raw(self.channel)
        self.raw_value = raw
        self.saturated = raw >= self.driver.raw_max * 0.99

        value = raw
        if self._median is not None:
            value = self._median.update(value)
        value = self._average.update(value)

        self.force_n = self._raw_to_force(value)
        return self.force_n

    def _raw_to_force(self, raw: float) -> float:
        excess = max(0.0, raw - self.raw_zero)
        ratio = min(1.0, excess / self.raw_span)
        if self.gamma == 1.0:
            return self.force_max_n * ratio
        return self.force_max_n * math.pow(ratio, self.gamma)

    def tare(self, samples: int = 50, delay_s: float = 0.005) -> float:
        """
        去皮：在夹爪完全张开（无接触）状态下采样，把当前读数设为新的零点。
        **每次开机、温度变化后都应该重新 tare**，FSR 的零点漂移很明显。
        """
        total = 0.0
        for _ in range(max(1, samples)):
            total += self.driver.read_raw(self.channel)
            time.sleep(delay_s)
        self.raw_zero = total / max(1, samples)
        self._median.reset()
        self._average.reset()
        return self.raw_zero

    @property
    def filtered_ready(self) -> bool:
        return self._average.ready

    def reset_filter(self) -> None:
        if self._median is not None:
            self._median.reset()
        self._average.reset()

    def describe(self) -> str:
        return (f"ForceSensor({self.name}) ch={self.channel} "
                f"zero={self.raw_zero:.1f} span={self.raw_span:.1f} "
                f"Fmax={self.force_max_n}N gamma={self.gamma}")


class ForceSensorPair:
    """
    左右两个传感器作为一个整体。

    对外提供：
      - ``update()``      → 采样两个通道
      - ``force_n``       → 平均力（控制环用这个）
      - ``asymmetry``     → 相对不平衡度，用于判断葡萄是否对中
    """

    def __init__(self, left: ForceSensor, right: ForceSensor):
        self.left = left
        self.right = right
        self.force_n: float = 0.0
        self.force_left_n: float = 0.0
        self.force_right_n: float = 0.0

    def update(self) -> float:
        self.force_left_n = self.left.update()
        self.force_right_n = self.right.update()
        # 对称夹持时两侧力接近相等，取平均比取和大更能反映"每侧受力"
        self.force_n = 0.5 * (self.force_left_n + self.force_right_n)
        return self.force_n

    @property
    def max_force_n(self) -> float:
        """取较大的一侧，用于安全判定（保守）。"""
        return max(self.force_left_n, self.force_right_n)

    @property
    def asymmetry(self) -> float:
        """
        不平衡度 0~1。0 = 完全对称，1 = 只有一侧受力。

        用途：夹持时若 asymmetry > 0.5，说明葡萄没夹在中间，
        继续加力会让它侧向挤出 → 应该先退开重新对位，而不是硬夹。
        """
        total = self.force_left_n + self.force_right_n
        if total <= 1e-6:
            return 0.0
        return abs(self.force_left_n - self.force_right_n) / total

    def tare(self, samples: int = 50) -> tuple[float, float]:
        return self.left.tare(samples), self.right.tare(samples)

    def reset_filter(self) -> None:
        self.left.reset_filter()
        self.right.reset_filter()

    def describe(self) -> str:
        return (f"[{self.left.describe()}] [{self.right.describe()}] "
                f"driver={self.left.driver.describe()}")


def create_sensor_pair(cfg, driver: Optional[AdcDriver] = None) -> ForceSensorPair:
    """根据配置创建左右传感器对。"""
    driver = driver or create_adc(cfg)
    window = cfg.get_int("sensor.filter_window", 7)
    median = cfg.get_int("sensor.median_window", 5)
    left = ForceSensor(
        "left", driver, cfg.get_int("sensor.channel_left", 0),
        cfg.get("sensor.calib.left", {}) or {},
        filter_window=window, median_window=median,
    )
    right = ForceSensor(
        "right", driver, cfg.get_int("sensor.channel_right", 1),
        cfg.get("sensor.calib.right", {}) or {},
        filter_window=window, median_window=median,
    )
    return ForceSensorPair(left, right)
