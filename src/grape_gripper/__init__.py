# -*- coding: utf-8 -*-
"""
葡萄无损夹取 —— 力闭环夹爪控制包。

模块划分::

    config.py    配置加载（零依赖 YAML 子集解析）
    sensor.py    ADC 驱动 + 力传感器标定 + 滤波
    actuator.py  舵机驱动（sysfs PWM / 软件 PWM / 模拟）
    gripper.py   ★ 力闭环状态机（核心）
    safety.py    独立安全看门狗
    vision.py    视觉识别（阶段二）

快速上手::

    from grape_gripper import load_default_config, validate_grip_config
    from grape_gripper.sensor import create_sensor_pair
    from grape_gripper.actuator import create_servo
    from grape_gripper.gripper import GripperController

    cfg = load_default_config()
    for w in validate_grip_config(cfg):
        print("配置警告:", w)

    sensors = create_sensor_pair(cfg)
    servo = create_servo(cfg, adc=sensors.left.driver)
    ctrl = GripperController(cfg, sensors, servo, log=print)

    ctrl.start_grip()
    while not ctrl.finished:
        ctrl.step()
        time.sleep(0.01)
    print(ctrl.result.summary())
"""

from .config import Config, ConfigError, load_default_config, validate_grip_config
from .gripper import AbortReason, GripResult, GripperController, GripState
from .sensor import (
    AdcDriver,
    ForceSensor,
    ForceSensorPair,
    SensorError,
    create_adc,
    create_sensor_pair,
)
from .actuator import ActuatorError, ServoBase, create_servo

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ConfigError",
    "load_default_config",
    "validate_grip_config",
    "AdcDriver",
    "ForceSensor",
    "ForceSensorPair",
    "SensorError",
    "create_adc",
    "create_sensor_pair",
    "ActuatorError",
    "ServoBase",
    "create_servo",
    "AbortReason",
    "GripResult",
    "GripperController",
    "GripState",
    "__version__",
]
