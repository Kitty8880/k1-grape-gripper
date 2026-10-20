# -*- coding: utf-8 -*-
"""
极简 YAML 子集解析器 + 配置加载。

为什么不直接用 PyYAML？
    目标板 Bianbu OS 上 pip 装 pyyaml 通常没问题，但我不想让"第一步就卡在
    装依赖"上。config.yaml 只用到 YAML 的一个很小子集（嵌套映射、标量、
    注释、行内注释），这里自己解析，零依赖、一定能跑。

支持的语法：
    key: value
    key:
      nested: value
    # 整行注释
    key: value   # 行尾注释
    key: "带引号的字符串"
    key: [1, 2, 3]          # 行内列表
    key:
      - item1               # 块列表
      - item2

不支持（本项目也用不到）：多行字符串、锚点、复杂嵌套列表、flow mapping。
"""

from __future__ import annotations

import os
import re
from typing import Any


class ConfigError(Exception):
    """配置解析或校验出错。"""


# ---------------------------------------------------------------------------
# YAML 子集解析
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    """去掉行尾注释，但保留引号内的 #。"""
    out = []
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            # # 前面必须是空白或行首，才算注释（避免把 a#b 当成注释）
            if i == 0 or line[i - 1].isspace():
                break
        out.append(ch)
    return "".join(out).rstrip()


_SCALAR_TRUE = {"true", "yes", "on"}
_SCALAR_FALSE = {"false", "no", "off"}
_NULL = {"null", "~", ""}


def _parse_scalar(text: str) -> Any:
    """把字符串转成 bool / int / float / list / str。"""
    text = text.strip()
    if text == "":
        return None

    # 引号字符串
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]

    low = text.lower()
    if low in _SCALAR_TRUE:
        return True
    if low in _SCALAR_FALSE:
        return False
    if low in _NULL:
        return None

    # 行内列表 [a, b, c]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]

    # 数字（整数优先，避免 40 被解析成 40.0）
    try:
        return int(text, 0) if re.fullmatch(r"[+-]?0[xXoObB][0-9a-fA-F_]+", text) else int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass

    return text


def parse_yaml(source: str) -> dict:
    """解析 YAML 子集，返回嵌套 dict。"""
    root: dict = {}
    # 栈元素：(缩进宽度, 该层的容器)
    stack: list[tuple[int, Any]] = [(-1, root)]
    # 记录每一层"最后一个 key 对应的容器"，用于块列表挂载
    pending_list_key: list[tuple[int, dict, str]] = []

    for lineno, raw_line in enumerate(source.splitlines(), start=1):
        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip())]:
            raise ConfigError(f"第 {lineno} 行：缩进不能用 Tab，请用空格")

        line = _strip_comment(raw_line)
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()

        # 弹栈：回到缩进小于当前行的层
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        if content.startswith("- "):
            # 块列表项
            if not pending_list_key:
                raise ConfigError(f"第 {lineno} 行：列表项没有对应的 key")
            _, parent, key = pending_list_key[-1]
            if not isinstance(parent.get(key), list):
                parent[key] = []
            parent[key].append(_parse_scalar(content[2:]))
            continue

        if ":" not in content:
            raise ConfigError(f"第 {lineno} 行：不是合法的 'key: value' —— {content!r}")

        key, _, value = content.partition(":")
        key = key.strip()
        value = value.strip()
        parent = stack[-1][1]

        if value == "":
            # 可能是嵌套映射，也可能是块列表，先建 dict 占位
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
            pending_list_key.append((indent, parent, key))
        else:
            parent[key] = _parse_scalar(value)
            # 该 key 是标量，清掉可能残留的列表挂载点
            while pending_list_key and pending_list_key[-1][0] >= indent:
                pending_list_key.pop()

    return root


# ---------------------------------------------------------------------------
# 配置访问辅助
# ---------------------------------------------------------------------------

class Config:
    """dict 的薄包装，提供点号路径访问和类型安全取值。"""

    def __init__(self, data: dict, path: str = "<memory>"):
        self._data = data
        self._path = path

    @classmethod
    def load(cls, path: str) -> "Config":
        if not os.path.isfile(path):
            raise ConfigError(f"配置文件不存在：{path}")
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        data = parse_yaml(text)
        if not isinstance(data, dict):
            raise ConfigError(f"配置文件顶层必须是映射：{path}")
        return cls(data, path)

    def get(self, dotted: str, default: Any = None) -> Any:
        """按 'a.b.c' 取值，缺失返回 default。"""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """按 'a.b.c' 取值，缺失抛异常（用于必填项）。"""
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"配置缺少必填项：{dotted} （来自 {self._path}）")
        return value

    def get_float(self, dotted: str, default: float | None = None) -> float:
        value = self.get(dotted, default)
        if value is None:
            raise ConfigError(f"配置项 {dotted} 缺失且无默认值")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"配置项 {dotted} 应为数字，实际为 {value!r}") from exc

    def get_int(self, dotted: str, default: int | None = None) -> int:
        return int(self.get_float(dotted, default))

    def get_bool(self, dotted: str, default: bool = False) -> bool:
        value = self.get(dotted, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in _SCALAR_TRUE
        return bool(value)

    def get_str(self, dotted: str, default: str = "") -> str:
        value = self.get(dotted, default)
        if value is None:
            return default
        return str(value)

    def as_dict(self) -> dict:
        return self._data

    # ---- 常用快捷属性 ----

    @property
    def simulate(self) -> bool:
        """全局仿真开关。"""
        return self.get_bool("hardware.simulate", True)


def default_config_path() -> str:
    """从当前文件位置推导项目根目录下的 config.yaml。"""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(root, "config.yaml")


def load_default_config() -> Config:
    return Config.load(default_config_path())


# ---------------------------------------------------------------------------
# 配置校验：把"参数写错"在启动时就暴露，而不是夹爆葡萄之后
# ---------------------------------------------------------------------------

def validate_grip_config(cfg: Config) -> list[str]:
    """检查关键安全参数是否自相矛盾。返回警告列表（空 = 全部正常）。"""
    warnings: list[str] = []

    target = cfg.get_float("grip.target_force_n", 0.8)
    fmax = cfg.get_float("grip.force_max_n", 1.5)
    approach = cfg.get_float("grip.force_approach_n", 0.25)

    if fmax <= target:
        warnings.append(
            f"grip.force_max_n ({fmax}) 必须大于 grip.target_force_n ({target})，"
            "否则一接触就会急停"
        )
    if approach >= target:
        warnings.append(
            f"grip.force_approach_n ({approach}) 应小于 target_force_n ({target})，"
            "否则接触判定会太晚"
        )
    if target > 3.0:
        warnings.append(
            f"grip.target_force_n = {target} N 偏大：葡萄破裂力约 1~3N，建议 ≤1.0N"
        )

    pulse_min = cfg.get_float("servo.pulse_min_us", 500)
    pulse_max = cfg.get_float("servo.pulse_max_us", 2500)
    period = cfg.get_float("servo.period_us", 20000)
    if pulse_max >= period:
        warnings.append(
            f"servo.pulse_max_us ({pulse_max}) 必须小于 period_us ({period})"
        )
    if pulse_min >= pulse_max:
        warnings.append("servo.pulse_min_us 必须小于 pulse_max_us")

    if cfg.simulate:
        warnings.append("当前 hardware.simulate = true，不会驱动真实硬件")

    return warnings
