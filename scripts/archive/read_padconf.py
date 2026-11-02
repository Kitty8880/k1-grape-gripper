#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接读 K1 的 PADCONF 寄存器，确认 GPIO 的复用功能（mux）。

K1 pinctrl 基址 0xd401e000，每个 PAD 占 4 字节：
    PADCONF(GPIO n) = 0xd401e000 + n*4

低 3 位是 mux 功能号：
    0 = GPIO
    1 = 第二功能（如 r_uart0_tx/rx）
    2/3/4/5 = 其他复用

需要 root（/dev/mem）。
"""
import mmap
import struct
import sys

BASE = 0xD401E000
PAGE = 0x1000

# 关注的 GPIO（按 40Pin 相关）
TARGETS = {
    "GPIO47 (脚33? R_UART0_TXD 候选)": 47,
    "GPIO48 (脚35? R_UART0_RXD 候选)": 48,
    "GPIO49 (脚22?)": 49,
    "GPIO68 (uart0_2 txd)": 68,
    "GPIO69 (uart0_2 rxd)": 69,
    "GPIO70 (脚7  uart5_txd)": 70,
    "GPIO71 (脚11 uart5_rxd)": 71,
    "GPIO75 (脚23 uart8_txd)": 75,
    "GPIO76 (脚24 uart8_rxd)": 76,
}


def main() -> int:
    try:
        with open("/dev/mem", "rb") as f:
            mm = mmap.mmap(f.fileno(), PAGE, offset=BASE)
    except (OSError, PermissionError) as exc:
        print(f"❌ 打不开 /dev/mem: {exc}")
        print("   需要 root，且内核允许访问（CONFIG_STRICT_DEVMEM）")
        return 2

    print(f"K1 PADCONF 基址 0x{BASE:08X}")
    print("=" * 62)
    print(f"{'GPIO':<40} {'寄存器值':<12} {'mux'}")
    print("=" * 62)

    for label, gpio in TARGETS.items():
        off = gpio * 4
        if off + 4 > PAGE:
            continue
        val = struct.unpack("<I", mm[off:off + 4])[0]
        mux = val & 0x7
        mark = ""
        if mux == 0:
            mark = "  ← GPIO（未复用）"
        elif mux == 1:
            mark = "  ← mux1（可能是 r_uart0）"
        print(f"{label:<40} 0x{val:08X}   {mux}{mark}")

    print()
    print("提示：")
    print("  mux=0 表示该引脚仍是普通 GPIO（UART 收不到数据）")
    print("  r_uart0 的 txd/rxd 需要 mux=1")
    mm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
