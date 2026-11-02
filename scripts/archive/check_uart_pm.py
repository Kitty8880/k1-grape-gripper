#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 r_uart0 在打开串口时是否从 runtime suspend 唤醒。"""
import subprocess
import time

RT = "/sys/devices/platform/soc/c0881000.r_uart0/power/runtime_status"


def rtpm() -> str:
    try:
        return open(RT).read().strip()
    except OSError as exc:
        return f"读取失败: {exc}"


def reg(addr: str) -> str:
    r = subprocess.run(["busybox", "devmem", addr],
                       capture_output=True, text=True)
    return r.stdout.strip()


print("=== 打开串口前 ===")
print("rtpm:", rtpm())
print("LCR :", reg("0xc088100c"))
print("LSR :", reg("0xc0881014"))

import serial
s = serial.Serial("/dev/ttyS10", 115200, timeout=0.2)
try:
    print()
    print("=== 打开串口后 ===")
    print("rtpm:", rtpm())
    print("LCR :", reg("0xc088100c"))
    print("LSR :", reg("0xc0881014"))
    print("DLL :", reg("0xc0881000"))
    print("IER :", reg("0xc0881004"))

    print()
    print("=== 发送数据测试 ===")
    s.write(b"HELLO_K1\r\n")
    s.flush()
    time.sleep(0.2)
    print("发送后 LSR:", reg("0xc0881014"))
    print("发送后 rtpm:", rtpm())

    print()
    print("=== 尝试读取（5 秒）===")
    t0 = time.time()
    total = 0
    while time.time() - t0 < 5:
        d = s.read(64)
        if d:
            total += len(d)
            print("收到:", repr(d))
    print(f"共收到 {total} 字节")
finally:
    s.close()

print()
print("=== 关闭串口后 ===")
time.sleep(0.5)
print("rtpm:", rtpm())
