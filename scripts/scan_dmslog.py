# -*- coding: utf-8 -*-
"""
分析 Device Monitoring Studio 的 .dmslog 二进制日志，找出 PS01Tool 实际发送的命令。

思路：
    RX（传感器输出）里有很多可识别的文本，比如 "[INFO]  JSON module initialized"。
    TX（PS01Tool 发送）里是命令字节。
    在文件里搜这些特征，对比上下文，就能推断记录结构并定位 TX。

搜索目标：
    - b"gnst"                   启动命令候选
    - b"gned"                   停止命令候选
    - b"JSON module initialized"  RX 特征（用来识别记录结构）
    - b"Data callback registered" RX 特征
    - b"measurement started"     响应特征
"""
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PATH = r"D:\文档\DMS Log Files\USB-SERIAL CH340 (COM11)$20261017-142839.dmslog8"

TARGETS = [
    b"gnst",
    b"gned",
    b"measurement started",
    b"Data callback registered",
    b"JSON module initialized",
]

CHUNK = 4 * 1024 * 1024          # 4MB 一块
OVERLAP = 64                      # 块间重叠，防止特征跨块被切


def hexdump(data: bytes) -> str:
    out = []
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        hx = " ".join(f"{b:02X}" for b in chunk)
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        out.append(f"    {i:04X}  {hx:<48} |{asc}|")
    return "\n".join(out)


def main() -> int:
    if not os.path.isfile(PATH):
        print(f"文件不存在：{PATH}")
        return 1

    size = os.path.getsize(PATH)
    print(f"文件大小：{size} 字节（{size/1024/1024:.1f} MB）\n")

    hits = {t: [] for t in TARGETS}
    pos = 0
    tail = b""

    with open(PATH, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            buf = tail + chunk
            base = pos - len(tail)
            for t in TARGETS:
                start = 0
                while True:
                    idx = buf.find(t, start)
                    if idx < 0:
                        break
                    abs_pos = base + idx
                    if len(hits[t]) < 6:      # 每类最多记 6 处
                        hits[t].append(abs_pos)
                    start = idx + 1
            tail = buf[-OVERLAP:] if len(buf) > OVERLAP else buf
            pos += len(chunk)
            print(f"  已扫描 {pos // (1024*1024):4d} MB …", end="\r", flush=True)

    print("\n")
    print("=" * 78)
    print("命中统计")
    print("=" * 78)
    for t, positions in hits.items():
        label = t.decode("ascii", "replace")
        print(f"  {label!r:<32} {len(positions)} 处（显示前 {len(positions)}）")

    # 把每个命中点附近的字节 dump 出来
    print()
    for t in TARGETS:
        positions = hits[t]
        if not positions:
            continue
        label = t.decode("ascii", "replace")
        print("=" * 78)
        print(f"目标 {label!r}")
        print("=" * 78)
        with open(PATH, "rb") as fh:
            for p in positions[:3]:
                lo = max(0, p - 48)
                fh.seek(lo)
                data = fh.read(len(t) + 96)
                print(f"\n  ---- 文件偏移 {p} (0x{p:X})，上下文 {lo}.. ----")
                print(hexdump(data))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
