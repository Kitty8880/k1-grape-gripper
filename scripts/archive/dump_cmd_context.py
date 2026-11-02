# -*- coding: utf-8 -*-
"""dump 主脚本里关键字符串附近的字节，定位命令拼接方式。"""
import struct
import sys
import io
import zlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PATH = r"D:\文档\deepseek-harness\default-workspace\ps01tool.exe"
data = open(PATH, "rb").read()

pkg_len = 107504659
toc_off = 107460427
toc_len = 44144
archive_start = len(data) - pkg_len
toc_start = archive_start + toc_off

p = toc_start
main = None
while p < toc_start + toc_len:
    entry_len = struct.unpack(">i", data[p:p + 4])[0]
    if entry_len < 18:
        break
    name = data[p + 18:p + entry_len].split(b"\x00")[0].decode("utf-8", "replace")
    if name == "PS01Tool":
        data_pos = struct.unpack(">i", data[p + 4:p + 8])[0]
        length = struct.unpack(">i", data[p + 12:p + 16])[0]
        cflag = data[p + 16]
        main = (data_pos, length, cflag)
        break
    p += entry_len

data_pos, length, cflag = main
raw = data[archive_start + data_pos: archive_start + data_pos + length]
if cflag:
    raw = zlib.decompress(raw)

targets = [
    "发送开始命令失败".encode("utf-8"),
    "发送停止命令失败".encode("utf-8"),
    "采集已开启".encode("utf-8"),
    b"ON",
    b"OFF",
]

for t in targets:
    print("=" * 72)
    print(f"目标: {t!r}")
    print("=" * 72)
    idx = raw.find(t)
    n = 0
    while idx != -1 and n < 3:
        lo = max(0, idx - 100)
        hi = min(len(raw), idx + len(t) + 100)
        seg = raw[lo:hi]
        print(f"\n  ---- 位置 {idx} (上下文 {lo}..{hi}) ----")
        # hex
        hx = " ".join(f"{b:02X}" for b in seg)
        print(f"  HEX: {hx}")
        # ascii
        asc = "".join(chr(b) if 32 <= b <= 126 else ("·" if b < 128 else "?") for b in seg)
        print(f"  ASC: {asc}")
        idx = raw.find(t, idx + 1)
        n += 1
    print()
