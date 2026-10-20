# -*- coding: utf-8 -*-
"""核对 TOC 条目字段与数据位置。"""
import struct
import sys
import io
import zlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PATH = r"D:\文档\deepseek-harness\default-workspace\ps01tool.exe"
data = open(PATH, "rb").read()

cookie_pos = 107784123
pkg_len = 107504659
toc_off = 107460427
toc_len = 44144
archive_start = len(data) - pkg_len
toc_start = archive_start + toc_off

print(f"archive_start={archive_start} toc_start={toc_start}")

p = toc_start
for n in range(8):
    entry_len = struct.unpack(">i", data[p:p + 4])[0]
    name_pos = struct.unpack(">i", data[p + 4:p + 8])[0]
    data_pos = struct.unpack(">i", data[p + 8:p + 12])[0]
    length = struct.unpack(">i", data[p + 12:p + 16])[0]
    rest = data[p + 16:p + entry_len]
    print(f"\n[{n}] entry_len={entry_len} name_pos={name_pos} "
          f"data_pos={data_pos} length={length}")
    print(f"    rest 前 40 字节: {rest[:40].hex(' ')}")
    print(f"    rest ASCII     : {rest[:40]!r}")

    # 检查 data_pos 位置
    for label, base in (("archive_start+data_pos", archive_start),
                        ("toc 相对", 0), ("文件绝对", 0)):
        off = base + data_pos
        if 0 <= off < len(data) - 4:
            sample = data[off:off + 4]
            zlib_ok = ""
            if sample[:1] == b"\x78":
                try:
                    zlib.decompress(data[off:off + 64])
                    zlib_ok = "  ← 疑似 zlib 头"
                except zlib.error:
                    pass
            print(f"    {label}: offset={off} 前4字节={sample.hex(' ')}{zlib_ok}")

    p += entry_len
