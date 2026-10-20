# -*- coding: utf-8 -*-
"""扫描 archive 里的 zlib 头，定位真实数据布局。"""
import struct
import sys
import io
import zlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PATH = r"D:\文档\deepseek-harness\default-workspace\ps01tool.exe"
data = open(PATH, "rb").read()

cookie_pos = 107784123
pkg_len = 107504659
archive_start = len(data) - pkg_len

print(f"archive_start = {archive_start}")

# 1) dump archive 开头 256 字节（看是不是名字表）
print("\n=== archive 开头 256 字节 ===")
blob = data[archive_start:archive_start + 256]
for i in range(0, len(blob), 32):
    chunk = blob[i:i + 32]
    hx = " ".join(f"{b:02X}" for b in chunk)
    asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
    print(f"  +{i:04X}  {hx:<96}  |{asc}|")

# 2) 在 archive 前 100KB 里扫 zlib 头（78 01/9C/DA）
print("\n=== 前 100KB 内的 zlib 头位置（前 20 个）===")
count = 0
for off in range(0, min(100_000, len(data) - archive_start - 2)):
    if data[archive_start + off] == 0x78 and data[archive_start + off + 1] in (0x01, 0x9C, 0xDA):
        try:
            zlib.decompressobj().decompress(data[archive_start + off:archive_start + off + 512])
            print(f"  offset {off:6d}  (0x{off:04X})  zlib 头 {data[archive_start+off]:02X} {data[archive_start+off+1]:02X}")
            count += 1
            if count >= 20:
                break
        except zlib.error:
            pass

# 3) 验证 TOC 前几条的 data_pos 处是什么
print("\n=== TOC 前 6 条的 data_pos 处字节 ===")
toc_off = 107460427
toc_len = 44144
toc_start = archive_start + toc_off
p = toc_start
for n in range(6):
    entry_len = struct.unpack(">i", data[p:p + 4])[0]
    name_pos = struct.unpack(">i", data[p + 4:p + 8])[0]
    data_pos = struct.unpack(">i", data[p + 8:p + 12])[0]
    length = struct.unpack(">i", data[p + 12:p + 16])[0]
    cflag = data[p + 16]
    typ = chr(data[p + 17])
    rest = data[p + 16:p + entry_len]
    name = rest[1:].split(b"\x00")[0].decode("utf-8", "replace")
    off = archive_start + data_pos
    sample = data[off:off + 8]
    print(f"\n[{n}] name={name!r} type={typ} cflag={cflag} "
          f"data_pos={data_pos} len={length}")
    print(f"     数据处字节: {sample.hex(' ')}")
    # 尝试解压
    try:
        out = zlib.decompress(data[off:off + length])
        print(f"     ✓ zlib 解压成功 {len(out)} 字节, 头: {out[:8].hex(' ')}")
    except zlib.error as e:
        print(f"     ✗ zlib 解压失败: {e}")
    p += entry_len
