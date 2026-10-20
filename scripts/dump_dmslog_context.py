# -*- coding: utf-8 -*-
"""
dump .dmslog 里 TX 记录前后的原始字节，找 PS01Tool 在发 gnst 之前/之后还做了什么。
同时尝试解析记录结构（长度 + 方向 + 时间戳 + 数据）。
"""
import sys
import io
import os
import struct
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PATH = r"D:\文档\DMS Log Files\USB-SERIAL CH340 (COM11)$20261017-142839.dmslog8"

# gnst 的两处位置
GNST_POS = [126213939, 126216394]
# gned 的一处
GNED_POS = [183080111]


def hexdump(data: bytes, base: int = 0) -> str:
    out = []
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        hx = " ".join(f"{b:02X}" for b in chunk)
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        out.append(f"  {base+i:08X}  {hx:<48} |{asc}|")
    return "\n".join(out)


def filetime_to_str(raw: bytes) -> str:
    """Windows FILETIME -> 可读时间。"""
    try:
        val = struct.unpack("<Q", raw)[0]
        if val == 0:
            return "0"
        # FILETIME: 1601-01-01 起的 100ns
        dt = datetime(1601, 1, 1) + timedelta(microseconds=val / 10)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except (struct.error, OverflowError, ValueError):
        return "?"


def main() -> int:
    if not os.path.isfile(PATH):
        print(f"文件不存在：{PATH}")
        return 1

    size = os.path.getsize(PATH)
    with open(PATH, "rb") as fh:
        for label, positions in (("gnst", GNST_POS), ("gned", GNED_POS)):
            for pos in positions:
                lo = max(0, pos - 200)
                hi = min(size, pos + 200)
                fh.seek(lo)
                data = fh.read(hi - lo)
                print("=" * 78)
                print(f"{label} @ 文件偏移 {pos}（dump {lo}..{hi}）")
                print("=" * 78)
                print(hexdump(data, lo))
                print()

    # 尝试按"长度(4) + 类型(4) + 时间戳(8) + 数据"解析一段
    print("=" * 78)
    print("尝试解析记录结构（从 gnst 前一段开始）")
    print("=" * 78)
    start = GNST_POS[0] - 96
    with open(PATH, "rb") as fh:
        fh.seek(start)
        blob = fh.read(400)

    off = 0
    for i in range(8):
        if off + 16 > len(blob):
            break
        length = struct.unpack("<I", blob[off:off + 4])[0]
        kind = struct.unpack("<I", blob[off + 4:off + 8])[0]
        ts = blob[off + 8:off + 16]
        print(f"\n  记录 @+{off:3d}: length={length:<6} kind={kind:<6} "
              f"ts={filetime_to_str(ts)}")
        if 0 < length <= 200 and off + 16 + length <= len(blob):
            payload = blob[off + 16:off + 16 + length]
            asc = "".join(chr(b) if 32 <= b < 127 else "." for b in payload)
            print(f"      payload({length}): {payload[:64].hex(' ')}")
            print(f"      ascii          : {asc[:64]}")
            off += 16 + length
        else:
            print(f"      （长度不合理，停止解析）")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
