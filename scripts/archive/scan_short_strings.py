# -*- coding: utf-8 -*-
"""
重新扫描 PS01Tool 主脚本，这次包含 1~2 字符的短字符串和 bytes 常量。
重点找 send_text_command 的实参。
"""
import struct
import sys
import io
import re
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

print(f"主脚本 {len(raw)} 字节\n")

# ---- 1) send_text_command 附近的字节码 ----
print("=" * 74)
print("① send_text_command 方法体附近（找 LOAD_CONST 的参数）")
print("=" * 74)
# Python 3.9 字节码：LOAD_METHOD=0xA0(160), LOAD_CONST=0x64(100), CALL_METHOD=0xA1(161)
idx = raw.find(b"send_text_command")
while idx != -1:
    lo = max(0, idx - 60)
    hi = min(len(raw), idx + 200)
    seg = raw[lo:hi]
    asc = "".join(chr(b) if 32 <= b < 127 else "." for b in seg)
    print(f"\n  @{idx}  {asc}")
    idx = raw.find(b"send_text_command", idx + 1)

# ---- 2) LOAD_METHOD + LOAD_CONST + CALL_METHOD 模式 ----
print("\n" + "=" * 74)
print("② 形如 LOAD_METHOD x; LOAD_CONST y; CALL_METHOD 1 的字节码序列")
print("=" * 74)
pat = re.compile(rb"\xa0(.)\x64(..)\xa1\x01", re.DOTALL)  # A0 xx 64 xx xx A1 01
cnt = 0
for m in pat.finditer(raw):
    lo = max(0, m.start() - 30)
    hi = min(len(raw), m.end() + 30)
    seg = raw[lo:hi]
    asc = "".join(chr(b) if 32 <= b < 127 else "." for b in seg)
    method_idx = m.group(1)[0]
    const_idx = struct.unpack("<H", m.group(2))[0]
    print(f"  @{m.start():6d}  method#{method_idx:<3} const#{const_idx:<4}  {asc}")
    cnt += 1
    if cnt >= 40:
        print("  ...（截断）")
        break

# ---- 3) 所有 1~2 字符的可打印串 ----
print("\n" + "=" * 74)
print("③ 1~2 字符的可打印串（命令可能是这种）")
print("=" * 74)
found = set()
for m in re.finditer(rb"[A-Za-z0-9_+\-.]{1,2}", raw):
    s = m.group().decode("ascii")
    found.add(s)
# 只显示看起来像命令的
likely = sorted(s for s in found
                if s.isalpha() or s.isdigit()
                or s in ("++", "--", "GO", "OK", "ON", "1", "0", "S", "s", "R", "W"))
print("  " + "  ".join(repr(s) for s in likely[:120]))

# ---- 4) bytes 常量（通常是二进制命令）----
print("\n" + "=" * 74)
print("④ 疑似 bytes 常量（marshal 's'/'b' 类型，长度 1~8）")
print("=" * 74)
# marshal 里 bytes('s') 编码：0x73 + 4字节长度 + 数据；'z' 短ASCII：0x7a + 1字节长度
for m in re.finditer(rb"[\x73\x7a](.)", raw):
    typ = m.group()[0:1]
    if typ == b"s":
        ln_bytes = raw[m.start() + 1:m.start() + 5]
        if len(ln_bytes) < 4:
            continue
        ln = struct.unpack("<I", ln_bytes)[0]
        if 1 <= ln <= 8:
            start = m.start() + 5
            payload = raw[start:start + ln]
            if all(32 <= b < 127 or b in (0, 13, 10) for b in payload):
                print(f"  @{m.start():6d} len={ln}  {payload!r}")
    else:
        ln = raw[m.start() + 1]
        if 1 <= ln <= 8:
            start = m.start() + 2
            payload = raw[start:start + ln]
            if all(32 <= b < 127 or b in (0, 13, 10) for b in payload):
                print(f"  @{m.start():6d} len={ln}  {payload!r}")
