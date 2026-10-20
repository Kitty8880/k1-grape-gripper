# -*- coding: utf-8 -*-
"""
诊断 marshal 解析失败原因，并尝试反编译 PS01Tool 的 send_text_command。

Python 3.9 打包的 pyc 用 Python 3.14 的 marshal 可能读不了，
先确认是不是版本问题，再决定是否手写解析器。
"""
import marshal
import struct
import sys
import io
import zlib
import dis

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PATH = r"D:\文档\deepseek-harness\default-workspace\ps01tool.exe"
data = open(PATH, "rb").read()

pkg_len = 107504659
toc_off = 107460427
toc_len = 44144
archive_start = len(data) - pkg_len
toc_start = archive_start + toc_off

p = toc_start
main_entry = None
while p < toc_start + toc_len:
    entry_len = struct.unpack(">i", data[p:p + 4])[0]
    if entry_len < 18:
        break
    name = data[p + 18:p + entry_len].split(b"\x00")[0].decode("utf-8", "replace")
    if name == "PS01Tool":
        main_entry = (
            struct.unpack(">i", data[p + 4:p + 8])[0],
            struct.unpack(">i", data[p + 12:p + 16])[0],
            data[p + 16],
        )
        break
    p += entry_len

data_pos, length, cflag = main_entry
raw = data[archive_start + data_pos: archive_start + data_pos + length]
if cflag:
    raw = zlib.decompress(raw)

print(f"主脚本大小 {len(raw)} 字节")
print(f"前 16 字节（pyc 头）: {raw[:16].hex(' ')}")
print(f"  magic: {raw[:4].hex(' ')}")
if len(raw) >= 16:
    flags, mtime, srcsize = struct.unpack("<III", raw[4:16])
    print(f"  flags={flags} mtime={mtime} srcsize={srcsize}")

print("\n" + "=" * 70)
print("尝试 marshal.loads（跳过 16 字节 pyc 头）")
print("=" * 70)
for header in (16, 12):
    try:
        code = marshal.loads(raw[header:])
        print(f"✅ header={header} 解析成功！")
        print(f"   co_name={code.co_name}")
        print(f"   co_filename={code.co_filename}")
        print(f"   常量数={len(code.co_consts)} 名字数={len(code.co_names)}")
        break
    except Exception as exc:  # noqa: BLE001
        print(f"❌ header={header} 失败: {type(exc).__name__}: {exc}")

print("\n" + "=" * 70)
print("手动检查 marshal 首字节类型")
print("=" * 70)
body = raw[16:]
if body:
    print(f"首字节: 0x{body[0]:02X} ({chr(body[0]) if 32 <= body[0] < 127 else '?'})")
    # Python 3.9 marshal type codes
    TYPES = {
        0x63: "TYPE_CODE ('a' in older, 'c'=code)",
        0x73: "TYPE_STRING (bytes, 4字节长度)",
        0x74: "TYPE_INTERNED",
        0x75: "TYPE_UNICODE ('u')",
        0x61: "TYPE_ASCII ('a')",
        0x41: "TYPE_ASCII_INTERNED ('A')",
        0x7A: "TYPE_SHORT_ASCII ('z')",
        0x5A: "TYPE_SHORT_ASCII_INTERNED ('Z')",
        0x28: "TYPE_TUPLE ('(')",
        0x29: "TYPE_SMALL_TUPLE (')')",
        0x69: "TYPE_INT ('i')",
    }
    print(f"类型猜测: {TYPES.get(body[0], '未知')}")
