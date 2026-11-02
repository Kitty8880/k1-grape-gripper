# -*- coding: utf-8 -*-
"""
PyInstaller 解包 + 提取字符串常量，用于逆向 PS01Tool 的串口命令协议。

原理：
    PyInstaller onefile 的 exe 尾部有一个 CArchive，里面装着
    PYZ（压缩的 Python 字节码归档）和各种依赖。
    把 PYZ 解压出来 → marshal.loads 得到 code object →
    递归遍历 co_consts 就能拿到所有字符串常量（不需要反编译）。

这样能直接看到程序里写死的命令格式，比如 {"cmd": "start"}。
"""
import marshal
import os
import re
import struct
import sys
import io
import zlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PATH = r"D:\文档\deepseek-harness\default-workspace\ps01tool.exe"
OUTDIR = r"D:\文档\deepseek-harness\default-workspace\ps01_unpacked"

# PyInstaller 2.1+ 的 magic
PYINSTALLER_MAGIC = b"MEI\014\013\012\013\016"


def find_cookie(data: bytes):
    """在文件末尾搜索 PyInstaller cookie。"""
    # cookie 可能出现在最后 8KB 里
    tail_start = max(0, len(data) - 8192)
    idx = data.rfind(PYINSTALLER_MAGIC)
    if idx < 0:
        return None
    return idx


def parse_archive(data: bytes, cookie_pos: int):
    """
    解析 CArchive 的 TOC。

    旧版 cookie 布局（PyInstaller < 4.0）：
        magic(8) pkgLen(4) toc(4) tocLen(4) pyver(4)
    新版 cookie 布局（>= 4.0）：
        magic(8) pkgLen(4) toc(4) tocLen(4) pyver(4) pylibname(64) pylib(4) -> 88 字节
    """
    pos = cookie_pos
    magic = data[pos:pos + 8]
    pos += 8
    pkg_len, toc_off, toc_len, pyver = struct.unpack(">iiii", data[pos:pos + 16])
    pos += 16

    print(f"  magic         : {magic!r}")
    print(f"  package len   : {pkg_len}")
    print(f"  TOC offset    : {toc_off}")
    print(f"  TOC length    : {toc_len}")
    print(f"  python version: {pyver}")

    # archive 起点：pkg_len 包含 88 字节的 cookie 本身，
    # 所以 archive_start = 文件大小 - pkg_len（也等于 cookie_pos + 88 - pkg_len）
    # 验证：archive_start + pkg_len == 文件末尾
    archive_start = len(data) - pkg_len
    toc_start = archive_start + toc_off
    print(f"  验证 archive_start + pkg_len == 文件大小: "
          f"{archive_start + pkg_len == len(data)}")
    print(f"  验证 TOC 结束 == cookie 起点: "
          f"{toc_start + toc_len == cookie_pos}")

    entries = []
    p = toc_start
    toc_end = toc_start + toc_len
    # PyInstaller 6.x 的 TOC 条目格式（通过扫描 zlib 头实测确认）：
    #   entry_len (4)  —— 该条目总长度，含本字段
    #   data_pos  (4)  —— 相对 archive_start 的压缩数据偏移
    #   ulen      (4)  —— 解压后大小
    #   length    (4)  —— 压缩后大小
    #   cflag     (1)  —— 是否压缩（1=zlib）
    #   typecode  (1)  —— 's'脚本 / 'm'模块 / 'z'PYZ / 'b'二进制 / 'x'数据
    #   name      (变长)—— 文件名，0 结尾，剩余填充 0
    while p < toc_end:
        if p + 18 > len(data):
            break
        entry_len = struct.unpack(">i", data[p:p + 4])[0]
        if entry_len < 18 or p + entry_len > len(data):
            break
        data_pos = struct.unpack(">i", data[p + 4:p + 8])[0]
        ulen = struct.unpack(">i", data[p + 8:p + 12])[0]
        length = struct.unpack(">i", data[p + 12:p + 16])[0]
        cflag = data[p + 16]
        typecode = chr(data[p + 17])
        name = data[p + 18:p + entry_len].split(b"\x00")[0].decode("utf-8", "replace")
        entries.append({
            "name": name, "data_pos": data_pos, "len": length,
            "ulen": ulen, "cflag": cflag, "type": typecode,
        })
        p += entry_len

    return archive_start, entries


def extract_entry(data: bytes, archive_start: int, entry: dict) -> bytes:
    """按 data_pos/length 取出压缩数据，若 cflag=1 则 zlib 解压。"""
    start = archive_start + entry["data_pos"]
    raw = data[start:start + entry["len"]]
    if entry["cflag"]:
        try:
            return zlib.decompress(raw)
        except zlib.error:
            # 长度字段偶尔有偏差，放宽一个字节重试
            try:
                return zlib.decompressobj().decompress(raw)
            except zlib.error:
                return raw
    return raw


def walk_code(code, out: list, depth: int = 0):
    """递归收集 code object 里的字符串常量。"""
    if depth > 8:
        return
    for const in code.co_consts:
        if isinstance(const, str):
            if 2 <= len(const) <= 500:
                out.append(const)
        elif isinstance(const, type(code)):
            walk_code(const, out, depth + 1)


def load_pyz(blob: bytes) -> dict:
    """
    PYZ 结构：magic(4 'PYZ\\0') + pyc magic(4) + TOC 偏移(4) + TOC(marshal)

    PyInstaller 6.x 的 TOC 是 list：[(name, (typecode, offset, length)), ...]
    """
    result = {}
    if blob[:4] != b"PYZ\x00":
        print(f"  ⚠️ PYZ magic 不是 PYZ\\x00，实际：{blob[:8]!r}")
        return result

    toc_off = struct.unpack(">I", blob[8:12])[0]
    toc = marshal.loads(blob[toc_off:])
    print(f"  PYZ TOC 类型: {type(toc).__name__}")

    # 兼容 list 和 dict 两种形态
    if isinstance(toc, dict):
        items = list(toc.items())
    else:
        items = list(toc)

    print(f"  PYZ 内含 {len(items)} 个模块")

    for item in items:
        try:
            name, entry = item
            if isinstance(entry, (tuple, list)):
                typ, offset, length = entry[0], entry[1], entry[2]
            else:
                continue
        except (ValueError, TypeError):
            continue
        try:
            raw = zlib.decompress(blob[offset:offset + length])
        except zlib.error:
            continue
        try:
            code = marshal.loads(raw)
        except (ValueError, EOFError, TypeError):
            continue
        result[name] = code
    return result


def main() -> int:
    if not os.path.isfile(PATH):
        print(f"文件不存在：{PATH}")
        return 1

    print("=" * 78)
    print("PyInstaller 解包")
    print("=" * 78)
    with open(PATH, "rb") as fh:
        data = fh.read()
    print(f"文件大小: {len(data)} 字节")

    cookie = find_cookie(data)
    if cookie is None:
        print("❌ 没找到 PyInstaller magic，可能不是 PyInstaller 打的包")
        return 2
    print(f"找到 cookie @ {cookie}\n")

    archive_start, entries = parse_archive(data, cookie)
    print(f"\n  CArchive 起点: {archive_start}")
    print(f"  TOC 条目数   : {len(entries)}\n")

    os.makedirs(OUTDIR, exist_ok=True)

    # 找 PYZ 和主脚本（PyInstaller 6 里主脚本 typecode='s'，名字不带 .pyc）
    pyz_entry = None
    script_entries = []
    for e in entries:
        if e["type"] == "z" or e["name"].endswith(".pyz"):
            pyz_entry = e
        if e["type"] == "s" or e["name"].endswith(".pyc"):
            script_entries.append(e)

    print(f"PYZ entry : {pyz_entry['name'] if pyz_entry else '未找到'}")
    print(f"脚本条目  : {[(e['name'], e['type']) for e in script_entries]}")

    # ---- 提取字符串常量 ----
    # 注意：主程序和 PYZ 里的 .pyc 是 Python 3.9 的 marshal 格式，
    # 用当前 Python(3.14) 的 marshal.loads 可能失败。
    # 但 marshal 对 str 类型存的是"长度 + 明文 UTF-8 字节"，所以
    # 直接在解压后的原始字节里扫可打印字符串即可，不需要反序列化。
    all_strings: list = []
    ASCII_RE2 = re.compile(rb"[\x20-\x7e]{4,}")

    def harvest(raw: bytes, tag: str) -> int:
        found = 0
        for m in ASCII_RE2.finditer(raw):
            s = m.group()
            if len(s) <= 400:
                all_strings.append(s.decode("ascii", "replace"))
                found += 1
        print(f"  ✓ {tag}: {len(raw)} 字节 → {found} 条字符串")
        return found

    # 1) 主脚本
    for e in script_entries:
        try:
            raw = extract_entry(data, archive_start, e)
            # 去掉 pyc 头部（3.7+ 是 16 字节）
            harvest(raw[16:] if len(raw) > 16 else raw, f"脚本 {e['name']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ 提取脚本 {e['name']} 失败: {exc}")

    # 2) PYZ 里的所有模块
    if pyz_entry:
        try:
            blob = extract_entry(data, archive_start, pyz_entry)
            if blob[:4] == b"PYZ\x00":
                toc_off = struct.unpack(">I", blob[8:12])[0]
                toc = marshal.loads(blob[toc_off:])
                items = list(toc.items()) if isinstance(toc, dict) else list(toc)
                print(f"  PYZ TOC 条目: {len(items)}")
                ok = 0
                for item in items:
                    try:
                        name, entry = item
                        if isinstance(entry, (tuple, list)):
                            _typ, offset, length = entry[0], entry[1], entry[2]
                        else:
                            continue
                        raw = zlib.decompress(blob[offset:offset + length])
                    except (zlib.error, ValueError, TypeError):
                        continue
                    harvest(raw[16:] if len(raw) > 16 else raw, f"模块 {name}")
                    ok += 1
                print(f"  ✓ PYZ 中成功提取 {ok}/{len(items)} 个模块")
            else:
                print(f"  ⚠️ PYZ magic 异常：{blob[:8]!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ PYZ 处理失败: {exc}")

    print(f"\n共收集字符串常量 {len(all_strings)} 条")

    # 去重，保存
    uniq = sorted(set(all_strings))
    dump = os.path.join(OUTDIR, "strings.txt")
    with open(dump, "w", encoding="utf-8", errors="replace") as fh:
        fh.write("\n".join(uniq))
    print(f"全部字符串已保存：{dump}")

    # ---- 搜索命令相关 ----
    print("\n" + "=" * 78)
    print("搜索命令协议相关字符串")
    print("=" * 78)

    patterns = [
        r'"?cmd"?', r'command', r'start', r'stop', r'acq', r'stream',
        r'json', r'pd1|pd2|pd3', r'sensor', r'sample', r'rate', r'freq',
        r'"\{.*\}"', r'status',
    ]
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        hits = [s for s in uniq if rx.search(s) and len(s) < 200]
        # 排除明显的库/文件路径噪音
        hits = [h for h in hits
                if not re.search(r'\.(py|pyc|pyd|dll|so|png|jpg|dat|npz|csv|json\.|txt|gz)$', h, re.I)
                and '\\' not in h and '/' not in h]
        if not hits:
            continue
        print(f"\n--- /{pat}/ : {len(hits)} 条 ---")
        for h in sorted(hits, key=len)[:30]:
            print(f"  {h!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
