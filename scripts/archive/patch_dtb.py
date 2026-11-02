#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给 r_uart0 节点补上 pinctrl-0，让它自己应用 r_uart0 的引脚复用。

背景：
    dtb 里 pinctrl_rcpu_grp（GPIO47/48 → r_uart0 txd/rxd）只挂在 rcpu
    （remoteproc）节点下，而 remoteproc 驱动没有把它应用到硬件，
    导致 GPIO47/48 仍是普通 GPIO，/dev/ttyS10 收不到任何数据。

    修复：给 r_uart0@c0881000 节点本身加
        pinctrl-names = "default";
        pinctrl-0 = <0x2d>;      (pinctrl_rcpu_grp 的 phandle)

本脚本只做「生成新 dts + 编译 + 校验」，不替换系统文件。
"""
import os
import shutil
import subprocess
import sys

SRC_DTS = "/tmp/mp.dts"
NEW_DTS = "/tmp/mp_patched.dts"
NEW_DTB = "/tmp/mp_patched.dtb"
TARGET_NODE = "r_uart0@c0881000"
PHANDLE = "0x2d"          # pinctrl_rcpu_grp


def main() -> int:
    if not os.path.isfile(SRC_DTS):
        print(f"❌ {SRC_DTS} 不存在")
        return 1

    with open(SRC_DTS, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()

    out = []
    in_node = False
    depth = 0
    inserted = False
    indent = "\t\t\t"          # 与其它属性保持一致

    for ln in lines:
        if not in_node and TARGET_NODE in ln and "{" in ln:
            in_node = True
            depth = 1
            out.append(ln)
            continue

        if in_node:
            out.append(ln)
            depth += ln.count("{") - ln.count("}")
            # 在 status 行后插入 pinctrl
            if (not inserted) and ("status" in ln) and ("okay" in ln):
                out.append(f'{indent}pinctrl-names = "default";')
                out.append(f"{indent}pinctrl-0 = <{PHANDLE}>;")
                inserted = True
                print(f"✅ 已在 {TARGET_NODE} 的 status 后插入 pinctrl 属性")
            if depth <= 0:
                in_node = False
        else:
            out.append(ln)

    if not inserted:
        print("❌ 没有插入成功：可能没找到节点或 status 行")
        return 2

    with open(NEW_DTS, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"✅ 新 dts 已生成：{NEW_DTS}")

    # 编译
    r = subprocess.run(
        ["dtc", "-I", "dts", "-O", "dtb", "-o", NEW_DTB, NEW_DTS],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("❌ dtc 编译失败：")
        print(r.stderr[-2000:])
        return 3
    print(f"✅ 编译成功：{NEW_DTB}  ({os.path.getsize(NEW_DTB)} 字节)")

    # 校验：反编译回来确认属性在
    r = subprocess.run(
        ["dtc", "-I", "dtb", "-O", "dts", "-o", "/tmp/mp_verify.dts", NEW_DTB],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        with open("/tmp/mp_verify.dts", "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        idx = text.find(TARGET_NODE)
        seg = text[idx:idx + 700] if idx >= 0 else ""
        if "pinctrl-0" in seg:
            print("✅ 校验通过：编译后的 dtb 里 r_uart0 已带 pinctrl-0")
            print("\n--- r_uart0 节点片段 ---")
            for line in seg.splitlines()[:16]:
                print(line)
        else:
            print("⚠️ 校验：反编译结果里没找到 pinctrl-0")

    print()
    print("下一步（需要你确认后执行）：")
    print("  1. 备份 /boot/spacemit/6.6.63/k1-x_MUSE-Pi-Pro.dtb")
    print("  2. 替换为 /tmp/mp_patched.dtb")
    print("  3. 重启")
    return 0


if __name__ == "__main__":
    sys.exit(main())
