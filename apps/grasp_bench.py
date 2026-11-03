#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抓取成功率测试 —— 连续跑 N 次完整抓取，统计成功率。

和 vision_bench / asr_bench 的分工
----------------------------------
    vision_bench.py  测"看得准不准"（只拍照识别，不动机器）
    asr_bench.py     测"听得准不准"（只录音识别）
    grasp_bench.py   测"抓得成不成"（真的动机器人）

前两个可以无人值守连跑，这个不行 —— 每次抓完要有人把葡萄放回原位。
所以本脚本**每次之间会停下来等你**（可以设 --interval 自动跑）。

判定标准
--------
沿用 grasp_with_verify.py 的退出码和输出：
    退出码 0 → 成功
    退出码 1 → 失败（原因看输出里的 [DECIDE] / ❌ 行）

用法::

    # 最常用：10 次，每次之间等你回车
    sudo python3 apps/grasp_bench.py --object grape --times 10

    # 全自动（每次间隔 8 秒，自己记得把葡萄放回去）
    sudo python3 apps/grasp_bench.py --object grape --times 10 --auto --interval 8

    # 结果存文件
    sudo python3 apps/grasp_bench.py --object grape --times 10 --csv grasp.csv
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

GRASP_SCRIPT = os.path.join(_HERE, "grasp_with_verify.py")

# 从输出里抓关键行，用于归因
_PATTERNS = [
    ("servo_stall",      re.compile(r"舵机已堵转")),
    ("weak_grip",        re.compile(r"力不足")),
    ("over_force",       re.compile(r"超过硬上限|超力")),
    ("slipped",          re.compile(r"滑移|掉了下来|没夹牢")),
    ("not_in_gripper",   re.compile(r"不在两指之间|被挤出去")),
    ("converged",        re.compile(r"力达到目标")),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="抓取成功率测试")
    ap.add_argument("--object", default="grape",
                    help="物体类型（grape / egg / pingpong / unknown）")
    ap.add_argument("--times", type=int, default=10, help="抓取次数（默认 10）")
    ap.add_argument("--min-force", type=float, default=1.0,
                    help="最低可接受夹持力，传给 grasp_with_verify")
    ap.add_argument("--auto", action="store_true",
                    help="不等待回车，自动连续跑")
    ap.add_argument("--interval", type=float, default=8.0,
                    help="--auto 模式下每次之间的间隔秒数")
    ap.add_argument("--csv", default=None, help="结果另存为 CSV")
    ap.add_argument("--grasp-arg", action="append", default=[],
                    help="透传给 grasp_with_verify 的参数（用 --grasp-arg=--xxx=值）")
    args = ap.parse_args()

    if not os.path.exists(GRASP_SCRIPT):
        print(f"❌ 找不到 {GRASP_SCRIPT}")
        return 1

    cmd = [sys.executable, GRASP_SCRIPT, "--object", args.object,
           "--min-force", str(args.min_force), "--auto-lift"]
    cmd.extend(args.grasp_arg)

    print("=" * 70)
    print("抓取成功率测试")
    print("=" * 70)
    print(f"  物体　：{args.object}")
    print(f"  次数　：{args.times}")
    print(f"  最低力：{args.min_force} N")
    print(f"  模式　：{'自动连续' if args.auto else '每次等你回车'}")
    print()
    print("  ⚠️ 每次抓完请把物体放回原位、重新摆正")
    print(f"  ⚠️ 手上留着电源开关，出现异常立刻断电")
    print("=" * 70)

    rows = []
    try:
        for i in range(1, args.times + 1):
            print()
            print("─" * 70)
            print(f"第 {i} / {args.times} 次")
            print("─" * 70)

            if not args.auto:
                try:
                    input("把物体摆好，按回车开始（Ctrl+C 中止）...")
                except EOFError:
                    print("（stdin 不可读，直接开始）")
            else:
                if i > 1:
                    print(f"（{args.interval:.0f} 秒后开始下一次）")
                    time.sleep(args.interval)

            t0 = time.time()
            try:
                proc = subprocess.run(
                    cmd, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=180, text=True, encoding="utf-8",
                    errors="replace")
                out = proc.stdout or ""
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                print("  ❌ 超时（180 秒未完成）")
                rows.append({"i": i, "ok": 0, "rc": -1, "sec": 180,
                             "note": "超时"})
                continue

            elapsed = time.time() - t0
            # 只打印关键行，避免刷屏
            for line in out.splitlines():
                if any(k in line for k in ("✅", "❌", "⚠️", "力达到目标",
                                           "[DECIDE]", "[VERIFY]",
                                           "[RECHECK]", "夹牢", "滑移")):
                    print("   " + line.strip())

            ok = 1 if rc == 0 else 0
            note = ""
            if not ok:
                for name, pat in _PATTERNS:
                    if pat.search(out):
                        note = name
                        break
                if not note:
                    note = f"退出码 {rc}"

            print()
            print(f"  结果：{'✅ 成功' if ok else '❌ 失败'}"
                  f"   耗时 {elapsed:.1f}s"
                  + (f"   原因={note}" if note else ""))
            rows.append({"i": i, "ok": ok, "rc": rc,
                         "sec": elapsed, "note": note})

    except KeyboardInterrupt:
        print("\n\n用户中断，用已采集的数据出报告。")

    # ---------------- 报告 ----------------
    n = len(rows)
    if n == 0:
        print("没有有效数据。")
        return 1
    okn = sum(r["ok"] for r in rows)
    rate = okn / n
    secs = [r["sec"] for r in rows]

    print()
    print()
    print("=" * 70)
    print("测试结果")
    print("=" * 70)
    print(f"  ✅ 抓取成功率：{rate*100:5.1f}%   （{okn}/{n}）")
    print(f"  单次耗时　　：平均 {sum(secs)/len(secs):.1f}s　"
          f"最短 {min(secs):.1f}s　最长 {max(secs):.1f}s")

    fails = [r for r in rows if not r["ok"]]
    if fails:
        print()
        print(f"  失败明细（{len(fails)} 次）")
        print("  " + "-" * 60)
        for r in fails:
            print(f"    第 {r['i']:2} 次   原因={r['note'] or '未识别'}")
        from collections import Counter
        cat = Counter(r["note"] or "未识别" for r in fails)
        print()
        print("  失败原因归类")
        print("  " + "-" * 60)
        for k, v in cat.most_common():
            print(f"    {k:<20} {v} 次")

    print()
    print("  " + "-" * 60)
    if rate >= 0.90:
        g = "≥90%"
    elif rate >= 0.80:
        g = "≥80%"
    elif rate >= 0.70:
        g = "≥70%"
    else:
        g = "<70%"
    print(f"  成功率 {rate*100:.0f}% 落在「{g}」档")

    if args.csv:
        try:
            with open(args.csv, "w", encoding="utf-8-sig") as fh:
                fh.write("序号,成功,退出码,耗时s,失败原因\n")
                for r in rows:
                    fh.write(f"{r['i']},{r['ok']},{r['rc']},"
                             f"{r['sec']:.1f},{r['note']}\n")
            print(f"\n  明细已存：{args.csv}")
        except OSError as exc:
            print(f"\n  ⚠️ 写 CSV 失败：{exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())