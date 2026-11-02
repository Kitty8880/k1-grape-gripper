#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
让 SenseVoice ASR 跑在 K1 的 NPU 上 —— 不改官方代码的猴子补丁方案。

背景
----
官方 `spacemit_asr` 在 models/utils.py 里这样建 session：

    from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
    ...
    self.session = InferenceSession(model_file, sess_options=sess_opt,
                                    providers=EP_list)

而 `EP_list` 里只有 CPU（官方代码是为 x86/GPU 写的），所以 ASR 跑在 CPU 上。

NPU 的启用条件（读 spacemit_ort 源码得出）
------------------------------------------
    import onnxruntime        # 必须先
    import spacemit_ort       # 必须后
    ...
    InferenceSession(model, providers=["SpaceMITExecutionProvider", "CPUExecutionProvider"])

`spacemit_ort` 会包装 initialize_session，看到 providers 里有
"SpaceMITExecutionProvider" 就自动挂上 libspacemit_ep.so。

本脚本的做法
------------
在导入 spacemit_asr **之前**，替换掉
`onnxruntime.capi.onnxruntime_inference_collection.InferenceSession`，
让它在建 session 时自动把 NPU 插到 provider 列表最前面。

因为官方代码是 `from ... import InferenceSession`，只要我们在它 import
之前完成替换，它拿到的就是我们的版本 —— 不用改它一个字。

用法::

    python3 apps/asr_npu_test.py --once
    python3 apps/asr_npu_test.py --device 0 --rate 44100
"""
from __future__ import annotations

import argparse
import os
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

# ⚠️ 不能用 os.path.expanduser("~")：sudo 下 HOME 会变成 /root，
#    会去找 /root/spacemit-demo（不存在），且模型缓存也会指错地方。
#    用 __file__ 反推主目录，无论是否 sudo 都正确。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HOME = os.path.dirname(_ROOT)
if os.path.isdir(os.path.join(_HOME, "spacemit-demo")):
    os.environ["HOME"] = _HOME

_NLP_DIR = os.path.join(_HOME, "spacemit-demo", "examples", "NLP")
if not os.path.isdir(_NLP_DIR):
    print(f"❌ 找不到 {_NLP_DIR}")
    sys.exit(1)

# 官方示例的包（spacemit_asr / spacemit_audio）不在本项目里
sys.path.insert(0, _NLP_DIR)

# ---------------------------------------------------------------------------
# ★ 关键：顺序不能变，且补丁必须在 import spacemit_asr 之前装好
# ---------------------------------------------------------------------------
import onnxruntime                      # noqa: E402  必须在前
import spacemit_ort                     # noqa: E402  必须在后

import onnxruntime.capi.onnxruntime_inference_collection as _coll  # noqa: E402

NPU_EP = "SpaceMITExecutionProvider"
_OrigSession = _coll.InferenceSession


class _NpuFirstSession(_OrigSession):
    """把 NPU 插到 provider 列表最前面的 InferenceSession。"""

    def __init__(self, path_or_bytes, sess_options=None, providers=None,
                 provider_options=None, **kwargs):
        prov = list(providers) if providers else ["CPUExecutionProvider"]
        opts = list(provider_options) if provider_options else [{} for _ in prov]

        if NPU_EP not in prov:
            prov.insert(0, NPU_EP)
            opts.insert(0, {})

        super().__init__(path_or_bytes, sess_options, prov, opts, **kwargs)


_coll.InferenceSession = _NpuFirstSession
print(f"[补丁] InferenceSession 已替换，provider 列表将以 {NPU_EP} 开头")


def main() -> int:
    ap = argparse.ArgumentParser(description="ASR on NPU 测试")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--rate", type=int, default=44100)
    ap.add_argument("--max-time", type=int, default=5)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU（做对比）")
    args = ap.parse_args()

    if args.cpu:
        _coll.InferenceSession = _OrigSession
        print("[补丁] 已还原为原始 InferenceSession（强制 CPU）")

    print("=" * 70)
    print(f"ASR on K1 {'CPU（对照）' if args.cpu else 'NPU'} 测试")
    print("=" * 70)

    print("\n[1/2] 加载 ASR 模型...")
    t0 = time.time()
    try:
        from spacemit_asr import ASRModel
        asr = ASRModel()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 加载失败：{exc}")
        return 1
    print(f"      ✅ 用时 {time.time()-t0:.1f}s")

    # 看看模型实际用了哪些 provider
    try:
        sess = getattr(asr._model, "session", None)
        if sess is not None:
            print(f"      实际 provider：{sess.get_providers()}")
        else:
            print("      （拿不到 session 对象，运行时看推理耗时判断）")
    except Exception as exc:  # noqa: BLE001
        print(f"      （查 provider 失败：{exc}）")

    print("[2/2] 初始化录音...")
    try:
        from spacemit_audio import RecAudioVadThread
        rec = RecAudioVadThread(sld=1.0, max_time=args.max_time,
                                channels=1, rate=args.rate,
                                device_index=args.device,
                                trig_on=0.30, trig_off=0.35)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ {exc}")
        return 1
    print("      ✅")

    print()
    print("按回车说话，说完停顿 1 秒自动结束。Ctrl+C 退出")

    try:
        while True:
            try:
                input("\n按回车开始说话...")
            except EOFError:
                break
            print("🎤 请说话...")
            try:
                audio = rec.record_audio()
            except Exception as exc:  # noqa: BLE001
                print(f"❌ 录音失败：{exc}")
                continue
            if audio is None or (hasattr(audio, "size") and audio.size == 0):
                print("⚠️ 没录到有效音频")
                continue

            t0 = time.time()
            try:
                text = asr.generate(audio)
            except Exception as exc:  # noqa: BLE001
                print(f"❌ 推理失败：{exc}")
                print("   如果报 provider / 算子不支持，说明该模型不适合 NPU，")
                print("   用 --cpu 跑对照，我们就继续用 CPU 版本。")
                continue
            dt = time.time() - t0

            dur = (audio.size / 16000.0) if hasattr(audio, "size") else 0.0
            print()
            print(f"  🗣  识别：{text}")
            print(f"  ⏱  音频 {dur:.2f}s   推理 {dt:.2f}s   "
                  f"RTF {dt/dur if dur else 0:.2f}")

            if args.once:
                break
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        try:
            rec.cleanup()
        except Exception:  # noqa: BLE001
            pass
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())