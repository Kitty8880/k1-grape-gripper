#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摄像头实时预览（浏览器里看，不需要板子接显示器）。

为什么做这个
------------
2026-11-05 拍 31 张验证图时，结果是**全部不合格**，原因全在成像：

    · 清晰度（Laplacian 方差）最高只有 38.7，而边缘分析需要 ≥80
    · 葡萄偏在画面最上沿（偏心 42%）
    · 视角顺着夹爪方向，葡萄被传感器挡住（set_b 只检出 3,698 px）

**这些问题在按快门之前就能发现** —— 只要能看到画面。

板子没有显示器，但电脑有浏览器。所以做一个 MJPEG 网页流：

    电脑浏览器打开  http://192.168.137.125:8080/

画面上实时叠加 4 个东西，全部是为了"调好再拍"：

    ① 清晰度分数（大字，颜色随分数变化）
       ≥80 绿（够做边缘分析） / 50~80 黄 / <50 红
       —— 一边调相机距离，一边看这个数变大

    ② 中央取景框（60% 区域）
       —— 把葡萄放进框里，构图就合格

    ③ 葡萄检测圈 + 占比
       —— 直径和面积占比实时显示，调到 5~10% 最合适

    ④ 帧率
       —— 确认是连续流（15.6 fps）还是退回单张模式（约 1 fps）

右下角还有个「拍照」按钮，点一下存全分辨率原图到板子，
文件名带回显，方便下一步 scp。

用法::

    # 先停服务，否则摄像头被占
    sudo systemctl stop grape-gripper
    sudo python3 apps/camera_preview.py

    # 然后电脑浏览器打开 http://192.168.137.125:8080/

    # 换端口
    sudo python3 apps/camera_preview.py --port 8081

    # 只用单张模式（连续流失败时的兜底）
    sudo python3 apps/camera_preview.py --single-shot
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

import cv2
import numpy as np

#: 葡萄 HSV（与 scripts/grape_detect.py 的 darkpurple 一致）
GRAPE_HSV = ((120, 15, 15), (180, 255, 130))

#: 清晰度分级阈值（实测：本机最好约 50，历史 100% 识别的那张只有 27）
FOCUS_EDGE = 80.0     # 够做边缘分析
FOCUS_COLOR = 40.0    # 够做颜色检测

#: 输出流的分辨率（全 1920 太吃 CPU，且浏览器也没必要）
STREAM_W = 960
STREAM_H = 540


class FrameSource:
    """
    统一的取帧接口：优先连续流（15.6 fps），失败退回 cam-test（约 1 fps）。

    两种源的行为差异很大，所以这里把 fps 也暴露出去，
    让预览界面能提示用户"你现在是单张模式"。
    """

    def __init__(self, single_shot: bool = False):
        self.mode = "?"
        self.err = ""
        self._cam = None
        self._stream = None
        if not single_shot:
            try:
                from k1_camera_stream import K1CameraStream
                self._stream = K1CameraStream(use_sudo=False, verbose=False)
                self._stream.open()
                # 试读一帧确认可用
                ok = self._stream.read_frame(timeout=8.0) is not None
                if not ok:
                    raise RuntimeError("连续流打开成功但读不到帧")
                self.mode = "stream"
                print("  ✅ 连续流模式（约 15.6 fps）", flush=True)
                return
            except Exception as exc:  # noqa: BLE001
                self.err = str(exc)
                print(f"  ⚠️ 连续流失效：{exc}", flush=True)
                print("     退回单张模式（约 1 fps，仍可用于对焦和构图）",
                      flush=True)
                try:
                    self._stream.close()
                except Exception:  # noqa: BLE001
                    pass
                self._stream = None

        from k1_camera_capture import K1Camera
        self._cam = K1Camera(use_sudo=False)
        self.mode = "single"
        print("  ✅ 单张模式（cam-test，约 1 fps）", flush=True)

    def read(self):
        if self.mode == "stream":
            return self._stream.read_frame(timeout=3.0)
        return self._cam.capture()

    def close(self):
        for o in (self._stream, self._cam):
            try:
                if o is not None:
                    o.close()
            except Exception:  # noqa: BLE001
                pass


def analyze(bgr):
    """一帧的检测 + 质量分析。返回叠加所需的所有数字。"""
    h, w = bgr.shape[:2]

    # ⚠️ 清晰度必须算**全画面**，不能只算中央 40%。
    #    第一版只算中央，结果葡萄在画面上沿时中间是空桌面，
    #    Laplacian 只有 4.46 —— 明明是"画面里没有边缘"，却被读成"失焦"。
    #    为了省 CPU，先缩到 640 宽再算（Laplacian 对缩放不敏感，
    #    但绝对值和全分辨率不完全一致，所以下面同时给出葡萄区域的数）。
    g_small = cv2.cvtColor(
        cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY)
    focus = float(cv2.Laplacian(g_small, cv2.CV_64F).var())
    focus_grape = None

    # 葡萄检测（0.5 倍加速）
    small = cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(GRAPE_HSV[0], np.uint8),
                    np.array(GRAPE_HSV[1], np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    grape = None
    best = None
    for idx in range(1, n):
        a = int(stats[idx, cv2.CC_STAT_AREA])
        if a < 200:                     # 半分辨率下 200 px ≈ 全图 800 px
            continue
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0 or min(bw, bh) / max(bw, bh) < 0.55:
            continue
        if best is None or a > best[1]:
            best = (idx, a)
    if best is not None:
        idx, a = best
        cx, cy = cents[idx]
        # 半径：用面积换算（半分辨率下的等效圆半径）→ 再放大 2 倍
        r = float(np.sqrt(a / np.pi)) * 2.0
        grape = (cx * 2.0, cy * 2.0, r, a * 4.0 / (w * h))

        # 葡萄区域的清晰度 —— 这才是"葡萄拍清楚了吗"的直接答案
        gx, gy, gr, _ = grape
        x0 = max(0, int(gx - gr))
        x1 = min(w, int(gx + gr))
        y0 = max(0, int(gy - gr))
        y1 = min(h, int(gy + gr))
        if x1 - x0 > 20 and y1 - y0 > 20:
            crop = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            crop = cv2.resize(crop, (240, 240),
                              interpolation=cv2.INTER_AREA)
            focus_grape = float(cv2.Laplacian(crop, cv2.CV_64F).var())
    return focus, focus_grape, grape


def overlay(bgr, focus, focus_grape, grape, fps, mode):
    """在画面上画出指导性的叠加信息。"""
    h, w = bgr.shape[:2]
    vis = bgr.copy()

    # ---- ① 中央取景框（60%）----
    x0, x1 = int(w * 0.2), int(w * 0.8)
    y0, y1 = int(h * 0.2), int(h * 0.8)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 255), 3)
    cv2.drawMarker(vis, (w // 2, h // 2), (0, 200, 255),
                   cv2.MARKER_CROSS, 60, 2)

    # ---- ② 葡萄圈 + 占比 ----
    if grape is not None:
        gx, gy, gr, frac = grape
        in_box = x0 <= gx <= x1 and y0 <= gy <= y1
        col = (0, 255, 0) if (in_box and 0.05 <= frac <= 0.10) else (0, 0, 255)
        cv2.circle(vis, (int(gx), int(gy)), int(gr), col, 3)
        cv2.circle(vis, (int(gx), int(gy)), 4, col, -1)
        txt = f"grape {frac*100:.1f}%  d={int(gr*2)}px"
        cv2.putText(vis, txt, (int(gx) - 120, int(gy) - int(gr) - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    else:
        cv2.putText(vis, "no grape", (w // 2 - 90, h // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # ---- ③ 清晰度大字 ----
    if focus >= FOCUS_EDGE:
        fcol, fjudge = (0, 255, 0), "OK for edge"
    elif focus >= FOCUS_COLOR:
        fcol, fjudge = (0, 215, 255), "color only"
    else:
        fcol, fjudge = (0, 0, 255), "TOO BLURRY"
    ftext = f"FOCUS {focus:5.1f}"
    if focus_grape is not None:
        ftext += f"   grape {focus_grape:5.1f}"
    ftext += f"  {fjudge}"
    cv2.putText(vis, ftext, (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, fcol, 3)
    # 进度条：0~120 映射
    bar_w = int(min(focus / 120.0, 1.0) * 500)
    cv2.rectangle(vis, (30, 95), (530, 120), (60, 60, 60), -1)
    cv2.rectangle(vis, (30, 95), (30 + bar_w, 120), fcol, -1)
    # 80 的刻度线
    x80 = 30 + int(80 / 120.0 * 500)
    cv2.line(vis, (x80, 90), (x80, 125), (255, 255, 255), 2)

    # ---- ④ 帧率 / 模式 ----
    mstr = "STREAM" if mode == "stream" else "SINGLE(1fps)"
    cv2.putText(vis, f"{fps:4.1f} fps   {mstr}", (30, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # ---- 提示 ----
    hint = "put grape inside box, aim for 5-10%"
    cv2.putText(vis, hint, (30, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return vis


class State:
    """预览线程与 HTTP 线程之间共享的状态。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None          # 最新一帧（已叠加、已缩放）
        self.snap_request = False
        self.snap_msg = ""
        self.last_full = None     # 最近一次全分辨率原图（用于拍照）
        self.focus = 0.0
        self.mode = "?"
        self.fps = 0.0


def capture_loop(src: FrameSource, st: State, save_dir: str):
    t_prev = time.time()
    fps = 0.0
    while True:
        try:
            bgr = src.read()
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ 取帧失败：{exc}", flush=True)
            time.sleep(1.0)
            continue
        if bgr is None:
            time.sleep(0.2)
            continue

        now = time.time()
        dt = now - t_prev
        t_prev = now
        if dt > 0:
            fps = 0.8 * fps + 0.2 * (1.0 / dt) if fps else 1.0 / dt

        # 分析每 2 帧做一次（分析和 JPEG 编码是 CPU 大头，
        # 1080p 全量分析会把帧率压到 4 fps，隔帧后约 8 fps，
        # 对"调焦/构图"这个用途完全够）
        nonlocal_state = capture_loop
        if not hasattr(capture_loop, "n"):
            capture_loop.n = 0
            capture_loop.last = (0.0, None, None)
        capture_loop.n += 1
        if capture_loop.n % 2 == 1:
            try:
                focus, fg, grape = analyze(bgr)
                capture_loop.last = (focus, fg, grape)
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠️ 分析失败：{exc}", flush=True)
        focus, fg, grape = capture_loop.last

        vis = overlay(bgr, focus, fg, grape, fps, src.mode)
        small = cv2.resize(vis, (STREAM_W, STREAM_H),
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        with st.lock:
            if ok:
                st.jpeg = buf.tobytes()
            st.focus = focus
            st.mode = src.mode
            st.fps = fps
            st.last_full = bgr
            # 处理拍照请求
            if st.snap_request:
                st.snap_request = False
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    name = time.strftime("snap_%m%d_%H%M%S") + ".jpg"
                    path = os.path.join(save_dir, name)
                    okf, bf = cv2.imencode(".jpg", bgr,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if okf:
                        bf.tofile(path)
                        st.snap_msg = f"saved {name} (focus {focus:.1f})"
                        print(f"  📷 {path}  focus={focus:.1f}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    st.snap_msg = f"save failed: {exc}"


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>K1 Camera Preview</title>
<style>
 body{margin:0;background:#111;color:#ddd;font-family:monospace;text-align:center}
 h3{margin:8px;font-weight:normal;font-size:14px}
 img{max-width:98vw;border:1px solid #333}
 button{font-size:18px;padding:10px 28px;margin:10px;cursor:pointer;
        background:#2a7;color:#fff;border:0;border-radius:6px}
 button:hover{background:#3b8}
 #msg{color:#8f8;font-size:13px;min-height:18px}
</style></head><body>
<h3>K1 MUSE Pi Pro &mdash; 实时预览（FOCUS &ge; 80 才够做边缘分析）</h3>
<img src="/stream" width="960">
<div><button onclick="snap()">📷 拍照（存全分辨率）</button></div>
<div id="msg"></div>
<script>
async function snap(){
  const r = await fetch('/snap');
  document.getElementById('msg').textContent = await r.text();
}
setInterval(async()=>{
  try{const r=await fetch('/snapmsg');document.getElementById('msg').textContent=await r.text();}
  catch(e){}
}, 1500);
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    st: State = None
    save_dir: str = ""

    def log_message(self, *a):        # 静默，避免刷屏
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with self.st.lock:
                        jpg = self.st.jpeg
                    if jpg:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                return

        elif self.path == "/snap":
            with self.st.lock:
                self.st.snap_request = True
                self.st.snap_msg = "拍照中..."
            body = f"已请求拍照，存到 {self.save_dir}".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/snapmsg":
            with self.st.lock:
                msg = self.st.snap_msg or f"focus {self.st.focus:.1f}  " \
                                          f"{self.st.fps:.1f}fps  {self.st.mode}"
            body = msg.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/stats":
            with self.st.lock:
                s = f"focus={self.st.focus:.2f} fps={self.st.fps:.2f} " \
                    f"mode={self.st.mode}"
            body = s.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_error(404)


def main() -> int:
    ap = argparse.ArgumentParser(description="摄像头实时预览（浏览器 MJPEG）")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--single-shot", action="store_true",
                    help="不用连续流，退回 cam-test（约 1 fps）")
    ap.add_argument("--save-dir", default="debug/live")
    args = ap.parse_args()

    print("=" * 68)
    print("摄像头实时预览")
    print("=" * 68)

    try:
        src = FrameSource(single_shot=args.single_shot)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 摄像头打不开：{exc}")
        print("   排查：sudo systemctl stop grape-gripper")
        print("         sudo pkill -f cam-test")
        return 1

    st = State()
    st.mode = src.mode
    save_dir = os.path.join(_ROOT, args.save_dir)

    th = threading.Thread(target=capture_loop,
                          args=(src, st, save_dir), daemon=True)
    th.start()

    Handler.st = st
    Handler.save_dir = save_dir
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)

    ip = "192.168.137.125"
    print()
    print("=" * 68)
    print(f"  电脑浏览器打开：  http://{ip}:{args.port}/")
    print("=" * 68)
    print(f"  模式　：{'连续流 15.6 fps' if src.mode == 'stream' else '单张 约 1 fps'}")
    print(f"  拍照存：{save_dir}")
    print()
    print("  看什么：")
    print(f"    · FOCUS 数字 ≥ {FOCUS_EDGE:.0f} 才能做边缘分析（现在需要）")
    print("    · 把葡萄放进黄色框，占比 5~10%")
    print("    · 一边调相机距离，一边看 FOCUS 变大")
    print()
    print("  Ctrl+C 退出")
    print()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        srv.shutdown()
        src.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())