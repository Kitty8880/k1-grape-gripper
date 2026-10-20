# 葡萄无损抓取机器人 · K1 MUSE Pi Pro

> **RISV-C 进迭时空赛道**参赛项目
> 在 RISC-V 单板计算机上，用 AI 视觉识别葡萄、以力闭环控制夹爪、实现**不破皮**的无损抓取。

[![Platform](https://img.shields.io/badge/Platform-RISC--V%20K1-blue)]()
[![OS](https://img.shields.io/badge/OS-Bianbu%20LXQt-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 一、项目简介

一颗巨峰葡萄约 12 g，直径 22 mm，**捏破它只需要 1~3 N 的力**。这个项目要解决的核心问题是：

> 如何用机械夹爪**稳定地夹住**一颗葡萄，同时**不把它捏破**？

答案不是"用力夹住"，而是"**轻轻碰住，靠摩擦吊起来**"：

- 目标夹持力只设 **0.8 N**（葡萄承受力的 1/3 左右）
- 双侧硅胶垫提供摩擦（μ≈1.0），0.8 N × 2 点 ≈ 1.6 N 摩擦力 ≫ 葡萄自重 0.12 N
- 用**速率限幅**逼近目标力，避免冲击过冲
- 用**光学力传感器**做闭环，实时感知接触力

多模态方面，本项目实现 **CV（视觉）+ ASR（语音）** 两种模态协同，均在本板本地推理。

---

## 二、当前进度

| 模块 | 状态 | 说明 |
|---|---|---|
| **K1 开发板环境** | ✅ 完成 | Bianbu OS / 内核 6.6.63 / 密码免登录 SSH + sudo |
| **PS01 光学力传感器** | ✅ **已通** | 逆向出隐藏命令字节 `0xD4`，稳定读到 ~135 Hz 数据 |
| **OV5647 MIPI 摄像头** | ✅ **已通** | 官方 `cam-test` + 交互模式，**13 fps 连续取帧** |
| **葡萄视觉识别** | 🔶 代码完成 | HSV + 凸包几何检测，待真实葡萄调参 |
| **夹爪力闭环控制** | 🔶 代码完成 | 状态机 + PI 控制 + 安全层，模拟验证 20/20 通过 |
| **舵机驱动** | ⏸ 待硬件 | 电源已到，等转接口 |
| **ASR 语音** | ⏸ 待硬件 | 麦克风未采购 |
| **机械臂结构** | ⏸ 未开始 | |

---

## 三、系统架构

```
┌──────────────────── K1 MUSE Pi Pro (Bianbu OS / RISC-V) ────────────────────┐
│                                                                             │
│   OV5647 (MIPI CSI)                  USB 麦克风 (待接入)                     │
│        │                                    │                               │
│        ▼                                    ▼                               │
│   ┌─────────────┐                    ┌─────────────┐                        │
│   │  视觉识别   │                    │  语音识别   │                        │
│   │ HSV+几何    │                    │  ASR        │                        │
│   └──────┬──────┘                    └──────┬──────┘                        │
│          │ 葡萄像素坐标                      │ 指令                          │
│          └──────────────┬───────────────────┘                              │
│                         ▼                                                   │
│              ┌─────────────────────┐                                        │
│              │   决策 / 运动解算    │                                        │
│              └──────────┬──────────┘                                        │
│                         │ 目标力 F* / 位置                                   │
│                         ▼                                                   │
│         ┌───────────────────────────────┐       ┌──────────────┐            │
│         │      夹爪力闭环控制器          │──────▶│  夹爪舵机     │            │
│         │  PI + 速率限幅 + 安全层        │       │  PWM 50Hz    │            │
│         └───────────────┬───────────────┘       └──────────────┘            │
│                         │ 力反馈                                            │
│                         ▼                                                   │
│              ┌─────────────────────┐                                        │
│              │ PS01 光学力传感器    │  ← AS7150 + LED + PD，UART 115200      │
│              └─────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、关键技术问题与解决（本项目的核心内容）

### 4.1 PS01 力传感器：隐藏的 `0xD4` 字节

**问题**：传感器完全不响应任何文本命令。尝试过 `gnst`、`gned`、各种波特率、各种接线组合，全部无输出。

**排查过程**：
1. 确认硬件接线正确（CH340 USB-TTL 回环测试 3/3 通过）
2. 确认 K1 的 UART 本身健康（`/dev/ttyS10` 回环 3/3 通过）
3. 确认 PADCONF 复用寄存器已正确配置（读 `0xd401e000` 基址）
4. 用 DMS 逻辑分析仪抓包 —— **仍然看不到有效数据**

**根因**：反编译厂商工具 `PS01Tool.exe` 后，从字节码里找到真相：

```python
def send_text_command(self, cmd):
    self.ser.write(bytes([212]))          # ← 0xD4！先发一个非打印字节
    self.ser.write((cmd + "\r\n").encode())
    time.sleep(0.02)
```

**命令实际是 `D4 67 6E 73 74 0D 0A`（`\xd4gnst\r\n`），不是 `gnst\r\n`。**

DMS 逻辑分析仪之所以没抓到，是因为它把不可打印的 `0xD4` 过滤掉了。

**结果**：修正后传感器立刻工作，稳定输出 ~135 Hz：

```
samples: 508, raw_lines: 1644, parse_errors: 0
PD1≈681k, PD2≈708k, PD3≈691k
```

> 详细记录见 [`docs/技术文档素材-持续积累.md`](docs/技术文档素材-持续积累.md)

### 4.2 OV5647 摄像头：不能直接用 OpenCV 打开

**问题**：`cv2.VideoCapture()` 打不开任何 video 节点；`v4l2-ctl --list-formats-ext` 返回空格式列表。

**原因**：K1 的 MIPI 摄像头走 SpacemiT 私有框架（`platform-cam`），
sensor → CSI → ISP → CPP 这条链路**必须由官方 SDK 建立**，之后 `/dev/video6~19` 才有数据。

**解决**：使用官方 `cam-test` 工具，并按 CSI 接口选择配置：

```bash
# 摄像头接在 CSI1 上
sudo cam-test /usr/share/camera_json/csi1_camera_detect.json
```

首次运行会自动探测传感器型号，生成可用配置：

```
I: ov5647_sensor.c(930): "detect sensor0 success"
I: cam_sensors_module.c(240): "detect ov5647_spm sensors in csi1: success"
I: update_json_file(732): "save json to .../csi1_camera_auto.json success"
```

### 4.3 从"拍一张"到"13 fps 视频流"

**问题**：`cam-test` 是**一次性**测试工具 —— 跑完就关闭 pipeline，
每采一帧要重建整条链路，耗时 0.85 秒（约 1.2 fps）。

**解决**：使用 `cam-test` 的**交互模式**（`auto_run: 0`），让 pipeline 常开：

| 按键 | 作用 |
|---|---|
| `s` | 开始推流（内部 15.6 fps） |
| `d` | 保存当前帧到 NV12 文件 |
| `c` | 停止推流 |
| `q` | 退出 |

Python 侧把 `cam-test` 作为子进程管理，循环发送 `d` 并读取新帧。

**关键坑**：必须用 **pty（伪终端）**启动子进程。因为 C 程序的 `printf` 输出到**管道**时是全缓冲（4KB 一块），日志会卡在缓冲区里读不到；输出到**终端**才是行缓冲。

**结果**：**13 fps**（77 ms/帧），提升 11 倍。

> 实现见 [`scripts/k1_camera_stream.py`](scripts/k1_camera_stream.py)

### 4.4 力闭环控制律

```
F*(目标力) ──▶ PI 控制器 ──▶ 期望速度 ──▶ 速率限幅 ──▶ 位置增量 ──▶ PWM
                  ▲                                                      │
                  └──── 力误差 e = F* - F ◀── 滑动平均滤波 ◀── 传感器 ◀──┘
```

**安全层**（独立于控制层，任一触发立即急停）：

| 保护 | 触发条件 | 动作 |
|---|---|---|
| 超力保护 | `F > F_max` | 立即反向松开 |
| 失稳保护 | 力持续上升但到不了目标 | 判定滑移，停机 |
| 通信超时 | 主循环卡顿 > 500 ms | 舵机卸力 |
| 双侧力差 | 左右传感器差值超阈值 | 判定未对中，停机 |

---

## 五、快速开始

### 5.1 硬件要求

| 类别 | 型号 | 关键参数 |
|---|---|---|
| 主控 | **K1 MUSE Pi Pro** | RISC-V 8 核，2 TOPS NPU，16 GB |
| 操作系统 | **Bianbu LXQt** | 内核 6.6.63 |
| 力传感器 | **PS01**（AS7150 光学力传感 AFE） | 3.3V TTL，UART 115200 8N1 |
| USB-TTL | CH340 模块 | 传感器与板子之间的桥接 |
| 摄像头 | **OV5647** MIPI 模组 | 500 万像素，2-lane CSI |
| 舵机 | MG996R 级别 | ≥ 15 kg·cm，**独立 5V 供电** |

### 5.2 传感器接线（已验证可用）

```
PS01 传感器           CH340 模块
  3V3   ────────────   3V3
  GND   ────────────   GND
  TXD   ────────────   RXD
  RXD   ────────────   TXD
                        │
                        └── USB ──▶ K1 的 USB 口（/dev/ttyUSB0）
```

> ⚠️ **注意**：不要用 K1 的 40Pin 3V3 给传感器供电（电流不足，实测无输出）。
> 一根 USB 线同时解决供电和通信，这也是比赛的推荐接线方式。

### 5.3 运行

```bash
# 1. 读力传感器
sudo python3 scripts/ps01_driver.py --port /dev/ttyUSB0

# 2. 摄像头初始化（首次运行，探测传感器并生成配置）
sudo cam-test /usr/share/camera_json/csi1_camera_detect.json

# 3. 摄像头连续取帧（13 fps）
sudo python3 scripts/k1_camera_stream.py --frames 30 --save-every 10

# 4. 葡萄识别
sudo python3 scripts/grape_detect.py --frames 10 --color purple \
    --save-raw /tmp/raw.jpg --save-ann /tmp/ann.jpg

# 5. 力闭环抓取（模拟模式，不需要硬件）
python3 apps/grip_test.py --sim
```

---

## 六、目录结构

```
grape-gripper/
├── README.md                    ← 本文件
├── LICENSE                      ← MIT
├── config.yaml                  ← 所有硬件/算法参数集中配置
│
├── src/grape_gripper/           ← 核心库
│   ├── config.py                ← 配置加载
│   ├── sensor.py                ← 力传感器采集 + 滤波 + 标定
│   ├── actuator.py              ← 舵机 PWM 驱动（sysfs / 模拟）
│   ├── gripper.py               ← ★ 夹爪力闭环状态机
│   ├── vision.py                ← 葡萄检测
│   └── safety.py                ← 安全监控层
│
├── apps/                        ← 可运行的应用
│   ├── grip_test.py             ← 单次抓取测试（支持 --sim 模拟）
│   ├── mg996r_test.py           ← 舵机测试与标定
│   └── mic_test.py              ← 麦克风检测与录音
│
├── scripts/                     ← ★ 板级驱动与工具
│   ├── ps01_driver.py           ← PS01 力传感器驱动（含 0xD4 命令）
│   ├── k1_camera_stream.py      ← 摄像头连续取帧（pty + 交互模式，13 fps）
│   ├── k1_camera_capture.py     ← 摄像头单帧采集
│   ├── grape_detect.py          ← 葡萄识别（HSV + 凸包几何）
│   ├── k1_check.sh              ← 板子环境自检
│   └── (其余为排查过程脚本)
│
└── docs/
    ├── 01-板子上手.md           ← 板子初始配置记录
    ├── PS01通信问题-求助描述.md  ← 问题描述（求助用）
    └── 技术文档素材-持续积累.md  ← ★ 技术文档素材库
```

---

## 七、路线图

- [x] **第 0 步**：板子上手、SSH、环境确认
- [x] **第 1 步**：PS01 力传感器通信打通（逆向出 `0xD4`）
- [x] **第 2 步**：OV5647 摄像头出图 + 13 fps 连续取流
- [ ] **第 3 步**：舵机转动 + 夹爪装配（等转接口到货）
- [ ] **第 4 步**：力闭环实机验证（先夹海绵 → 小番茄 → 葡萄）
- [ ] **第 5 步**：葡萄识别调参（等真实葡萄）
- [ ] **第 6 步**：ASR 语音指令
- [ ] **第 7 步**：机械臂结构 + 手眼标定
- [ ] **第 8 步**：系统联调 + 演示视频

---

## 八、致谢与参考

- [Bit-Brick K1 文档](https://docs.bit-brick.com/) — 摄像头开发指南、MIPI 相机使用说明
- [SpacemiT k1x-cam 源码](https://gitee.com/spacemit-buildroot/k1x-cam) — 官方摄像头 SDK
- [Bianbu Linux](https://gitee.com/bianbu-linux) — 系统源码

---

## 九、许可

本项目采用 [MIT License](LICENSE)。

第三方组件（如厂商工具 `PS01Tool.exe`）的逆向分析结果仅用于**互操作性**目的，
原始软件版权归其各自所有者。