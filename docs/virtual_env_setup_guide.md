# Open Duck Mini — 虚拟环境运行指南

本文档介绍如何在 Python 虚拟环境中安装并运行 Open Duck Mini 项目（MuJoCo 仿真）。

---

## 前置条件

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11、Ubuntu 20.04+ 或 macOS |
| Python | 3.10 – 3.11（推荐 3.10，部分依赖对 3.12+ 兼容性尚不完善） |
| Git | 已安装 |
| GPU（可选） | 训练 RL 策略时建议有 CUDA GPU；仅运行仿真查看无需 GPU |

---

## 1. 克隆仓库

```bash
git clone https://github.com/apirrone/Open_Duck_Mini.git
cd Open_Duck_Mini
```

如果你已经有了本地仓库，直接 `cd` 进入即可。

---

## 2. 创建并激活虚拟环境

### 方式 A：使用 venv（Python 内置）

```bash
# 创建虚拟环境
python -m venv .venv

# 激活（Windows PowerShell）
.venv\Scripts\Activate.ps1

# 激活（Windows CMD）
.venv\Scripts\activate.bat

# 激活（Linux / macOS）
source .venv/bin/activate
```

### 方式 B：使用 Conda

```bash
conda create -n openduck python=3.10 -y
conda activate openduck
```

---

## 3. 安装项目及依赖

项目使用 `setup.cfg` + `pyproject.toml` 管理依赖。安装方式如下：

```bash
# 安装基础包（以可编辑模式安装，方便开发）
pip install -e ".[all]"
```

这会安装 `setup.cfg` 中 `[options.extras_require] > all` 列出的全部依赖，包括：

- `mujoco==3.1.5` — 物理仿真引擎
- `mujoco-python-viewer==0.1.4` — MuJoCo 可视化查看器
- `gymnasium[mujoco]==0.29.1` — Gymnasium 强化学习环境
- `stable-baselines3[extra]==2.3.2` — RL 训练框架
- `sb3_contrib==2.3.0` — SB3 额外算法（TQC 等）
- `placo==0.5.0` — 运动学步态引擎
- `onshape-to-robot==0.3.25` — CAD 导出工具
- `imitation==1.0.0` — 模仿学习库
- `h5py==3.11.0` — HDF5 数据读写
- 等

> **注意**：如果只是在真实机器人上运行，可以用 `pip install -e ".[robot]"` 安装精简依赖。

---

## 4. 验证安装

```bash
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
python -c "import mini_bdx; print('mini_bdx imported successfully')"
```

两行都无报错即安装成功。

---

## 5. 运行 MuJoCo 仿真 Demo

### 5.1 Placo 步态引擎仿真（bdx 版本）

```bash
cd experiments/placo
python placo_walk_engine_mujoco.py
```

运行后会弹出 MuJoCo 可视化窗口，显示机器人站立。按键盘方向键 ↑ 可让机器人前进。

### 5.2 使用预训练 ONNX 策略运行仿真（推荐体验）

项目提供了预训练的 ONNX 策略文件。先下载策略文件（参见 README 中的链接），然后运行：

```bash
cd experiments/v2
python onnx_AWD_mujoco_motor_control.py -o /path/to/BEST_WALK_ONNX_2.onnx
```

添加 `-k` 参数可以用键盘控制机器人方向（需要 pygame）：

```bash
python onnx_AWD_mujoco_motor_control.py -o /path/to/BEST_WALK_ONNX_2.onnx -k
```

键盘映射（AZERTY 布局）：
- `Z` / `S` — 前进 / 后退
- `Q` / `D` — 左移 / 右移
- `A` / `E` — 左转 / 右转

> **注意**：此脚本还依赖 `mini_bdx_runtime` 包，需要额外克隆并安装
> [Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) 仓库：
>
> ```bash
> git clone https://github.com/apirrone/Open_Duck_Mini_Runtime.git
> cd Open_Duck_Mini_Runtime
> pip install -e .
> ```

### 5.3 修改 scene.xml 路径

部分实验脚本中 `scene.xml` 的路径是硬编码的绝对路径（如 `/home/antoine/...`）。运行前需将其改为你本地的正确路径。Open Duck Mini v2 的模型文件位于：

```
mini_bdx/robots/open_duck_mini_v2/scene.xml
```

旧版 BDX 模型文件位于：

```
mini_bdx/robots/bdx/scene.xml
```

---

## 6. 训练自己的 RL 策略（进阶）

项目目前推荐使用 [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground)（基于 MuJoCo Playground）来训练策略。

### 6.1 使用本仓库中的旧版训练脚本

```bash
cd experiments/RL/new
python train.py -a SAC -d cuda
```

支持的算法：`SAC`、`TD3`、`A2C`、`TQC`、`PPO`。

如果没有 GPU，可以指定 CPU：

```bash
python train.py -a PPO -d cpu
```

### 6.2 使用 Open Duck Playground（推荐）

```bash
git clone https://github.com/apirrone/Open_Duck_Playground.git
cd Open_Duck_Playground
pip install -e .
```

详细训练流程参见 [sim2real 文档](sim2real.md) 和 Playground 仓库的 README。

---

## 7. 常见问题

### Q: `pip install -e ".[all]"` 报错怎么办？

- 确认 Python 版本为 3.10 或 3.11
- 尝试先升级 pip：`pip install --upgrade pip setuptools wheel`
- 如果 `placo` 安装失败，可能需要系统级依赖（如 `libeigen3-dev`），Linux 下执行：
  ```bash
  sudo apt install libeigen3-dev
  ```

### Q: MuJoCo 窗口打不开 / 渲染异常？

- 确认显卡驱动已安装
- 在远程服务器上需要配置虚拟显示（如 `xvfb`）或使用 `EGL` 渲染后端：
  ```bash
  export MUJOCO_GL=egl
  ```

### Q: Windows 上 `placo` 安装失败？

`placo` 目前主要支持 Linux。如果仅想在 Windows 上运行 ONNX 策略仿真，可以跳过 placo，手动安装其他依赖：

```bash
pip install mujoco==3.1.5 mujoco-python-viewer==0.1.4 numpy scipy onnxruntime pygame
```

### Q: `mini_bdx_runtime` 是什么？

这是在真实机器人上运行策略的运行时库，也被部分仿真脚本引用（如 ONNX 推理工具）。仓库地址：https://github.com/apirrone/Open_Duck_Mini_Runtime

---

## 8. 项目结构速览

```
Open_Duck_Mini/
├── mini_bdx/                  # 核心 Python 包
│   ├── mini_bdx/
│   │   ├── placo_walk_engine/ # Placo 步态引擎
│   │   ├── utils/             # 工具函数（MuJoCo、RL、控制器等）
│   │   └── old_walk_engine/   # 旧版步态引擎
│   └── robots/
│       ├── bdx/               # 旧版 BDX 机器人模型
│       │   └── scene.xml
│       └── open_duck_mini_v2/ # Open Duck Mini v2 机器人模型
│           └── scene.xml
├── experiments/               # 各类实验脚本
│   ├── v2/                    # v2 相关实验（ONNX 推理等）
│   ├── mujoco/                # MuJoCo 仿真实验
│   ├── placo/                 # Placo 步态相关实验
│   └── RL/                    # 强化学习训练脚本
├── docs/                      # 文档
├── setup.cfg                  # 依赖与包配置
├── pyproject.toml             # 构建系统配置
└── README.md
```

---

## 相关资源

- [Open Duck Mini GitHub](https://github.com/apirrone/Open_Duck_Mini)
- [Open Duck Playground（训练框架）](https://github.com/apirrone/Open_Duck_Playground)
- [Open Duck Mini Runtime（机器人运行时）](https://github.com/apirrone/Open_Duck_Mini_Runtime)
- [参考动作生成器](https://github.com/apirrone/Open_Duck_reference_motion_generator)
- [Discord 社区](https://discord.gg/UtJZsgfQGe)
