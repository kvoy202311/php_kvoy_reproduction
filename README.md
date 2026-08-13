# PHP-Kvoy Reproduction

基于 Isaac Lab 的 ELF3 机器人动作跟踪与攀爬专家策略训练工程。当前主要任务是让一个策略同时学习同一技能的多个 NPZ 动作片段，并在 0.65 m 平台上完成攀爬和最终站立。

## 环境与安装

需要已安装 Isaac Sim 4.5、Isaac Lab 2.1 和 Python 3.10，并使用包含 Isaac Lab 的 conda 环境（示例环境名为 `mimic`）。

```bash
conda activate mimic
cd /home/kvoy/Desktop/php_kvoy_reproduction
python -m pip install -e source/php_kvoy_reproduction
```

本地动作数据默认位于：

```text
data/processed_motions/elf3/climb_50hz/
```

目录训练会读取其中全部 `.npz` 文件（当前为同一攀爬技能的 4 个动作）。请确认动作数据和 ELF3 资产已经存在；数据文件较大时不会随代码仓库自动获得。

## 任务

| 任务 | 用途 |
| --- | --- |
| `Tracking-Climb-ELF3-v0` | 0.65 m 平台攀爬专家训练与评估 |
| `Tracking-Flat-ELF3-v0` | 平地动作跟踪 |

## 训练

### 多动作（推荐）

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Climb-ELF3-v0 \
  --motion_dir /home/kvoy/Desktop/php_kvoy_reproduction/data/processed_motions/elf3/climb_50hz \
  --num_envs 2048 \
  --max_iterations 100000 \
  --logger tensorboard \
  --run_name elf3_climb \
  --device cuda:0 \
  --headless
```

### 单动作

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Climb-ELF3-v0 \
  --motion_file /path/to/one_motion.npz \
  --num_envs 2048 \
  --max_iterations 100000 \
  --logger tensorboard \
  --run_name elf3_climb_single \
  --device cuda:0 \
  --headless
```

训练时动作来源三选一：`--motion_file`（单个本地 NPZ）、`--motion_dir`（目录中全部 NPZ）或 `--registry_name`（W&B 动作 artifact）。本工程推荐使用本地 `--motion_dir`。

常用训练参数：

```text
--task TASK                 任务注册名
--motion_file FILE          单个 NPZ；不能与另外两个动作来源同时使用
--motion_dir DIR            多个 NPZ 的目录；不能与另外两个动作来源同时使用
--registry_name NAME        W&B registry artifact（可选）
--num_envs N                并行环境数；4090 可先从 2048 开始
--max_iterations N          PPO 训练迭代数
--seed N                    随机种子
--experiment_name NAME      日志根目录名，默认使用任务配置
--run_name NAME             本次运行的目录后缀
--logger {tensorboard,wandb,neptune}
--log_project_name NAME     W&B/Neptune 项目名
--device {cuda:0,cpu}       仿真和训练设备
--headless                  无界面训练
--video                     训练期间录制视频
--video_length N            视频长度（步）
--video_interval N          录制间隔（步）
--resume / --no-resume      是否恢复 checkpoint（完整恢复优化器等状态）
--warm_start                只加载策略、价值网络和观测归一化参数，重新开始优化器/采样器状态
--load_run NAME             要加载的日志运行目录
--checkpoint FILE           指定 checkpoint 文件名
```

修改奖励函数或环境逻辑后，建议重新训练；若只想复用已有策略参数，使用 `--warm_start`，不要把旧实验当作严格续训。

训练日志默认写入 `logs/rsl_rl/<experiment_name>/<时间>_<run_name>/`。

## 播放策略

`--load_run` 是日志运行目录名，`--checkpoint` 是其中的模型文件名，例如 `model_100000.pt`。

### 播放全部动作（从第 0 帧开始）

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Climb-ELF3-v0 \
  --motion_dir /home/kvoy/Desktop/php_kvoy_reproduction/data/processed_motions/elf3/climb_50hz \
  --playback_mode full_clip \
  --load_run 2026-08-12_某次运行 \
  --checkpoint model_100000.pt \
  --num_envs 4 \
  --device cuda:0
```

### 固定播放第 `motion_id` 个动作

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Climb-ELF3-v0 \
  --motion_dir /home/kvoy/Desktop/php_kvoy_reproduction/data/processed_motions/elf3/climb_50hz \
  --playback_mode fixed_clip \
  --motion_id 0 \
  --load_run 2026-08-12_某次运行 \
  --checkpoint model_100000.pt \
  --num_envs 1 \
  --device cuda:0
```

播放模式：`training` 保持训练配置；`full_clip` 从第 0 帧按顺序播放全部动作；`fixed_clip` 从第 0 帧固定播放一个动作。`--free_camera` 使用世界坐标相机，可以在 Isaac Sim 窗口中手动拖动视角：

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Climb-ELF3-v0 \
  --motion_dir /home/kvoy/Desktop/php_kvoy_reproduction/data/processed_motions/elf3/climb_50hz \
  --playback_mode full_clip \
  --free_camera \
  --load_run 2026-08-12_某次运行 \
  --checkpoint model_100000.pt \
  --num_envs 4 \
  --device cuda:0
```

还可使用 `--video --video_length 500` 录制播放视频；`--headless` 用于无界面运行。播放脚本会将策略导出到 checkpoint 目录下的 `exported/policy.onnx`。

常用播放参数：

```text
--motion_file FILE          播放单个 NPZ（与 --motion_dir 二选一）
--motion_dir DIR            播放目录中的多个 NPZ
--playback_mode MODE        training / full_clip / fixed_clip
--motion_id N               fixed_clip 使用的动作编号，从 0 开始
--load_run NAME             日志运行目录名
--checkpoint FILE           checkpoint 文件名
--wandb_path PATH           从 W&B 运行下载模型（可选）
--num_envs N                播放环境数
--free_camera               禁止相机跟随机器人，允许手动拖动
--video                     录制播放视频
--video_length N            视频长度（步）
--headless                  无界面播放
```

## 确定性验收评估

该脚本让每个 NPZ 的独立试验都从第 0 帧开始，并分别统计成功率、跟踪失败和最终站立失败，同时检查动作边界以及固定平台与高程图的一致性。

```bash
python scripts/rsl_rl/evaluate_elf3_climb.py \
  --task Tracking-Climb-ELF3-v0 \
  --motion_dir /home/kvoy/Desktop/php_kvoy_reproduction/data/processed_motions/elf3/climb_50hz \
  --checkpoint /home/kvoy/Desktop/php_kvoy_reproduction/logs/rsl_rl/elf3_climb/某次运行/model_100000.pt \
  --trials_per_motion 10 \
  --headless \
  --json_output climb_eval.json \
  --csv_output climb_eval.csv
```

评估随机障碍物时增加 `--randomized_obstacles`；固定平台默认要求每个 NPZ 成功率为 100%，随机障碍物默认要求至少 90%。可用 `--min_success_rate 0.8` 自定义阈值，`--seed` 固定评估随机性，`--platform_tolerance` 设置平台/高程图允许误差。

## 代码结构

```text
source/php_kvoy_reproduction/php_kvoy_reproduction/
├── assets/                 ELF3 机器人资产
├── tasks/tracking/
│   ├── config/elf3/        ELF3 平地/攀爬环境与 PPO 配置
│   └── mdp/                观测、奖励、事件、终止和动作采样逻辑
└── utils/                  ONNX 导出等工具
scripts/rsl_rl/             训练、播放和确定性评估入口
data/processed_motions/     处理后的 NPZ 动作
logs/rsl_rl/                训练日志与 checkpoint
tests/                      自动化测试
```

运行基础测试：

```bash
python -m pytest -q
```
