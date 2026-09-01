# PHP-Kvoy Reproduction

基于 Isaac Lab 的 ELF3 机器人动作跟踪与平台技能专家策略训练工程。当前包含固定 0.66 m 平台的攀爬与 down-roll 专家任务。

## 环境与安装

需要已安装 Isaac Sim 4.5、Isaac Lab 2.1 和 Python 3.10，并使用包含 Isaac Lab 的 conda 环境（示例环境名为 `mimic`）。

```bash
conda activate mimic
cd /home/kvoy/Desktop/php_kvoy_reproduction
python -m pip install -e source/php_kvoy_reproduction
```

推荐使用已修复自然开场姿态的数据：

```text
data/processed_motions/elf3/climb_50hz_default_start_v1/
```

目录训练会读取其中全部 `.npz` 文件（当前为同一攀爬技能的 4 个动作）。旧的 `climb_50hz/` 数据保持不变，可用于对照。

若上述新目录尚未生成，先在包含 MuJoCo 的 Holosoma 环境中重建原始 WBT 数据，再用 `mimic` 转换为训练格式：

```bash
conda run -p /home/kvoy/.holosoma_deps/miniconda3/envs/hsretargeting \
  python scripts/rebuild_elf3_climb_default_start.py \
  data/motions/elf3/climb_50hz \
  data/motions/elf3/climb_50hz_default_start_v1 \
  --holosoma-root /home/kvoy/Desktop/PHP-kvoy/holosoma

conda run -n mimic python scripts/convert_elf3_holosoma2wbt_npz.py \
  data/motions/elf3/climb_50hz_default_start_v1 \
  data/processed_motions/elf3/climb_50hz_default_start_v1
```

0.66 m down-roll 使用独立的离线接触修正版：

```text
data/processed_motions/elf3/down_roll_50hz_platform_0p66_v1/
```

该版本只在足底会穿入 0.66 m 台面的离台阶段平滑修正根部 Z；29 个关节、根部 XY/姿态和首次地面接触后的滚翻恢复保持原数据不变。重新生成命令为：

```bash
conda run -p /home/kvoy/.holosoma_deps/miniconda3/envs/hsretargeting \
  python scripts/rebuild_elf3_down_roll_platform_height.py \
  data/motions/elf3/down_roll_50hz \
  data/motions/elf3/down_roll_50hz_platform_0p66_v1 \
  --holosoma-root /home/kvoy/Desktop/PHP-kvoy/holosoma \
  --platform-height 0.66

conda run -n mimic python scripts/convert_elf3_holosoma2wbt_npz.py \
  data/motions/elf3/down_roll_50hz_platform_0p66_v1 \
  data/processed_motions/elf3/down_roll_50hz_platform_0p66_v1
```

## 任务

| 任务 | 用途 |
| --- | --- |
| `Tracking-Climb-ELF3-v0` | 固定 0.66 m 平台攀爬专家训练与评估 |
| `Tracking-DownRoll-ELF3-v0` | 固定 0.66 m 平台下台翻滚专家训练与评估 |
| `Tracking-Flat-ELF3-v0` | 平地动作跟踪 |
| `Distillation-MultiSkill-ELF3-v0` | locomotion、climb 和 down-roll 三专家视觉蒸馏 |

## 训练

### 多动作（推荐）

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Climb-ELF3-v0 \
  --motion_dir /home/kvoy/Desktop/php_kvoy_reproduction/data/processed_motions/elf3/climb_50hz_default_start_v1 \
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

### Down-roll 专家

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-DownRoll-ELF3-v0 \
  --motion_dir /home/kvoy/Desktop/php_kvoy_reproduction/data/processed_motions/elf3/down_roll_50hz_platform_0p66_v1 \
  --num_envs 2048 \
  --max_iterations 200000 \
  --logger tensorboard \
  --run_name elf3_down_roll \
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

## 视觉多专家蒸馏

蒸馏任务将冻结的 locomotion、climb 和 down-roll 三个专家统一蒸馏到一个视觉 Student。Student Actor
观测包含 8 帧本体历史、二维速度命令和 `87 x 58` 头部深度图，不直接观察专家编号或特权平台参数。
网络使用共享视觉编码器、三分类技能选择器和三个互相独立的动作头；选择器只做硬路由，不对不同专家动作
加权平均。部署接口不接收 `skill_id`，而是由 Student 从视觉、本体历史和速度请求自主选择。climb/down-roll
一旦启动会经过迟滞确认并锁定执行，不能直接互相跳转，完成后再恢复 locomotion。
当前 50 Hz 控制配置要求 motion 概率至少为 `0.60` 并连续确认 3 帧后才启动；启动后至少锁定 100 帧
（2.0 s），超过最短时长后需 locomotion 概率至少为 `0.55` 并连续确认 5 帧才释放。单次 motion 的安全
上限为 400 帧（8.0 s），释放后有 25 帧（0.5 s）重触发冷却。训练 checkpoint 会校验这些部署状态机
参数，不能在训练、播放或后续部署时静默改成另一套数值。
当前头部 D435i 按水平向下 42 度固定，深度采集频率为 30 Hz；训练假设部署时将完整的
`848 x 480` 深度图直接缩放到 `87 x 58`，再按照训练代码裁剪、归一化。

### 构建教师 bundle

每台训练机器第一次使用时，先从仓库内三个源模型生成带哈希、观测维度和动作契约校验的教师
bundle。若 `data/expert_models/elf3/distillation_bundle_v1/` 已存在且三个 manifest 校验正常，无需重复生成。
三个源 `.pt` 是 Git LFS 对象；新机器 clone/pull 代码后，必须先拉取真实权重，不能使用 LFS 指针文件：

```bash
cd ~/Desktop/php_kvoy_reproduction

git lfs pull
git lfs checkout
```

确认下面三个文件均为 MB 量级后再构建 bundle：

```bash
ls -lh \
  data/expert_models/elf3/source/locomotion/policy.pt \
  data/expert_models/elf3/source/climb/model_36000.pt \
  data/expert_models/elf3/source/down_roll/model_12000.pt
```

```bash
cd ~/Desktop/php_kvoy_reproduction

python scripts/tools/build_teacher_artifacts.py \
  --locomotion_source data/expert_models/elf3/source/locomotion/policy.pt \
  --climb_checkpoint data/expert_models/elf3/source/climb/model_36000.pt \
  --down_roll_checkpoint data/expert_models/elf3/source/down_roll/model_12000.pt \
  --output_dir data/expert_models/elf3/distillation_bundle_v1
```

成功后应生成：

```text
data/expert_models/elf3/distillation_bundle_v1/
├── locomotion_manifest.json
├── climb_manifest.json
├── down_roll_manifest.json
├── locomotion_actor.pt
├── climb_actor.pt
├── down_roll_actor.pt
├── elf3_action_contract.json
├── elf3.urdf
└── elf3.usd
```

源专家或机器人动作契约改变后，需要重新审计源模型并使用 `--overwrite` 重建；不要手工修改生成的
manifest。

### 训练前冒烟检查

正式长训前先运行 4 环境、1 轮检查。该命令会实际加载三个专家、创建头部 RTX 深度相机，并校验
教师观测、29-DoF 动作顺序、动作缩放和资产哈希。

```bash
cd ~/Desktop/php_kvoy_reproduction

python scripts/rsl_rl/train_distillation.py \
  --task Distillation-MultiSkill-ELF3-v0 \
  --climb_motion_dir data/processed_motions/elf3/climb_50hz_default_start_v1 \
  --down_roll_motion_dir data/processed_motions/elf3/down_roll_50hz_platform_0p66_v1 \
  --locomotion_manifest data/expert_models/elf3/distillation_bundle_v1/locomotion_manifest.json \
  --climb_manifest data/expert_models/elf3/distillation_bundle_v1/climb_manifest.json \
  --down_roll_manifest data/expert_models/elf3/distillation_bundle_v1/down_roll_manifest.json \
  --training_stage atomic \
  --num_envs 4 \
  --max_iterations 1 \
  --seed 42 \
  --experiment_name elf3_multi_skill_distillation_smoke \
  --run_name visual_student_smoke \
  --logger tensorboard \
  --device cuda:0 \
  --headless
```

确认环境初始化完成、六个观测组维度正确、训练完成 `Iteration 0/0`，且没有 manifest、哈希、观测维度、
动作契约或 CUDA 错误后，再开始正式训练。第一轮尚无完整 episode 时，reward 和 episode length 显示
`nan` 是正常的；loss 或动作出现非有限值则不正常。

### 三阶段正式训练

必须按 `atomic -> transition -> full` 顺序训练。`atomic` 先在标准 0.66 m 平台、无深度噪声/延迟和无相机
外参扰动的条件下学习三个完整独立技能；`transition` 保持标准平台，恢复部署侧深度噪声、延迟和相机外参
扰动，并使用连续物理状态学习 locomotion、climb 和 down-roll 之间的切换；`full` 再加入完整平台几何
随机化、完整组合 episode 和锁定期遥控请求扰动。atomic 的随机相位只用于学习独立动作头；transition/full
的自主路由 episode 都从部署可复现的 locomotion 状态开始，不向 Student 泄露重置技能。两个专家边界不做人工关节插值：approach 仍是正常
响应 `(vx, vy)` 的 locomotion；入口/边缘
settle、完整 climb/down-roll 和动作结束后的短暂安全释放属于 motion control lock。锁定期间内部 locomotion
Teacher 可以使用零速度完成稳定，climb/down-roll Teacher 不接收速度命令；Student 仍始终观察遥控器最新
请求，并通过深度图与本体历史隐式学会暂时忽略它。安全释放完成后，locomotion 立即恢复最新请求，而不是固定
恢复平台前向速度。只有关节姿态、实际关节速度、躯干直立度、平台相对位置、朝向和待启动专家的运动学作用域
全部满足时才切换 Teacher。settle 不再使用随机等待时长；切换只由上述可观测门控决定。

路由训练采用按 episode 采样的 teacher forcing，而不是逐帧随机切换：atomic 始终使用 oracle 路由来先学稳
三个动作头；transition 在前 20000 轮从 `100%` 线性降到 `25%`；full 从 `25%` 线性降到 `0%`。无论实际
执行路由是否由 oracle 提供，选择器始终使用 oracle 分类标签训练；PPO 只重算 rollout 当时真正执行的动作头，
DAgger 只监督 oracle 对应动作头。

外部二维速度请求及 `--vx/--vy` 使用世界坐标，Student Actor 接收该请求在机器人机体坐标系中的二维投影；
请求必须满足 `sqrt(vx^2 + vy^2) <= 1.0 m/s`。down-roll 只会在请求方向与机器人朝向都对齐平台前向时
触发；仅有很小前向分量的侧向命令不会误触发 down-roll。

为避免一开始同时学习所有困难，atomic 和 full 阶段会在 motion control lock 内以 1--2 s 间隔改变仅 Actor
可见的遥控请求，Teacher 动作和专家轨迹保持不变；transition 阶段关闭该扰动，先学习可靠的标准平台连续
切换。climb/down-roll 因此不会被中途遥控命令打断，但结束后可以响应锁定期间收到的最后一条命令。

下面以 RTX 4090、2048 环境为例。若显存不足，优先将 `--num_envs` 依次降为 1024 或 512；不要改变深度图
尺寸、教师 manifest 或动作缩放来规避显存问题。

第一阶段从头训练：

```bash
cd ~/Desktop/php_kvoy_reproduction

python scripts/rsl_rl/train_distillation.py \
  --task Distillation-MultiSkill-ELF3-v0 \
  --climb_motion_dir data/processed_motions/elf3/climb_50hz_default_start_v1 \
  --down_roll_motion_dir data/processed_motions/elf3/down_roll_50hz_platform_0p66_v1 \
  --locomotion_manifest data/expert_models/elf3/distillation_bundle_v1/locomotion_manifest.json \
  --climb_manifest data/expert_models/elf3/distillation_bundle_v1/climb_manifest.json \
  --down_roll_manifest data/expert_models/elf3/distillation_bundle_v1/down_roll_manifest.json \
  --training_stage atomic \
  --num_envs 2048 \
  --max_iterations 20000 \
  --seed 42 \
  --experiment_name elf3_multi_skill_distillation \
  --run_name elf3_visual_student_atomic \
  --logger tensorboard \
  --device cuda:0 \
  --headless
```

确认 fixed-skill 的三个技能均能完整执行后，用第一阶段 checkpoint 启动新的 transition run（示例时间目录需
改为实际值）：

```bash
python scripts/rsl_rl/train_distillation.py \
  --task Distillation-MultiSkill-ELF3-v0 \
  --climb_motion_dir data/processed_motions/elf3/climb_50hz_default_start_v1 \
  --down_roll_motion_dir data/processed_motions/elf3/down_roll_50hz_platform_0p66_v1 \
  --locomotion_manifest data/expert_models/elf3/distillation_bundle_v1/locomotion_manifest.json \
  --climb_manifest data/expert_models/elf3/distillation_bundle_v1/climb_manifest.json \
  --down_roll_manifest data/expert_models/elf3/distillation_bundle_v1/down_roll_manifest.json \
  --training_stage transition \
  --warm_start \
  --load_run 2026-08-28_12-00-00_elf3_visual_student_atomic \
  --checkpoint model_19999.pt \
  --num_envs 2048 \
  --max_iterations 20000 \
  --seed 42 \
  --experiment_name elf3_multi_skill_distillation \
  --run_name elf3_visual_student_transition \
  --logger tensorboard \
  --device cuda:0 \
  --headless
```

确认 composed 模式可以完成 climb、台上 locomotion，并在短台面进入 down-roll 后，再由 transition
checkpoint warm-start 完整随机化阶段：

```bash
python scripts/rsl_rl/train_distillation.py \
  --task Distillation-MultiSkill-ELF3-v0 \
  --climb_motion_dir data/processed_motions/elf3/climb_50hz_default_start_v1 \
  --down_roll_motion_dir data/processed_motions/elf3/down_roll_50hz_platform_0p66_v1 \
  --locomotion_manifest data/expert_models/elf3/distillation_bundle_v1/locomotion_manifest.json \
  --climb_manifest data/expert_models/elf3/distillation_bundle_v1/climb_manifest.json \
  --down_roll_manifest data/expert_models/elf3/distillation_bundle_v1/down_roll_manifest.json \
  --training_stage full \
  --warm_start \
  --load_run 2026-08-28_18-00-00_elf3_visual_student_transition \
  --checkpoint model_19999.pt \
  --num_envs 2048 \
  --max_iterations 100000 \
  --seed 42 \
  --experiment_name elf3_multi_skill_distillation \
  --run_name elf3_visual_student_full \
  --logger tensorboard \
  --device cuda:0 \
  --headless
```

`--warm_start` 只继承 Student/normalizer 权重并为新阶段重置优化器和课程轮次；只允许 atomic checkpoint
初始化 transition、transition checkpoint 初始化 full。`--resume` 仅用于同一阶段、同一环境语义的中断续训。
checkpoint 会保存训练阶段、头部相机、FOV、深度裁剪、Actor 观测语义与动作契约指纹，训练和播放遇到不一致
时会拒绝加载。当前命令语义是“实时遥控请求始终可见、motion lock 内忽略控制作用、释放后恢复最新请求”；
采用旧命令语义或旧 checkpoint 格式的 run 不可作为本流程起点，必须从 `atomic` 重新训练。尤其是
`php_multi_teacher_student_v3` 单动作头 checkpoint 与当前分层硬路由网络结构不兼容。

日志写入：

```text
logs/rsl_rl/elf3_multi_skill_distillation/<时间>_elf3_visual_student_<阶段>/
```

默认每 500 轮保存一次 checkpoint，并在训练结束时额外保存最后一轮。例如单阶段训练 20000 轮的最终文件名是
`model_19999.pt`。TensorBoard 命令：

```bash
tensorboard --logdir logs/rsl_rl/elf3_multi_skill_distillation
```

终端会直接显示 `selector` 损失、`acc`（选择器相对 oracle 的准确率）、`route`（实际执行路由与 oracle 的
一致率）和 `forced`（当前 batch 的路由 teacher-forcing 比例）。TensorBoard 还记录
`Loss/selector_accuracy/<skill>`、`Loss/dagger_mse/<skill>`、Student/Teacher 动作最大绝对值等细分指标。

不要只根据总 reward 或训练轮数切换阶段。atomic 结束前应分别播放 locomotion、climb、down-roll，确认三个
动作头都能完整执行；transition 结束前应使用 composed 播放确认可以自主完成
`locomotion -> climb -> 台上 locomotion`，并在短平台接近远端时进入 down-roll；full 阶段则需在训练的完整
平台尺寸、深度噪声和相机外参随机化范围内重复验证。随着 teacher forcing 降低，应同时确认 `acc` 保持较高、
`route` 没有持续下降、三个 `Loss/dagger_mse/<skill>` 均稳定，并且实际播放没有提前触发、反复切换或 motion
中途退出。指标与画面必须共同通过，不能用单一 loss 代替真实成功率。

### 播放视觉 Student

固定技能模式用于分别检查三个技能。下面以 climb、动作 0 为例；可将 `--skill` 改成 `locomotion` 或
`down_roll`。该模式同时固定环境 oracle 技能和 Student 动作头，并通过训练使用的 Option Controller 从第一帧
执行对应动作头，因此与 atomic 的路由 teacher forcing 对齐；它只验收独立动作头，不代表部署时向 Student
输入技能 ID。固定 climb/down-roll 中的 `--vx/--vy` 仍作为 Actor 可见的遥控输入，用于验证 motion lock 内
动作不被命令打断；固定 locomotion 才直接响应速度请求。
`--checkpoint_path` 必须指向视觉 Student checkpoint，而不是三个
专家的 checkpoint。

```bash
cd ~/Desktop/php_kvoy_reproduction

python scripts/rsl_rl/play_distillation.py \
  --task Distillation-MultiSkill-ELF3-v0 \
  --climb_motion_dir data/processed_motions/elf3/climb_50hz_default_start_v1 \
  --down_roll_motion_dir data/processed_motions/elf3/down_roll_50hz_platform_0p66_v1 \
  --locomotion_manifest data/expert_models/elf3/distillation_bundle_v1/locomotion_manifest.json \
  --climb_manifest data/expert_models/elf3/distillation_bundle_v1/climb_manifest.json \
  --down_roll_manifest data/expert_models/elf3/distillation_bundle_v1/down_roll_manifest.json \
  --checkpoint_path logs/rsl_rl/elf3_multi_skill_distillation/2026-08-28_12-00-00_elf3_visual_student_atomic/model_19999.pt \
  --playback_mode fixed_skill \
  --skill climb \
  --motion_id 0 \
  --vx 0.6 \
  --vy 0.0 \
  --num_envs 4 \
  --device cuda:0
```

组合模式不固定 Student 动作头，使用与部署相同的自主 Option Controller；由实际平台长度和视觉输入决定何时
从 locomotion 切换至 climb、在台面继续 locomotion，或在接近远端时进入 down-roll。它用于验收
transition/full checkpoint，不能同时指定 `--skill`，也不应使用 atomic checkpoint 判断自主切换能力：

```bash
cd ~/Desktop/php_kvoy_reproduction

python scripts/rsl_rl/play_distillation.py \
  --task Distillation-MultiSkill-ELF3-v0 \
  --climb_motion_dir data/processed_motions/elf3/climb_50hz_default_start_v1 \
  --down_roll_motion_dir data/processed_motions/elf3/down_roll_50hz_platform_0p66_v1 \
  --locomotion_manifest data/expert_models/elf3/distillation_bundle_v1/locomotion_manifest.json \
  --climb_manifest data/expert_models/elf3/distillation_bundle_v1/climb_manifest.json \
  --down_roll_manifest data/expert_models/elf3/distillation_bundle_v1/down_roll_manifest.json \
  --checkpoint_path logs/rsl_rl/elf3_multi_skill_distillation/2026-08-28_18-00-00_elf3_visual_student_transition/model_19999.pt \
  --playback_mode composed \
  --motion_id 0 \
  --vx 0.6 \
  --vy 0.0 \
  --num_envs 4 \
  --device cuda:0
```

交互播放默认持续运行；可加 `--max_steps 1000` 自动停止。播放在创建场景前读取 checkpoint 的
`training_stage`，自动匹配 atomic/transition/full 的平台和视觉课程，并恢复 checkpoint 训练轮次，从而沿用
训练时相同的终止阈值；reset 时会打印实际终止原因，只想看画面时可加 `--quiet_reset_log`。排除视觉噪声
影响时可以临时加入 `--no_depth_noise`，它同时关闭深度像素噪声和相机外参随机化，不应用于最终鲁棒性验收。

## 播放策略

`--load_run` 是日志运行目录名，`--checkpoint` 是其中的模型文件名，例如 `model_100000.pt`。

### 播放全部动作（从第 0 帧开始）

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Climb-ELF3-v0 \
  --motion_dir /home/kvoy/Desktop/php_kvoy_reproduction/data/processed_motions/elf3/climb_50hz_default_start_v1 \
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
  --motion_dir /home/kvoy/Desktop/php_kvoy_reproduction/data/processed_motions/elf3/climb_50hz_default_start_v1 \
  --playback_mode fixed_clip \
  --motion_id 0 \
  --load_run 2026-08-12_某次运行 \
  --checkpoint model_100000.pt \
  --num_envs 1 \
  --device cuda:0
```

播放模式：`training` 保持训练配置；`full_clip` 从第 0 帧按顺序播放全部动作；`fixed_clip` 从第 0 帧固定播放一个动作。`full_clip` / `fixed_clip` 默认在第一个环境执行完专家末帧后停止；仅在需要观察末帧之后的自由物理演化时加入 `--no-stop_at_motion_end`。`--free_camera` 使用世界坐标相机，可以在 Isaac Sim 窗口中手动拖动视角。

需要核对跟踪姿态时，可在 `full_clip` 或 `fixed_clip` 播放中加入 `--debug_vis`。它显示当前机器人和专家目标的关键 body 三维坐标轴（不是 29 个关节的数值）；建议使用 `--num_envs 1`、配合 `--free_camera`，且不要加 `--headless`：

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Climb-ELF3-v0 \
  --motion_dir /home/kvoy/Desktop/php_kvoy_reproduction/data/processed_motions/elf3/climb_50hz_default_start_v1 \
  --playback_mode fixed_clip \
  --motion_id 0 \
  --free_camera \
  --debug_vis \
  --load_run 2026-08-12_某次运行 \
  --checkpoint model_100000.pt \
  --num_envs 1 \
  --device cuda:0
```

还可使用 `--video --video_length 500` 录制播放视频；`--headless` 用于无界面运行。播放脚本会将策略导出到 checkpoint 目录下的 `exported/policy.onnx`。

常用播放参数：

```text
--motion_file FILE          播放单个 NPZ（与 --motion_dir 二选一）
--motion_dir DIR            播放目录中的多个 NPZ
--playback_mode MODE        training / full_clip / fixed_clip
--motion_id N               fixed_clip 使用的动作编号，从 0 开始
--[no-]stop_at_motion_end   full/fixed 默认在专家末帧停止；no- 前缀允许无限保持末帧
--load_run NAME             日志运行目录名
--checkpoint FILE           checkpoint 文件名
--wandb_path PATH           从 W&B 运行下载模型（可选）
--num_envs N                播放环境数
--free_camera               禁止相机跟随机器人，允许手动拖动
--debug_vis                 显示当前机器人与专家目标关键 body 的三维坐标轴（full/fixed 播放）
--video                     录制播放视频
--video_length N            视频长度（步）
--headless                  无界面播放
```

## 确定性验收评估

该脚本让每个 NPZ 的独立试验都从第 0 帧开始，并统计专家轨迹完成率与提前跟踪失败，同时检查动作边界以及固定平台与高程图的一致性。当前攀爬任务在专家末帧结束，不再追加额外的最终站立保持阶段。

```bash
python scripts/rsl_rl/evaluate_elf3_climb.py \
  --task Tracking-Climb-ELF3-v0 \
  --motion_dir /home/kvoy/Desktop/php_kvoy_reproduction/data/processed_motions/elf3/climb_50hz_default_start_v1 \
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
