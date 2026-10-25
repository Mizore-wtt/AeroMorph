# AeroMorph GPU 飞行与在线变构

本实现使用本机 Isaac Lab `v3.0.0-beta2.patch1`、Isaac Sim `6.0.1-rc.7`。默认 GPU PhysX，重力开启，依靠四路旋翼的实际外力和反作用力矩飞行。飞行模型保留九个刚体和四个变构自由度，右侧滑块可在飞行中调节关节目标。展示与飞行入口使用同一份 `src/aeromorph_description/urdf/AeroMorph.urdf`。

## 新模型状态（2026-10-25）

模型已切换到最新 CAD 导出版本：根连杆为 `link3`，变构关节为 `joint1/joint2/joint3/joint4`，旋翼关节仍为 `fly_joint1`～`fly_joint4`。URDF、CSV 和 9 个 STL 统一放在 ROS 2 包的 `urdf/`、`meshes/` 中，展示与飞行读取同一套模型。

上版旋翼网格相对转轴的偏移已经修复，四桨扫掠半径均约 0.060024 m。新模型质量约 0.693176806 kg，零位质心在 `link3` 下约为 `(-0.141284185, -0.000043982, 0.029702735)` m。

**展示保持 CAD 零位；飞行默认使用非零 nominal 构型。** CAD 零位四个桨毂共线，分配矩阵秩为 3，会被构型检查拒绝。为适配新几何，YAML 的三个预设已更新如下，角度顺序为 `joint1/joint2/joint3/joint4`：

| 构型 | 关节角 [rad] | 离线悬停推力，fly1～fly4 [N] |
|---|---|---|
| nominal | `[-0.35, 0.25, 0.25, -0.35]` | `[1.6221, 1.7797, 1.6204, 1.7779]` |
| swept | `[-0.55, 0.40, 0.35, -0.50]` | `[1.9844, 1.0672, 2.3328, 1.4156]` |
| asymmetric | `[-0.30, 0.25, 0.30, -0.40]` | `[1.2100, 2.5948, 0.8053, 2.1900]` |

三个预设均通过离线扫掠圆盘、分配矩阵、推力及力矩余量检查。在线变构增加了缓存运动学、完整路径检查、参考速度/加速度限制，以及变构角动量补偿。第 7 节的旧模型飞行结果仅作历史归档，不能用来证明新版本的性能。

## 1. 启动

在项目目录中，第一次新增启动文件后先构建 ROS 包：

```bash
cd ~/ST_Innovation/AeroMorph
source /opt/ros/humble/setup.bash
colcon build --packages-select aeromorph_description
source install/setup.bash
ros2 launch aeromorph_description flight.launch.py
```

以后打开新终端，source 上述两个 `setup.bash` 后直接运行最后一条命令即可。无需 `isaaclab_display.sh`，也无需安装 ROS/Isaac ROS Bridge。ROS launch 负责调用 Isaac Sim 自带 Python，物理仿真和控制在同一个进程中运行。

通过构型检查并打开窗口后，无人机质心位于世界坐标 `(0, 0, 1.2)` m，自动悬停。视口与展示程序一样使用深灰背景和网格，地面碰撞体保留但不可见。相机从初始航向的后上方朝前观察，航向改变时初始相机方向也随之旋转；运行后可自由调整视角。点击视口再操作；终端每仿真秒输出一次实际状态。右侧面板显示飞行状态及四个变构滑块。关闭窗口或在启动终端按 Ctrl+C 可结束。

更换构型或初始航向：

```bash
ros2 launch aeromorph_description flight.launch.py configuration:=swept
ros2 launch aeromorph_description flight.launch.py configuration:=asymmetric yaw:=1.57079632679
ros2 launch aeromorph_description flight.launch.py joints:="-0.30 0.25 0.30 -0.40"
```

`joints` 顺序固定为 **joint1、joint2、joint3、joint4**，单位 rad；覆盖所选构型的四个关节角。关节名称和范围以 URDF 为准。修改 YAML 不需要重新 colcon build，重新启动飞行即可。

也可以直接运行 Python 入口：

```bash
cd ~/ST_Innovation/AeroMorph
PYTHONPATH="$HOME/IsaacLab/source/isaaclab" \
  "$HOME/IsaacLab/_isaac_sim/python.sh" isaaclab/scripts/fly.py --configuration swept
```

推荐使用 ROS launch，它还会隔离 Conda、ROS Python 和动态库环境。默认采用 `cuda:0` GPU PhysX，并开启 Fabric 直接同步物理状态；窗口渲染同样使用 NVIDIA GPU。可用 `device:=cuda:1` 选择另一块可见 GPU，或 `device:=cpu` 使用 CPU。直接 Python 入口也默认 `--device cuda:0`。安装路径不同时可传 `isaaclab_root:=/path/to/IsaacLab workspace:=/path/to/AeroMorph`。

GPU 接触/碰撞缓冲按单机场景设置，避免默认的大规模并行容量占用过多显存。在本机 6 GB RTX 4050 上，交互窗口约占用 3.3 GB 显存；训练时应按并行环境规模重新配置缓冲。飞行网格由无质量、无碰撞的参考线显示，播放时仍然可见，不参与飞行动力学。

当前 NumPy/SciPy 飞控与约束求解仍在 CPU，GPU 状态每周期读回。此版本提供 GPU 关节动力学基础；后续大批量训练还需要张量化控制及并行环境，不能把单机键盘程序直接当作高吞吐训练环境。

## 2. 键盘和坐标系

| 按键 | 指令 |
|---|---|
| W / S | 沿初始航向前进 / 后退 |
| A / D | 沿初始航向左移 / 右移 |
| R / F | 上升 / 下降 |
| H | 制动并保持停止后的位置 |
| 松开运动键 | 相应轴自动制动，然后捕获当前位置保持 |

支持 W+R、A+R、W+D 等组合。水平合速度目标限制为 0.5 m/s，垂直目标限制为 0.3 m/s；对角线不会把水平目标提高到 √2 倍。参考速度有加速度限制，实际速度受闭环控制，过渡过程中允许有限超调。

**空格保留给 Isaac Sim 时间轴**，因此悬停键采用 H。按 H 后，即使方向键尚未松开，也先制动；方向键全部松开后才重新接受运动输入。窗口失焦会清空运动键，恢复焦点后需要重新按键。

- 世界系 W：Z 向上，重力 `(0, 0, -9.81)` m/s²。
- 机体系 B：原点为当前时刻的整机质心，坐标轴平行 URDF 的 `link3`。机体 +X 为前、+Y 为左、+Z 为上。
- 操纵系 H：只绕世界 Z 轴旋转初始航向 ψ₀，整个飞行期间固定。`v_W = Rz(ψ₀) v_H`。例如 ψ₀=90° 时，W 对应世界 +Y，A 对应世界 −X。
- 姿态 `R_WB` 把 B 系向量旋转到 W 系。角速度反馈和控制力矩使用 B 系。本机 Isaac Lab 3.0 接口的四元数顺序为 **XYZW**。

H 和松键都通过速度反馈制动，再捕获新位置，不会直接回到起飞点。没有单独的航向操纵键，目标航向始终为初始航向。

### 飞行中调节构型

拖动右侧 `joint1`～`joint4` 滑块即可设置目标角度，单位 rad。滑块范围取自 URDF 关节限位，但该范围内并非所有构型都可飞。面板的 `Actual [rad]` 显示真实关节角，与滑块目标分开；关节不会瞬间跳到滑块位置。

- 请求先检查目标及加减速/反向过程中经过的路径，包括桨盘间距、分配矩阵秩/条件数、非负推力、推力上限和悬停余量；目标还检查三轴双向力矩余量。四关节沿共同进度协调变化，中途换方向先制动，再进入新路径，避免独立到位导致中间构型失去余量。
- 默认参考速度不超过 `0.08 rad/s`，参考加速度不超过 `0.08 rad/s²`，再由有阻尼、有限力矩的物理关节驱动跟踪。URDF 的 CAD 限位不等于已验证可飞范围。
- 倾角超过 `0.15 rad`、角速度超过 `0.6 rad/s`、关节跟踪误差超过 `0.04 rad` 或分配残差过大时，变构参考减速暂停；飞行闭环继续工作，状态恢复后自动继续。
- `MOVING/HOLD/PAUSED` 分别表示正在变构、到位和暂停；`REJECTED/BLOCKED` 显示拒绝原因。无效请求不会替换已接受的可行目标，之前的合法变构仍可继续。不要通过零位在两侧构型之间切换，直线路径可能穿越退化区域。
- 可以同时飞行和变构。推荐先悬停、小幅调节，观察实际角度与状态，再组合运动；连续拖动会更新最新目标。

这些检查是当前平行 +Z 推进布局的可行性保护，包含桨盘间距但不是完整机构自碰撞检测。为避免 CAD 包围盒在铰链处产生假碰撞，内部自碰撞关闭；不模拟叶片气动干涉、转子陀螺效应、舵机间隙或风场。稳定性结论仅适用于配置参数及已验收的变构路径。

### 螺旋桨显示动画

飞行窗口中，四个螺旋桨绕各自桨毂旋转，分别根据各自电机一阶响应后的实际推力自动选择低、中、高显示转速。按各电机最大推力归一化，低/中、中/高的切换阈值分别为 `30% / 50%`，各保留 `2%` 回差以避免临界值附近频繁跳档（例如低→中需达到 32%，中→低需低于 28%）。各桨独立分档，右侧 `Rotor display speed (automatic)` 只显示 R1～R4 当前档位，顺序与推力及配置中的旋翼顺序一致，无需手动选择。

三档显示速度为 `1.2 / 2.8 / 4.8` 转/秒，是上一版各档速度的两倍，不代表真实电机转速；切档保持转角连续。默认渲染提高为 50 Hz，让加速后的叶片运动仍可辨认。

动画只修改旋翼的外观子节点，随真实连杆移动和变构，不改质量、惯量、物理关节或推力。显示旋向与配置的机体反扭矩相反。按仿真时间推进，时间轴暂停时停止；某桨实际推力降至 `0.001 N` 以下时停止该桨动画。无界面运行跳过动画。

## 3. 构型和物理参数

全部可调参数放在 [`configs/flight.yaml`](configs/flight.yaml)。包括三个示例构型、旋翼参数、控制增益、速度/加速度限制、倾角限制、时序和诊断阈值。

| 参数 | 单位 / 含义 |
|---|---|
| `configurations` | 四个变构关节角，rad |
| `thrust_sign` | 相对于 URDF 旋翼关节轴的正推力侧，±1 |
| `reaction_sign` | 沿正推力方向、作用在机体上的反扭矩符号，±1 |
| `kappa_m` | 反扭矩 / 推力，N·m/N=m |
| `max_thrust_n` | 每桨推力上限，N |
| `motor_tau_s` | 推力一阶响应时间常数，s |
| `position_gain_s2` | 位置误差到加速度的增益，s⁻² |
| `velocity_gain_s` | 速度误差到加速度的增益，s⁻¹ |
| `attitude_frequency_rad_s` | 姿态环各轴自然频率，rad/s |
| `attitude_damping_ratio` | 姿态环各轴阻尼比，无量纲 |
| `max_angular_accel_rad_s2` | 姿态环角加速度指令限制，rad/s² |
| `thrust_rate_n_s` | 各桨推力命令变化率限制，N/s |
| `reference_length_m` | 分配矩阵无量纲化的参考长度，m |
| `morphing.max_speed_rad_s` / `max_accel_rad_s2` | 变构参考速度与加速度上限 |
| `morphing.servo_*` | 舵机刚度、阻尼和力矩上限，均为仿真假设 |
| `morphing.minimum_hover_margin` | 在线变构要求的更保守悬停余量，默认 8% |

**URDF 没有定义实际旋向、电机推力上限、扭矩系数及电机响应。** 当前 YAML 明确标注为 `simulation_assumption`：机体反扭矩符号为 `[+1, -1, -1, +1]`，`κ=0.018 m`、每桨最大 `4.5 N`、时间常数 `0.02 s`。这些值用于建立可运行的仿真系统，实机参数确认后应替换，再重新检查构型。URDF 的关节 axis 不等于已知真实 CW/CCW 旋向。

质量、各连杆惯量及惯性原点、关节变换、旋翼毂位置、旋翼轴和外观 STL 都读取当前 URDF，不读取旧展示 USD 的惯量。新模型总质量约 0.693176806 kg，CAD 零位整机质心约为 `(-0.141284185, -0.000043982, 0.029702735)` m；飞行预设的质心、惯量和悬停推力按其关节角重新求解。

启动前可单独检查任意构型，不开启仿真窗口：

```bash
~/IsaacLab/_isaac_sim/python.sh isaaclab/scripts/check_configuration.py --configuration swept
~/IsaacLab/_isaac_sim/python.sh isaaclab/scripts/check_configuration.py --joints -0.30 0.25 0.30 -0.40
```

检查内容包含 URDF 限位、质量/惯量物理性、旋翼扫掠圆盘间距、无量纲矩阵的秩/条件数、悬停推力上下界与余量，以及保持总推力和其他轴力矩时的三轴双向力矩范围。悬停推力不够、要求负推力、布局退化等会说明原因并退出。它不是完整的机构碰撞检测器。

## 4. 实现方法

1. **自由基座关节模型。** 从原 STL/URDF 生成九个刚体的 articulation，保留 `joint1`～`joint4` 四个转动自由度，以有限力矩的隐式 PD 驱动跟踪平滑参考。旋翼自转关节固定，升力及反扭矩由连续推力模型表示。整机运动及舵机内力反作用均由 PhysX 求解；运行中不写根位姿、速度或关节状态。
2. **正确质量特性与反馈。** 每个连杆分别写入质量、质心、主惯量和主轴，起飞时与 PhysX 读回值核对。每步对所有连杆的实际 COM 位置/速度做质量加权，得到整机 COM 反馈；用实际连杆姿态和完整惯量，结合平行轴定理更新整机惯量。不能把根连杆质心当作整机质心。机身/臂采用包围盒地面碰撞体，桨叶不模拟叶片碰撞。
3. **外环。** 每个操纵轴分别处于 `VELOCITY → BRAKE → HOLD`。速度参考做斜率限制，位置 PD/速度反馈加参考加速度前馈和重力补偿，得到期望世界合力，再由合力方向和初始航向构造期望姿态。
4. **内环与变构补偿。** SO(3) 姿态误差和角速度反馈求角加速度，使用实际整机完整惯量计算力矩，包含 `ω×Jω`。从各连杆的实际速度求整机角动量 `H=J(q)ω+h`，增加经过低通滤波的 `Jdot·ω + hdot + ω×h`，补偿形变引起的惯量变化和相对运动。没有积分器。内部关节可以驱动，但不能产生整机的凭空外力。
5. **实时控制分配。** 每桨列向量为 `[dᵢ; rᵢ×dᵢ + sᵢκᵢdᵢ]`。使用实际桨毂位置与朝向，在每个姿态周期刷新 6×4 的 `B(q)`；对已验证的平行 +Z 布局取总推力与三轴力矩做有界加权最小二乘。刷新映射保留上一周期推力，保证上下界、变化率限制和电机状态连续。
6. **实际受力。** 电机一阶响应后，在四个旋翼刚体的真实桨毂处分别累计局部推力和反扭矩，经 wrench composer 传入 PhysX。局部作用点相对各旋翼自身 COM 设置；力臂力矩只计算一次。控制器的期望力矩只经推力分配实现。

关节驱动的建模依据见 [OpenUSD Physics 关节与驱动定义](https://openusd.org/dev/api/usd_physics_page_front.html)；Isaac Lab 的配置方式可参考 [Articulation 配置说明](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/write_articulation_cfg.html)。本项目 API 以本机安装版本源码为准。

若以后修改 URDF 导致旋翼轴不平行，程序会报告六维映射秩并明确拒绝启动。当前实现没有把四路输入当成六个独立力/力矩通道；一般非平行推进布局需要另行设计受控子空间及姿态生成方法。

默认物理频率 500 Hz，姿态/分配 250 Hz，外环 50 Hz，日志 50 Hz，窗口 50 Hz。GUI 尽量按真实时间节拍运行，机器性能不足时仿真时间会慢于墙钟时间；控制周期仍以物理步长为准。无界面验收不等待真实时间。

达到高度、距离、倾角或角速度边界，出现非有限状态，或分配残差持续超限时，终端说明原因并暂停/结束仿真。持续不可实现的指令不会触发隐藏的位置或姿态修正。

## 5. 文件组织和运行产物

```text
isaaclab/
  aeromorph_control/
    config.py          # 配置读取与有效性检查
    model.py           # URDF 质量特性检查、共享网格/碰撞几何快照
    articulated.py     # 九刚体/四关节 USD、整机反馈、角动量补偿与逐桨受力
    morphology.py      # 缓存运动学、完整变构路径检查与平滑舵机参考
    scene.py           # 灰黑背景、无碰撞参考网格与后方相机
    allocator.py       # 布局可控性、悬停可行性、有界推力分配
    controller.py      # 指令参考、位置/速度和 SO(3) 姿态控制
    rotors.py          # 四路电机的一阶推力响应
    rotor_visuals.py   # 旋翼三档显示动画，不参与动力学
    keyboard.py        # 键盘、失焦处理、简洁状态面板
    telemetry.py       # 实际状态与控制量日志
  configs/flight.yaml
  scripts/fly.py
  scripts/check_configuration.py
  scripts/validate_flight.py
  tests/test_control.py
src/aeromorph_description/launch/flight.launch.py
```

自动生成的 USD 和模型摘要放在 `build/isaaclab/flight/<hash>/`。哈希包含 URDF、STL、关节角及旋翼配置，修改源模型后自动生成对应资产；不手工修改缓存 USD。每次飞行日志放在 `log/flight/<时间戳>/`：

- `configuration.json`：本次完整参数、质量特性、分配矩阵及控制余量。
- `asset.json`：实际 USD 路径、模型哈希、PhysX 读回值和测试场景。
- `flight.csv`：实际 COM 位置/速度、RPY、角速度、XYZW 四元数、输入、参考速度、四桨命令/实际推力、期望控制量、实现六维合力/力矩、模式、饱和及残差；还记录真实关节角/速度、平滑参考、已接受目标、角动量补偿力矩、变构状态及拒绝次数。
- `summary.json`：结束状态、范围、末段漂移和速度等指标。

`build/`、`log/` 已被 Git 忽略；源码、参数、说明文件留在对应目录。

## 6. 复现验证

离线回归测试：

```bash
~/IsaacLab/_isaac_sim/python.sh -m unittest discover -s isaaclab/tests -v
```

32 s 无界面悬停或组合运动，直接运行同一个飞控主程序：

```bash
ros2 launch aeromorph_description flight.launch.py headless:=true scenario:=hover duration:=32
ros2 launch aeromorph_description flight.launch.py headless:=true scenario:=sequence duration:=32 configuration:=swept
```

`sequence` 使用相同指令输入链路：0–3 s 悬停，3–6 s 前进+上升，6–10 s 松键，10–13 s 左移，13–17 s 松键，17–20 s 后退+右移+下降，之后松键悬停。它用于验收，不取代默认键盘操作。

针对终端给出的日志目录执行验收：

```bash
~/IsaacLab/_isaac_sim/python.sh isaaclab/scripts/validate_flight.py log/flight/<运行目录> --scenario sequence
```

悬停场景改为 `--scenario hover`。检查真实运动方向、姿态倾斜、航向、推力边界、末段速度/漂移和分配饱和。`--scenario motors_off --duration 0.30` 是排除伪悬停的短时自由落体检查；较长时间会因低于高度边界而按预期停止。

直接 Python 入口还支持 `--initial-offset x y z`、`--initial-tilt roll pitch`、`--run-dir`，便于做带初始偏差的回归测试。所有测试均使用同一套控制器和 YAML 增益。

GPU 在线变构回归：

```bash
ros2 launch aeromorph_description flight.launch.py headless:=true scenario:=morph duration:=38
ros2 launch aeromorph_description flight.launch.py headless:=true scenario:=morph_sequence duration:=38
```

`morph` 悬停变构，`morph_sequence` 同时执行上述组合飞行动作。两者在 3 s 请求 swept、14 s 请求 asymmetric、25 s 故意请求不可飞零位、26 s 请求启动构型；最后等待稳定。它们与滑块使用同一个请求检查及舵机参考链路。

验收对应日志：

```bash
~/IsaacLab/_isaac_sim/python.sh isaaclab/scripts/validate_flight.py log/flight/<运行目录> --scenario morph_sequence
```

验收除了飞行方向、末段速度/漂移和姿态，还检查实际到达 swept/asymmetric/启动构型，关节跟踪误差，参考速度/加速度上限，以及只拒绝一次故意提交的零位请求。悬停变构用 `--scenario morph`。

### 当前模型 GPU 验证（2026-10-25）

- 19 项离线回归通过，覆盖 URDF/缓存运动学一致性、九刚体四自由度及逐连杆完整惯量、材质绑定、参考网格不引入质量/碰撞、跨退化路径拒绝、速度/加速度限制、暂停及中途反向、分配更新时的推力历史连续性。
- RTX 4050 上的 38 s `morph_sequence` 通过：初始 COM 偏移 `(0.08, -0.06, 0.10)` m、roll/pitch `(0.06, -0.05)` rad；边执行组合飞行边遍历三个构型。日志为 `log/flight/validation-gpu-morph-sequence/`。
- 16 s GPU 窗口滑块检查通过：nominal → swept → 拒绝零位 → nominal，悬停位置 RMSE `0.00761 m`；后方视角、灰黑网格和实际构型已截图确认。日志为 `log/flight/validation-gpu-flight-gui/`，README 包含实测截图。
- 3 s CPU 悬停兼容检查通过；关闭全部旋翼的 0.30 s GPU 检查下落约 `0.336 m`、末段垂直速度约 `−2.56 m/s`，确认没有隐藏悬停外力。汇总结果保存在 `log/flight/verification.json`。

| 指标 | 实测 |
|---|---|
| 最后 2 s 最大速度 | `0.00540 m/s` |
| 最后 2 s 最大单轴位置跨度 | `0.00613 m` |
| 全程最大倾角（含初始扰动与机动） | `6.96°` |
| 最大关节跟踪误差 | `0.00380 rad` |
| 最大关节参考速度 / 加速度 | `0.0800 rad/s` / `0.0800 rad/s²` |
| 分配饱和样本占比 | `0.105%` |
| 零位拒绝 | `1` 次，拒绝后继续稳定飞行 |

这些结果对应当前无风、连续推力和假设舵机参数的模型，仅覆盖上述构型及路径，不代表任意关节角或实机性能保证。

## 7. 旧模型验证记录（2026-10-24，仅归档）

以下数值来自替换前的旧模型，不适用于本次新 CAD 导出版本；当前模型的 GPU 飞行与变构验证见第 6 节。

旧模型已在本机 RTX 4050 Laptop 6 GB 上运行实际 Isaac Sim/PhysX。三种构型各做 32 s 悬停和 32 s 组合运动，均加入初始 COM 偏移 `(0.08, -0.06, 0.10)` m、初始 roll/pitch `(0.06, -0.05)` rad，控制器和增益完全相同。

| 构型 | 悬停位置 RMSE | 组合运动最后 2 s 最大速度 | 悬停配平推力 [N] |
|---|---|---|---|
| nominal | 0.0246 m | 0.0000104 m/s | 2.6223, 0.9940, 2.8444, 1.2161 |
| swept | 0.0246 m | 0.0000099 m/s | 1.7830, 2.4856, 1.3529, 2.0555 |
| asymmetric | 0.0246 m | 0.0000109 m/s | 2.1296, 1.8864, 1.9521, 1.7089 |

其他已通过检查：

- 10 项离线回归：质量/非对角惯量、三构型配平、负悬停推力、低推力上限、奇异布局、非平行轴拒绝、分配限幅/变化率、初始航向和组合速度、松键位置捕获、姿态反馈、电机响应及 H 覆盖指令。
- 三构型均通过 PhysX 质量/完整惯量读回，以及实际四点外力合成与 `B(q)f` 一致性检查。
- 初始航向旋转 90° 的 32 s 组合运动通过，W 对应世界 +Y。
- 关闭全部旋翼的 0.30 s 检查通过：末次日志已下落约 0.339 m，垂直速度约 −2.56 m/s。
- 物理步长从 0.002 s 减半至 0.001 s，保持控制周期不变，32 s 组合运动通过；最终位置与基准相差约 1.37 mm。
- 三构型组合运动最大倾角约 6.93°，分配饱和样本占比 0.125%，松键后恢复 `HOLD`。
- ROS launch 实际打开 GUI，并通过真实键盘事件测试 W+R、A、H 覆盖和松键制动。120 s 窗口运行正常结束；另用临时测试窗口确认，方向键仍按住时失焦也会清空指令并制动。

验收记录保存在 `log/flight/validation-*`。这些结果对应替换前的旧 URDF、明确的电机仿真假设和无风刚体模型；没有将理想仿真中的微小稳态误差当作实机性能指标。
