# AeroMorph 空中变构无人机

本项目为 AeroMorph 空中可变构无人机的仿真与飞行控制训练。

## 一、环境配置

本项目基于 Issaclab 仿真环境开展，环境配置参考网站 `https://docs.robotsfan.com/isaaclab/index.html`

## 二、模型展示

- 模型包含 9 个连杆和 8 个活动关节
- `joint1`、`joint2`、`joint4`、`joint5` 是变构关节，`fly_joint1`～`fly_joint4` 是旋翼关节

在 rviz2 中进行模型展示
```bash
colcon build --packages-select aeromorph_description
source install setup.bash

ros2 launch aeromorph_description display.launch.py
```

模型展示如下：
![](images/readme/2_display_rviz2.png)

## Isaac Lab 展示与移动

本机已安装 Isaac Lab `v3.0.0-beta2.patch1` / Isaac Sim 6.0.1，在项目目录运行：

```bash
./scripts/isaaclab_display.sh
```

启动后显示模型和控制面板。先点击三维视口，再使用键盘：

| 操作 | 按键 |
|---|---|
| 沿世界 X 轴移动 | W / S |
| 沿世界 Y 轴移动 | A / D |
| 升高 / 降低 | R / F |
| 偏航旋转 | Q / E |
| 开关自动移动与变构演示 | T，或面板 Toggle demo |
| 恢复位置和关节零位 | Home，或面板 Reset |
| 调整四个变构关节 | 面板 joint1 / joint2 / joint4 / joint5 滑块，单位 rad |

自动演示入口：

```bash
./scripts/isaaclab_display.sh --motion demo
```

当前是直接设置位姿和关节状态的展示模式，关闭重力和碰撞；移动、悬空和变构效果用于检查模型，不代表推力驱动的稳定飞行。滑块范围暂设为 ±0.6 rad，仅用于观察，不是已验证的机构安全范围。真正的动力学飞行与强化学习路径见 [report.md](report.md)。

脚本默认只启动一个模型，物理计算使用 CPU，RTX 用于窗口渲染。首次运行会自动把 URDF 转为 USD，放在 `build/isaaclab/assets/`；运行记录放在 `log/isaaclab/`。更换 STL 后加 `--rebuild` 重新转换。脚本使用 Isaac Sim 自带 Python，不需要激活 ROS 或 Conda。

无窗口检查并自动退出：

```bash
./scripts/isaaclab_display.sh --visualizer none --motion demo --steps 120
```

关闭 Isaac Sim 窗口或终端按 `Ctrl+C` 退出。安装位置不同时可设置 `ISAACLAB_ROOT`。

## RViz2 启动

在项目目录完成构建后启动：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select aeromorph_description
source install/local_setup.bash
export ROS_LOG_DIR="$PWD/log/ros"
ros2 launch aeromorph_description display.launch.py
```

无桌面调试时追加 `gui:=false rviz:=false`。

## 模型与操作

- 模型包含 9 个连杆和 8 个活动关节；默认使用 CAD 导出的零位姿态。
- `joint1`、`joint2`、`joint4`、`joint5` 是变构关节，`fly_joint1`～`fly_joint4` 是旋翼关节。
- 在 Joint State Publisher 窗口拖动滑块，RViz2 会同步更新。角度单位为弧度；Center 按钮恢复零位。
- RViz2 已设置 `base_link` 为固定坐标系，并配置网格、RobotModel 和观察视角；需要查看坐标系时勾选 TF。
- 保留 CAD 原有的几何、质量、惯量、关节连接关系及限位。导出的变构关节范围为约 ±π，仅用于模型展示，未验证实际机构的安全范围；滑块不会计算机构碰撞或驱动真实无人机。
- RViz2 和 Isaac Lab 当前均用于模型展示，尚未配置飞行动力学、飞控或硬件控制。

## 目录

```text
AeroMorph/
├── src/aeromorph_description/  # ROS 2 模型包
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── launch/                # ROS 2 启动文件
│   ├── meshes/                # 9 个原始 STL 网格
│   ├── rviz/                  # RViz2 显示配置
│   └── urdf/                  # 机器人结构与物理参数
├── isaaclab/scripts/          # Isaac Lab 模型展示入口
├── scripts/isaaclab_display.sh # 使用本机 Isaac 环境启动
├── report.md                 # 动力学及强化学习实施路径
├── build/                     # 自动生成的构建文件
├── install/                   # 自动生成的 ROS 2 安装空间
└── log/                       # 构建及运行日志
```


