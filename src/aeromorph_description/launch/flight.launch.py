'''
在 Isaaclab 中启动仿真飞行的 launch 文件
author:GPT-6.1 Sol
reviewer:wutong
date:10.25
'''

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def default_workspace():
    # 寻找项目目录
    for parent in Path(__file__).resolve().parents:
        if (parent / "isaaclab/scripts/fly.py").is_file():
            return str(parent)
    return str(Path.cwd())


def start(context):
    # 启动飞行仿真
    def value(name):
        return LaunchConfiguration(name).perform(context)

    # 设置脚本与环境
    root = Path(value("workspace")).expanduser().resolve()
    lab = Path(value("isaaclab_root")).expanduser().resolve()
    python = lab / "_isaac_sim/python.sh"
    script = root / "isaaclab/scripts/fly.py"
    if not python.is_file() or not script.is_file():
        raise RuntimeError("请正确设置 workspace 和 isaaclab_root；需要 Isaac Sim 自带 python.sh")
    environment = dict(os.environ)
    for key in ("CONDA_PREFIX", "VIRTUAL_ENV", "PYTHONHOME", "LD_LIBRARY_PATH", "PYTHONEXE"):
        environment.pop(key, None)
    environment["PYTHONPATH"] = str(lab / "source/isaaclab")

    # 构建启动命令
    command = [str(python), str(script), 
               "--configuration", value("configuration"),
               "--device", value("device"), 
               "--duration", value("duration"),
               "--scenario", value("scenario")
               ]
    for key in ("config", "yaw"):
        if value(key):
            command.extend(["--"+key, value(key)])
    if value("joints"):
        angles = value("joints").split()
        if len(angles) != 4:
            raise RuntimeError("joints 参数必须是四个空格分隔的角度，顺序 joint1 joint2 joint3 joint4")
        command.extend(["--joints", *angles])
    if value("headless") == "true":
        command.extend(["--visualizer", "none"])
    return [ExecuteProcess(cmd=command, cwd=str(root), env=environment, output="screen",
                           sigterm_timeout="20", sigkill_timeout="10")]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("workspace", default_value=default_workspace()),
        DeclareLaunchArgument("isaaclab_root",
                              default_value=os.environ.get("ISAACLAB_ROOT", str(Path.home()/"IsaacLab"))),
        DeclareLaunchArgument("configuration", default_value="nominal", description="flight.yaml 中的构型名称"),
        DeclareLaunchArgument("config", default_value="", description="可选 YAML 配置绝对路径"),
        DeclareLaunchArgument("joints", default_value="", description="可选四个关节角 [rad]，空格分隔"),
        DeclareLaunchArgument("yaw", default_value="", description="可选初始航向 [rad]"),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("headless", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("duration", default_value="0", description="运行仿真秒数；0 为持续运行"),
        DeclareLaunchArgument("scenario", default_value="keyboard",
                              choices=["keyboard", "hover", "sequence", "morph", "morph_sequence", "motors_off"]),
        OpaqueFunction(function=start),
    ])
