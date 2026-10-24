'''
在 Isaaclab 中展示机器人模型的 launch 文件
author:GPT-6.1 high
review:wutong
date:10.24
'''

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _default_workspace():
    # 寻找项目的工作空间(AeroMorph)
    for parent in Path(__file__).resolve().parents:
        if (parent / 'isaaclab/scripts/display.py').is_file():
            return str(parent)
    return str(Path.cwd())

def _start_display(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    # 设置脚本和环境的路径
    workspace = Path(value('workspace')).expanduser().resolve()
    isaac_root = Path(value('isaaclab_root')).expanduser().resolve()
    script = workspace / 'isaaclab/scripts/display.py'
    python = isaac_root / '_isaac_sim/python.sh'
    if not script.is_file():
        raise RuntimeError(
            f'Display script not found: {script}. Set workspace:=/path/to/AeroMorph.'
        )
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(
            f'Isaac Sim Python not found: {python}. Set isaaclab_root:=/path/to/IsaacLab.'
        )
    # 处理环境冲突：ROS 2 使用 Python 3.10; Isaac Sim 6.0.1 使用 Python 3.12.
    environment = dict(os.environ)
    for name in ('CONDA_PREFIX', 'VIRTUAL_ENV', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'PYTHONEXE'):
        environment.pop(name, None)
    environment['PYTHONPATH'] = str(isaac_root / 'source/isaaclab')

    # 只启动模型展示；关节滑块在 display.py 中创建。
    return [
        ExecuteProcess(
            cmd=[str(python), str(script)], cwd=str(workspace), env=environment, output='screen',
            sigterm_timeout='20', sigkill_timeout='10',
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('workspace', default_value=_default_workspace(),
                              description='AeroMorph project root containing isaaclab/scripts/display.py.'),
        DeclareLaunchArgument('isaaclab_root',
                              default_value=os.environ.get('ISAACLAB_ROOT', str(Path.home() / 'IsaacLab')),
                              description='Isaac Lab installation containing _isaac_sim/python.sh.'),
        OpaqueFunction(function=_start_display),
    ])
