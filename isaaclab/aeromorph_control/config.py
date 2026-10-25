"""配置集中读取；缺项和非物理数值在启动前报错。"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "src/aeromorph_description/urdf/AeroMorph.urdf"
JOINTS = ("joint1", "joint2", "joint3", "joint4")


def load_config(path=None, configuration=None, joints=None):
    path = Path(path) if path else ROOT / "isaaclab/configs/flight.yaml"
    cfg = yaml.safe_load(path.read_text())
    name = configuration or cfg["default_configuration"]
    if name not in cfg["configurations"]:
        raise ValueError(f"未知构型 {name}; 可选: {list(cfg['configurations'])}")
    cfg = deepcopy(cfg)
    cfg["configuration"] = name
    cfg["joints"] = deepcopy(cfg["configurations"][name])
    if joints:
        if len(joints) != 4:
            raise ValueError("关节顺序为 joint1 joint2 joint3 joint4，必须提供四个角度")
        cfg["joints"] = dict(zip(JOINTS, map(float, joints)))
    if set(cfg["joints"]) != set(JOINTS):
        raise ValueError("构型必须恰好指定 joint1/joint2/joint3/joint4")
    if not np.isfinite(list(cfg["joints"].values())).all():
        raise ValueError("关节角必须为有限数值，单位 rad")
    rotors = cfg["rotors"]
    if len(rotors) != 4 or len({r["link"] for r in rotors}) != 4:
        raise ValueError("必须指定四个独立旋翼 link")
    for rotor in rotors:
        for key in ("kappa_m", "max_thrust_n", "motor_tau_s"):
            val = rotor.get(key)
            if val is None or not np.isfinite(val) or val <= 0:
                raise ValueError(f"{rotor['link']}.{key} 必须为正的有限数")
        if rotor.get("reaction_sign") not in (-1, 1):
            raise ValueError(f"{rotor['link']} 缺少 +1/-1 反扭矩符号")
        if rotor.get("thrust_sign") not in (-1, 1):
            raise ValueError(f"{rotor['link']} 缺少 +1/-1 推力方向符号")
    sim = cfg["simulation"]
    if not np.isfinite(sim["dt_s"]) or sim["dt_s"] <= 0:
        raise ValueError("物理步长必须为正")
    for key in ("attitude_decimation", "position_decimation", "render_decimation", "log_decimation"):
        if not isinstance(sim[key], int) or sim[key] < 1:
            raise ValueError(f"{key} 必须为正整数")
    if sim["position_decimation"] % sim["attitude_decimation"]:
        raise ValueError("外环周期必须是姿态环周期的整数倍")
    for section, key in (("controller", "position_gain_s2"), ("controller", "velocity_gain_s"),
                         ("controller", "attitude_frequency_rad_s"),
                         ("controller", "attitude_damping_ratio"),
                         ("controller", "max_angular_accel_rad_s2")):
        val = np.asarray(cfg[section][key], dtype=float)
        if val.shape != (3,) or not np.isfinite(val).all() or (val <= 0).any():
            raise ValueError(f"{section}.{key} 必须为三个正数")
    for section, keys in {
        "command": ("horizontal_speed_m_s", "vertical_speed_m_s", "horizontal_accel_m_s2",
                    "vertical_accel_m_s2", "capture_speed_m_s", "capture_time_s"),
        "allocation": ("reference_length_m", "thrust_rate_n_s", "max_condition", "minimum_moment_n_m"),
    }.items():
        for key in keys:
            if not np.isfinite(cfg[section][key]) or cfg[section][key] <= 0:
                raise ValueError(f"{section}.{key} 必须为正数")
    a = cfg["allocation"]
    if not 0 <= a["minimum_hover_margin"] < 0.5 or not np.isfinite(a["regularization"]) or a["regularization"] < 0:
        raise ValueError("悬停余量或分配正则参数无效")
    if len(a["weights"]) != 4 or not np.isfinite(a["weights"]).all() or not np.all(np.asarray(a["weights"]) > 0):
        raise ValueError("分配权重必须为四个正数")
    if not 0 < cfg["controller"]["max_tilt_rad"] < cfg["limits"]["maximum_tilt_rad"] < 1.4:
        raise ValueError("姿态限制顺序无效")
    limits = cfg["limits"]
    if not np.isfinite(list(limits.values())).all() or min(limits.values()) <= 0:
        raise ValueError("运行限制必须为正的有限数")
    if not limits["minimum_height_m"] < sim["height_m"] < limits["maximum_height_m"]:
        raise ValueError("初始高度必须处于运行高度限制内")
    if not np.isfinite(sim["yaw_rad"]):
        raise ValueError("初始航向必须为有限数")
    morph = cfg["morphing"]
    for key, value in morph.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"morphing.{key} 必须为正的有限数")
    if not a["minimum_hover_margin"] <= morph["minimum_hover_margin"] < 0.5:
        raise ValueError("变构悬停余量必须不小于飞行余量且小于 0.5")
    if morph["pause_tilt_rad"] >= cfg["limits"]["maximum_tilt_rad"]:
        raise ValueError("变构暂停倾角必须小于飞行停止倾角")
    return cfg
