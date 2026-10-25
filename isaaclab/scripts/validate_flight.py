"""对飞行 CSV 做验收；失败返回非零退出码，便于重复验证。"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aeromorph_control.config import JOINTS


def validate_morph(array, rows, t, parameters):
    if t[-1] < 37:
        raise ValueError("变构验收至少需要运行 38 s，覆盖三构型、拒绝和末段稳定")
    joints, reference = array("joint_position", 4), array("joint_reference", 4)
    rate = array("joint_velocity", 4)
    if not np.isfinite(np.c_[joints, reference, rate]).all():
        raise ValueError("关节日志包含 NaN/Inf")
    morph = parameters["morphing"]
    if np.abs(joints-reference).max() > morph["maximum_tracking_error_rad"]:
        raise ValueError("关节跟踪误差超限")
    reference_velocity = np.diff(reference, axis=0)/np.diff(t)[:, None]
    if np.abs(reference_velocity).max() > morph["max_speed_rad_s"]+1e-5:
        raise ValueError("变构参考速度超限")
    reference_acceleration = np.diff(reference_velocity, axis=0)/np.diff(t)[1:, None]
    if np.abs(reference_acceleration).max() > morph["max_accel_rad_s2"]*1.05+1e-4:
        raise ValueError("变构参考加速度超限")
    for start, end, name in ((11, 13, "swept"), (23, 24.5, "asymmetric")):
        target = np.array([parameters["configurations"][name][j] for j in JOINTS])
        if np.abs(joints[(t > start) & (t < end)]-target).max() > .015:
            raise ValueError(f"未实际到达 {name} 构型")
    target = np.array([parameters["joints"][j] for j in JOINTS])
    if np.abs(joints[t > t[-1]-2]-target).max() > .015 or rows[-1]["morph_status"] != "HOLD":
        raise ValueError("末段未恢复启动构型或变构未结束")
    rejections = int(rows[-1]["morph_rejections"])
    if rejections != 1:
        raise ValueError(f"预期只拒绝一次不可飞零位请求，实际 {rejections} 次")
    return {"max_joint_tracking_error_rad": float(np.abs(joints-reference).max()),
            "max_joint_reference_rate_rad_s": float(np.abs(reference_velocity).max()),
            "max_joint_reference_accel_rad_s2": float(np.abs(reference_acceleration).max()),
            "rejected_requests": rejections}

def validate(folder, scenario):
    parameters = json.loads((folder / "configuration.json").read_text())["parameters"]
    summary = json.loads((folder / "summary.json").read_text())
    rows = list(csv.DictReader((folder / "flight.csv").open()))
    if not summary["completed"] or not rows:
        raise ValueError(f"仿真未完成: {summary['error']}")
    array = lambda key, n: np.array([[float(row[f"{key}_{i}"]) for i in range(n)] for row in rows])
    p, v, rpy, thrust = array("position", 3), array("velocity", 3), array("rpy", 3), array("thrust_actual", 4)
    t = np.array([float(row["time_s"]) for row in rows])
    if not np.isfinite(np.c_[p, v, rpy, thrust]).all():
        raise ValueError("日志含 NaN/Inf")
    maximum = np.array([r["max_thrust_n"] for r in parameters["rotors"]])
    if (thrust < -1e-8).any() or (thrust > maximum+1e-8).any():
        raise ValueError("旋翼推力越界")
    if scenario == "motors_off":
        if p[0, 2]-p[-1, 2] < .2 or v[-1, 2] > -1:
            raise ValueError("关闭旋翼后没有按重力下落")
        return {"free_fall_drop_m": float(p[0, 2]-p[-1, 2]), "final_vertical_speed_m_s": float(v[-1, 2])}
    if summary["tail_max_speed_m_s"] > .05 or max(summary["tail_position_span_m"]) > .03:
        raise ValueError("末段速度/位置漂移未达到悬停标准")
    if summary["tail_max_attitude_error_rad"] > .03 or any(summary["last"]["mode"][i] != "HOLD" for i in range(3)):
        raise ValueError("末段姿态或模式未恢复悬停")
    yaw_error = np.arctan2(np.sin(rpy[:, 2]-parameters["simulation"]["yaw_rad"]),
                           np.cos(rpy[:, 2]-parameters["simulation"]["yaw_rad"]))
    if np.abs(yaw_error).max() > .08:
        raise ValueError("航向偏差过大")
    morph_result = validate_morph(array, rows, t, parameters) if scenario in ("morph", "morph_sequence") else {}
    if scenario == "morph":
        target = np.array([0, 0, parameters["simulation"]["height_m"]])
        rmse = float(np.sqrt(np.mean(np.sum((p-target)**2, axis=1))))
        if rmse > .1:
            raise ValueError(f"变构悬停位置 RMSE {rmse:.3f} m > 0.10 m")
        return {"position_rmse_m": rmse, **morph_result}
    if scenario == "hover":
        target = np.array([0, 0, parameters["simulation"]["height_m"]])
        rmse = float(np.sqrt(np.mean(np.sum((p-target)**2, axis=1))))
        if rmse > .1:
            raise ValueError(f"悬停位置 RMSE {rmse:.3f} m > 0.10 m")
        return {"position_rmse_m": rmse, "max_yaw_error_rad": float(np.abs(yaw_error).max()),
                "tail_max_speed_m_s": summary["tail_max_speed_m_s"]}
    if t[-1] < 30:
        raise ValueError("组合动作验收至少需要运行 30 s")
    H = Rotation.from_euler("z", parameters["simulation"]["yaw_rad"]).as_matrix()
    velocity_h = v @ H
    # 进入每段指令后等待 1.5 s，核对初始航向坐标系下的实际速度符号和幅值。
    for start, stop, direction in [(3, 6, [1, 0, 1]), (10, 13, [0, 1, 0]), (17, 20, [-1, -1, -1])]:
        measured = velocity_h[(t > start+1.5) & (t < stop)].mean(axis=0)
        for i, sign in enumerate(direction):
            if sign and measured[i]*sign < .15:
                raise ValueError(f"动作 {start}s 的第 {i} 轴方向/速度错误: {measured}")
    if summary["max_tilt_rad"] < .01:
        raise ValueError("水平移动未产生可测的真实倾斜")
    if summary["saturation_fraction"] > .1:
        raise ValueError("推力分配饱和占比过高")
    return {"tail_max_speed_m_s": summary["tail_max_speed_m_s"],
            "tail_position_span_m": summary["tail_position_span_m"],
            "max_tilt_deg": float(np.degrees(summary["max_tilt_rad"])),
            "saturation_fraction": summary["saturation_fraction"], **morph_result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--scenario", choices=("hover", "sequence", "morph", "morph_sequence", "motors_off"), default="sequence")
    args = parser.parse_args()
    try:
        print(json.dumps({"PASS": str(args.run_dir), **validate(args.run_dir, args.scenario)}, indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"验收失败: {exc}\n")
