"""记录实际物理状态和控制量，保留可复核的 CSV 与运行摘要。"""

import csv
import json

import numpy as np
from scipy.spatial.transform import Rotation


class Telemetry:
    def __init__(self, folder, cfg, model, allocator):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "configuration.json").write_text(json.dumps({
            "parameters": cfg, "mass_kg": model.mass, "urdf_com_m": model.com.tolist(),
            "inertia_kg_m2": model.inertia.tolist(), "hubs_com_m": model.hubs.tolist(),
            "axes": model.axes.tolist(), **allocator.report(),
        }, indent=2, ensure_ascii=False)+"\n")
        self.file = (folder / "flight.csv").open("w", newline="")
        self.writer = csv.writer(self.file)
        groups = [("position", 3), ("velocity", 3), ("rpy", 3), ("omega", 3),
                  ("command_h", 3), ("target_velocity", 3), ("thrust_command", 4), ("thrust_actual", 4),
                  ("quaternion_xyzw", 4), ("desired_T_Mxyz", 4), ("actual_Fxyz_Mxyz", 6),
                  ("joint_position", 4), ("joint_velocity", 4), ("joint_reference", 4),
                  ("joint_target", 4), ("morph_compensation", 3)]
        self.writer.writerow(
            ["time_s"] + [f"{key}_{i}" for key, n in groups for i in range(n)]
            + ["mode_x", "mode_y", "mode_z", "saturated", "allocation_residual", "attitude_error_rad",
               "morph_status", "morph_rejections"]
        )
        self.tail = []
        self.count = self.saturated = 0
        self.min_position = np.full(3, np.inf)
        self.max_position = np.full(3, -np.inf)
        self.max_speed = self.max_tilt = self.max_attitude_error = 0.0
        self.last = None

    def record(self, t, p, v, rpy, omega, command, controller, allocator, thrust, actual, R,
               vehicle, planner):
        attitude_error = controller.desired_attitude_error(R)
        self.writer.writerow([t, *p, *v, *rpy, *omega, *command, *controller.velocity_target_w,
                              *thrust, *actual, *Rotation.from_matrix(R).as_quat(),
                              *allocator.desired, *(allocator.B @ actual),
                              *vehicle.joints, *vehicle.joint_velocity, *planner.position, *planner.target,
                              *vehicle.compensation, *controller.mode, int(allocator.saturated),
                              allocator.normalized_residual, attitude_error, planner.status, planner.rejections])
        self.count += 1
        self.saturated += allocator.saturated
        self.min_position = np.minimum(self.min_position, p)
        self.max_position = np.maximum(self.max_position, p)
        self.max_speed = max(self.max_speed, float(np.linalg.norm(v)))
        self.max_tilt = max(self.max_tilt, float(np.arccos(np.clip(R[2, 2], -1, 1))))
        self.max_attitude_error = max(self.max_attitude_error, attitude_error)
        self.last = {"time_s": t, "position_m": p.tolist(), "velocity_m_s": v.tolist(),
                     "rpy_rad": rpy.tolist(), "mode": controller.mode.tolist()}
        self.tail.append(np.r_[t, p, np.linalg.norm(v), attitude_error])
        self.tail = [row for row in self.tail if t-row[0] <= 2]
        if self.count % 50 == 0:
            self.file.flush()

    def close(self, error=None):
        self.file.close()
        tail = np.array(self.tail)
        result = {"completed": error is None, "error": error, "last": self.last,
                  "samples": self.count, "saturation_fraction": self.saturated/max(1, self.count)}
        if len(tail):
            result.update({"position_min_m": self.min_position.tolist(),
                           "position_max_m": self.max_position.tolist(),
                           "max_speed_m_s": self.max_speed, "max_tilt_rad": self.max_tilt,
                           "max_attitude_error_rad": self.max_attitude_error,
                           "tail_position_span_m": np.ptp(tail[:, 1:4], axis=0).tolist(),
                           "tail_max_speed_m_s": float(tail[:, 4].max()),
                           "tail_max_attitude_error_rad": float(tail[:, 5].max())})
        (self.folder / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
        return result
