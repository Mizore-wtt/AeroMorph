"""质心平移外环 + SO(3) 姿态内环；不调用任何物理状态写入接口。"""

import numpy as np
from scipy.spatial.transform import Rotation


def clip_norm(value, limit):
    return value * min(1.0, limit / max(np.linalg.norm(value), 1e-12))


class FlightController:
    def __init__(self, model, cfg, initial_position, yaw):
        self.model, self.cfg = model, cfg
        self.H = Rotation.from_euler("z", yaw).as_matrix()  # H -> W，只保存初始航向。
        self.yaw = yaw
        self.reference_h = self.H.T @ np.asarray(initial_position)
        self.velocity_h = np.zeros(3)
        self.mode = np.array(["HOLD"]*3, dtype="<U8")
        self.capture_time = np.zeros(3)
        self.force_w = np.array([0.0, 0.0, model.mass*9.81])
        self.desired_rotation = self.H.copy()
        self.position_error = np.zeros(3)
        self.velocity_target_w = np.zeros(3)

    def position_update(self, position_w, velocity_w, command_h, dt):
        cfg = self.cfg["command"]
        gains = self.cfg["controller"]
        p, v = self.H.T @ position_w, self.H.T @ velocity_w
        target = np.asarray(command_h, dtype=float).copy()
        target[:2] = clip_norm(target[:2], 1) * cfg["horizontal_speed_m_s"]
        target[2] = np.clip(target[2], -1, 1) * cfg["vertical_speed_m_s"]
        for i in range(3):
            if abs(target[i]) > 1e-8:
                self.mode[i] = "VELOCITY"
                self.capture_time[i] = 0
            elif self.mode[i] == "VELOCITY":
                self.mode[i] = "BRAKE"
        change = target - self.velocity_h
        change[:2] = clip_norm(change[:2], cfg["horizontal_accel_m_s2"]*dt)
        change[2] = np.clip(change[2], -cfg["vertical_accel_m_s2"]*dt, cfg["vertical_accel_m_s2"]*dt)
        self.velocity_h += change
        for i in range(3):
            if self.mode[i] == "BRAKE":
                ready = abs(v[i]) < cfg["capture_speed_m_s"] and abs(self.velocity_h[i]) < 1e-5
                self.capture_time[i] = self.capture_time[i]+dt if ready else 0
                if self.capture_time[i] >= cfg["capture_time_s"]:
                    self.reference_h[i] = p[i]
                    self.mode[i] = "HOLD"
        error = np.clip(self.reference_h-p, -0.5, 0.5) * (self.mode == "HOLD")
        accel_h = np.array(gains["position_gain_s2"])*error + np.array(gains["velocity_gain_s"])*(self.velocity_h-v)
        accel_h += change/dt  # 平滑参考速度的前馈加速度。
        accel_h[:2] = clip_norm(accel_h[:2], cfg["horizontal_accel_m_s2"]*2)
        accel_h[2] = np.clip(accel_h[2], -cfg["vertical_accel_m_s2"]*2, cfg["vertical_accel_m_s2"]*2)
        accel = self.H @ accel_h
        accel[:2] = clip_norm(accel[:2], (9.81+accel[2])*np.tan(gains["max_tilt_rad"]))
        self.force_w = self.model.mass * (accel + [0, 0, 9.81])
        b3 = self.force_w / np.linalg.norm(self.force_w)
        heading = self.H[:, 0]
        b2 = np.cross(b3, heading)
        b2 /= np.linalg.norm(b2)
        self.desired_rotation = np.column_stack((np.cross(b2, b3), b2, b3))
        self.velocity_target_w = self.H @ self.velocity_h
        self.position_error = self.H @ error

    def attitude_update(self, R_wb, omega_b):
        gains = self.cfg["controller"]
        skew = self.desired_rotation.T @ R_wb - R_wb.T @ self.desired_rotation
        error = 0.5 * np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        frequency = np.asarray(gains["attitude_frequency_rad_s"])
        damping = np.asarray(gains["attitude_damping_ratio"])
        alpha = -frequency**2*error - 2*damping*frequency*omega_b
        limit = np.asarray(gains["max_angular_accel_rad_s2"])
        alpha = np.clip(alpha, -limit, limit)
        J = self.model.inertia  # 完整惯量含非对角项，每种构型自动重新计算。
        moment = J @ alpha + np.cross(omega_b, J @ omega_b)
        thrust = max(0.0, float(self.force_w @ R_wb[:, 2]))
        return np.r_[thrust, moment]

    def desired_attitude_error(self, R_wb):
        cosine = (np.trace(self.desired_rotation.T @ R_wb)-1)/2
        return float(np.arccos(np.clip(cosine, -1, 1)))
