"""缓存运动学和受约束的在线变构参考；不写入仿真位姿或关节状态。"""

from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .allocator import Allocator
from .config import JOINTS, URDF
from .model import Snapshot, origin


class Kinematics:
    """启动时缓存 URDF 与网格，飞行中只更新小矩阵。"""

    def __init__(self, cfg, initial):
        self.cfg, self.initial = cfg, initial
        tree = ET.parse(URDF).getroot()
        self.joints = list(tree.findall("joint"))
        self.links = {link.get("name"): link for link in tree.findall("link")}
        self.root = (set(self.links) - {j.find("child").get("link") for j in self.joints}).pop()
        self.properties = {}
        for name, link in self.links.items():
            inertial = link.find("inertial")
            T = origin(inertial.find("origin"))
            a = {key: float(value) for key, value in inertial.find("inertia").attrib.items()}
            I = np.array([[a["ixx"], a["ixy"], a["ixz"]],
                          [a["ixy"], a["iyy"], a["iyz"]],
                          [a["ixz"], a["iyz"], a["izz"]]])
            self.properties[name] = (float(inertial.find("mass").get("value")), T[:3, 3],
                                     T[:3, :3] @ I @ T[:3, :3].T)
        self.edges = []
        self.limits = {}
        for j in self.joints:
            name = j.get("name")
            axis_element = j.find("axis")
            axis = (np.fromstring(axis_element.get("xyz"), sep=" ")
                    if axis_element is not None else np.array([0, 0, 1.]))
            axis /= np.linalg.norm(axis)
            self.edges.append((name, j.find("parent").get("link"), j.find("child").get("link"),
                               origin(j.find("origin")), axis))
            if name in JOINTS:
                limit = j.find("limit")
                self.limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
        self.rotor_axes = []
        for rotor in cfg["rotors"]:
            edge = next(e for e in self.edges if e[2] == rotor["link"])
            self.rotor_axes.append(edge[4] * rotor["thrust_sign"])

    def at(self, angles):
        """由实际关节角求模型快照，坐标原点取整机质心、轴平行根连杆。"""
        if not isinstance(angles, dict):
            angles = dict(zip(JOINTS, angles))
        if not np.isfinite(list(angles.values())).all():
            raise ValueError("关节角必须为有限数")
        for name, (low, high) in self.limits.items():
            if not low <= angles[name] <= high:
                raise ValueError(f"{name} 超出 URDF 限位 [{low:.3f}, {high:.3f}]")
        transforms = {self.root: np.eye(4)}
        pending = self.edges.copy()
        while pending:
            for edge in pending[:]:
                name, parent, child, T, axis = edge
                if parent not in transforms:
                    continue
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(axis * angles.get(name, 0.)).as_matrix()
                transforms[child] = transforms[parent] @ T @ motion
                pending.remove(edge)
        centers, inertias, masses = [], [], []
        for name, (m, c, I) in self.properties.items():
            T = transforms[name]
            masses.append(m)
            centers.append(T[:3, 3] + T[:3, :3] @ c)
            inertias.append(T[:3, :3] @ I @ T[:3, :3].T)
        masses, centers = np.array(masses), np.array(centers)
        com = np.sum(masses[:, None] * centers, axis=0) / masses.sum()
        J = np.zeros((3, 3))
        for m, c, I in zip(masses, centers, inertias):
            r = c - com
            J += I + m * ((r @ r) * np.eye(3) - np.outer(r, r))
        hubs, axes = [], []
        for rotor, axis in zip(self.cfg["rotors"], self.rotor_axes):
            T = transforms[rotor["link"]]
            hubs.append(T[:3, 3] - com)
            axes.append(T[:3, :3] @ axis)
        return Snapshot(float(masses.sum()), com, J, transforms, self.initial.links,
                        np.array(hubs), np.array(axes), self.initial.radii, self.initial.mesh_hash)


class MorphPlanner:
    """滑块请求先验证完整运动路径，再生成有限速度/加速度的舵机参考。"""

    def __init__(self, cfg, kinematics):
        self.cfg, self.kinematics = cfg, kinematics
        self.settings = cfg["morphing"]
        self.position = np.array([cfg["joints"][name] for name in JOINTS], dtype=float)
        self.velocity = np.zeros(4)
        self.target = self.position.copy()
        self.requested = self.target.copy()
        self.status = "HOLD"
        self.reason = ""
        self.guard_cfg = deepcopy(cfg)
        self.guard_cfg["allocation"]["minimum_hover_margin"] = self.settings["minimum_hover_margin"]
        self.rejections = 0
        self.accepted = 0

    def _advance(self, q, v, target, dt, allowed=True):
        a, vmax = self.settings["max_accel_rad_s2"], self.settings["max_speed_rad_s"]
        error = target-q
        distance, speed = np.max(np.abs(error)), np.max(np.abs(v))
        direction = error/max(distance, 1e-12)
        # 用同一个标量进度协调四关节，避免各轴独立到位造成危险的弯曲路径。
        # 移动中换方向先沿原方向减速，再从停止点进入新直线路径。
        aligned = speed < 1e-10 or np.max(np.abs(v/speed-direction)) < 1e-5
        desired = direction * min(vmax, np.sqrt(2*a*distance), distance*2)
        if not allowed or not aligned:
            desired = np.zeros(4)
        change = desired-v
        next_v = v + change*min(1., a*dt/max(np.max(np.abs(change)), 1e-12))
        return q + (v+next_v)*(.5*dt), next_v

    def _stop(self, q, v):
        return q + v*np.max(np.abs(v))/(2*self.settings["max_accel_rad_s2"])

    def _check(self, q, full=False):
        return Allocator(self.kinematics.at(q), self.guard_cfg, check_moments=full)

    def request(self, target):
        """验证新目标和路径；拒绝时保留上一次接受的目标，不中断飞行闭环。"""
        target = np.asarray(target, dtype=float)
        if target.shape != (4,) or not np.isfinite(target).all():
            self.status, self.reason = "REJECTED", "需要四个有限关节角"
            self.rejections += 1
            return False
        if np.array_equal(target, self.requested):
            return False
        self.requested = target.copy()
        try:
            self._check(target, full=True)
            # 用同一个参考生成器预演到达过程，检查加减速/反向造成的弯曲路径。
            q, v, checked = self.position.copy(), self.velocity.copy(), self.position.copy()
            horizon = 4 * np.max(np.abs(target-q))/self.settings["max_speed_rad_s"] + 10
            preview_dt = .02
            for _ in range(int(horizon/preview_dt)+1):
                q, v = self._advance(q, v, target, preview_dt)
                if np.max(np.abs(q-checked)) >= self.settings["path_resolution_rad"]:
                    self._check(q)
                    # 也核验此处暂停时的制动距离。
                    stop = self._stop(q, v)
                    self._check(stop)
                    checked = q.copy()
                if np.max(np.abs(q-target)) < 1e-5 and np.max(np.abs(v)) < 1e-4:
                    break
            else:
                raise ValueError("变构路径未能收敛")
        except ValueError as exc:
            self.status, self.reason = "REJECTED", str(exc)
            self.rejections += 1
            return False
        self.target = target.copy()
        self.status, self.reason = "MOVING", ""
        self.accepted += 1
        return True

    def step(self, dt, actual, allowed=True):
        """推进舵机参考；失稳或跟踪落后时减速，始终返回位置/速度目标。"""
        tracking = np.max(np.abs(np.asarray(actual)-self.position))
        allowed = allowed and tracking < self.settings["maximum_tracking_error_rad"]
        q, v = self._advance(self.position, self.velocity, self.target, dt, allowed)
        try:
            self._check(q)
            self._check(self._stop(q, v))
        except ValueError as exc:
            # 极端边界的最后一道保护：保留最后的可行参考，不继续穿越退化区域。
            self.velocity[:] = 0
            self.target = self.position.copy()
            self.status, self.reason = "BLOCKED", str(exc)
            return self.position.copy(), self.velocity.copy()
        self.position, self.velocity = q, v
        if self.status not in ("REJECTED", "BLOCKED"):
            if not allowed:
                self.status = "PAUSED"
            elif np.max(np.abs(q-self.target)) < 1e-4 and np.max(np.abs(v)) < 1e-3:
                self.status = "HOLD"
            else:
                self.status = "MOVING"
        return q.copy(), v.copy()
