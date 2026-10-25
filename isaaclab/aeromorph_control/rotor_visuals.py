"""旋翼外观动画：按各桨实际推力自动分档，不修改飞行动力学。"""

import numpy as np


# 单位为显示转数/s；为原动画的两倍，不代表真实电机转速。
DISPLAY_SPEEDS_RPS = np.array([0., 1.2, 2.8, 4.8])
DISPLAY_SPEED_LABELS = ("Stopped", "Low", "Medium", "High")
# 按各电机最大推力归一化；阈值附近保留 2% 回差，防止显示档位抖动。
THRUST_THRESHOLDS = (0.30, 0.50)
LEVEL_HYSTERESIS = 0.02


class RotorVisuals:
    def __init__(self, stage, cfg, axes, prim_path="/World/AeroMorph"):
        from pxr import Gf, UsdGeom

        self.axes = np.asarray(axes, dtype=float)
        # 机体反扭矩与桨叶自转相反，沿配置的正推力轴确定显示旋向。
        self.directions = -np.array([rotor["reaction_sign"] for rotor in cfg["rotors"]])
        self.maximum = np.array([rotor["max_thrust_n"] for rotor in cfg["rotors"]])
        self.levels = np.zeros(len(cfg["rotors"]), dtype=int)
        self.angles = np.zeros(len(cfg["rotors"]))
        self.operations = []
        for rotor in cfg["rotors"]:
            link = stage.GetPrimAtPath(f"{prim_path}/{rotor['link']}")
            operations = []
            for prim in link.GetChildren():
                if not prim.GetName().startswith("visual_"):
                    continue
                visual = UsdGeom.Xformable(prim)
                original = visual.GetOrderedXformOps()
                spin = visual.AddOrientOp(opSuffix="rotorSpin")
                spin.Set(Gf.Quatf(1))
                # 自转放在原 visual 变换外侧，绕 link 原点（桨毂）而非网格质心旋转。
                visual.SetXformOpOrder([spin, *original])
                operations.append(spin)
            if not operations:
                raise ValueError(f"旋翼 {rotor['link']} 缺少可动画的 visual 子节点")
            self.operations.append(operations)

    def advance(self, dt, thrust):
        """只在渲染前调用；dt 是两帧间的仿真时间，停桨后保持当前角度。"""
        from pxr import Gf

        thrust = np.asarray(thrust)
        ratio = np.clip(thrust/self.maximum, 0, 1)
        self.levels = np.maximum(self.levels, 1)
        # 每桨独立判断，推力大幅变化时可直接跨档；切档仅改变角速度，保留累计转角。
        for level, boundary in enumerate(THRUST_THRESHOLDS, start=2):
            self.levels[(self.levels < level) & (ratio >= boundary+LEVEL_HYSTERESIS)] = level
            self.levels[(self.levels >= level) & (ratio < boundary-LEVEL_HYSTERESIS)] = level-1
        self.levels[thrust <= 1e-3] = 0
        turns_per_second = DISPLAY_SPEEDS_RPS[self.levels]
        self.angles = (self.angles + self.directions*360*turns_per_second*dt) % 360
        for angle, axis, operations in zip(self.angles, self.axes, self.operations):
            rotation = Gf.Rotation(Gf.Vec3d(*axis.tolist()), float(angle)).GetQuat()
            value = Gf.Quatf(rotation)
            for operation in operations:
                operation.Set(value)
