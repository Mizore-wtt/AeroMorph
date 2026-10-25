"""构型相关六维映射、悬停可行性检查和带约束四路推力分配。"""

import numpy as np
from scipy.optimize import linprog, lsq_linear


class Allocator:
    def __init__(self, model, cfg, check_moments=True):
        self.model, self.cfg = model, cfg["allocation"]
        self.maximum = np.array([r["max_thrust_n"] for r in cfg["rotors"]])
        reaction = np.array([r["reaction_sign"] * r["kappa_m"] for r in cfg["rotors"]])
        self.B = np.vstack((model.axes.T, (np.cross(model.hubs, model.axes) + reaction[:, None]*model.axes).T))
        if not np.allclose(model.axes, [0, 0, 1], atol=1e-6):
            raise ValueError(f"当前控制器仅支持已验证的平行 +Z 推力；B6 秩={np.linalg.matrix_rank(self.B)}")
        for i in range(4):
            for j in range(i):
                if np.linalg.norm(model.hubs[i]-model.hubs[j]) < model.radii[i]+model.radii[j]+0.005:
                    raise ValueError(f"旋翼 {j+1}/{i+1} 扫掠圆盘相交，拒绝此构型")
        self.A = self.B[[2, 3, 4, 5]]
        weight = model.mass * 9.81
        self.scale = np.array([weight] + [weight*self.cfg["reference_length_m"]]*3)
        normalized = self.A * self.maximum[None, :] / self.scale[:, None]
        self.rank = int(np.linalg.matrix_rank(normalized, tol=1e-7))
        self.condition = float(np.linalg.cond(normalized))
        if self.rank != 4 or self.condition > self.cfg["max_condition"]:
            raise ValueError(f"不可控制/病态构型: rank={self.rank}, normalized condition={self.condition:.1f}")
        self.hover = np.linalg.solve(self.A, np.array([weight, 0, 0, 0]))
        margin = np.minimum(self.hover/self.maximum, 1-self.hover/self.maximum)
        if margin.min() < self.cfg["minimum_hover_margin"]:
            raise ValueError(f"悬停推力不可行或余量不足: thrust={self.hover.tolist()}, margins={margin.tolist()}")
        # 保持总推力和另外两轴力矩，求每轴正负控制余量。
        self.moments = []
        for row in (range(1, 4) if check_moments else []):
            keep = [i for i in range(4) if i != row]
            constraints = dict(A_eq=self.A[keep], b_eq=np.array([weight, 0, 0, 0])[keep],
                               bounds=list(zip(np.zeros(4), self.maximum)), method="highs")
            lo = linprog(self.A[row], **constraints)
            hi = linprog(-self.A[row], **constraints)
            if not lo.success or not hi.success:
                raise ValueError("控制余量检查求解失败")
            self.moments.append([float(lo.fun), float(-hi.fun)])
        if self.moments and min(min(-a, b) for a, b in self.moments) < self.cfg["minimum_moment_n_m"]:
            raise ValueError(f"悬停附近缺少双向力矩余量: {self.moments}")
        self.previous = self.hover.copy()
        self.desired = self.A @ self.hover
        self.saturated = False
        self.residual = np.zeros(4)
        self.normalized_residual = 0.0
        w = np.asarray(self.cfg["weights"]) / self.scale
        reg = np.sqrt(self.cfg["regularization"]) / self.maximum
        self.matrix = np.vstack((w[:, None]*self.A, np.diag(reg)))
        self.weights, self.reg = w, reg

    def update_model(self, model, cfg):
        """按实际构型刷新映射，保留推力历史，避免变构时重置变化率限制。"""
        updated = Allocator(model, cfg, check_moments=False)
        for key in ("model", "B", "A", "scale", "rank", "condition", "hover", "matrix", "weights", "reg"):
            setattr(self, key, getattr(updated, key))

    def solve(self, desired, dt):
        desired = np.asarray(desired, dtype=float)
        if not np.isfinite(desired).all():
            raise RuntimeError("分配输入包含非有限数")
        self.desired = desired.copy()
        delta = self.cfg["thrust_rate_n_s"] * dt
        lower = np.maximum(0, self.previous-delta)
        upper = np.minimum(self.maximum, self.previous+delta)
        rhs = np.r_[self.weights*desired, self.reg*self.previous]
        result = np.linalg.lstsq(self.matrix, rhs, rcond=None)[0]
        if np.any(result < lower) or np.any(result > upper):
            fit = lsq_linear(self.matrix, rhs, bounds=(lower, upper), method="bvls", tol=1e-10)
            if not fit.success:
                raise RuntimeError(f"推力分配失败: {fit.message}")
            result = fit.x
        self.previous = result
        self.residual = self.A @ result - desired
        self.normalized_residual = float(np.linalg.norm(self.residual/self.scale))
        self.saturated = bool(np.any(result < 0.001) or np.any(result > self.maximum-0.001)
                              or self.normalized_residual > 0.02)
        return result.copy()

    def report(self):
        return {"rank": self.rank, "normalized_condition": self.condition,
                "hover_thrust_n": self.hover.tolist(), "moment_ranges_n_m": self.moments,
                "B6": self.B.tolist()}
