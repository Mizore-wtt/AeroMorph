"""四路电机的一阶推力响应；多刚体受力实现位于 articulated.py。"""

import numpy as np


class RotorDynamics:
    def __init__(self, cfg, initial):
        self.actual = initial.copy()
        self.tau = np.array([r["motor_tau_s"] for r in cfg["rotors"]])
        self.maximum = np.array([r["max_thrust_n"] for r in cfg["rotors"]])
        self.reaction = np.array([r["reaction_sign"]*r["kappa_m"] for r in cfg["rotors"]])

    def step(self, command, dt):
        command = np.clip(command, 0, self.maximum)
        self.actual = command + (self.actual-command)*np.exp(-dt/self.tau)
        return self.actual
