"""物理含义明确的离线回归检查，不需要加载 Kit。"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aeromorph_control.allocator import Allocator
from aeromorph_control.config import load_config
from aeromorph_control.controller import FlightController
from aeromorph_control.keyboard import Keyboard
from aeromorph_control.model import inspect_model
from aeromorph_control.rotors import RotorDynamics


class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.model = inspect_model(cls.cfg)

    def test_urdf_mass_and_non_diagonal_inertia(self):
        model = inspect_model(load_config(joints=[0, 0, 0, 0]))
        self.assertAlmostEqual(model.mass, .693176806052, places=10)
        np.testing.assert_allclose(model.com, [-.141284185301, -.000043981815, .029702735067], atol=1e-11)
        np.testing.assert_allclose(model.inertia, [
            [.000492150738, .000000541301, -.000133015303],
            [.000000541301, .067737879211, .000000550252],
            [-.000133015303, .000000550252, .067917539315],
        ], atol=1e-11)
        np.testing.assert_allclose(model.axes, np.tile([0, 0, 1], (4, 1)), atol=1e-12)
        self.assertIn("link3", model.transforms)
        np.testing.assert_allclose(model.radii, np.full(4, .060024019214), atol=1e-10)

    def test_collinear_cad_zero_is_rejected(self):
        cfg = load_config(joints=[0, 0, 0, 0])
        model = inspect_model(cfg)
        np.testing.assert_allclose(model.hubs[:, 1] + model.com[1], 0, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "不可控制/病态.*rank=3"):
            Allocator(model, cfg)

    def test_feasible_configurations_hover_with_same_algorithm(self):
        thrusts = []
        for name in self.cfg["configurations"]:
            cfg = load_config(configuration=name)
            model = inspect_model(cfg)
            allocator = Allocator(model, cfg)
            np.testing.assert_allclose(allocator.B @ allocator.hover, [0, 0, model.mass*9.81, 0, 0, 0], atol=1e-12)
            self.assertTrue(np.all(allocator.hover > 0))
            self.assertEqual(allocator.rank, 4)
            thrusts.append(allocator.hover)
            if name != self.cfg["configuration"]:
                self.assertGreater(np.linalg.norm(model.com-self.model.com), .001)
                self.assertGreater(np.linalg.norm(model.inertia-self.model.inertia), .0001)
        self.assertGreater(np.linalg.norm(thrusts[0]-thrusts[1]), .5)

    def test_negative_hover_is_rejected_even_when_full_rank(self):
        cfg = load_config(joints=[.6, -.5, -.5, 0])
        with self.assertRaisesRegex(ValueError, "悬停推力"):
            Allocator(inspect_model(cfg), cfg)

    def test_insufficient_thrust_and_singular_geometry(self):
        cfg = deepcopy(self.cfg)
        for rotor in cfg["rotors"]:
            rotor["max_thrust_n"] = 1.0
        with self.assertRaisesRegex(ValueError, "悬停推力"):
            Allocator(self.model, cfg)
        model = deepcopy(self.model)
        model.hubs[:, 0] = [-1.5, -.5, .5, 1.5]
        model.hubs[:, 1] = 0
        with self.assertRaisesRegex(ValueError, "不可控制/病态"):
            Allocator(model, self.cfg)

    def test_nonparallel_axes_are_explicitly_rejected(self):
        model = deepcopy(self.model)
        model.axes[0] = [0, .1, np.sqrt(.99)]
        with self.assertRaisesRegex(ValueError, "平行"):
            Allocator(model, self.cfg)

    def test_allocation_bounds_and_rate_limit(self):
        allocator = Allocator(self.model, self.cfg)
        previous = allocator.previous.copy()
        command = allocator.solve([30, 1, 2, 1], .004)
        self.assertTrue(np.all(command >= 0) and np.all(command <= allocator.maximum))
        self.assertLessEqual(float(np.abs(command-previous).max()), 80*.004+1e-10)
        self.assertTrue(allocator.saturated)
        self.assertGreater(allocator.normalized_residual, .02)

    def test_heading_and_combined_speed_limits(self):
        controller = FlightController(self.model, self.cfg, [0, 0, 1.2], np.pi/2)
        for _ in range(100):
            controller.position_update(np.array([0, 0, 1.2]), np.zeros(3), [1, 1, 1], .02)
        target = controller.velocity_target_w
        self.assertLess(target[0], 0)
        self.assertGreater(target[1], 0)
        self.assertAlmostEqual(np.linalg.norm(target[:2]), .5)
        self.assertAlmostEqual(target[2], .3)

    def test_release_captures_current_position(self):
        controller = FlightController(self.model, self.cfg, [0, 0, 1.2], 0)
        controller.position_update(np.array([0, 0, 1.2]), np.zeros(3), [1, 0, 0], .02)
        for _ in range(100):
            controller.position_update(np.array([1.7, 0, 1.2]), np.zeros(3), [0, 0, 0], .02)
        self.assertEqual(controller.mode[0], "HOLD")
        self.assertAlmostEqual(controller.reference_h[0], 1.7)
        np.testing.assert_allclose(controller.force_w, [0, 0, self.model.mass*9.81], atol=1e-12)

    def test_attitude_feedback_corrects_roll(self):
        controller = FlightController(self.model, self.cfg, [0, 0, 1.2], 0)
        R = Rotation.from_euler("x", .05).as_matrix()
        _, *moment = controller.attitude_update(R, np.zeros(3))
        alpha = np.linalg.solve(self.model.inertia, moment)
        self.assertLess(alpha[0], 0)
        np.testing.assert_allclose(alpha[1:], 0, atol=1e-10)

    def test_rotor_lag_and_keyboard_hold(self):
        motors = RotorDynamics(self.cfg, np.ones(4))
        np.testing.assert_allclose(motors.step(np.zeros(4), .02), np.full(4, np.exp(-1)))
        keys = Keyboard()
        keys.keys = {"W", "R"}
        np.testing.assert_equal(keys.command(), [1, 0, 1])
        keys.hold()
        np.testing.assert_equal(keys.command(), [0, 0, 0])
        keys.keys.clear()
        keys.command()
        keys.keys.add("A")
        np.testing.assert_equal(keys.command(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
