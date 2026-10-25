"""在线模型、变构路径与有限舵机参考的物理回归。"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aeromorph_control.allocator import Allocator
from aeromorph_control.config import JOINTS, load_config
from aeromorph_control.model import inspect_model
from aeromorph_control.morphology import Kinematics, MorphPlanner


class MorphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.initial = inspect_model(cls.cfg)
        cls.kinematics = Kinematics(cls.cfg, cls.initial)

    def test_cached_model_matches_urdf_aggregation(self):
        for name in self.cfg["configurations"]:
            cfg = load_config(configuration=name)
            expected = inspect_model(cfg)
            actual = self.kinematics.at(cfg["joints"])
            for attr in ("mass", "com", "inertia", "hubs", "axes", "radii"):
                np.testing.assert_allclose(getattr(actual, attr), getattr(expected, attr), atol=1e-12)

    def test_invalid_target_does_not_change_flight_reference(self):
        planner = MorphPlanner(self.cfg, self.kinematics)
        before = planner.position.copy()
        self.assertFalse(planner.request(np.zeros(4)))
        self.assertIn("rank=3", planner.reason)
        np.testing.assert_equal(planner.target, before)
        self.assertFalse(planner.request([np.nan]*4))
        self.assertFalse(planner.request([10]*4))

    def test_feasible_endpoints_do_not_allow_crossing_singular_path(self):
        planner = MorphPlanner(self.cfg, self.kinematics)
        opposite = -planner.position
        Allocator(self.kinematics.at(opposite), self.cfg)
        self.assertFalse(planner.request(opposite))
        np.testing.assert_equal(planner.target, planner.position)

    def test_rate_acceleration_tracking_and_pause(self):
        planner = MorphPlanner(self.cfg, self.kinematics)
        goal = np.array([self.cfg["configurations"]["swept"][j] for j in JOINTS])
        self.assertTrue(planner.request(goal), planner.reason)
        actual = planner.position.copy()
        dt, previous_velocity = .02, np.zeros(4)
        for index in range(500):
            allowed = not 80 <= index < 140
            q, velocity = planner.step(dt, actual, allowed)
            self.assertLessEqual(np.abs(velocity).max(), self.cfg["morphing"]["max_speed_rad_s"]+1e-12)
            self.assertLessEqual(np.abs(velocity-previous_velocity).max(),
                                 self.cfg["morphing"]["max_accel_rad_s2"]*dt+1e-12)
            if index == 139:
                np.testing.assert_allclose(velocity, 0, atol=1e-12)
                self.assertEqual(planner.status, "PAUSED")
            actual, previous_velocity = q, velocity
        np.testing.assert_allclose(actual, goal, atol=1e-4)
        self.assertEqual(planner.status, "HOLD")
        # 跟踪偏差大时停止继续推进目标，而不是不断拉大舵机误差。
        self.assertTrue(planner.request(np.array(list(self.cfg["joints"].values()))))
        planner.step(dt, actual+0.1)
        self.assertEqual(planner.status, "PAUSED")

    def test_allocation_refresh_preserves_motor_history(self):
        allocator = Allocator(self.initial, self.cfg)
        previous = allocator.solve([self.initial.mass*9.81, 0.001, 0, 0], .004)
        cfg = load_config(configuration="swept")
        changed = self.kinematics.at(cfg["joints"])
        allocator.update_model(changed, self.cfg)
        np.testing.assert_equal(allocator.previous, previous)
        command = allocator.solve([changed.mass*9.81, 0, 0, 0], .004)
        self.assertLessEqual(np.abs(command-previous).max(), .004*self.cfg["allocation"]["thrust_rate_n_s"]+1e-10)
        np.testing.assert_allclose(allocator.B @ allocator.hover, [0, 0, changed.mass*9.81, 0, 0, 0], atol=1e-12)

    def test_coordinated_path_and_mid_motion_reversal(self):
        planner = MorphPlanner(self.cfg, self.kinematics)
        actual = planner.position.copy()
        dt = .02
        for name in ("swept", "asymmetric", "nominal"):
            goal = np.array([self.cfg["configurations"][name][j] for j in JOINTS])
            self.assertTrue(planner.request(goal), planner.reason)
            for _ in range(500):
                actual, _ = planner.step(dt, actual)
            np.testing.assert_allclose(actual, goal, atol=1e-4)
        goal = np.array([self.cfg["configurations"]["swept"][j] for j in JOINTS])
        self.assertTrue(planner.request(goal), planner.reason)
        for _ in range(50):
            actual, _ = planner.step(dt, actual)
        initial = np.array([self.cfg["joints"][j] for j in JOINTS])
        self.assertTrue(planner.request(initial), planner.reason)
        previous = planner.velocity.copy()
        for _ in range(400):
            actual, velocity = planner.step(dt, actual)
            self.assertLessEqual(np.abs(velocity-previous).max(), .08*dt+1e-12)
            previous = velocity
        np.testing.assert_allclose(actual, initial, atol=1e-4)

    def test_articulation_asset_has_physical_joints_and_correct_link_inertia(self):
        from pxr import Usd, UsdPhysics, UsdShade
        from scipy.spatial.transform import Rotation
        from aeromorph_control.articulated import build_articulation_asset

        stage = Usd.Stage.Open(str(build_articulation_asset(self.initial, self.cfg, self.kinematics)))
        bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        self.assertEqual(len(bodies), 9)
        self.assertEqual(sum(p.IsA(UsdPhysics.RevoluteJoint) for p in stage.Traverse()), 4)
        self.assertEqual(sum(p.IsA(UsdPhysics.FixedJoint) for p in stage.Traverse()), 4)
        self.assertEqual(sum(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse()), 1)
        for prim in bodies:
            mesh = stage.GetPrimAtPath(f"{prim.GetPath()}/visual_0/mesh")
            self.assertTrue(UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0])
            m, center, inertia = self.kinematics.properties[prim.GetName()]
            properties = UsdPhysics.MassAPI(prim)
            self.assertAlmostEqual(properties.GetMassAttr().Get(), m, places=7)
            np.testing.assert_allclose(properties.GetCenterOfMassAttr().Get(), center, atol=1e-8)
            quat = properties.GetPrincipalAxesAttr().Get()
            R = Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()
            reconstructed = R @ np.diag(properties.GetDiagonalInertiaAttr().Get()) @ R.T
            np.testing.assert_allclose(reconstructed, inertia, atol=1e-9)

    def test_grid_does_not_add_mass_or_collisions(self):
        from pxr import Usd, UsdGeom, UsdPhysics
        from aeromorph_control.scene import create_grid

        stage = Usd.Stage.CreateInMemory()
        create_grid(stage)
        self.assertTrue(any(p.IsA(UsdGeom.Mesh) for p in stage.Traverse()))
        for prim in stage.Traverse():
            self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertFalse(prim.HasAPI(UsdPhysics.MassAPI))
            self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI))


if __name__ == "__main__":
    unittest.main(verbosity=2)
