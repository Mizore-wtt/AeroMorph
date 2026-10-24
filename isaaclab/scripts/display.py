"""Inspect AeroMorph in Isaac Lab 3.0 beta2 / Isaac Sim 6.0.1.

This is a pose-driven visualization, with gravity and collisions disabled.
It is deliberately separate from any future flight-control or RL task.
"""

import argparse
import json
import math
from pathlib import Path
import time
import traceback

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/aeromorph_description"
FOLD_JOINTS = ("joint1", "joint2", "joint4", "joint5")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", choices=("manual", "demo"), default="manual")
parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps; 0 runs until closed.")
parser.add_argument("--speed", type=float, default=0.35, help="Keyboard translation speed in m/s.")
parser.add_argument("--rebuild", action="store_true", help="Reconvert USD, including changed STL files.")
parser.add_argument("--run-dir", type=Path, default=ROOT / "log/isaaclab/latest")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu", visualizer=["kit"])
args = parser.parse_args()
if args.steps < 0 or args.speed <= 0:
    parser.error("--steps must be nonnegative and --speed must be positive")
args.run_dir.mkdir(parents=True, exist_ok=True)

launcher = AppLauncher(args, width=1280, height=720, window_width=1280, window_height=800)
simulation_app = launcher.app

# Simulator modules must be imported after AppLauncher starts Kit.
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg


class Controls:
    """World-frame keyboard motion and four independent joint sliders."""

    def __init__(self):
        self.keys = set()
        self.pose = [0.0, 0.0, 1.0, 0.0]  # x/y/z [m], yaw [rad]
        self.angles = [0.0] * 4
        self.demo = args.motion == "demo"
        self.phase = 0.0
        self.window = None
        self.keyboard_subscription = None
        self.syncing_sliders = False

    def reset(self):
        self.pose[:] = [0.0, 0.0, 1.0, 0.0]
        self.angles[:] = [0.0] * 4
        self.phase = 0.0
        self.keys.clear()
        self.demo = False

    def attach_gui(self):
        import carb.input
        import omni.appwindow
        import omni.ui as ui

        self.input_interface = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self.keyboard_subscription = self.input_interface.subscribe_to_keyboard_events(
            self.keyboard, self._on_key
        )
        self.window = ui.Window("AeroMorph controls", width=350, height=350)
        self.window.position_x = 910
        self.window.position_y = 80
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Pose-driven display (no flight physics)")
                ui.Label("Click viewport before using keyboard.")
                ui.Label("W/S: world X    A/D: world Y")
                ui.Label("R/F: up/down    Q/E: yaw")
                ui.Label("T: toggle demo    HOME: reset")
                with ui.HStack(height=25):
                    ui.Button("Toggle demo", clicked_fn=self.toggle_demo)
                    ui.Button("Reset", clicked_fn=self.reset)
                self.slider_models = []
                for index, name in enumerate(FOLD_JOINTS):
                    with ui.HStack(height=25):
                        ui.Label(f"{name} [rad]", width=95)
                        slider = ui.FloatSlider(min=-0.6, max=0.6, step=0.01)
                        slider.model.add_value_changed_fn(
                            lambda model, i=index: self._set_angle(i, model.as_float)
                        )
                        self.slider_models.append(slider.model)
                self.status = ui.Label("", height=40, word_wrap=True)

    def toggle_demo(self):
        self.demo = not self.demo
        self.phase = 0.0

    def _set_angle(self, index, value):
        if self.syncing_sliders:
            return
        self.angles[index] = value
        self.demo = False

    def _on_key(self, event, *_):
        import carb.input

        key = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.keys.add(key)
            if key == "T":
                self.toggle_demo()
            elif key == "HOME":
                self.reset()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.keys.discard(key)
        return True

    def advance(self, dt):
        if self.demo:
            self.phase += dt
            t = self.phase
            self.pose[:] = [0.45 * math.sin(0.5 * t), 0.25 * math.sin(t),
                            1.0 + 0.12 * math.sin(0.7 * t), 0.25 * math.sin(0.5 * t)]
            angle = 0.25 * math.sin(0.7 * t)
            self.angles[:] = [angle, -angle, angle, -angle]
        else:
            axis = lambda positive, negative: int(positive in self.keys) - int(negative in self.keys)
            self.pose[0] += axis("W", "S") * args.speed * dt
            self.pose[1] += axis("A", "D") * args.speed * dt
            self.pose[2] += axis("R", "F") * args.speed * dt
            self.pose[3] += axis("Q", "E") * 0.6 * dt
            self.pose[0] = max(-2.0, min(2.0, self.pose[0]))
            self.pose[1] = max(-2.0, min(2.0, self.pose[1]))
            self.pose[2] = max(0.25, min(2.5, self.pose[2]))

    def refresh_gui(self):
        if self.window is not None:
            self.syncing_sliders = True
            try:
                for model, angle in zip(self.slider_models, self.angles):
                    if abs(model.as_float - angle) > 0.0001:
                        model.set_value(angle)
            finally:
                self.syncing_sliders = False
            self.status.text = (
                f"{'DEMO' if self.demo else 'MANUAL'}  "
                f"XYZ: {self.pose[0]:.2f}, {self.pose[1]:.2f}, {self.pose[2]:.2f} m  "
                f"Yaw: {math.degrees(self.pose[3]):.0f} deg\n"
                f"Joints: {', '.join(f'{v:.2f}' for v in self.angles)} rad"
            )

    def close(self):
        if self.keyboard_subscription is not None:
            self.input_interface.unsubscribe_to_keyboard_events(self.keyboard, self.keyboard_subscription)
        if self.window is not None:
            self.window.destroy()


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=1.0 / 60.0, device=args.device, gravity=(0.0, 0.0, 0.0), use_fabric=False,
        log_dir=str(args.run_dir),
    ))
    # Use local geometry: the display does not need online environment assets.
    floor = sim_utils.CuboidCfg(
        size=(6.0, 6.0, 0.02),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.22, 0.28)),
    )
    floor.func("/World/Floor", floor, translation=(0.0, 0.0, -0.02))
    light = sim_utils.DomeLightCfg(intensity=500.0, color=(0.85, 0.9, 1.0))
    light.func("/World/Light", light)

    robot = Articulation(ArticulationCfg(
        prim_path="/World/AeroMorph",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(PACKAGE / "urdf/AeroMorph.urdf"),
            usd_dir=str(ROOT / "build/isaaclab/assets"),
            fix_base=False, merge_fixed_joints=False, self_collision=False,
            ros_package_paths=[{"name": "aeromorph_description", "path": str(PACKAGE)}],
            robot_type="Aerial", run_asset_transformer=False, run_multi_physics_conversion=False,
            force_usd_conversion=args.rebuild,
            joint_drive=UrdfConverterCfg.JointDriveCfg(target_type="none"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        actuators={"display": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=0.0, damping=0.0,
            effort_limit_sim=1.0, velocity_limit_sim=100.0,
        )},
    ))
    sim.reset()
    sim.set_camera_view(eye=[1.4, 1.6, 1.8], target=[-0.12, 0.0, 1.0])
    assert robot.num_joints == 8, f"Expected 8 joints, got {robot.joint_names}"
    fold_ids, _ = robot.find_joints(list(FOLD_JOINTS), preserve_order=True)
    assert len(fold_ids) == 4, robot.joint_names
    root_pose = robot.data.default_root_pose.torch.clone()
    root_velocity = torch.zeros_like(robot.data.default_root_vel.torch)
    joint_position = robot.data.default_joint_pos.torch.clone()
    joint_velocity = torch.zeros_like(robot.data.default_joint_vel.torch)
    controls = Controls()
    # In this beta, Kit can be an active visualizer even when has_gui is False.
    gui_enabled = not args.headless and "kit" in (args.visualizer or [])
    if gui_enabled:
        controls.attach_gui()
    print(f"[AeroMorph] READY: {robot.num_bodies} bodies, {robot.num_joints} joints", flush=True)
    print("[AeroMorph] Pose-driven visualization: gravity/collisions disabled; no flight controller.", flush=True)
    count = 0
    max_pose_error = 0.0
    max_joint_error = 0.0
    first_position = None
    last_position = None
    try:
        while simulation_app.is_running() and (args.steps == 0 or count < args.steps):
            start = time.perf_counter()
            if not sim.is_playing():
                sim.render()
                continue
            controls.advance(sim.get_physics_dt())
            x, y, z, yaw = controls.pose
            root_pose[0] = torch.tensor(
                [x, y, z, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], device=sim.device
            )
            joint_position[0, fold_ids] = torch.tensor(controls.angles, device=sim.device)
            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            robot.write_root_velocity_to_sim_index(root_velocity=root_velocity)
            robot.write_joint_position_to_sim_index(position=joint_position)
            robot.write_joint_velocity_to_sim_index(velocity=joint_velocity)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            actual = robot.data.root_pos_w.torch
            actual_joints = robot.data.joint_pos.torch
            if not torch.isfinite(actual).all() or not torch.isfinite(actual_joints).all():
                raise RuntimeError("Non-finite model state")
            max_pose_error = max(max_pose_error, (actual - root_pose[:, :3]).abs().max().item())
            max_joint_error = max(max_joint_error, (actual_joints - joint_position).abs().max().item())
            last_position = actual[0].tolist()
            if first_position is None:
                first_position = last_position
            controls.refresh_gui()
            count += 1
            # Keep manual controls and demo timing usable on fast machines.
            if gui_enabled:
                time.sleep(max(0.0, sim.get_physics_dt() - (time.perf_counter() - start)))
    finally:
        controls.close()
        summary = {
            "mode": "pose-driven visualization", "steps": count,
            "bodies": robot.num_bodies, "joints": robot.joint_names,
            "first_position_m": first_position, "last_position_m": last_position,
            "max_position_error_m": max_pose_error, "max_joint_error_rad": max_joint_error,
        }
        (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[AeroMorph] Summary: {args.run_dir / 'summary.json'}", flush=True)
    if max_pose_error > 0.001 or max_joint_error > 0.001:
        raise RuntimeError(f"Display state mismatch: position={max_pose_error}, joints={max_joint_error}")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close(exit_code=exit_code)
