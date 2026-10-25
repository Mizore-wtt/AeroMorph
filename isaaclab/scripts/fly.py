"""GPU AeroMorph 飞行与四关节在线变构；Isaac Lab 3.0 / Isaac Sim 6.0.1。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation

from aeromorph_control.allocator import Allocator
from aeromorph_control.articulated import ArticulatedRotorForces, ArticulatedState, build_articulation_asset
from aeromorph_control.config import JOINTS, ROOT, load_config
from aeromorph_control.controller import FlightController
from aeromorph_control.keyboard import Keyboard
from aeromorph_control.model import inspect_model
from aeromorph_control.morphology import Kinematics, MorphPlanner
from aeromorph_control.rotors import RotorDynamics
from aeromorph_control.rotor_visuals import RotorVisuals
from aeromorph_control.scene import configure_view
from aeromorph_control.telemetry import Telemetry
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path)
parser.add_argument("--configuration")
parser.add_argument("--joints", nargs=4, type=float, metavar=("J1", "J2", "J3", "J4"))
parser.add_argument("--yaw", type=float, help="初始航向 [rad]；控制方向随此航向固定")
parser.add_argument("--duration", type=float, default=0, help="仿真秒数；0 为持续运行")
parser.add_argument(
    "--scenario", choices=("keyboard", "hover", "sequence", "morph", "morph_sequence", "motors_off"),
    default="keyboard",
)
parser.add_argument("--initial-offset", nargs=3, type=float, default=[0, 0, 0], help="测试用初始质心偏移 [m]")
parser.add_argument("--initial-tilt", nargs=2, type=float, default=[0, 0], help="测试用初始 roll/pitch [rad]")
parser.add_argument("--run-dir", type=Path)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cuda:0", visualizer=["kit"])
args = parser.parse_args()
if args.duration < 0 or not np.isfinite([args.duration, *args.initial_offset, *args.initial_tilt]).all():
    parser.error("运行时长和初始偏移必须为有限数；时长不能为负")
try:
    cfg = load_config(args.config, args.configuration, args.joints)
    if args.yaw is not None:
        if not np.isfinite(args.yaw):
            raise ValueError("--yaw 必须为有限数")
        cfg["simulation"]["yaw_rad"] = args.yaw
    model = inspect_model(cfg)
    allocator = Allocator(model, cfg)  # 无效构型在启动图形程序前就拒绝。
    kinematics = Kinematics(cfg, model)
    planner = MorphPlanner(cfg, kinematics)
except (ValueError, KeyError) as exc:
    parser.exit(2, f"[AeroMorph] 构型检查失败: {exc}\n")
run_dir = args.run_dir or ROOT / "log/flight" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
run_dir.mkdir(parents=True, exist_ok=True)
print(f"[AeroMorph] 构型={cfg['configuration']} 质量={model.mass:.6f} kg "
      f"cond={allocator.condition:.2f} 悬停推力={allocator.hover.round(4)} N", flush=True)
print(f"[AeroMorph] 电机参数: {cfg['parameter_source']}；日志: {run_dir}", flush=True)
launcher = AppLauncher(args, width=1280, height=720, window_width=1280, window_height=800)
simulation_app = launcher.app

# Kit 启动后再导入仿真模块，避免插件尚未注册。
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg


def scenario_command(t):
    if args.scenario not in ("sequence", "morph_sequence"):
        return np.zeros(3)
    # 与键盘共用同一指令生成器，包含组合输入和充分的松键制动时间。
    if 3 <= t < 6:
        return np.array([1, 0, 1])
    if 10 <= t < 13:
        return np.array([0, 1, 0])
    if 17 <= t < 20:
        return np.array([-1, -1, -1])
    return np.zeros(3)


def scenario_joints(t):
    if args.scenario not in ("morph", "morph_sequence"):
        return None
    if t < 3:
        return np.array([cfg["joints"][j] for j in JOINTS])
    if t < 14:
        return np.array([cfg["configurations"]["swept"][j] for j in JOINTS])
    if t < 25:
        return np.array([cfg["configurations"]["asymmetric"][j] for j in JOINTS])
    if t < 26:
        return np.zeros(4)  # 共线退化请求必须拒绝，飞行继续。
    return np.array([cfg["joints"][j] for j in JOINTS])


def main():
    # 显式建立 Kit 场景，并让模型生成、视口和 PhysX 共用同一个 stage。
    import omni.usd

    context = omni.usd.get_context()
    context.new_stage()
    with sim_utils.use_stage(context.get_stage()):
        run_flight()


def run_flight():
    s = cfg["simulation"]
    dt = s["dt_s"]
    gui = not args.headless and "kit" in (args.visualizer or [])
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt, render_interval=s["render_decimation"], device=args.device,
        gravity=(0, 0, -9.81), use_fabric=True, log_dir=str(run_dir),
        # 这里只模拟单机，按小场景预分配 GPU 缓冲，避免训练规模的默认容量挤占显存。
        physics=PhysxCfg(
            enable_external_forces_every_iteration=True,
            gpu_max_rigid_contact_count=2**15, gpu_max_rigid_patch_count=2**13,
            gpu_found_lost_pairs_capacity=2**13, gpu_found_lost_aggregate_pairs_capacity=2**13,
            gpu_total_aggregate_pairs_capacity=2**13, gpu_collision_stack_size=2**24,
            gpu_heap_capacity=2**24, gpu_temp_buffer_capacity=2**22,
            gpu_max_soft_body_contacts=2**10, gpu_max_particle_contacts=2**10,
        ),
    ))
    ground = sim_utils.CuboidCfg(
        size=(20, 20, 0.1), visible=False,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )
    ground.func("/World/Ground", ground, translation=(0, 0, -.05))
    light = sim_utils.DomeLightCfg(intensity=500)
    light.func("/World/Light", light)
    target_position = np.array([0, 0, s["height_m"]])
    initial_rotation = Rotation.from_euler("xyz", [*args.initial_tilt, s["yaw_rad"]])
    initial_root = target_position+args.initial_offset-initial_rotation.apply(model.com)
    asset_path = build_articulation_asset(model, cfg, kinematics)
    morph = cfg["morphing"]
    robot = Articulation(ArticulationCfg(
        prim_path="/World/AeroMorph",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, kinematic_enabled=False, linear_damping=0, angular_damping=0,
                enable_gyroscopic_forces=True, max_linear_velocity=20, max_angular_velocity=20,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(initial_root), rot=tuple(initial_rotation.as_quat()), joint_pos=cfg["joints"],
        ),
        actuators={"morph": ImplicitActuatorCfg(
            joint_names_expr=list(cfg["joints"]), stiffness=morph["servo_stiffness_n_m_rad"],
            damping=morph["servo_damping_n_m_s_rad"], effort_limit_sim=morph["servo_effort_limit_n_m"],
            velocity_limit_sim=.5,
        )},
    ))
    sim.reset()
    configure_view(sim, target_position, s["yaw_rad"], gui)
    vehicle = ArticulatedState(robot, kinematics, cfg)
    # 仅在起飞初始化时设置一次；控制循环绝不写入位姿或根速度。
    pose = torch.tensor(np.r_[initial_root, initial_rotation.as_quat()][None],
                        dtype=torch.float32, device=sim.device)
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=sim.device))
    q0 = torch.tensor([[cfg["joints"][j] for j in robot.joint_names]], dtype=torch.float32, device=sim.device)
    robot.write_joint_position_to_sim_index(position=q0)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
    robot.reset()
    robot.set_joint_position_target_index(target=q0)
    sim.forward()
    robot.update(dt)
    controller = FlightController(model, cfg, target_position, s["yaw_rad"])
    motors = RotorDynamics(cfg, allocator.hover)
    forces = ArticulatedRotorForces(robot, motors, vehicle)
    keyboard = Keyboard()
    telemetry = Telemetry(run_dir, cfg, model, allocator)
    (run_dir / "asset.json").write_text(json.dumps({
        "usd_path": str(asset_path), "source_hash": model.mesh_hash,
        "device": sim.device, "model_type": "articulation", "physics_mass_kg": vehicle.mass,
        "physics_link_inertias_kg_m2": vehicle.inertias.tolist(),
        "scenario": args.scenario, "initial_offset_m": args.initial_offset,
        "initial_tilt_rad": args.initial_tilt,
    }, indent=2)+"\n")
    if gui:
        keyboard.attach(cfg["configuration"], cfg["joints"], kinematics.limits)
    rotor_visuals = RotorVisuals(sim.stage, cfg, vehicle.rotor_axes_local) if gui else None
    print(f"[AeroMorph] READY: device={sim.device}；四关节物理变构；W/S A/D R/F，H 制动悬停。", flush=True)
    count, bad_allocation_time = 0, 0.0
    thrust, command = allocator.hover.copy(), np.zeros(3)
    error = None
    try:
        while simulation_app.is_running() and (args.duration == 0 or count*dt < args.duration):
            wall_start = time.perf_counter()
            if not sim.is_playing():
                keyboard.keys.clear()
                sim.render()
                continue
            t = count*dt
            p, v, R, omega = vehicle.read(dt)
            tilt = np.arccos(np.clip(R[2, 2], -1, 1))
            limits = cfg["limits"]
            if not limits["minimum_height_m"] < p[2] < limits["maximum_height_m"]:
                raise RuntimeError(f"超出高度限制，停止仿真: z={p[2]:.3f} m")
            if (np.linalg.norm(p[:2]) > limits["maximum_distance_m"]
                    or tilt > limits["maximum_tilt_rad"]
                    or np.linalg.norm(omega) > limits["maximum_rate_rad_s"]):
                raise RuntimeError("超出位置/姿态/角速度限制，停止仿真；请检查构型或控制参数")
            if count % s["position_decimation"] == 0:
                command = keyboard.command() if args.scenario == "keyboard" else scenario_command(t)
                controller.position_update(p, v, command, dt*s["position_decimation"])
                requested = keyboard.joint_command() if args.scenario == "keyboard" else scenario_joints(t)
                if requested is not None:
                    old_rejections = planner.rejections
                    planner.request(requested)
                    if planner.rejections != old_rejections:
                        print(f"[AeroMorph] 变构请求拒绝，保持可行构型: {planner.reason}", flush=True)
            if count % s["attitude_decimation"] == 0:
                controller.model = vehicle.model
                allocator.update_model(vehicle.model, cfg)
                allowed = (tilt < morph["pause_tilt_rad"] and np.linalg.norm(omega) < morph["pause_rate_rad_s"]
                           and allocator.normalized_residual < .01)
                q_ref, dq_ref = planner.step(dt*s["attitude_decimation"], vehicle.joints, allowed)
                robot.set_joint_position_target_index(
                    target=torch.tensor(q_ref[None], dtype=torch.float32, device=sim.device),
                    joint_ids=vehicle.joint_ids,
                )
                robot.set_joint_velocity_target_index(
                    target=torch.tensor(dq_ref[None], dtype=torch.float32, device=sim.device),
                    joint_ids=vehicle.joint_ids,
                )
                desired = controller.attitude_update(R, omega)
                desired[1:] += vehicle.compensation
                thrust = allocator.solve(desired, dt*s["attitude_decimation"])
            if args.scenario == "motors_off":
                thrust = np.zeros(4)
            motors.step(thrust, dt)
            forces.apply()
            if count == 0:
                composer = robot.permanent_wrench_composer
                f_b = composer.out_force_b.torch[0].cpu().numpy()
                tau_b = composer.out_torque_b.torch[0].cpu().numpy()
                applied = np.zeros(6)
                for rotor, body in enumerate(vehicle.rotor_ids):
                    name = robot.body_names[body]
                    rotation = vehicle.model.transforms[name][:3, :3]
                    force_b = rotation @ f_b[body]
                    torque_b = rotation @ tau_b[body]
                    center = vehicle.model.hubs[rotor] + rotation @ vehicle.centers_local[body]
                    applied += np.r_[force_b, torque_b+np.cross(center, force_b)]
                if not np.allclose(applied, allocator.B @ motors.actual, atol=2e-6):
                    raise RuntimeError(f"实际旋翼合力/力矩与 B(q) 不符: {applied}")
            bad_allocation_time = bad_allocation_time+dt if allocator.normalized_residual > .02 else 0
            if bad_allocation_time > limits["saturation_timeout_s"]:
                raise RuntimeError("控制分配持续不可实现；已停止仿真，请降低指令或更换构型")
            if count % s["log_decimation"] == 0:
                rpy = Rotation.from_matrix(R).as_euler("xyz")
                telemetry.record(t, p, v, rpy, omega, command, controller, allocator, thrust, motors.actual, R,
                                 vehicle, planner)
                keyboard.refresh(p, v, rpy, omega, motors.actual, controller.mode, allocator.saturated)
                keyboard.refresh_morph(planner, vehicle.joints)
                if count % max(1, round(1/dt)) == 0:
                    print(f"[flight {t:5.1f}s] p={p.round(3)} v={v.round(3)} "
                          f"rpy={np.degrees(rpy).round(1)} f={motors.actual.round(2)} "
                          f"mode={'/'.join(controller.mode)} sat={allocator.saturated} "
                          f"q={vehicle.joints.round(3)} morph={planner.status}", flush=True)
            sim.step(render=False)
            robot.update(dt)
            count += 1
            if gui and count % s["render_decimation"] == 0:
                # 仅更新外观子节点；自转跟随桨毂，不向 PhysX 写入旋翼关节状态。
                rotor_visuals.advance(dt*s["render_decimation"], motors.actual)
                keyboard.refresh_rotors(rotor_visuals.levels)
                sim.render()
            # 可视化时尽量按真实时间运行，保持键盘的速度含义清晰。
            if gui:
                time.sleep(max(0, dt-(time.perf_counter()-wall_start)))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        forces.close()
        keyboard.close()
        result = telemetry.close(error)
        print("[AeroMorph] SUMMARY " + json.dumps(result, ensure_ascii=False), flush=True)
        sim.pause()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        pass
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close(exit_code=exit_code)
