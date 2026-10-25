"""真实四关节飞行模型、整机 COM 反馈及变构角动量补偿。"""

import hashlib

import numpy as np
from scipy.spatial.transform import Rotation

from .config import JOINTS, ROOT
from .model import build_visual_asset, origin


def build_articulation_asset(model, cfg, kinematics):
    """保留九个刚体、四个变构自由度；旋翼自转用连续推力模型等效。"""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    visual_asset = build_visual_asset(model, cfg)
    key = hashlib.sha256((str(visual_asset) + "articulated-v2").encode()).hexdigest()[:16]
    folder = ROOT / "build/isaaclab/flight" / key
    path = folder / "articulated.usdc"
    if path.exists():
        return path
    folder.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, "/AeroMorph")
    stage.SetDefaultPrim(root.GetPrim())
    # 在根连杆上指定自由基座 articulation，不生成连到世界的固定关节。
    for name, visuals in model.links:
        link = UsdGeom.Xform.Define(stage, f"/AeroMorph/{name}")
        T = model.transforms[name]
        link.AddTranslateOp().Set(Gf.Vec3d(*T[:3, 3].tolist()))
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        link.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3].tolist())))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        if name == kinematics.root:
            UsdPhysics.ArticulationRootAPI.Apply(link.GetPrim())
        m, c, I = kinematics.properties[name]
        mass = UsdPhysics.MassAPI.Apply(link.GetPrim())
        mass.CreateMassAttr(m)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*c.tolist()))
        values, E = np.linalg.eigh(I)
        # 特征向量组成主惯量坐标系，修正为右手系后再转为 USD 的实部/虚部四元数。
        if np.linalg.det(E) < 0:
            E[:, -1] *= -1
        quat = Rotation.from_matrix(E).as_quat()
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*values.tolist()))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(float(quat[3]), Gf.Vec3f(*quat[:3].tolist())))
        for i in range(len(visuals)):
            # 共享几何子树；每个连杆的刚体与惯量仅在上文定义一次。
            part = UsdGeom.Xform.Define(stage, f"{link.GetPath()}/visual_{i}")
            part.GetPrim().GetReferences().AddReference(
                str(visual_asset), Sdf.Path(f"/AeroMorph/links/{name}/visual_{i}"))
            material_path = f"/AeroMorph/materials/{name}_{i}"
            material = UsdShade.Material.Define(stage, material_path)
            material.GetPrim().GetReferences().AddReference(str(visual_asset), Sdf.Path(material_path))
            # 子树引用不包含外部材质目标，需要在当前资产内显式重新绑定。
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{part.GetPath()}/mesh")).Bind(material)
    for element in kinematics.joints:
        name = element.get("name")
        parent, child = element.find("parent").get("link"), element.find("child").get("link")
        T = origin(element.find("origin"))
        if name in JOINTS:
            joint = UsdPhysics.RevoluteJoint.Define(stage, f"/AeroMorph/joints/{name}")
            axis = np.fromstring(element.find("axis").get("xyz"), sep=" ")
            # 当前模型为 +Z 转轴；换模型时明确拒绝，不隐式改错关节坐标系。
            if not np.allclose(axis, [0, 0, 1]):
                raise ValueError(f"关节 {name} 尚不支持非 +Z 转轴")
            joint.CreateAxisAttr("Z")
            low, high = kinematics.limits[name]
            joint.CreateLowerLimitAttr(float(np.degrees(low)))
            joint.CreateUpperLimitAttr(float(np.degrees(high)))
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            # 实际 SI 增益由 Isaac Lab actuator 在初始化时写入 PhysX。
            drive.CreateStiffnessAttr(0.)
            drive.CreateDampingAttr(0.)
            drive.CreateTargetPositionAttr(float(np.degrees(cfg["joints"][name])))
        else:
            joint = UsdPhysics.FixedJoint.Define(stage, f"/AeroMorph/joints/{name}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/AeroMorph/{parent}")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/AeroMorph/{child}")])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*T[:3, 3].tolist()))
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        joint.CreateLocalRot0Attr(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3].tolist())))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateCollisionEnabledAttr(False)
    stage.GetRootLayer().Save()
    return path


class ArticulatedState:
    """从所有连杆的实际状态计算整机反馈，不把根连杆 COM 当成整机 COM。"""

    def __init__(self, robot, kinematics, cfg):
        self.robot, self.kinematics, self.cfg = robot, kinematics, cfg
        self.joint_ids, _ = robot.find_joints(list(JOINTS), preserve_order=True)
        self.rotor_ids, _ = robot.find_bodies([r["link"] for r in cfg["rotors"]], preserve_order=True)
        self.masses = robot.data.body_mass.torch[0].detach().cpu().numpy().astype(float)
        self.inertias = robot.data.body_inertia.torch[0].detach().cpu().numpy().reshape(-1, 3, 3).astype(float)
        self.centers_local = np.array([kinematics.properties[n][1] for n in robot.body_names])
        for i, name in enumerate(robot.body_names):
            m, _, I = kinematics.properties[name]
            if not np.isclose(self.masses[i], m, rtol=1e-5) or not np.allclose(self.inertias[i], I, atol=1e-7):
                raise RuntimeError(f"PhysX 连杆 {name} 的质量/惯量与 URDF 不符")
        if set(robot.body_names) != set(kinematics.properties) or set(robot.joint_names) != set(JOINTS):
            raise RuntimeError(f"变构模型自由度不符: bodies={robot.body_names}, joints={robot.joint_names}")
        self.mass = float(self.masses.sum())
        self.previous_J = self.previous_shape_momentum = None
        self.compensation = np.zeros(3)
        self.joints = np.array([cfg["joints"][j] for j in JOINTS])
        self.joint_velocity = np.zeros(4)
        self.model = kinematics.at(self.joints)
        self.rotor_axes_local = np.asarray(kinematics.rotor_axes)

    def read(self, dt):
        """读取真实多刚体状态，聚合质心/惯量/角动量；所有控制力矩表达在根连杆轴系。"""
        import torch

        data = self.robot.data
        # 一次打包读回，避免在每个旋翼/连杆上分别同步 GPU。
        packed = torch.cat((data.body_link_pos_w.torch[0], data.body_link_quat_w.torch[0],
                            data.body_com_pos_w.torch[0], data.body_com_vel_w.torch[0]), dim=1)
        links = packed.detach().cpu().numpy().astype(float)
        joints = torch.stack((data.joint_pos.torch[0], data.joint_vel.torch[0])).detach().cpu().numpy()
        self.joints, self.joint_velocity = joints[:, self.joint_ids].astype(float)
        if not np.isfinite(links).all() or not np.isfinite(joints).all():
            raise RuntimeError("物理状态出现非有限值")
        rotations = Rotation.from_quat(links[:, 3:7]).as_matrix()
        root_id = self.robot.body_names.index(self.kinematics.root)
        R = rotations[root_id]
        coms, velocities, rates = links[:, 7:10], links[:, 10:13], links[:, 13:16]
        p = np.sum(self.masses[:, None]*coms, axis=0)/self.mass
        v = np.sum(self.masses[:, None]*velocities, axis=0)/self.mass
        omega = R.T @ rates[root_id]
        relative = (coms-p) @ R
        relative_velocity = (velocities-v) @ R
        local_rotations = R.T[None] @ rotations
        J = np.zeros((3, 3))
        H = np.zeros(3)
        for i, (m, r) in enumerate(zip(self.masses, relative)):
            I = local_rotations[i] @ self.inertias[i] @ local_rotations[i].T
            J += I + m*((r @ r)*np.eye(3)-np.outer(r, r))
            H += I @ (R.T @ rates[i]) + np.cross(r, m*relative_velocity[i])
        # 扣除整机随根连杆一起转动的部分，余下的是内部变构相对运动的角动量。
        shape_momentum = H-J @ omega
        if self.previous_J is not None:
            # H = J(q)ω + h(q,qdot)，外部力矩还需 Jdot·ω + hdot + ω×h。
            correction = ((J-self.previous_J)/dt @ omega
                          + (shape_momentum-self.previous_shape_momentum)/dt
                          + np.cross(omega, shape_momentum))
            blend = 1-np.exp(-dt/self.cfg["morphing"]["momentum_filter_s"])
            self.compensation += blend*(correction-self.compensation)
        self.previous_J, self.previous_shape_momentum = J.copy(), shape_momentum.copy()
        # 分配器使用实际桨毂位置和朝向，包含关节跟踪偏差。
        self.model = self.kinematics.at(self.joints)
        self.model.inertia = J
        self.model.com = R.T @ (p-links[root_id, :3])
        self.model.hubs = (links[self.rotor_ids, :3]-p) @ R
        self.model.axes = np.array([R.T @ rotations[i] @ axis
                                    for i, axis in zip(self.rotor_ids, self.rotor_axes_local)])
        return p, v, R, omega


class ArticulatedRotorForces:
    """推力施加到真实旋翼刚体，内力/舵机反作用由 PhysX 求解。"""

    def __init__(self, robot, dynamics, vehicle):
        import torch

        self.robot, self.dynamics, self.vehicle = robot, dynamics, vehicle
        self.body_ids = torch.tensor(vehicle.rotor_ids, dtype=torch.int32, device=robot.device)
        self.axes = torch.tensor(vehicle.rotor_axes_local[None], dtype=torch.float32, device=robot.device)
        # composer 的局部作用点相对各刚体 COM；桨毂在旋翼 link 原点。
        self.positions = torch.tensor(-vehicle.centers_local[vehicle.rotor_ids][None],
                                      dtype=torch.float32, device=robot.device)
        self.reaction = torch.tensor(dynamics.reaction[None, :, None], dtype=torch.float32, device=robot.device)

    def apply(self):
        import torch

        force = self.axes * torch.as_tensor(self.dynamics.actual[None, :, None], dtype=torch.float32,
                                            device=self.robot.device)
        composer = self.robot.permanent_wrench_composer
        composer.reset()
        composer.add_forces_and_torques_index(forces=force, torques=force*self.reaction,
                                              positions=self.positions, body_ids=self.body_ids, is_global=False)
        self.robot.write_data_to_sim()

    def close(self):
        self.robot.permanent_wrench_composer.reset()
