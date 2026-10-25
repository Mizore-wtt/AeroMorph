"""URDF 质量特性检查及共享几何快照；飞行关节模型由 articulated.py 生成。"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .config import ROOT, URDF


def origin(element):
    T = np.eye(4)
    if element is not None:
        T[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        T[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return T


def mesh_path(uri):
    prefix = "package://aeromorph_description/"
    if not uri.startswith(prefix):
        raise ValueError(f"不支持的模型路径: {uri}")
    return ROOT / "src/aeromorph_description" / uri[len(prefix):]


def stl_data(path):
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + count * 50 or count == 0:
        raise ValueError(f"需要有效二进制 STL: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    data = np.frombuffer(raw, dtype=dtype, offset=84, count=count)
    if not np.isfinite(data["vertices"]).all():
        raise ValueError(f"STL 存在非有限顶点: {path}")
    return data["vertices"].reshape(-1, 3).copy(), data["normal"].copy()


@dataclass
class Snapshot:
    mass: float
    com: np.ndarray
    inertia: np.ndarray
    transforms: dict
    links: list
    hubs: np.ndarray
    axes: np.ndarray
    radii: np.ndarray
    mesh_hash: str


def inspect_model(cfg):
    tree = ET.parse(URDF).getroot()
    link_elements = {e.get("name"): e for e in tree.findall("link")}
    joints = tree.findall("joint")
    parent_joint = {j.find("child").get("link"): j for j in joints}
    children = {j.find("child").get("link") for j in joints}
    roots = set(link_elements) - children
    if len(roots) != 1:
        raise ValueError("URDF 必须为单根树")
    transforms = {roots.pop(): np.eye(4)}
    pending = list(joints)
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            kind, name = joint.get("type"), joint.get("name")
            angle = cfg["joints"].get(name, 0.0)
            if kind == "revolute":
                limit = joint.find("limit")
                if not float(limit.get("lower")) <= angle <= float(limit.get("upper")):
                    raise ValueError(f"关节 {name}={angle} 超出 URDF 限位")
            if kind not in ("fixed", "revolute", "continuous"):
                raise ValueError(f"暂不支持关节类型: {kind}")
            motion = np.eye(4)
            if kind != "fixed":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                motion[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
            child = joint.find("child").get("link")
            transforms[child] = transforms[parent] @ origin(joint.find("origin")) @ motion
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("URDF 存在闭环或未连接关节")
    links, masses, centers, inertias = [], [], [], []
    digest = hashlib.sha256(URDF.read_bytes())
    for name, link in link_elements.items():
        inertial = link.find("inertial")
        m = float(inertial.find("mass").get("value"))
        a = inertial.find("inertia").attrib
        I = np.array([[float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
                      [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
                      [float(a["ixz"]), float(a["iyz"]), float(a["izz"])]])
        eig = np.linalg.eigvalsh(I)
        if m <= 0 or not np.isfinite(m) or not np.isfinite(I).all() or eig[0] <= 0 or eig[-1] > eig[:2].sum() + 1e-10:
            raise ValueError(f"{name} 质量/惯量不满足物理约束")
        TI = transforms[name] @ origin(inertial.find("origin"))
        masses.append(m)
        centers.append(TI[:3, 3])
        inertias.append(TI[:3, :3] @ I @ TI[:3, :3].T)
        visuals = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            path = mesh_path(mesh.get("filename"))
            points, normals = stl_data(path)
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            points *= scale
            color = visual.find("material/color")
            rgb = np.fromstring(color.get("rgba"), sep=" ")[:3] if color is not None else np.ones(3) * .8
            visuals.append((origin(visual.find("origin")), points, normals, rgb))
            digest.update(path.read_bytes())
        links.append((name, visuals))
    masses, centers = np.array(masses), np.array(centers)
    mass = float(masses.sum())
    com = (masses[:, None] * centers).sum(axis=0) / mass
    J = np.zeros((3, 3))
    for m, center, I in zip(masses, centers, inertias):
        d = center - com
        J += I + m * ((d @ d) * np.eye(3) - np.outer(d, d))
    hubs, axes, radii = [], [], []
    for rotor in cfg["rotors"]:
        name = rotor["link"]
        if name not in transforms:
            raise ValueError(f"未找到旋翼连杆: {name}")
        T = transforms[name]
        joint = parent_joint.get(name)
        if joint is None or joint.get("type") not in ("continuous", "revolute"):
            raise ValueError(f"{name} 需要 URDF 中的旋翼转轴关节")
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
            raise ValueError(f"{name} 的 URDF 旋翼转轴无效")
        axis = axis / np.linalg.norm(axis) * rotor["thrust_sign"]
        hubs.append(T[:3, 3] - com)
        axes.append(T[:3, :3] @ axis)
        V, points, _, _ = next(v for n, v in links if n == name)[0]
        points = points @ V[:3, :3].T + V[:3, 3]
        radii.append(float(np.linalg.norm(points - (points @ axis)[:, None] * axis, axis=1).max()))
    return Snapshot(mass, com, J, transforms, links, np.array(hubs), np.array(axes), np.array(radii), digest.hexdigest())


def build_visual_asset(snapshot, cfg):
    """缓存 STL 网格、材质和包围盒碰撞几何，不在此定义飞行刚体。"""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    source = (snapshot.mesh_hash + json.dumps(cfg["joints"], sort_keys=True)
              + json.dumps(cfg["rotors"], sort_keys=True) + "flight-visual-v1")
    key = hashlib.sha256(source.encode()).hexdigest()[:16]
    folder = ROOT / "build/isaaclab/flight" / key
    path = folder / "visuals.usdc"
    if path.exists():
        return path
    folder.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/AeroMorph")
    stage.SetDefaultPrim(root.GetPrim())
    for name, visuals in snapshot.links:
        T = snapshot.transforms[name].copy()
        T[:3, 3] -= snapshot.com
        link = UsdGeom.Xform.Define(stage, f"/AeroMorph/links/{name}")
        link.AddTransformOp().Set(Gf.Matrix4d(*T.T.reshape(-1).tolist()))
        for i, (V, points, normals, rgb) in enumerate(visuals):
            part = UsdGeom.Xform.Define(stage, f"{link.GetPath()}/visual_{i}")
            part.AddTransformOp().Set(Gf.Matrix4d(*V.T.reshape(-1).tolist()))
            mesh = UsdGeom.Mesh.Define(stage, f"{part.GetPath()}/mesh")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.CreateExtentAttr(UsdGeom.Mesh.ComputeExtent(mesh.GetPointsAttr().Get()))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(points)//3, 3, dtype=np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(len(points), dtype=np.int32)))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            material = UsdShade.Material.Define(stage, f"/AeroMorph/materials/{name}_{i}")
            shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb.tolist()))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
            if not name.startswith("fly"):
                low, high = points.min(axis=0), points.max(axis=0)
                box = UsdGeom.Cube.Define(stage, f"{part.GetPath()}/collider")
                box.CreateSizeAttr(1.0)
                box.AddTranslateOp().Set(Gf.Vec3d(*((low+high)/2).tolist()))
                box.AddScaleOp().Set(Gf.Vec3f(*(high-low).tolist()))
                box.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    stage.GetRootLayer().Save()
    (folder / "model.json").write_text(json.dumps({
        "source": str(URDF), "source_hash": snapshot.mesh_hash, "joints_rad": cfg["joints"],
        "mass_kg": snapshot.mass, "urdf_com_m": snapshot.com.tolist(),
        "inertia_kg_m2": snapshot.inertia.tolist(), "hubs_com_m": snapshot.hubs.tolist(),
        "axes": snapshot.axes.tolist(), "parameter_source": cfg["parameter_source"],
    }, indent=2) + "\n")
    return path
