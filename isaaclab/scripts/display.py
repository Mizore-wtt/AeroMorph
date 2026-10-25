'''
在 Isaac Lab 中显示固定的 AeroMorph，仅提供四个变构关节滑块
author:GPT-6.1 high
review:wutong
date:10.24
'''

import math
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/aeromorph_description"
FOLD_JOINTS = ("joint1", "joint2", "joint3", "joint4")


def create_model(stage):
    """读取 URDF 文件，转换为 USD 格式"""
    import numpy as np
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    urdf = ET.parse(PACKAGE / "urdf/AeroMorph.urdf").getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    children = {}
    child_links = set()
    for joint in urdf.findall("joint"):
        parent = joint.find("parent").get("link")
        children.setdefault(parent, []).append(joint)
        child_links.add(joint.find("child").get("link"))
    roots = set(links) - child_links
    if len(roots) != 1:
        raise ValueError("AeroMorph URDF 必须有且仅有一个根连杆")
    joint_ops = {}

    def set_origin(xform, element):
        if element is None:
            return
        xyz = [float(v) for v in element.get("xyz", "0 0 0").split()]
        rpy = [math.degrees(float(v)) for v in element.get("rpy", "0 0 0").split()]
        xform.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rpy))

    def add_link(name, path):
        # 加载 STL 模型
        UsdGeom.Xform.Define(stage, path)
        for index, visual in enumerate(links[name].findall("visual")):
            part = UsdGeom.Xform.Define(stage, f"{path}/visual_{index}")
            set_origin(part, visual.find("origin"))
            mesh_element = visual.find("geometry/mesh")
            uri = mesh_element.get("filename")
            prefix = "package://aeromorph_description/"
            if not uri.startswith(prefix):
                raise ValueError(f"不支持的模型路径: {uri}")
            mesh_path = PACKAGE / uri[len(prefix):]
            raw = mesh_path.read_bytes()
            count = int.from_bytes(raw[80:84], "little")
            if not count or len(raw) != 84 + count * 50:
                raise ValueError(f"需要有效二进制 STL: {mesh_path}")
            triangle_type = np.dtype([
                ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2"),
            ])
            triangles = np.frombuffer(raw, dtype=triangle_type, count=count, offset=84)
            points = triangles["vertices"].reshape(-1, 3).copy()
            scale = np.array([float(v) for v in mesh_element.get("scale", "1 1 1").split()])
            points *= scale
            mesh = UsdGeom.Mesh.Define(stage, f"{part.GetPath()}/mesh")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(count, 3, dtype=np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(count * 3, dtype=np.int32)))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateExtentAttr(UsdGeom.Mesh.ComputeExtent(mesh.GetPointsAttr().Get()))
            color = visual.find("material/color")
            rgb = [float(v) for v in color.get("rgba").split()[:3]] if color is not None else [0.8] * 3
            material = UsdShade.Material.Define(stage, f"/World/Materials/{name}_{index}")
            shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

        for joint in children.get(name, []):
            # 建立关节层级关系
            joint_path = f"{path}/{joint.get('name')}"
            xform = UsdGeom.Xform.Define(stage, joint_path)
            set_origin(xform, joint.find("origin"))
            if joint.get("type") != "fixed":
                axis = Gf.Vec3d(*[float(v) for v in joint.find("axis").get("xyz").split()])
                op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                op.Set(Gf.Quatd(1.0))
                joint_ops[joint.get("name")] = (op, axis)
            child = joint.find("child").get("link")
            add_link(child, f"{joint_path}/{child}")

    UsdGeom.Xform.Define(stage, "/World")
    root = UsdGeom.Xform.Define(stage, "/World/AeroMorph")
    root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
    root_name = roots.pop()
    add_link(root_name, f"{root.GetPath()}/{root_name}")
    return joint_ops


def create_joint_sliders(joint_ops):
    # 创建关节控制滑块
    import omni.ui as ui
    from pxr import Gf

    window = ui.Window("AeroMorph joints", width=320, height=190)
    window.position_x = 940
    window.position_y = 80
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label("Joint angles [rad]")
            for name in FOLD_JOINTS:
                op, axis = joint_ops[name]
                with ui.HStack(height=28):
                    ui.Label(name, width=70)
                    # 关节控制范围：从 -3.0～3.0(rad)
                    slider = ui.FloatSlider(min=-3.0, max=3.0, step=0.01)
                    slider.model.add_value_changed_fn(
                        lambda model, op=op, axis=axis: op.Set(
                            Gf.Rotation(axis, math.degrees(model.as_float)).GetQuat()
                        )
                    )
    return window


def main():
    from isaaclab.app import AppLauncher

    log_dir = ROOT / "log/isaaclab"
    log_dir.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(
        device="cuda:0", visualizer=["kit"],
        width=1280, height=720, 
        window_width=1280, window_height=800,
        kit_args=f"--/log/file={log_dir}/kit.log",
    ).app
    window = None
    exit_code = 0

    try:
        # Kit 启动后再导入其模块；这里只渲染外观，不启动物理仿真。
        import carb.settings
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, UsdGeom, UsdLux

        # 创建 USD 场景，加载机器人
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        joint_ops = create_model(stage)

        # 保留穹顶照明，用深灰色覆盖视口背景。
        light = UsdLux.DomeLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(500.0)
        settings = carb.settings.get_settings()
        settings.set_int("/rtx/background/source/type", 2)
        settings.set("/rtx/background/source/color", [0.025, 0.025, 0.025])
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(1.4, 1.6, 1.8), Gf.Vec3d(-0.12, 0.0, 1.0), Gf.Vec3d(0, 0, 1),
        ).GetInverse())
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        camera.CreateFocalLengthAttr(24.0)
        viewport = get_active_viewport()
        viewport.camera_path = str(camera.GetPath())
        settings.set_bool("/app/viewport/grid/enabled", True)
        settings.set_bool(f"/persistent/app/viewport/{viewport.id}/guide/grid/visible", True)
        
        # 创建滑块
        window = create_joint_sliders(joint_ops)
        print("[AeroMorph] 模型已加载，拖动四个滑块调整变构关节。", flush=True)
        while app.is_running():
            app.update()
            
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if window is not None:
            window.destroy()
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
