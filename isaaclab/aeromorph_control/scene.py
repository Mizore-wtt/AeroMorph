"""飞行场景的灰黑背景、地面参考网格和初始后方视角。"""

import numpy as np
from scipy.spatial.transform import Rotation


def create_grid(stage):
    """绘制不参与碰撞/动力学的参考线，避免播放时编辑器辅助网格被隐藏。"""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    groups = {"minor": [], "major": [], "x_axis": [], "y_axis": []}
    for index in range(-20, 21):
        if index == 0:
            continue
        value = index*.5
        group = "major" if index % 2 == 0 else "minor"
        groups[group].extend([((-10., value), (10., value)), ((value, -10.), (value, 10.))])
    groups["x_axis"] = [((-10., 0.), (10., 0.))]
    groups["y_axis"] = [((0., -10.), (0., 10.))]
    colors = {"minor": (.045,)*3, "major": (.075,)*3,
              "x_axis": (.20, .055, .055), "y_axis": (.055, .18, .055)}
    for name, segments in groups.items():
        width = .003 if name == "minor" else .005
        points = []
        for start, end in segments:
            start, end = np.asarray(start), np.asarray(end)
            delta = end-start
            normal = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)*(width/2)
            points.extend([[*xy, .002] for xy in (start-normal, end-normal, end+normal, start+normal)])
        mesh = UsdGeom.Mesh.Define(stage, f"/World/Grid/{name}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(segments), 4, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(len(points), dtype=np.int32)))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateExtentAttr(UsdGeom.Mesh.ComputeExtent(mesh.GetPointsAttr().Get()))
        material = UsdShade.Material.Define(stage, f"/World/Grid/materials/{name}")
        shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
        # 自发光仅用于稳定显示参考线，网格没有刚体或碰撞 API，不会给无人机提供力。
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colors[name]))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def configure_view(sim, target, yaw, gui):
    """相机位于初始航向后上方；只在启动时设置，允许用户随后自由调整。"""
    heading = Rotation.from_euler("z", yaw).as_matrix()
    eye = target+heading @ np.array([-2.3, 0., 1.3])
    sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
    if not gui:
        return
    import carb.settings
    import omni.ui as ui
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf, UsdGeom

    settings = carb.settings.get_settings()
    settings.set_int("/rtx/background/source/type", 2)
    settings.set("/rtx/background/source/color", [0.025, 0.025, 0.025])
    viewport = get_active_viewport()
    camera = UsdGeom.Camera.Define(sim.stage, "/World/FlightCamera")
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye.tolist()), Gf.Vec3d(*target.tolist()), Gf.Vec3d(0, 0, 1)).GetInverse())
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
    camera.CreateFocalLengthAttr(24.)
    viewport.camera_path = str(camera.GetPath())
    # 用场景内参考线保证 GPU 播放时也有网格，关闭编辑器网格以免重复绘制。
    settings.set_bool("/app/viewport/grid/enabled", False)
    create_grid(sim.stage)
    output_window = ui.Workspace.get_window("Simulation Output Settings")
    if output_window is not None:
        output_window.visible = False
