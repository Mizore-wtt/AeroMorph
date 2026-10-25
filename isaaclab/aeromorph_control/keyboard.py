"""键盘只生成运动指令；失焦、H 或面板按钮均触发闭环制动。"""

import numpy as np

from .config import JOINTS
from .rotor_visuals import DISPLAY_SPEED_LABELS


class Keyboard:
    def __init__(self):
        self.keys = set()
        self.window = None
        self.subscription = None
        self.holding = False
        self.joint_models = []

    def attach(self, configuration, joints, joint_limits=None):
        import carb.input
        import omni.appwindow
        import omni.ui as ui

        self.app_window = omni.appwindow.get_default_app_window()
        self.interface = carb.input.acquire_input_interface()
        self.keyboard = self.app_window.get_keyboard()
        self.subscription = self.interface.subscribe_to_keyboard_events(self.keyboard, self.on_event)
        self.window = ui.Window("AeroMorph flight", width=375, height=720)
        self.window.position_x, self.window.position_y = 895, 40
        with self.window.frame:
            with ui.VStack(spacing=4):
                ui.Label("PHYSICS FLIGHT | gravity ON", height=22)
                ui.Label(f"Configuration: {configuration}", height=22)
                ui.Label("Click viewport to control; initial heading frame", height=22)
                ui.Label("W/S forward/back | A/D left/right | R/F up/down", height=22)
                ui.Label("H: brake + hold | release keys: brake + hold", height=22)
                ui.Label("Space is reserved for the simulator timeline.", height=22)
                ui.Button("Brake and hold", clicked_fn=self.hold, height=25)
                self.status = ui.Label("Initializing...", word_wrap=True, height=110)
                ui.Separator(height=8)
                ui.Label("In-flight morphing | target angles [rad]", height=22)
                for name in JOINTS:
                    low, high = joint_limits[name]
                    with ui.HStack(height=28):
                        ui.Label(name, width=55)
                        slider = ui.FloatSlider(min=low, max=high, step=0.01)
                        slider.model.set_value(joints[name])
                        self.joint_models.append(slider.model)
                ui.Label("Targets are checked, then followed at limited speed.", height=22)
                self.morph_status = ui.Label("Morph: HOLD", word_wrap=True, height=65)
                ui.Separator(height=8)
                ui.Label("Rotor display speed (automatic)", height=22)
                self.rotor_status = ui.Label("Initializing...", word_wrap=True, height=40)

    def refresh_rotors(self, levels):
        if self.window is None:
            return
        labels = [f"R{i+1}: {DISPLAY_SPEED_LABELS[level]}" for i, level in enumerate(levels)]
        self.rotor_status.text = " | ".join(labels[:2])+"\n"+" | ".join(labels[2:])

    def joint_command(self):
        return np.array([model.as_float for model in self.joint_models]) if self.joint_models else None

    def refresh_morph(self, planner, actual):
        if self.window is None:
            return
        angles = ", ".join(f"{q:+.3f}" for q in actual)
        self.morph_status.text = f"Actual [rad]: {angles}\nMorph: {planner.status}\n{planner.reason}"

    def hold(self):
        # 若 H 与方向键同时按着，要等方向键全部松开才能重新发出运动指令。
        self.holding = True

    def on_event(self, event, *_):
        import carb.input

        # 不同 Kit 版本可能返回按键枚举或字符串，统一为按键名称。
        key = getattr(event.input, "name", event.input)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.keys.add(key)
            if key == "H":
                self.hold()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.keys.discard(key)
        return True

    def command(self):
        if self.subscription is not None and not self.app_window.is_focused():
            self.keys.clear()
        if self.holding:
            self.holding = bool(self.keys & set("WSADRF"))
            return np.zeros(3)
        axis = lambda positive, negative: int(positive in self.keys)-int(negative in self.keys)
        return np.array([axis("W", "S"), axis("A", "D"), axis("R", "F")], dtype=float)

    def refresh(self, p, v, rpy, omega, thrust, mode, saturated):
        if self.window is None:
            return
        fmt = lambda values: ", ".join(f"{x:+.2f}" for x in values)
        self.status.text = (f"XYZ [m]: {fmt(p)}\nVelocity [m/s]: {fmt(v)}\n"
                            f"RPY [deg]: {fmt(np.degrees(rpy))}\nRates [rad/s]: {fmt(omega)}\n"
                            f"Thrust [N]: {fmt(thrust)}\n"
                            f"Mode XYZ: {' / '.join(mode)}\nAllocation saturation: {saturated}")

    def close(self):
        if self.subscription is not None:
            self.interface.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)
        if self.window is not None:
            self.window.destroy()
