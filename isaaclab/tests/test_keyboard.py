"""检查不同 Kit 按键格式下的运动指令和悬停行为，无需启动仿真。"""

from enum import Enum, auto
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aeromorph_control.keyboard import Keyboard


class EventType(Enum):
    KEY_PRESS = auto()
    KEY_RELEASE = auto()
    KEY_REPEAT = auto()


Key = Enum("Key", "W S A D R F H")


class KeyboardTests(unittest.TestCase):
    def setUp(self):
        # 仅替代外部事件类型，测试使用生产代码中的事件处理与指令生成逻辑。
        carb = ModuleType("carb")
        carb.input = ModuleType("carb.input")
        carb.input.KeyboardEventType = EventType
        modules = patch.dict(sys.modules, {"carb": carb, "carb.input": carb.input})
        modules.start()
        self.addCleanup(modules.stop)

    def test_press_and_release_for_strings_and_enums(self):
        directions = {
            "W": [1, 0, 0], "S": [-1, 0, 0],
            "A": [0, 1, 0], "D": [0, -1, 0],
            "R": [0, 0, 1], "F": [0, 0, -1],
        }
        for encode in (str, Key.__getitem__):
            for name, expected in directions.items():
                with self.subTest(format=encode, key=name):
                    keyboard = Keyboard()
                    event = SimpleNamespace(input=encode(name), type=EventType.KEY_PRESS)
                    self.assertTrue(keyboard.on_event(event))
                    np.testing.assert_array_equal(keyboard.command(), expected)
                    event.type = EventType.KEY_REPEAT
                    keyboard.on_event(event)
                    np.testing.assert_array_equal(keyboard.command(), expected)
                    event.type = EventType.KEY_RELEASE
                    keyboard.on_event(event)
                    np.testing.assert_array_equal(keyboard.command(), [0, 0, 0])
                    self.assertFalse(keyboard.keys)

    def test_h_brakes_until_movement_keys_are_released(self):
        for encode in (str, Key.__getitem__):
            with self.subTest(format=encode):
                keyboard = Keyboard()

                def send(name, event_type):
                    keyboard.on_event(SimpleNamespace(input=encode(name), type=event_type))

                send("W", EventType.KEY_PRESS)
                send("R", EventType.KEY_PRESS)
                np.testing.assert_array_equal(keyboard.command(), [1, 0, 1])
                send("H", EventType.KEY_PRESS)
                np.testing.assert_array_equal(keyboard.command(), [0, 0, 0])
                send("H", EventType.KEY_RELEASE)
                np.testing.assert_array_equal(keyboard.command(), [0, 0, 0])
                send("W", EventType.KEY_RELEASE)
                np.testing.assert_array_equal(keyboard.command(), [0, 0, 0])
                send("R", EventType.KEY_RELEASE)
                np.testing.assert_array_equal(keyboard.command(), [0, 0, 0])
                send("A", EventType.KEY_PRESS)
                np.testing.assert_array_equal(keyboard.command(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
