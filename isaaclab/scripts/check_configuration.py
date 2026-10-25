"""离线检查构型，无需启动 Isaac Sim；使用 Isaac Sim 自带 Python 执行。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aeromorph_control.config import load_config
from aeromorph_control.model import inspect_model
from aeromorph_control.allocator import Allocator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config")
parser.add_argument("--configuration")
parser.add_argument("--joints", nargs=4, type=float, metavar=("J1", "J2", "J3", "J4"))
args = parser.parse_args()
try:
    cfg = load_config(args.config, args.configuration, args.joints)
    model = inspect_model(cfg)
    allocator = Allocator(model, cfg)
    print(json.dumps({"configuration": cfg["configuration"], "joints_rad": cfg["joints"],
                      "mass_kg": model.mass, "com_root_m": model.com.tolist(),
                      "inertia_kg_m2": model.inertia.tolist(),
                      "parameter_source": cfg["parameter_source"], **allocator.report()}, indent=2))
except (ValueError, KeyError) as exc:
    parser.exit(2, f"构型检查失败: {exc}\n")
