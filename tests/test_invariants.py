"""The repository's invariants (AGENTS.md), each as a test."""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_the_truth_stays_in_the_world():
    for path in (ROOT / "tcs").rglob("*.py"):
        assert not imports(path) & {"world", "evals"}, path


def test_the_world_does_not_read_the_products_settings():
    for path in (ROOT / "world").rglob("*.py"):
        text = path.read_text()
        assert "tcs" not in imports(path), path
        assert not any(name in text for name in ("platform.toml", "designs.toml", "api.toml")), path


def test_settings_live_in_config():
    """Every setting in config/platform.toml is read somewhere, and the plane hides no tuning constant."""
    cfg = tomllib.loads((ROOT / "config" / "platform.toml").read_text())
    leaves: set[str] = set()

    def walk(d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                for item in v:
                    walk(item)
            else:
                leaves.add(k)

    walk(cfg)
    code = "".join(p.read_text() for d in ("tcs", "evals") for p in (ROOT / d).rglob("*.py"))
    unused = {k for k in leaves if f'"{k}"' not in code}
    assert not unused, f"settings nobody reads: {sorted(unused)}"
    allowed = {0.0, 0.5, 1.0, 3600.0, 1e9}  # units and guards
    for node in ast.walk(ast.parse((ROOT / "tcs" / "plane" / "plane.py").read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            assert node.value in allowed, f"literal {node.value} belongs in config/platform.toml"


def test_every_design_sets_every_flag():
    from tcs import config

    designs = config.designs()
    assert {"baseline", "llm_style", "platform"} <= set(designs)


def test_reports_are_generated():
    from evals import readme

    if not (ROOT / "reports" / "dev.json").exists():
        return
    assert readme.render_block() in (ROOT / "README.md").read_text()


def test_held_out_ran_once_behind_a_lock():
    out = ROOT / "reports" / "heldout.json"
    if not out.exists():
        return
    lock = json.loads((ROOT / "reports" / "heldout.lock").read_text())
    report = json.loads(out.read_text())
    assert lock["commit"] == report["commit"] and len(lock["commit"]) == 40
    assert report["world"] == "heldout"


def test_the_heldout_world_differs_from_the_dev_world():
    dev = tomllib.loads((ROOT / "config" / "world.toml").read_text())
    held = tomllib.loads((ROOT / "config" / "heldout.toml").read_text())
    assert dev["campaign"]["dev_seeds"] != held["campaign"]["dev_seeds"]
    assert dev["physics"] != held["physics"] and dev["traffic"] != held["traffic"]
    assert [g["name"] for g in dev["gpu"]] == [g["name"] for g in held["gpu"]]  # same inventory the plane knows
