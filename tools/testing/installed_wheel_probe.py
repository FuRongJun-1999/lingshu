#!/usr/bin/env python3
"""Probe the real CLI and asset from an isolated installed-wheel interpreter.

The acceptance driver copies this file outside the checkout and invokes it with
``python -I``. Product imports must resolve to that interpreter's site-packages.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

MODES = {"roleplay", "world", "story", "history", "causal"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def strict_json(raw):
    def reject(value):
        raise ValueError("non-JSON numeric constant: " + value)
    return json.loads(raw, parse_constant=reject)


def validate_views(data, expected):
    require(data.get("schema_version") == 1 and data.get("fixture") is True,
            "CLI did not return the fixture schema")
    require(set(data["views"]) == expected, "CLI omitted or added a requested task")
    for mode, view in data["views"].items():
        require(view["mode"] == mode and view["state"] == "DEFER", "fixture qualification changed")
        require(bool(view["facts"]) and all(node["provenance"]["source"] == "fixture"
                                           for node in view["facts"]),
                "fixture facts lost their provenance")
    if expected == MODES:
        views = data["views"]
        role = views["roleplay"]
        require(role["payload"]["identity"][0]["id"] == "keeper", "role identity disappeared")
        require(not {"other-role", "other-session"} & {node["id"] for node in role["facts"]},
                "role or session filtering changed")
        require({node["id"] for node in views["history"]["facts"]} == {"catalogue"},
                "history domain filtering changed")
        require(views["world"]["payload"]["entities"][0]["state"] == "on", "world state was not projected")
        paths = views["causal"]["payload"]["paths"]
        require(len(paths) == 1 and len(paths[0]["edges"]) == 8 and paths[0]["hypothesis"] is True,
                "causal path lost edges or qualification")
        outline = [node["node_id"] for node in views["story"]["payload"]["outline"]]
        require(outline.index("action") < outline.index("result"), "story order violates causality")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("light", "full"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    require(not Path.cwd().resolve().is_relative_to(source), "probe cwd is inside source")
    require("PYTHONPATH" not in os.environ and "PYTHONHOME" not in os.environ,
            "installed probe inherited a Python source path")
    require(sys.flags.isolated == 1, "installed probe requires isolated Python")

    import lingshu

    imported = Path(lingshu.__file__).resolve()
    roots = {Path(sysconfig.get_path(name)).resolve() for name in ("purelib", "platlib")}
    require(any(imported.is_relative_to(root) for root in roots), "product import is outside site-packages")
    require(not imported.is_relative_to(source), "probe imported source checkout")
    distribution = importlib.metadata.distribution("lingshu")
    direct = distribution.read_text("direct_url.json")
    require(direct is not None and strict_json(direct).get("dir_info", {}).get("editable") is not True,
            "probe is not a regular wheel installation")
    available = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "PIL")}
    if args.phase == "light":
        require(not any(available.values()), "light environment contains NumPy or Pillow")
    else:
        require(all(available.values()), "full extra did not install NumPy and Pillow")

    cli_results = []
    for mode in ("all", *sorted(MODES)):
        command = [sys.executable, "-I", "-B", "-X", "utf8", "-m", "lingshu.world.scenario_cli",
                   "demo", "--mode", mode]
        result = subprocess.run(command, cwd=output, capture_output=True, timeout=60)
        (output / (mode + ".stdout.json")).write_bytes(result.stdout)
        (output / (mode + ".stderr.log")).write_bytes(result.stderr)
        require(result.returncode == 0, "installed CLI failed for " + mode)
        data = strict_json(result.stdout.decode("utf-8"))
        validate_views(data, MODES if mode == "all" else {mode})
        cli_results.append({"mode": mode, "views": sorted(data["views"]), "returncode": result.returncode,
                            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest()})
    report = {"accepted": False, "phase": args.phase, "python": sys.version,
              "package_file": str(imported), "site_packages": sorted(str(root) for root in roots),
              "isolated": True, "editable": False, "optional_packages_available": available,
              "cli_results": cli_results}
    if args.phase == "full":
        png = output / "world.png"
        result = subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", "-m",
                                 "lingshu.world.scenario_cli", "demo", "--mode", "world",
                                 "--render", str(png)], cwd=output, capture_output=True, timeout=60)
        (output / "render.stdout.json").write_bytes(result.stdout)
        (output / "render.stderr.log").write_bytes(result.stderr)
        require(result.returncode == 0 and png.is_file(), "installed render did not produce an image")
        data = strict_json(result.stdout.decode("utf-8"))
        validate_views(data, {"world"})
        require(data["render"]["evidence"] is False, "render changed its evidence qualification")
        from PIL import Image
        with Image.open(png) as image:
            image.load()
            require(image.format == "PNG" and image.size == (500, 500), "render image format or dimensions changed")
            colors = image.convert("RGB").getcolors(250001)
            require(colors is not None and len(colors) > 1, "render image is uniform or unreadable")
            report["render"] = {"size": list(image.size), "colors": len(colors),
                                 "sha256": hashlib.sha256(png.read_bytes()).hexdigest()}
        from importlib.resources import files
        import numpy as np
        from lingshu.nn.hex_cnn import hex_conv, load_consolidated
        asset = files("lingshu.nn").joinpath("assets/kernels_img0.json")
        require(asset.is_file(), "installed package asset is missing")
        raw = asset.read_bytes()
        kernels = load_consolidated(str(asset))
        require(bool(kernels) and all(kernel.shape == (7,) and np.isfinite(kernel).all()
                                     for kernel in kernels.values()), "installed kernels are not usable")
        field = np.full((4, 4, 3), 0.5, dtype=np.float32)
        for name, kernel in kernels.items():
            response = hex_conv(field, kernel)
            expected = 0.5 * float(kernel.sum())
            require(math.isfinite(expected) and response.shape == field.shape,
                    "installed kernel response has wrong shape")
            np.testing.assert_allclose(response, expected, atol=1e-6, rtol=0,
                                       err_msg="installed asset constant-field oracle: " + name)
        report["asset_sha256"] = hashlib.sha256(raw).hexdigest()
        report["asset_kernels_checked"] = sorted(kernels)
        report["installed_dependencies"] = {name: importlib.metadata.version(name) for name in ("numpy", "Pillow")}
    report["accepted"] = True
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                   allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": True, "phase": args.phase, "package_file": str(imported)}))


if __name__ == "__main__":
    main()
