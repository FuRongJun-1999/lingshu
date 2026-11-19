"""Generate independent multi-view geometry truth using Blender Cycles on CPU.

Run with normal Python and an explicit ``--blender`` executable. NumPy is used
inside Blender, which supplies its own Python. No DA3, SfM, downloaded texture,
or reconstruction result participates in geometry, camera or depth truth.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys


def scene_specs():
    """Twenty fixed independent geometry/material/camera configurations."""
    scenes = []
    for index in range(20):
        family = ("layered_boxes" if index < 7 else
                  "curves_slopes" if index < 14 else "texture_light_baseline")
        scenes.append({
            "id": f"scene-{index:02d}", "index": index, "family": family,
            "seed": 20261119 + 137 * index,
            "objects": 3 + index % 4, "focal_mm": 35 + (index % 4) * 4,
            "camera_radius": 5.4 + .35 * (index % 5),
            "camera_arc_radians": .12 if index == 19 else .32 + .035 * (index % 4),
            "camera_height": 2.5 + .15 * (index % 3),
            "texture_contrast": .22 if index in (15, 18) else .85,
            "light_energy": 450 if index % 3 else 220,
        })
    return scenes


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _reset_scene(bpy, args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = args.threads
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.world = bpy.data.worlds.new("truth-world")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = .65
    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    render = nodes.new("CompositorNodeRLayers")
    depth = nodes.new("CompositorNodeOutputFile")
    depth.format.file_format = "OPEN_EXR"
    depth.format.color_depth = "32"
    depth.format.color_mode = "BW"
    scene.node_tree.links.new(render.outputs["Depth"], depth.inputs[0])
    return scene, depth


def _camera(bpy, scene, focal, eye, target, args):
    Vector = importlib.import_module("mathutils").Vector
    camera_data = bpy.data.cameras.new("truth-camera")
    camera_data.lens = focal
    camera_data.sensor_width = 36
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.clip_start, camera_data.clip_end = .05, 100
    camera = bpy.data.objects.new("truth-camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = eye
    camera.rotation_euler = (Vector(target) - Vector(eye)).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    return camera


def _camera_record(camera, args):
    np = importlib.import_module("numpy")
    bpy = importlib.import_module("bpy")
    bpy.context.view_layer.update()
    world_to_blender = np.asarray(camera.matrix_world.inverted(), dtype=np.float64)
    world_to_camera = np.diag((1., -1., -1.)) @ world_to_blender[:3, :]
    focal = camera.data.lens / camera.data.sensor_width * args.width
    return {"width": args.width, "height": args.height,
            "intrinsics": [[focal, 0, args.width / 2], [0, focal, args.height / 2], [0, 0, 1]],
            "cam_from_world": world_to_camera.tolist()}


def _load_image_pixels(bpy, path, height, width):
    np = importlib.import_module("numpy")
    image = bpy.data.images.load(str(Path(path).resolve()), check_existing=False)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    result = pixels.reshape(height, width, image.channels)[::-1].copy()
    bpy.data.images.remove(image)
    return result


def _render(bpy, scene, depth_output, output, name, args):
    raw = output / "truth/raw_depth"
    raw.mkdir(parents=True, exist_ok=True)
    image = output / "images" / name
    image.parent.mkdir(parents=True, exist_ok=True)
    depth_output.base_path = str(raw.resolve())
    depth_output.file_slots[0].path = Path(name).stem + "-"
    scene.render.filepath = str(image.resolve())
    scene.frame_set(1)
    bpy.ops.render.render(write_still=True)
    candidates = list(raw.glob(Path(name).stem + "-*.exr"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one Blender depth pass for {name}; got {candidates}")
    depth = _load_image_pixels(bpy, candidates[0], args.height, args.width)[:, :, 0]
    rgba = _load_image_pixels(bpy, image, args.height, args.width)
    mask = (rgba[:, :, 3] > .5) & (depth > 0) & (depth < 100)
    return depth, mask


def _ray_norm(camera):
    np = importlib.import_module("numpy")
    ys, xs = np.mgrid[:camera["height"], :camera["width"]]
    k = np.asarray(camera["intrinsics"])
    return np.sqrt(((xs + .5 - k[0, 2]) / k[0, 0]) ** 2
                   + ((ys + .5 - k[1, 2]) / k[1, 1]) ** 2 + 1)


def _depth_calibration(bpy, args):
    """Verify depth-pass semantics against a known frontoparallel plane."""
    np = importlib.import_module("numpy")
    scene, depth_output = _reset_scene(bpy, args)
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0))
    _camera(bpy, scene, 36, (0, 0, 5), (0, 0, 0), args)
    camera = _camera_record(scene.camera, args)
    output = args.output / "calibration"
    raw, mask = _render(bpy, scene, depth_output, output, "plane.png", args)
    candidates = {"camera_z": raw, "euclidean_ray_distance": raw / _ray_norm(camera)}
    errors = {name: float(np.median(np.abs(depth[mask] - 5))) for name, depth in candidates.items()}
    selected = min(errors, key=errors.get)
    if not mask.any() or errors[selected] > .005:
        raise RuntimeError(f"Blender independent plane depth calibration failed: {errors}")
    result = {"truth_source": "Blender Cycles CPU", "blender_version": bpy.app.version_string,
              "selected_depth_semantics": selected, "expected_plane_z_m": 5,
              "median_abs_error_m": errors, "valid_pixels": int(mask.sum()),
              "pixel_center_offset": .5, "depth_is_independent_of_reconstruction": True}
    _write_json(output / "depth_calibration.json", result)
    return result


def _material(bpy, seed, contrast):
    np = importlib.import_module("numpy")
    rng = np.random.default_rng(seed)
    side = 256
    texture = np.zeros((side, side, 3), dtype=np.float64)
    for cells, weight in ((8, .25), (16, .35), (32, .25), (64, .15)):
        grid = rng.random((cells, cells, 3))
        texture += weight * np.repeat(np.repeat(grid, side // cells, axis=0), side // cells, axis=1)
    for axis in (0, 1):
        texture = (np.roll(texture, 1, axis) + 2 * texture + np.roll(texture, -1, axis)) / 4
    texture = np.clip(.5 + contrast * (texture - .5) * 2.3, .04, .96)
    rgba = np.concatenate((texture, np.ones((side, side, 1))), axis=2).astype(np.float32)
    image = bpy.data.images.new(f"independent-texture-{seed}", width=side, height=side)
    image.pixels.foreach_set(rgba.ravel())
    material = bpy.data.materials.new(f"material-{seed}")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    shader = nodes.get("Principled BSDF")
    shader.inputs["Roughness"].default_value = .9
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.image = image
    image_node.interpolation = "Linear"
    material.node_tree.links.new(image_node.outputs["Color"], shader.inputs["Base Color"])
    return material


def _geometry(bpy, spec):
    np = importlib.import_module("numpy")
    rng = np.random.default_rng(spec["seed"])
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, -.03))
    floor = bpy.context.object
    floor.scale = (3.4 + .2 * (spec["index"] % 3), 2.8, 1)
    floor.data.materials.append(_material(bpy, spec["seed"] - 1, spec["texture_contrast"]))
    for index in range(spec["objects"]):
        x = (index - (spec["objects"] - 1) / 2) * .75 + rng.uniform(-.12, .12)
        y = rng.uniform(-.65, .8)
        height = rng.uniform(.7, 1.8)
        if spec["family"] == "curves_slopes" and index % 3 == 0:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=.5,
                                               location=(x, y, height / 2))
            obj = bpy.context.object
            obj.scale = (rng.uniform(.8, 1.2), .9, height)
        elif spec["family"] != "layered_boxes" and index % 3 == 1:
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=.35, depth=height,
                                                location=(x, y, height / 2))
            obj = bpy.context.object
            obj.rotation_euler[2] = rng.uniform(-.4, .4)
        else:
            bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, height / 2))
            obj = bpy.context.object
            obj.scale = (rng.uniform(.45, .8), rng.uniform(.55, 1), height)
            obj.rotation_euler[2] = rng.uniform(-.35, .35)
            if spec["family"] == "curves_slopes":
                obj.rotation_euler[1] = .23
        obj.data.materials.append(_material(bpy, spec["seed"] + index, spec["texture_contrast"]))
    light_data = bpy.data.lights.new("truth-area", type="AREA")
    light_data.energy, light_data.size = spec["light_energy"], 5
    light = bpy.data.objects.new("truth-area", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = (-3, -4, 7)
    Vector = importlib.import_module("mathutils").Vector
    light.rotation_euler = (Vector((0, 0, 1)) - light.location).to_track_quat("-Z", "Y").to_euler()


def _save_mesh(bpy, output):
    np = importlib.import_module("numpy")
    vertices, triangles = [], []
    bpy.context.view_layer.update()
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.to_mesh()
        mesh.calc_loop_triangles()
        offset = len(vertices)
        vertices.extend(tuple(obj.matrix_world @ vertex.co) for vertex in mesh.vertices)
        triangles.extend(tuple(offset + v for v in face.vertices) for face in mesh.loop_triangles)
        obj.to_mesh_clear()
    vertices, triangles = np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32)
    np.savez_compressed(output / "mesh.npz", vertices=vertices, triangles=triangles)
    with (output / "mesh.ply").open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(vertices)}\n")
        stream.write("property double x\nproperty double y\nproperty double z\n")
        stream.write(f"element face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n")
        for vertex in vertices:
            stream.write(" ".join(f"{v:.12g}" for v in vertex) + "\n")
        for triangle in triangles:
            stream.write("3 " + " ".join(str(v) for v in triangle) + "\n")
    return {"vertices": len(vertices), "triangles": len(triangles)}


def _generate_scene(bpy, args, spec, calibration):
    np = importlib.import_module("numpy")
    output = args.output / spec["id"]
    if (output / "scene.json").exists():
        print(f"TRUTH_ALREADY_GENERATED {spec['id']}", flush=True)
        return
    truth = output / "truth"
    (truth / "depth").mkdir(parents=True, exist_ok=True)
    scene, depth_output = _reset_scene(bpy, args)
    scene.cycles.seed = spec["seed"] % 100000
    _geometry(bpy, spec)
    geometry = _save_mesh(bpy, truth)
    cameras = []
    target = (0, .1, .8)
    for index, angle in enumerate(np.linspace(-spec["camera_arc_radians"], spec["camera_arc_radians"], args.views)):
        eye = (math.sin(angle) * spec["camera_radius"], -math.cos(angle) * spec["camera_radius"],
               spec["camera_height"] + .12 * math.cos(angle * 4 + .4 * spec["index"]))
        camera = _camera(bpy, scene, spec["focal_mm"], eye, target, args)
        record = _camera_record(camera, args)
        name = f"view-{index:03d}.png"
        raw_depth, mask = _render(bpy, scene, depth_output, output, name, args)
        z = raw_depth / _ray_norm(record) if calibration["selected_depth_semantics"] == "euclidean_ray_distance" else raw_depth
        z = np.where(mask, z, np.nan).astype(np.float32)
        depth_relative = f"truth/depth/{Path(name).stem}.npz"
        np.savez_compressed(output / depth_relative, depth_z=z, mask=mask.astype(np.uint8))
        record.update(image=name, depth=depth_relative, foreground_pixels=int(mask.sum()))
        cameras.append(record)
        scene.camera = None
        bpy.data.objects.remove(camera, do_unlink=True)
        print(f"TRUTH_VIEW {spec['id']} {index + 1}/{args.views}", flush=True)
    metadata = {"schema": "lingshu.independent-reconstruction-truth/1", "truth_source": "Blender Cycles CPU",
                "blender_version": bpy.app.version_string, "coordinate_units": "meters",
                "world_axes": "Blender x/y horizontal, z up; cameras x right y down z forward",
                "pixel_center_offset": .5, "depth_semantics": "camera-forward z in meters",
                "calibration": calibration, "cameras": cameras}
    _write_json(truth / "cameras.json", metadata)
    _write_json(output / "scene.json", {**spec, "truth_source": "Blender Cycles CPU", "geometry": geometry,
                                       "views": args.views, "width": args.width, "height": args.height,
                                       "cycles_samples": args.samples, "threads": args.threads,
                                       "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(f"TRUTH_SCENE_DONE {spec['id']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=Path, help="explicit Blender executable; never downloaded automatically")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", help="scene IDs to generate; suite denominator remains 20")
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--inside-blender", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.views < 3 or min(args.width, args.height, args.samples, args.threads) < 1:
        parser.error("at least 3 views and positive render settings are required")
    specs = scene_specs()
    if args.cases is not None and not set(args.cases) <= {spec["id"] for spec in specs}:
        parser.error("unknown --cases ID")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    suite = {"schema": "lingshu.independent-reconstruction-suite/1", "truth_source": "Blender Cycles CPU",
             "fixed_denominator": 20, "views_per_scene": args.views, "width": args.width, "height": args.height,
             "source_of_truth": "independent procedural scene/cameras/rendered depth; no DA3 or SfM",
             "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "scenes": specs}
    suite_path = args.output / "suite.json"
    if suite_path.exists() and _read_existing_suite(suite_path) != suite:
        parser.error("existing suite configuration differs; use a new output directory")
    _write_json(suite_path, suite)
    if args.inside_blender:
        bpy = importlib.import_module("bpy")
        calibration = _depth_calibration(bpy, args)
        for spec in specs:
            if args.cases is None or spec["id"] in args.cases:
                _generate_scene(bpy, args, spec, calibration)
        return 0
    if args.blender is None or not args.blender.is_file():
        parser.error("--blender must name an available Blender executable")
    command = [str(args.blender.resolve()), "--background", "--factory-startup", "--threads", str(args.threads),
               "--python", str(Path(__file__).resolve()), "--", "--inside-blender", "--output", str(args.output),
               "--views", str(args.views), "--width", str(args.width), "--height", str(args.height),
               "--samples", str(args.samples), "--threads", str(args.threads)]
    if args.cases is not None:
        command += ["--cases", *args.cases]
    return subprocess.run(command).returncode


def _read_existing_suite(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--inside-blender" in sys.argv and "--" in sys.argv else None
    raise SystemExit(main(arguments))
