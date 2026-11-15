"""One-time cooked collision geometry for the production manipulation solver.

Cold preparation loads collision shapes only, without planners, rendering,
simulation steps or task execution. M11 freezes the resulting geometry; M12's
candidate loop is pure CPU and never loads assets or starts a simulator.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from transforms3d.quaternions import axangle2quat, qmult

from if_benchmark.schema import artifact_hash

CATALOG_VERSION = "cooked_operation_geometry_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cooking_implementation_sources(root):
    """Invalidate cooked shapes when extraction or Actor interpretation changes."""
    paths = [Path(__file__), Path(__file__).with_name("operation_space_live.py"),
             Path(__file__).parents[1] / "grasp_validation/convex_cover.py",
             Path(__file__).parents[1] / "initial_pose/runtime.py",
             Path(root) / "envs/utils/actor_utils.py",
             Path(root) / "envs/utils/create_actor.py"]
    return {str(path.resolve()): digest(path) for path in paths}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, allow_nan=False, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@lru_cache(maxsize=1)
def _cooking_scene(process_id):
    import sapien.core as sapien
    engine = sapien.Engine()
    scene = engine.create_scene(sapien.SceneConfig())
    scene.default_physical_material = scene.create_physical_material(.5, .5, 0.)
    return engine, scene


class OperationGeometryCatalog:
    def __init__(self, root, cache=None):
        self.root = Path(root).resolve()
        self.cache = Path(cache) if cache else self.root / ".cache/if_benchmark/operation_geometry"
        self.memory = {}

    def _get(self, identity, build):
        implementation = cooking_implementation_sources(self.root)
        identity = {**identity, "files": {**identity.get("files", {}), **implementation}}
        key = artifact_hash({"version": CATALOG_VERSION, "source": identity,
                             "implementation": implementation})
        if key in self.memory:
            return self.memory[key]
        path = self.cache / (key + ".json")
        if path.exists():
            result = json.loads(path.read_text())
            if result.get("key") != key or result.get("content_hash") != artifact_hash(result, omit_keys=("content_hash",)):
                raise ValueError("corrupt operation geometry cache")
        else:
            result = {"key": key, "identity": identity, "geometry": build()}
            result["content_hash"] = artifact_hash(result)
            atomic_json(path, result)
        self.memory[key] = result
        return result

    def robot(self):
        import yaml
        root = self.root / "assets/embodiments/aloha-agilex"
        config = yaml.safe_load((root / "config.yml").read_text())
        urdf = (root / config["urdf_path"]).resolve()
        tree = ET.parse(urdf)
        paths = {urdf, root / "config.yml"}
        for node in tree.findall(".//collision/geometry/mesh"):
            name = node.get("filename")
            if name.startswith("package://"):
                raise ValueError("unresolved robot collision package URI")
            paths.add((urdf.parent / name).resolve())
        source = {"files": {str(p.resolve()): digest(p) for p in sorted(paths)}}
        def build():
            from .operation_space_live import extract_robot_geometry
            _, scene = _cooking_scene(os.getpid())
            loader = scene.create_urdf_loader()
            loader.fix_root_link = True
            actor = loader.load(str(urdf))
            if actor is None:
                raise ValueError("operation robot collision load failed")
            try:
                robot = SimpleNamespace()
                for index, arm in enumerate(("left", "right")):
                    names = config["gripper_name"][index]
                    joints = [(actor.find_joint_by_name(names["base"]), 1., 0.)]
                    joints += [(actor.find_joint_by_name(n), float(m), float(b)) for n, m, b in names["mimic"]]
                    setattr(robot, arm + "_entity", actor)
                    setattr(robot, arm + "_urdf_path", str(urdf))
                    setattr(robot, arm + "_gripper", joints)
                    setattr(robot, arm + "_gripper_scale", config["gripper_scale"])
                return {arm: extract_robot_geometry(robot, arm) for arm in ("left", "right")}
            finally:
                scene.remove_articulation(actor)
        return self._get(source, build)

    def subject(self, binding, local):
        from envs.utils.create_actor import get_glb_or_obj_file
        root = self.root / "assets/objects" / str(binding["asset"])
        model_id = None if str(binding["model_id"]) == "default" else int(binding["model_id"])
        metadata = root / ("model_data.json" if model_id is None else f"model_data{model_id}.json")
        mesh = get_glb_or_obj_file(root / "collision", model_id) if (root / "collision").exists() else None
        if not mesh or not mesh.exists():
            mesh = get_glb_or_obj_file(root, model_id)
        if not mesh or not mesh.exists():
            raise ValueError(f"operation geometry missing: {binding}")
        if (local.get("articulated_joint_state") or {}).get("qpos"):
            raise ValueError("articulated manipulated object has no rigid operation template")
        source = {"asset": binding["asset"], "model_id": binding["model_id"],
                  "mesh": digest(mesh), "metadata": digest(metadata),
                  "files": {str(mesh.resolve()): digest(mesh), str(metadata.resolve()): digest(metadata)},
                  "pose": {k: local.get(k) for k in ("q0_wxyz", "relative_yaw_deg", "scale_xyz", "canonical_state_hash")}}
        def build():
            import sapien.core as sapien
            from envs.utils.actor_utils import Actor
            from if_benchmark.initial_pose.runtime import bind_loaded_actor_scale
            from if_benchmark.grasp_validation.convex_cover import live_collision_pieces
            data = json.loads(metadata.read_text())
            scale = np.asarray(local["scale_xyz"], float)
            # Match create_actor: legacy metadata without scale falls back to
            # the canonical caller scale and returns no Actor contact config.
            # Such assets remain valid static landmarks, not grasp candidates.
            runtime_data = data if "scale" in data else None
            if not np.allclose(scale, data.get("scale", scale), atol=1e-9, rtol=0):
                raise ValueError("frozen and runtime asset scales disagree")
            _, scene = _cooking_scene(os.getpid())
            builder = scene.create_actor_builder()
            builder.set_physx_body_type("static")
            builder.add_multiple_convex_collisions_from_file(filename=str(mesh), scale=scale)
            physical = builder.build(name="operation_space_geometry")
            try:
                yaw = np.deg2rad(float(local.get("relative_yaw_deg", 0.)))
                q = qmult(axangle2quat([0, 0, 1], yaw), local["q0_wxyz"])
                physical.set_pose(sapien.Pose([0, 0, 0], q))
                actor = Actor(physical, runtime_data, mass=None)
                bind_loaded_actor_scale(actor, scale)
                geometry = live_collision_pieces(actor)
                contacts = ({str(i): np.asarray(c, float).tolist() for i, c in actor.iter_contact_points("matrix")}
                            if (actor.config or {}).get("contact_points_pose") else {})
                return {"pieces": geometry["pieces"], "sources": geometry["sources"],
                        "contacts": contacts, "quat_wxyz": q.tolist()}
            finally:
                scene.remove_actor(physical)
        return self._get(source, build)

    def obstacle(self, binding, local):
        """Complete receiver geometry at the frozen joint state, including cavities."""
        state = local.get("articulated_joint_state") or {}
        if not state.get("qpos"):
            return self.subject(binding, local)
        import yaml
        if not hasattr(self, "_inventory"):
            self._inventory = yaml.load((self.root / "if_benchmark/data/inventory/geometry_summary.yml").read_text(),
                                        Loader=yaml.CSafeLoader)["models"]
        ref = f"{binding['asset']}|{binding['model_id']}"
        metadata = self.root / self._inventory[ref]["source_path"]
        urdf = metadata.parent / "mobility.urdf"
        paths = {metadata.resolve(), urdf.resolve()}
        for node in ET.parse(urdf).findall(".//collision/geometry/mesh"):
            name = node.get("filename")
            if name.startswith("package://"):
                raise ValueError("unresolved receiver collision package URI")
            paths.add((urdf.parent / name).resolve())
        source = {"files": {str(p): digest(p) for p in sorted(paths)},
                  "pose": {k: local.get(k) for k in ("q0_wxyz", "relative_yaw_deg", "scale_xyz")},
                  "joint_state": state}
        def build():
            import sapien.core as sapien
            from if_benchmark.grasp_validation.convex_cover import live_collision_pieces
            scale = np.asarray(local["scale_xyz"], float)
            if not np.allclose(scale, scale[0], rtol=0, atol=1e-12):
                raise ValueError("URDF receiver requires uniform frozen scale")
            _, scene = _cooking_scene(os.getpid())
            loader = scene.create_urdf_loader()
            loader.scale = float(scale[0]); loader.fix_root_link = True
            loader.load_multiple_collisions_from_file = True
            actor = loader.load(str(urdf))
            if actor is None:
                raise ValueError("receiver collision cooking failed")
            try:
                names = [str(j.get_name()) for j in actor.get_active_joints()]
                rows = sorted(state["joints"], key=lambda r: r["joint_index"])
                if names != [r["joint_name"] for r in rows] or len(state["qpos"]) != len(names):
                    raise ValueError("receiver frozen joint identity mismatch")
                actor.set_qpos(np.asarray(state["qpos"], float))
                yaw = np.deg2rad(float(local.get("relative_yaw_deg", 0.)))
                q = qmult(axangle2quat([0, 0, 1], yaw), local["q0_wxyz"])
                actor.set_pose(sapien.Pose([0, 0, 0], q))
                geometry = live_collision_pieces(actor)
                return {"pieces": geometry["pieces"], "sources": geometry["sources"],
                        "link_matrices": {str(link.get_name()): link.get_pose().to_transformation_matrix().tolist()
                                          for link in actor.get_links()}}
            finally:
                scene.remove_articulation(actor)
        return self._get(source, build)
