"""Worker for Milestone 5 pair terminal-settle probes.

The worker validates whether a requested pair terminal state can physically
exist after settling. It intentionally does not load a robot or run motion
planning; canonical actor skill validation is a later stage.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import importlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml

from if_benchmark.asset_probe_pose import candidate_pose_defaults, resolve_canonical_pose
from if_benchmark.initial_pose.runtime import (
    apply_dynamic_mass_properties,
    apply_world_yaw,
    approved_world_yaw_deg,
    bind_loaded_actor_scale,
    validate_loaded_initial_pose,
)
from if_benchmark.initial_pose.core import InitialPoseContractError
from if_benchmark.schema import artifact_hash
from if_benchmark.pair_validation import require_active_relation_selection
from if_benchmark.success_contract import (
    SuccessContract,
    build_success_snapshot,
    evaluate_m5_terminal_success,
    support_relation_metrics,
)


TABLE_TOP_Z = 0.741
SUCCESS_CONTRACT = SuccessContract.load()
_STABILITY_CONTRACT = SUCCESS_CONTRACT.section("stability")
_RELATION_CONTRACT = SUCCESS_CONTRACT.section("relation")
SETTLE_MIN_STEPS = int(_STABILITY_CONTRACT["minimum_settle_steps"])
SETTLE_MAX_STEPS = int(_STABILITY_CONTRACT["maximum_settle_steps"])
STABILITY_WINDOW_STEPS = int(_STABILITY_CONTRACT["observation_window_steps"])
STABLE_TRANSLATION_EPS_M = float(_STABILITY_CONTRACT["translation_drift_m"])
STABLE_ROTATION_EPS_DEG = float(_STABILITY_CONTRACT["rotation_drift_deg"])
CONTACT_IMPULSE_EPS = float(
    SUCCESS_CONTRACT.section("active_contact")["impulse_epsilon"]
)
RELATION_DEPTH_SPLIT_M = float(_RELATION_CONTRACT["containment"]["depth_split_m"])
RELATION_DEPTH_TOLERANCE_M = float(_RELATION_CONTRACT["containment"]["depth_tolerance_m"])
RELATION_CENTER_RIM_TOLERANCE_M = float(_RELATION_CONTRACT["containment"]["center_rim_tolerance_m"])
LOW_CLEARANCE_RELATIONS = {"place_on", "stack", "stamp_on", "hang_on"}
HIGH_CLEARANCE_RELATIONS = {"place_in", "dump_into"}
LOW_CLEARANCE_LADDER_M = (0.004, 0.008, 0.015)
HIGH_CLEARANCE_LADDER_M = (0.015, 0.03, 0.05)

SUBJECT_CANDIDATE_USE = "pick_subject"
RECEIVER_CANDIDATE_BY_RELATION = {
    "place_in": "place_in_receiver",
    "place_on": "place_on_receiver",
    "stack": "stack_base",
}

PAIR_QPOS_HINTS = {
    ("hang_on", "subject"): [0.707, 0.707, 0.0, 0.0],
    ("hang_on", "receiver"): [-0.22, -0.22, 0.67, 0.67],
    ("dump_into", "subject"): [0.651892, 0.651428, 0.274378, 0.274584],
    ("dump_into", "receiver"): [0.5, 0.5, 0.5, 0.5],
    ("stamp_on", "subject"): [0.5, 0.5, 0.5, 0.5],
    ("stamp_on", "receiver"): [1.0, 0.0, 0.0, 0.0],
}


class _TerminalSettleImports:
    """Process-local immutable imports used by every terminal-settle scene."""

    def __init__(
        self,
        *,
        sapien: Any,
        create_actor: Any,
        create_sphere: Any,
        create_table: Any,
        set_mass_and_scale_inertia: Any,
    ) -> None:
        self.sapien = sapien
        self.create_actor = create_actor
        self.create_sphere = create_sphere
        self.create_table = create_table
        self.set_mass_and_scale_inertia = set_mass_and_scale_inertia


@lru_cache(maxsize=1)
def _terminal_settle_imports() -> _TerminalSettleImports:
    """Import SAPIEN and static loader helpers once in each worker process."""

    import sapien
    from sapien.render import set_global_config

    # ``envs.utils.__init__`` eagerly imports Open3D, HDF5, OpenCV, and every
    # data-generation helper.  M5 needs only the exact actor/primitives source
    # files.  Load those files as a lightweight private package so their
    # relative imports and shared mass/inertia implementation remain intact,
    # without paying ~6.5 s per batch process for unrelated modules.
    create_actor_module = sys.modules.get("envs.utils.create_actor")
    physical_properties_module = sys.modules.get(
        "envs.utils.physical_properties"
    )
    if create_actor_module is None or physical_properties_module is None:
        package_name = "_robotwin_m5_lightweight_env_utils"
        package = sys.modules.get(package_name)
        if package is None:
            package = ModuleType(package_name)
            package.__path__ = [
                str(Path(__file__).resolve().parents[1] / "envs" / "utils")
            ]
            sys.modules[package_name] = package
        create_actor_module = importlib.import_module(
            f"{package_name}.create_actor"
        )
        physical_properties_module = importlib.import_module(
            f"{package_name}.physical_properties"
        )

    try:
        set_global_config(max_num_materials=50000, max_num_textures=50000)
    except Exception:
        pass
    return _TerminalSettleImports(
        sapien=sapien,
        create_actor=create_actor_module.create_actor,
        create_sphere=create_actor_module.create_sphere,
        create_table=create_actor_module.create_table,
        set_mass_and_scale_inertia=(
            physical_properties_module.set_mass_and_scale_inertia
        ),
    )


class _TerminalSettleResources:
    """Process-static imports and optional render resources shared by a batch.

    Actors, builders, and physics scenes deliberately do not live here.  They
    belong to exactly one :class:`_TerminalSettleScene` attempt.
    """

    def __init__(self, *, enable_rendering: bool = False) -> None:
        self.imports = _terminal_settle_imports()
        self.enable_rendering = bool(enable_rendering)
        self.engine = None
        self.renderer = None
        if enable_rendering:
            # Keep the established diagnostic-render path. Canonical M5 never
            # constructs these Vulkan resources.
            self.engine = self.imports.sapien.Engine()
            self.renderer = self.imports.sapien.SapienRenderer()
            self.engine.set_renderer(self.renderer)

    def create_scene(self, scene_config: Any) -> Any:
        sapien = self.imports.sapien
        if not self.enable_rendering:
            # SAPIEN 3's deprecated Engine.create_scene() silently creates a
            # RenderSystem. Build an explicit CPU-only system list so the
            # canonical terminal-settle gate is genuinely headless.
            physx = getattr(sapien, "physx", None)
            scene_type = getattr(sapien, "Scene", None)
            cpu_system_type = getattr(physx, "PhysxCpuSystem", None)
            set_scene_config = getattr(physx, "set_scene_config", None)
            if (
                scene_type is not None
                and cpu_system_type is not None
                and set_scene_config is not None
            ):
                set_scene_config(scene_config)
                return scene_type([cpu_system_type()])

        # Diagnostic rendering and older SAPIEN releases retain the
        # compatible Engine path.
        if self.engine is None:
            self.engine = sapien.Engine()
        return self.engine.create_scene(scene_config)

    def close(self) -> None:
        self.renderer = None
        self.engine = None


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    return _run_attempt(args)


def run_batch(args: argparse.Namespace) -> list[dict[str, Any]]:
    batch = _load_json_required(Path(args.batch_file))
    attempts = batch.get("attempts") if isinstance(batch, dict) else batch
    if not isinstance(attempts, list):
        raise ValueError(f"expected attempts list in {args.batch_file}")

    catalog = _load_yaml_required(Path(args.asset_catalog))
    reusable_runtime = PairTerminalSettleRuntime(catalog) if args.reuse_task_env else None
    results: list[dict[str, Any]] = []
    output_path = Path(args.batch_output) if args.batch_output else None
    output_stream = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_stream = output_path.open("w", encoding="utf-8")

    try:
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                raise ValueError(f"expected attempt mapping at batch index {index}")
            attempt_args = argparse.Namespace(
                asset_catalog=args.asset_catalog,
                pair_key=attempt.get("pair_key"),
                relation=attempt.get("relation"),
                seed=int(attempt.get("seed")),
                attempt_id=attempt.get("attempt_id"),
                job_id=attempt.get("job_id"),
                job_index=attempt.get("job_index"),
                warmup=bool(attempt.get("warmup", False)),
                probe_job=attempt.get("probe_job"),
                probe_job_file=None,
            )
            result = _run_attempt(
                attempt_args,
                catalog=catalog,
                reusable_runtime=reusable_runtime,
            )
            results.append(result)
            if output_stream is not None:
                json.dump(
                    _jsonable(result),
                    output_stream,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                output_stream.write("\n")
                # Keep every completed attempt crash-visible without paying
                # one open/close cycle per JSONL row.
                output_stream.flush()
    finally:
        if output_stream is not None:
            output_stream.close()
        if reusable_runtime is not None:
            reusable_runtime.close()
    return results


def _run_attempt(
    args: argparse.Namespace,
    *,
    catalog: dict[str, Any] | None = None,
    reusable_runtime: "PairTerminalSettleRuntime | None" = None,
) -> dict[str, Any]:
    timings: dict[str, Any] = {}
    total_started = time.monotonic()
    try:
        started = time.monotonic()
        loaded_catalog = catalog if catalog is not None else _load_yaml_required(Path(args.asset_catalog))
        probe_job = _build_probe_job(loaded_catalog, args)
        _record_timing(timings, "build_probe_job_sec", started)
    except Exception as exc:
        return _failure(args, "missing_pair_metadata", exc, timings=timings, total_started=total_started)

    runtime = reusable_runtime or PairTerminalSettleRuntime(loaded_catalog)
    timings["task_env_reused"] = reusable_runtime is not None
    try:
        _validate_probe_job_initial_state_bindings(probe_job)
        started = time.monotonic()
        result = runtime.run(probe_job, seed=int(args.seed), warmup=bool(getattr(args, "warmup", False)))
        _record_timing(timings, "terminal_settle_sec", started)
        result.update(
            {
                "attempt_id": getattr(args, "attempt_id", None),
                "job_id": getattr(args, "job_id", None),
                "job_index": getattr(args, "job_index", None),
                "pair_key": probe_job["pair_key"],
                "relation": probe_job["relation"],
                "subject": _endpoint_identity(probe_job["subject"]),
                "receiver": _endpoint_identity(probe_job["receiver"]),
                "seed": int(args.seed),
                "warmup": bool(getattr(args, "warmup", False)),
            }
        )
        checker_metrics = result.get("metrics") or {}
        for key in (
            "success_contract_version",
            "success_contract_hash",
            "relation_contract_hash",
            "success_snapshot_hash",
            "predicate",
            "status",
            "failed_predicates",
        ):
            if key in checker_metrics:
                result[key] = checker_metrics[key]
        _record_total(timings, total_started)
        result["timings"] = {**result.get("timings", {}), **timings}
        result["duration_sec"] = result["timings"]["total_worker_sec"]
        return result
    except Exception as exc:
        reason = (
            "source_initial_pose_mismatch"
            if isinstance(exc, InitialPoseContractError)
            or "source_initial_pose" in str(exc)
            else "worker_exception"
        )
        return _failure(args, reason, exc, timings=timings, total_started=total_started)
    finally:
        if reusable_runtime is None:
            runtime.close()


def _validate_probe_job_initial_state_bindings(probe_job: dict[str, Any]) -> None:
    """Reject incomplete or cross-wired R01 bindings before SAPIEN starts."""

    bindings = probe_job.get("canonical_initial_states")
    if not isinstance(bindings, dict):
        raise RuntimeError("source_initial_pose_binding_missing:canonical_initial_states")
    for endpoint_name in ("subject", "receiver"):
        endpoint = probe_job.get(endpoint_name)
        canonical = bindings.get(endpoint_name)
        if not isinstance(endpoint, dict) or not isinstance(canonical, dict):
            raise RuntimeError(f"source_initial_pose_binding_missing:{endpoint_name}")
        endpoint_asset = str(endpoint.get("asset"))
        endpoint_model = str(endpoint.get("model_id"))
        canonical_asset = str(canonical.get("asset"))
        canonical_model = str(canonical.get("model_id"))
        if endpoint_asset != canonical_asset or endpoint_model != canonical_model:
            raise RuntimeError(
                "source_initial_pose_binding_mismatch:"
                f"{endpoint_name}:{endpoint_asset}:{endpoint_model}!="
                f"{canonical_asset}:{canonical_model}"
            )


class PairTerminalSettleRuntime:
    def __init__(self, catalog: dict[str, Any]) -> None:
        self.catalog = catalog
        self._resources: _TerminalSettleResources | None = None

    def _attempt_resources(self) -> _TerminalSettleResources:
        if self._resources is None:
            self._resources = _TerminalSettleResources(enable_rendering=False)
        return self._resources

    def close(self) -> None:
        if self._resources is not None:
            self._resources.close()
            self._resources = None

    def run(self, probe_job: dict[str, Any], *, seed: int, warmup: bool = False) -> dict[str, Any]:
        relation = str(probe_job.get("relation"))
        recipe = probe_job.get("placement_recipe") or {}
        clearances = (float(recipe.get("clearance_m")),) if recipe.get("clearance_m") is not None else (clearance_ladder(relation)[1],)
        attempts: list[dict[str, Any]] = []
        for clearance in clearances:
            # Reuse only process-level imports/loader resources. A fresh Scene
            # (and therefore fresh PhysX system, actors, and state) is
            # mandatory for every physical attempt.
            scene = _TerminalSettleScene(
                self.catalog,
                probe_job,
                seed=seed,
                clearance_m=clearance,
                resources=self._attempt_resources(),
            )
            try:
                result = scene.run()
            finally:
                scene.close()
            attempts.append(result)
            if result.get("success"):
                return {
                    "plan_success": True,
                    "success": True,
                    "failure_reason": "ok",
                    "metrics": {**result.get("metrics", {}), "clearance_attempts": attempts},
                    "info": {
                        "terminal_settle": result.get("info", {}),
                        "recipe_id": probe_job.get("recipe_id"),
                        "yaw_witness": probe_job.get("yaw_witness") or {},
                    },
                }
            if result.get("failure_reason") in {"actor_load_failed", "receiver_unstable", "unsupported_relation"}:
                break
        best = attempts[-1] if attempts else {"failure_reason": "checker_false", "metrics": {}}
        return {
            "plan_success": True,
            "success": False,
            "failure_reason": best.get("failure_reason") or "checker_false",
            "metrics": {**best.get("metrics", {}), "clearance_attempts": attempts},
            "info": {
                "terminal_settle": best.get("info", {}),
                "recipe_id": probe_job.get("recipe_id"),
                "yaw_witness": probe_job.get("yaw_witness") or {},
            },
        }


def render_terminal_settle_job(
    *,
    asset_catalog_path: Path,
    plan_path: Path,
    pair_key: str,
    seed: int,
    output_dir: Path,
    clearance_m: float | None = None,
    band_id: str | None = None,
) -> dict[str, Any]:
    """Render four diagnostic phases for one exact terminal-settle job."""

    require_active_relation_selection(pair_keys={pair_key})
    catalog = _load_yaml_required(asset_catalog_path)
    plan = _load_yaml_required(plan_path)
    job = next(
        (
            row
            for row in plan.get("jobs") or []
            if str(row.get("pair_key")) == pair_key
            and (band_id is None or str((row.get("yaw_witness") or {}).get("band_id")) == str(band_id))
        ),
        None,
    )
    if job is None:
        raise ValueError(f"pair key not found in terminal-settle plan: {pair_key}")
    probe_job = _build_probe_job(
        catalog,
        argparse.Namespace(probe_job={
            "scope": job.get("scope", "pair"),
            "stage": job.get("stage", "terminal_settle"),
            "relation": job.get("relation"),
            "subject": job.get("subject") or {},
            "receiver": job.get("receiver") or {},
            "pair_key": job.get("pair_key"),
            "probe_spec": job.get("probe_spec") or {},
            "geometry_metrics": job.get("geometry_metrics") or {},
            "rubric_review": job.get("rubric_review") or {},
            "source_candidate_id": job.get("source_candidate_id"),
            "yaw_witness": job.get("yaw_witness") or {},
            "placement_recipe": job.get("placement_recipe") or {},
            "recipe_id": job.get("recipe_id"),
        }, probe_job_file=None),
    )
    recipe_clearance = (job.get("placement_recipe") or {}).get("clearance_m")
    clearance = (
        float(clearance_m)
        if clearance_m is not None
        else float(recipe_clearance)
        if recipe_clearance is not None
        else float(clearance_ladder(str(job.get("relation")))[1])
    )
    resolved_band_id = str((job.get("yaw_witness") or {}).get("band_id") or "fixed")
    attempt_dir = output_dir / _safe_path_component(pair_key) / _safe_path_component(resolved_band_id) / f"seed_{int(seed)}_clearance_{clearance:.3f}"
    scene = _TerminalSettleScene(
        catalog,
        probe_job,
        seed=int(seed),
        clearance_m=clearance,
        render_dir=attempt_dir,
    )
    try:
        result = scene.run()
        frames = list(scene.rendered_frames)
    finally:
        scene.close()
    manifest = {
        "schema_version": 1,
        "kind": "terminal_settle_render",
        "pair_key": pair_key,
        "relation": job.get("relation"),
        "seed": int(seed),
        "clearance_m": clearance,
        "band_id": resolved_band_id,
        "yaw_witness": job.get("yaw_witness") or {},
        "placement_recipe": job.get("placement_recipe") or {},
        "recipe_id": job.get("recipe_id"),
        "source_probe_plan_path": _path_string(plan_path),
        "source_probe_plan_hash": plan.get("probe_plan_hash"),
        "frames": frames,
        "result": result,
    }
    manifest_path = attempt_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": _path_string(manifest_path)}


class _TerminalSettleScene:
    def __init__(
        self,
        catalog: dict[str, Any],
        probe_job: dict[str, Any],
        *,
        seed: int,
        clearance_m: float,
        render_dir: Path | None = None,
        resources: _TerminalSettleResources | None = None,
    ) -> None:
        self.catalog = catalog
        self.probe_job = probe_job
        self.seed = int(seed)
        self.clearance_m = float(clearance_m)
        self.relation = str(probe_job.get("relation"))
        self.engine = None
        self.renderer = None
        self.scene = None
        self.table = None
        self.subject = None
        self.receiver = None
        self.proxy_spheres: list[Any] = []
        self.render_dir = render_dir
        self.debug_camera = None
        self.rendered_frames: list[str] = []
        self._resources = resources
        self._owns_resources = resources is None

    def run(self) -> dict[str, Any]:
        np.random.seed(self.seed)
        self._setup_scene()
        receiver_is_static = self._receiver_is_static()
        self.receiver = self._load_endpoint(
            "receiver", is_static=receiver_is_static
        )
        if self.receiver is None:
            return _attempt_result(False, "actor_load_failed", {"actor": "receiver"})
        self.receiver.set_name(f"receiver:{self.probe_job['receiver']['asset']}")
        _set_actor_mass(
            self.receiver,
            0.5,
            is_static=receiver_is_static,
        )
        self._place_actor_on_table(
            self.receiver,
            endpoint_name="receiver",
            xy=self._receiver_xy(),
            qpos=self._approved_initial_qpos("receiver"),
        )
        self._validate_receiver_initial_pose()
        self._capture("01_receiver_pre_settle")
        receiver_stable = self._settle_and_measure([self.receiver])
        self._capture("02_receiver_settled")
        if not receiver_stable["stable"]:
            return _attempt_result(False, "receiver_unstable", {"receiver_stability": receiver_stable})

        if self.relation == "dump_into":
            return self._run_dump_into(receiver_stable)

        self.subject = self._load_endpoint("subject", is_static=False)
        if self.subject is None:
            return _attempt_result(False, "actor_load_failed", {"actor": "subject"})
        self.subject.set_name(f"subject:{self.probe_job['subject']['asset']}")
        _set_actor_mass(self.subject, 0.01)

        self._place_subject_terminal()
        self._capture("03_terminal_pre_settle")
        terminal_stable = self._settle_and_measure([self.subject, self.receiver])
        self._capture("04_terminal_settled")
        metrics = self._relation_metrics(receiver_stable, terminal_stable)
        success = bool(metrics.get("success"))
        reason = "ok" if success else str(metrics.get("failure_reason") or "checker_false")
        return _attempt_result(success, reason, metrics, info={"clearance_m": self.clearance_m})

    def close(self) -> None:
        owned_resources = self._resources if self._owns_resources else None
        self.proxy_spheres = []
        self.subject = None
        self.receiver = None
        self.table = None
        self.debug_camera = None
        self.scene = None
        self.renderer = None
        self.engine = None
        self._resources = None
        self._owns_resources = False
        if owned_resources is not None:
            owned_resources.close()

    def _setup_scene(self) -> None:
        if self._resources is None:
            self._resources = _TerminalSettleResources(
                enable_rendering=self.render_dir is not None
            )
            self._owns_resources = True
        resources = self._resources
        sapien = resources.imports.sapien
        self.engine = resources.engine
        self.renderer = resources.renderer
        # Canonical M5 probes are physics-only.  Creating a Vulkan renderer for
        # every attempt serializes otherwise independent CPU workers on one GPU
        # and dominates the full sweep.  Keep rendering only for the explicit
        # diagnostic path that owns a render directory.
        if self.render_dir is not None and self.renderer is None:
            raise RuntimeError("terminal_settle_render_resources_missing")
        scene_config = sapien.SceneConfig()
        self.scene = resources.create_scene(scene_config)
        self.scene.set_timestep(1 / 250)
        try:
            self.scene.add_ground(0, render=self.render_dir is not None)
        except TypeError:
            # SAPIEN 2 does not expose the render switch.
            self.scene.add_ground(0)
        self.scene.default_physical_material = self.scene.create_physical_material(0.5, 0.5, 0)
        if self.renderer is not None:
            self.scene.set_ambient_light([0.5, 0.5, 0.5])
            self.scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5], shadow=False)
        self.table = resources.imports.create_table(
            self.scene,
            sapien.Pose(p=[0, 0, 0.74]),
            length=1.2,
            width=0.7,
            height=0.74,
            thickness=0.05,
            is_static=True,
            name="table",
            load_visual=self.render_dir is not None,
        )
        if self.render_dir is not None:
            self.render_dir.mkdir(parents=True, exist_ok=True)
            self.debug_camera = self.scene.add_camera(
                name="pair_terminal_debug_camera",
                width=640,
                height=480,
                fovy=np.deg2rad(70),
                near=0.05,
                far=10.0,
            )
            camera_position = np.array([0.0, 0.38, 1.35], dtype=float)
            camera_forward = np.array([0.0, -0.55, -0.62], dtype=float)
            camera_forward /= np.linalg.norm(camera_forward)
            camera_left = np.array([1.0, 0.0, 0.0], dtype=float)
            camera_up = np.cross(camera_forward, camera_left)
            camera_matrix = np.eye(4)
            camera_matrix[:3, :3] = np.stack([camera_forward, camera_left, camera_up], axis=1)
            camera_matrix[:3, 3] = camera_position
            self.debug_camera.entity.set_pose(sapien.Pose(camera_matrix))

    def _capture(self, label: str) -> None:
        if self.debug_camera is None or self.render_dir is None:
            return
        from PIL import Image

        self.scene.update_render()
        self.debug_camera.take_picture()
        rgba = self.debug_camera.get_picture("Color")
        rgb = (np.asarray(rgba)[..., :3] * 255).clip(0, 255).astype(np.uint8)
        path = self.render_dir / f"{label}.png"
        Image.fromarray(rgb).save(path)
        self.rendered_frames.append(_path_string(path))

    def _load_endpoint(self, endpoint_name: str, *, is_static: bool) -> Any:
        if self._resources is None:
            raise RuntimeError("terminal_settle_resources_missing")
        sapien = self._resources.imports.sapien
        create_actor = self._resources.imports.create_actor

        endpoint = self.probe_job[endpoint_name]
        canonical = (self.probe_job.get("canonical_initial_states") or {}).get(
            endpoint_name
        )
        if not isinstance(canonical, dict):
            raise RuntimeError(
                f"source_initial_pose_binding_missing:{endpoint_name}"
            )
        actor = create_actor(
            self.scene,
            pose=sapien.Pose([0, 0, TABLE_TOP_Z + 0.2], self._endpoint_qpos(endpoint_name)),
            modelname=str(endpoint["asset"]),
            scale=canonical["scale_xyz"],
            model_id=_model_id(endpoint.get("model_id")),
            convex=True,
            is_static=is_static,
            mass=None,
            # Canonical M5 is physics-only and must not pay to parse visual
            # meshes.  The explicit diagnostic renderer shares this loader,
            # so keep visuals there or the reviewed actors would disappear.
            load_visual=self.render_dir is not None,
        )
        if actor is not None:
            bind_loaded_actor_scale(actor, canonical["scale_xyz"])
            if not is_static:
                apply_dynamic_mass_properties(actor, canonical)
        return actor

    def _place_actor_on_table(
        self,
        actor: Any,
        *,
        endpoint_name: str,
        xy: tuple[float, float],
        qpos: list[float],
    ) -> None:
        if self._resources is None:
            raise RuntimeError("terminal_settle_resources_missing")
        sapien = self._resources.imports.sapien

        canonical = (self.probe_job.get("canonical_initial_states") or {}).get(endpoint_name)
        if not isinstance(canonical, dict):
            raise RuntimeError(f"source_initial_pose_binding_missing:{endpoint_name}")
        support = canonical["support"]
        z = float(support["origin_z"]) + TABLE_TOP_Z - float(support["plane_z"])
        actor.actor.set_pose(sapien.Pose([xy[0], xy[1], z], qpos))

    def _validate_receiver_initial_pose(self) -> None:
        canonical = (self.probe_job.get("canonical_initial_states") or {}).get(
            "receiver"
        )
        if not isinstance(canonical, dict):
            raise RuntimeError("source_initial_pose_binding_missing:receiver")
        pose = self.receiver.get_pose()
        scale = (self.receiver.config or {}).get("scale") or [1.0, 1.0, 1.0]
        if isinstance(scale, (int, float)):
            scale = [float(scale)] * 3
        elif len(scale) == 1:
            scale = list(scale) * 3
        runtime_state = dict(canonical)
        runtime_state["support"] = {
            **canonical["support"],
            "origin_z": float(canonical["support"]["origin_z"])
            + TABLE_TOP_Z
            - float(canonical["support"]["plane_z"]),
        }
        validate_loaded_initial_pose(
            canonical_state=runtime_state,
            actual_q_wxyz=np.asarray(pose.q, dtype=float).tolist(),
            actual_joint_qpos=[],
            actual_scale_xyz=list(scale),
            actual_origin_z=float(pose.p[2]),
            expected_world_yaw_deg=approved_world_yaw_deg(canonical),
        )

    def _approved_initial_qpos(self, endpoint_name: str) -> list[float]:
        canonical = (self.probe_job.get("canonical_initial_states") or {}).get(
            endpoint_name
        )
        if not isinstance(canonical, dict):
            raise RuntimeError(
                f"source_initial_pose_binding_missing:{endpoint_name}"
            )
        return apply_world_yaw(
            canonical["q0_wxyz"], approved_world_yaw_deg(canonical)
        )

    def _place_subject_terminal(self) -> None:
        if self._resources is None:
            raise RuntimeError("terminal_settle_resources_missing")
        sapien = self._resources.imports.sapien

        qpos = self._endpoint_qpos("subject")
        self.subject.actor.set_pose(sapien.Pose([0.0, -0.1, TABLE_TOP_Z + 0.2], qpos))
        target = self._subject_target_pose()
        self.subject.actor.set_pose(sapien.Pose([target[0], target[1], TABLE_TOP_Z + 0.2], qpos))
        if self.relation == "hang_on":
            self._place_hang_subject_terminal()
            return
        pose = self.subject.get_pose()
        aabb = actor_aabb(self.subject)
        centered_xy = centered_root_xy(pose.p, aabb["center"], target[:2])
        self.subject.actor.set_pose(sapien.Pose(centered_xy, pose.q))
        if self.relation == "stamp_on":
            self._align_subject_contact_face(target)
        aabb = actor_aabb(self.subject)
        z = nonpenetrating_z(
            current_z=float(self.subject.get_pose().p[2]),
            current_min_z=aabb["min"][2],
            target_top_z=float(target[2]),
            clearance_m=self.clearance_m,
        )
        pose = self.subject.get_pose()
        self.subject.actor.set_pose(sapien.Pose([pose.p[0], pose.p[1], z], pose.q))

    def _place_hang_subject_terminal(self) -> None:
        if self._resources is None:
            raise RuntimeError("terminal_settle_resources_missing")
        sapien = self._resources.imports.sapien

        subject_fp_matrix = _actor_functional_matrix(self.subject, 0)
        receiver_fp_matrix = _actor_functional_matrix(self.receiver, 0)
        if subject_fp_matrix is None or receiver_fp_matrix is None:
            return
        actor_matrix = self.subject.get_pose().to_transformation_matrix()
        root_matrix = aligned_functional_root_pose(
            actor_matrix,
            subject_fp_matrix,
            receiver_fp_matrix,
            offset_along_receiver_axis_m=0.05,
        )
        self.subject.actor.set_pose(sapien.Pose(root_matrix))

    def _align_subject_contact_face(self, target: np.ndarray) -> None:
        if self._resources is None:
            raise RuntimeError("terminal_settle_resources_missing")
        sapien = self._resources.imports.sapien

        points = [_actor_contact_point(self.subject, idx) for idx in (4, 5, 6, 7)]
        points = [point for point in points if point is not None]
        if not points:
            return
        pose = self.subject.get_pose()
        center = np.mean(np.array(points, dtype=float)[:, :3], axis=0)
        delta = np.array([target[0] - center[0], target[1] - center[1], 0.0])
        self.subject.actor.set_pose(sapien.Pose((pose.p + delta).tolist(), pose.q))

    def _run_dump_into(self, receiver_stable: dict[str, Any]) -> dict[str, Any]:
        if self._resources is None:
            raise RuntimeError("terminal_settle_resources_missing")
        sapien = self._resources.imports.sapien
        create_sphere = self._resources.imports.create_sphere
        set_mass_and_scale_inertia = (
            self._resources.imports.set_mass_and_scale_inertia
        )

        target = self._receiver_target()
        receiver_aabb = actor_aabb(self.receiver)
        radius = float(((self.probe_job.get("probe_spec") or {}).get("proxy_contents") or {}).get("radius_m") or 0.008)
        count = int(((self.probe_job.get("probe_spec") or {}).get("proxy_contents") or {}).get("count") or 5)
        # Terminal settle validates that the receiver can contain the proxy
        # contents. The source container, pour pose, and release trajectory
        # belong to the later canonical skill gate and are intentionally not
        # loaded into this terminal-only scene.
        release_center = np.array([target[0], target[1], receiver_aabb["max"][2] + 0.04], dtype=float)
        release_offsets = non_overlapping_proxy_offsets(count=count, radius_m=radius, seed=self.seed)
        release_positions: list[list[float]] = []
        for index, offset in enumerate(release_offsets):
            position = release_center + np.array(offset, dtype=float)
            sphere = create_sphere(
                self.scene,
                pose=sapien.Pose(
                    position.tolist(),
                    [1, 0, 0, 0],
                ),
                radius=radius,
                color=[1, 0, 0],
                name=f"proxy_content:{index}",
                load_visual=self.render_dir is not None,
            )
            _set_entity_dynamic_mass(
                sphere,
                component_type=sapien.physx.PhysxRigidDynamicComponent,
                target_mass=0.0001,
                update=set_mass_and_scale_inertia,
            )
            self.proxy_spheres.append(sphere)
            release_positions.append([_clean_float(value) for value in position.tolist()])
        self._capture("03_proxy_contents_released")
        sphere_stable = self._settle_entities(self.proxy_spheres, rotation_invariant=True)
        self._capture("04_proxy_contents_settled")
        states = [_entity_state(sphere, radius=radius) for sphere in self.proxy_spheres]
        metrics = dump_into_success_metrics(receiver_aabb, states, TABLE_TOP_Z, sphere_stable)
        metrics["proxy_stability"] = sphere_stable
        metrics["proxy_release_layout"] = {
            "shape": "sphere",
            "count": count,
            "radius_m": radius,
            "non_overlapping": True,
            "positions": release_positions,
        }
        metrics["receiver_stability"] = receiver_stable
        metrics["receiver_aabb"] = receiver_aabb
        metrics["release_target"] = [_clean_float(value) for value in np.array(target[:3], dtype=float).tolist()]
        metrics["source_container_in_terminal_scene"] = False
        return _attempt_result(bool(metrics["success"]), "ok" if metrics["success"] else metrics["failure_reason"], metrics, info={"clearance_m": self.clearance_m})

    def _settle_and_measure(self, actors: list[Any]) -> dict[str, Any]:
        entities = [actor.actor for actor in actors]
        return self._settle_entities(entities)

    def _settle_entities(self, entities: list[Any], *, rotation_invariant: bool = False) -> dict[str, Any]:
        for _ in range(SETTLE_MIN_STEPS):
            self.scene.step()
        steps = SETTLE_MIN_STEPS
        last: dict[str, Any] = {"stable": False, "failure_reason": "empty_stability_window"}
        while steps + STABILITY_WINDOW_STEPS <= SETTLE_MAX_STEPS:
            snapshots: list[list[dict[str, Any]]] = []
            for _ in range(STABILITY_WINDOW_STEPS):
                self.scene.step()
                snapshots.append([_entity_pose_state(entity) for entity in entities])
            steps += STABILITY_WINDOW_STEPS
            last = stability_from_snapshots(snapshots, rotation_invariant=rotation_invariant)
            last["settle_steps"] = steps
            if last.get("stable"):
                return last
        return last

    def _relation_metrics(self, receiver_stable: dict[str, Any], terminal_stable: dict[str, Any]) -> dict[str, Any]:
        subject_state = actor_state(self.subject)
        receiver_state = actor_state(self.receiver)
        contact_diagnostics = {
            "subject_receiver": contact_metrics(self.scene, self.subject.actor, self.receiver.actor),
            "subject_table": contact_metrics(self.scene, self.subject.actor, self.table),
            "receiver_table": contact_metrics(self.scene, self.receiver.actor, self.table),
        }
        contacts = {key: bool(value["active"]) for key, value in contact_diagnostics.items()}
        canonical_relation = {
            "place_in": "in",
            "place_on": "on",
            "stack": "stack",
        }.get(self.relation)
        if canonical_relation is not None:
            active_contacts = []
            for name, endpoints in {
                "subject_receiver": ("subject", "receiver"),
                "subject_table": ("subject", "support:table"),
            }.items():
                row = contact_diagnostics[name]
                if row.get("active"):
                    active_contacts.append(
                        {
                            "entity_a": endpoints[0],
                            "entity_b": endpoints[1],
                            "summed_impulse": row.get("summed_impulse"),
                            "summed_impulse_norm": row.get("summed_impulse_norm"),
                            "points": row.get("points", []),
                        }
                    )
            entity_stability = {}
            for index, name in enumerate(("subject", "receiver")):
                values = terminal_stable.get("per_entity") or []
                entity_stability[name] = (
                    dict(values[index])
                    if index < len(values)
                    else {"stable": bool(terminal_stable.get("stable"))}
                )
            terminal_state = {
                "entities": {
                    "subject": {"pose": {"position": subject_state["p"], "quat_wxyz": subject_state["q"]}, "aabb": subject_state["aabb"]},
                    "receiver": {"pose": {"position": receiver_state["p"], "quat_wxyz": receiver_state["q"]}, "aabb": receiver_state["aabb"]},
                },
                "active_contacts": active_contacts,
                "grippers": {},
            }
            source_hashes = {
                "pair_probe_job": str(
                    self.probe_job.get("job_hash")
                    or self.probe_job.get("config_hash")
                    or artifact_hash(self.probe_job)
                )
            }
            success_snapshot = build_success_snapshot(
                initial=terminal_state,
                terminal=terminal_state,
                stability={
                    "entities": entity_stability,
                    "whole_scene": {"stable": bool(terminal_stable.get("stable"))},
                },
                execution_events={},
                source_hashes=source_hashes,
                contract=SUCCESS_CONTRACT,
            )
            checker = evaluate_m5_terminal_success(
                "relation_holds",
                {
                    "subject": "subject",
                    "target": "receiver",
                    "relation": canonical_relation,
                },
                success_snapshot,
                SUCCESS_CONTRACT,
            )
            metrics = {
                **checker["metrics"],
                "success": checker.get("status") == "passed",
                "status": checker["status"],
                "predicate": checker["predicate"],
                "failed_predicates": checker["failed_predicates"],
                "failure_reason": (
                    "ok"
                    if checker.get("status") == "passed"
                    else checker["failed_predicates"][0]
                ),
                **SUCCESS_CONTRACT.identity,
                "relation_contract_hash": success_snapshot["relation_contract_hash"],
                "success_snapshot_hash": success_snapshot["success_snapshot_hash"],
            }
            metrics.update(
                {
                    "clearance_m": self.clearance_m,
                    "geometry_metrics": self.probe_job.get("geometry_metrics") or {},
                    "rubric_review": self.probe_job.get("rubric_review") or {},
                    "subject_pose": {key: subject_state[key] for key in ("p", "q")},
                    "receiver_pose": {key: receiver_state[key] for key in ("p", "q")},
                    "subject_aabb": subject_state["aabb"],
                    "receiver_aabb": receiver_state["aabb"],
                    "contacts": contacts,
                    "contact_diagnostics": contact_diagnostics,
                    "receiver_stability": receiver_stable,
                    "terminal_stability": terminal_stable,
                }
            )
            return metrics
        probe_spec = self.probe_job.get("probe_spec") or {}
        if self.relation == "stack":
            metrics = stack_success_metrics(subject_state, receiver_state, contacts, terminal_stable)
        elif self.relation == "place_on":
            metrics = place_on_success_metrics(subject_state, receiver_state, contacts, terminal_stable)
        elif self.relation == "place_in":
            metrics = place_in_success_metrics(subject_state, receiver_state, contacts, terminal_stable)
        elif self.relation == "hang_on":
            metrics = hang_on_success_metrics(subject_state, receiver_state, terminal_stable)
        elif self.relation == "stamp_on":
            metrics = stamp_on_success_metrics(subject_state, receiver_state, contacts, terminal_stable, probe_spec)
        else:
            metrics = {"success": False, "failure_reason": "unsupported_relation"}
        metrics.update(
            {
                "clearance_m": self.clearance_m,
                "geometry_metrics": self.probe_job.get("geometry_metrics") or {},
                "rubric_review": self.probe_job.get("rubric_review") or {},
                "subject_pose": {key: subject_state[key] for key in ("p", "q")},
                "receiver_pose": {key: receiver_state[key] for key in ("p", "q")},
                "subject_aabb": subject_state["aabb"],
                "receiver_aabb": receiver_state["aabb"],
                "contacts": contacts,
                "contact_diagnostics": contact_diagnostics,
                "receiver_stability": receiver_stable,
                "terminal_stability": terminal_stable,
            }
        )
        return metrics

    def _subject_target_pose(self) -> np.ndarray:
        target = self._receiver_target()
        if self.relation in {"place_on", "stack", "stamp_on"}:
            receiver_aabb = actor_aabb(self.receiver)
            target[2] = receiver_aabb["max"][2]
        return target

    def _receiver_target(self) -> np.ndarray:
        probe_spec = self.probe_job.get("probe_spec") or {}
        anchor = probe_spec.get("receiver_target") or probe_spec.get("terminal_anchor") or {}
        if self.relation == "hang_on":
            root = np.array(self.receiver.get_pose().p, dtype=float)
            fp = _actor_functional_point(self.receiver, 0)
            if fp is not None:
                return (root + fp[:3]) / 2
            return root
        if anchor.get("type") == "functional_point":
            point = _actor_functional_point(self.receiver, int(anchor.get("id") or 0))
            if point is not None:
                return point[:3].copy()
        if anchor.get("type") == "target_pose":
            point = _actor_target_point(self.receiver, int(anchor.get("id") or 0))
            if point is not None:
                return point[:3].copy()
        aabb = actor_aabb(self.receiver)
        return np.array([(aabb["min"][0] + aabb["max"][0]) / 2, (aabb["min"][1] + aabb["max"][1]) / 2, aabb["max"][2]], dtype=float)

    def _endpoint_qpos(self, endpoint_name: str) -> list[float]:
        candidate = self.probe_job.get(f"{endpoint_name}_candidate") or {}
        canonical = ((self.probe_job.get("probe_spec") or {}).get("canonical_poses") or {}).get(endpoint_name)
        if not isinstance(canonical, dict):
            hint = PAIR_QPOS_HINTS.get((self.relation, endpoint_name))
            if hint is not None:
                return [float(value) for value in hint]
        if not isinstance(canonical, dict):
            canonical = resolve_canonical_pose(
                candidate,
                "pick_subject" if endpoint_name == "subject" else str(candidate.get("use") or "receiver"),
                relation=self.relation,
                role=endpoint_name,
            )
        qpos = list(canonical.get("q0_wxyz") or candidate_pose_defaults(candidate, "pick_left")["qpos"])
        if endpoint_name == "subject" and self.relation in {"place_in", "place_on", "stack"}:
            rotation_deg = float((self.probe_job.get("yaw_witness") or {}).get("witness_delta_deg") or 0.0)
            if abs(rotation_deg) > 1e-6:
                yaw = math.radians(rotation_deg)
                qpos = _quat_multiply_wxyz([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], qpos)
        return [float(value) for value in qpos]

    def _receiver_xy(self) -> tuple[float, float]:
        rng = np.random.default_rng(self.seed)
        if self.relation == "hang_on":
            return (float(0.18 + rng.uniform(-0.01, 0.01)), float(0.15 + rng.uniform(-0.005, 0.005)))
        if self.relation == "dump_into":
            return (float(-0.08 + rng.uniform(-0.01, 0.01)), float(-0.08 + rng.uniform(-0.01, 0.01)))
        return (float(rng.uniform(-0.015, 0.015)), float(-0.1 + rng.uniform(-0.015, 0.015)))

    def _receiver_is_static(self) -> bool:
        return self.relation in {"hang_on", "dump_into", "stamp_on"}


def _build_probe_job(catalog: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    probe_job = getattr(args, "probe_job", None)
    if not isinstance(probe_job, dict):
        probe_job_file = getattr(args, "probe_job_file", None)
        if probe_job_file:
            probe_job = _load_json_required(Path(probe_job_file))
        else:
            raise ValueError("pair terminal settle worker requires probe_job")
    out = dict(probe_job)
    relation = str(out.get("relation"))
    subject = _endpoint_identity(out.get("subject") or {})
    receiver = _endpoint_identity(out.get("receiver") or {})
    pair_candidate = _pair_candidate_by_id(catalog, out.get("source_candidate_id"))
    out["subject"] = _merge_endpoint_from_pair_candidate(subject, pair_candidate, "subject")
    out["receiver"] = _merge_endpoint_from_pair_candidate(receiver, pair_candidate, "receiver")
    out["subject_candidate"] = _candidate_for_endpoint(catalog, out["subject"], SUBJECT_CANDIDATE_USE)
    out["receiver_candidate"] = _candidate_for_endpoint(catalog, out["receiver"], RECEIVER_CANDIDATE_BY_RELATION.get(relation))
    return out


def _candidate_for_endpoint(catalog: dict[str, Any], endpoint: dict[str, Any], preferred_use: str | None) -> dict[str, Any]:
    asset = str(endpoint.get("asset"))
    model_id = str(endpoint.get("model_id"))
    model = ((catalog.get("assets") or {}).get(asset) or {}).get("models", {}).get(model_id) or {}
    candidates = model.get("candidates") or {}
    candidate = candidates.get(preferred_use) if preferred_use else None
    if not isinstance(candidate, dict):
        candidate = candidates.get(SUBJECT_CANDIDATE_USE)
    if not isinstance(candidate, dict) and candidates:
        candidate = next((item for item in candidates.values() if isinstance(item, dict)), {})
    merged = dict(candidate or {})
    merged.setdefault("asset", asset)
    merged.setdefault("model_id", model_id)
    for key in ("category", "inventory_id", "metadata"):
        if endpoint.get(key) is not None:
            merged[key] = endpoint.get(key)
    return merged


def _pair_candidate_by_id(catalog: dict[str, Any], candidate_id: Any) -> dict[str, Any] | None:
    if not candidate_id:
        return None
    for candidate in catalog.get("pair_candidates") or []:
        if str(candidate.get("candidate_id")) == str(candidate_id):
            return candidate
    return None


def _merge_endpoint_from_pair_candidate(endpoint: dict[str, Any], pair_candidate: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if not isinstance(pair_candidate, dict):
        return endpoint
    pair_endpoint = pair_candidate.get(name) or {}
    out = dict(endpoint)
    for key in ("category", "inventory_id", "metadata"):
        if out.get(key) is None and pair_endpoint.get(key) is not None:
            out[key] = pair_endpoint.get(key)
    if not out.get("metadata") and pair_endpoint.get("metadata"):
        out["metadata"] = pair_endpoint.get("metadata")
    return out


def clearance_ladder(relation: str) -> tuple[float, ...]:
    if relation in HIGH_CLEARANCE_RELATIONS:
        return HIGH_CLEARANCE_LADDER_M
    if relation in LOW_CLEARANCE_RELATIONS:
        return LOW_CLEARANCE_LADDER_M
    return LOW_CLEARANCE_LADDER_M


def non_overlapping_proxy_offsets(*, count: int, radius_m: float, seed: int) -> list[list[float]]:
    """Return a compact, deterministic XY release layout without sphere overlap."""

    if count <= 0:
        return []
    spacing = max(2.0 * float(radius_m) + 0.002, 2.1 * float(radius_m))
    columns = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / columns))
    points = np.array(
        [
            [
                (index % columns - (columns - 1) / 2.0) * spacing,
                (index // columns - (rows - 1) / 2.0) * spacing,
            ]
            for index in range(count)
        ],
        dtype=float,
    )
    points -= points.mean(axis=0, keepdims=True)
    angle = float(np.random.default_rng(int(seed)).uniform(0.0, 2.0 * math.pi))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=float,
    )
    points = points @ rotation.T
    return [[_clean_float(point[0]), _clean_float(point[1]), 0.0] for point in points]


def actor_state(actor: Any) -> dict[str, Any]:
    state = _entity_pose_state(actor.actor)
    state["aabb"] = actor_aabb(actor)
    fp0 = _actor_functional_point(actor, 0)
    if fp0 is not None:
        state["functional_point_0"] = fp0.tolist()
    contact_face = _contact_face_center(actor, [4, 5, 6, 7])
    if contact_face is not None:
        state["contact_face_4_5_6_7"] = contact_face.tolist()
    return state


def _entity_state(entity: Any, *, radius: float) -> dict[str, Any]:
    pose_state = _entity_pose_state(entity)
    p = np.array(pose_state["p"], dtype=float)
    pose_state["aabb"] = {
        "min": [float(p[0] - radius), float(p[1] - radius), float(p[2] - radius)],
        "max": [float(p[0] + radius), float(p[1] + radius), float(p[2] + radius)],
        "center": [float(p[0]), float(p[1]), float(p[2])],
    }
    return pose_state


def actor_aabb(actor: Any) -> dict[str, list[float]]:
    return world_aabb_from_config(actor.config or {}, actor.get_pose().to_transformation_matrix())


def world_aabb_from_config(config: dict[str, Any], pose_matrix: Any) -> dict[str, list[float]]:
    extents = _vector3(config.get("extents"), default=[0.1, 0.1, 0.1])
    scale = _scale3(config.get("scale", 1.0))
    half = extents * scale / 2.0
    center = _vector3_any(config.get("center"), default=[0.0, 0.0, 0.0]) * scale
    corners = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    ) * half + center
    matrix = np.array(pose_matrix, dtype=float)
    transformed = matrix[:3, :3] @ corners.T + matrix[:3, 3].reshape(3, 1)
    mins = transformed.min(axis=1)
    maxs = transformed.max(axis=1)
    center = (mins + maxs) / 2.0
    return {
        "min": [_clean_float(value) for value in mins.tolist()],
        "max": [_clean_float(value) for value in maxs.tolist()],
        "center": [_clean_float(value) for value in center.tolist()],
    }


def nonpenetrating_z(*, current_z: float, current_min_z: float, target_top_z: float, clearance_m: float) -> float:
    return _clean_float(float(current_z) + (float(target_top_z) + float(clearance_m) - float(current_min_z)))


def centered_root_xy(root_position: Any, aabb_center: Any, target_xy: Any) -> list[float]:
    root = np.array(root_position, dtype=float).copy()
    center = np.array(aabb_center, dtype=float)
    target = np.array(target_xy, dtype=float)
    root[:2] += target[:2] - center[:2]
    return root.tolist()


def stability_from_snapshots(
    snapshots: list[list[dict[str, Any]]],
    *,
    rotation_invariant: bool = False,
) -> dict[str, Any]:
    if not snapshots:
        return {"stable": False, "failure_reason": "empty_stability_window", "per_entity": []}
    entity_count = len(snapshots[-1])
    max_translation = 0.0
    max_rotation = 0.0
    unstable_indices: set[int] = set()
    per_entity: list[dict[str, Any]] = []
    for index in range(entity_count):
        entity_translation = 0.0
        entity_rotation = 0.0
        final = snapshots[-1][index]
        final_p = np.array(final["p"], dtype=float)
        final_q = np.array(final["q"], dtype=float)
        for frame in snapshots:
            current = frame[index]
            translation = float(np.linalg.norm(np.array(current["p"], dtype=float) - final_p))
            rotation = quat_distance_deg(np.array(current["q"], dtype=float), final_q)
            max_translation = max(max_translation, translation)
            max_rotation = max(max_rotation, rotation)
            entity_translation = max(entity_translation, translation)
            entity_rotation = max(entity_rotation, rotation)
            if translation > STABLE_TRANSLATION_EPS_M or (
                not rotation_invariant and rotation > STABLE_ROTATION_EPS_DEG
            ):
                unstable_indices.add(index)
        per_entity.append(
            {
                "index": index,
                "stable": index not in unstable_indices,
                "max_translation_m": _clean_float(entity_translation),
                "max_rotation_deg": _clean_float(entity_rotation),
            }
        )
    stable = not unstable_indices
    return {
        "stable": stable,
        "rotation_invariant": bool(rotation_invariant),
        "unstable_indices": sorted(unstable_indices),
        "max_translation_m": _clean_float(max_translation),
        "max_rotation_deg": _clean_float(max_rotation),
        "per_entity": per_entity,
    }


def stack_success_metrics(subject: dict[str, Any], receiver: dict[str, Any], contacts: dict[str, bool], stability: dict[str, Any]) -> dict[str, Any]:
    return _shared_support_metrics("stack", subject, receiver, contacts, stability)


def place_on_success_metrics(subject: dict[str, Any], receiver: dict[str, Any], contacts: dict[str, bool], stability: dict[str, Any]) -> dict[str, Any]:
    return _shared_support_metrics("on", subject, receiver, contacts, stability)


def _shared_support_metrics(
    relation: str,
    subject: dict[str, Any],
    receiver: dict[str, Any],
    contacts: dict[str, bool],
    stability: dict[str, Any],
) -> dict[str, Any]:
    metrics = support_relation_metrics(
        relation,
        subject["aabb"],
        receiver["aabb"],
        subject_receiver_contact=bool(contacts.get("subject_receiver")),
        subject_table_contact=bool(contacts.get("subject_table")),
        stable=bool(stability.get("stable")),
        contract=SUCCESS_CONTRACT,
    )
    checks = dict(metrics["checks"])
    checks["stable"] = checks["relation_entities_stable"]
    metrics["checks"] = checks
    metrics["failed_checks"] = list(metrics["failed_predicates"])
    metrics["failure_reason"] = (
        "ok"
        if metrics["success"]
        else _semantic_failure_reason(metrics["failed_checks"])
    )
    metrics["relation"] = "place_on" if relation == "on" else relation
    if relation in {"on", "stack"}:
        metrics.update(
            {
                "xy_distance": metrics["xy_distance_m"],
                "xy_limit": metrics["xy_limit_m"],
                "bottom_top_gap": metrics["bottom_top_gap_m"],
                "subject_center_top_delta": (
                    float(subject["aabb"]["center"][2])
                    - float(receiver["aabb"]["max"][2])
                ),
            }
        )
    else:
        center_delta = metrics["subject_center_rim_delta_m"]
        depth = metrics["containment_depth_m"]
        metrics.update(
            {
                "xy_inside_margin": metrics["xy_inside_margin_m"],
                "subject_center_rim_delta": center_delta,
                "subject_bottom_rim_delta": -depth,
                "inside_by_center_rim": center_delta
                <= RELATION_CENTER_RIM_TOLERANCE_M,
                "inside_by_containment_depth": depth
                >= RELATION_DEPTH_SPLIT_M - RELATION_DEPTH_TOLERANCE_M,
            }
        )
    metrics.update(
        {
            "relation_depth_split_m": RELATION_DEPTH_SPLIT_M,
            "relation_depth_tolerance_m": RELATION_DEPTH_TOLERANCE_M,
            "relation_depth_boundary_m": RELATION_DEPTH_SPLIT_M
            - RELATION_DEPTH_TOLERANCE_M,
            "center_rim_tolerance_m": RELATION_CENTER_RIM_TOLERANCE_M,
        }
    )
    return metrics


def place_in_success_metrics(subject: dict[str, Any], receiver: dict[str, Any], contacts: dict[str, bool], stability: dict[str, Any]) -> dict[str, Any]:
    return _shared_support_metrics("in", subject, receiver, contacts, stability)


def hang_on_success_metrics(subject: dict[str, Any], receiver: dict[str, Any], stability: dict[str, Any]) -> dict[str, Any]:
    mug_fp = np.array(subject.get("functional_point_0") or subject["p"], dtype=float)[:3]
    rack_root = np.array(receiver["p"], dtype=float)
    rack_fp = np.array(receiver.get("functional_point_0") or receiver["p"], dtype=float)[:3]
    midpoint = (rack_root + rack_fp) / 2.0
    xy_error = np.abs(mug_fp[:2] - midpoint[:2])
    checks = {
        "stable": bool(stability.get("stable")),
        "xy_midpoint_aligned": bool(np.all(xy_error < 0.02)),
        "mug_functional_point_high": bool(mug_fp[2] > 0.86),
    }
    failed_checks = _failed_checks(checks)
    return {
        "success": all(checks.values()),
        "failure_reason": "ok" if all(checks.values()) else _semantic_failure_reason(failed_checks),
        "failed_checks": failed_checks,
        "relation": "hang_on",
        "checks": checks,
        "xy_error": [_clean_float(value) for value in xy_error.tolist()],
        "mug_functional_point_z": _clean_float(float(mug_fp[2])),
        "rack_midpoint": [_clean_float(value) for value in midpoint.tolist()],
    }


def stamp_on_success_metrics(
    subject: dict[str, Any],
    receiver: dict[str, Any],
    contacts: dict[str, bool],
    stability: dict[str, Any],
    probe_spec: dict[str, Any],
) -> dict[str, Any]:
    face = np.array(subject.get("contact_face_4_5_6_7") or subject["aabb"]["center"], dtype=float)[:3]
    target = np.array(receiver.get("functional_point_0") or receiver["aabb"]["center"], dtype=float)[:3]
    xy_error = np.abs(face[:2] - target[:2])
    checks = {
        "stable": bool(stability.get("stable")),
        "subject_receiver_contact": bool(contacts.get("subject_receiver")),
        "stamp_contact_ids": list(probe_spec.get("stamp_contact_ids") or []) == [4, 5, 6, 7],
        "xy_aligned": bool(np.all(xy_error < 0.015)),
    }
    failed_checks = _failed_checks(checks)
    return {
        "success": all(checks.values()),
        "failure_reason": "ok" if all(checks.values()) else _semantic_failure_reason(failed_checks),
        "failed_checks": failed_checks,
        "relation": "stamp_on",
        "checks": checks,
        "xy_error": [_clean_float(value) for value in xy_error.tolist()],
    }


def dump_into_success_metrics(receiver_aabb: dict[str, Any], sphere_states: list[dict[str, Any]], table_top_z: float, stability: dict[str, Any]) -> dict[str, Any]:
    receiver_min = np.array(receiver_aabb["min"], dtype=float)
    receiver_max = np.array(receiver_aabb["max"], dtype=float)
    sphere_checks = []
    for state in sphere_states:
        center = np.array(state["aabb"]["center"], dtype=float)
        inside_xy = bool(np.all(center[:2] >= receiver_min[:2] - 0.01) and np.all(center[:2] <= receiver_max[:2] + 0.01))
        above_table = bool(center[2] > table_top_z + 0.005)
        sphere_checks.append({"inside_xy": inside_xy, "above_table": above_table, "center": [_clean_float(value) for value in center.tolist()]})
    checks = {
        "all_proxy_contents_inside_receiver_xy": all(item["inside_xy"] for item in sphere_checks),
        "all_proxy_contents_above_table": all(item["above_table"] for item in sphere_checks),
    }
    failed_checks = _failed_checks(checks)
    return {
        "success": all(checks.values()),
        "failure_reason": "ok" if all(checks.values()) else _semantic_failure_reason(failed_checks),
        "failed_checks": failed_checks,
        "relation": "dump_into",
        "checks": checks,
        "proxy_contents": sphere_checks,
        "proxy_motion_diagnostic": {
            "within_generic_stability_limit": bool(stability.get("stable")),
            "max_translation_m": stability.get("max_translation_m"),
        },
    }


def has_contact(scene: Any, entity_a: Any, entity_b: Any) -> bool:
    return bool(contact_metrics(scene, entity_a, entity_b)["active"])


def contact_metrics(scene: Any, entity_a: Any, entity_b: Any) -> dict[str, Any]:
    pair_count = 0
    point_count = 0
    max_impulse = 0.0
    summed_impulse = np.zeros(3, dtype=float)
    min_separation: float | None = None
    points: list[dict[str, Any]] = []
    for contact in scene.get_contacts():
        first = contact.bodies[0].entity
        second = contact.bodies[1].entity
        if (first == entity_a and second == entity_b) or (first == entity_b and second == entity_a):
            pair_count += 1
            for point in contact.points:
                point_count += 1
                impulse = np.asarray(getattr(point, "impulse", [0.0, 0.0, 0.0]), dtype=float)
                max_impulse = max(max_impulse, float(np.linalg.norm(impulse)))
                summed_impulse += impulse
                separation = getattr(point, "separation", None)
                if separation is not None:
                    value = float(separation)
                    min_separation = value if min_separation is None else min(min_separation, value)
                points.append(
                    {
                        "position": np.asarray(
                            getattr(point, "position", [0.0, 0.0, 0.0]),
                            dtype=float,
                        ).tolist(),
                        "normal": np.asarray(
                            getattr(point, "normal", [0.0, 0.0, 0.0]),
                            dtype=float,
                        ).tolist(),
                        "impulse": impulse.tolist(),
                        "separation": float(separation or 0.0),
                    }
                )
    summed_impulse_norm = float(np.linalg.norm(summed_impulse))
    return {
        "active": bool(summed_impulse_norm > CONTACT_IMPULSE_EPS),
        "pair_count": pair_count,
        "point_count": point_count,
        "max_impulse_norm": _clean_float(max_impulse),
        "summed_impulse": [_clean_float(value) for value in summed_impulse],
        "summed_impulse_norm": _clean_float(summed_impulse_norm),
        "min_separation_m": _clean_float(min_separation) if min_separation is not None else None,
        "impulse_epsilon": CONTACT_IMPULSE_EPS,
        "points": points,
    }


def quat_distance_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    dot = min(1.0, max(-1.0, abs(float(np.dot(q1, q2)))))
    return float(2.0 * math.degrees(math.acos(dot)))


def _actor_functional_point(actor: Any, idx: int) -> np.ndarray | None:
    try:
        point = actor.get_functional_point(idx)
    except Exception:
        return None
    if point is None:
        return None
    return np.array(point[:3], dtype=float)


def _actor_functional_matrix(actor: Any, idx: int) -> np.ndarray | None:
    try:
        matrix = actor.get_functional_point(idx, "matrix")
    except Exception:
        return None
    if matrix is None:
        return None
    return np.array(matrix, dtype=float)


def aligned_functional_root_pose(
    actor_matrix: Any,
    subject_functional_matrix: Any,
    receiver_functional_matrix: Any,
    *,
    offset_along_receiver_axis_m: float,
) -> np.ndarray:
    """Align subject FP frame to receiver FP frame with canonical hang offset."""

    actor_matrix = np.array(actor_matrix, dtype=float)
    subject_functional_matrix = np.array(subject_functional_matrix, dtype=float)
    desired_functional_matrix = np.array(receiver_functional_matrix, dtype=float).copy()
    desired_functional_matrix[:3, 3] += (
        desired_functional_matrix[:3, 2] * float(offset_along_receiver_axis_m)
    )
    local_functional_matrix = np.linalg.inv(actor_matrix) @ subject_functional_matrix
    return desired_functional_matrix @ np.linalg.inv(local_functional_matrix)


def _actor_target_point(actor: Any, idx: int) -> np.ndarray | None:
    try:
        point = actor.get_target_point(idx)
    except Exception:
        return None
    if point is None:
        return None
    return np.array(point[:3], dtype=float)


def _actor_contact_point(actor: Any, idx: int) -> np.ndarray | None:
    try:
        point = actor.get_contact_point(idx)
    except Exception:
        return None
    if point is None:
        return None
    return np.array(point[:3], dtype=float)


def _contact_face_center(actor: Any, contact_ids: list[int]) -> np.ndarray | None:
    points = [_actor_contact_point(actor, idx) for idx in contact_ids]
    points = [point for point in points if point is not None]
    if not points:
        return None
    return np.mean(np.array(points, dtype=float), axis=0)


def _entity_pose_state(entity: Any) -> dict[str, Any]:
    pose = entity.get_pose()
    return {
        "p": [_clean_float(value) for value in np.array(pose.p, dtype=float).tolist()],
        "q": [_clean_float(value) for value in np.array(pose.q, dtype=float).tolist()],
    }


def _set_actor_mass(
    actor: Any,
    mass: float,
    *,
    is_static: bool = False,
) -> None:
    """Apply an explicit coupled mass target to a declared dynamic actor.

    M5 uses this after loading the R01 state to impose its terminal-validation
    dynamics: 0.5 kg for a dynamic receiver and 0.01 kg for the subject.  A
    receiver declared static is immobile by construction and has no dynamic
    mass component to update.
    """

    dynamic_count, static_count = _actor_mass_body_counts(actor)
    if is_static:
        if dynamic_count != 0 or static_count <= 0:
            raise RuntimeError("declared_static_actor_body_mismatch")
        return
    if dynamic_count <= 0:
        raise RuntimeError("dynamic_actor_mass_component_missing")
    actor.set_mass(float(mass))


def _actor_mass_body_counts(actor: Any) -> tuple[int, int]:
    """Return exact SAPIEN dynamic/static body counts for an actor wrapper."""

    import sapien

    entity = getattr(actor, "actor", None)
    if isinstance(entity, sapien.physx.PhysxArticulation):
        return len(entity.get_links()), 0
    if not isinstance(entity, sapien.Entity):
        raise RuntimeError("actor_physics_entity_missing")
    components = entity.get_components()
    dynamic_count = sum(
        isinstance(component, sapien.physx.PhysxRigidDynamicComponent)
        for component in components
    )
    static_count = sum(
        isinstance(component, sapien.physx.PhysxRigidStaticComponent)
        for component in components
    )
    return dynamic_count, static_count


def _set_entity_dynamic_mass(
    entity: Any,
    *,
    component_type: type,
    target_mass: float,
    update: Any,
) -> None:
    """Apply a coupled update to one required dynamic entity component."""

    component = entity.find_component_by_type(component_type)
    if component is None:
        raise RuntimeError("dynamic_mass_component_missing")
    update(component, float(target_mass))


def _vector3(value: Any, *, default: list[float]) -> np.ndarray:
    try:
        vector = np.array(value, dtype=float)
    except Exception:
        return np.array(default, dtype=float)
    if vector.shape != (3,) or np.any(vector <= 0):
        return np.array(default, dtype=float)
    return vector


def _vector3_any(value: Any, *, default: list[float]) -> np.ndarray:
    try:
        vector = np.array(value, dtype=float)
    except Exception:
        return np.array(default, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return np.array(default, dtype=float)
    return vector


def _scale3(value: Any) -> np.ndarray:
    try:
        vector = np.array(value, dtype=float)
    except Exception:
        return np.ones(3, dtype=float)
    if vector.shape == ():
        scalar = float(vector)
        return np.array([scalar, scalar, scalar], dtype=float)
    if vector.shape != (3,):
        return np.ones(3, dtype=float)
    return vector


def _attempt_result(success: bool, failure_reason: str, metrics: dict[str, Any], *, info: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "plan_success": True,
        "success": bool(success),
        "failure_reason": "ok" if success else str(failure_reason),
        "metrics": _jsonable(metrics),
        "info": _jsonable(info or {}),
    }


def _failure(
    args: argparse.Namespace,
    reason: str,
    exc: Exception,
    *,
    timings: dict[str, Any] | None = None,
    total_started: float | None = None,
) -> dict[str, Any]:
    timings = timings or {}
    if total_started is not None:
        _record_total(timings, total_started)
    return {
        "attempt_id": getattr(args, "attempt_id", None),
        "job_id": getattr(args, "job_id", None),
        "job_index": getattr(args, "job_index", None),
        "pair_key": getattr(args, "pair_key", None),
        "relation": getattr(args, "relation", None),
        "seed": int(getattr(args, "seed", 0)),
        "plan_success": False,
        "success": False,
        "failure_reason": reason,
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback_tail": traceback.format_exc()[-4000:],
        "timings": timings,
        "duration_sec": timings.get("total_worker_sec"),
        "warmup": bool(getattr(args, "warmup", False)),
    }


def _failed_checks(checks: dict[str, bool]) -> list[str]:
    return [key for key, value in checks.items() if not value]


def _semantic_failure_reason(failed_checks: list[str]) -> str:
    priority = (
        ("subject_not_table_contact", "subject_resting_on_table"),
        ("subject_receiver_contact", "missing_subject_receiver_contact"),
        ("xy_inside_receiver", "outside_receiver_cavity"),
        ("subject_meaningfully_inside_receiver", "outside_receiver_cavity"),
        ("subject_center_above_receiver_top", "subject_inside_receiver"),
        ("subject_not_meaningfully_inside_receiver", "subject_inside_receiver"),
        ("all_proxy_contents_inside_receiver_xy", "proxy_contents_outside_receiver"),
        ("all_proxy_contents_above_table", "proxy_contents_on_table"),
        ("xy_midpoint_aligned", "terminal_misaligned"),
        ("xy_aligned", "terminal_misaligned"),
        ("mug_functional_point_high", "hang_height_too_low"),
        ("stamp_contact_ids", "invalid_stamp_contact_face"),
        ("stable", "terminal_unstable"),
    )
    failed = set(failed_checks)
    for check, reason in priority:
        if check in failed:
            return reason
    return failed_checks[0] if failed_checks else "checker_false"


def _quat_multiply_wxyz(first: list[float], second: list[float]) -> list[float]:
    w1, x1, y1, z1 = (float(value) for value in first)
    w2, x2, y2, z2 = (float(value) for value in second)
    product = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )
    product /= max(float(np.linalg.norm(product)), 1e-12)
    return product.tolist()


def _endpoint_identity(endpoint: dict[str, Any]) -> dict[str, Any]:
    row = {
        "asset": str(endpoint.get("asset")),
        "model_id": str(endpoint.get("model_id")),
    }
    for key in ("category", "inventory_id", "metadata"):
        if endpoint.get(key) is not None:
            row[key] = endpoint.get(key)
    return row


def _model_id(value: Any) -> int | None:
    if value is None or str(value) == "None":
        return None
    return int(value)


def _record_timing(timings: dict[str, Any], key: str, started: float) -> None:
    timings[key] = round(float(time.monotonic() - started), 12)


def _record_total(timings: dict[str, Any], started: float) -> None:
    timings["total_worker_sec"] = round(float(time.monotonic() - started), 12)


def _load_yaml_required(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.CSafeLoader)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def _load_json_required(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON mapping in {path}")
    return data


def _clean_float(value: float) -> float:
    return round(float(value), 12)


def _path_string(path: Path) -> str:
    return str(path).replace("\\", "/")


def _safe_path_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, sort_keys=True, ensure_ascii=False, indent=2)
        f.write("\n")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Milestone 5 pair terminal-settle probe attempts.")
    parser.add_argument("--asset-catalog", required=True)
    parser.add_argument("--pair-key")
    parser.add_argument("--relation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-file")
    parser.add_argument("--batch-output")
    parser.add_argument("--reuse-task-env", action="store_true")
    parser.add_argument("--probe-job-file")
    args = parser.parse_args(argv)
    if args.batch_file:
        results = run_batch(args)
        print(json.dumps({"batch_output": args.batch_output, "batch_result_count": len(results)}, sort_keys=True, ensure_ascii=False))
        return 0
    missing = [name for name in ("seed",) if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required one-shot arguments: {', '.join(missing)}")
    result = run_worker(args)
    print(json.dumps(_jsonable(result), sort_keys=True, ensure_ascii=False))
    return 0 if result.get("success") or result.get("failure_reason") in {"checker_false", "receiver_unstable"} else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
