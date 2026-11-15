from if_benchmark.grasp_validation.carry_motion import propose_carry_motion, verify_carry_motion_receipt
import mplib.planner
import mplib
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pdb
import traceback
from if_benchmark.execution_policy import GRIPPER_PLAN_STEPS
import numpy as np
import toppra as ta
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
import transforms3d as t3d
from if_benchmark.planner_batch import (
    CUROBO_FIXED_BATCH_SIZE,
    normalize_candidate_search_budget,
)
from if_benchmark.grasp_validation.native_constraint_diagnostics import (
    NativeConstraintDiagnostics, observe_constraint_terms,
)
from if_benchmark.grasp_validation.collision_revalidation import (
    guard_start_state_constraints,
    preserve_collision_constraint_states,
)
from if_benchmark.grasp_validation.core import trajectory_summary
from if_benchmark.schema import artifact_hash
from if_benchmark.grasp_validation.convex_cover import conservative_piece_cover, verify_piece_cover
from if_benchmark.grasp_validation.world_geometry import (
    WORLD_AABB_VERSION, native_cuboid_cover, verify_native_cuboid_cover,
)
from if_benchmark.grasp_validation.held_camera import (
    bind_camera_pair, require_camera_pair, validate_camera_path, observe_camera_pair,
)
from if_benchmark.grasp_validation.acquisition_lift_validation import validate_acquisition_lift_path
from if_benchmark.grasp_validation.vertical_lift import (
    SerialChain, VerticalLiftFailure, propose_vertical_lift, verify_vertical_lift_receipt,
)
from if_benchmark.grasp_validation.table_path_validation import table_candidate_world, validate_table_path


try:
    # ********************** CuroboPlanner (optional) **********************
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.math import Pose as CuroboPose
    import time
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
        PoseCostMetric,
    )
    from curobo.util import logger
    import torch
    import yaml
    from curobo.util import logger
    logger.setup_logger(level="error", logger_name="curobo")

    class CuroboPlanner:

        ATTACHED_LINK_NAME = "attached_object"
        ATTACHED_SPHERE_COUNT = 32
        ATTACHED_SURFACE_RADIUS_M = 0.002
        CARRY_COLLISION_MODE = "m15_carry_conservative_v1"

        def __init__(
            self,
            robot_origion_pose,
            active_joints_name,
            all_joints,
            yml_path=None,
        ):
            super().__init__()
            from if_benchmark.grasp_validation.curobo_interpolation import install_stable_interpolation
            self._interpolation_contract = install_stable_interpolation()
            ta.setup_logging("CRITICAL")  # hide logging
            logger.setup_logger(level="error", logger_name="'curobo")

            if yml_path != None:
                self.yml_path = yml_path
            else:
                raise ValueError("[Planner.py]: CuroboPlanner yml_path is None!")
            self.robot_origion_pose = robot_origion_pose
            self.active_joints_name = active_joints_name
            self.all_joints = all_joints

            # translate from baselink to arm's base
            with open(self.yml_path, "r") as f:
                yml_data = yaml.safe_load(f)
            self.frame_bias = yml_data["planner"]["frame_bias"]
            robot_config = self._robot_config_with_attachment(yml_data)
            self._vertical_kinematic_config = deepcopy(robot_config["robot_cfg"]["kinematics"])
            self._world_revision = 0
            self._world_hash = None
            self._attachment_revision = 0
            self._attachment_hash = None
            self._collision_geometry_mode = "acquisition"
            self._collision_geometry_arm_tag = None
            self._last_world_obstacles = []

            # motion generation
            if True:
                world_config = {
                    "cuboid": {
                        "table": {
                            "dims": [0.7, 2, 0.04],  # x, y, z
                            "pose": [
                                self.robot_origion_pose.p[1],
                                0.0,
                                0.74 - self.robot_origion_pose.p[2],
                                1,
                                0,
                                0,
                                0.0,
                            ],  # x, y, z, qw, qx, qy, qz
                        },
                    }
                }
            collision_cache = {"obb": 64, "mesh": 0}
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_config,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=1,
                collision_cache=collision_cache,
            )

            self.motion_gen = MotionGen(motion_gen_config)
            guard_start_state_constraints(self.motion_gen)
            self.motion_gen.warmup()
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                deepcopy(robot_config),
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=1,
                num_graph_seeds=1,
                collision_cache=collision_cache,
            )
            self.motion_gen_batch = MotionGen(motion_gen_config)
            for generator in (self.motion_gen, self.motion_gen_batch):
                guard_start_state_constraints(generator)
            # Batch width is a planner throughput shape.  It is unrelated to
            # the removed legacy per-Contact ten-rotation candidate expansion.
            self.motion_gen_batch.warmup(batch=CUROBO_FIXED_BATCH_SIZE)
            self.update_world_obstacles([])

        @staticmethod
        def _stable_hash(value):
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        def _robot_config_with_attachment(self, yml_data):
            """Add a disabled collision-sphere link to the in-memory config.

            RoboTwin's embodiment assets are runtime inputs and are ignored by
            Git.  Injecting this fixed link here keeps the capability in the
            tracked planner adapter while leaving every non-attached query
            unchanged (all reserved spheres start with negative radii).
            """

            value = deepcopy(yml_data)
            kinematics = value["robot_cfg"]["kinematics"]
            config_dir = Path(self.yml_path).resolve().parent
            for key, relative_parent in (
                ("urdf_path", "urdf"),
                ("collision_spheres", ""),
            ):
                configured = Path(str(kinematics[key]))
                if configured.is_file():
                    continue
                local = config_dir / relative_parent / configured.name
                if not local.is_file():
                    raise FileNotFoundError(
                        f"CuRobo {key} does not exist: configured={configured} "
                        f"local_fallback={local}"
                    )
                kinematics[key] = str(local)
            ee_link = str(kinematics["ee_link"])
            self._held_camera_urdf = str(kinematics["urdf_path"])
            self._held_camera_ee_link = ee_link
            collision_links = list(kinematics.get("collision_link_names") or [])
            if self.ATTACHED_LINK_NAME not in collision_links:
                collision_links.append(self.ATTACHED_LINK_NAME)
            kinematics["collision_link_names"] = collision_links
            extra_spheres = dict(kinematics.get("extra_collision_spheres") or {})
            extra_spheres[self.ATTACHED_LINK_NAME] = self.ATTACHED_SPHERE_COUNT
            kinematics["extra_collision_spheres"] = extra_spheres
            extra_links = dict(kinematics.get("extra_links") or {})
            extra_links[self.ATTACHED_LINK_NAME] = {
                "parent_link_name": ee_link,
                "link_name": self.ATTACHED_LINK_NAME,
                "fixed_transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "joint_type": "FIXED",
                "joint_name": "attached_object_fixed_joint",
            }
            kinematics["extra_links"] = extra_links
            ignore = dict(kinematics.get("self_collision_ignore") or {})
            terminal_neighbors = list(ignore.get(ee_link) or [])
            if self.ATTACHED_LINK_NAME not in terminal_neighbors:
                terminal_neighbors.append(self.ATTACHED_LINK_NAME)
            ignore[ee_link] = terminal_neighbors
            grasping_links = [
                name
                for name in collision_links
                if name == ee_link or name.endswith("link7") or name.endswith("link8")
            ]
            ignore[self.ATTACHED_LINK_NAME] = grasping_links
            kinematics["self_collision_ignore"] = ignore
            buffers = dict(kinematics.get("self_collision_buffer") or {})
            buffers[self.ATTACHED_LINK_NAME] = 0.0
            kinematics["self_collision_buffer"] = buffers
            return value

        def set_collision_geometry_mode(self, mode, *, arm_tag, table_obstacles=None):
            """Switch only an explicitly selected arm's post-lift carry geometry.

            Acquisition retains the frozen Pick/provider convention. Changing
            convention invalidates both hashes and removes the old EE proxy;
            the caller must rebind its existing physical held object.
            """
            if mode not in ("acquisition", self.CARRY_COLLISION_MODE):
                raise ValueError(f"invalid collision geometry mode: {mode}")
            if arm_tag not in ("left", "right"):
                raise ValueError(f"invalid collision geometry arm_tag: {arm_tag}")
            tables = []
            if mode == self.CARRY_COLLISION_MODE:
                if not isinstance(table_obstacles, (list, tuple)) or not table_obstacles:
                    raise ValueError("corrected carry requires live table collision OBBs")
                if len(table_obstacles) > 64:
                    raise ValueError("live table obstacle cache exceeded")
                names = set()
                for index, row in enumerate(table_obstacles):
                    if not isinstance(row, dict):
                        raise ValueError(f"invalid table obstacle {index}")
                    dims = np.asarray(row.get("dims"), dtype=float)
                    pose = np.asarray(row.get("pose"), dtype=float)
                    name = str(row.get("name") or f"m15_table_shape_{index}")
                    if (dims.shape != (3,) or pose.shape != (7,)
                            or not np.isfinite(dims).all() or not np.isfinite(pose).all()
                            or np.any(dims <= 0) or np.linalg.norm(pose[3:]) < 1e-12
                            or name in names):
                        raise ValueError(f"invalid table obstacle {index}: {row!r}")
                    names.add(name)
                    tables.append({"name": name, "dims": dims.tolist(), "pose": pose.tolist()})
            previous_mode = getattr(self, "_collision_geometry_mode", "acquisition")
            previous_arm = getattr(self, "_collision_geometry_arm_tag", None)
            changed = (previous_mode != mode or (mode != "acquisition" and (
                previous_arm != arm_tag or tables != getattr(self, "_carry_table_obstacles", []))))
            if changed:
                obstacles = deepcopy(getattr(self, "_last_world_obstacles", []))
                self.detach_collision_object()
                self._world_hash = None
                self._collision_geometry_mode = mode
                self._collision_geometry_arm_tag = arm_tag if mode != "acquisition" else None
                self._carry_table_obstacles = tables
                self.update_world_obstacles(obstacles)
            return {"updated": True, "mode": mode, "arm_tag": arm_tag, "changed": changed}

        def _collision_pose_in_base(self, pose):
            if getattr(self, "_collision_geometry_mode", "acquisition") == self.CARRY_COLLISION_MODE:
                return self._target_pose_in_base(
                    SimpleNamespace(p=np.asarray(pose[:3]), q=np.asarray(pose[3:])),
                    self._collision_geometry_arm_tag,
                )
            world_base_pose = np.concatenate([
                np.asarray(self.robot_origion_pose.p, dtype=float),
                np.asarray(self.robot_origion_pose.q, dtype=float),
            ])
            return self._trans_from_world_to_base(world_base_pose, pose)

        def _collision_table_cuboids(self):
            if getattr(self, "_omit_table_candidate", False):
                return []
            if getattr(self, "_collision_geometry_mode", "acquisition") != self.CARRY_COLLISION_MODE:
                # Keep the frozen acquisition/provider geometry unchanged.
                return [Cuboid(name="table", dims=[0.7, 2.0, 0.04], pose=[
                    self.robot_origion_pose.p[1], 0.0, 0.74 - self.robot_origion_pose.p[2],
                    1.0, 0.0, 0.0, 0.0,
                ])]
            rows = getattr(self, "_carry_table_obstacles", [])
            if not rows:
                raise RuntimeError("corrected carry has no live table collision OBBs")
            cuboids = []
            for row in rows:
                position, quaternion = self._collision_pose_in_base(np.asarray(row["pose"], dtype=float))
                cuboids.append(Cuboid(name=row["name"], dims=list(row["dims"]),
                                      pose=[*position.tolist(), *quaternion.tolist()]))
            return cuboids

        def update_world_obstacles(self, obstacles):
            """Strictly install live world-frame OBBs in both CuRobo solvers."""

            if not isinstance(obstacles, (list, tuple)):
                raise ValueError("world obstacles must be a sequence")
            cuboids = self._collision_table_cuboids()
            if len(obstacles) + len(cuboids) > 64:
                raise ValueError(
                    f"world obstacle cache exceeded: {len(obstacles)} scene + {len(cuboids)} table > 64"
                )
            for index, row in enumerate(obstacles):
                if not isinstance(row, dict):
                    raise ValueError(f"world obstacle {index} is not a mapping")
                dims = np.asarray(row.get("dims"), dtype=float)
                pose = np.asarray(row.get("pose"), dtype=float)
                if (
                    dims.shape != (3,)
                    or pose.shape != (7,)
                    or not np.isfinite(dims).all()
                    or not np.isfinite(pose).all()
                    or np.any(dims <= 0.0)
                ):
                    raise ValueError(f"invalid world obstacle {index}: {row!r}")
                base_p, base_q = self._collision_pose_in_base(pose)
                cuboids.append(
                    Cuboid(
                        name=str(row.get("name") or f"scene_obstacle_{index}"),
                        dims=dims.tolist(),
                        pose=[*base_p.tolist(), *base_q.tolist()],
                    )
                )
            world = WorldConfig(cuboid=cuboids)
            # A partial solver update must never leave a usable old receipt.
            self._world_hash = None
            for label, generator in (("single", self.motion_gen), ("batch", self.motion_gen_batch)):
                if not cuboids:
                    # CuRobo's empty-OBB loader returns without disabling the
                    # previous world. Clear in place so captured CUDA graphs
                    # keep their tensor references, then install/reset normally.
                    generator.clear_world_cache()
                generator.update_world(world)
                if not cuboids:
                    checker = generator.world_coll_checker
                    if (bool(checker._cube_tensor_list[2].any().item())
                            or bool(checker._env_n_obbs.any().item())):
                        raise RuntimeError(f"empty_world_native_cache_not_cleared:{label}")
            native_coverage = self._preserve_complete_world_bounds(obstacles, cuboids)
            payload = [
                {
                    "name": str(row.get("name") or f"scene_obstacle_{index}"),
                    "dims": [float(value) for value in row["dims"]],
                    "pose": [float(value) for value in row["pose"]],
                }
                for index, row in enumerate(obstacles)
            ]
            if getattr(self, "_collision_geometry_mode", "acquisition") == self.CARRY_COLLISION_MODE or native_coverage:
                payload = {
                    "collision_geometry_mode": getattr(self, "_collision_geometry_mode", "acquisition"),
                    "arm_tag": self._collision_geometry_arm_tag,
                    "world_obstacles": payload,
                    "internal_cuboids": [
                        {"name": box.name, "dims": [float(x) for x in box.dims],
                         "pose": [float(x) for x in box.pose]}
                        for box in cuboids
                    ],
                    "complete_world_native_coverage": native_coverage,
                }
            self._last_world_obstacles = deepcopy(obstacles)
            self._world_revision += 1
            self._world_hash = self._stable_hash(payload)
            return {
                "updated": True,
                "planner": "curobo",
                "dynamic_obstacle_count": len(obstacles),
                "total_cuboid_count": len(cuboids),
                "world_revision": self._world_revision,
                "world_hash": self._world_hash,
                **({"complete_world_native_coverage": native_coverage} if native_coverage else {}),
            }

        def _preserve_complete_world_bounds(self, obstacles, cuboids):
            """Keep complete world boxes covered after actual native pose rounding.

            CuRobo stores float32 inverse poses. Read those exact poses from
            each solver, enlarge dimensions in place without changing CUDA
            graph cache references, then independently verify the readback.
            Ordinary acquisition OBBs retain their existing representation.
            """
            from itertools import product
            selected = [(i, row) for i, row in enumerate(obstacles)
                        if row.get("source") == WORLD_AABB_VERSION]
            if not selected:
                return []
            origin_p, origin_q = self._collision_pose_in_base([0., 0., 0., 1., 0., 0., 0.])
            internal_world = t3d.affines.compose(origin_p, t3d.quaternions.quat2mat(origin_q), [1, 1, 1])
            signs = np.asarray(list(product((-1., 1.), repeat=3)))
            offset = len(cuboids) - len(obstacles)
            receipts = []
            for scene_index, row in selected:
                cache_index = offset + scene_index
                pose = np.asarray(row['pose'], dtype=float)
                corners = ((signs * np.asarray(row['dims']) / 2)
                           @ t3d.quaternions.quat2mat(pose[3:]).T + pose[:3])
                points = corners @ internal_world[:3, :3].T + internal_world[:3, 3]
                slots = []
                for label, generator in (("single", self.motion_gen), ("batch", self.motion_gen_batch)):
                    checker = generator.world_coll_checker
                    cache = checker._cube_tensor_list
                    if (checker._env_obbs_names[0][cache_index] != row['name']
                            or not bool(cache[2][0, cache_index].item())):
                        raise RuntimeError(f"complete_world_native_slot_mismatch:{label}")
                    inverse = cache[1][0, cache_index, :7].detach().cpu().numpy().copy()
                    dims = cache[0][0, cache_index, :3].detach().cpu().numpy().copy()
                    if inverse.dtype != np.float32 or dims.dtype != np.float32:
                        raise RuntimeError(f"complete_world_native_dtype_mismatch:{label}")
                    cover = native_cuboid_cover(points, inverse, minimum_dims=dims)
                    slots.append((label, cache, inverse, np.asarray(cover['dims'], dtype=np.float32), dims))
                dimensions = np.maximum(slots[0][3], slots[1][3])
                # The model and both actual tensor caches share one outward
                # extent. The receipt binds inverse poses as well as extents.
                cuboids[cache_index].dims = dimensions.astype(float).tolist()
                for label, cache, inverse, _, previous_dims in slots:
                    if np.any(dimensions < previous_dims):
                        raise RuntimeError(f"complete_world_native_extent_shrank:{label}")
                    cache[0][0, cache_index, :3].copy_(torch.as_tensor(
                        dimensions, device=cache[0].device, dtype=cache[0].dtype))
                    actual = cache[0][0, cache_index, :3].detach().cpu().numpy().copy()
                    pose_after = cache[1][0, cache_index, :7].detach().cpu().numpy().copy()
                    if not np.array_equal(pose_after, inverse) or not np.array_equal(actual, dimensions):
                        raise RuntimeError(f"complete_world_native_readback_mismatch:{label}")
                    proof = verify_native_cuboid_cover(points, inverse, actual)
                    receipts.append({"obstacle": row['name'], "source_coverage_hash": row['coverage_hash'],
                                     "solver": label, "native_inverse_pose": inverse.astype(float).tolist(),
                                     "native_dimensions": actual.astype(float).tolist(), "proof": proof})
            return receipts

        def _validate_expected_state(
            self,
            *,
            expected_world_hash=None,
            expected_attachment_hash=None,
        ):
            if (
                expected_world_hash is not None
                and str(expected_world_hash) != str(self._world_hash)
            ):
                raise RuntimeError(
                    "stale_collision_world:"
                    f"expected={expected_world_hash}:actual={self._world_hash}"
                )
            if (
                expected_attachment_hash is not None
                and str(expected_attachment_hash) != str(self._attachment_hash)
            ):
                raise RuntimeError(
                    "stale_attached_object:"
                    f"expected={expected_attachment_hash}:"
                    f"actual={self._attachment_hash}"
                )

        def attach_collision_object(self, curr_joint_pos, descriptor):
            """Attach one conservative world-frame cuboid to the EE link."""

            if not isinstance(descriptor, dict):
                raise ValueError("attached collision descriptor must be a mapping")
            dims = np.asarray(descriptor.get("dims"), dtype=float)
            pose = np.asarray(descriptor.get("pose"), dtype=float)
            if (
                dims.shape != (3,)
                or pose.shape != (7,)
                or not np.isfinite(dims).all()
                or not np.isfinite(pose).all()
                or np.any(dims <= 0.0)
            ):
                raise ValueError(f"invalid attached collision object: {descriptor!r}")
            base_p, base_q = self._collision_pose_in_base(pose)
            obstacle = Cuboid(
                name=str(descriptor.get("name") or "attached_subject"),
                dims=dims.tolist(),
                pose=[*base_p.tolist(), *base_q.tolist()],
            )
            corrected = getattr(self, "_collision_geometry_mode", "acquisition") == self.CARRY_COLLISION_MODE
            joint_angles = self._active_joint_angles(curr_joint_pos)
            if not corrected:
                joint_angles = [round(angle, 5) for angle in joint_angles]
            joint_state = JointState.from_position(
                torch.tensor(joint_angles, dtype=torch.float32)
                .cuda()
                .reshape(1, -1),
                joint_names=self.active_joints_name,
            )
            self.detach_collision_object()
            coverage = None
            try:
                if corrected:
                    coverage = self._attach_conservative_cuboid(joint_state, obstacle, descriptor)
                    camera_pair = bind_camera_pair(self, joint_angles, descriptor, coverage)
                else:
                    for generator in (self.motion_gen, self.motion_gen_batch):
                        attached = generator.attach_external_objects_to_robot(
                            joint_state,
                            [obstacle],
                            surface_sphere_radius=self.ATTACHED_SURFACE_RADIUS_M,
                            link_name=self.ATTACHED_LINK_NAME,
                        )
                        if not attached:
                            raise RuntimeError("attached_object_sphere_fit_failed")
            except BaseException as error:
                # A failed batch install cannot leave the single solver armed
                # with a geometry that has no complete, verified receipt.
                try:
                    self.detach_collision_object()
                except BaseException as cleanup_error:
                    if hasattr(error, "add_note"):
                        error.add_note(f"attachment cleanup also failed: {cleanup_error!r}")
                raise
            payload = {
                "name": obstacle.name,
                "dims": dims.tolist(),
                "pose": pose.tolist(),
                "geometry_hash": descriptor.get("geometry_hash"),
                "padding_m": descriptor.get("padding_m"),
                "start_qpos": joint_angles,
            }
            if corrected:
                payload.update({
                    "collision_geometry_mode": self.CARRY_COLLISION_MODE,
                    "arm_tag": self._collision_geometry_arm_tag,
                    "internal_cuboid_pose": list(obstacle.pose),
                    "cover_version": coverage["version"],
                    "sphere_tensor_sha256": coverage["sphere_tensor_sha256"],
                    "coverage_certificate": coverage,
                    "fixed_camera_pair": camera_pair,
                })
            self._attachment_revision += 1
            self._attachment_hash = self._stable_hash(payload)
            self._held_camera_attachment_hash = self._attachment_hash if corrected else None
            return {
                **({"collision_geometry_mode": self.CARRY_COLLISION_MODE,
                    "cover_version": coverage["version"],
                    "sphere_tensor_sha256": coverage["sphere_tensor_sha256"],
                    "coverage_certificate": coverage,
                    "fixed_camera_pair": camera_pair} if corrected else {}),
                "attached": True,
                "attachment_revision": self._attachment_revision,
                "attachment_hash": self._attachment_hash,
                "sphere_count": self.ATTACHED_SPHERE_COUNT,
                "descriptor": payload,
            }

        def _attach_conservative_cuboid(self, joint_state, obstacle, descriptor):
            # The reserved attached link is an identity child of the planning
            # EE. Use the very same FK frame as native attachment, once for
            # both solvers, without stochastic sphere fitting.
            kinematics = self.motion_gen.compute_kinematics(joint_state)
            ee_pose = kinematics.ee_pose
            ee_position = ee_pose.position.detach().to("cpu").numpy().reshape(-1, 3)
            ee_quaternion = ee_pose.quaternion.detach().to("cpu").numpy().reshape(-1, 4)
            if len(ee_position) != 1 or len(ee_quaternion) != 1:
                raise RuntimeError("attached cover requires exactly one EE pose")
            internal_ee = t3d.affines.compose(
                ee_position[0], t3d.quaternions.quat2mat(ee_quaternion[0]), [1, 1, 1]
            )
            internal_cuboid = t3d.affines.compose(
                obstacle.pose[:3], t3d.quaternions.quat2mat(obstacle.pose[3:]), [1, 1, 1]
            )
            origin_p, origin_q = self._collision_pose_in_base([0., 0., 0., 1., 0., 0., 0.])
            internal_world = t3d.affines.compose(origin_p, t3d.quaternions.quat2mat(origin_q), [1, 1, 1])
            coverage = conservative_piece_cover(
                descriptor["collision_pieces"], world_to_parent=np.linalg.inv(internal_ee) @ internal_world,
                dimension_envelope_m=descriptor["dimension_envelope_m"],
                padding_m=descriptor["padding_m"], slot_count=self.ATTACHED_SPHERE_COUNT,
            )
            verify_piece_cover(
                coverage, descriptor["collision_pieces"],
                world_to_parent=np.linalg.inv(internal_ee) @ internal_world,
                dimension_envelope_m=descriptor["dimension_envelope_m"],
                padding_m=descriptor["padding_m"], slot_count=self.ATTACHED_SPHERE_COUNT,
            )
            # Retain the diagnostic AABB transform; it is not the collision
            # shape after the live whole-piece cover replaces its filled hull.
            coverage["cuboid_to_parent"] = (np.linalg.inv(internal_ee) @ internal_cuboid).tolist()
            world_to_parent = np.asarray(coverage["world_to_parent"])
            from itertools import product
            offsets = np.asarray(list(product(
                (-descriptor["dimension_envelope_m"], descriptor["dimension_envelope_m"]), repeat=3
            ))) @ world_to_parent[:3, :3].T
            points = np.vstack(descriptor["collision_pieces"]["pieces"])
            parent_points = points @ world_to_parent[:3, :3].T + world_to_parent[:3, 3]
            exact_geometry = {
                "vertices": (parent_points[:, None, :] + offsets[None, :, :]).reshape(-1, 3),
                "raw_vertices": parent_points,
                "source_raw_vertices_world": points.copy(),
                "world_to_parent": world_to_parent.copy(),
                "padding_m": descriptor["padding_m"],
                "geometry_sha256": coverage["geometry_sha256"],
            }
            expected = np.asarray(coverage["spheres"], dtype="<f4")
            sphere_sha = hashlib.sha256(expected.tobytes(order="C")).hexdigest()
            sphere_tensor = torch.as_tensor(expected.copy(), dtype=torch.float32).cuda()
            verified = []
            for label, generator in (("single", self.motion_gen), ("batch", self.motion_gen_batch)):
                generator.attach_spheres_to_robot(
                    sphere_tensor=sphere_tensor, link_name=self.ATTACHED_LINK_NAME
                )
                actual = (generator.robot_cfg.kinematics.kinematics_config
                          .get_link_spheres(self.ATTACHED_LINK_NAME)
                          .detach().to("cpu").numpy())
                actual_sha = hashlib.sha256(
                    np.asarray(actual, dtype="<f4").tobytes(order="C")
                ).hexdigest()
                if actual.shape != expected.shape or actual.dtype != np.float32 or actual_sha != sphere_sha:
                    raise RuntimeError(f"attached_object_tensor_readback_mismatch:{label}")
                verified.append(label)
            self._carry_exact_table_geometry = exact_geometry
            coverage.update(sphere_tensor_sha256=sphere_sha, verified_solver_slots=verified)
            coverage["cover_hash"] = artifact_hash(coverage, omit_keys=("cover_hash",))
            return coverage

        def detach_collision_object(self):
            was_attached = self._attachment_hash is not None
            self._attachment_revision += 1
            self._attachment_hash = None
            self._carry_exact_table_geometry = None
            self._held_camera_gate = None
            self._held_camera_observation_failed = False
            self._held_camera_attachment_hash = None
            first_error = None
            if getattr(self, "_held_camera_mask", None) is not None:
                try:
                    self._held_camera_mask.restore()
                except BaseException as error:
                    first_error = error
            for generator in (self.motion_gen, self.motion_gen_batch):
                try:
                    generator.detach_object_from_robot(self.ATTACHED_LINK_NAME)
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error
            return {
                "detached": True,
                "was_attached": was_attached,
                "attachment_revision": self._attachment_revision,
                "attachment_hash": None,
            }

        def reset_collision_state(self):
            for generator in (self.motion_gen, self.motion_gen_batch):
                guard_start_state_constraints(generator)
            detached = self.detach_collision_object()
            self._collision_geometry_mode = "acquisition"
            self._collision_geometry_arm_tag = None
            self._carry_table_obstacles = []
            world = self.update_world_obstacles([])
            self.motion_gen.reset(reset_seed=True)
            self.motion_gen_batch.reset(reset_seed=True)
            return {"reset": True, "detached": detached, "world": world}

        def validate_path_collision_world(self, qpos_rows, *, expected_world_hash, held_table_validation=False):
            """Check the original path against current native CuRobo constraints."""
            self._validate_expected_state(expected_world_hash=expected_world_hash)
            values = np.asarray(qpos_rows, dtype=float)
            if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
                raise ValueError("collision-world validation requires a finite 2D path")
            if held_table_validation:
                if (getattr(self, "_collision_geometry_mode", None) != self.CARRY_COLLISION_MODE
                        or not self._attachment_hash):
                    raise RuntimeError("held_handoff_requires_complete_attachment")
                attachment_hash = self._attachment_hash
                with table_candidate_world(self) as binding:
                    native = self.validate_path_collision_world(values, expected_world_hash=self._world_hash)
                    table = self._validate_carry_table_path(self.motion_gen.tensor_args.to_device(values))
                    locked = self.motion_gen.kinematics.lock_jointstate
                    locks = {} if locked is None else dict(zip(locked.joint_names,
                        locked.position.detach().to("cpu").numpy().reshape(-1).astype(float).tolist()))
                self._validate_expected_state(expected_world_hash=expected_world_hash,
                                              expected_attachment_hash=attachment_hash)
                return {**native, "world_hash": self._world_hash,
                        "validator": "if_held_handoff_native_and_exact_table_v1",
                        "collision_free": native["collision_free"] and table["collision_free"],
                        "native_non_table_validation": native, "table_validation": table,
                        "attachment_hash": attachment_hash, "planning_lock_joint_positions": locks,
                        **binding}
            pair = validate_camera_path(self, values)
            active = np.asarray([self._active_joint_angles(row) for row in values])
            rollout = self.motion_gen.rollout_fn
            if rollout.cuda_graph_instance:
                raise RuntimeError("native collision validation requires the non-graph rollout")
            failed = []
            diagnostics = NativeConstraintDiagnostics()
            with preserve_collision_constraint_states(self.motion_gen):
                for name in ("primitive_collision_constraint", "robot_self_collision_constraint"):
                    constraint = getattr(rollout, name)
                    constraint.enable_cost()
                    if not constraint.enabled:
                        raise RuntimeError(f"required collision constraint is disabled: {name}")
                for offset in range(0, len(active), 128):
                    positions = torch.tensor(active[offset:offset + 128], dtype=torch.float32).cuda()
                    with observe_constraint_terms(rollout) as terms:
                        metrics = rollout.rollout_constraint(positions.unsqueeze(1), use_batch_env=False)
                    feasible = metrics.feasible.detach().to("cpu").numpy().reshape(-1)
                    if len(feasible) != len(positions):
                        raise RuntimeError("native collision result has the wrong waypoint count")
                    diagnostics.add(terms, feasible, offset=offset)
                    failed.extend(offset + int(index) for index in np.flatnonzero(~feasible))
            self._validate_expected_state(expected_world_hash=expected_world_hash)
            return {
                "validator": "curobo_native_path_constraints_v1",
                "collision_free": not failed and (pair is None or pair["collision_free"]),
                **({"fixed_camera_path": pair} if pair is not None else {}),
                "world_hash": str(self._world_hash),
                "trajectory_sha256": trajectory_summary(values)["trajectory_sha256"],
                "checked_waypoint_count": len(values),
                "failed_waypoint_indices": failed,
                "constraints": ["joint_bounds", "world_collision", "self_collision"],
                "constraint_diagnostics": diagnostics.receipt(),
            }

        def _planner_state(self):
            return {
                **({"trajectory_interpolation": deepcopy(self._interpolation_contract)}
                   if getattr(self, "_interpolation_contract", None) is not None else {}),
                **({"collision_geometry_mode": self.CARRY_COLLISION_MODE,
                    "collision_geometry_arm_tag": self._collision_geometry_arm_tag}
                   if getattr(self, "_collision_geometry_mode", "acquisition") == self.CARRY_COLLISION_MODE else {}),
                "world_revision": self._world_revision,
                "world_hash": self._world_hash,
                "attachment_revision": self._attachment_revision,
                "attachment_hash": self._attachment_hash,
            }

        def observe_held_camera_pair(self, qpos, descriptor, *, expected_attachment_hash):
            return observe_camera_pair(self, qpos, descriptor, expected_attachment_hash=expected_attachment_hash)

        def get_collision_state(self):
            return dict(self._planner_state())

        def _target_pose_in_base(self, target_gripper_pose, arms_tag):
            world_base_pose = np.concatenate(
                [
                    np.asarray(self.robot_origion_pose.p, dtype=float),
                    np.asarray(self.robot_origion_pose.q, dtype=float),
                ]
            )
            world_target_pose = np.concatenate(
                [
                    np.asarray(target_gripper_pose.p, dtype=float),
                    np.asarray(target_gripper_pose.q, dtype=float),
                ]
            )
            target_pose_p, target_pose_q = self._trans_from_world_to_base(
                world_base_pose, world_target_pose
            )
            if not ("aloha-agilex" in self.yml_path):
                target_pose_p += np.asarray(self.frame_bias, dtype=float)
                return target_pose_p, target_pose_q
            target_matrix = t3d.affines.compose(
                target_pose_p,
                t3d.quaternions.quat2mat(target_pose_q),
                [1, 1, 1],
            )
            bias_matrix = t3d.affines.compose(
                self.frame_bias, np.eye(3), [1, 1, 1]
            )
            if arms_tag == "left":
                angle = -0.02
            elif arms_tag == "right":
                angle = -0.01
            else:
                raise ValueError(f"Invalid arms_tag: {arms_tag}")
            rotation_matrix = t3d.affines.compose(
                [0, 0, 0],
                t3d.axangles.axangle2mat([0, 0, 1], angle),
                [1, 1, 1],
            )
            transformed = rotation_matrix @ bias_matrix @ target_matrix
            return (
                transformed[:3, 3],
                t3d.quaternions.mat2quat(transformed[:3, :3]),
            )

        def _target_pose_from_internal_base(self, position, quaternion, arms_tag):
            internal = t3d.affines.compose(
                np.asarray(position, dtype=float),
                t3d.quaternions.quat2mat(np.asarray(quaternion, dtype=float)),
                [1, 1, 1],
            )
            if "aloha-agilex" in self.yml_path:
                if arms_tag == "left":
                    angle = -0.02
                elif arms_tag == "right":
                    angle = -0.01
                else:
                    raise ValueError(f"Invalid arms_tag: {arms_tag}")
                rotation = t3d.affines.compose(
                    [0, 0, 0],
                    t3d.axangles.axangle2mat([0, 0, 1], angle),
                    [1, 1, 1],
                )
                bias = t3d.affines.compose(
                    self.frame_bias, np.eye(3), [1, 1, 1]
                )
                target_base = np.linalg.inv(rotation @ bias) @ internal
            else:
                target_base = internal.copy()
                target_base[:3, 3] -= np.asarray(self.frame_bias, dtype=float)
            world_base = t3d.affines.compose(
                np.asarray(self.robot_origion_pose.p, dtype=float),
                t3d.quaternions.quat2mat(
                    np.asarray(self.robot_origion_pose.q, dtype=float)
                ),
                [1, 1, 1],
            )
            world = world_base @ target_base
            return [
                *world[:3, 3].tolist(),
                *t3d.quaternions.mat2quat(world[:3, :3]).tolist(),
            ]

        def fk_endlink_path(self, qpos_rows, arms_tag=None):
            values = np.asarray(qpos_rows, dtype=float)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            active = np.asarray(
                [self._active_joint_angles(row) for row in values], dtype=float
            )
            state = JointState.from_position(
                torch.tensor(active, dtype=torch.float32).cuda(),
                joint_names=self.active_joints_name,
            )
            kinematics = self.motion_gen.compute_kinematics(state)
            positions = kinematics.ee_pos_seq.detach().to("cpu").numpy()
            quaternions = kinematics.ee_quat_seq.detach().to("cpu").numpy()
            return [
                self._target_pose_from_internal_base(position, quaternion, arms_tag)
                for position, quaternion in zip(positions, quaternions)
            ]

        def fk_collision_spheres(self, qpos, arms_tag=None):
            """Return non-attached robot collision spheres in world frame."""

            active = np.asarray(
                self._active_joint_angles(qpos), dtype=float
            ).reshape(1, -1)
            tensor = torch.tensor(active, dtype=torch.float32).cuda()
            sphere_rows = self.motion_gen.kinematics.get_robot_as_spheres(
                tensor
            )[0]
            attached_indices = set(
                int(value)
                for value in self.motion_gen.kinematics.kinematics_config
                .get_sphere_index_from_link_name(self.ATTACHED_LINK_NAME)
                .detach()
                .to("cpu")
                .numpy()
                .tolist()
            )
            result = []
            for index, sphere in enumerate(sphere_rows):
                if index in attached_indices:
                    continue
                world_pose = self._target_pose_from_internal_base(
                    sphere.pose[:3], [1.0, 0.0, 0.0, 0.0], arms_tag
                )
                result.append(
                    {
                        "name": f"{arms_tag}_source_sphere_{index}",
                        "center": world_pose[:3],
                        "radius": float(sphere.radius),
                    }
                )
            return result

        def _active_joint_angles(self, curr_joint_pos):
            """Accept either full articulation qpos or chained arm qpos."""

            values = np.asarray(curr_joint_pos, dtype=float).reshape(-1)
            if len(values) == len(self.active_joints_name):
                return values.tolist()
            joint_indices = [
                self.all_joints.index(name)
                for name in self.active_joints_name
                if name in self.all_joints
            ]
            if len(joint_indices) != len(self.active_joints_name):
                raise ValueError("active arm joint is missing from planner joint order")
            if not joint_indices or max(joint_indices) >= len(values):
                raise ValueError(
                    "planner qpos is neither full-articulation nor active-arm sized: "
                    f"got={len(values)} active={len(self.active_joints_name)} "
                    f"required_full_index={max(joint_indices, default=-1)}"
                )
            return [float(values[index]) for index in joint_indices]

        def _validate_carry_table_path(self, positions, *, route_lift_support=False,
                                       acquisition_clearance_m=None):
            values = positions.detach().to("cpu").numpy()
            if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
                raise ValueError("exact table validation requires the complete interpolated path")
            state = self.motion_gen.kinematics.get_state(positions)
            spheres = state.get_link_spheres().detach().to("cpu").numpy().copy()
            indices = (self.motion_gen.kinematics.kinematics_config
                       .get_sphere_index_from_link_name(self.ATTACHED_LINK_NAME)
                       .detach().to("cpu").numpy().reshape(-1))
            robot_spheres = np.delete(spheres, indices, axis=1)
            origin_p, origin_q = self._collision_pose_in_base([0., 0., 0., 1., 0., 0., 0.])
            internal_world = t3d.affines.compose(origin_p, t3d.quaternions.quat2mat(origin_q), [1, 1, 1])
            world_internal = np.linalg.inv(internal_world)
            robot_spheres[:, :, :3] = (robot_spheres[:, :, :3] @ world_internal[:3, :3].T
                                       + world_internal[:3, 3])
            ee_positions = state.ee_position.detach().to("cpu").numpy()
            ee_quaternions = state.ee_quaternion.detach().to("cpu").numpy()
            ee_matrices = [world_internal @ t3d.affines.compose(p, t3d.quaternions.quat2mat(q), [1, 1, 1])
                           for p, q in zip(ee_positions, ee_quaternions)]
            geometry = self._carry_exact_table_geometry
            if acquisition_clearance_m is not None:
                kcfg = self.motion_gen.kinematics.kinematics_config
                mapping = kcfg.link_sphere_idx_map.detach().to("cpu").numpy().reshape(-1)
                names = {int(value): str(name) for name, value in kcfg.link_name_to_idx_map.items()}
                links = [names[int(value)] for value in np.delete(mapping, indices)]
                if len(mapping) != spheres.shape[1]:
                    raise RuntimeError("acquisition_native_sphere_link_count_mismatch")
                return validate_acquisition_lift_path(
                    robot_spheres_world=robot_spheres, sphere_link_names=links,
                    ee_world_matrices=ee_matrices,
                    raw_held_vertices_parent=geometry["raw_vertices"],
                    source_raw_vertices_world=geometry["source_raw_vertices_world"],
                    expanded_held_vertices_parent=geometry["vertices"],
                    held_expansion_world_to_parent=geometry["world_to_parent"],
                    padding_m=geometry["padding_m"], table_obstacles=self._carry_table_obstacles,
                    required_pick_clearance_m=acquisition_clearance_m,
                    arm=self._collision_geometry_arm_tag,
                    trajectory_sha256=trajectory_summary(values)["trajectory_sha256"],
                    geometry_sha256=geometry["geometry_sha256"],
                )
            return validate_table_path(
                robot_spheres_world=robot_spheres, ee_world_matrices=ee_matrices,
                held_vertices_parent=geometry["vertices"], padding_m=geometry["padding_m"],
                table_obstacles=self._carry_table_obstacles,
                trajectory_sha256=trajectory_summary(values)["trajectory_sha256"],
                geometry_sha256=geometry["geometry_sha256"],
                route_lift_support=route_lift_support,
            )

        def _plan_vertical_lift(self, start, target_position, target_quaternion):
            return self._plan_cartesian_motion(start, target_position, target_quaternion)

        def _plan_cartesian_motion(self, start, target_position, target_quaternion, *, carry_mode=None):
            """Generate an upward Cartesian candidate; original gates admit it."""
            generator = self.motion_gen
            origin_p, origin_q = self._collision_pose_in_base([0., 0., 0., 1., 0., 0., 0.])
            world_base = np.linalg.inv(t3d.affines.compose(origin_p, t3d.quaternions.quat2mat(origin_q), [1, 1, 1]))
            target = world_base @ t3d.affines.compose(target_position, t3d.quaternions.quat2mat(target_quaternion), [1, 1, 1])
            locked = generator.kinematics.lock_jointstate
            locks = {} if locked is None else dict(zip(locked.joint_names, locked.position.detach().cpu().numpy().reshape(-1)))
            config = self._vertical_kinematic_config
            chain = SerialChain(config["urdf_path"], config["base_link"], config["ee_link"], self.active_joints_name, locks)
            native_limits = generator.kinematics.get_joint_limits()
            if list(native_limits.joint_names) != list(self.active_joints_name):
                raise RuntimeError("vertical_lift_native_joint_order_mismatch")
            limits = {key: getattr(native_limits, key).detach().cpu().numpy()
                      for key in ("position", "velocity", "acceleration", "jerk")}

            def native_fk(rows):
                poses = []
                for offset in range(0, len(rows), 128):
                    state = generator.kinematics.get_state(generator.tensor_args.to_device(rows[offset:offset + 128]))
                    poses.extend(world_base @ t3d.affines.compose(p, t3d.quaternions.quat2mat(q), [1, 1, 1])
                                 for p, q in zip(state.ee_position.detach().cpu().numpy(),
                                                 state.ee_quaternion.detach().cpu().numpy()))
                return np.asarray(poses)

            try:
                path, velocity, proof = (propose_vertical_lift if carry_mode is None else propose_carry_motion)(
                    chain, world_base, start.position.detach().cpu().numpy().reshape(-1),
                    target, limits, native_fk, dt=float(generator.interpolation_dt),
                    **({"mode": carry_mode} if carry_mode is not None else {}))
            except VerticalLiftFailure as exc:
                return SimpleNamespace(success=torch.tensor(False), status=str(exc))
            return SimpleNamespace(success=torch.tensor(True), status=("CARTESIAN_VERTICAL_LIFT" if carry_mode is None else "CARTESIAN_CARRY_" + carry_mode.upper()),
                **({"vertical_lift": proof} if carry_mode is None else {"carry_motion": proof}), interpolated_plan=SimpleNamespace(
                    position=generator.tensor_args.to_device(path), velocity=generator.tensor_args.to_device(velocity)))

        def plan_path(
            self,
            curr_joint_pos,
            target_gripper_pose,
            constraint_pose=None,
            arms_tag=None,
            expected_world_hash=None,
            expected_attachment_hash=None,
            m15_exact_table_validation=False,
            m15_route_lift_support=False,
            m15_route_acquisition_clearance_m=None,
            m15_carry_cartesian_motion=None,
        ):
            carry_mode = m15_carry_cartesian_motion
            acquisition = m15_route_acquisition_clearance_m is not None
            if carry_mode is not None and (carry_mode not in {"duplicate", "descent"}
                    or acquisition or m15_route_lift_support or constraint_pose is not None
                    or not m15_exact_table_validation
                    or getattr(self, "_collision_geometry_mode", None) != self.CARRY_COLLISION_MODE):
                raise ValueError("Cartesian Carry requires corrected geometry, exact table and its own constraint")
            if acquisition and (not m15_exact_table_validation or m15_route_lift_support
                    or getattr(self, "_collision_geometry_mode", None) != self.CARRY_COLLISION_MODE):
                raise ValueError("acquisition lift requires corrected geometry and its distinct table gate")
            self._validate_expected_state(
                expected_world_hash=expected_world_hash,
                expected_attachment_hash=expected_attachment_hash,
            )
            camera_gate = require_camera_pair(self)
            if camera_gate is not None and not camera_gate["passed"]:
                return {**self._planner_state(), "status": "Fail",
                        "planner_status": "HELD_FIXED_CAMERA_PAIR_REJECTED",
                        "fixed_camera_pair": deepcopy(camera_gate)}
            target_pose_p, target_pose_q = self._target_pose_in_base(
                target_gripper_pose, arms_tag
            )

            goal_pose_of_ee = CuroboPose.from_list(list(target_pose_p) + list(target_pose_q))
            joint_angles = self._active_joint_angles(curr_joint_pos)
            if getattr(self, "_collision_geometry_mode", "acquisition") != self.CARRY_COLLISION_MODE:
                joint_angles = [round(angle, 5) for angle in joint_angles]
            start_joint_states = JointState.from_position(
                torch.tensor(joint_angles).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )
            # plan
            plan_config = MotionGenPlanConfig(max_attempts=10)
            if acquisition:
                if constraint_pose is not None:
                    raise ValueError("acquisition lift owns its world-Z proposal constraint")
            elif constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                )
                plan_config.pose_cost_metric = pose_cost_metric

            table_validation = None
            if m15_exact_table_validation:
                if not expected_world_hash or not expected_attachment_hash:
                    raise RuntimeError("exact table path requires both collision bindings")
                with table_candidate_world(self) as table_binding:
                    result = (self._plan_vertical_lift(start_joint_states, target_pose_p, target_pose_q)
                              if acquisition else self._plan_cartesian_motion(
                                  start_joint_states, target_pose_p, target_pose_q, carry_mode=carry_mode)
                              if carry_mode is not None else self.motion_gen.plan_single(
                                  start_joint_states, goal_pose_of_ee, plan_config))
                    if result.success.item():
                        table_validation = {
                            **self._validate_carry_table_path(
                                result.interpolated_plan.position, route_lift_support=m15_route_lift_support,
                                acquisition_clearance_m=m15_route_acquisition_clearance_m,
                            ),
                            **table_binding,
                            "attachment_hash": self._attachment_hash,
                        }
                        if acquisition or carry_mode is not None:
                            native = self.validate_path_collision_world(
                                result.interpolated_plan.position.detach().to("cpu").numpy(),
                                expected_world_hash=self._world_hash,
                            )
                            if (native["trajectory_sha256"] != table_validation["trajectory_sha256"]
                                    or native["checked_waypoint_count"] != table_validation["checked_waypoint_count"]):
                                raise RuntimeError("acquisition_native_path_binding_mismatch")
                            table_validation["native_non_table_validation"] = native
                            if carry_mode is not None:
                                verify_carry_motion_receipt(result.carry_motion,
                                    result.interpolated_plan.position.detach().cpu().numpy(), mode=carry_mode)
                                table_validation["carry_motion"] = result.carry_motion
                                table_validation["collision_free"] = table_validation["collision_free"] and native["collision_free"]
                        if acquisition:
                            vertical = result.vertical_lift
                            verify_vertical_lift_receipt(vertical, result.interpolated_plan.position.detach().cpu().numpy())
                            table_validation.update(
                                vertical_lift=vertical,
                                native_non_table_validation=native,
                                requires_full_robot_fcl=True,
                                planner_checks_passed=(table_validation["geometry_checks_passed"]
                                                       and native["collision_free"]),
                            )
                        locked = self.motion_gen.kinematics.lock_jointstate
                        table_validation["planning_lock_joint_positions"] = (
                            {} if locked is None else dict(zip(
                                locked.joint_names,
                                locked.position.detach().to("cpu").numpy().reshape(-1).astype(float).tolist(),
                            ))
                        )
                        table_validation["validation_hash"] = artifact_hash(
                            table_validation, omit_keys=("validation_hash",)
                        )
            else:
                result = self.motion_gen.plan_single(start_joint_states, goal_pose_of_ee, plan_config)

            # Bind the replacement gate to the same complete, unmodified path.
            pair = None
            if result.success.item():
                pair = validate_camera_path(self, result.interpolated_plan.position.detach().to("cpu").numpy())
                if pair is not None and table_validation is not None:
                    table_validation["fixed_camera_path"] = pair
                    table_validation["validation_hash"] = artifact_hash(table_validation, omit_keys=("validation_hash",))
            res_result = {**self._planner_state()}
            if acquisition and hasattr(result, "vertical_lift"):
                res_result["vertical_lift"] = result.vertical_lift
            if carry_mode is not None and hasattr(result, "carry_motion"):
                res_result["carry_motion"] = result.carry_motion
            if pair is not None:
                res_result["fixed_camera_path"] = pair
                if not pair["collision_free"]:
                    return {**res_result, "status": "Fail", "planner_status": "HELD_FIXED_CAMERA_PATH_REJECTED"}
            if table_validation is not None:
                res_result["table_validation"] = table_validation
                if not table_validation["planner_checks_passed" if acquisition else "collision_free"]:
                    return {**res_result, "status": "Fail", "planner_status": "TABLE_PATH_VALIDATION_FAILED"}
            raw_status = getattr(result, "status", None)
            if raw_status is not None:
                res_result["planner_status"] = str(raw_status)
            if result.success.item() == False:
                res_result["status"] = "Fail"
                return res_result
            else:
                res_result["status"] = "Success"
                res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
                res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))
                return res_result

        def plan_batch(
            self,
            curr_joint_pos,
            target_gripper_pose_list,
            constraint_pose=None,
            arms_tag=None,
            expected_world_hash=None,
            expected_attachment_hash=None,
            candidate_search_budget=None,
        ):
            """
            Plan a batch of trajectories for multiple target poses.

            Input:
                - curr_joint_pos: List of current joint angles (1 x n)
                - target_gripper_pose_list: List of target poses [sapien.Pose, sapien.Pose, ...]

            Output:
                - result['status']: numpy array of string values indicating "Success"/"Fail" for each pose
                - result['position']: numpy array of joint positions with shape (n x m x l)
                  where n is number of target poses, m is number of waypoints, l is number of joints
                - result['velocity']: numpy array of joint velocities with same shape as position
            """

            budget = normalize_candidate_search_budget(candidate_search_budget)
            if budget is not None and expected_attachment_hash is not None:
                raise ValueError("candidate search budget cannot limit attached mandatory paths")
            self._validate_expected_state(
                expected_world_hash=expected_world_hash,
                expected_attachment_hash=expected_attachment_hash,
            )
            num_poses = len(target_gripper_pose_list)
            camera_gate = require_camera_pair(self)
            if camera_gate is not None and not camera_gate["passed"]:
                return {**self._planner_state(), "status": ["Fail"] * num_poses,
                        "planner_status": "HELD_FIXED_CAMERA_PAIR_REJECTED",
                        "fixed_camera_pair": deepcopy(camera_gate)}
            poses_list = []
            for target_gripper_pose in target_gripper_pose_list:
                base_target_pose_p, base_target_pose_q = self._target_pose_in_base(
                    target_gripper_pose, arms_tag
                )

                base_target_pose_list = list(base_target_pose_p) + list(base_target_pose_q)
                poses_list.append(base_target_pose_list)

            poses_cuda = torch.tensor(poses_list, dtype=torch.float32).cuda()
            goal_pose_of_ee = CuroboPose(poses_cuda[:, :3], poses_cuda[:, 3:])
            raw_joint_angles = np.asarray(curr_joint_pos, dtype=float)
            if raw_joint_angles.ndim == 1:
                joint_angles = self._active_joint_angles(raw_joint_angles)
                joint_rows = [joint_angles for _ in range(num_poses)]
            elif raw_joint_angles.ndim == 2:
                if len(raw_joint_angles) != num_poses:
                    raise ValueError(
                        "batched start qpos must contain one row per target pose: "
                        f"got={len(raw_joint_angles)} targets={num_poses}"
                    )
                joint_rows = [
                    self._active_joint_angles(row) for row in raw_joint_angles
                ]
            else:
                raise ValueError(
                    "planner batch qpos must be one or two dimensional"
                )
            joint_rows = [
                [round(angle, 5) for angle in row]
                for row in joint_rows
            ]  # avoid the precision problem
            joint_angles_cuda = torch.tensor(
                joint_rows, dtype=torch.float32
            ).cuda()
            start_joint_states = JointState.from_position(joint_angles_cuda, joint_names=self.active_joints_name)
            # plan
            plan_config = MotionGenPlanConfig(
                **({"max_attempts": 10} if budget is None else budget)
            )
            if constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                )
                plan_config.pose_cost_metric = pose_cost_metric

            try:
                result = self.motion_gen_batch.plan_batch(start_joint_states, goal_pose_of_ee, plan_config)
            except TimeoutError:
                # A worker deadline cancels the attempt; it is not a solver result.
                raise
            except Exception as e:
                return {
                    "status": ["Exception" for _ in range(num_poses)],
                    "exception": f"{type(e).__name__}:{e}",
                    **self._planner_state(),
                }

            # output
            res_result = {**self._planner_state()}
            if budget is not None:
                res_result.update({
                    "candidate_search_budget": budget,
                    "candidate_search_budget_applied": True,
                })
            raw_status = getattr(result, "status", None)
            if raw_status is not None:
                res_result["planner_status"] = str(raw_status)
            # Convert boolean success values to "Success"/"Failure" strings
            success_array = result.success.cpu().numpy()
            status_array = np.array(["Success" if s else "Failure" for s in success_array], dtype=object)
            res_result["status"] = status_array

            last_tstep = getattr(result, "path_buffer_last_tstep", None)
            if last_tstep is not None:
                if hasattr(last_tstep, "detach"):
                    last_tstep = last_tstep.detach().to("cpu").numpy()
                else:
                    converted_steps = []
                    for value in last_tstep:
                        if hasattr(value, "detach"):
                            value = value.detach().to("cpu")
                        if hasattr(value, "item"):
                            value = value.item()
                        converted_steps.append(int(value))
                    last_tstep = converted_steps
                res_result["path_buffer_last_tstep"] = np.asarray(
                    last_tstep, dtype=int
                ).reshape(-1)

            if np.all(res_result["status"] == "Failure"):
                return res_result

            res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
            res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))
            if camera_gate is not None:
                checks = []
                for index, status in enumerate(res_result["status"]):
                    pair = None
                    if status == "Success":
                        pair = validate_camera_path(self, res_result["position"][index], generator=self.motion_gen_batch)
                        if not pair["collision_free"]:
                            res_result["status"][index] = "Fail"
                    checks.append(pair)
                res_result["fixed_camera_paths"] = checks
            return res_result

        def solve_ik_batch(self, target_pose_list, arms_tag=None):
            """Solve collision-aware IK for end-link poses in fixed-size chunks.

            This deliberately answers only pose feasibility in the currently
            installed world.  The generic reach-envelope caller installs an
            empty dynamic world before invoking it, so the resulting artifact
            is not scene-specific path evidence.
            """

            camera_gate = require_camera_pair(self)
            if camera_gate is not None and not camera_gate["passed"]:
                raise RuntimeError("held_fixed_camera_pair_rejected_before_ik")
            if not target_pose_list:
                return {
                    "status": [],
                    "solution": [],
                    "position_error": [],
                    "rotation_error": [],
                    "joint_names": list(self.active_joints_name),
                    "joint_limits": [],
                }
            pose_rows = []
            for pose in target_pose_list:
                position, quaternion = self._target_pose_in_base(pose, arms_tag)
                pose_rows.append([*position.tolist(), *quaternion.tolist()])

            output = []
            for offset in range(0, len(pose_rows), CUROBO_FIXED_BATCH_SIZE):
                chunk = pose_rows[offset : offset + CUROBO_FIXED_BATCH_SIZE]
                valid_count = len(chunk)
                if valid_count < CUROBO_FIXED_BATCH_SIZE:
                    chunk.extend([chunk[-1]] * (CUROBO_FIXED_BATCH_SIZE - valid_count))
                values = torch.tensor(chunk, dtype=torch.float32).cuda()
                result = self.motion_gen_batch.ik_solver.solve_batch(
                    CuroboPose(values[:, :3], values[:, 3:])
                )
                success = result.success.detach().to("cpu").numpy()
                solutions = result.solution.detach().to("cpu").numpy()
                position_error = result.position_error.detach().to("cpu").numpy()
                rotation_error = result.rotation_error.detach().to("cpu").numpy()
                for index in range(valid_count):
                    success_row = np.asarray(success[index]).reshape(-1)
                    solution_row = np.asarray(solutions[index])
                    position_row = np.asarray(position_error[index]).reshape(-1)
                    rotation_row = np.asarray(rotation_error[index]).reshape(-1)
                    valid = np.flatnonzero(success_row)
                    selected = int(valid[0]) if len(valid) else int(
                        np.argmin(position_row + rotation_row)
                    )
                    if solution_row.ndim == 1:
                        selected_solution = solution_row
                    else:
                        selected_solution = solution_row[selected]
                    output.append(
                        {
                            "success": bool(len(valid)),
                            "solution": selected_solution.tolist(),
                            "position_error": float(position_row[selected]),
                            "rotation_error": float(rotation_row[selected]),
                        }
                    )

            if camera_gate is not None:
                for row in output:
                    if row["success"]:
                        pair = validate_camera_path(self, np.asarray(row["solution"])[None], generator=self.motion_gen_batch)
                        row["success"] = pair["collision_free"]
            limits = self.motion_gen.kinematics.get_joint_limits()
            position_limits = limits.position.detach().to("cpu").numpy()
            if position_limits.shape == (2, len(limits.joint_names)):
                position_limits = position_limits.T
            if position_limits.shape != (len(limits.joint_names), 2):
                raise RuntimeError(
                    "unexpected CuRobo joint-limit shape: "
                    f"{position_limits.shape}"
                )
            return {
                "status": ["Success" if row["success"] else "Fail" for row in output],
                "solution": [row["solution"] for row in output],
                "position_error": [row["position_error"] for row in output],
                "rotation_error": [row["rotation_error"] for row in output],
                "joint_names": list(limits.joint_names),
                "joint_limits": position_limits.tolist(),
                **self._planner_state(),
            }

        def plan_grippers(self, now_val, target_val):
            num_step = GRIPPER_PLAN_STEPS
            dis_val = target_val - now_val
            step = dis_val / num_step
            res = {}
            vals = np.linspace(now_val, target_val, num_step)
            res["num_step"] = num_step
            res["per_step"] = step
            res["result"] = vals
            return res

        def _trans_from_world_to_base(self, base_pose, target_pose):
            '''
                transform target pose from world frame to base frame
                base_pose: np.array([x, y, z, qw, qx, qy, qz])
                target_pose: np.array([x, y, z, qw, qx, qy, qz])
            '''
            base_p, base_q = base_pose[0:3], base_pose[3:]
            target_p, target_q = target_pose[0:3], target_pose[3:]
            rel_p = target_p - base_p
            wRb = t3d.quaternions.quat2mat(base_q)
            wRt = t3d.quaternions.quat2mat(target_q)
            result_p = wRb.T @ rel_p
            result_q = t3d.quaternions.mat2quat(wRb.T @ wRt)
            return result_p, result_q
    
except Exception as e:
    print('[planner.py]: Something wrong happened when importing CuroboPlanner! Please check if Curobo is installed correctly. If the problem still exists, you can install Curobo from https://github.com/NVlabs/curobo manually.')
    print('Exception traceback:')
    traceback.print_exc()


# ********************** MplibPlanner **********************
class MplibPlanner:
    # links=None, joints=None
    def __init__(
        self,
        urdf_path,
        srdf_path,
        move_group,
        robot_origion_pose,
        robot_entity,
        planner_type="mplib_RRT",
        scene=None,
    ):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging

        links = [link.get_name() for link in robot_entity.get_links()]
        joints = [joint.get_name() for joint in robot_entity.get_active_joints()]

        if scene is None:
            self.planner = mplib.Planner(
                urdf=urdf_path,
                srdf=srdf_path,
                move_group=move_group,
                user_link_names=links,
                user_joint_names=joints,
                use_convex=False,
            )
            self.planner.set_base_pose(robot_origion_pose)
        else:
            planning_world = SapienPlanningWorld(scene, [robot_entity])
            self.planner = SapienPlanner(planning_world, move_group)

        self.planner_type = planner_type
        self.plan_step_lim = 2500
        self.TOPP = self.planner.TOPP

    def show_info(self):
        print("joint_limits", self.planner.joint_limits)
        print("joint_acc_limits", self.planner.joint_acc_limits)

    def plan_pose(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        try_times=2,
        log=True,
    ):
        result = {}
        result["status"] = "Fail"

        now_try_times = 1
        while result["status"] != "Success" and now_try_times < try_times:
            result = self.planner.plan_pose(
                goal_pose=target_pose,
                current_qpos=np.array(now_qpos),
                time_step=1 / 250,
                planning_time=5,
                # rrt_range=0.05
                # =================== mplib 0.1.1 ===================
                # use_point_cloud=use_point_cloud,
                # use_attach=use_attach,
                # planner_name="RRTConnect"
            )
            now_try_times += 1

        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
        else:
            n_step = result["position"].shape[0]
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_screw(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=False,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        result = self.planner.plan_screw(
            goal_pose=target_pose,
            current_qpos=now_qpos,
            time_step=1 / 250,
            # =================== mplib 0.1.1 ===================
            # use_point_cloud=use_point_cloud,
            # use_attach=use_attach,
        )

        # plan fail
        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
            # return result
        else:
            n_step = result["position"].shape[0]
            # plan step lim
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_path(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=True,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        if self.planner_type == "mplib_RRT":
            result = self.plan_pose(
                now_qpos,
                target_pose,
                use_point_cloud,
                use_attach,
                arms_tag,
                try_times=10,
                log=log,
            )
        elif self.planner_type == "mplib_screw":
            result = self.plan_screw(now_qpos, target_pose, use_point_cloud, use_attach, arms_tag, log)

        return result

    def plan_grippers(self, now_val, target_val):
        num_step = GRIPPER_PLAN_STEPS
        dis_val = target_val - now_val
        per_step = dis_val / num_step
        res = {}
        vals = np.linspace(now_val, target_val, num_step)
        res["num_step"] = num_step
        res["per_step"] = per_step  # dis per step
        res["result"] = vals
        return res
