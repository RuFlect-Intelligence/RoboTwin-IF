import sapien.core as sapien
import numpy as np
import pdb
from .planner import MplibPlanner
import numpy as np
import toppra as ta
import math
import yaml
import os
import transforms3d as t3d
from copy import deepcopy
import sapien.core as sapien
import envs._GLOBAL_CONFIGS as CONFIGS
from envs.utils import transforms
from .planner import CuroboPlanner
import torch.multiprocessing as mp
from mplib.sapien_utils import SapienPlanningWorld
from mplib import ArticulatedModel, PlanningWorld, Pose as MplibPose
from mplib.collision_detection.fcl import (
    Box,
    BVHModel,
    Capsule,
    CollisionObject,
    Convex,
    Cylinder,
    FCLObject,
    Halfspace,
    Sphere,
)
from mplib.sapien_utils.conversion import (
    convert_object_name,
    export_kinematic_chain_urdf,
    export_srdf,
)
from sapien.physx import (
    PhysxArticulationLinkComponent,
    PhysxCollisionShapeBox,
    PhysxCollisionShapeCapsule,
    PhysxCollisionShapeConvexMesh,
    PhysxCollisionShapeCylinder,
    PhysxCollisionShapePlane,
    PhysxCollisionShapeSphere,
    PhysxCollisionShapeTriangleMesh,
)


def _normalize_curobo_planner_arms(value):
    if value is None:
        return ("left", "right")
    raw = [value] if isinstance(value, str) else list(value)
    arms = tuple(arm for arm in ("left", "right") if arm in set(raw))
    invalid = sorted(set(str(arm) for arm in raw) - {"left", "right"})
    if invalid or not arms:
        raise ValueError(
            f"curobo_planner_arms must contain left/right, got {raw!r}"
        )
    return arms


def _plan_gripper_trajectory(now_val, target_val):
    """Planner-independent equivalent of CuroboPlanner.plan_grippers."""

    num_step = 200
    per_step = (target_val - now_val) / num_step
    return {
        "num_step": num_step,
        "per_step": per_step,
        "result": np.linspace(now_val, target_val, num_step),
    }


def _scaled_mesh_vertices(vertices, scale):
    """Apply one PhysX mesh scale without mutating the simulator shape."""

    points = np.asarray(vertices, dtype=float)
    factors = np.asarray(scale, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("PhysX mesh vertices must have shape (N, 3)")
    if factors.shape != (3,) or not np.isfinite(factors).all():
        raise ValueError("PhysX mesh scale must contain three finite values")
    if np.any(factors <= 0.0):
        raise ValueError("PhysX mesh scale must be positive")
    return points * factors[None, :]


def _convert_physx_component_preserving_scale(component):
    """Convert exact PhysX geometry while preserving mesh-local scale.

    mplib's stock converter asserts that convex meshes have unit scale.  IF
    assets routinely use scaled convex collision meshes, and skipping those
    actors removes the Pick subject from the read-only Pre-Grasp collision
    world.  This equivalent converter bakes the scale into copied vertices;
    it never changes the live SAPIEN component or actor pose.
    """

    shapes = []
    shape_poses = []
    used_scaled_mesh = False
    for shape in component.collision_shapes:
        shape_pose = MplibPose(shape.local_pose)
        if isinstance(shape, PhysxCollisionShapeBox):
            geometry = Box(side=np.asarray(shape.half_size, dtype=float) * 2.0)
        elif isinstance(shape, PhysxCollisionShapeCapsule):
            geometry = Capsule(
                radius=float(shape.radius),
                lz=float(shape.half_length) * 2.0,
            )
            shape_pose *= MplibPose(q=t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0))
        elif isinstance(shape, PhysxCollisionShapeConvexMesh):
            scale = np.asarray(shape.scale, dtype=float)
            vertices = _scaled_mesh_vertices(shape.vertices, scale)
            used_scaled_mesh = used_scaled_mesh or not np.allclose(scale, 1.0)
            geometry = Convex(vertices=vertices, faces=shape.triangles)
        elif isinstance(shape, PhysxCollisionShapeCylinder):
            geometry = Cylinder(
                radius=float(shape.radius),
                lz=float(shape.half_length) * 2.0,
            )
            shape_pose *= MplibPose(q=t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0))
        elif isinstance(shape, PhysxCollisionShapePlane):
            normal = shape_pose.to_transformation_matrix()[:3, 0]
            offset = normal.dot(shape_pose.p)
            geometry = Halfspace(n=normal, d=offset)
            shape_pose = MplibPose()
        elif isinstance(shape, PhysxCollisionShapeSphere):
            geometry = Sphere(radius=float(shape.radius))
        elif isinstance(shape, PhysxCollisionShapeTriangleMesh):
            scale = np.asarray(shape.scale, dtype=float)
            vertices = _scaled_mesh_vertices(shape.vertices, scale)
            used_scaled_mesh = used_scaled_mesh or not np.allclose(scale, 1.0)
            geometry = BVHModel()
            geometry.begin_model()
            geometry.add_sub_model(vertices=vertices, faces=shape.triangles)
            geometry.end_model()
        else:
            raise TypeError(f"Unknown PhysX collision shape: {type(shape)}")
        shapes.append(CollisionObject(geometry))
        shape_poses.append(shape_pose)

    if not shapes:
        return None, used_scaled_mesh
    name = (
        component.name
        if isinstance(component, PhysxArticulationLinkComponent)
        else convert_object_name(component.entity)
    )
    return (
        FCLObject(
            name,
            component.entity.pose,
            shapes,
            shape_poses,
        ),
        used_scaled_mesh,
    )


class Robot:

    def __init__(self, scene, need_topp=False, **kwargs):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging
        self.scene = scene
        self._physical_collision_world = None
        self._physical_collision_world_signature = None
        self._physical_collision_skipped_entities = []
        self.left_planner = None
        self.right_planner = None
        self._init_robot_(scene, need_topp, **kwargs)

    def _init_robot_(self, scene, need_topp=False, **kwargs):
        # self.dual_arm = dual_arm_tag
        # self.plan_success = True

        self.left_js = None
        self.right_js = None

        left_embodiment_args = kwargs["left_embodiment_config"]
        right_embodiment_args = kwargs["right_embodiment_config"]
        left_robot_file = kwargs["left_robot_file"]
        right_robot_file = kwargs["right_robot_file"]

        self.need_topp = need_topp
        self.curobo_planner_arms = _normalize_curobo_planner_arms(
            kwargs.get("curobo_planner_arms")
        )

        self.left_urdf_path = os.path.join(left_robot_file, left_embodiment_args["urdf_path"])
        self.left_srdf_path = left_embodiment_args.get("srdf_path", None)
        self.left_curobo_yml_path = os.path.join(left_robot_file, "curobo.yml")
        if self.left_srdf_path is not None:
            self.left_srdf_path = os.path.join(left_robot_file, self.left_srdf_path)
        self.left_joint_stiffness = left_embodiment_args.get("joint_stiffness", 1000)
        self.left_joint_damping = left_embodiment_args.get("joint_damping", 200)
        self.left_gripper_stiffness = left_embodiment_args.get("gripper_stiffness", 1000)
        self.left_gripper_damping = left_embodiment_args.get("gripper_damping", 200)
        self.left_planner_type = left_embodiment_args.get("planner", "mplib_RRT")
        self.left_move_group = left_embodiment_args["move_group"][0]
        self.left_ee_name = left_embodiment_args["ee_joints"][0]
        self.left_arm_joints_name = left_embodiment_args["arm_joints_name"][0]
        self.left_gripper_name = left_embodiment_args["gripper_name"][0]
        self.left_gripper_bias = left_embodiment_args["gripper_bias"]
        self.left_gripper_scale = left_embodiment_args["gripper_scale"]
        self.left_homestate = left_embodiment_args.get("homestate", [[0] * len(self.left_arm_joints_name)])[0]
        self.left_fix_gripper_name = left_embodiment_args.get("fix_gripper_name", [])
        self.left_delta_matrix = np.array(left_embodiment_args.get("delta_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.left_inv_delta_matrix = np.linalg.inv(self.left_delta_matrix)
        self.left_global_trans_matrix = np.array(
            left_embodiment_args.get("global_trans_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

        _entity_origion_pose = left_embodiment_args.get("robot_pose", [[0, -0.65, 0, 1, 0, 0, 1]])[0]
        _entity_origion_pose = sapien.Pose(_entity_origion_pose[:3], _entity_origion_pose[-4:])
        self.left_entity_origion_pose = deepcopy(_entity_origion_pose)

        self.right_urdf_path = os.path.join(right_robot_file, right_embodiment_args["urdf_path"])
        self.right_srdf_path = right_embodiment_args.get("srdf_path", None)
        if self.right_srdf_path is not None:
            self.right_srdf_path = os.path.join(right_robot_file, self.right_srdf_path)
        self.right_curobo_yml_path = os.path.join(right_robot_file, "curobo.yml")
        self.right_joint_stiffness = right_embodiment_args.get("joint_stiffness", 1000)
        self.right_joint_damping = right_embodiment_args.get("joint_damping", 200)
        self.right_gripper_stiffness = right_embodiment_args.get("gripper_stiffness", 1000)
        self.right_gripper_damping = right_embodiment_args.get("gripper_damping", 200)
        self.right_planner_type = right_embodiment_args.get("planner", "mplib_RRT")
        self.right_move_group = right_embodiment_args["move_group"][1]
        self.right_ee_name = right_embodiment_args["ee_joints"][1]
        self.right_arm_joints_name = right_embodiment_args["arm_joints_name"][1]
        self.right_gripper_name = right_embodiment_args["gripper_name"][1]
        self.right_gripper_bias = right_embodiment_args["gripper_bias"]
        self.right_gripper_scale = right_embodiment_args["gripper_scale"]
        self.right_homestate = right_embodiment_args.get("homestate", [[1] * len(self.right_arm_joints_name)])[1]
        self.right_fix_gripper_name = right_embodiment_args.get("fix_gripper_name", [])
        self.right_delta_matrix = np.array(right_embodiment_args.get("delta_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.right_inv_delta_matrix = np.linalg.inv(self.right_delta_matrix)
        self.right_global_trans_matrix = np.array(
            right_embodiment_args.get("global_trans_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

        _entity_origion_pose = right_embodiment_args.get("robot_pose", [[0, -0.65, 0, 1, 0, 0, 1]])
        _entity_origion_pose = _entity_origion_pose[0 if len(_entity_origion_pose) == 1 else 1]
        _entity_origion_pose = sapien.Pose(_entity_origion_pose[:3], _entity_origion_pose[-4:])
        self.right_entity_origion_pose = deepcopy(_entity_origion_pose)
        self.is_dual_arm = kwargs["dual_arm_embodied"]

        self.left_rotate_lim = left_embodiment_args.get("rotate_lim", [0, 0])
        self.right_rotate_lim = right_embodiment_args.get("rotate_lim", [0, 0])

        self.left_perfect_direction = left_embodiment_args.get("grasp_perfect_direction",
                                                               ["front_right", "front_left"])[0]
        self.right_perfect_direction = right_embodiment_args.get("grasp_perfect_direction",
                                                                 ["front_right", "front_left"])[1]

        if self.is_dual_arm:
            loader: sapien.URDFLoader = scene.create_urdf_loader()
            loader.fix_root_link = True
            self._entity = loader.load(self.left_urdf_path)
            self.left_entity = self._entity
            self.right_entity = self._entity
        else:
            arms_dis = kwargs["embodiment_dis"]
            self.left_entity_origion_pose.p += [-arms_dis / 2, 0, 0]
            self.right_entity_origion_pose.p += [arms_dis / 2, 0, 0]
            left_loader: sapien.URDFLoader = scene.create_urdf_loader()
            left_loader.fix_root_link = True
            right_loader: sapien.URDFLoader = scene.create_urdf_loader()
            right_loader.fix_root_link = True
            self.left_entity = left_loader.load(self.left_urdf_path)
            self.right_entity = right_loader.load(self.right_urdf_path)

        self.left_entity.set_root_pose(self.left_entity_origion_pose)
        self.right_entity.set_root_pose(self.right_entity_origion_pose)

    def reset(self, scene, need_topp=False, **kwargs):
        # ``Base_Task`` creates a fresh SAPIEN scene for every episode even
        # when the Python task/Robot object is reused by a batched worker.
        # Exact-mesh collision validation must therefore follow the new scene
        # and must never retain an FCL world compiled from the prior episode.
        self.scene = scene
        self._physical_collision_world = None
        self._physical_collision_world_signature = None
        self._physical_collision_skipped_entities = []
        # Carry geometry uses the calibrated planner frame only after a
        # physical pick is certified. A reused solver must start every new
        # episode in the unchanged acquisition mode.
        for arm in tuple(getattr(self, "_carry_collision_mode_arms", ())):
            self.set_collision_geometry_mode(arm, "acquisition")
        self._init_robot_(scene, need_topp, **kwargs)

        communication_flag = bool(getattr(self, "communication_flag", False))
        if communication_flag:
            if hasattr(self, "left_conn") and self.left_conn:
                self.left_conn.send({"cmd": "reset"})
                _ = self.left_conn.recv()
            if hasattr(self, "right_conn") and self.right_conn:
                self.right_conn.send({"cmd": "reset"})
                _ = self.right_conn.recv()
        else:
            # A failed first Warp/CuRobo compile may leave this reusable Robot
            # alive without one or both planner attributes. The next episode
            # must rebuild that half-initialized state instead of cascading an
            # AttributeError through the rest of the batch.
            missing_planner = any(
                not isinstance(
                    getattr(self, f"{arm}_planner", None),
                    CuroboPlanner,
                )
                for arm in self.curobo_planner_arms
            )
            if missing_planner:
                self.set_planner(
                    scene=scene,
                    curobo_planner_arms=self.curobo_planner_arms,
                )
            # Match the subprocess planner's ``reset`` command. The Robot and
            # expensive solver allocations survive an episode, but attached
            # object proxies, obstacle worlds and planner RNG state must not.
            for arm in self.curobo_planner_arms:
                getattr(self, f"{arm}_planner").reset_collision_state()

        self.init_joints()

    def get_grasp_perfect_direction(self, arm_tag):
        if arm_tag == "left":
            return self.left_perfect_direction
        elif arm_tag == "right":
            return self.right_perfect_direction

    def create_target_pose_list(self, origin_pose, center_pose, arm_tag=None):
        res_lst = []
        rotate_lim = (self.left_rotate_lim if arm_tag == "left" else self.right_rotate_lim)
        rotate_step = (rotate_lim[1] - rotate_lim[0]) / CONFIGS.ROTATE_NUM
        for i in range(CONFIGS.ROTATE_NUM):
            now_pose = transforms.rotate_along_axis(
                origin_pose,
                center_pose,
                [0, 1, 0],
                rotate_step * i + rotate_lim[0],
                axis_type="target",
                towards=[0, -1, 0],
            )
            res_lst.append(now_pose)
        return res_lst

    def get_constraint_pose(self, ori_vec: list, arm_tag=None):
        inv_delta_matrix = (self.left_inv_delta_matrix if arm_tag == "left" else self.right_inv_delta_matrix)
        return ori_vec[:3] + (ori_vec[-3:] @ np.linalg.inv(inv_delta_matrix)).tolist()

    def init_joints(self):
        if self.left_entity is None or self.right_entity is None:
            raise ValueError("Robote entity is None")

        self.left_active_joints = self.left_entity.get_active_joints()
        self.right_active_joints = self.right_entity.get_active_joints()

        self.left_ee = self.left_entity.find_joint_by_name(self.left_ee_name)
        self.right_ee = self.right_entity.find_joint_by_name(self.right_ee_name)

        self.left_gripper_val = 0.0
        self.right_gripper_val = 0.0

        self.left_arm_joints = [self.left_entity.find_joint_by_name(i) for i in self.left_arm_joints_name]
        self.right_arm_joints = [self.right_entity.find_joint_by_name(i) for i in self.right_arm_joints_name]

        def get_gripper_joints(find, gripper_name: str):
            gripper = [(find(gripper_name["base"]), 1.0, 0.0)]
            for g in gripper_name["mimic"]:
                gripper.append((find(g[0]), g[1], g[2]))
            return gripper

        self.left_gripper = get_gripper_joints(self.left_entity.find_joint_by_name, self.left_gripper_name)
        self.right_gripper = get_gripper_joints(self.right_entity.find_joint_by_name, self.right_gripper_name)
        self.gripper_name = deepcopy(self.left_fix_gripper_name) + deepcopy(self.right_fix_gripper_name)

        for g in self.left_gripper:
            self.gripper_name.append(g[0].child_link.get_name())
        for g in self.right_gripper:
            self.gripper_name.append(g[0].child_link.get_name())

        # camera link id
        self.left_camera = self.left_entity.find_link_by_name("left_camera")
        if self.left_camera is None:
            self.left_camera = self.left_entity.find_link_by_name("camera")
            if self.left_camera is None:
                print("No left camera link")
                self.left_camera = self.left_entity.get_links()[0]

        self.right_camera = self.right_entity.find_link_by_name("right_camera")
        if self.right_camera is None:
            self.right_camera = self.right_entity.find_link_by_name("camera")
            if self.right_camera is None:
                print("No right camera link")
                self.right_camera = self.right_entity.get_links()[0]

        for i, joint in enumerate(self.left_active_joints):
            if joint not in self.left_gripper:
                joint.set_drive_property(stiffness=self.left_joint_stiffness, damping=self.left_joint_damping)
        for i, joint in enumerate(self.right_active_joints):
            if joint not in self.right_gripper:
                joint.set_drive_property(
                    stiffness=self.right_joint_stiffness,
                    damping=self.right_joint_damping,
                )

        for joint in self.left_gripper:
            joint[0].set_drive_property(stiffness=self.left_gripper_stiffness, damping=self.left_gripper_damping)
        for joint in self.right_gripper:
            joint[0].set_drive_property(
                stiffness=self.right_gripper_stiffness,
                damping=self.right_gripper_damping,
            )

    def move_to_homestate(self):
        for i, joint in enumerate(self.left_arm_joints):
            joint.set_drive_target(self.left_homestate[i])

        for i, joint in enumerate(self.right_arm_joints):
            joint.set_drive_target(self.right_homestate[i])

    def set_origin_endpose(self):
        self.left_original_pose = self.get_left_ee_pose()
        self.right_original_pose = self.get_right_ee_pose()

    def print_info(self):
        print(
            "active joints: ",
            [joint.get_name() for joint in self.left_active_joints + self.right_active_joints],
        )
        print(
            "all links: ",
            [link.get_name() for link in self.left_entity.get_links() + self.right_entity.get_links()],
        )
        print("left arm joints: ", [joint.get_name() for joint in self.left_arm_joints])
        print("right arm joints: ", [joint.get_name() for joint in self.right_arm_joints])
        print("left gripper: ", [joint[0].get_name() for joint in self.left_gripper])
        print("right gripper: ", [joint[0].get_name() for joint in self.right_gripper])
        print("left ee: ", self.left_ee.get_name())
        print("right ee: ", self.right_ee.get_name())

    def set_planner(self, scene=None, curobo_planner_arms=None):
        planner_arms = _normalize_curobo_planner_arms(
            self.curobo_planner_arms
            if curobo_planner_arms is None
            else curobo_planner_arms
        )
        self.curobo_planner_arms = planner_arms
        abs_left_curobo_yml_path = os.path.join(CONFIGS.ROOT_PATH, self.left_curobo_yml_path)
        abs_right_curobo_yml_path = os.path.join(CONFIGS.ROOT_PATH, self.right_curobo_yml_path)

        self.communication_flag = (abs_left_curobo_yml_path != abs_right_curobo_yml_path)

        if self.is_dual_arm:
            abs_left_curobo_yml_path = abs_left_curobo_yml_path.replace("curobo.yml", "curobo_left.yml")
            abs_right_curobo_yml_path = abs_right_curobo_yml_path.replace("curobo.yml", "curobo_right.yml")

        if not self.communication_flag:
            if "left" in planner_arms:
                self.left_planner = CuroboPlanner(
                    self.left_entity_origion_pose,
                    self.left_arm_joints_name,
                    [joint.get_name() for joint in self.left_entity.get_active_joints()],
                    yml_path=abs_left_curobo_yml_path,
                )
            if "right" in planner_arms:
                self.right_planner = CuroboPlanner(
                    self.right_entity_origion_pose,
                    self.right_arm_joints_name,
                    [joint.get_name() for joint in self.right_entity.get_active_joints()],
                    yml_path=abs_right_curobo_yml_path,
                )
        else:
            if set(planner_arms) != {"left", "right"}:
                raise ValueError(
                    "selected-arm CuRobo initialization is unsupported for "
                    "separate embodiment planner processes"
                )
            self.left_conn, left_child_conn = mp.Pipe()
            self.right_conn, right_child_conn = mp.Pipe()

            left_args = {
                "origin_pose": self.left_entity_origion_pose,
                "joints_name": self.left_arm_joints_name,
                "all_joints": [joint.get_name() for joint in self.left_entity.get_active_joints()],
                "yml_path": abs_left_curobo_yml_path
            }

            right_args = {
                "origin_pose": self.right_entity_origion_pose,
                "joints_name": self.right_arm_joints_name,
                "all_joints": [joint.get_name() for joint in self.right_entity.get_active_joints()],
                "yml_path": abs_right_curobo_yml_path
            }

            self.left_proc = mp.Process(target=planner_process_worker, args=(left_child_conn, left_args))
            self.right_proc = mp.Process(target=planner_process_worker, args=(right_child_conn, right_args))

            self.left_proc.daemon = True
            self.right_proc.daemon = True

            self.left_proc.start()
            self.right_proc.start()

        if self.need_topp:
            if set(planner_arms) != {"left", "right"}:
                raise ValueError(
                    "selected-arm CuRobo initialization requires need_topp=False"
                )
            self.left_mplib_planner = MplibPlanner(
                self.left_urdf_path,
                self.left_srdf_path,
                self.left_move_group,
                self.left_entity_origion_pose,
                self.left_entity,
                self.left_planner_type,
                scene,
            )
            self.right_mplib_planner = MplibPlanner(
                self.right_urdf_path,
                self.right_srdf_path,
                self.right_move_group,
                self.right_entity_origion_pose,
                self.right_entity,
                self.right_planner_type,
                scene,
            )

    def update_world_pcd(self, world_pcd):
        try:
            for arm in self.curobo_planner_arms:
                planner = getattr(self, f"{arm}_planner")
                planner.update_point_cloud(world_pcd, resolution=0.02)
        except:
            print("Update world pointcloud wrong!")

    def update_world_obstacles(self, obstacles):
        """Strict R02 collision-world refresh; exceptions are never hidden."""

        payload = deepcopy(list(obstacles))
        if self.communication_flag:
            self.left_conn.send(
                {"cmd": "update_world_obstacles", "obstacles": payload}
            )
            left = self.left_conn.recv()
            self.right_conn.send(
                {"cmd": "update_world_obstacles", "obstacles": payload}
            )
            right = self.right_conn.recv()
        else:
            results = {
                arm: getattr(self, f"{arm}_planner").update_world_obstacles(
                    payload
                )
                for arm in self.curobo_planner_arms
            }
            left = results.get("left")
            right = results.get("right")
        for arm, result in (
            (arm, left if arm == "left" else right)
            for arm in self.curobo_planner_arms
        ):
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(
                    f"{arm} collision world update failed: {result['error']}"
                )
            if not isinstance(result, dict) or not result.get("updated"):
                raise RuntimeError(
                    f"{arm} collision world update was not acknowledged: {result!r}"
                )
        return {
            "updated": True,
            "dynamic_obstacle_count": len(payload),
            "left": left,
            "right": right,
        }

    def set_collision_geometry_mode(self, arm_tag, mode, *, table_obstacles=None):
        """Switch one solver's geometry contract, with explicit arm routing."""

        arm = str(arm_tag)
        if arm not in {"left", "right"}:
            raise ValueError(f"invalid arm tag: {arm!r}")
        if mode not in {"acquisition", "m15_carry_conservative_v1"}:
            raise ValueError(f"invalid collision geometry mode: {mode!r}")
        if arm not in set(self.curobo_planner_arms):
            raise RuntimeError(f"{arm} CuRobo planner is not initialized")
        touched = getattr(self, "_carry_collision_mode_arms", None)
        if touched is None:
            touched = self._carry_collision_mode_arms = set()
        # Keep a reset obligation even if an RPC mutates the remote solver
        # and its acknowledgement then fails.
        touched.add(arm)
        options = ({"table_obstacles": deepcopy(list(table_obstacles))}
                   if table_obstacles is not None else {})
        if self.communication_flag:
            connection = self.left_conn if arm == "left" else self.right_conn
            connection.send({"cmd": "set_collision_geometry_mode", "mode": mode,
                             "arm_tag": arm, **options})
            result = connection.recv()
        else:
            result = getattr(self, f"{arm}_planner").set_collision_geometry_mode(
                mode, arm_tag=arm, **options
            )
        if (not isinstance(result, dict) or not result.get("updated")
                or result.get("mode") != mode or result.get("arm_tag") != arm):
            raise RuntimeError(
                f"{arm} collision geometry mode was not acknowledged: {result!r}"
            )
        if mode == "acquisition":
            touched.discard(arm)
        return result

    def update_arm_world_obstacles(self, arm_tag, obstacles):
        """Refresh only the planner that will consume the exact world.

        Planning one arm never queries the opposite planner. Leaving that
        planner's old world in place is safe because its next query must call
        this method again, while avoiding a duplicate CuRobo world rebuild for
        every M12 target.
        """

        arm = str(arm_tag)
        if arm not in {"left", "right"}:
            raise ValueError(f"invalid arm tag: {arm!r}")
        if arm not in set(self.curobo_planner_arms):
            raise RuntimeError(f"{arm} CuRobo planner is not initialized")
        payload = deepcopy(list(obstacles))
        if self.communication_flag:
            connection = self.left_conn if arm == "left" else self.right_conn
            connection.send(
                {"cmd": "update_world_obstacles", "obstacles": payload}
            )
            result = connection.recv()
        else:
            result = getattr(self, f"{arm}_planner").update_world_obstacles(
                payload
            )
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(
                f"{arm} collision world update failed: {result['error']}"
            )
        if not isinstance(result, dict) or not result.get("updated"):
            raise RuntimeError(
                f"{arm} collision world update was not acknowledged: {result!r}"
            )
        return {
            "updated": True,
            "dynamic_obstacle_count": len(payload),
            arm: result,
        }

    def get_collision_state(self, arm_tag):
        if arm_tag not in {"left", "right"}:
            raise ValueError(f"invalid arm tag: {arm_tag!r}")
        if self.communication_flag:
            conn = self.left_conn if arm_tag == "left" else self.right_conn
            conn.send({"cmd": "get_collision_state"})
            result = conn.recv()
        else:
            planner = self.left_planner if arm_tag == "left" else self.right_planner
            result = planner.get_collision_state()
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(
                f"{arm_tag} collision state query failed: {result['error']}"
            )
        return result

    def get_collision_spheres(self, arm_tag, qpos):
        if arm_tag not in {"left", "right"}:
            raise ValueError(f"invalid arm tag: {arm_tag!r}")
        if self.communication_flag:
            conn = self.left_conn if arm_tag == "left" else self.right_conn
            conn.send(
                {
                    "cmd": "fk_collision_spheres",
                    "qpos": np.asarray(qpos, dtype=float).tolist(),
                    "arms_tag": arm_tag,
                }
            )
            result = conn.recv()
        else:
            planner = self.left_planner if arm_tag == "left" else self.right_planner
            result = planner.fk_collision_spheres(qpos, arms_tag=arm_tag)
        if not isinstance(result, list):
            raise RuntimeError(
                f"{arm_tag} collision-sphere FK returned invalid data: {result!r}"
            )
        return result

    def validate_arm_path_collision_world(self, arm_tag, qpos_rows, *, expected_world_hash, held_table_validation=False):
        if arm_tag not in {"left", "right"}:
            raise ValueError(f"invalid arm tag: {arm_tag!r}")
        values = np.asarray(qpos_rows, dtype=float)
        if self.communication_flag:
            conn = self.left_conn if arm_tag == "left" else self.right_conn
            conn.send({"cmd": "validate_path_collision_world", "qpos_rows": values.tolist(),
                       "expected_world_hash": expected_world_hash,
                       **({"held_table_validation": True} if held_table_validation else {})})
            result = conn.recv()
        else:
            planner = self.left_planner if arm_tag == "left" else self.right_planner
            result = planner.validate_path_collision_world(values, expected_world_hash=expected_world_hash,
                **({"held_table_validation": True} if held_table_validation else {}))
        if not isinstance(result, dict) or result.get("error"):
            raise RuntimeError(f"{arm_tag} native collision validation failed: {result!r}")
        return result

    def validate_arm_path_physical_collisions(
        self,
        arm_tag,
        qpos_rows,
        *,
        sample_stride=4,
        excluded_entities=(),
        planning_lock_joint_positions=None,
    ):
        """Check one compiled path against exact SAPIEN collision meshes.

        The independent FCL planning world is read-only with respect to the
        simulator, so this preflight never teleports the physical robot.
        An explicit planner lock map changes only this independent model's
        non-arm joints during validation; its original qpos is always restored.
        """

        from collections.abc import Mapping

        positions = np.asarray(qpos_rows, dtype=float)
        if positions.ndim != 2 or len(positions) == 0:
            raise ValueError("physical collision validation needs a 2D path")
        stride = max(1, int(sample_stride))
        articulations = list(self.scene.get_all_articulations())
        actors = list(self.scene.get_all_actors())
        excluded_ids = {
            id(getattr(value, "actor", value)) for value in excluded_entities
        }

        def pose_signature(pose):
            return tuple(
                np.round(
                    np.concatenate(
                        [
                            np.asarray(pose.p, dtype=float),
                            np.asarray(pose.q, dtype=float),
                        ]
                    ),
                    9,
                ).tolist()
            )

        signature = (
            tuple(sorted(excluded_ids)),
            tuple(
                (
                    id(value),
                    pose_signature(value.root_pose),
                    tuple(np.round(value.qpos, 9).tolist()),
                )
                for value in articulations
            ),
            tuple(
                (id(value), pose_signature(value.pose)) for value in actors
            ),
        )
        if (
            self._physical_collision_world is None
            or self._physical_collision_world_signature != signature
        ):
            planned = {id(self.left_entity), id(self.right_entity)}
            world = PlanningWorld([])
            scaled_entities = set()
            for articulation in articulations:
                if id(articulation) in excluded_ids:
                    continue
                try:
                    collision_links = []
                    for link in articulation.links:
                        value, scaled = _convert_physx_component_preserving_scale(link)
                        if value is not None:
                            collision_links.append(value)
                        if scaled:
                            scaled_entities.add(str(articulation.name))
                    model = ArticulatedModel.create_from_urdf_string(
                        export_kinematic_chain_urdf(articulation),
                        export_srdf(articulation),
                        collision_links=collision_links,
                        gravity=self.scene.get_physx_system().config.gravity,
                        link_names=[link.name for link in articulation.links],
                        joint_names=[
                            joint.name for joint in articulation.active_joints
                        ],
                        verbose=False,
                    )
                except TimeoutError:
                    raise
                except Exception as exc:
                    raise RuntimeError(
                        f"exact collision geometry conversion failed for "
                        f"articulation {articulation.name}: {type(exc).__name__}: {exc}"
                    ) from exc
                model.set_base_pose(articulation.root_pose)
                model.set_qpos(articulation.qpos, full=True)
                world.add_articulation(
                    model,
                    planned=id(articulation) in planned,
                )
            for entity in actors:
                if id(entity) in excluded_ids:
                    continue
                component = next(
                    (
                        value
                        for value in entity.components
                        if hasattr(value, "collision_shapes")
                    ),
                    None,
                )
                if component is None:
                    continue
                try:
                    collision_object, scaled = (
                        _convert_physx_component_preserving_scale(component)
                    )
                except TimeoutError:
                    raise
                except Exception as exc:
                    raise RuntimeError(
                        f"exact collision geometry conversion failed for actor "
                        f"{entity.name}: {type(exc).__name__}: {exc}"
                    ) from exc
                if scaled:
                    scaled_entities.add(str(entity.name))
                if collision_object is not None:
                    world.add_object(collision_object)
            self._physical_collision_world = world
            self._physical_collision_skipped_entities = []
            self._physical_collision_scaled_entities = sorted(scaled_entities)
            self._physical_collision_world_signature = signature

        world = self._physical_collision_world
        arm_names = (
            self.left_arm_joints_name
            if str(arm_tag) == "left"
            else self.right_arm_joints_name
        )
        model = next(
            (
                value
                for value in world.get_planned_articulations()
                if set(arm_names).issubset(set(value.get_user_joint_names()))
            ),
            None,
        )
        if model is None:
            raise RuntimeError(
                f"physical collision world has no {arm_tag} articulation"
            )
        joint_names = list(model.get_user_joint_names())
        joint_indices = [joint_names.index(name) for name in arm_names]
        if positions.shape[1] != len(joint_indices):
            raise ValueError(
                "compiled arm path does not match physical arm joint count"
            )
        original_qpos = np.asarray(model.get_qpos(), dtype=float).copy()
        validation_base_qpos = original_qpos.copy()
        lock_joint_evidence = {}
        if planning_lock_joint_positions is not None:
            if not isinstance(planning_lock_joint_positions, Mapping):
                raise ValueError("planning lock joint positions must be a mapping")
            lock_positions = {}
            for name, value in planning_lock_joint_positions.items():
                if not isinstance(name, str) or name not in joint_names:
                    raise ValueError(f"unknown planning lock joint: {name!r}")
                if name in arm_names:
                    raise ValueError(f"planning lock joint conflicts with active arm: {name!r}")
                position = np.asarray(value, dtype=float)
                if position.ndim != 0 or not np.isfinite(position):
                    raise ValueError(f"planning lock joint position must be a finite scalar: {name!r}")
                lock_positions[name] = float(position)
            for name, value in lock_positions.items():
                validation_base_qpos[joint_names.index(name)] = value
            lock_joint_evidence["planning_lock_joint_positions"] = lock_positions
        sample_indices = list(range(0, len(positions), stride))
        if sample_indices[-1] != len(positions) - 1:
            sample_indices.append(len(positions) - 1)
        try:
            for checked_count, waypoint_index in enumerate(sample_indices, 1):
                qpos = validation_base_qpos.copy()
                qpos[joint_indices] = positions[waypoint_index]
                model.set_qpos(qpos, full=True)
                collisions = list(world.check_collision())
                if collisions:
                    return {
                        "collision_free": False,
                        "validator": "sapien_exact_mesh_fcl_v1",
                        "sample_stride": stride,
                        "excluded_entity_count": len(excluded_ids),
                        "skipped_scaled_entities": list(
                            self._physical_collision_skipped_entities
                        ),
                        "converted_scaled_entities": list(
                            self._physical_collision_scaled_entities
                        ),
                        "checked_waypoint_count": checked_count,
                        **lock_joint_evidence,
                        "first_collision_waypoint_index": int(waypoint_index),
                        "collisions": [
                            {
                                "collision_type": str(value.collision_type),
                                "object_name1": str(value.object_name1),
                                "object_name2": str(value.object_name2),
                                "link_name1": str(value.link_name1),
                                "link_name2": str(value.link_name2),
                            }
                            for value in collisions
                        ],
                    }
        finally:
            model.set_qpos(original_qpos, full=True)
        return {
            "collision_free": True,
            "validator": "sapien_exact_mesh_fcl_v1",
            "sample_stride": stride,
            "excluded_entity_count": len(excluded_ids),
            "skipped_scaled_entities": list(
                self._physical_collision_skipped_entities
            ),
            "converted_scaled_entities": list(
                self._physical_collision_scaled_entities
            ),
            "checked_waypoint_count": len(sample_indices),
            **lock_joint_evidence,
            "first_collision_waypoint_index": None,
            "collisions": [],
        }

    def attach_collision_object(
        self,
        arm_tag,
        descriptor,
        *,
        last_qpos=None,
    ):
        if arm_tag not in {"left", "right"}:
            raise ValueError(f"invalid arm tag: {arm_tag!r}")
        entity = self.left_entity if arm_tag == "left" else self.right_entity
        qpos = entity.get_qpos() if last_qpos is None else deepcopy(last_qpos)
        descriptor = deepcopy(descriptor)
        if "collision_pieces" in descriptor and "dimension_envelope_m" in descriptor:
            from if_benchmark.grasp_validation.held_camera import live_camera_descriptor
            descriptor["fixed_camera"] = live_camera_descriptor(self, arm_tag)
        if self.communication_flag:
            conn = self.left_conn if arm_tag == "left" else self.right_conn
            conn.send(
                {
                    "cmd": "attach_collision_object",
                    "qpos": qpos,
                    "descriptor": deepcopy(descriptor),
                }
            )
            result = conn.recv()
        else:
            planner = self.left_planner if arm_tag == "left" else self.right_planner
            result = planner.attach_collision_object(qpos, deepcopy(descriptor))
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(
                f"{arm_tag} attached-object update failed: {result['error']}"
            )
        if not isinstance(result, dict) or not result.get("attached"):
            raise RuntimeError(
                f"{arm_tag} attached-object update was not acknowledged: {result!r}"
            )
        return result

    def observe_held_camera_pair(self, arm_tag, qpos, descriptor, *, expected_attachment_hash):
        from if_benchmark.grasp_validation.held_camera import live_camera_descriptor
        descriptor = deepcopy(descriptor)
        descriptor["fixed_camera"] = live_camera_descriptor(self, arm_tag)
        options = {"expected_attachment_hash": expected_attachment_hash}
        if self.communication_flag:
            conn = self.left_conn if arm_tag == "left" else self.right_conn
            conn.send({"cmd": "observe_held_camera_pair", "qpos": np.asarray(qpos).tolist(),
                       "descriptor": descriptor, **options})
            result = conn.recv()
        else:
            planner = self.left_planner if arm_tag == "left" else self.right_planner
            result = planner.observe_held_camera_pair(qpos, descriptor, **options)
        if not isinstance(result, dict) or result.get("error"):
            raise RuntimeError(f"{arm_tag} held camera observation failed: {result!r}")
        return result

    def detach_collision_object(self, arm_tag):
        if arm_tag not in {"left", "right"}:
            raise ValueError(f"invalid arm tag: {arm_tag!r}")
        if self.communication_flag:
            conn = self.left_conn if arm_tag == "left" else self.right_conn
            conn.send({"cmd": "detach_collision_object"})
            result = conn.recv()
        else:
            planner = self.left_planner if arm_tag == "left" else self.right_planner
            result = planner.detach_collision_object()
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(
                f"{arm_tag} attached-object detach failed: {result['error']}"
            )
        return result

    def solve_ik_batch(self, arm_tag, target_pose_list):
        if arm_tag not in {"left", "right"}:
            raise ValueError(f"invalid arm tag: {arm_tag!r}")
        transformed = [
            self._trans_from_gripper_to_endlink(deepcopy(pose), arm_tag=arm_tag)
            for pose in target_pose_list
        ]
        if self.communication_flag:
            conn = self.left_conn if arm_tag == "left" else self.right_conn
            conn.send(
                {
                    "cmd": "solve_ik_batch",
                    "target_pose_list": transformed,
                    "arms_tag": arm_tag,
                }
            )
            result = conn.recv()
        else:
            planner = self.left_planner if arm_tag == "left" else self.right_planner
            result = planner.solve_ik_batch(transformed, arms_tag=arm_tag)
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(f"{arm_tag} IK batch failed: {result['error']}")
        return result

    def _trans_from_gripper_to_endlink(self, target_pose, arm_tag=None):
        gripper_bias = (self.left_gripper_bias if arm_tag == "left" else self.right_gripper_bias)
        inv_delta_matrix = (self.left_inv_delta_matrix if arm_tag == "left" else self.right_inv_delta_matrix)
        target_pose_arr = np.array(target_pose)
        gripper_pose_pos, gripper_pose_quat = deepcopy(target_pose_arr[0:3]), deepcopy(target_pose_arr[-4:])
        gripper_pose_mat = t3d.quaternions.quat2mat(gripper_pose_quat)
        gripper_pose_pos += gripper_pose_mat @ np.array([0.12 - gripper_bias, 0, 0]).T
        gripper_pose_mat = gripper_pose_mat @ inv_delta_matrix
        gripper_pose_quat = t3d.quaternions.mat2quat(gripper_pose_mat)
        return sapien.Pose(gripper_pose_pos, gripper_pose_quat)

    def left_plan_grippers(self, now_val, target_val):
        if self.communication_flag:
            self.left_conn.send({"cmd": "plan_grippers", "now_val": now_val, "target_val": target_val})
            return self.left_conn.recv()
        else:
            if self.left_planner is not None:
                return self.left_planner.plan_grippers(now_val, target_val)
            return _plan_gripper_trajectory(now_val, target_val)

    def right_plan_grippers(self, now_val, target_val):
        if self.communication_flag:
            self.right_conn.send({"cmd": "plan_grippers", "now_val": now_val, "target_val": target_val})
            return self.right_conn.recv()
        else:
            if self.right_planner is not None:
                return self.right_planner.plan_grippers(now_val, target_val)
            return _plan_gripper_trajectory(now_val, target_val)

    def left_plan_multi_path(
        self,
        target_lst,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
        expected_world_hash=None,
        expected_attachment_hash=None,
        candidate_search_budget=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="left")
        if last_qpos is None:
            now_qpos = self.left_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)
        target_lst_copy = deepcopy(target_lst)
        for i in range(len(target_lst_copy)):
            target_lst_copy[i] = self._trans_from_gripper_to_endlink(target_lst_copy[i], arm_tag="left")

        if self.communication_flag:
            self.left_conn.send({
                "cmd": "plan_batch",
                "qpos": now_qpos,
                "target_pose_list": target_lst_copy,
                "constraint_pose": constraint_pose,
                "arms_tag": "left",
                "expected_world_hash": expected_world_hash,
                "expected_attachment_hash": expected_attachment_hash,
                **({"candidate_search_budget": candidate_search_budget}
                   if candidate_search_budget is not None else {}),
            })
            return self.left_conn.recv()
        else:
            return self.left_planner.plan_batch(
                now_qpos,
                target_lst_copy,
                constraint_pose=constraint_pose,
                arms_tag="left",
                expected_world_hash=expected_world_hash,
                expected_attachment_hash=expected_attachment_hash,
                **({"candidate_search_budget": candidate_search_budget}
                   if candidate_search_budget is not None else {}),
            )

    def right_plan_multi_path(
        self,
        target_lst,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
        expected_world_hash=None,
        expected_attachment_hash=None,
        candidate_search_budget=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="right")
        if last_qpos is None:
            now_qpos = self.right_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)
        target_lst_copy = deepcopy(target_lst)
        for i in range(len(target_lst_copy)):
            target_lst_copy[i] = self._trans_from_gripper_to_endlink(target_lst_copy[i], arm_tag="right")

        if self.communication_flag:
            self.right_conn.send({
                "cmd": "plan_batch",
                "qpos": now_qpos,
                "target_pose_list": target_lst_copy,
                "constraint_pose": constraint_pose,
                "arms_tag": "right",
                "expected_world_hash": expected_world_hash,
                "expected_attachment_hash": expected_attachment_hash,
                **({"candidate_search_budget": candidate_search_budget}
                   if candidate_search_budget is not None else {}),
            })
            return self.right_conn.recv()
        else:
            return self.right_planner.plan_batch(
                now_qpos,
                target_lst_copy,
                constraint_pose=constraint_pose,
                arms_tag="right",
                expected_world_hash=expected_world_hash,
                expected_attachment_hash=expected_attachment_hash,
                **({"candidate_search_budget": candidate_search_budget}
                   if candidate_search_budget is not None else {}),
            )

    def left_plan_path(
        self,
        target_pose,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
        expected_world_hash=None,
        expected_attachment_hash=None,
        m15_exact_table_validation=False,
        m15_route_lift_support=False,
        m15_route_acquisition_clearance_m=None,
        m15_carry_cartesian_motion=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="left")
        if last_qpos is None:
            now_qpos = self.left_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)

        trans_target_pose = self._trans_from_gripper_to_endlink(target_pose, arm_tag="left")

        if self.communication_flag:
            self.left_conn.send({
                "cmd": "plan_path",
                "qpos": now_qpos,
                "target_pose": trans_target_pose,
                "constraint_pose": constraint_pose,
                "arms_tag": "left",
                "expected_world_hash": expected_world_hash,
                "expected_attachment_hash": expected_attachment_hash,
                **({"m15_exact_table_validation": True} if m15_exact_table_validation else {}),
                **({"m15_route_lift_support": True} if m15_route_lift_support else {}),
                **({"m15_route_acquisition_clearance_m": m15_route_acquisition_clearance_m}
                   if m15_route_acquisition_clearance_m is not None else {}),
                **({"m15_carry_cartesian_motion": m15_carry_cartesian_motion}
                   if m15_carry_cartesian_motion is not None else {}),
            })
            return self.left_conn.recv()
        else:
            return self.left_planner.plan_path(
                now_qpos,
                trans_target_pose,
                constraint_pose=constraint_pose,
                arms_tag="left",
                expected_world_hash=expected_world_hash,
                expected_attachment_hash=expected_attachment_hash,
                **({"m15_exact_table_validation": True} if m15_exact_table_validation else {}),
                **({"m15_route_lift_support": True} if m15_route_lift_support else {}),
                **({"m15_route_acquisition_clearance_m": m15_route_acquisition_clearance_m}
                   if m15_route_acquisition_clearance_m is not None else {}),
                **({"m15_carry_cartesian_motion": m15_carry_cartesian_motion}
                   if m15_carry_cartesian_motion is not None else {}),
            )

    def right_plan_path(
        self,
        target_pose,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
        expected_world_hash=None,
        expected_attachment_hash=None,
        m15_exact_table_validation=False,
        m15_route_lift_support=False,
        m15_route_acquisition_clearance_m=None,
        m15_carry_cartesian_motion=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="right")
        if last_qpos is None:
            now_qpos = self.right_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)

        trans_target_pose = self._trans_from_gripper_to_endlink(target_pose, arm_tag="right")

        if self.communication_flag:
            self.right_conn.send({
                "cmd": "plan_path",
                "qpos": now_qpos,
                "target_pose": trans_target_pose,
                "constraint_pose": constraint_pose,
                "arms_tag": "right",
                "expected_world_hash": expected_world_hash,
                "expected_attachment_hash": expected_attachment_hash,
                **({"m15_exact_table_validation": True} if m15_exact_table_validation else {}),
                **({"m15_route_lift_support": True} if m15_route_lift_support else {}),
                **({"m15_route_acquisition_clearance_m": m15_route_acquisition_clearance_m}
                   if m15_route_acquisition_clearance_m is not None else {}),
                **({"m15_carry_cartesian_motion": m15_carry_cartesian_motion}
                   if m15_carry_cartesian_motion is not None else {}),
            })
            return self.right_conn.recv()
        else:
            return self.right_planner.plan_path(
                now_qpos,
                trans_target_pose,
                constraint_pose=constraint_pose,
                arms_tag="right",
                expected_world_hash=expected_world_hash,
                expected_attachment_hash=expected_attachment_hash,
                **({"m15_exact_table_validation": True} if m15_exact_table_validation else {}),
                **({"m15_route_lift_support": True} if m15_route_lift_support else {}),
                **({"m15_route_acquisition_clearance_m": m15_route_acquisition_clearance_m}
                   if m15_route_acquisition_clearance_m is not None else {}),
                **({"m15_carry_cartesian_motion": m15_carry_cartesian_motion}
                   if m15_carry_cartesian_motion is not None else {}),
            )

    # The data of gripper has been normalized
    def get_left_arm_jointState(self) -> list:
        jointState_list = []
        for joint in self.left_arm_joints:
            jointState_list.append(joint.get_drive_target()[0].astype(float))
        jointState_list.append(self.get_left_gripper_val())
        return jointState_list

    def get_right_arm_jointState(self) -> list:
        jointState_list = []
        for joint in self.right_arm_joints:
            jointState_list.append(joint.get_drive_target()[0].astype(float))
        jointState_list.append(self.get_right_gripper_val())
        return jointState_list

    def get_left_arm_real_jointState(self) -> list:
        jointState_list = []
        left_joints_qpos = self.left_entity.get_qpos()
        left_active_joints = self.left_entity.get_active_joints()
        for joint in self.left_arm_joints:
            jointState_list.append(left_joints_qpos[left_active_joints.index(joint)])
        jointState_list.append(self.get_left_gripper_val())
        return jointState_list

    def get_right_arm_real_jointState(self) -> list:
        jointState_list = []
        right_joints_qpos = self.right_entity.get_qpos()
        right_active_joints = self.right_entity.get_active_joints()
        for joint in self.right_arm_joints:
            jointState_list.append(right_joints_qpos[right_active_joints.index(joint)])
        jointState_list.append(self.get_right_gripper_val())
        return jointState_list

    def get_left_gripper_val(self):
        if None in self.left_gripper:
            print("No gripper")
            return 0
        return self.left_gripper_val

    def get_right_gripper_val(self):
        if None in self.right_gripper:
            print("No gripper")
            return 0
        return self.right_gripper_val

    def is_left_gripper_open(self):
        return self.left_gripper_val > 0.8

    def is_right_gripper_open(self):
        return self.right_gripper_val > 0.8

    def is_left_gripper_open_half(self):
        return self.left_gripper_val > 0.45

    def is_right_gripper_open_half(self):
        return self.right_gripper_val > 0.45

    def is_left_gripper_close(self):
        return self.left_gripper_val < 0.2

    def is_right_gripper_close(self):
        return self.right_gripper_val < 0.2

    # get move group joint pose
    def get_left_ee_pose(self):
        return self._trans_endpose(arm_tag="left", is_endpose=False)

    def get_right_ee_pose(self):
        return self._trans_endpose(arm_tag="right", is_endpose=False)

    # get gripper centor pose
    def get_left_tcp_pose(self):
        return self._trans_endpose(arm_tag="left", is_endpose=True)

    def get_right_tcp_pose(self):
        return self._trans_endpose(arm_tag="right", is_endpose=True)

    def get_left_orig_endpose(self):
        pose = self.left_ee.global_pose
        global_trans_matrix = self.left_global_trans_matrix
        pose.p = pose.p - self.left_entity_origion_pose.p
        pose.p = t3d.quaternions.quat2mat(self.left_entity_origion_pose.q).T @ pose.p
        return (pose.p.tolist() + t3d.quaternions.mat2quat(
            t3d.quaternions.quat2mat(self.left_entity_origion_pose.q).T @ t3d.quaternions.quat2mat(pose.q)
            @ global_trans_matrix).tolist())

    def get_right_orig_endpose(self):
        pose = self.right_ee.global_pose
        global_trans_matrix = self.right_global_trans_matrix
        pose.p = pose.p - self.right_entity_origion_pose.p
        pose.p = t3d.quaternions.quat2mat(self.right_entity_origion_pose.q).T @ pose.p
        return (pose.p.tolist() + t3d.quaternions.mat2quat(
            t3d.quaternions.quat2mat(self.right_entity_origion_pose.q).T @ t3d.quaternions.quat2mat(pose.q)
            @ global_trans_matrix).tolist())

    def _trans_endpose(self, arm_tag=None, is_endpose=False):
        if arm_tag is None:
            print("No arm tag")
            return
        gripper_bias = (self.left_gripper_bias if arm_tag == "left" else self.right_gripper_bias)
        global_trans_matrix = (self.left_global_trans_matrix if arm_tag == "left" else self.right_global_trans_matrix)
        delta_matrix = (self.left_delta_matrix if arm_tag == "left" else self.right_delta_matrix)
        ee_pose = (self.left_ee.global_pose if arm_tag == "left" else self.right_ee.global_pose)
        endpose_arr = np.eye(4)
        endpose_arr[:3, :3] = (t3d.quaternions.quat2mat(ee_pose.q) @ global_trans_matrix @ delta_matrix)
        dis = gripper_bias
        if is_endpose == False:
            dis -= 0.12
        endpose_arr[:3, 3] = ee_pose.p + endpose_arr[:3, :3] @ np.array([dis, 0, 0]).T
        res = (endpose_arr[:3, 3].tolist() + t3d.quaternions.mat2quat(endpose_arr[:3, :3]).tolist())
        return res

    def _entity_qf(self, entity):
        qf = entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        entity.set_qf(qf)

    def set_arm_joints(self, target_position, target_velocity, arm_tag):
        self._entity_qf(self.left_entity)
        self._entity_qf(self.right_entity)

        joint_lst = self.left_arm_joints if arm_tag == "left" else self.right_arm_joints
        for j in range(len(joint_lst)):
            joint = joint_lst[j]
            joint.set_drive_target(target_position[j])
            joint.set_drive_velocity_target(target_velocity[j])

    def get_normal_real_gripper_val(self):
        normal_left_gripper_val = (self.left_gripper[0][0].get_drive_target()[0] - self.left_gripper_scale[0]) / (
            self.left_gripper_scale[1] - self.left_gripper_scale[0])
        normal_right_gripper_val = (self.right_gripper[0][0].get_drive_target()[0] - self.right_gripper_scale[0]) / (
            self.right_gripper_scale[1] - self.right_gripper_scale[0])
        normal_left_gripper_val = np.clip(normal_left_gripper_val, 0, 1)
        normal_right_gripper_val = np.clip(normal_right_gripper_val, 0, 1)
        return [normal_left_gripper_val, normal_right_gripper_val]

    def set_gripper(self, gripper_val, arm_tag, gripper_eps=0.1):  # gripper_val in [0,1]
        self._entity_qf(self.left_entity)
        self._entity_qf(self.right_entity)
        gripper_val = np.clip(gripper_val, 0, 1)

        if arm_tag == "left":
            joints = self.left_gripper
            self.left_gripper_val = gripper_val
            gripper_scale = self.left_gripper_scale
            real_gripper_val = self.get_normal_real_gripper_val()[0]
        else:
            joints = self.right_gripper
            self.right_gripper_val = gripper_val
            gripper_scale = self.right_gripper_scale
            real_gripper_val = self.get_normal_real_gripper_val()[1]

        if not joints:
            print("No gripper")
            return

        if (gripper_val - real_gripper_val > gripper_eps
                and gripper_eps > 0) or (gripper_val - real_gripper_val < gripper_eps and gripper_eps < 0):
            gripper_val = real_gripper_val + gripper_eps  # TODO

        real_gripper_val = gripper_scale[0] + gripper_val * (gripper_scale[1] - gripper_scale[0])

        for joint in joints:
            real_joint: sapien.physx.PhysxArticulationJoint = joint[0]
            drive_target = real_gripper_val * joint[1] + joint[2]
            drive_velocity_target = (np.clip(drive_target - real_joint.drive_target, -1.0, 1.0) * 0.05)
            real_joint.set_drive_target(drive_target)
            real_joint.set_drive_velocity_target(drive_velocity_target)


def planner_process_worker(conn, args):
    import os
    from .planner import CuroboPlanner  # 或者绝对路径导入

    planner = CuroboPlanner(args["origin_pose"], args["joints_name"], args["all_joints"], yml_path=args["yml_path"])

    while True:
        try:
            msg = conn.recv()
            if msg["cmd"] == "plan_path":
                result = planner.plan_path(
                    msg["qpos"],
                    msg["target_pose"],
                    constraint_pose=msg.get("constraint_pose", None),
                    arms_tag=msg["arms_tag"],
                    expected_world_hash=msg.get("expected_world_hash"),
                    expected_attachment_hash=msg.get("expected_attachment_hash"),
                    **({"m15_exact_table_validation": True} if msg.get("m15_exact_table_validation") else {}),
                    **({"m15_route_lift_support": True} if msg.get("m15_route_lift_support") else {}),
                    **({"m15_route_acquisition_clearance_m": msg["m15_route_acquisition_clearance_m"]}
                       if msg.get("m15_route_acquisition_clearance_m") is not None else {}),
                    **({"m15_carry_cartesian_motion": msg["m15_carry_cartesian_motion"]}
                       if msg.get("m15_carry_cartesian_motion") is not None else {}),
                )
                conn.send(result)

            elif msg["cmd"] == "plan_batch":
                result = planner.plan_batch(
                    msg["qpos"],
                    msg["target_pose_list"],
                    constraint_pose=msg.get("constraint_pose", None),
                    arms_tag=msg["arms_tag"],
                    expected_world_hash=msg.get("expected_world_hash"),
                    expected_attachment_hash=msg.get("expected_attachment_hash"),
                    **({"candidate_search_budget": msg["candidate_search_budget"]}
                       if msg.get("candidate_search_budget") is not None else {}),
                )
                conn.send(result)

            elif msg["cmd"] == "plan_grippers":
                result = planner.plan_grippers(
                    msg["now_val"],
                    msg["target_val"],
                )
                conn.send(result)

            elif msg["cmd"] == "update_point_cloud":
                planner.update_point_cloud(msg["pcd"], resolution=msg.get("resolution", 0.02))
                conn.send("ok")

            elif msg["cmd"] == "update_world_obstacles":
                conn.send(
                    planner.update_world_obstacles(msg.get("obstacles") or [])
                )

            elif msg["cmd"] == "set_collision_geometry_mode":
                conn.send(planner.set_collision_geometry_mode(
                    msg["mode"], arm_tag=msg["arm_tag"],
                    **({"table_obstacles": msg["table_obstacles"]}
                       if "table_obstacles" in msg else {})
                ))

            elif msg["cmd"] == "get_collision_state":
                conn.send(planner.get_collision_state())

            elif msg["cmd"] == "validate_path_collision_world":
                conn.send(planner.validate_path_collision_world(
                    msg["qpos_rows"], expected_world_hash=msg["expected_world_hash"],
                    **({"held_table_validation": True} if msg.get("held_table_validation") else {})
                ))

            elif msg["cmd"] == "fk_collision_spheres":
                conn.send(
                    planner.fk_collision_spheres(
                        msg["qpos"], arms_tag=msg["arms_tag"]
                    )
                )

            elif msg["cmd"] == "observe_held_camera_pair":
                conn.send(planner.observe_held_camera_pair(
                    msg["qpos"], msg["descriptor"], expected_attachment_hash=msg["expected_attachment_hash"]))
            elif msg["cmd"] == "attach_collision_object":
                conn.send(
                    planner.attach_collision_object(
                        msg["qpos"], msg["descriptor"]
                    )
                )

            elif msg["cmd"] == "detach_collision_object":
                conn.send(planner.detach_collision_object())

            elif msg["cmd"] == "solve_ik_batch":
                conn.send(
                    planner.solve_ik_batch(
                        msg["target_pose_list"], arms_tag=msg["arms_tag"]
                    )
                )

            elif msg["cmd"] == "reset":
                conn.send(planner.reset_collision_state())

            elif msg["cmd"] == "exit":
                conn.close()
                break

            else:
                conn.send({"error": f"Unknown command {msg['cmd']}"})

        except EOFError:
            break
        except Exception as e:
            conn.send({"error": str(e)})
