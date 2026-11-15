"""Versioned, simulation-independent evaluation contracts."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

from if_benchmark.manifest.io import file_hash, read, seal, verify
from if_benchmark.schema.serialization import artifact_hash

VERSION = "m17_shared_initial_rollout_v1"
AXES = ("D1a", "D1b", "D1c", "D4")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "if_benchmark/data/cutover"


@dataclass(frozen=True)
class EvaluationPolicy:
    version: str = VERSION
    seeds: tuple = (0, 1, 2)
    success_seconds: float = 60.0
    pick_seconds: float = 30.0
    arm_seconds: float = 2.0
    max_actions: int = 1000
    success_check_steps: int = 20
    contact_impulse: float = 1e-6
    contact_steps: int = 2
    arm_activity: float = 0.005
    arm_dominance: float = 2.0
    gripper_in_arm_activity: bool = True
    waypoint_hz: int = 20
    render_updates: str = "observation_boundary_only_v1"
    runtime_reset: str = "fresh_scene_cached_planners_v1"

    def validate(self):
        if self.version != VERSION or tuple(self.seeds) != (0, 1, 2):
            raise ValueError("unsupported evaluation version/seeds")
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"invalid evaluation policy: {key}")
        if self.success_seconds < max(self.pick_seconds, self.arm_seconds):
            raise ValueError("success horizon must contain both intent windows")
        if self.arm_dominance <= 1 or not self.gripper_in_arm_activity:
            raise ValueError("unsupported D4 policy")
        if self.render_updates != "observation_boundary_only_v1":
            raise ValueError("unsupported render update policy")
        if self.runtime_reset != "fresh_scene_cached_planners_v1":
            raise ValueError("unsupported runtime reset policy")

    @property
    def payload(self):
        self.validate()
        return seal({"artifact_kind": "m17_evaluation_policy", **asdict(self)})


def implementation_identity():
    from if_benchmark.manifest.core import _code_hash
    paths = [*Path(__file__).parent.glob("*.py"),
             ROOT / "script/benchmark_v1/m17_env.sh",
             ROOT / "envs/if_scene_validation.py", ROOT / "envs/_base_task.py", ROOT / "envs/camera/camera.py",
             * (ROOT / "envs/robot").glob("*.py"),
             * (ROOT / "if_benchmark/success_contract").glob("*.py"),
             ROOT / "if_benchmark/data/success_contract/success_contract_v1.yml",
             ROOT / "if_benchmark/data/success_contract/embodiment_profiles/aloha_agilex_gripper_v1.yml"]
    paths.extend(ROOT / "task_config" / name for name in
                 ("demo_clean.yml", "_camera_config.yml", "_embodiment_config.yml"))
    embodiment = ROOT / "assets/embodiments/aloha-agilex"
    paths.extend(embodiment.glob("*.yml"))
    paths.extend((embodiment / "urdf").glob("*.urdf"))
    paths.extend((embodiment / "srdf").glob("*.srdf"))
    existing = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths) if p.is_file()}
    existing["m16_semantic_implementation"] = _code_hash()
    from if_benchmark.manifest.reader_compatibility import evaluation_identity_files
    return artifact_hash(evaluation_identity_files(existing))


def load_plan(path):
    plan = read(Path(path)); verify(plan)
    if plan.get("version") != VERSION:
        raise ValueError("unsupported evaluation plan")
    verify(plan["policy"])
    if plan["implementation_hash"] != implementation_identity():
        raise ValueError("evaluation implementation changed; regenerate plan")
    return plan
