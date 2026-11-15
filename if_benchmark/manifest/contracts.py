"""Versioned M16 policy and typed public records, independent of frozen M6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from .io import IntegrityError, read
from if_benchmark.schema.serialization import artifact_hash


VERSION = "m16_publication_graph_v1"
DEFAULT_RELEASE = Path("if_benchmark/data/pipeline_runs/release")
DEFAULT_COLLECTION = Path("if_benchmark/data/scene_validation/collection")
DEFAULT_OUTPUT = Path("if_benchmark/data/manifest")
ARTIFACTS = (
    "benchmark_manifest.yml", "derived_contrast_sets.yml", "publication_graph_report.yml",
    "coverage_report.yml", "survivor_bias_report.yml", "train_test_split.yml",
    "scene_counterfactual_bundles.yml", "split_feasibility_report.yml",
    "split_leakage_report.yml", "publish_block_report.yml",
    "input_evidence_index.yml", "task_admission_index.yml",
)


class SkillKey(TypedDict):
    verb: str
    asset: str


class AttemptRef(TypedDict):
    attempt_id: str
    attempt_hash: str
    task_record_id: str
    seed: int
    path: str
    classification: str | None
    qualified_success: bool


class TaskAdmission(TypedDict):
    task_record_id: str
    validated: bool
    status: Literal["validated", "excluded", "not_executed", "blocked"]
    success_count: int
    denominator: int
    attempts: list[AttemptRef]
    reasons: list[str]


@dataclass(frozen=True)
class ManifestConfig:
    version: str = VERSION
    input_policy: str = "frozen_artifacts_local_code_v2"
    identity_policy: str = "structured_goal_checker_skill_same_scene_v1"
    admission_policy: str = "any_complete_alias_two_of_three_all_qualified_success_refs_v1"
    overlap_policy: str = "all_recorded_cross_matches_block_affected_parent_context_v1"
    split_policy: str = "realized_family_atomic_dual_dcode_skill_coverage_v1"
    test_fraction: float = 0.1
    split_seed: int = 0
    split_time_limit_seconds: int = 600
    split_node_limit: int = 100000
    solver_version: str = "1.16.3"

    def validate(self) -> None:
        defaults = ManifestConfig()
        for field in ("version", "input_policy", "identity_policy", "admission_policy", "overlap_policy", "split_policy"):
            if getattr(self, field) != getattr(defaults, field):
                raise IntegrityError(f"unsupported M16 policy: {field}")
        if not 0 < self.test_fraction < 1:
            raise IntegrityError("test_fraction must lie strictly between zero and one")
        if type(self.split_seed) is not int or self.split_time_limit_seconds <= 0 or self.split_node_limit <= 0:
            raise IntegrityError("invalid split seed or solver limits")

    @property
    def hash(self) -> str:
        return artifact_hash(asdict(self))

    @classmethod
    def load(cls, path: Path | None = None) -> "ManifestConfig":
        result = cls(**read(path)) if path is not None else cls()
        result.validate()
        return result


def issue(code: str, *, owner: str = "source", blocking: bool = True, **context: Any) -> dict[str, Any]:
    return {"code": code, "owner": owner, "blocking": blocking, **context}
