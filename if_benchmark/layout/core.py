"""Milestone 12 deterministic constructive layout solver and artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import inspect
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import yaml

from if_benchmark.collision_geometry import frozen_support_plane_z
from if_benchmark.layout.cuda_devices import normalize_cuda_device_selectors
from if_benchmark.layout.search_budget import (
    FamilyLiveSearchBudget,
    LIVE_SEED_BUDGET_EXHAUSTED,
    contains_limited_planning_failure,
    failure_summary_counts,
    normalize_search_budget,
)
from if_benchmark.layout.search_budget_compatibility import equivalent_attempt_identity
from if_benchmark.schema.cli import validate_document
from if_benchmark.schema.serialization import (
    artifact_hash,
    build_provenance,
    canonical_data,
    deterministic_id,
)
from if_benchmark.grasp_validation.contact_contract import EXACT_GRASP_POLICY
from if_benchmark.grasp_validation.planning_contract import (
    DEFAULT_PICK_PRE_GRASP_M,
    PickPlanningProviderError,
    build_pick_planning_request,
    classify_pick_planning_failure,
    next_non_d4_pick_arm_assignment,
    pick_planning_identity,
    single_arm_pick_requirement_groups,
    validate_pick_planning_witness,
)
from if_benchmark.success_contract.contract import SuccessContract
from if_benchmark.success_contract.relation import (
    RelationContract,
    static_relation_geometry_evidence,
)
from if_benchmark.articulation_capability import (
    ARTICULATION_CAPABILITY_VERSION,
    articulation_capability_errors,
)
from if_benchmark.touch_capability import (
    TOUCH_CAPABILITY_VERSION,
    touch_capability_errors,
)
from if_benchmark.orientation_adjustment import (
    normalize_orientation_adjustment_params,
    target_quaternion_wxyz,
)
from if_benchmark.units.core import CONFIG as UNIT_BUILD_CONFIG
from if_benchmark.units.core import UNIT_BUILD_VERSION
from if_benchmark.units.core import (
    m4_articulation_evidence_required_entities,
    m4_touch_evidence_required_entities,
    required_operation_anchor_kinds,
)


SCHEMA_VERSION = 1
LAYOUT_SOLVER_VERSION = 24
PUBLICATION_POLICY = "validated_solved_subset_with_audited_unsat_v1"
UNSAT_ATTEMPT_CACHE_VERSION = "m12_deterministic_unsat_attempt_cache_v4"
GEOMETRY_EPSILON = 1e-10
SUCCESS_CONTRACT = SuccessContract.load()
RELATION_CONTRACT = RelationContract.from_success_contract(SUCCESS_CONTRACT)

DEFAULT_REALIZATION_DIR = Path(
    "if_benchmark/data/pipeline_runs/release/m10_realized_families"
)
DEFAULT_UNIT_DIR = Path(
    "if_benchmark/data/pipeline_runs/release/m11_units"
)
DEFAULT_CONFIG_PATH = Path("if_benchmark/data/layout/layout_config.yml")
DEFAULT_OUTPUT_DIR = Path(
    "if_benchmark/data/pipeline_runs/release/m12_scene_specs"
)
DEFAULT_PROCESS_WORKERS = 32
DEFAULT_MAX_IN_FLIGHT = 4096
DEFAULT_LAYOUT_SUCCESS_COUNT = 3
DEFAULT_MAX_LAYOUT_SEED_ATTEMPTS = 24
DEFAULT_LAYOUT_SEEDS = tuple(range(DEFAULT_MAX_LAYOUT_SEED_ATTEMPTS))

REPORT_FILE = "layout_report.yml"
UNSAT_FILE = "layout_unsat_report.yml"
SHORTCUT_FILE = "layout_shortcut_features.yml"
UNSAT_ATTEMPT_CACHE_DIR = ".m12_unsat_attempt_cache"
SUCCESS_ATTEMPT_CACHE_DIR = ".m12_success_attempt_cache"

GEOMETRIC_UNSAT_CODES = frozenset(
    {
        "footprint_outside_workspace",
        "camera_bounds",
        "reach_bounds",
        "pick_planning_unreachable",
        "direct_single_arm_reach_bounds",
        "terminal_reach_bounds",
        "constraint_domain_empty",
        "layout_search_exhausted",
    }
)
SEED_LOCAL_UNSOLVED_CODES = GEOMETRIC_UNSAT_CODES | frozenset(
    {
        "operation_contact_domain_empty",
        "operation_space_search_exhausted",
        "pick_planning_search_budget_exhausted",
        "search_budget_exhausted",
        LIVE_SEED_BUDGET_EXHAUSTED,
    }
)


def _layout_seed_failure_disposition(
    failure: Mapping[str, Any],
) -> str:
    """Classify one seed without confusing search budget with infra failure."""

    return (
        "seed_unsolved"
        if str(failure.get("code") or "") in SEED_LOCAL_UNSOLVED_CODES
        else "blocked"
    )


class PickPlanningWitnessProvider(Protocol):
    """Simulator-owned, planning-only oracle injected into the pure M12 core."""

    def __call__(
        self,
        request: Mapping[str, Any],
        *,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


# Compatibility name for existing CLI/scripts; semantics are generic Pick.
D4PlanningWitnessProvider = PickPlanningWitnessProvider


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=yaml.CSafeLoader)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _try_load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _load_yaml(path)
    except (OSError, ValueError, yaml.YAMLError):
        return None


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            yaml.safe_dump(
                canonical_data(dict(value)),
                stream,
                sort_keys=True,
                allow_unicode=True,
                width=100,
            )
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _publish_layout_family_artifacts(
    successful: Sequence[Mapping[str, Any]],
    *,
    success_quota: int,
) -> None:
    """Transactionally replace one family's complete public seed set.

    Successful attempts are prepared in the private success cache.  A resumed
    run can therefore select a different canonical seed set while old public
    seed files are still present.  Publishing the new files one-by-one and
    waiting for the run-wide cleanup leaves a mixed four-or-more-seed public
    directory when the coordinator is interrupted.

    Stage the complete quota first, then retire the old public set and install
    the new set with same-filesystem atomic renames.  Python-level failures
    roll back to the old set.  The transaction is intentionally confined to
    ``rsf_<id>/*.yml``; private success/unsat caches and aggregate audit files
    are not touched.
    """

    quota = int(success_quota)
    if quota < 1 or len(successful) != quota:
        raise ValueError("family publication requires the complete success quota")
    expected_paths = [Path(str(result["expected_path"])) for result in successful]
    family_dirs = {path.parent for path in expected_paths}
    if len(family_dirs) != 1:
        raise ValueError("family publication spans multiple public directories")
    family_dir = next(iter(family_dirs))
    expected_names = [path.name for path in expected_paths]
    if len(set(expected_names)) != quota:
        raise ValueError("family publication contains duplicate seed artifacts")
    if any(path.suffix != ".yml" for path in expected_paths):
        raise ValueError("family publication contains a non-YAML artifact")

    transaction_root = family_dir.parent / ".m12_family_publish_transactions"
    transaction_root.mkdir(parents=True, exist_ok=True)
    transaction_dir = Path(
        tempfile.mkdtemp(prefix=f"{family_dir.name}.", dir=transaction_root)
    )
    staged_dir = transaction_dir / "staged"
    retired_dir = transaction_dir / "retired"
    staged_dir.mkdir()
    retired_dir.mkdir()
    committed = False
    try:
        expected_artifacts: dict[str, Mapping[str, Any]] = {}
        for result, expected_path in zip(successful, expected_paths, strict=True):
            artifact = result.get("artifact")
            if not isinstance(artifact, Mapping):
                raise ValueError("family publication is missing a seed artifact")
            expected_artifacts[expected_path.name] = artifact
            _write_yaml(staged_dir / expected_path.name, artifact)

        # No public artifact is touched until every member of the canonical
        # success set has been serialized successfully.
        family_dir.mkdir(parents=True, exist_ok=True)
        old_public = sorted(family_dir.glob("*.yml"))
        try:
            for old_path in old_public:
                os.replace(old_path, retired_dir / old_path.name)
            for name in expected_names:
                os.replace(staged_dir / name, family_dir / name)

            public_names = sorted(path.name for path in family_dir.glob("*.yml"))
            if public_names != sorted(expected_names):
                raise OSError("family publication did not install the exact seed set")
            for name, expected_artifact in expected_artifacts.items():
                if canonical_data(_load_yaml(family_dir / name)) != canonical_data(
                    dict(expected_artifact)
                ):
                    raise OSError(f"family publication verification failed: {name}")
            committed = True
        except BaseException:
            # Restore the complete previous public view for ordinary I/O or
            # validation failures.  A hard process kill may leave the public
            # view incomplete, but all successful attempts remain in the
            # private cache and the next resume will rebuild this transaction.
            for path in sorted(family_dir.glob("*.yml")):
                path.unlink(missing_ok=True)
            for old_path in sorted(retired_dir.glob("*.yml")):
                os.replace(old_path, family_dir / old_path.name)
            raise
    finally:
        if committed:
            shutil.rmtree(transaction_dir, ignore_errors=True)
            # This parent is shared by every publishing family. Another worker
            # can be between mkdir(exist_ok=True) and mkdtemp, so even an empty
            # parent must remain until all publishers have exited. Each worker
            # cleans up only the unique transaction directory it owns.


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _callable_source_path(value: Any) -> Path:
    source = inspect.getsourcefile(value)
    if source is None:
        raise OSError(f"no source file for {value!r}")
    path = Path(source).resolve()
    if not path.is_file():
        raise OSError(f"source file is unavailable: {path}")
    return path


def _unsat_cache_solver_identity() -> dict[str, Any] | None:
    """Bind cached decisions to the deterministic code and policy surface.

    A version constant alone is too easy to forget while iterating on the
    solver.  The source manifest makes a local code change automatically stale
    even before its public artifact version is bumped.  Missing source files
    disable cache reuse rather than weakening the identity.
    """

    try:
        sources = {
            "layout_core": Path(__file__).resolve(),
            "collision_geometry": _callable_source_path(
                frozen_support_plane_z
            ),
            "schema_validation": _callable_source_path(validate_document),
            "schema_serialization": _callable_source_path(artifact_hash),
            "pick_planning_contract": _callable_source_path(
                build_pick_planning_request
            ),
            "relation_contract": _callable_source_path(
                static_relation_geometry_evidence
            ),
            "unit_contract": _callable_source_path(
                required_operation_anchor_kinds
            ),
            "articulation_capability": _callable_source_path(
                articulation_capability_errors
            ),
        }
        root = Path(__file__).resolve().parents[1]
        for relative in ("manipulation_policy.py", "layout/manipulation.py",
                         "layout/operation_space.py", "layout/operation_catalog.py",
                         "layout/initial_candidates.py", "layout/planner_proxy_candidates.py",
                         "layout/sparse_candidates.py",
                         "grasp_validation/convex_cover.py",
                         "reach_validation/candidates.py", "reach_validation/contract.py"):
            sources[relative] = root / relative
        source_hashes = {
            label: _sha256_file(path) for label, path in sorted(sources.items())
        }
    except OSError:
        return None
    result = {
        "cache_policy_version": UNSAT_ATTEMPT_CACHE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "layout_solver_version": LAYOUT_SOLVER_VERSION,
        "source_hashes": source_hashes,
        "unit_builder": {
            "version": UNIT_BUILD_VERSION,
            "config_hash": artifact_hash(UNIT_BUILD_CONFIG),
        },
        "success_contract": SUCCESS_CONTRACT.identity,
        "relation_contract": RELATION_CONTRACT.identity,
        "articulation_capability_version": ARTICULATION_CAPABILITY_VERSION,
        "pick_planning": pick_planning_identity(),
    }
    return canonical_data(result)


def _provider_semantic_identity(
    provider: PickPlanningWitnessProvider | None,
) -> dict[str, Any] | None:
    """Return a stable semantic identity or fail closed for planner caching.

    Arbitrary closures can depend on uninspectable mutable state, so they are
    not cacheable unless they expose ``m12_unsat_cache_identity``.  Canonical
    provider objects are additionally bound to their implementation source and
    scene-validation configuration.  Operational values such as CUDA device,
    concurrency, timeout and evidence directory intentionally do not enter the
    identity.
    """

    if provider is None:
        return None
    try:
        explicit = getattr(provider, "m12_unsat_cache_identity", None)
        if callable(explicit):
            explicit = explicit()
    except Exception:
        return None
    if explicit is not None and not isinstance(explicit, Mapping):
        return None
    if inspect.isfunction(provider) or inspect.ismethod(provider):
        if explicit is None:
            return None
        target = provider
    else:
        target = type(provider)
    try:
        source_path = _callable_source_path(target)
        source_hashes = {"provider": _sha256_file(source_path)}
        worker_path = source_path.with_name("planning_worker.py")
        if worker_path.is_file():
            source_hashes["planning_worker"] = _sha256_file(worker_path)
    except OSError:
        return None

    semantic_config: dict[str, Any] = {}
    validation_config = getattr(provider, "validation_config", None)
    if isinstance(validation_config, Mapping):
        semantic_config["validation_config_hash"] = artifact_hash(
            canonical_data(dict(validation_config))
        )
    validation_config_path = getattr(
        provider, "validation_config_path", None
    )
    if validation_config_path is not None:
        try:
            semantic_config["validation_config_file_hash"] = _sha256_file(
                Path(validation_config_path)
            )
        except OSError:
            return None
    if explicit is None and not semantic_config:
        return None
    try:
        payload = {
            "callable": {
                "module": str(getattr(target, "__module__", "")),
                "qualname": str(getattr(target, "__qualname__", "")),
            },
            "source_hashes": source_hashes,
            "semantic_config": semantic_config,
            "explicit_identity": (
                None if explicit is None else canonical_data(dict(explicit))
            ),
        }
        return canonical_data(
            {**payload, "identity_hash": artifact_hash(payload)}
        )
    except (TypeError, ValueError):
        return None


def _unsat_attempt_cache_path(
    output_dir: Path, realized_family_id: str, layout_seed: int
) -> Path:
    cache_key = hashlib.sha256(
        f"{realized_family_id}\0{int(layout_seed)}".encode("utf-8")
    ).hexdigest()
    return (
        output_dir
        / UNSAT_ATTEMPT_CACHE_DIR
        / cache_key[:2]
        / f"{cache_key}.yml"
    )


def _success_attempt_cache_path(
    output_dir: Path, realized_family_id: str, layout_seed: int
) -> Path:
    family_key = hashlib.sha256(
        str(realized_family_id).encode("utf-8")
    ).hexdigest()
    return (
        output_dir
        / SUCCESS_ATTEMPT_CACHE_DIR
        / family_key[:2]
        / family_key
        / f"{int(layout_seed)}.yml"
    )


def _clear_unsat_attempt_cache(output_dir: Path) -> None:
    cache_root = output_dir / UNSAT_ATTEMPT_CACHE_DIR
    if cache_root.is_symlink() or cache_root.is_file():
        cache_root.unlink(missing_ok=True)
    elif cache_root.is_dir():
        shutil.rmtree(cache_root)


def _discard_unsat_cache_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _unsat_cache_has_entries(output_dir: Path) -> bool:
    cache_root = output_dir / UNSAT_ATTEMPT_CACHE_DIR
    if not cache_root.is_dir():
        return False
    try:
        return next(cache_root.glob("*/*.yml"), None) is not None
    except OSError:
        return False


def _clean(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def _vec(values: Iterable[float]) -> list[float]:
    return [_clean(value) for value in values]


@dataclass(frozen=True)
class LayoutConfig:
    """Versioned static geometry and search policy."""

    data: dict[str, Any]

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "LayoutConfig":
        value = _load_yaml(path)
        if int(value.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(f"unsupported layout config schema: {path}")
        if int(value.get("layout_solver_version", 0)) != LAYOUT_SOLVER_VERSION:
            raise ValueError(f"unsupported layout solver config: {path}")
        from .initial_candidates import POLICY as INITIAL_POLICY
        initial_policy = value.get("search", {}).get("initial_candidate_policy", "baseline")
        from .planner_proxy_candidates import POLICY as PROXY_POLICY
        from .sparse_candidates import POLICY as SPARSE_POLICY
        if initial_policy not in {"baseline", INITIAL_POLICY["version"], PROXY_POLICY["version"], SPARSE_POLICY["version"]}:
            raise ValueError(f"unsupported initial candidate policy: {initial_policy}")
        result = cls(canonical_data(value))
        _ = result.max_pick_planning_layout_attempts_per_seed
        return result

    @property
    def hash(self) -> str:
        # M4's canonical probe pose is asset-evidence provenance, not an M12
        # world-coordinate constraint.  M12 hashes only its own layout policy;
        # the shared Contact/planner identities are frozen in the exact Pick
        # planning request and witness.
        return artifact_hash({"layout_config": self.data})

    @property
    def workspace(self) -> tuple[float, float, float, float]:
        bounds = self.data["workspace"]["bounds_xy"]
        return (
            float(bounds["x"][0]),
            float(bounds["x"][1]),
            float(bounds["y"][0]),
            float(bounds["y"][1]),
        )

    @property
    def table_z(self) -> float:
        return float(self.data["table"]["surface_z"])

    @property
    def camera_polygon(self) -> list[tuple[float, float]]:
        return [
            (float(point[0]), float(point[1]))
            for point in self.data["camera"]["table_visible_polygon_xy"]
        ]

    @property
    def margin(self) -> float:
        return float(self.data["margin_m"])

    @property
    def max_pick_planning_layout_attempts_per_seed(self) -> int:
        value = self.data.get("search", {}).get(
            "max_pick_planning_layout_attempts_per_seed"
        )
        if isinstance(value, bool):
            raise ValueError(
                "max_pick_planning_layout_attempts_per_seed must be a positive integer"
            )
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "max_pick_planning_layout_attempts_per_seed must be a positive integer"
            ) from exc
        if parsed < 1 or parsed != value:
            raise ValueError(
                "max_pick_planning_layout_attempts_per_seed must be a positive integer"
            )
        return parsed


class LayoutSolveError(ValueError):
    """Attributed layout failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        realized_family_id: str | None = None,
        layout_seed: int | None = None,
        unit_ids: Sequence[str] = (),
        constraint_ids: Sequence[str] = (),
        blockers: Sequence[Mapping[str, Any]] = (),
        owner: str = "layout",
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.realized_family_id = realized_family_id
        self.layout_seed = layout_seed
        self.unit_ids = tuple(unit_ids)
        self.constraint_ids = tuple(constraint_ids)
        self.blockers = tuple(canonical_data(list(blockers)))
        self.owner = owner
        self.metrics = canonical_data(dict(metrics or {}))

    def to_record(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "realized_family_id": self.realized_family_id,
            "layout_seed": self.layout_seed,
            "unit_ids": list(self.unit_ids),
            "constraint_ids": list(self.constraint_ids),
            "blockers": list(self.blockers),
            "owner": self.owner,
            "metrics": self.metrics,
            "geometric_unsat": (self.code in GEOMETRIC_UNSAT_CODES and
                self.metrics.get("failure_classification", {}).get("disposition") != "operation_search_exhausted"),
        }


def _layout_solve_error_from_record(
    raw: Mapping[str, Any],
) -> LayoutSolveError:
    return LayoutSolveError(
        str(raw["code"]),
        str(raw["message"]),
        realized_family_id=(
            None
            if raw.get("realized_family_id") is None
            else str(raw["realized_family_id"])
        ),
        layout_seed=(
            None if raw.get("layout_seed") is None else int(raw["layout_seed"])
        ),
        unit_ids=[str(value) for value in raw.get("unit_ids") or []],
        constraint_ids=[
            str(value) for value in raw.get("constraint_ids") or []
        ],
        blockers=[
            dict(value)
            for value in raw.get("blockers") or []
            if isinstance(value, Mapping)
        ],
        owner=str(raw.get("owner") or "layout"),
        metrics=(
            dict(raw.get("metrics") or {})
            if isinstance(raw.get("metrics"), Mapping)
            else {}
        ),
    )


def _self_hash_valid(value: Mapping[str, Any]) -> bool:
    return value.get("artifact_hash") == artifact_hash(
        value, omit_keys=("artifact_hash",)
    )


def _report_hash_valid(value: Mapping[str, Any]) -> bool:
    return value.get("report_hash") == artifact_hash(
        value, omit_keys=("report_hash",)
    )


def _unit_join_valid(
    realized_artifact: Mapping[str, Any],
    unit_artifact: Mapping[str, Any],
) -> bool:
    return (
        _self_hash_valid(realized_artifact)
        and unit_artifact.get("realized_family_id")
        == realized_artifact.get("realized_family_id")
        and unit_artifact.get("source_realization_hash")
        == realized_artifact.get("artifact_hash")
        and unit_artifact.get("unit_build_version") == UNIT_BUILD_VERSION
        and unit_artifact.get("config_hash") == artifact_hash(UNIT_BUILD_CONFIG)
        and unit_artifact.get("preflight", {}).get("status") == "passed"
        and _self_hash_valid(unit_artifact)
    )


def _unit_local_poses(unit: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    params = unit.get("params") or {}
    unit_type = unit.get("type")
    if unit_type == "ObjectUnit":
        return {str(params["object_key"]): dict(params["local_pose"])}
    if unit_type == "LandmarkRelationUnit":
        return {
            str(key): dict(value)
            for key, value in params["internal_layout"].items()
        }
    if unit_type == "ReceiverReservationUnit":
        return {
            str(key): dict(value)
            for key, value in params["receiver_layout"].items()
        }
    raise LayoutSolveError(
        "unsupported_unit_type",
        f"unsupported Unit type {unit_type!r}",
        unit_ids=[str(unit.get("id"))],
        owner="unit_builder",
    )


def _unit_bounds_local(
    unit: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    fp = unit["params"]["footprint"]
    cx, cy = [float(value) for value in fp["center_xy"]]
    hx, hy = [float(value) for value in fp["half_extents_xy"]]
    return cx - hx, cx + hx, cy - hy, cy + hy


def _global_bounds(
    local: tuple[float, float, float, float], position: Sequence[float]
) -> tuple[float, float, float, float]:
    return (
        local[0] + float(position[0]),
        local[1] + float(position[0]),
        local[2] + float(position[1]),
        local[3] + float(position[1]),
    )


def _overlap_with_margin(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    margin: float,
) -> bool:
    return not (
        first[1] + margin <= second[0] + GEOMETRY_EPSILON
        or second[1] + margin <= first[0] + GEOMETRY_EPSILON
        or first[3] + margin <= second[2] + GEOMETRY_EPSILON
        or second[3] + margin <= first[2] + GEOMETRY_EPSILON
    )


def _point_in_convex_polygon(
    point: Sequence[float], polygon: Sequence[Sequence[float]]
) -> bool:
    signs: set[int] = set()
    px, py = float(point[0]), float(point[1])
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        cross = (float(end[0]) - float(start[0])) * (
            py - float(start[1])
        ) - (float(end[1]) - float(start[1])) * (px - float(start[0]))
        if abs(cross) > 1e-10:
            signs.add(1 if cross > 0 else -1)
    return len(signs) <= 1


def _bounds_in_polygon(
    bounds: tuple[float, float, float, float],
    polygon: Sequence[Sequence[float]],
) -> bool:
    return all(
        _point_in_convex_polygon(point, polygon)
        for point in (
            (bounds[0], bounds[2]),
            (bounds[0], bounds[3]),
            (bounds[1], bounds[2]),
            (bounds[1], bounds[3]),
        )
    )


def _bounds_in_workspace(
    bounds: tuple[float, float, float, float], config: LayoutConfig
) -> bool:
    x0, x1, y0, y1 = config.workspace
    return (
        bounds[0] >= x0 - GEOMETRY_EPSILON
        and bounds[1] <= x1 + GEOMETRY_EPSILON
        and bounds[2] >= y0 - GEOMETRY_EPSILON
        and bounds[3] <= y1 + GEOMETRY_EPSILON
    )


def _rectangle_halfplanes(
    bounds: tuple[float, float, float, float],
) -> list[tuple[float, float, float]]:
    """Return ``a*x + b*y <= c`` rows for one closed rectangle."""

    x0, x1, y0, y1 = bounds
    return [
        (-1.0, 0.0, -float(x0)),
        (1.0, 0.0, float(x1)),
        (0.0, -1.0, -float(y0)),
        (0.0, 1.0, float(y1)),
    ]


def _convex_polygon_halfplanes(
    polygon: Sequence[Sequence[float]],
) -> list[tuple[float, float, float]]:
    """Project a convex polygon to normalized closed halfplanes."""

    if len(polygon) < 3:
        raise ValueError("a convex polygon requires at least three points")
    area2 = sum(
        float(start[0]) * float(polygon[(index + 1) % len(polygon)][1])
        - float(polygon[(index + 1) % len(polygon)][0]) * float(start[1])
        for index, start in enumerate(polygon)
    )
    if abs(area2) <= GEOMETRY_EPSILON:
        raise ValueError("camera polygon has zero signed area")
    ccw = area2 > 0.0
    result: list[tuple[float, float, float]] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        x0, y0 = float(start[0]), float(start[1])
        dx = float(end[0]) - x0
        dy = float(end[1]) - y0
        if math.hypot(dx, dy) <= GEOMETRY_EPSILON:
            raise ValueError("camera polygon contains a zero-length edge")
        if ccw:
            a, b, c = dy, -dx, dy * x0 - dx * y0
        else:
            a, b, c = -dy, dx, -dy * x0 + dx * y0
        norm = math.hypot(a, b)
        result.append((a / norm, b / norm, c / norm))
    return result


def _camera_anchor_halfplanes(
    local_bounds: tuple[float, float, float, float],
    polygon: Sequence[Sequence[float]],
) -> list[tuple[float, float, float]]:
    """Erode a convex camera polygon by an axis-aligned Unit footprint."""

    left, right, bottom, top = local_bounds
    corners = (
        (left, bottom),
        (left, top),
        (right, bottom),
        (right, top),
    )
    result = []
    for a, b, c in _convex_polygon_halfplanes(polygon):
        support = max(a * x + b * y for x, y in corners)
        result.append((a, b, c - support))
    return result


def _halfplane_vertices(
    halfplanes: Sequence[tuple[float, float, float]],
) -> list[tuple[float, float]]:
    """Enumerate every vertex of a bounded closed halfplane intersection.

    Pairwise boundary intersections deliberately preserve regions that collapse
    to a line segment or a single point; random or grid sampling has zero
    probability of finding those domains.
    """

    points: set[tuple[float, float]] = set()
    for first_index, (a1, b1, c1) in enumerate(halfplanes):
        for a2, b2, c2 in halfplanes[first_index + 1 :]:
            determinant = a1 * b2 - a2 * b1
            if abs(determinant) <= GEOMETRY_EPSILON:
                continue
            x = (c1 * b2 - c2 * b1) / determinant
            y = (a1 * c2 - a2 * c1) / determinant
            if all(
                a * x + b * y <= c + GEOMETRY_EPSILON
                for a, b, c in halfplanes
            ):
                points.add((_clean(x), _clean(y)))
    if len(points) <= 1:
        return sorted(points)
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    return sorted(
        points,
        key=lambda point: (
            math.atan2(point[1] - center_y, point[0] - center_x),
            point,
        ),
    )


def _halfplane_domain_candidates(
    halfplanes: Sequence[tuple[float, float, float]],
) -> tuple[list[tuple[float, float]], str]:
    """Return vertices, boundary midpoints and a convex-domain center."""

    vertices = _halfplane_vertices(halfplanes)
    if not vertices:
        return [], "empty"
    if len(vertices) == 1:
        return vertices, "point"
    points = set(vertices)
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        points.add(
            (
                _clean((start[0] + end[0]) / 2.0),
                _clean((start[1] + end[1]) / 2.0),
            )
        )
    center = (
        _clean(sum(point[0] for point in vertices) / len(vertices)),
        _clean(sum(point[1] for point in vertices) / len(vertices)),
    )
    points.add(center)
    if len(vertices) == 2:
        dimension = "line"
    else:
        origin = vertices[0]
        dimension = (
            "area"
            if any(
                abs(
                    (first[0] - origin[0]) * (second[1] - origin[1])
                    - (first[1] - origin[1]) * (second[0] - origin[0])
                )
                > GEOMETRY_EPSILON
                for first in vertices[1:]
                for second in vertices[2:]
            )
            else "line"
        )
    return sorted(points), dimension


def _intersect_anchor_bounds(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    x0, x1 = max(first[0], second[0]), min(first[1], second[1])
    y0, y1 = max(first[2], second[2]), min(first[3], second[3])
    if x0 > x1 + GEOMETRY_EPSILON or y0 > y1 + GEOMETRY_EPSILON:
        return None
    if x0 > x1:
        x0 = x1 = (x0 + x1) / 2.0
    if y0 > y1:
        y0 = y1 = (y0 + y1) / 2.0
    return (_clean(x0), _clean(x1), _clean(y0), _clean(y1))


def _maximal_anchor_bounds(
    bounds: Sequence[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Deduplicate a rectangle union and remove contained components."""

    unique = sorted(set(bounds))
    result = []

    def contains(
        outer: tuple[float, float, float, float],
        inner: tuple[float, float, float, float],
    ) -> bool:
        return (
            outer[0] <= inner[0] + GEOMETRY_EPSILON
            and outer[1] >= inner[1] - GEOMETRY_EPSILON
            and outer[2] <= inner[2] + GEOMETRY_EPSILON
            and outer[3] >= inner[3] - GEOMETRY_EPSILON
        )

    for index, candidate in enumerate(unique):
        if any(
            index != other_index
            and contains(other, candidate)
            and (
                not contains(candidate, other)
                or other_index < index
            )
            for other_index, other in enumerate(unique)
        ):
            continue
        result.append(candidate)
    return result


def _anchor_domain_rectangles(
    workspace_anchor_bounds: tuple[float, float, float, float],
    constraint_groups: Sequence[Mapping[str, Any]],
) -> list[tuple[float, float, float, float]]:
    """Intersect an AND of reach groups, each represented as an OR of AABBs."""

    domains = [workspace_anchor_bounds]
    for group in constraint_groups:
        options = [
            tuple(float(value) for value in option)
            for option in group.get("options", [])
            if len(option) == 4
        ]
        if not options:
            return []
        next_domains = [
            intersection
            for domain in domains
            for option in options
            for intersection in [_intersect_anchor_bounds(domain, option)]
            if intersection is not None
        ]
        domains = _maximal_anchor_bounds(next_domains)
        if not domains:
            return []
    return domains


def _halton(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0
    value = index
    while value:
        fraction /= base
        result += fraction * (value % base)
        value //= base
    return result


def _seed_offset(realized_id: str, unit_id: str, layout_seed: int) -> int:
    payload = f"{realized_id}|{unit_id}|{layout_seed}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:12], 16) % 100003


def _candidate_positions(
    realized_id: str,
    unit: Mapping[str, Any],
    layout_seed: int,
    config: LayoutConfig,
    *,
    constraint_groups: Sequence[Mapping[str, Any]] = (),
    anchor_domains: Sequence[tuple[float, float, float, float]] | None = None,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Generate constraint-derived anchors before coverage fallback points."""

    wx0, wx1, wy0, wy1 = config.workspace
    left, right, bottom, top = _unit_bounds_local(unit)
    x0, x1 = wx0 - left, wx1 - right
    y0, y1 = wy0 - bottom, wy1 - top
    workspace_anchor_bounds = (x0, x1, y0, y1)
    metrics = {
        "workspace_bounds_xy": {"x": [wx0, wx1], "y": [wy0, wy1]},
        "unit_local_bounds_xy": {
            "x": [_clean(left), _clean(right)],
            "y": [_clean(bottom), _clean(top)],
        },
        "anchor_bounds_xy": {
            "x": [_clean(x0), _clean(x1)],
            "y": [_clean(y0), _clean(y1)],
        },
    }
    if x0 > x1 or y0 > y1:
        return [], metrics
    if anchor_domains is None:
        anchor_domains = _anchor_domain_rectangles(
            workspace_anchor_bounds, constraint_groups
        )
    camera_halfplanes = _camera_anchor_halfplanes(
        (left, right, bottom, top), config.camera_polygon
    )
    base_candidates, base_dimension = _halfplane_domain_candidates(
        [
            *_rectangle_halfplanes(workspace_anchor_bounds),
            *camera_halfplanes,
        ]
    )
    constraint_candidates: set[tuple[float, float]] = set()
    constraint_dimensions: Counter[str] = Counter()
    if constraint_groups:
        for domain in anchor_domains:
            points, dimension = _halfplane_domain_candidates(
                [*_rectangle_halfplanes(domain), *camera_halfplanes]
            )
            constraint_candidates.update(points)
            constraint_dimensions[dimension] += 1

    # Coverage points are diversity-only.  Correctness comes from the exact
    # domain tier above; the former whole-table 5x5 grid is intentionally gone.
    fallback_points: set[tuple[float, float]] = set()
    count = int(config.data["search"]["halton_candidates_per_unit"])
    offset = _seed_offset(realized_id, str(unit["id"]), layout_seed)
    for index in range(1, count + 1):
        sequence_index = offset + index
        fallback_points.add(
            (
                x0 + (x1 - x0) * _halton(sequence_index, 2),
                y0 + (y1 - y0) * _halton(sequence_index, 3),
            )
        )
    camera = config.camera_polygon
    fallback_candidates = {
        (_clean(x), _clean(y))
        for x, y in fallback_points
        if _bounds_in_polygon(
            _global_bounds((left, right, bottom, top), (x, y)), camera
        )
    }

    def ordered(
        points: Iterable[tuple[float, float]], tier: str
    ) -> list[tuple[float, float]]:
        return sorted(
            set(points),
            key=lambda point: artifact_hash(
                {
                    "realized_family_id": realized_id,
                    "unit_id": unit["id"],
                    "layout_seed": layout_seed,
                    "candidate_tier": tier,
                    "point": point,
                }
            ),
        )

    exact = ordered(constraint_candidates, "constraint_domain")
    base = ordered(set(base_candidates) - set(exact), "camera_workspace")
    fallback = ordered(
        fallback_candidates - set(exact) - set(base), "coverage_fallback"
    )
    metrics.update(
        {
            "candidate_policy": "constraint_domain_vertices_centers_v1",
            "constraint_group_count": len(constraint_groups),
            "anchor_domain_count": len(anchor_domains),
            "anchor_domain_bounds_xy": [
                {
                    "x": [_clean(domain[0]), _clean(domain[1])],
                    "y": [_clean(domain[2]), _clean(domain[3])],
                }
                for domain in anchor_domains
            ],
            "camera_workspace_domain_dimension": base_dimension,
            "constraint_domain_dimensions": dict(
                sorted(constraint_dimensions.items())
            ),
            "constraint_candidate_count": len(exact),
            "camera_workspace_candidate_count": len(base),
            "fallback_candidate_count": len(fallback),
        }
    )
    return [*exact, *base, *fallback], metrics


def _role_entities(realized: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(role): str(binding["entity_key"])
        for role, binding in realized["role_bindings"].items()
    }


def _entity_units(
    units: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    unit_by_entity: dict[str, str] = {}
    pose_by_entity: dict[str, dict[str, Any]] = {}
    for unit in units:
        poses = _unit_local_poses(unit)
        for entity, pose in poses.items():
            if entity in unit_by_entity:
                raise LayoutSolveError(
                    "duplicate_entity_membership",
                    f"entity {entity} appears in multiple Units",
                    unit_ids=[unit_by_entity[entity], str(unit["id"])],
                    owner="unit_builder",
                )
            unit_by_entity[entity] = str(unit["id"])
            pose_by_entity[entity] = pose
    return unit_by_entity, pose_by_entity


def _reach_requirements(
    realized: Mapping[str, Any],
) -> dict[str, str]:
    requirements: dict[str, set[str]] = defaultdict(set)
    for variant in realized.get("goal_variants", []):
        goal = variant.get("goal") or {}
        entity = goal.get("subject")
        if entity:
            arm = goal.get("arm")
            requirements[str(entity)].add("any" if arm is None else str(arm))
    roles = _role_entities(realized)
    for constraint in realized.get("layout_constraints", []):
        if constraint.get("type") != "reach_zone":
            continue
        if (constraint.get("params") or {}).get("zone") == "both":
            for role in constraint.get("roles", []):
                if str(role) in roles:
                    requirements[roles[str(role)]] = {"left", "right"}
    result: dict[str, str] = {}
    for entity, values in requirements.items():
        if {"left", "right"} <= values:
            result[entity] = "both"
        elif "left" in values:
            result[entity] = "left"
        elif "right" in values:
            result[entity] = "right"
        else:
            result[entity] = "any"
    return result


def _operation_anchor_kinds(
    realized: Mapping[str, Any],
) -> dict[str, set[str]]:
    """Return the shared M11/M12 operation-point requirements."""

    return required_operation_anchor_kinds(realized)


def _pick_operation_entities(realized: Mapping[str, Any]) -> set[str]:
    """Return subjects whose skill requires a grasp acquisition placement."""

    result: set[str] = set()
    for variant in realized.get("goal_variants", []):
        goal = variant.get("goal") or {}
        subject = goal.get("subject")
        if not subject:
            continue
        if (
            str(goal.get("action_family") or "") == "O2O_relation"
            or str(goal.get("verb") or "")
            in {"pick", "adjust_orientation"}
        ):
            result.add(str(subject))
    return result


def _pick_operation_use(
    realized: Mapping[str, Any], entity_key: str, arm: str
) -> str:
    """Select the M4 Contact evidence used by one exact acquisition gate."""

    verbs = {
        str((variant.get("goal") or {}).get("verb") or "")
        for variant in realized.get("goal_variants", [])
        if str((variant.get("goal") or {}).get("subject") or "")
        == str(entity_key)
    }
    families = {
        str((variant.get("goal") or {}).get("action_family") or "")
        for variant in realized.get("goal_variants", [])
        if str((variant.get("goal") or {}).get("subject") or "")
        == str(entity_key)
    }
    if verbs == {"adjust_orientation"} and families == {"A2O_state"}:
        return "adjust_orientation_subject"
    return f"pick_{arm}"


def _rigid_pick_capable_arms(
    realized: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    """Return arm choices backed by exact M11 rigid-Pick M4 evidence."""

    contracts = _operation_anchor_contracts(units)
    result: dict[str, set[str]] = {}
    for entity in sorted(_pick_operation_entities(realized)):
        contract = contracts.get(entity) or {}
        rows = (contract.get("source") or {}).get(
            "validated_contact_contracts"
        ) or []
        arms = {
            arm
            for arm in ("left", "right")
            if any(
                str(row.get("use") or "") == f"pick_{arm}"
                and isinstance(row.get("binding"), Mapping)
                for row in rows
            )
        }
        # Orientation adjustment has one arm-neutral acquisition use and is
        # handled by its existing Contact path. Only constrain ordinary rigid
        # Pick/O2O subjects that carry explicit pick_left/pick_right evidence.
        if arms:
            result[entity] = arms
    return result


def _orientation_targets_for_entity(
    realized: Mapping[str, Any],
    entity_key: str,
    initial_quaternion_wxyz: Sequence[float],
    *,
    goal_variant_ids: set[str] | None = None,
) -> list[list[float]]:
    rows: list[tuple[int, list[float]]] = []
    for variant in realized.get("goal_variants", []):
        goal_variant_id = str(variant.get("id") or "")
        goal = variant.get("goal") or {}
        if (
            str(goal.get("subject") or "") != str(entity_key)
            or str(goal.get("verb") or "") != "adjust_orientation"
            or (
                goal_variant_ids is not None
                and goal_variant_id not in goal_variant_ids
            )
        ):
            continue
        try:
            params = normalize_orientation_adjustment_params(
                goal.get("params") or {}
            )
            target = target_quaternion_wxyz(
                initial_quaternion_wxyz, params
            )
        except ValueError as exc:
            raise LayoutSolveError(
                "pick_planning_witness_source_invalid",
                f"invalid orientation goal for {entity_key}: {exc}",
                owner="realization",
            ) from exc
        rows.append((int(params["rotation_delta_deg"]), target))
    return [target for _, target in sorted(rows)]


def _operation_anchor_contracts(
    units: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for unit in units:
        for entity, contract in (
            (unit.get("params") or {}).get("operation_anchors") or {}
        ).items():
            if str(entity) in result:
                raise LayoutSolveError(
                    "duplicate_operation_anchor_contract",
                    f"entity {entity} has multiple operation-anchor contracts",
                    unit_ids=[str(unit.get("id") or "")],
                    owner="unit_builder",
                )
            result[str(entity)] = contract
    return result


def _articulation_capabilities_by_entity(
    realized: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Read exact M4 per-arm capability only from the frozen M11 proof."""

    contracts = _operation_anchor_contracts(units)
    required = m4_articulation_evidence_required_entities(realized)
    projected: dict[str, dict[str, Any]] = {}
    for entity, uses in sorted(required.items()):
        contract = contracts.get(entity)
        rows = list(
            ((contract or {}).get("source") or {}).get(
                "validated_articulation_executions"
            )
            or []
        )
        use_projection: dict[str, Any] = {}
        for use in sorted(uses):
            matches = [
                row
                for row in rows
                if str(row.get("use") or "") == str(use)
            ]
            binding = (
                (matches[0].get("binding") or {}) if len(matches) == 1 else {}
            )
            failures = articulation_capability_errors(binding, use=use)
            if len(matches) != 1 or failures:
                raise LayoutSolveError(
                    "articulation_capability_source_invalid",
                    f"{entity} lacks exact M4 per-arm capability for {use}",
                    owner="unit_builder",
                )
            arm_capabilities = canonical_data(
                binding.get("arm_capabilities") or []
            )
            use_projection[use] = {
                "capable_arms": sorted(
                    str(row["arm"]) for row in arm_capabilities
                ),
                "arm_capabilities": arm_capabilities,
                "sampling_version": binding.get("sampling_version"),
                "requested_arms": binding.get("requested_arms"),
                "arm_results": binding.get("arm_results"),
                "failures": binding.get("failures"),
            }
        projected[entity] = {
            "capability_version": ARTICULATION_CAPABILITY_VERSION,
            "sampling_version": next(
                (
                    str(row.get("binding", {}).get("sampling_version"))
                    for row in rows
                    if row.get("binding", {}).get("sampling_version")
                ),
                None,
            ),
            "uses": use_projection,
            "evidence": "m11_exact_m4_articulation_capability_v1",
        }
    return canonical_data(projected)


def _touch_capabilities_by_entity(
    realized: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Read exact M4 click/press per-arm evidence from the frozen M11 Unit."""

    contracts = _operation_anchor_contracts(units)
    required = m4_touch_evidence_required_entities(realized)
    projected: dict[str, dict[str, Any]] = {}
    for entity, uses in sorted(required.items()):
        rows = list(
            (((contracts.get(entity) or {}).get("source") or {}).get(
                "validated_touch_executions"
            ))
            or []
        )
        use_projection: dict[str, Any] = {}
        for use in sorted(uses):
            matches = [
                row for row in rows if str(row.get("use") or "") == use
            ]
            binding = (
                (matches[0].get("binding") or {}) if len(matches) == 1 else {}
            )
            failures = touch_capability_errors(binding, use=use)
            if len(matches) != 1 or failures:
                raise LayoutSolveError(
                    "touch_capability_source_invalid",
                    f"{entity} lacks exact M4 per-arm capability for {use}",
                    owner="unit_builder",
                )
            arm_capabilities = canonical_data(
                binding.get("arm_capabilities") or []
            )
            use_projection[use] = {
                "capable_arms": sorted(
                    str(row["arm"]) for row in arm_capabilities
                ),
                "arm_capabilities": arm_capabilities,
                "sampling_version": binding.get("sampling_version"),
                "placement_policy_version": binding.get(
                    "placement_policy_version"
                ),
                "placement_policy_hash": binding.get(
                    "placement_policy_hash"
                ),
                "requested_arms": binding.get("requested_arms"),
                "arm_results": binding.get("arm_results"),
                "failures": binding.get("failures"),
                "canonical_placement_witness_hash": binding.get(
                    "canonical_placement_witness_hash"
                ),
            }
        projected[entity] = {
            "capability_version": TOUCH_CAPABILITY_VERSION,
            "uses": use_projection,
            "evidence": "m11_exact_m4_touch_capability_v1",
        }
    return canonical_data(projected)


def _articulation_use_from_goal(goal: Mapping[str, Any]) -> str | None:
    return {
        "open": "open",
        "close": "close",
        "on": "switch_on",
        "off": "switch_off",
        "switch_on": "switch_on",
        "switch_off": "switch_off",
    }.get(str(goal.get("verb") or goal.get("state") or ""))


def _touch_use_from_goal(goal: Mapping[str, Any]) -> str | None:
    return {
        "click": "click_subject",
        "press": "press_subject",
    }.get(str(goal.get("verb") or ""))


def _articulation_operation_reach_groups(
    realized: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build AND groups whose options are only M4-successful Contacts.

    Each exact ``(entity, use, arm)`` goal becomes one group. Candidate
    Contacts within a group are OR alternatives; distinct arm/use groups are
    conjoined. Thus D4 no longer requires one point in the left/right reach
    intersection.
    """

    capabilities = _articulation_capabilities_by_entity(realized, units)
    reach_intent = _reach_requirements(realized)
    groups_by_entity: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for variant in realized.get("goal_variants", []):
        goal = variant.get("goal") or {}
        entity = str(goal.get("subject") or "")
        use = _articulation_use_from_goal(goal)
        if not entity or use is None:
            continue
        use_row = ((capabilities.get(entity) or {}).get("uses") or {}).get(use)
        capable_arms = set((use_row or {}).get("capable_arms") or [])
        requested = goal.get("arm")
        if requested is not None:
            arms = [str(requested)]
            mode = "all"
        elif reach_intent.get(entity) == "both":
            arms = ["left", "right"]
            mode = "all"
        elif reach_intent.get(entity) in {"left", "right"}:
            arms = [str(reach_intent[entity])]
            mode = "all"
        else:
            arms = sorted(capable_arms)
            mode = "any"
        if not arms or any(arm not in capable_arms for arm in arms):
            raise LayoutSolveError(
                "articulation_arm_capability_missing",
                f"{entity} use {use} lacks requested arm capability {arms}",
                owner="asset_validation",
            )
        arm_rows = {
            str(row.get("arm")): row
            for row in (use_row or {}).get("arm_capabilities") or []
        }
        if mode == "all":
            arm_sets = [[arm] for arm in arms]
        else:
            arm_sets = [arms]
        for arm_set in arm_sets:
            options: list[dict[str, Any]] = []
            for arm in arm_set:
                # _articulation_capabilities_by_entity has already validated
                # the complete binding before exposing this exact arm row.
                arm_capability = arm_rows.get(arm)
                for candidate in (arm_capability or {}).get(
                    "successful_candidates", []
                ):
                    contact = candidate.get("candidate") or {}
                    if contact.get("contact_id") is not None:
                        options.append(
                            {
                                "arm": arm,
                                "contact_id": int(contact["contact_id"]),
                            }
                        )
            group_id = f"articulation_reach:{entity}:{use}:" + (
                arm_set[0] if len(arm_set) == 1 else "any"
            )
            if not options:
                raise LayoutSolveError(
                    "articulation_arm_capability_missing",
                    f"{group_id} has no M4-successful Contact candidate",
                    owner="asset_validation",
                )
            groups_by_entity[entity][group_id] = {
                "id": group_id,
                "kind": "articulation",
                "entity_key": entity,
                "use": use,
                "mode": "any",
                "options": sorted(
                    options, key=lambda row: (row["arm"], row["contact_id"])
                ),
            }
    return {
        entity: [groups[group_id] for group_id in sorted(groups)]
        for entity, groups in sorted(groups_by_entity.items())
    }


def _touch_operation_reach_groups(
    realized: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Conjoin exact left/right click/press contact-anchor reach at one pose."""

    capabilities = _touch_capabilities_by_entity(realized, units)
    groups_by_entity: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for variant in realized.get("goal_variants", []):
        goal = variant.get("goal") or {}
        entity = str(goal.get("subject") or "")
        use = _touch_use_from_goal(goal)
        arm = str(goal.get("arm") or "")
        if not entity or use is None:
            continue
        use_row = ((capabilities.get(entity) or {}).get("uses") or {}).get(use)
        arm_rows = {
            str(row.get("arm")): row
            for row in (use_row or {}).get("arm_capabilities") or []
        }
        capability = arm_rows.get(arm)
        if not isinstance(capability, Mapping) or capability.get("contact_id") is None:
            raise LayoutSolveError(
                "touch_arm_capability_missing",
                f"{entity} use {use} is not validated for {arm}",
                owner="asset_validation",
            )
        group_id = f"touch_reach:{entity}:{use}:{arm}"
        groups_by_entity[entity][group_id] = {
            "id": group_id,
            "kind": "touch",
            "entity_key": entity,
            "use": use,
            "mode": "any",
            "options": [
                {"arm": arm, "contact_id": int(capability["contact_id"])}
            ],
        }
    return {
        entity: [groups[group_id] for group_id in sorted(groups)]
        for entity, groups in sorted(groups_by_entity.items())
    }


def _operation_reach_groups(
    realized: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in (
        _articulation_operation_reach_groups(realized, units),
        _touch_operation_reach_groups(realized, units),
    ):
        for entity, groups in source.items():
            merged[entity].extend(canonical_data(groups))
    return {
        entity: sorted(groups, key=lambda row: str(row["id"]))
        for entity, groups in sorted(merged.items())
    }


def _effective_reach_requirements(
    realized: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Conjoin geometric reach intent with exact per-arm M4 capability."""

    requirements = _reach_requirements(realized)
    capabilities = _articulation_capabilities_by_entity(realized, units)
    touch_capabilities = _touch_capabilities_by_entity(realized, units)
    rigid_pick_capabilities = _rigid_pick_capable_arms(realized, units)
    forced: dict[str, set[str]] = defaultdict(set)
    for entity, requirement in requirements.items():
        if requirement == "both":
            forced[entity].update(("left", "right"))
        elif requirement in {"left", "right"}:
            forced[entity].add(requirement)
    for entity, capable in rigid_pick_capabilities.items():
        requirement = requirements.get(entity, "any")
        required = (
            {"left", "right"}
            if requirement == "both"
            else {str(requirement)}
            if requirement in {"left", "right"}
            else set()
        )
        if required and not required.issubset(capable):
            raise LayoutSolveError(
                "pick_arm_capability_missing",
                f"{entity} lacks required rigid Pick arm capability "
                f"{sorted(required - capable)}",
                owner="asset_validation",
            )
        if requirement == "any" and len(capable) == 1:
            forced[entity].update(capable)
    for variant in realized.get("goal_variants", []):
        goal = variant.get("goal") or {}
        entity = str(goal.get("subject") or "")
        use = _articulation_use_from_goal(goal)
        if not entity or use is None:
            continue
        use_row = ((capabilities.get(entity) or {}).get("uses") or {}).get(use)
        capable = set((use_row or {}).get("capable_arms") or [])
        semantic_arm = goal.get("arm")
        if semantic_arm is not None:
            requested = str(semantic_arm)
            if requested not in capable:
                raise LayoutSolveError(
                    "articulation_arm_capability_missing",
                    f"{entity} use {use} is not validated for {requested}",
                    owner="asset_validation",
                )
            forced[entity].add(requested)
        elif not capable:
            raise LayoutSolveError(
                "articulation_arm_capability_missing",
                f"{entity} use {use} has no validated arm",
                owner="asset_validation",
            )
        elif len(capable) == 1:
            forced[entity].update(capable)
    for entity, arms in forced.items():
        if {"left", "right"} <= arms:
            requirements[entity] = "both"
        elif "left" in arms:
            requirements[entity] = "left"
        elif "right" in arms:
            requirements[entity] = "right"
    for variant in realized.get("goal_variants", []):
        goal = variant.get("goal") or {}
        entity = str(goal.get("subject") or "")
        use = _touch_use_from_goal(goal)
        if not entity or use is None:
            continue
        capable = set(
            (((touch_capabilities.get(entity) or {}).get("uses") or {}).get(use) or {}).get(
                "capable_arms"
            )
            or []
        )
        semantic_arm = str(goal.get("arm") or "")
        if semantic_arm not in capable:
            raise LayoutSolveError(
                "touch_arm_capability_missing",
                f"{entity} use {use} is not validated for {semantic_arm}",
                owner="asset_validation",
            )
    return requirements


def _pick_planning_request(
    realized_artifact: Mapping[str, Any],
    unit_artifact: Mapping[str, Any],
    scene_spec: Mapping[str, Any],
    config: LayoutConfig,
    *,
    prior_failure_counts: Mapping[tuple[str, str], int] | None = None,
) -> dict[str, Any] | None:
    """Project every required Pick target into exact-scene oracle requests."""

    realized = (
        realized_artifact.get("realized_scene_family") or realized_artifact
    )
    from .manipulation import selected_realized
    original_realized = realized
    realized = selected_realized(realized, scene_spec)
    try:
        requirement_groups = single_arm_pick_requirement_groups(
            original_realized, scene_spec
        )
    except ValueError as exc:
        error = str(exc)
        reach_failure = error.startswith(
            ("pick_planning_no_direct_arm:", "pick_planning_semantic_arm_unreachable:")
        )
        raise LayoutSolveError(
            (
                "direct_single_arm_reach_bounds"
                if reach_failure
                else "pick_planning_source_invalid"
            ),
            (
                f"no direct singleton-arm Pick assignment exists: {exc}"
                if reach_failure
                else f"invalid Pick GoalVariant identity: {exc}"
            ),
            realized_family_id=str(realized["id"]),
            layout_seed=int(scene_spec["layout_seed"]),
            constraint_ids=[
                "layout.direct_single_arm_reach"
                if reach_failure
                else "realization.goal_variant_identity"
            ],
            owner="layout" if reach_failure else "realization",
        ) from exc
    if not requirement_groups:
        return None
    arms_by_entity: dict[str, set[str]] = defaultdict(set)
    for group in requirement_groups:
        arms_by_entity[str(group["entity_key"])].add(
            str(group["selected_arm"])
        )
    pick_entities = sorted(arms_by_entity)
    contracts = _operation_anchor_contracts(
        list(unit_artifact.get("units") or [])
    )
    entities = scene_spec.get("entities") or {}
    reach = scene_spec.get("reach") or {}
    targets: list[dict[str, Any]] = []
    for entity in pick_entities:
        contract = contracts.get(entity)
        spec = entities.get(entity)
        if not isinstance(contract, Mapping) or not isinstance(spec, Mapping):
            raise LayoutSolveError(
                "pick_planning_witness_source_invalid",
                f"Pick subject {entity} has no exact M11 operation-anchor source",
                realized_family_id=str(realized["id"]),
                layout_seed=int(scene_spec["layout_seed"]),
                owner="unit_builder",
                blockers=[{"entity_key": entity}],
            )
        arms = [
            arm for arm in ("left", "right") if arm in arms_by_entity[entity]
        ]
        if not arms:
            raise LayoutSolveError(
                "pick_planning_witness_source_invalid",
                f"Pick subject {entity} has no concrete arm requirement",
                realized_family_id=str(realized["id"]),
                layout_seed=int(scene_spec["layout_seed"]),
                owner="layout",
                blockers=[{"entity_key": entity}],
            )
        evidence_rows = (contract.get("source") or {}).get(
            "validated_contact_contracts"
        ) or []
        for arm in arms:
            assigned_goal_variant_ids = {
                str(group["goal_variant_id"])
                for group in requirement_groups
                if str(group["entity_key"]) == entity
                and str(group["selected_arm"]) == arm
            }
            validated_use = _pick_operation_use(realized, entity, arm)
            matches = [
                row
                for row in evidence_rows
                if str(row.get("use") or "") == validated_use
                and isinstance(row.get("binding"), Mapping)
            ]
            if len(matches) != 1:
                raise LayoutSolveError(
                    "pick_planning_witness_source_invalid",
                    (
                        f"Pick subject {entity} requires one exact "
                        f"{validated_use} Contact binding, found {len(matches)}"
                    ),
                    realized_family_id=str(realized["id"]),
                    layout_seed=int(scene_spec["layout_seed"]),
                    owner="unit_builder",
                    blockers=[
                        {
                            "entity_key": entity,
                            "arm": arm,
                            "validated_use": validated_use,
                        }
                    ],
                )
            binding = canonical_data(dict(matches[0]["binding"]))
            validated_by_arm = matches[0].get(
                "validated_contact_ids_by_arm"
            ) or {}
            validated_contact_ids = (
                validated_by_arm.get(arm)
                if isinstance(validated_by_arm, Mapping)
                else None
            )
            if not validated_contact_ids:
                validated_contact_ids = matches[0].get(
                    "validated_contact_ids"
                )
            if not isinstance(validated_contact_ids, list) or not (
                validated_contact_ids
            ):
                # Old M11 artifacts remain readable, but newly rebuilt M11
                # always carries M4-proven per-arm Contact priorities.
                validated_contact_ids = list(
                    range(int(binding["contact_count"]))
                )
            validated_contact_ids = [
                int(value) for value in validated_contact_ids
            ]
            if (
                len(validated_contact_ids)
                != len(set(validated_contact_ids))
                or any(
                    value < 0
                    or value >= int(binding["contact_count"])
                    for value in validated_contact_ids
                )
            ):
                raise LayoutSolveError(
                    "pick_planning_witness_source_invalid",
                    f"Pick subject {entity} has invalid validated Contact ids "
                    f"for {arm}",
                    realized_family_id=str(realized["id"]),
                    layout_seed=int(scene_spec["layout_seed"]),
                    owner="unit_builder",
                    blockers=[{"entity_key": entity, "arm": arm}],
                )
            try:
                support_plane_z = frozen_support_plane_z(
                    spec.get("initial_pose") or {}
                )
            except RuntimeError as exc:
                raise LayoutSolveError(
                    "pick_planning_witness_source_invalid",
                    f"Pick subject {entity} has no frozen support plane",
                    realized_family_id=str(realized["id"]),
                    layout_seed=int(scene_spec["layout_seed"]),
                    owner="unit_builder",
                    blockers=[{"entity_key": entity, "arm": arm}],
                ) from exc
            world_yaw = float(
                (spec.get("initial_pose") or {}).get("world_yaw_deg", 0.0)
            )
            planning_request = {
                "subject": entity,
                "arm": arm,
                "asset": str(spec["asset"]),
                "model_id": str(spec["model_id"]),
                "use": str(binding["candidate_use"]),
                "use_profile": str(binding["use_profile"]),
                "requested_pre_grasp_m": DEFAULT_PICK_PRE_GRASP_M,
                "table_acquisition": True,
                "support_plane_z_m": support_plane_z,
                "contact_groups": binding["contact_groups"],
                "group_index": None,
                "contact_ids": list(range(int(binding["contact_count"]))),
                "priority_contact_ids": validated_contact_ids,
                "maximum_attempts": int(
                    EXACT_GRASP_POLICY["maximum_physical_contact_attempts"]
                ),
                "minimum_normalized_joint_margin": float(
                    EXACT_GRASP_POLICY["minimum_normalized_joint_margin"]
                ),
                "world_yaw_deg": world_yaw,
                "reach_zone": str(
                    (reach.get(entity) or {}).get("zone", "both")
                ),
                "post_lift_target_quats_wxyz": (
                    _orientation_targets_for_entity(
                        realized,
                        entity,
                        spec["pose"]["quat_wxyz"],
                        goal_variant_ids=assigned_goal_variant_ids,
                    )
                ),
            }
            op = ((scene_spec.get("provenance") or {}).get("metadata") or {}).get("operation_space") or {}
            options = op.get("goal_options") or {}
            domains = [{int(row["contact_id"]) for row in options[gid] if row["arm"] == arm}
                       for gid in assigned_goal_variant_ids if gid in options]
            if domains:
                allowed = set.intersection(*domains)
                if not allowed:
                    raise LayoutSolveError("operation_contact_domain_empty", "no common Contact connects assigned goals",
                        realized_family_id=str(realized["id"]), layout_seed=int(scene_spec["layout_seed"]))
                planning_request["contact_ids"] = sorted(allowed)
                planning_request["priority_contact_ids"] = [cid for cid in validated_contact_ids if cid in allowed]
            targets.append(
                {
                    "entity_key": entity,
                    "arm": arm,
                    "validated_use": validated_use,
                    "entity_pose": spec["pose"],
                    "operation_anchor_contract_hash": str(
                        contract.get("contract_hash") or ""
                    ),
                    "contact_contract_binding": binding,
                    "planning_request": planning_request,
                    **({"operation_constraint": {
                        "constraint_hash": op["constraint_hash"],
                        "goal_variant_ids": sorted(gid for gid in assigned_goal_variant_ids if gid in options),
                        "allowed_contact_ids": sorted(allowed),
                    }} if domains else {}),
                }
            )
    target_by_key = {
        (str(target["entity_key"]), str(target["arm"])): target
        for target in targets
    }
    failure_counts = dict(prior_failure_counts or {})

    def reach_slack(entity: str, arm: str) -> float:
        position = (
            target_by_key[(entity, arm)].get("entity_pose", {}).get(
                "position"
            )
            or []
        )
        bounds = config.data.get("reach", {}).get(arm) or {}
        x_bounds = bounds.get("x") or []
        y_bounds = bounds.get("y") or []
        if (
            len(position) < 2
            or len(x_bounds) != 2
            or len(y_bounds) != 2
        ):
            return 0.0
        x, y = float(position[0]), float(position[1])
        x0, x1 = float(x_bounds[0]), float(x_bounds[1])
        y0, y1 = float(y_bounds[0]), float(y_bounds[1])
        x_scale = max(0.5 * (x1 - x0), 1e-9)
        y_scale = max(0.5 * (y1 - y0), 1e-9)
        return min(
            (x - x0) / x_scale,
            (x1 - x) / x_scale,
            (y - y0) / y_scale,
            (y1 - y) / y_scale,
        )

    def group_order_key(group: Mapping[str, Any]) -> tuple[Any, ...]:
        entity = str(group["entity_key"])
        arms = [str(arm) for arm in group["arms"]]
        return (
            -max(
                (failure_counts.get((entity, arm), 0) for arm in arms),
                default=0,
            ),
            0 if str(group["mode"]) == "all" else 1,
            len(arms),
            min((reach_slack(entity, arm) for arm in arms), default=0.0),
            entity,
        )

    evaluation_order: list[dict[str, str]] = []
    scheduled_targets: set[tuple[str, str]] = set()
    for group in sorted(requirement_groups, key=group_order_key):
        entity = str(group["entity_key"])
        mode = str(group["mode"])
        arms = [str(arm) for arm in group["arms"]]
        arms.sort(
            key=lambda arm: (
                -failure_counts.get((entity, arm), 0),
                (
                    -reach_slack(entity, arm)
                    if mode == "any"
                    else reach_slack(entity, arm)
                ),
                arm,
            )
        )
        for arm in arms:
            key = (entity, arm)
            if key in scheduled_targets:
                continue
            scheduled_targets.add(key)
            evaluation_order.append({"entity_key": entity, "arm": arm})

    return build_pick_planning_request(
        tentative_scene_hash=artifact_hash(scene_spec),
        realized_family_id=str(realized["id"]),
        layout_seed=int(scene_spec["layout_seed"]),
        source_realization_hash=str(realized_artifact["artifact_hash"]),
        source_unit_hash=str(unit_artifact["artifact_hash"]),
        config_hash=config.hash,
        requirement_groups=requirement_groups,
        targets=targets,
        evaluation_order=evaluation_order,
    )


def _pick_failure_counts_from_rejections(
    rejections: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    """Count concrete failed targets from earlier deterministic layouts."""

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for rejection in rejections:
        witnesses = [
            row.get("witness")
            for row in rejection.get("arm_selection_retries") or []
            if isinstance(row, Mapping)
        ]
        witnesses.append(rejection.get("witness"))
        for witness in witnesses:
            if not isinstance(witness, Mapping):
                continue
            for row in witness.get("results") or []:
                if not isinstance(row, Mapping) or row.get("status") != "failed":
                    continue
                code = str(row.get("failure_code") or "")
                if (
                    code.startswith("pruned_after_")
                    or code == "initial_scene_unstable"
                ):
                    continue
                entity = str(row.get("entity_key") or "")
                arm = str(row.get("arm") or "")
                if entity and arm in {"left", "right"}:
                    counts[(entity, arm)] += 1
    return dict(counts)


def _scene_with_pick_arm_assignment(
    realized: Mapping[str, Any],
    scene_spec: Mapping[str, Any],
    selected_arm_by_goal_variant: Mapping[str, str],
) -> dict[str, Any]:
    """Return the same physical layout with a new legal singleton-arm binding."""

    result = canonical_data(dict(scene_spec))
    groups = single_arm_pick_requirement_groups(
        realized,
        result,
        arm_overrides=selected_arm_by_goal_variant,
        respect_scene_assignment=False,
    )
    for reach in (result.get("reach") or {}).values():
        if isinstance(reach, dict):
            reach.pop("pick_planning_requirements", None)
    for group in groups:
        entity = str(group["entity_key"])
        marker = {
            "goal_variant_id": str(group["goal_variant_id"]),
            "selected_arm": str(group["selected_arm"]),
            "required_arms": [str(group["selected_arm"])],
            "requirement_mode": "all",
            "route": "direct",
            "selection": str(group["selection"]),
            "scope": "exact_scene_planning_required",
            "planning_witness_required": True,
            "final_feasibility_proven": False,
        }
        result["reach"][entity].setdefault(
            "pick_planning_requirements", []
        ).append(marker)
    for reach in (result.get("reach") or {}).values():
        if isinstance(reach, dict) and "pick_planning_requirements" in reach:
            reach["pick_planning_requirements"] = sorted(
                reach["pick_planning_requirements"],
                key=lambda row: str(row["goal_variant_id"]),
            )
    return canonical_data(result)


def _validated_pick_witness(
    request: Mapping[str, Any] | None,
    witness: Mapping[str, Any] | None,
    *,
    required: bool,
) -> dict[str, Any] | None:
    if request is None:
        if witness is not None:
            raise ValueError("non-Pick artifact contains a Pick planning witness")
        return None
    if witness is None:
        if required:
            raise ValueError("Pick planning witness is required")
        return None
    return validate_pick_planning_witness(request, witness)


def _enforce_pick_witness_result(
    request: Mapping[str, Any],
    witness: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one passing witness or reproduce its typed gate failure.

    Failed witnesses are retained verbatim in the M12 unsat ledger.  This lets
    ``validate`` re-check the exact frozen request/witness evidence without
    launching a second simulator query.
    """

    checked = validate_pick_planning_witness(request, witness)
    if checked["passed"] is True:
        return checked
    failures = [
        row for row in checked["results"] if row["status"] != "passed"
    ]
    classification = classify_pick_planning_failure(checked)
    disposition = str(classification["disposition"])
    if disposition == "infrastructure_failure":
        code = "pick_planning_infrastructure_failed"
        message = "Pick planning oracle reported an infrastructure failure"
    elif disposition == "source_failure":
        code = "pick_planning_source_invalid"
        message = (
            "Pick planning oracle reported invalid source/Contact evidence"
        )
    elif disposition == "contract_failure":
        code = "pick_planning_failure_unclassified"
        message = "Pick planning oracle returned an unclassified failure"
    else:
        code = "pick_planning_unreachable"
        message = "tentative layout lacks a required exact-scene Pick path"
    raise LayoutSolveError(
        code,
        message,
        realized_family_id=str(request["realized_family_id"]),
        layout_seed=int(request["layout_seed"]),
        constraint_ids=["layout.pick_planning_witness"],
        blockers=failures,
        owner=str(classification["owner"]),
        metrics={
            "request_hash": request["request_hash"],
            "tentative_scene_hash": request["tentative_scene_hash"],
            "witness_hash": checked["witness_hash"],
            "failed_arm_count": len(failures),
            "failure_classification": classification,
            "frozen_request": canonical_data(dict(request)),
            "frozen_witness": canonical_data(dict(checked)),
        },
    )


def _invoke_pick_witness_provider(
    provider: PickPlanningWitnessProvider | None,
    request: Mapping[str, Any] | None,
    *,
    required: bool,
    context: Mapping[str, Any],
) -> dict[str, Any] | None:
    if request is None:
        return None
    if provider is None:
        if required:
            raise LayoutSolveError(
                "pick_planning_witness_provider_missing",
                "strict Pick planning gate requires a simulator planning provider",
                realized_family_id=str(request["realized_family_id"]),
                layout_seed=int(request["layout_seed"]),
                owner="planning_oracle",
            )
        return None
    try:
        supplied = provider(
            canonical_data(dict(request)),
            context=canonical_data(dict(context)),
        )
    except PickPlanningProviderError as exc:
        disposition = str(exc.disposition)
        code = (
            "pick_planning_source_invalid"
            if disposition == "source_failure"
            else "pick_planning_infrastructure_failed"
            if disposition == "infrastructure_failure"
            else "pick_planning_witness_invalid"
        )
        owner = (
            "unit_builder"
            if disposition == "source_failure"
            else "planning_oracle"
        )
        raise LayoutSolveError(
            code,
            f"Pick planning provider {disposition}: {exc}",
            realized_family_id=str(request["realized_family_id"]),
            layout_seed=int(request["layout_seed"]),
            owner=owner,
            metrics={"provider_disposition": disposition},
        ) from exc
    except Exception as exc:
        raise LayoutSolveError(
            "pick_planning_witness_provider_failed",
            f"Pick planning provider raised {type(exc).__name__}: {exc}",
            realized_family_id=str(request["realized_family_id"]),
            layout_seed=int(request["layout_seed"]),
            owner="planning_oracle",
        ) from exc
    if not isinstance(supplied, Mapping):
        raise LayoutSolveError(
            "pick_planning_witness_invalid",
            "Pick planning provider returned a non-mapping witness",
            realized_family_id=str(request["realized_family_id"]),
            layout_seed=int(request["layout_seed"]),
            owner="planning_oracle",
        )
    try:
        witness = validate_pick_planning_witness(request, supplied)
    except ValueError as exc:
        raise LayoutSolveError(
            "pick_planning_witness_invalid",
            f"Pick planning witness failed validation: {exc}",
            realized_family_id=str(request["realized_family_id"]),
            layout_seed=int(request["layout_seed"]),
            owner="planning_oracle",
        ) from exc
    return _enforce_pick_witness_result(request, witness)


def _terminal_reach_requirements(
    realized: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    arms_by_goal: dict[tuple[str, str], set[str]] = defaultdict(set)
    for variant in realized.get("goal_variants", []):
        goal = variant.get("goal") or {}
        subject = goal.get("subject")
        relation = goal.get("relation")
        if not subject or not relation:
            continue
        arm = goal.get("arm")
        arms_by_goal[(str(subject), str(relation))].add(
            "any" if arm is None else str(arm)
        )
    requirements: dict[str, dict[str, Any]] = {}
    for unit in units:
        unit_id = str(unit["id"])
        for region in (unit.get("params") or {}).get(
            "reserved_regions", []
        ):
            region_id = str(region.get("id") or "")
            subject = str(
                region.get("subject_key") or region.get("subject") or ""
            )
            relation = str(region.get("relation") or "")
            arms = arms_by_goal.get((subject, relation), set())
            if not region_id or not arms:
                continue
            requirement = (
                "both"
                if {"left", "right"} <= arms
                else "left"
                if "left" in arms
                else "right"
                if "right" in arms
                else "any"
            )
            requirements[region_id] = {
                "unit_id": unit_id,
                "subject": subject,
                "relation": relation,
                "required_arm": requirement,
                "center_xy": list(
                    (region.get("footprint") or {}).get(
                        "center_xy", [0.0, 0.0]
                    )
                ),
            }
    return requirements


def _in_reach(point: Sequence[float], arm: str, config: LayoutConfig) -> bool:
    def inside(name: str) -> bool:
        region = config.data["reach"][name]
        return (
            float(region["x"][0]) - GEOMETRY_EPSILON
            <= float(point[0])
            <= float(region["x"][1]) + GEOMETRY_EPSILON
            and float(region["y"][0]) - GEOMETRY_EPSILON
            <= float(point[1])
            <= float(region["y"][1]) + GEOMETRY_EPSILON
        )

    if arm == "left":
        return inside("left")
    if arm == "right":
        return inside("right")
    if arm == "both":
        return inside("left") and inside("right")
    return inside("left") or inside("right")


def _reach_region_bounds(
    requirement: str, config: LayoutConfig
) -> list[tuple[float, float, float, float]]:
    """Return the rectangular components of one static reach requirement."""

    def bounds(arm: str) -> tuple[float, float, float, float]:
        region = config.data["reach"][arm]
        return (
            float(region["x"][0]),
            float(region["x"][1]),
            float(region["y"][0]),
            float(region["y"][1]),
        )

    left, right = bounds("left"), bounds("right")
    if requirement == "left":
        return [left]
    if requirement == "right":
        return [right]
    if requirement == "both":
        intersection = _intersect_anchor_bounds(left, right)
        return [] if intersection is None else [intersection]
    return _maximal_anchor_bounds([left, right])


def _entity_operation_offsets_xy(
    entity: str,
    unit_by_entity: Mapping[str, str],
    pose_by_entity: Mapping[str, Mapping[str, Any]],
    operation_contracts: Mapping[str, Mapping[str, Any]],
    kinds: set[str],
    contact_ids: set[int] | None = None,
) -> list[tuple[float, float]]:
    """Transform eligible actor-local anchors into the shared Unit frame."""

    if entity not in unit_by_entity:
        return []
    contract = operation_contracts.get(entity)
    if not isinstance(contract, Mapping):
        return []
    local_pose = pose_by_entity[entity]
    translation = local_pose["translation_xyz"]
    yaw_rad = math.radians(float(local_pose.get("relative_yaw_deg", 0.0)))
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    result: set[tuple[float, float]] = set()
    for row in contract.get("anchors", []):
        if str(row.get("kind")) not in kinds:
            continue
        if contact_ids is not None and (
            str(row.get("kind")) != "grasp_contact"
            or row.get("point_index") is None
            or int(row["point_index"]) not in contact_ids
        ):
            continue
        if row.get("operation_eligible", True) is not True:
            continue
        position = row.get("position_xyz_m") or []
        if len(position) != 3:
            continue
        local_x = cosine * float(position[0]) - sine * float(position[1])
        local_y = sine * float(position[0]) + cosine * float(position[1])
        result.add(
            (
                _clean(float(translation[0]) + local_x),
                _clean(float(translation[1]) + local_y),
            )
        )
    return sorted(result)


def _entity_position_xy(
    entity: str,
    placements: Mapping[str, Sequence[float]],
    unit_by_entity: Mapping[str, str],
    pose_by_entity: Mapping[str, Mapping[str, Any]],
) -> tuple[float, float] | None:
    unit_id = unit_by_entity.get(entity)
    if unit_id not in placements:
        return None
    local = pose_by_entity[entity]["translation_xyz"]
    anchor = placements[unit_id]
    return (
        float(anchor[0]) + float(local[0]),
        float(anchor[1]) + float(local[1]),
    )


def _entity_operation_positions_xy(
    entity: str,
    placements: Mapping[str, Sequence[float]],
    unit_by_entity: Mapping[str, str],
    pose_by_entity: Mapping[str, Mapping[str, Any]],
    operation_contracts: Mapping[str, Mapping[str, Any]],
    kinds: set[str],
) -> list[tuple[float, float]]:
    unit_id = unit_by_entity.get(entity)
    if unit_id not in placements:
        return []
    unit_anchor = placements[unit_id]
    return [
        (
            float(unit_anchor[0]) + offset[0],
            float(unit_anchor[1]) + offset[1],
        )
        for offset in _entity_operation_offsets_xy(
            entity,
            unit_by_entity,
            pose_by_entity,
            operation_contracts,
            kinds,
        )
    ]


def _landmark_reference(realized: Mapping[str, Any]) -> str | None:
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("basis") == "landmark" and value.get("reference"):
                found.add(str(value["reference"]))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for variant in realized.get("goal_variants", []):
        visit((variant.get("goal") or {}).get("descriptors") or {})
    return next(iter(found)) if len(found) == 1 else None


def _constraints_hold(
    realized: Mapping[str, Any],
    placements: Mapping[str, Sequence[float]],
    unit_by_entity: Mapping[str, str],
    pose_by_entity: Mapping[str, Mapping[str, Any]],
    reach_requirements: Mapping[str, str],
    terminal_reach_requirements: Mapping[str, Mapping[str, Any]],
    config: LayoutConfig,
    operation_anchor_kinds: Mapping[str, set[str]] | None = None,
    operation_contracts: Mapping[str, Mapping[str, Any]] | None = None,
    articulation_reach_groups: Mapping[
        str, Sequence[Mapping[str, Any]]
    ]
    | None = None,
) -> bool:
    operation_anchor_kinds = operation_anchor_kinds or {}
    operation_contracts = operation_contracts or {}
    articulation_reach_groups = articulation_reach_groups or {}
    for entity, requirement in reach_requirements.items():
        # The constructive solver checks partial placements one Unit at a
        # time.  An active entity whose Unit has not been placed yet is not a
        # missing-anchor case; defer it until that Unit enters the partial
        # assignment.
        unit_id = unit_by_entity.get(entity)
        if unit_id not in placements:
            continue
        if entity in articulation_reach_groups:
            for group in articulation_reach_groups[entity]:
                group_passed = False
                for option in group.get("options", []):
                    offsets = _entity_operation_offsets_xy(
                        entity,
                        unit_by_entity,
                        pose_by_entity,
                        operation_contracts,
                        {"grasp_contact"},
                        contact_ids={int(option["contact_id"])},
                    )
                    for offset in offsets:
                        point = (
                            float(placements[unit_id][0]) + offset[0],
                            float(placements[unit_id][1]) + offset[1],
                        )
                        if _in_reach(point, str(option["arm"]), config):
                            group_passed = True
                            break
                    if group_passed:
                        break
                if not group_passed:
                    return False
            continue
        required_anchor_kinds = operation_anchor_kinds.get(
            entity, {"grasp_contact"}
        )
        points = _entity_operation_positions_xy(
            entity,
            placements,
            unit_by_entity,
            pose_by_entity,
            operation_contracts,
            required_anchor_kinds,
        )
        if not points and entity in operation_anchor_kinds:
            return False
        if not points:
            fallback = _entity_position_xy(
                entity, placements, unit_by_entity, pose_by_entity
            )
            points = [] if fallback is None else [fallback]
        if points and not any(
            _in_reach(point, requirement, config) for point in points
        ):
            return False
    for row in terminal_reach_requirements.values():
        unit_id = str(row["unit_id"])
        if unit_id not in placements:
            continue
        anchor = placements[unit_id]
        center = row["center_xy"]
        point = (
            float(anchor[0]) + float(center[0]),
            float(anchor[1]) + float(center[1]),
        )
        if not _in_reach(point, str(row["required_arm"]), config):
            return False
    # A direct transfer is one single-arm task.  Independent ``any`` checks
    # for the acquisition point and terminal point are insufficient because
    # they can be satisfied by opposite arms and silently imply handover.
    # Require at least one concrete arm to satisfy both ends of every O2O
    # GoalVariant; D4 left/right variants therefore remain two independent
    # singleton checks on the same SceneSpec.
    for variant in realized.get("goal_variants", []):
        goal = variant.get("goal") or {}
        if str(goal.get("action_family") or "") != "O2O_relation":
            continue
        entity = str(goal.get("subject") or "")
        relation = str(goal.get("relation") or "")
        unit_id = unit_by_entity.get(entity)
        if not entity or not relation or unit_id not in placements:
            continue
        points = _entity_operation_positions_xy(
            entity,
            placements,
            unit_by_entity,
            pose_by_entity,
            operation_contracts,
            operation_anchor_kinds.get(entity, {"grasp_contact"}),
        )
        if not points:
            return False
        terminal_points: list[tuple[float, float]] = []
        terminal_incomplete = False
        for row in terminal_reach_requirements.values():
            if (
                str(row.get("subject") or "") != entity
                or str(row.get("relation") or "") != relation
            ):
                continue
            terminal_unit_id = str(row["unit_id"])
            if terminal_unit_id not in placements:
                terminal_incomplete = True
                break
            terminal_anchor = placements[terminal_unit_id]
            center = row["center_xy"]
            terminal_points.append(
                (
                    float(terminal_anchor[0]) + float(center[0]),
                    float(terminal_anchor[1]) + float(center[1]),
                )
            )
        if terminal_incomplete:
            continue
        if not terminal_points:
            return False
        semantic_arm = goal.get("arm")
        candidate_arms = (
            [str(semantic_arm)]
            if semantic_arm is not None
            else ["left", "right"]
        )
        effective_requirement = str(reach_requirements.get(entity) or "any")
        if effective_requirement in {"left", "right"}:
            candidate_arms = [
                arm for arm in candidate_arms if arm == effective_requirement
            ]
        if not any(
            any(_in_reach(point, arm, config) for point in points)
            and all(_in_reach(point, arm, config) for point in terminal_points)
            for arm in candidate_arms
        ):
            return False
    role_entities = _role_entities(realized)
    for constraint in realized.get("layout_constraints", []):
        if constraint.get("type") != "ordering_constraint":
            continue
        params = constraint.get("params") or {}
        basis = str(params.get("basis"))
        if basis == "size":
            continue
        entities = [
            role_entities[str(role)]
            for role in constraint.get("roles", [])
            if str(role) in role_entities
        ]
        if len(entities) != 2:
            return False
        points = [
            _entity_position_xy(
                entity, placements, unit_by_entity, pose_by_entity
            )
            for entity in entities
        ]
        if any(point is None for point in points):
            continue
        first, second = points  # type: ignore[misc]
        if basis == "position":
            gap = float(config.data["ordering"]["position_gap_m"])
            if float(first[0]) + gap > float(second[0]):
                return False
        elif basis in {"robot", "landmark"}:
            gap = float(config.data["ordering"]["distance_gap_m"])
            if basis == "robot":
                reference = config.data["actor"]["base_pose"]["position"][:2]
            else:
                reference_entity = _landmark_reference(realized)
                if reference_entity is None:
                    return False
                reference = _entity_position_xy(
                    reference_entity,
                    placements,
                    unit_by_entity,
                    pose_by_entity,
                )
                if reference is None:
                    continue
            first_distance = math.dist(first, reference)
            second_distance = math.dist(second, reference)
            if first_distance + gap > second_distance:
                return False
    return True


def _shift_reach_bounds_to_anchor(
    bounds: tuple[float, float, float, float],
    offset: Sequence[float],
) -> tuple[float, float, float, float]:
    return (
        _clean(bounds[0] - float(offset[0])),
        _clean(bounds[1] - float(offset[0])),
        _clean(bounds[2] - float(offset[1])),
        _clean(bounds[3] - float(offset[1])),
    )


def _unit_anchor_constraint_groups(
    unit: Mapping[str, Any],
    unit_by_entity: Mapping[str, str],
    pose_by_entity: Mapping[str, Mapping[str, Any]],
    reach_requirements: Mapping[str, str],
    terminal_reach_requirements: Mapping[str, Mapping[str, Any]],
    config: LayoutConfig,
    *,
    operation_anchor_kinds: Mapping[str, set[str]],
    operation_contracts: Mapping[str, Mapping[str, Any]],
    articulation_reach_groups: Mapping[
        str, Sequence[Mapping[str, Any]]
    ]
    | None = None,
) -> list[dict[str, Any]]:
    """Build exact Unit-anchor AABB unions for static operation reach."""

    unit_id = str(unit["id"])
    articulation_reach_groups = articulation_reach_groups or {}
    groups: list[dict[str, Any]] = []
    for entity in sorted(
        set(str(value) for value in unit.get("entity_keys", []))
        & set(reach_requirements)
    ):
        if entity in articulation_reach_groups:
            for articulation_group in articulation_reach_groups[entity]:
                options: list[tuple[float, float, float, float]] = []
                for option in articulation_group.get("options", []):
                    offsets = _entity_operation_offsets_xy(
                        entity,
                        unit_by_entity,
                        pose_by_entity,
                        operation_contracts,
                        {"grasp_contact"},
                        contact_ids={int(option["contact_id"])},
                    )
                    reach_regions = _reach_region_bounds(
                        str(option["arm"]), config
                    )
                    options.extend(
                        _shift_reach_bounds_to_anchor(region, offset)
                        for offset in offsets
                        for region in reach_regions
                    )
                groups.append(
                    {
                        "id": str(articulation_group["id"]),
                        "kind": (
                            "touch_operation_reach"
                            if articulation_group.get("kind") == "touch"
                            else "articulation_operation_reach"
                        ),
                        "entity_key": entity,
                        "use": str(articulation_group["use"]),
                        "requirement": "per_arm_contact",
                        "options": _maximal_anchor_bounds(options),
                    }
                )
            continue
        kinds = operation_anchor_kinds.get(entity, {"grasp_contact"})
        offsets = _entity_operation_offsets_xy(
            entity,
            unit_by_entity,
            pose_by_entity,
            operation_contracts,
            kinds,
        )
        if not offsets and entity not in operation_anchor_kinds:
            local = pose_by_entity[entity]["translation_xyz"]
            offsets = [(_clean(local[0]), _clean(local[1]))]
        reach_regions = _reach_region_bounds(
            str(reach_requirements[entity]), config
        )
        options = _maximal_anchor_bounds(
            [
                _shift_reach_bounds_to_anchor(region, offset)
                for offset in offsets
                for region in reach_regions
            ]
        )
        groups.append(
            {
                "id": f"operation_reach:{entity}",
                "kind": "operation_reach",
                "entity_key": entity,
                "requirement": str(reach_requirements[entity]),
                "options": options,
            }
        )
    for region_id, row in sorted(terminal_reach_requirements.items()):
        if str(row["unit_id"]) != unit_id:
            continue
        reach_regions = _reach_region_bounds(
            str(row["required_arm"]), config
        )
        center = row["center_xy"]
        groups.append(
            {
                "id": f"terminal_reach:{region_id}",
                "kind": "terminal_reach",
                "region_id": region_id,
                "requirement": str(row["required_arm"]),
                "options": _maximal_anchor_bounds(
                    [
                        _shift_reach_bounds_to_anchor(region, center)
                        for region in reach_regions
                    ]
                ),
            }
        )
    return groups


def _dynamic_candidates(
    unit: Mapping[str, Any],
    base: Sequence[tuple[float, float]],
    placements: Mapping[str, Sequence[float]],
    unit_by_id: Mapping[str, Mapping[str, Any]],
    margin: float,
    *,
    realized_family_id: str,
    layout_seed: int,
    anchor_domains: Sequence[tuple[float, float, float, float]] = (),
    config: LayoutConfig | None = None,
) -> list[tuple[float, float]]:
    """Put exact Unit tangencies ahead without an axis-ordering bias."""

    left, right, bottom, top = _unit_bounds_local(unit)
    base_points = list(dict.fromkeys(base))
    tangent_points: set[tuple[float, float]] = set()
    xs = {point[0] for point in base}
    ys = {point[1] for point in base}
    camera_halfplanes = (
        _camera_anchor_halfplanes(
            (left, right, bottom, top), config.camera_polygon
        )
        if config is not None
        else []
    )
    all_tangent_x: set[float] = set()
    all_tangent_y: set[float] = set()

    def exact_line(axis: str, value: float) -> None:
        if not anchor_domains or config is None:
            return
        equality = (
            [(1.0, 0.0, value), (-1.0, 0.0, -value)]
            if axis == "x"
            else [(0.0, 1.0, value), (0.0, -1.0, -value)]
        )
        for domain in anchor_domains:
            candidates, _dimension = _halfplane_domain_candidates(
                [
                    *_rectangle_halfplanes(domain),
                    *camera_halfplanes,
                    *equality,
                ]
            )
            tangent_points.update(candidates)

    for placed_id, placed_position in placements.items():
        placed_bounds = _global_bounds(
            _unit_bounds_local(unit_by_id[placed_id]), placed_position
        )
        tangent_x = (
            placed_bounds[0] - margin - right,
            placed_bounds[1] + margin - left,
        )
        tangent_y = (
            placed_bounds[2] - margin - top,
            placed_bounds[3] + margin - bottom,
        )
        all_tangent_x.update(tangent_x)
        all_tangent_y.update(tangent_y)
        preferred_y = (
            float(placed_position[1]),
            min(ys, key=lambda value: abs(value - float(placed_position[1]))),
        )
        preferred_x = (
            float(placed_position[0]),
            min(xs, key=lambda value: abs(value - float(placed_position[0]))),
        )
        tangent_points.update((x, y) for x in tangent_x for y in preferred_y)
        tangent_points.update((x, y) for x in preferred_x for y in tangent_y)
    # Cross-axis tangencies may be owned by different already-placed Units.
    # Their Cartesian intersections are the exact packing corners that a
    # whole-table sampler cannot be expected to hit.
    tangent_points.update(
        (x, y) for x in all_tangent_x for y in all_tangent_y
    )
    for value in all_tangent_x:
        exact_line("x", value)
    for value in all_tangent_y:
        exact_line("y", value)
    if anchor_domains and config is not None:
        tangent_points = {
            point
            for point in tangent_points
            if any(
                domain[0] - GEOMETRY_EPSILON
                <= float(point[0])
                <= domain[1] + GEOMETRY_EPSILON
                and domain[2] - GEOMETRY_EPSILON
                <= float(point[1])
                <= domain[3] + GEOMETRY_EPSILON
                for domain in anchor_domains
            )
            and _bounds_in_polygon(
                _global_bounds(
                    (left, right, bottom, top), point
                ),
                config.camera_polygon,
            )
        }
    extras = {
        (_clean(point[0]), _clean(point[1])) for point in tangent_points
    } - set(base_points)
    unit_id = str(unit["id"])
    extras = sorted(
        extras,
        key=lambda point: (
            artifact_hash(
                {
                    "realized_family_id": str(realized_family_id),
                    "unit_id": unit_id,
                    "layout_seed": int(layout_seed),
                    "point_xy": _vec(point),
                }
            ),
            point,
        ),
    )
    return [*extras, *base_points]


def _unit_placement_hash(
    placements: Mapping[str, Sequence[float]],
    goal_variant_ids: Sequence[str] | None = None,
) -> str:
    """Return the canonical identity of one complete Unit-anchor layout."""

    return artifact_hash(
        {
            **({"goal_variant_ids": sorted(goal_variant_ids)} if goal_variant_ids is not None else {}),
            "unit_placements": [
                {
                    "unit_id": str(unit_id),
                    "position_xy": _vec(
                        [float(position[0]), float(position[1])]
                    ),
                }
                for unit_id, position in sorted(placements.items())
            ]
        }
    )


def _scene_unit_placement_hash(scene_spec: Mapping[str, Any]) -> str:
    placements: dict[str, Sequence[float]] = {}
    for row in scene_spec.get("unit_poses") or []:
        if not isinstance(row, Mapping):
            raise ValueError("SceneSpec unit_poses contains a non-mapping row")
        position = row.get("position")
        if not isinstance(position, Sequence) or len(position) < 2:
            raise ValueError("SceneSpec Unit pose lacks an XY position")
        placements[str(row["unit_id"])] = position
    if not placements:
        raise ValueError("SceneSpec contains no Unit poses")
    metadata = (scene_spec.get("provenance") or {}).get("metadata") or {}
    ids = scene_spec.get("goal_variant_ids") if metadata.get("d2b_task_domain") else None
    return _unit_placement_hash(placements, ids)


def _solve_unit_poses(
    realized: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    layout_seed: int,
    config: LayoutConfig,
    *,
    rejected_placement_hashes: Sequence[str] = (),
    operation_gate=None,
) -> dict[str, tuple[float, float]]:
    realized_id = str(realized["id"])
    unit_by_id = {str(unit["id"]): unit for unit in units}
    unit_by_entity, pose_by_entity = _entity_units(units)
    reach_requirements = _effective_reach_requirements(realized, units)
    operation_anchor_kinds = _operation_anchor_kinds(realized)
    operation_contracts = _operation_anchor_contracts(units)
    articulation_reach_groups = _operation_reach_groups(
        realized, units
    )
    terminal_reach_requirements = _terminal_reach_requirements(
        realized, units
    )
    candidates: dict[str, list[tuple[float, float]]] = {}
    candidate_metrics: dict[str, dict[str, Any]] = {}
    anchor_domains_by_unit: dict[
        str, list[tuple[float, float, float, float]]
    ] = {}
    for unit in units:
        unit_id = str(unit["id"])
        constraint_groups = _unit_anchor_constraint_groups(
            unit,
            unit_by_entity,
            pose_by_entity,
            reach_requirements,
            terminal_reach_requirements,
            config,
            operation_anchor_kinds=operation_anchor_kinds,
            operation_contracts=operation_contracts,
            articulation_reach_groups=articulation_reach_groups,
        )
        wx0, wx1, wy0, wy1 = config.workspace
        left, right, bottom, top = _unit_bounds_local(unit)
        workspace_anchor_bounds = (
            wx0 - left,
            wx1 - right,
            wy0 - bottom,
            wy1 - top,
        )
        anchor_domains = (
            []
            if workspace_anchor_bounds[0] > workspace_anchor_bounds[1]
            or workspace_anchor_bounds[2] > workspace_anchor_bounds[3]
            else _anchor_domain_rectangles(
                workspace_anchor_bounds, constraint_groups
            )
        )
        anchor_domains_by_unit[unit_id] = anchor_domains
        points, metrics = _candidate_positions(
            realized_id,
            unit,
            layout_seed,
            config,
            constraint_groups=constraint_groups,
            anchor_domains=anchor_domains,
        )
        candidate_metrics[unit_id] = metrics
        if not points:
            wx0, wx1, wy0, wy1 = config.workspace
            left, right, bottom, top = _unit_bounds_local(unit)
            too_large = right - left > wx1 - wx0 or top - bottom > wy1 - wy0
            code = (
                "footprint_outside_workspace" if too_large else "camera_bounds"
            )
            blocker = "workspace" if too_large else "camera"
            seed_invariant_proof = bool(too_large or not anchor_domains)
            raise LayoutSolveError(
                code,
                f"{unit_id} has no anchor inside {blocker} bounds",
                realized_family_id=realized_id,
                layout_seed=layout_seed,
                unit_ids=[unit_id],
                constraint_ids=[
                    (
                        "layout.workspace_containment"
                        if too_large
                        else "layout.camera_containment"
                    )
                ],
                blockers=[
                    {
                        "constraint_type": blocker,
                        "blocked_unit_id": unit_id,
                    }
                ],
                owner="layout",
                metrics={
                    **metrics,
                    "seed_invariant_proof": seed_invariant_proof,
                    "seed_invariant_reason": (
                        "unit_footprint_exceeds_workspace"
                        if too_large
                        else "anchor_constraint_intersection_empty"
                        if not anchor_domains
                        else None
                    ),
                },
            )
        candidates[unit_id] = points
    for unit in units:
        unit_id = str(unit["id"])
        reachable_points = [
            point
            for point in candidates[unit_id]
            if _constraints_hold(
                realized,
                {unit_id: point},
                unit_by_entity,
                pose_by_entity,
                reach_requirements,
                terminal_reach_requirements,
                config,
                operation_anchor_kinds=operation_anchor_kinds,
                operation_contracts=operation_contracts,
                articulation_reach_groups=articulation_reach_groups,
            )
        ]
        required_entities = sorted(
            set(unit["entity_keys"]) & set(reach_requirements)
        )
        required_regions = sorted(
            region_id
            for region_id, row in terminal_reach_requirements.items()
            if str(row["unit_id"]) == unit_id
        )
        if (required_entities or required_regions) and not reachable_points:
            code = (
                "terminal_reach_bounds"
                if required_regions and not required_entities
                else "reach_bounds"
            )
            raise LayoutSolveError(
                code,
                f"{unit_id} has no anchor satisfying static reach requirements",
                realized_family_id=realized_id,
                layout_seed=layout_seed,
                unit_ids=[unit_id],
                constraint_ids=[
                    "layout.static_reach_proxy",
                    *[
                        str(row["id"])
                        for row in realized.get("layout_constraints", [])
                        if row.get("type") == "reach_zone"
                        and set(row.get("roles", []))
                        & set(unit.get("roles", []))
                    ],
                ],
                blockers=[
                    {
                        "constraint_type": "reach",
                        "blocked_unit_id": unit_id,
                        "entity_keys": required_entities,
                        "terminal_region_ids": required_regions,
                    }
                ],
                metrics={
                    **candidate_metrics[unit_id],
                    "reach_requirements": {
                        entity: reach_requirements[entity]
                        for entity in required_entities
                    },
                },
            )
        candidates[unit_id] = reachable_points or candidates[unit_id]
    priorities = {
        "ReceiverReservationUnit": 0,
        "LandmarkRelationUnit": 1,
        "ObjectUnit": 2,
    }

    ordering_roles = Counter(
        str(role)
        for constraint in realized.get("layout_constraints", [])
        if constraint.get("type") == "ordering_constraint"
        for role in constraint.get("roles", [])
    )
    role_entities = _role_entities(realized)
    entity_roles = {entity: role for role, entity in role_entities.items()}

    def constraint_strength(unit: Mapping[str, Any]) -> int:
        return sum(
            2 * int(entity in reach_requirements)
            + ordering_roles.get(entity_roles.get(str(entity), ""), 0)
            for entity in unit.get("entity_keys", [])
        )

    ordered_units = sorted(
        units,
        key=lambda unit: (
            priorities[str(unit["type"])],
            -(
                (
                    _unit_bounds_local(unit)[1]
                    - _unit_bounds_local(unit)[0]
                )
                * (
                    _unit_bounds_local(unit)[3]
                    - _unit_bounds_local(unit)[2]
                )
            ),
            -constraint_strength(unit),
            artifact_hash(
                {
                    "unit_id": unit["id"],
                    "layout_seed": layout_seed,
                }
            ),
        ),
    )
    placements: dict[str, tuple[float, float]] = {}
    rejected_hashes = frozenset(
        str(value) for value in rejected_placement_hashes
    )
    nodes = 0
    max_nodes = int(config.data["search"]["max_nodes"])
    budget_exhausted = False

    from .initial_candidates import InitialCandidatePolicy, POLICY as INITIAL_POLICY
    from .planner_proxy_candidates import NativeProxyCandidatePolicy, POLICY as PROXY_POLICY
    from .sparse_candidates import SparseNativeProxyCandidatePolicy, POLICY as SPARSE_POLICY
    initial_policy = None
    policy_metadata = INITIAL_POLICY
    if operation_gate is not None and operation_gate.enabled:
        selected_policy = config.data.get("search", {}).get("initial_candidate_policy", "baseline")
        if selected_policy == INITIAL_POLICY["version"]:
            initial_policy = InitialCandidatePolicy(operation_gate)
        elif selected_policy == PROXY_POLICY["version"]:
            initial_policy = NativeProxyCandidatePolicy(operation_gate)
            policy_metadata = PROXY_POLICY
        elif selected_policy == SPARSE_POLICY["version"]:
            initial_policy = SparseNativeProxyCandidatePolicy(operation_gate)
            policy_metadata = SPARSE_POLICY
    portfolio = []

    def visit(index: int) -> bool:
        nonlocal nodes, budget_exhausted
        if index == len(ordered_units):
            passed = _constraints_hold(
                realized,
                placements,
                unit_by_entity,
                pose_by_entity,
                reach_requirements,
                terminal_reach_requirements,
                config,
                operation_anchor_kinds=operation_anchor_kinds,
                operation_contracts=operation_contracts,
                articulation_reach_groups=articulation_reach_groups,
            ) and _not_success_holds(units, placements)
            admitted = (
                passed
                and _unit_placement_hash(placements, [v["id"] for v in realized["goal_variants"]] if operation_gate and operation_gate.enabled else None) not in rejected_hashes
                and (operation_gate is None or operation_gate.evaluate(placements))
            )
            if admitted and initial_policy is not None:
                portfolio.append((initial_policy.score(placements),
                                  artifact_hash(dict(placements)), dict(placements)))
                return len(portfolio) >= policy_metadata["feasible_portfolio_size"]
            return admitted
        unit = ordered_units[index]
        unit_id = str(unit["id"])
        dynamic_points = _dynamic_candidates(
            unit,
            candidates[unit_id],
            placements,
            unit_by_id,
            config.margin,
            realized_family_id=realized_id,
            layout_seed=layout_seed,
            anchor_domains=anchor_domains_by_unit[unit_id],
            config=config,
        )
        if operation_gate is not None:
            dynamic_points = operation_gate.proposal_points(unit, placements, dynamic_points, layout_seed)
        if initial_policy is not None:
            dynamic_points = initial_policy.order(
                unit, placements, dynamic_points, layout_seed,
                anchor_domains=anchor_domains_by_unit[unit_id])
        for point in dynamic_points:
            nodes += 1
            if nodes > max_nodes:
                budget_exhausted = True
                return False
            bounds = _global_bounds(_unit_bounds_local(unit), point)
            if not _bounds_in_workspace(
                bounds, config
            ) or not _bounds_in_polygon(bounds, config.camera_polygon):
                continue
            if (initial_policy.overlaps(unit, point, placements) if initial_policy is not None else any(
                _overlap_with_margin(
                    bounds,
                    _global_bounds(
                        _unit_bounds_local(unit_by_id[other_id]),
                        other_position,
                    ),
                    config.margin,
                )
                for other_id, other_position in placements.items()
            )):
                continue
            placements[unit_id] = point
            if _constraints_hold(
                realized,
                placements,
                unit_by_entity,
                pose_by_entity,
                reach_requirements,
                terminal_reach_requirements,
                config,
                operation_anchor_kinds=operation_anchor_kinds,
                operation_contracts=operation_contracts,
                articulation_reach_groups=articulation_reach_groups,
            ) and visit(index + 1):
                return True
            placements.pop(unit_id)
        return False

    from .manipulation import OperationSpaceSearchExhausted
    try:
        solved = visit(0)
    except OperationSpaceSearchExhausted:
        solved = False
    if portfolio:
        score, _, placements = min(portfolio, key=lambda row: row[:2])
        operation_gate.evaluate(placements)  # Restore options for the chosen scene.
        operation_gate.initial_candidate_receipt = {
            "policy": policy_metadata, "feasible_layouts_ranked": len(portfolio),
            "operation_layouts_evaluated": len(operation_gate.cache),
            "score": list(score), "is_planning_proof": False,
        }
        if hasattr(initial_policy, "sampling_receipt"):
            operation_gate.initial_candidate_receipt["sampling"] = initial_policy.sampling_receipt()
        solved = True
    if not solved:
        code = (
            "search_budget_exhausted"
            if budget_exhausted
            else ("operation_space_search_exhausted"
                  if operation_gate is not None and operation_gate.enabled
                  else "layout_search_exhausted")
        )
        constraint_ids = [
            "layout.unit_non_overlap",
            "layout.unit_margin",
            *[
                str(row["id"])
                for row in realized.get("layout_constraints", [])
            ],
        ]
        raise LayoutSolveError(
            code,
            f"deterministic layout search exhausted after {nodes} nodes",
            realized_family_id=realized_id,
            layout_seed=layout_seed,
            unit_ids=[str(unit["id"]) for unit in ordered_units],
            constraint_ids=constraint_ids,
            blockers=[
                {
                    "constraint_type": "footprint/order/reservation/reach",
                    "blocked_unit_id": str(unit["id"]),
                    "candidate_count": len(candidates[str(unit["id"])]),
                }
                for unit in ordered_units
            ],
            metrics={
                "nodes": nodes,
                "max_nodes": max_nodes,
                "candidate_domains": candidate_metrics,
            },
        )
    return placements


def _quat_multiply(first: Sequence[float], second: Sequence[float]) -> list[float]:
    w1, x1, y1, z1 = [float(value) for value in first]
    w2, x2, y2, z2 = [float(value) for value in second]
    return _vec(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _yaw_quat(degrees: float) -> list[float]:
    radians = math.radians(float(degrees)) / 2.0
    return [_clean(math.cos(radians)), 0.0, 0.0, _clean(math.sin(radians))]


def _camera_quat(camera: Mapping[str, Any]) -> list[float]:
    forward = [float(value) for value in camera["forward"]]
    left = [float(value) for value in camera["left"]]
    forward_norm = math.sqrt(sum(value * value for value in forward))
    left_norm = math.sqrt(sum(value * value for value in left))
    forward = [value / forward_norm for value in forward]
    left = [value / left_norm for value in left]
    up = [
        forward[1] * left[2] - forward[2] * left[1],
        forward[2] * left[0] - forward[0] * left[2],
        forward[0] * left[1] - forward[1] * left[0],
    ]
    matrix = [
        [forward[0], left[0], up[0]],
        [forward[1], left[1], up[1]],
        [forward[2], left[2], up[2]],
    ]
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = [
            0.25 * scale,
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
        ]
    else:
        diagonal = max(range(3), key=lambda index: matrix[index][index])
        if diagonal == 0:
            scale = (
                math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2
            )
            quat = [
                (matrix[2][1] - matrix[1][2]) / scale,
                0.25 * scale,
                (matrix[0][1] + matrix[1][0]) / scale,
                (matrix[0][2] + matrix[2][0]) / scale,
            ]
        elif diagonal == 1:
            scale = (
                math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2
            )
            quat = [
                (matrix[0][2] - matrix[2][0]) / scale,
                (matrix[0][1] + matrix[1][0]) / scale,
                0.25 * scale,
                (matrix[1][2] + matrix[2][1]) / scale,
            ]
        else:
            scale = (
                math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2
            )
            quat = [
                (matrix[1][0] - matrix[0][1]) / scale,
                (matrix[0][2] + matrix[2][0]) / scale,
                (matrix[1][2] + matrix[2][1]) / scale,
                0.25 * scale,
            ]
    norm = math.sqrt(sum(value * value for value in quat))
    return _vec(value / norm for value in quat)


def _reach_metadata(point: Sequence[float], config: LayoutConfig) -> dict[str, Any]:
    left = _in_reach(point, "left", config)
    right = _in_reach(point, "right", config)
    arms = [name for name, value in (("left", left), ("right", right)) if value]
    zone = "both" if left and right else f"{arms[0]}_only" if arms else "none"
    return {
        "reachable_arms": arms,
        "zone": zone,
        "evidence": "benchmark_v1_static_reach_proxy_v1",
    }


def _operation_reach_metadata(
    points: Sequence[Sequence[float]], config: LayoutConfig
) -> dict[str, Any]:
    arms = [
        arm
        for arm in ("left", "right")
        if any(_in_reach(point, arm, config) for point in points)
    ]
    zone = "both" if len(arms) == 2 else f"{arms[0]}_only" if arms else "none"
    return {
        "reachable_arms": arms,
        "zone": zone,
        "evidence": "benchmark_v1_operation_anchor_reach_proxy_v2",
    }


def _entity_footprints(
    units: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for unit in units:
        params = unit["params"]
        if unit["type"] == "ObjectUnit":
            result[str(params["object_key"])] = params["footprint"]
        elif unit["type"] == "LandmarkRelationUnit":
            result.update(
                {
                    str(key): value
                    for key, value in params["member_footprints"].items()
                }
            )
        else:
            result.update(
                {
                    str(key): value
                    for key, value in params["receiver_footprints"].items()
                }
            )
    return result


def _entity_z_bounds(
    units: Sequence[Mapping[str, Any]],
) -> dict[str, Sequence[float]]:
    result: dict[str, Sequence[float]] = {}
    for unit in units:
        params = unit["params"]
        if unit["type"] == "ObjectUnit":
            result[str(params["object_key"])] = params.get(
                "local_z_bounds", [0.0, 0.0]
            )
        elif unit["type"] == "LandmarkRelationUnit":
            result.update(
                {
                    str(key): value
                    for key, value in (
                        params.get("member_z_bounds") or {}
                    ).items()
                }
            )
        elif unit["type"] == "ReceiverReservationUnit":
            result.update(
                {
                    str(key): value
                    for key, value in (
                        params.get("receiver_z_bounds") or {}
                    ).items()
                }
            )
    return result


def _layout_entity_geometry(
    units: Sequence[Mapping[str, Any]],
    placements: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, Any]]:
    unit_by_entity, local_pose_by_entity = _entity_units(units)
    footprints = _entity_footprints(units)
    z_bounds_by_entity = _entity_z_bounds(units)
    result: dict[str, dict[str, Any]] = {}
    for entity, unit_id in unit_by_entity.items():
        footprint = footprints[entity]
        center_xy = [float(value) for value in footprint["center_xy"]]
        half_xy = [float(value) for value in footprint["half_extents_xy"]]
        unit_position = placements[unit_id]
        pose = local_pose_by_entity[entity]
        translation = [
            float(value)
            for value in pose.get("translation_xyz", [0.0, 0.0, 0.0])
        ]
        while len(translation) < 3:
            translation.append(0.0)
        z_bounds = [
            float(value)
            for value in z_bounds_by_entity.get(entity, [0.0, 0.0])
        ]
        if len(z_bounds) < 2:
            z_bounds = [0.0, 0.0]
        minimum = [
            float(unit_position[0]) + center_xy[0] - half_xy[0],
            float(unit_position[1]) + center_xy[1] - half_xy[1],
            translation[2] + z_bounds[0],
        ]
        maximum = [
            float(unit_position[0]) + center_xy[0] + half_xy[0],
            float(unit_position[1]) + center_xy[1] + half_xy[1],
            translation[2] + z_bounds[1],
        ]
        result[entity] = {
            "position": [
                float(unit_position[0]) + translation[0],
                float(unit_position[1]) + translation[1],
                translation[2],
            ],
            "aabb": {
                "min": minimum,
                "max": maximum,
                "center": [
                    (minimum[index] + maximum[index]) * 0.5
                    for index in range(3)
                ],
            },
        }
    return result


def _not_success_holds(
    units: Sequence[Mapping[str, Any]],
    placements: Mapping[str, Sequence[float]],
) -> bool:
    entities = _layout_entity_geometry(units, placements)
    for receiver in units:
        if receiver["type"] != "ReceiverReservationUnit":
            continue
        for region in receiver["params"]["reserved_regions"]:
            subject = str(region["subject_key"])
            receiver_keys = [
                str(value)
                for value in (
                    region.get("receiver_keys")
                    or receiver["params"].get("receiver_keys")
                    or []
                )
            ]
            if subject not in entities or not receiver_keys:
                raise LayoutSolveError(
                    "constraint_source_invalid",
                    "not-success-at-t0 lacks relation endpoint geometry",
                    unit_ids=[str(receiver.get("id") or "")],
                    owner="unit_builder",
                )
            missing = [key for key in receiver_keys if key not in entities]
            if missing:
                raise LayoutSolveError(
                    "constraint_source_invalid",
                    "not-success-at-t0 receiver geometry is missing",
                    unit_ids=[str(receiver.get("id") or "")],
                    owner="unit_builder",
                    metrics={"missing_receiver_keys": missing},
                )
            evidence = static_relation_geometry_evidence(
                str(region["relation"]),
                entities[subject],
                entities[receiver_keys[0]],
                target2=(
                    entities[receiver_keys[1]]
                    if len(receiver_keys) > 1
                    else None
                ),
                contract=RELATION_CONTRACT,
            )
            if evidence["geometry_holds"]:
                return False
    return True


def solve_scene_spec(realized_artifact, unit_artifact, layout_seed, config, *, rejected_placement_hashes=()):
    """Solve a shared scene under an explicit >=2 D2b counterfactual domain.

    Domains change tasks, never the physical initial state between alternatives.
    Every omitted alternative is audited; D4 and non-D2b contexts stay fixed.
    """
    from .manipulation import goal_domains, POLICY
    realized = realized_artifact.get("realized_scene_family") or realized_artifact
    domains = goal_domains(realized)
    start = min(len(rejected_placement_hashes) // POLICY["maximum_layouts_per_goal_domain"], len(domains) - 1)
    domains = domains[start:] + domains[:start]
    failures = []
    for ids in domains:
        try:
            return _solve_scene_spec_domain(realized_artifact, unit_artifact, layout_seed, config,
                rejected_placement_hashes=rejected_placement_hashes, selected_goal_variant_ids=ids)
        except LayoutSolveError as exc:
            if exc.code not in SEED_LOCAL_UNSOLVED_CODES:
                raise
            failures.append(exc)
    last = failures[-1]
    last.metrics["D2b_task_domains_tried"] = len(domains)
    raise last


def _solve_scene_spec_domain(
    realized_artifact: Mapping[str, Any],
    unit_artifact: Mapping[str, Any],
    layout_seed: int,
    config: LayoutConfig,
    *,
    rejected_placement_hashes: Sequence[str] = (),
    selected_goal_variant_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Solve one variant-invariant SceneSpec."""

    if layout_seed < 0:
        raise ValueError("layout seed must be non-negative")
    realized = (
        realized_artifact.get("realized_scene_family") or realized_artifact
    )
    realized_id = str(realized["id"])
    if any(
        str((variant.get("goal") or {}).get(field) or "") == "handover"
        for variant in realized.get("goal_variants") or []
        if isinstance(variant, Mapping)
        for field in ("verb", "action_family", "route")
    ):
        raise LayoutSolveError(
            "handover_disabled",
            "M12 publication excludes pure handover GoalVariants",
            realized_family_id=realized_id,
            layout_seed=layout_seed,
            owner="realization",
        )
    expected_unit_config_hash = artifact_hash(UNIT_BUILD_CONFIG)
    if not _unit_join_valid(realized_artifact, unit_artifact):
        raise LayoutSolveError(
            "stale_unit_join",
            "M11 Unit delta does not match its source realization",
            realized_family_id=realized_id,
            layout_seed=layout_seed,
            owner="unit_builder",
            metrics={
                "source_realization_hash": realized_artifact.get("artifact_hash"),
                "unit_source_realization_hash": unit_artifact.get(
                    "source_realization_hash"
                ),
                "unit_build_version": unit_artifact.get("unit_build_version"),
                "expected_unit_build_version": UNIT_BUILD_VERSION,
                "unit_config_hash": unit_artifact.get("config_hash"),
                "expected_unit_config_hash": expected_unit_config_hash,
            },
        )
    from .manipulation import OperationLayoutGate, domain_receipt, selected_layout_units
    original_realized = realized
    if selected_goal_variant_ids is not None:
        selected_ids = set(selected_goal_variant_ids)
        realized = {**realized, "goal_variants": [v for v in realized["goal_variants"] if v["id"] in selected_ids]}
    units = list(unit_artifact.get("units", []))
    if selected_goal_variant_ids is not None:
        # Validate before filtering any semantic reservation requirements.
        domain_receipt(original_realized, selected_goal_variant_ids)
        if len(selected_goal_variant_ids) != len(original_realized["goal_variants"]):
            units = selected_layout_units(realized, units)
    operation_gate = OperationLayoutGate(realized, units, config)
    placements = _solve_unit_poses(
        realized,
        units,
        layout_seed,
        config,
        rejected_placement_hashes=rejected_placement_hashes,
        operation_gate=operation_gate,
    )
    if not _not_success_holds(units, placements):
        raise AssertionError(
            "layout search returned a not-success-at-t0 violation"
        )
    role_by_entity = {
        str(binding["entity_key"]): str(role)
        for role, binding in realized["role_bindings"].items()
    }
    unit_by_entity, local_pose_by_entity = _entity_units(units)
    operation_contracts = _operation_anchor_contracts(units)
    operation_anchor_kinds = _operation_anchor_kinds(realized)
    reach_requirements = _effective_reach_requirements(realized, units)
    articulation_capabilities = _articulation_capabilities_by_entity(
        realized, units
    )
    touch_capabilities = _touch_capabilities_by_entity(realized, units)
    rigid_pick_capabilities = _rigid_pick_capable_arms(realized, units)
    articulation_reach_groups = _operation_reach_groups(
        realized, units
    )
    pick_operation_entities = _pick_operation_entities(realized)
    entities: dict[str, dict[str, Any]] = {}
    reach: dict[str, dict[str, Any]] = {}
    for entity, role in sorted(role_by_entity.items()):
        unit_id = unit_by_entity[entity]
        anchor = placements[unit_id]
        local = local_pose_by_entity[entity]
        translation = local["translation_xyz"]
        relative_yaw = float(local.get("relative_yaw_deg", 0.0))
        yaw_policy = local.get("yaw_policy") or {}
        yaw_mode = str(yaw_policy.get("mode") or "")
        if yaw_mode == "fixed":
            fixed_yaw = float(yaw_policy.get("yaw_deg") or 0.0)
            yaw_error = ((relative_yaw - fixed_yaw + 180.0) % 360.0) - 180.0
            if abs(yaw_error) > 1e-6:
                raise LayoutSolveError(
                    "initial_pose_yaw_policy_violation",
                    f"entity {entity} requests yaw {relative_yaw} outside fixed Initial Pose yaw {fixed_yaw}",
                    realized_family_id=realized_id,
                    layout_seed=layout_seed,
                    blockers=[
                        {
                            "entity_key": entity,
                            "initial_state_id": local.get("initial_state_id"),
                            "requested_yaw_deg": relative_yaw,
                            "fixed_yaw_deg": fixed_yaw,
                        }
                    ],
                )
        elif yaw_mode not in {"", "free"}:
            raise LayoutSolveError(
                "initial_pose_yaw_policy_violation",
                f"entity {entity} has invalid Initial Pose yaw policy {yaw_mode!r}",
                realized_family_id=realized_id,
                layout_seed=layout_seed,
            )
        position = _vec(
            [
                float(anchor[0]) + float(translation[0]),
                float(anchor[1]) + float(translation[1]),
                config.table_z + float(translation[2]),
            ]
        )
        quat = _quat_multiply(_yaw_quat(relative_yaw), local["q0_wxyz"])
        binding = realized["role_bindings"][role]
        entity_spec = {
            "role_ref": role,
            "asset": str(binding["asset"]),
            "model_id": str(binding["model_id"]),
            "unit_id": unit_id,
            "pose": {"position": position, "quat_wxyz": quat},
        }
        if local.get("initial_state_id"):
            approved_support = local.get("support") or {}
            approved_bounds = local.get("origin_relative_bounds") or {}
            local_z_min = float((approved_bounds.get("min_xyz") or [0, 0, 0])[2])
            support_clearance = float(
                approved_support.get("clearance_m") or 0.0
            )
            placement_support = {
                "plane_z": _clean(
                    float(position[2]) + local_z_min - support_clearance
                ),
                "clearance_m": _clean(support_clearance),
                "origin_z": _clean(float(position[2])),
                "formula": "support_z-local_z_min+support_clearance",
            }
            entity_spec["initial_pose"] = {
                "initial_state_id": local.get("initial_state_id"),
                "revision": local.get("initial_state_revision"),
                "revision_hash": local.get("initial_state_revision_hash"),
                "geometry_hash": local.get("geometry_hash"),
                "canonical_state_hash": local.get(
                    "canonical_state_hash"
                ),
                "q0_wxyz": local.get("q0_wxyz"),
                "world_yaw_deg": relative_yaw,
                "yaw_policy": yaw_policy,
                "scale_xyz": local.get("scale_xyz"),
                "origin_relative_bounds": local.get("origin_relative_bounds"),
                "support": placement_support,
                "articulated_joint_state": local.get(
                    "articulated_joint_state"
                ),
                "dynamic_mass_properties": local.get(
                    "dynamic_mass_properties"
                ),
            }
            if local.get("assignment_state_revision"):
                entity_spec["initial_pose"]["assignment_state_revision"] = (
                    local["assignment_state_revision"]
                )
        entities[entity] = entity_spec
        operation_points = _entity_operation_positions_xy(
            entity,
            placements,
            unit_by_entity,
            local_pose_by_entity,
            operation_contracts,
            operation_anchor_kinds.get(entity, {"grasp_contact"}),
        )
        reach_record = (
            _operation_reach_metadata(operation_points, config)
            if operation_points
            else _reach_metadata(position[:2], config)
        )
        if entity in articulation_capabilities:
            reach_record["articulation_capability"] = canonical_data(
                articulation_capabilities[entity]
            )
        if entity in touch_capabilities:
            reach_record["touch_capability"] = canonical_data(
                touch_capabilities[entity]
            )
        if entity in articulation_reach_groups:
            group_rows: list[dict[str, Any]] = []
            reachable_capability_arms: set[str] = set()
            for group in articulation_reach_groups[entity]:
                options: list[dict[str, Any]] = []
                for option in group.get("options", []):
                    offsets = _entity_operation_offsets_xy(
                        entity,
                        unit_by_entity,
                        local_pose_by_entity,
                        operation_contracts,
                        {"grasp_contact"},
                        contact_ids={int(option["contact_id"])},
                    )
                    for offset in offsets:
                        point_xy = [
                            _clean(float(anchor[0]) + offset[0]),
                            _clean(float(anchor[1]) + offset[1]),
                        ]
                        reachable = _in_reach(
                            point_xy, str(option["arm"]), config
                        )
                        if reachable:
                            reachable_capability_arms.add(str(option["arm"]))
                        options.append(
                            {
                                **canonical_data(dict(option)),
                                "position_xy": point_xy,
                                "reachable": bool(reachable),
                            }
                        )
                group_rows.append(
                    {
                        "id": str(group["id"]),
                        "use": str(group["use"]),
                        "mode": "any",
                        "options": options,
                        "passed": any(option["reachable"] for option in options),
                    }
                )
            reach_record["reachable_arms"] = sorted(
                reachable_capability_arms
            )
            reach_record["zone"] = (
                "both"
                if len(reachable_capability_arms) == 2
                else f"{next(iter(reachable_capability_arms))}_only"
                if len(reachable_capability_arms) == 1
                else "none"
            )
            articulation_rows = [
                row
                for row, source in zip(
                    group_rows, articulation_reach_groups[entity], strict=True
                )
                if source.get("kind") == "articulation"
            ]
            touch_rows = [
                row
                for row, source in zip(
                    group_rows, articulation_reach_groups[entity], strict=True
                )
                if source.get("kind") == "touch"
            ]
            if articulation_rows:
                reach_record["articulation_operation_groups"] = articulation_rows
                reach_record["articulation_operation_groups_passed"] = all(
                    group["passed"] for group in articulation_rows
                )
            if touch_rows:
                reach_record["touch_operation_groups"] = touch_rows
                reach_record["touch_operation_groups_passed"] = all(
                    group["passed"] for group in touch_rows
                )
            reach_record["evidence"] = (
                "benchmark_v1_per_arm_m4_operation_contact_reach_proxy_v2"
            )
        if entity in pick_operation_entities and entity in rigid_pick_capabilities:
            geometric_arms = list(reach_record["reachable_arms"])
            reachable_pick_arms = [
                arm
                for arm in geometric_arms
                if arm in rigid_pick_capabilities[entity]
            ]
            reach_record["reachable_arms"] = reachable_pick_arms
            reach_record["zone"] = (
                "both"
                if len(reachable_pick_arms) == 2
                else f"{reachable_pick_arms[0]}_only"
                if len(reachable_pick_arms) == 1
                else "none"
            )
        reach[entity] = reach_record
    terminal_reach: dict[str, dict[str, Any]] = {}
    for region_id, row in sorted(
        _terminal_reach_requirements(realized, units).items()
    ):
        anchor = placements[str(row["unit_id"])]
        center = row["center_xy"]
        position_xy = _vec(
            [
                float(anchor[0]) + float(center[0]),
                float(anchor[1]) + float(center[1]),
            ]
        )
        metadata = _reach_metadata(position_xy, config)
        terminal_reach[region_id] = {
            "subject": str(row["subject"]),
            "relation": str(row["relation"]),
            "required_arm": str(row["required_arm"]),
            "position_xy": position_xy,
            **metadata,
            "proxy_passed": _in_reach(
                position_xy, str(row["required_arm"]), config
            ),
            "evidence": "benchmark_v1_terminal_static_reach_proxy_v1",
        }
    actor = {
        "embodiment": str(config.data["embodiment"]),
        "base_pose": canonical_data(config.data["actor"]["base_pose"]),
    }
    camera_config = config.data["camera"]
    camera = {
        "name": str(camera_config["name"]),
        "type": str(camera_config["type"]),
        "width": int(camera_config["width"]),
        "height": int(camera_config["height"]),
        "fovy_deg": float(camera_config["fovy_deg"]),
        "near": float(camera_config["near"]),
        "far": float(camera_config["far"]),
        "pose": {
            "position": _vec(camera_config["position"]),
            "quat_wxyz": _camera_quat(camera_config),
        },
        "table_visible_polygon_xy": canonical_data(
            camera_config["table_visible_polygon_xy"]
        ),
    }
    unit_poses = [
        {
            "unit_id": unit_id,
            "position": _vec([position[0], position[1], config.table_z]),
            "yaw_deg": 0.0,
        }
        for unit_id, position in sorted(placements.items())
    ]
    state_payload = {
        "actor": actor,
        "camera": camera,
        "entities": entities,
    }
    state_hash = artifact_hash(state_payload)
    scene_spec_id = deterministic_id(
        "ss",
        {
            "realized_family_id": realized_id,
            "layout_seed": layout_seed,
            "source_realization_hash": realized_artifact.get("artifact_hash"),
            "source_unit_hash": unit_artifact.get("artifact_hash"),
            "config_hash": config.hash,
        },
    )
    result = {
        "id": scene_spec_id,
        "realized_family_id": realized_id,
        "layout_seed": int(layout_seed),
        "entities": entities,
        "unit_poses": unit_poses,
        "actor": actor,
        "camera": camera,
        "reach": reach,
        "terminal_reach": terminal_reach,
        "initial_observation": {
            "observation_id": deterministic_id(
                "obs",
                {"scene_spec_id": scene_spec_id, "state_hash": state_hash},
            ),
            "scene_state_hash": state_hash,
            "variant_invariant": True,
            "artifact_path": None,
        },
        "goal_variant_ids": sorted(
            str(row["id"]) for row in realized.get("goal_variants", [])
        ),
        "provenance": build_provenance(
            "if_benchmark.layout",
            config.hash,
            [
                {
                    "kind": "realized_scene_family",
                    "hash": str(realized_artifact["artifact_hash"]),
                    "record_id": realized_id,
                },
                {
                    "kind": "unit_build",
                    "hash": str(unit_artifact["artifact_hash"]),
                    "record_id": realized_id,
                },
            ],
            metadata={
                "layout_solver_version": LAYOUT_SOLVER_VERSION,
                "unit_build_version": unit_artifact.get(
                    "unit_build_version"
                ),
            },
        ),
    }
    metadata = result["provenance"]["metadata"]
    if operation_gate.enabled or set(result["goal_variant_ids"]) != {
        v["id"] for v in original_realized["goal_variants"]
    }:
        metadata["d2b_task_domain"] = domain_receipt(original_realized, result["goal_variant_ids"])
    if operation_gate.enabled:
        payload = {"policy": "d2b_operation_space_joint_branches_v1",
                   "scene_state_hash": state_hash,
                   "resource_hashes": {k: v["content_hash"] for k, v in operation_gate.templates.resources.items()},
                   "goal_options": operation_gate.last_options}
        metadata["operation_space"] = {**payload, "constraint_hash": artifact_hash(payload)}
        if hasattr(operation_gate, "initial_candidate_receipt"):
            metadata["initial_candidate_selection"] = operation_gate.initial_candidate_receipt
    try:
        task_requirements = single_arm_pick_requirement_groups(original_realized, result)
    except ValueError as exc:
        error = str(exc)
        reach_failure = error.startswith(
            ("pick_planning_no_direct_arm:", "pick_planning_semantic_arm_unreachable:")
        )
        raise LayoutSolveError(
            (
                "direct_single_arm_reach_bounds"
                if reach_failure
                else "pick_planning_source_invalid"
            ),
            (
                f"no direct singleton-arm assignment exists: {exc}"
                if reach_failure
                else f"invalid Pick GoalVariant identity: {exc}"
            ),
            realized_family_id=realized_id,
            layout_seed=layout_seed,
            constraint_ids=[
                "layout.direct_single_arm_reach"
                if reach_failure
                else "realization.goal_variant_identity"
            ],
            owner="layout" if reach_failure else "realization",
        ) from exc
    for group in task_requirements:
        entity = str(group["entity_key"])
        marker = {
            "goal_variant_id": str(group["goal_variant_id"]),
            "selected_arm": str(group["selected_arm"]),
            "required_arms": [str(group["selected_arm"])],
            "requirement_mode": "all",
            "route": "direct",
            "selection": str(group["selection"]),
            "scope": "exact_scene_planning_required",
            "planning_witness_required": True,
            "final_feasibility_proven": False,
        }
        result["reach"][entity].setdefault(
            "pick_planning_requirements", []
        ).append(marker)
    for row in result["reach"].values():
        if "pick_planning_requirements" in row:
            row["pick_planning_requirements"] = sorted(
                row["pick_planning_requirements"],
                key=lambda marker: str(marker["goal_variant_id"]),
            )
    return canonical_data(result)


def preflight_scene_spec(
    realized: Mapping[str, Any],
    scene_spec: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    expected_bindings = {
        str(binding["entity_key"]): {
            "role_ref": str(role),
            "asset": str(binding["asset"]),
            "model_id": str(binding["model_id"]),
        }
        for role, binding in realized["role_bindings"].items()
    }
    entities = scene_spec.get("entities", {})
    if set(entities) != set(expected_bindings):
        failures.append(
            {
                "code": "entity_alignment",
                "message": "SceneSpec entities do not match role bindings",
            }
        )
    else:
        for entity, expected in expected_bindings.items():
            actual = entities[entity]
            if any(actual.get(key) != value for key, value in expected.items()):
                failures.append(
                    {
                        "code": "binding_alignment",
                        "message": (
                            f"SceneSpec binding for {entity} does not match "
                            "the realized family"
                        ),
                    }
                )
    unit_pose_ids = [
        str(row.get("unit_id")) for row in scene_spec.get("unit_poses", [])
    ]
    entity_unit_ids = {
        str(row.get("unit_id")) for row in entities.values()
    }
    if (
        len(unit_pose_ids) != len(set(unit_pose_ids))
        or set(unit_pose_ids) != entity_unit_ids
    ):
        failures.append(
            {
                "code": "unit_pose_alignment",
                "message": "SceneSpec Unit poses do not align with entity membership",
            }
        )
    expected_variants = sorted(
        str(row["id"]) for row in realized.get("goal_variants", [])
    )
    from .manipulation import valid_domain
    if not valid_domain(realized, scene_spec):
        failures.append(
            {
                "code": "variant_alignment",
                "message": "SceneSpec does not retain every realized variant",
            }
        )
    expected_terminal_goals = {
        (str(goal["subject"]), str(goal["relation"]))
        for variant in realized.get("goal_variants", [])
        if variant["id"] in scene_spec.get("goal_variant_ids", [])
        for goal in [variant.get("goal") or {}]
        if goal.get("subject") and goal.get("relation")
    }
    terminal_rows = scene_spec.get("terminal_reach") or {}
    observed_terminal_goals = {
        (str(row.get("subject")), str(row.get("relation")))
        for row in terminal_rows.values()
        if isinstance(row, Mapping)
    }
    if expected_terminal_goals - observed_terminal_goals:
        failures.append(
            {
                "code": "terminal_reach_alignment",
                "message": "SceneSpec omits required terminal reach evidence",
            }
        )
    if any(
        not bool(row.get("proxy_passed"))
        for row in terminal_rows.values()
        if isinstance(row, Mapping)
    ):
        failures.append(
            {
                "code": "terminal_reach_bounds",
                "message": "SceneSpec contains a failed terminal reach proxy",
            }
        )
    state_payload = {
        "actor": scene_spec.get("actor"),
        "camera": scene_spec.get("camera"),
        "entities": scene_spec.get("entities"),
    }
    if (
        scene_spec.get("initial_observation", {}).get("scene_state_hash")
        != artifact_hash(state_payload)
        or scene_spec.get("initial_observation", {}).get("variant_invariant")
        is not True
    ):
        failures.append(
            {
                "code": "initial_observation_hash",
                "message": "initial observation hash is not variant-invariant",
            }
        )
    schema = validate_document(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "scene_spec",
            "scene_spec": canonical_data(dict(scene_spec)),
        }
    )
    failures.extend(
        {
            "code": "scene_spec_schema",
            "message": f"{row['path']}: {row['message']}",
        }
        for row in schema["errors"]
    )
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }


PickPlanningAttemptInvoker = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any] | None,
]


def _validated_pick_arm_selection_retries(
    raw: Any,
) -> list[dict[str, Any]]:
    """Validate failed same-placement arm assignments retained for replay."""

    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("pick_planning_arm_selection_retries must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError("Pick arm-selection retry is not a mapping")
        request = row.get("request")
        witness = row.get("witness")
        if not isinstance(request, Mapping) or not isinstance(witness, Mapping):
            raise ValueError("Pick arm-selection retry lacks request/witness")
        request_hash = str(request.get("request_hash") or "")
        if not request_hash or request_hash in seen:
            raise ValueError("Pick arm-selection retry request is duplicate")
        seen.add(request_hash)
        try:
            _enforce_pick_witness_result(request, witness)
        except LayoutSolveError as exc:
            if exc.code != "pick_planning_unreachable":
                raise ValueError(
                    "Pick arm-selection retry is not deterministic no-path"
                ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid Pick arm-selection retry") from exc
        else:
            raise ValueError("Pick arm-selection retry unexpectedly passed")
        result.append(
            canonical_data(
                {"request": dict(request), "witness": dict(witness)}
            )
        )
    return result


def _validated_pick_planning_rejection(
    raw: Mapping[str, Any],
    *,
    expected_request: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one frozen deterministic no-path placement rejection."""

    placement_hash = str(raw.get("placement_hash") or "")
    arm_retries = _validated_pick_arm_selection_retries(
        raw.get("arm_selection_retries") or []
    )
    request = raw.get("request")
    witness = raw.get("witness")
    if (
        not placement_hash
        or not isinstance(request, Mapping)
        or not isinstance(witness, Mapping)
    ):
        raise ValueError("invalid frozen Pick layout rejection")
    if expected_request is not None and canonical_data(
        request
    ) != canonical_data(expected_request):
        raise ValueError(
            "frozen Pick layout rejection request does not match the CPU layout"
        )
    try:
        _enforce_pick_witness_result(request, witness)
    except LayoutSolveError as exc:
        if exc.code != "pick_planning_unreachable":
            raise ValueError(
                "frozen Pick layout rejection is not deterministic no-path"
            ) from exc
        failure = exc.to_record()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid frozen Pick layout rejection") from exc
    else:
        raise ValueError(
            "frozen Pick layout rejection contains a passing witness"
        )
    return (
        canonical_data(
            {
                "placement_hash": placement_hash,
                "arm_selection_retries": arm_retries,
                "request": dict(request),
                "witness": dict(witness),
            }
        ),
        canonical_data(failure),
    )


def _validated_pick_planning_rejections(
    raw: Any,
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("pick_planning_rejections must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError(
                "pick_planning_rejections contains a non-mapping row"
            )
        checked, _ = _validated_pick_planning_rejection(row)
        placement_hash = str(checked["placement_hash"])
        if placement_hash in seen:
            raise ValueError(
                "pick_planning_rejections contains a duplicate layout"
            )
        seen.add(placement_hash)
        result.append(checked)
    return result


def _pick_planning_rejection_from_error(
    scene_spec: Mapping[str, Any],
    exc: LayoutSolveError,
    *,
    arm_selection_retries: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if exc.code != "pick_planning_unreachable":
        raise ValueError(
            "only deterministic no-path can reject a layout candidate"
        )
    request = exc.metrics.get("frozen_request")
    witness = exc.metrics.get("frozen_witness")
    if not isinstance(request, Mapping) or not isinstance(witness, Mapping):
        raise ValueError(
            "Pick no-path failure lacks frozen request/witness evidence"
        )
    row, failure = _validated_pick_planning_rejection(
        {
            "placement_hash": _scene_unit_placement_hash(scene_spec),
            "arm_selection_retries": list(arm_selection_retries),
            "request": request,
            "witness": witness,
        }
    )
    if canonical_data(failure) != canonical_data(exc.to_record()):
        raise ValueError(
            "Pick no-path failure does not reproduce from its witness"
        )
    return row


def _pick_planning_trace_failures(
    rejections: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checked_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rejections:
        checked, failure = _validated_pick_planning_rejection(row)
        placement_hash = str(checked["placement_hash"])
        if placement_hash in seen:
            raise ValueError("Pick planning trace repeats a layout candidate")
        seen.add(placement_hash)
        checked_rows.append(checked)
        failures.append(failure)
    return checked_rows, failures


def _pick_planning_exhausted_error(
    rejections: Sequence[Mapping[str, Any]],
    cpu_exhaustion: LayoutSolveError,
) -> LayoutSolveError:
    checked_rows, failures = _pick_planning_trace_failures(rejections)
    if not checked_rows or cpu_exhaustion.code != "layout_search_exhausted":
        raise ValueError("invalid exhaustive Pick planning rejection trace")
    blockers: list[dict[str, Any]] = []
    for index, (row, failure) in enumerate(zip(checked_rows, failures)):
        for blocker in failure.get("blockers") or []:
            blockers.append(
                {
                    **canonical_data(dict(blocker)),
                    "layout_candidate_index": index,
                    "placement_hash": row["placement_hash"],
                }
            )
    last = checked_rows[-1]
    return LayoutSolveError(
        "pick_planning_unreachable",
        (
            "all "
            f"{len(checked_rows)} statically valid layout candidates lack a "
            "required exact-scene Pick path"
        ),
        realized_family_id=str(last["request"]["realized_family_id"]),
        layout_seed=int(last["request"]["layout_seed"]),
        constraint_ids=["layout.pick_planning_witness"],
        blockers=blockers,
        owner="layout",
        metrics={
            "planning_layout_candidate_count": len(checked_rows),
            "planning_rejections": checked_rows,
            "rejected_placement_hashes": [
                row["placement_hash"] for row in checked_rows
            ],
            "cpu_layout_exhaustion": cpu_exhaustion.to_record(),
            # Retain the last request/witness aliases for report readers that
            # have not yet learned the multi-candidate trace field.
            "frozen_request": last["request"],
            "frozen_witness": last["witness"],
        },
    )


def _pick_planning_budget_error(
    rejections: Sequence[Mapping[str, Any]],
    *,
    max_attempts: int,
    next_placement_hash: str | None = None,
    cpu_failure: LayoutSolveError | None = None,
) -> LayoutSolveError:
    checked_rows, failures = _pick_planning_trace_failures(rejections)
    if not checked_rows:
        raise ValueError(
            "Pick planning budget failure lacks attempted layouts"
        )
    last = checked_rows[-1]
    blockers = [
        {
            **canonical_data(dict(blocker)),
            "layout_candidate_index": len(checked_rows) - 1,
            "placement_hash": last["placement_hash"],
        }
        for blocker in failures[-1].get("blockers") or []
    ]
    message = (
        "CPU continuation search could not prove layout-domain exhaustion "
        f"after {len(checked_rows)} planner-gated candidates"
        if cpu_failure is not None
        else (
            "Pick planning layout search reached its deterministic per-seed "
            f"budget of {max_attempts} candidates before proving exhaustion"
        )
    )
    return LayoutSolveError(
        "pick_planning_search_budget_exhausted",
        message,
        realized_family_id=str(last["request"]["realized_family_id"]),
        layout_seed=int(last["request"]["layout_seed"]),
        constraint_ids=["layout.pick_planning_witness"],
        blockers=blockers,
        owner="layout",
        metrics={
            "planning_layout_candidate_count": len(checked_rows),
            "max_pick_planning_layout_attempts_per_seed": int(max_attempts),
            "planning_rejections": checked_rows,
            "rejected_placement_hashes": [
                row["placement_hash"] for row in checked_rows
            ],
            "next_placement_hash": next_placement_hash,
            "cpu_failure": (
                None if cpu_failure is None else cpu_failure.to_record()
            ),
        },
    )


def _solve_scene_spec_with_pick_planning(
    realized_artifact: Mapping[str, Any],
    unit_artifact: Mapping[str, Any],
    layout_seed: int,
    config: LayoutConfig,
    *,
    invoke_pick_planning: PickPlanningAttemptInvoker | None,
    require_pick_planning_witness: bool,
    new_planning_layout_limit: int | None = None,
) -> dict[str, Any]:
    """Search static layouts until one passes the exact Pick planning gate.

    The CPU solver remains simulator-independent.  The caller-owned invoker is
    consulted only after a complete tentative SceneSpec exists; a classified
    deterministic no-path rejects that placement hash and resumes CPU search.
    Every other provider disposition fails immediately.
    """

    rejected_hashes: list[str] = []
    rejections: list[dict[str, Any]] = []
    max_attempts = config.max_pick_planning_layout_attempts_per_seed
    if new_planning_layout_limit is not None:
        if not 1 <= new_planning_layout_limit <= max_attempts:
            raise ValueError("new planning budget must not exceed the frozen validation ceiling")
        max_attempts = new_planning_layout_limit

    def exhausted_error(exc: LayoutSolveError) -> LayoutSolveError:
        if contains_limited_planning_failure(rejections):
            error = _pick_planning_budget_error(rejections, max_attempts=max_attempts, cpu_failure=exc)
            error.metrics["candidate_search_budget_applied"] = True
            return error
        return _pick_planning_exhausted_error(rejections, exc)
    while True:
        try:
            scene_spec = solve_scene_spec(
                realized_artifact,
                unit_artifact,
                layout_seed,
                config,
                rejected_placement_hashes=rejected_hashes,
            )
        except LayoutSolveError as exc:
            if rejections and exc.code == "layout_search_exhausted":
                raise exhausted_error(exc) from exc
            if rejections and exc.code == "search_budget_exhausted":
                raise _pick_planning_budget_error(
                    rejections,
                    max_attempts=max_attempts,
                    cpu_failure=exc,
                ) from exc
            raise
        preflight = preflight_scene_spec(
            realized_artifact["realized_scene_family"], scene_spec
        )
        if preflight["status"] != "passed":
            raise LayoutSolveError(
                "scene_spec_preflight_failed",
                "; ".join(
                    str(row["message"]) for row in preflight["failures"][:10]
                ),
                realized_family_id=str(
                    realized_artifact["realized_family_id"]
                ),
                layout_seed=int(layout_seed),
                owner="layout",
            )
        arm_selection_retries: list[dict[str, Any]] = []
        while True:
            request = _pick_planning_request(
                realized_artifact,
                unit_artifact,
                scene_spec,
                config,
                prior_failure_counts=_pick_failure_counts_from_rejections(
                    rejections
                ),
            )
            context = {
                "realized_artifact": realized_artifact,
                "unit_artifact": unit_artifact,
                "scene_spec": scene_spec,
                "layout_config": config.data,
                "config_hash": config.hash,
            }
            try:
                if invoke_pick_planning is None:
                    witness = _invoke_pick_witness_provider(
                        None,
                        request,
                        required=require_pick_planning_witness,
                        context=context,
                    )
                else:
                    witness = (
                        invoke_pick_planning(request, context)
                        if request
                        else None
                    )
                if request is not None and witness is not None:
                    witness = _enforce_pick_witness_result(request, witness)
                break
            except LayoutSolveError as exc:
                if exc.code != "pick_planning_unreachable":
                    raise
                frozen_request = exc.metrics.get("frozen_request")
                frozen_witness = exc.metrics.get("frozen_witness")
                try:
                    if not isinstance(
                        frozen_request, Mapping
                    ) or not isinstance(frozen_witness, Mapping):
                        raise ValueError(
                            "Pick no-path failure lacks frozen request/witness"
                        )
                    realized = (
                        realized_artifact.get("realized_scene_family")
                        or realized_artifact
                    )
                    next_assignment = next_non_d4_pick_arm_assignment(
                        realized,
                        scene_spec,
                        frozen_request,
                        frozen_witness,
                        prior_requests=[
                            row["request"] for row in arm_selection_retries
                        ],
                    )
                    if next_assignment is not None:
                        retry_row = _validated_pick_arm_selection_retries(
                            [
                                {
                                    "request": frozen_request,
                                    "witness": frozen_witness,
                                }
                            ]
                        )[0]
                        arm_selection_retries.append(retry_row)
                        scene_spec = _scene_with_pick_arm_assignment(
                            realized, scene_spec, next_assignment
                        )
                        preflight = preflight_scene_spec(realized, scene_spec)
                        if preflight["status"] != "passed":
                            raise ValueError(
                                "fallback arm assignment invalidated SceneSpec"
                            )
                        continue
                    rejection = _pick_planning_rejection_from_error(
                        scene_spec,
                        exc,
                        arm_selection_retries=arm_selection_retries,
                    )
                except ValueError as trace_exc:
                    raise LayoutSolveError(
                        "pick_planning_failure_unclassified",
                        f"Pick no-path evidence is not replayable: {trace_exc}",
                        realized_family_id=str(
                            realized_artifact["realized_family_id"]
                        ),
                        layout_seed=int(layout_seed),
                        owner="planning_oracle",
                    ) from trace_exc
                placement_hash = str(rejection["placement_hash"])
                if placement_hash in rejected_hashes:
                    raise LayoutSolveError(
                        "pick_planning_failure_unclassified",
                        "Pick layout retry produced a duplicate placement",
                        realized_family_id=str(
                            realized_artifact["realized_family_id"]
                        ),
                        layout_seed=int(layout_seed),
                        owner="layout",
                    ) from exc
                rejections.append(rejection)
                rejected_hashes.append(placement_hash)
                if len(rejections) < max_attempts:
                    break
                try:
                    next_scene = solve_scene_spec(
                        realized_artifact,
                        unit_artifact,
                        layout_seed,
                        config,
                        rejected_placement_hashes=rejected_hashes,
                    )
                except LayoutSolveError as cpu_exc:
                    if cpu_exc.code == "layout_search_exhausted":
                        raise exhausted_error(cpu_exc) from cpu_exc
                    if cpu_exc.code == "search_budget_exhausted":
                        raise _pick_planning_budget_error(
                            rejections,
                            max_attempts=max_attempts,
                            cpu_failure=cpu_exc,
                        ) from cpu_exc
                    raise
                raise _pick_planning_budget_error(
                    rejections,
                    max_attempts=max_attempts,
                    next_placement_hash=_scene_unit_placement_hash(next_scene),
                )
        if rejections and _scene_unit_placement_hash(scene_spec) in rejected_hashes:
            # The inner loop exhausted this physical placement and the outer
            # loop must ask the CPU solver for its next candidate.
            continue
        return {
            "scene_spec": scene_spec,
            "preflight": preflight,
            "pick_planning_request": request,
            "pick_planning_witness": witness,
            "pick_planning_rejections": rejections,
            "pick_planning_arm_selection_retries": arm_selection_retries,
        }


class _FrozenPickPlanningReplay:
    """Replay one freeze-owned request/witness trace without a live oracle."""

    def __init__(
        self,
        rejections: Sequence[Mapping[str, Any]],
        *,
        final_arm_selection_retries: Sequence[Mapping[str, Any]] = (),
        final_request: Mapping[str, Any] | None = None,
        final_witness: Mapping[str, Any] | None = None,
    ) -> None:
        self.rejections = _validated_pick_planning_rejections(list(rejections))
        self.final_request = (
            None
            if final_request is None
            else canonical_data(dict(final_request))
        )
        self.final_witness = (
            None
            if final_witness is None
            else canonical_data(dict(final_witness))
        )
        if (self.final_request is None) != (self.final_witness is None):
            raise ValueError("frozen final Pick request/witness is incomplete")
        self.entries: list[dict[str, Any]] = []
        for rejection in self.rejections:
            for retry in rejection.get("arm_selection_retries") or []:
                self.entries.append(
                    {
                        "placement_hash": str(rejection["placement_hash"]),
                        "request": retry["request"],
                        "witness": retry["witness"],
                    }
                )
            self.entries.append(
                {
                    "placement_hash": str(rejection["placement_hash"]),
                    "request": rejection["request"],
                    "witness": rejection["witness"],
                }
            )
        for retry in _validated_pick_arm_selection_retries(
            list(final_arm_selection_retries)
        ):
            self.entries.append(
                {
                    "placement_hash": None,
                    "request": retry["request"],
                    "witness": retry["witness"],
                }
            )
        if self.final_request is not None:
            self.entries.append(
                {
                    "placement_hash": None,
                    "request": self.final_request,
                    "witness": self.final_witness,
                }
            )
        self.index = 0

    def __call__(
        self,
        request: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self.index < len(self.entries):
            row = self.entries[self.index]
            scene_spec = context.get("scene_spec")
            expected_placement = row["placement_hash"]
            if not isinstance(scene_spec, Mapping) or (
                expected_placement is not None
                and _scene_unit_placement_hash(scene_spec)
                != str(expected_placement)
            ):
                raise LayoutSolveError(
                    "pick_planning_witness_invalid",
                    "frozen Pick trace placement does not match the CPU layout",
                    realized_family_id=str(request["realized_family_id"]),
                    layout_seed=int(request["layout_seed"]),
                    owner="planning_oracle",
                )
            if canonical_data(request) != canonical_data(row["request"]):
                raise LayoutSolveError(
                    "pick_planning_witness_invalid",
                    "frozen Pick trace request does not match the CPU layout",
                    realized_family_id=str(request["realized_family_id"]),
                    layout_seed=int(request["layout_seed"]),
                    owner="planning_oracle",
                )
            self.index += 1
            assert isinstance(row["witness"], Mapping)
            return canonical_data(dict(row["witness"]))
        raise LayoutSolveError(
            "pick_planning_witness_invalid",
            "CPU layout search requested a Pick witness absent from the frozen trace",
            realized_family_id=str(request["realized_family_id"]),
            layout_seed=int(request["layout_seed"]),
            owner="planning_oracle",
        )

    def assert_consumed(self) -> None:
        if self.index != len(self.entries):
            raise ValueError(
                "CPU layout search did not consume every frozen Pick trace row"
            )


def _artifact(
    realized_artifact: Mapping[str, Any],
    unit_artifact: Mapping[str, Any],
    scene_spec: Mapping[str, Any],
    preflight: Mapping[str, Any],
    config: LayoutConfig,
    pick_planning_request: Mapping[str, Any] | None = None,
    pick_planning_witness: Mapping[str, Any] | None = None,
    *,
    pick_planning_required: bool = False,
    pick_planning_rejections: Sequence[Mapping[str, Any]] = (),
    pick_planning_arm_selection_retries: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "solved_scene_spec",
        "layout_solver_version": LAYOUT_SOLVER_VERSION,
        "realized_family_id": scene_spec["realized_family_id"],
        "layout_seed": scene_spec["layout_seed"],
        "source_realization_hash": realized_artifact["artifact_hash"],
        "source_unit_hash": unit_artifact["artifact_hash"],
        "config_hash": config.hash,
        "pick_planning_witness_required": bool(pick_planning_required),
        "pick_planning_witness_status": (
            "passed"
            if pick_planning_witness is not None
            else "missing"
            if pick_planning_required
            else "not_applicable"
        ),
        "pick_planning_rejections": canonical_data(
            list(pick_planning_rejections)
        ),
        "pick_planning_arm_selection_retries": canonical_data(
            list(pick_planning_arm_selection_retries)
        ),
        "scene_spec": canonical_data(dict(scene_spec)),
        "preflight": canonical_data(dict(preflight)),
    }
    if pick_planning_witness is not None:
        result["pick_planning_witness"] = canonical_data(
            dict(pick_planning_witness)
        )
    if pick_planning_request is not None:
        result["pick_planning_request"] = canonical_data(
            dict(pick_planning_request)
        )
    result["artifact_hash"] = artifact_hash(result)
    return result


def _reusable(
    artifact: Mapping[str, Any],
    realized_artifact: Mapping[str, Any],
    unit_artifact: Mapping[str, Any],
    layout_seed: int,
    config: LayoutConfig,
    *,
    require_pick_planning_witness: bool = False,
) -> bool:
    base_valid = (
        _unit_join_valid(realized_artifact, unit_artifact)
        and artifact.get("artifact_kind") == "solved_scene_spec"
        and artifact.get("layout_solver_version") == LAYOUT_SOLVER_VERSION
        and artifact.get("realized_family_id")
        == realized_artifact.get("realized_family_id")
        and artifact.get("layout_seed") == layout_seed
        and artifact.get("source_realization_hash")
        == realized_artifact.get("artifact_hash")
        and artifact.get("source_unit_hash") == unit_artifact.get("artifact_hash")
        and artifact.get("config_hash") == config.hash
        and artifact.get("preflight", {}).get("status") == "passed"
        and _self_hash_valid(artifact)
    )
    if not base_valid:
        return False
    try:
        rejections = _validated_pick_planning_rejections(
            artifact.get("pick_planning_rejections")
        )
        arm_selection_retries = _validated_pick_arm_selection_retries(
            artifact.get("pick_planning_arm_selection_retries") or []
        )
        request = _pick_planning_request(
            realized_artifact,
            unit_artifact,
            artifact.get("scene_spec") or {},
            config,
            prior_failure_counts=_pick_failure_counts_from_rejections(
                rejections
            ),
        )
        expected_required = request is not None
        expected_status = (
            "passed"
            if artifact.get("pick_planning_witness") is not None
            else "missing"
            if expected_required
            else "not_applicable"
        )
        if (
            artifact.get("pick_planning_witness_required")
            is not expected_required
            or artifact.get("pick_planning_witness_status")
            != expected_status
            or canonical_data(artifact.get("pick_planning_request"))
            != canonical_data(request)
        ):
            return False
        if request is None and (rejections or arm_selection_retries):
            return False
        if (
            len(rejections)
            >= config.max_pick_planning_layout_attempts_per_seed
        ):
            return False
        if request is not None and _scene_unit_placement_hash(
            artifact.get("scene_spec") or {}
        ) in {str(row["placement_hash"]) for row in rejections}:
            return False
        _validated_pick_witness(
            request,
            artifact.get("pick_planning_witness"),
            required=require_pick_planning_witness,
        )
        if rejections or arm_selection_retries:
            final_witness = artifact.get("pick_planning_witness")
            if request is None or not isinstance(final_witness, Mapping):
                return False
            replay = _FrozenPickPlanningReplay(
                rejections,
                final_arm_selection_retries=arm_selection_retries,
                final_request=request,
                final_witness=final_witness,
            )
            solved = _solve_scene_spec_with_pick_planning(
                realized_artifact,
                unit_artifact,
                layout_seed,
                config,
                invoke_pick_planning=replay,
                require_pick_planning_witness=require_pick_planning_witness,
            )
            replay.assert_consumed()
            if (
                canonical_data(solved["scene_spec"])
                != canonical_data(artifact.get("scene_spec"))
                or canonical_data(solved["preflight"])
                != canonical_data(artifact.get("preflight"))
                or canonical_data(solved["pick_planning_request"])
                != canonical_data(request)
                or canonical_data(solved["pick_planning_witness"])
                != canonical_data(final_witness)
                or canonical_data(solved["pick_planning_rejections"])
                != canonical_data(rejections)
                or canonical_data(
                    solved["pick_planning_arm_selection_retries"]
                )
                != canonical_data(arm_selection_retries)
            ):
                return False
    except (KeyError, TypeError, ValueError, LayoutSolveError):
        return False
    return True


def _unsat_attempt_identity(
    realized_artifact: Mapping[str, Any],
    unit_artifact: Mapping[str, Any],
    layout_seed: int,
    config: LayoutConfig,
    *,
    require_pick_planning_witness: bool,
    solver_identity: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    try:
        join_valid = _unit_join_valid(realized_artifact, unit_artifact)
    except Exception:
        return None
    if solver_identity is None or not join_valid:
        return None
    try:
        payload = {
            "realized_family_id": str(
                realized_artifact["realized_family_id"]
            ),
            "layout_seed": int(layout_seed),
            "m10_realization": {
                "schema_version": realized_artifact.get("schema_version"),
                "artifact_kind": realized_artifact.get("artifact_kind"),
                "realization_version": realized_artifact.get(
                    "realization_version"
                ),
                "artifact_hash": str(realized_artifact["artifact_hash"]),
                "config_hash": str(
                    realized_artifact.get("config_hash") or ""
                ),
            },
            "m11_unit_build": {
                "schema_version": unit_artifact.get("schema_version"),
                "artifact_kind": unit_artifact.get("artifact_kind"),
                "unit_build_version": unit_artifact.get("unit_build_version"),
                "artifact_hash": str(unit_artifact["artifact_hash"]),
                "source_realization_hash": str(
                    unit_artifact["source_realization_hash"]
                ),
                "config_hash": str(
                    unit_artifact.get("config_hash") or ""
                ),
            },
            "m12_layout": {
                "schema_version": SCHEMA_VERSION,
                "layout_solver_version": LAYOUT_SOLVER_VERSION,
                "config_hash": config.hash,
                "require_pick_planning_witness": bool(
                    require_pick_planning_witness
                ),
            },
            "solver_identity": canonical_data(dict(solver_identity)),
        }
        return canonical_data(
            {**payload, "identity_hash": artifact_hash(payload)}
        )
    except Exception:
        return None


def _unsat_failure_binding(
    failure: Mapping[str, Any],
    *,
    provider_identity: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return immutable outcome evidence only for deterministic layout unsat."""

    if (
        str(failure.get("code") or "") not in GEOMETRIC_UNSAT_CODES
        or failure.get("geometric_unsat") is not True
        or str(failure.get("owner") or "") != "layout"
        or contains_limited_planning_failure(failure)
    ):
        return None
    code = str(failure["code"])
    metrics = failure.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    if code != "pick_planning_unreachable":
        if (
            metrics.get("frozen_request") is not None
            or metrics.get("frozen_witness") is not None
            or metrics.get("planning_rejections") is not None
        ):
            return None
        return {
            "outcome": "deterministic_geometric_unsat",
            "request_status": "not_applicable",
            "request_hash": None,
            "witness_status": "not_applicable",
            "witness_hash": None,
            "provider_identity": None,
        }
    raw_rejections = metrics.get("planning_rejections")
    raw_cpu_exhaustion = metrics.get("cpu_layout_exhaustion")
    if provider_identity is None or not isinstance(raw_rejections, list):
        return None
    try:
        checked_rejections, _ = _pick_planning_trace_failures(raw_rejections)
        if not isinstance(raw_cpu_exhaustion, Mapping):
            return None
        cpu_exhaustion = _layout_solve_error_from_record(raw_cpu_exhaustion)
        reproduced = _pick_planning_exhausted_error(
            checked_rejections, cpu_exhaustion
        )
        if canonical_data(reproduced.to_record()) != canonical_data(failure):
            return None
    except (KeyError, TypeError, ValueError, LayoutSolveError):
        return None
    return canonical_data(
        {
            "outcome": "deterministic_pick_planning_unreachable",
            "request_status": "bound",
            "request_hashes": [
                str(row["request"].get("request_hash") or "")
                for row in checked_rejections
            ],
            "witness_status": "failed",
            "witness_hashes": [
                str(row["witness"].get("witness_hash") or "")
                for row in checked_rejections
            ],
            "placement_hashes": [
                str(row["placement_hash"]) for row in checked_rejections
            ],
            "provider_identity": dict(provider_identity),
        }
    )


def _verify_unsat_failure_against_current_solver(
    failure: Mapping[str, Any],
    realized_artifact: Mapping[str, Any],
    unit_artifact: Mapping[str, Any],
    layout_seed: int,
    config: LayoutConfig,
    *,
    verify_cpu_unsat: bool,
) -> bool:
    """Reproduce evidence needed to safely import or reuse one failure.

    Planner-unreachable records always rebuild the deterministic tentative
    SceneSpec and exact request before their frozen failed witness is accepted.
    A new sidecar may skip CPU re-solving for non-planner geometric failures
    because its stronger source-code identity already proves the solver is
    unchanged.  Legacy aggregate bootstrap lacks that identity and therefore
    sets ``verify_cpu_unsat``.
    """

    code = str(failure.get("code") or "")
    if code != "pick_planning_unreachable" and not verify_cpu_unsat:
        return True
    if code == "pick_planning_unreachable":
        metrics = failure.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        raw_rejections = metrics.get("planning_rejections")
        if not isinstance(raw_rejections, list):
            return False
        try:
            replay = _FrozenPickPlanningReplay(raw_rejections)
            _solve_scene_spec_with_pick_planning(
                realized_artifact,
                unit_artifact,
                layout_seed,
                config,
                invoke_pick_planning=replay,
                require_pick_planning_witness=True,
            )
        except LayoutSolveError as reproduced:
            try:
                replay.assert_consumed()
            except (NameError, ValueError):
                return False
            return canonical_data(reproduced.to_record()) == canonical_data(
                failure
            )
        except (KeyError, TypeError, ValueError):
            return False
        return False
    try:
        solve_scene_spec(realized_artifact, unit_artifact, layout_seed, config)
    except LayoutSolveError as reproduced:
        if reproduced.realized_family_id is None:
            reproduced.realized_family_id = str(
                realized_artifact["realized_family_id"]
            )
        if reproduced.layout_seed is None:
            reproduced.layout_seed = int(layout_seed)
        return canonical_data(reproduced.to_record()) == canonical_data(
            failure
        )
    except (KeyError, TypeError, ValueError):
        return False
    return False


def _unsat_cache_artifact(
    failure: Mapping[str, Any],
    attempt_identity: Mapping[str, Any],
    *,
    provider_identity: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    binding = _unsat_failure_binding(
        failure, provider_identity=provider_identity
    )
    if binding is None:
        return None
    payload = {
        "schema_version": 1,
        "artifact_kind": "m12_deterministic_unsat_attempt_cache",
        "cache_policy_version": UNSAT_ATTEMPT_CACHE_VERSION,
        "attempt_identity": canonical_data(dict(attempt_identity)),
        "failure_binding": binding,
        "failure": canonical_data(dict(failure)),
    }
    return canonical_data(
        {**payload, "artifact_hash": artifact_hash(payload)}
    )


def _write_unsat_attempt_cache(
    path: Path,
    failure: Mapping[str, Any],
    attempt_identity: Mapping[str, Any] | None,
    *,
    provider_identity: Mapping[str, Any] | None,
) -> None:
    if attempt_identity is None:
        return
    try:
        artifact = _unsat_cache_artifact(
            failure,
            attempt_identity,
            provider_identity=provider_identity,
        )
        if artifact is None:
            _discard_unsat_cache_file(path)
            return
        _write_yaml(path, artifact)
    except Exception:
        # Caching is an acceleration sidecar.  A cache filesystem or YAML
        # failure must not alter the canonical M12 classification.
        _discard_unsat_cache_file(path)


def _load_reusable_unsat_attempt(
    path: Path,
    realized_artifact: Mapping[str, Any],
    unit_artifact: Mapping[str, Any],
    layout_seed: int,
    config: LayoutConfig,
    *,
    attempt_identity: Mapping[str, Any] | None,
    provider_identity: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    cached = _try_load_yaml(path)
    if cached is None:
        if path.exists():
            _discard_unsat_cache_file(path)
        return None
    try:
        failure = cached.get("failure")
        valid = (
            attempt_identity is not None
            and cached.get("schema_version") == 1
            and cached.get("artifact_kind")
            == "m12_deterministic_unsat_attempt_cache"
            and cached.get("cache_policy_version")
            == UNSAT_ATTEMPT_CACHE_VERSION
            and equivalent_attempt_identity(cached.get("attempt_identity"), attempt_identity)
            and isinstance(failure, Mapping)
            and cached.get("artifact_hash")
            == artifact_hash(cached, omit_keys=("artifact_hash",))
        )
        if valid:
            binding = _unsat_failure_binding(
                failure, provider_identity=provider_identity
            )
            valid = (
                binding is not None
                and canonical_data(cached.get("failure_binding"))
                == canonical_data(binding)
                and str(failure.get("realized_family_id") or "")
                == str(realized_artifact["realized_family_id"])
                and int(failure.get("layout_seed", -1)) == int(layout_seed)
                and _verify_unsat_failure_against_current_solver(
                    failure,
                    realized_artifact,
                    unit_artifact,
                    layout_seed,
                    config,
                    verify_cpu_unsat=False,
                )
            )
    except Exception:
        valid = False
    if not valid:
        _discard_unsat_cache_file(path)
        return None
    assert isinstance(failure, Mapping)
    return canonical_data(dict(failure))


def _shortcut_record(
    realized: Mapping[str, Any], scene_spec: Mapping[str, Any]
) -> dict[str, Any]:
    entities = scene_spec["entities"]
    actor_xy = scene_spec["actor"]["base_pose"]["position"][:2]
    x_order = sorted(
        entities, key=lambda key: (entities[key]["pose"]["position"][0], key)
    )
    robot_order = sorted(
        entities,
        key=lambda key: (
            math.dist(entities[key]["pose"]["position"][:2], actor_xy),
            key,
        ),
    )
    entity_features = {
        key: {
            "x_rank": x_order.index(key),
            "robot_distance_rank": robot_order.index(key),
            "is_leftmost": key == x_order[0],
            "is_rightmost": key == x_order[-1],
            "is_nearest_robot": key == robot_order[0],
            "reach_zone": scene_spec["reach"][key]["zone"],
        }
        for key in sorted(entities)
    }
    frozen_pick_arms = {
        str(requirement["goal_variant_id"]): str(
            requirement["selected_arm"]
        )
        for reach_record in (scene_spec.get("reach") or {}).values()
        for requirement in reach_record.get("pick_planning_requirements", [])
    }
    variants: list[dict[str, Any]] = []
    for variant in realized.get("goal_variants", []):
        goal = variant.get("goal") or {}
        subject = goal.get("subject")
        if not subject or subject not in entity_features:
            continue
        target_keys = [
            str(goal[key])
            for key in ("target", "target2")
            if goal.get(key) in entities
        ]
        subject_xy = entities[subject]["pose"]["position"][:2]
        semantic_arm = goal.get("arm")
        reachable_arms = list(
            (scene_spec["reach"].get(str(subject)) or {}).get(
                "reachable_arms", []
            )
        )
        eligible_arms = set(str(arm) for arm in reachable_arms)
        if str(goal.get("action_family") or "") == "O2O_relation":
            terminal_sets = [
                {
                    str(arm)
                    for arm in terminal.get("reachable_arms", [])
                    if str(arm) in {"left", "right"}
                }
                for terminal in (
                    scene_spec.get("terminal_reach") or {}
                ).values()
                if str(terminal.get("subject") or "") == str(subject)
                and str(terminal.get("relation") or "")
                == str(goal.get("relation") or "")
                and terminal.get("proxy_passed") is True
            ]
            if terminal_sets:
                eligible_arms &= set.intersection(*terminal_sets)
        selected_arm = (
            str(semantic_arm)
            if semantic_arm in {"left", "right"}
            else frozen_pick_arms.get(str(variant["id"]))
        )
        target_distance_ranks = {}
        for target in target_keys:
            target_xy = entities[target]["pose"]["position"][:2]
            distance_order = sorted(
                entities,
                key=lambda key: (
                    math.dist(
                        entities[key]["pose"]["position"][:2], target_xy
                    ),
                    key,
                ),
            )
            target_distance_ranks[target] = distance_order.index(str(subject))
        variants.append(
            {
                "variant_id": str(variant["id"]),
                "subject_key": str(subject),
                "subject_features": entity_features[str(subject)],
                "execution_arm": selected_arm,
                "execution_arm_semantic": semantic_arm is not None,
                "execution_arm_dual_reachable": (
                    semantic_arm is None
                    and str(variant["id"]) in frozen_pick_arms
                    and {"left", "right"}.issubset(eligible_arms)
                ),
                "target_keys": target_keys,
                "target_distance_ranks": target_distance_ranks,
                "target_distances_m": [
                    _clean(
                        math.dist(
                            subject_xy,
                            entities[target]["pose"]["position"][:2],
                        )
                    )
                    for target in target_keys
                ],
                "same_side_as_first_target": (
                    (
                        float(subject_xy[0]) < 0
                        and float(entities[target_keys[0]]["pose"]["position"][0])
                        < 0
                    )
                    or (
                        float(subject_xy[0]) >= 0
                        and float(entities[target_keys[0]]["pose"]["position"][0])
                        >= 0
                    )
                )
                if target_keys
                else None,
            }
        )
    return {
        "scene_spec_id": scene_spec["id"],
        "realized_family_id": scene_spec["realized_family_id"],
        "layout_seed": scene_spec["layout_seed"],
        "entity_features": entity_features,
        "variant_features": variants,
    }


def _reports(
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    shortcuts: Sequence[Mapping[str, Any]],
    family_outcomes: Sequence[Mapping[str, Any]],
    *,
    expected_family_count: int,
    layout_seed_candidates: Sequence[int],
    success_quota: int,
    config: LayoutConfig,
    source_set_hashes: Mapping[str, str],
    search_budget: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    attempt_count = sum(
        len(outcome.get("attempted_seeds") or [])
        for outcome in family_outcomes
    )
    search_budget = normalize_search_budget(search_budget)
    geometric_unsat_count, budget_exhausted_count, input_error_count = failure_summary_counts(
        failures, budgeted=search_budget is not None
    )
    quota_met_family_count = sum(
        outcome.get("success_quota_met") is True
        for outcome in family_outcomes
    )
    shortfall_family_count = sum(
        outcome.get("terminal_status") == "shortfall"
        for outcome in family_outcomes
    )
    blocked_family_count = sum(
        outcome.get("terminal_status") == "blocked"
        for outcome in family_outcomes
    )
    discarded_success_count = sum(
        len(outcome.get("successful_seeds") or [])
        for outcome in family_outcomes
        if outcome.get("success_quota_met") is not True
    )
    missing_witness_count = sum(
        row.get("pick_planning_witness_required") is True
        and row.get("pick_planning_witness_status") == "missing"
        for row in rows
    )
    summary = {
        "expected_realized_family_count": expected_family_count,
        "layout_seed_candidates": list(layout_seed_candidates),
        "max_layout_seed_attempts": len(layout_seed_candidates),
        "success_quota_per_family": int(success_quota),
        "attempt_count": attempt_count,
        "solved_count": len(rows),
        "published_family_count": quota_met_family_count,
        "success_quota_met_family_count": quota_met_family_count,
        "shortfall_family_count": shortfall_family_count,
        "blocked_family_count": blocked_family_count,
        "discarded_success_count": discarded_success_count,
        "unsat_count": geometric_unsat_count,
        "input_error_count": input_error_count,
        "missing_witness_count": missing_witness_count,
    }
    if search_budget is not None:
        summary["search_budget_exhausted_count"] = budget_exhausted_count
    implementation_complete = (
        len(family_outcomes) == expected_family_count
        and len(rows) + len(failures) + discarded_success_count
        == attempt_count
        and input_error_count == 0
        and blocked_family_count == 0
    )
    coverage_complete = (
        implementation_complete
        and success_quota == DEFAULT_LAYOUT_SUCCESS_COUNT
        and shortfall_family_count == 0
        and missing_witness_count == 0
    )
    publication_ready = (
        implementation_complete
        and success_quota == DEFAULT_LAYOUT_SUCCESS_COUNT
        and bool(rows)
        and missing_witness_count == 0
    )
    artifact_rows = [
        {
            "id": f"{row['realized_family_id']}:{row['layout_seed']}",
            "hash": row["artifact_hash"],
        }
        for row in sorted(
            rows,
            key=lambda value: (
                str(value["realized_family_id"]),
                int(value["layout_seed"]),
            ),
        )
    ]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "layout_report",
        "layout_solver_version": LAYOUT_SOLVER_VERSION,
        "config_hash": config.hash,
        "source_set_hashes": dict(source_set_hashes),
        "summary": summary,
        "family_outcomes": canonical_data(list(family_outcomes)),
        "failure_code_counts": dict(
            sorted(Counter(str(row["code"]) for row in failures).items())
        ),
        "artifact_set_hash": artifact_hash(artifact_rows),
        "implementation_complete": implementation_complete,
        "publication_policy": PUBLICATION_POLICY,
        "publication_scope": "solved_scene_specs",
        "coverage_complete": coverage_complete,
        "publication_ready": publication_ready,
        "acceptance_ready": publication_ready,
    }
    if search_budget is not None:
        report["search_budget"] = search_budget
    report["report_hash"] = artifact_hash(report)
    unsat: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "layout_unsat_report",
        "layout_solver_version": LAYOUT_SOLVER_VERSION,
        "config_hash": config.hash,
        "source_set_hashes": dict(source_set_hashes),
        "failure_count": len(failures),
        "failures": canonical_data(list(failures)),
        "family_outcomes": canonical_data(list(family_outcomes)),
        "implementation_complete": implementation_complete,
        "publication_policy": PUBLICATION_POLICY,
        "publication_scope": "layout_unsat_audit",
        "coverage_complete": coverage_complete,
        "publication_ready": publication_ready,
    }
    if search_budget is not None:
        unsat["search_budget"] = search_budget
    unsat["report_hash"] = artifact_hash(unsat)
    reach_counts = Counter(
        str(feature["reach_zone"])
        for row in shortcuts
        for feature in row["entity_features"].values()
    )
    subject_extreme_counts = Counter(
        "leftmost"
        if row["subject_features"]["is_leftmost"]
        else "rightmost"
        if row["subject_features"]["is_rightmost"]
        else "interior"
        for shortcut in shortcuts
        for row in shortcut["variant_features"]
    )
    same_side_counts = Counter(
        "same_side"
        if row["same_side_as_first_target"] is True
        else "opposite_side"
        if row["same_side_as_first_target"] is False
        else "no_target"
        for shortcut in shortcuts
        for row in shortcut["variant_features"]
    )
    non_d4_arm_counts = Counter(
        str(row["execution_arm"])
        for shortcut in shortcuts
        for row in shortcut["variant_features"]
        if row.get("execution_arm_semantic") is False
        and row.get("execution_arm") in {"left", "right"}
    )
    non_d4_dual_reachable_arm_counts = Counter(
        str(row["execution_arm"])
        for shortcut in shortcuts
        for row in shortcut["variant_features"]
        if row.get("execution_arm_semantic") is False
        and row.get("execution_arm_dual_reachable") is True
        and row.get("execution_arm") in {"left", "right"}
    )
    variant_count = sum(len(row["variant_features"]) for row in shortcuts)

    def dominant_share(counts: Mapping[str, int]) -> float:
        total = sum(counts.values())
        return (
            _clean(max(counts.values(), default=0) / total) if total else 0.0
        )

    dominant_reach_share = dominant_share(reach_counts)
    dominant_extreme_share = dominant_share(subject_extreme_counts)
    same_side_known = same_side_counts.get(
        "same_side", 0
    ) + same_side_counts.get("opposite_side", 0)
    same_side_share = (
        _clean(same_side_counts.get("same_side", 0) / same_side_known)
        if same_side_known
        else 0.0
    )
    risks = []
    if dominant_reach_share >= 0.95:
        risks.append("reach_zone_dominance")
    if dominant_extreme_share >= 0.95:
        risks.append("subject_extreme_dominance")
    if same_side_known and (same_side_share >= 0.95 or same_side_share <= 0.05):
        risks.append("same_side_dominance")
    shortcut: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "layout_shortcut_features",
        "layout_solver_version": LAYOUT_SOLVER_VERSION,
        "config_hash": config.hash,
        "source_set_hashes": dict(source_set_hashes),
        "record_count": len(shortcuts),
        "records": canonical_data(list(shortcuts)),
        "aggregate": {
            "variant_count": variant_count,
            "reach_zone_counts": dict(sorted(reach_counts.items())),
            "subject_extreme_counts": dict(sorted(subject_extreme_counts.items())),
            "same_side_counts": dict(sorted(same_side_counts.items())),
            "dominant_reach_zone_share": dominant_reach_share,
            "dominant_subject_extreme_share": dominant_extreme_share,
            "same_side_share": same_side_share,
            "non_d4_pick_execution_arm_counts": dict(
                sorted(non_d4_arm_counts.items())
            ),
            "non_d4_pick_dual_reachable_arm_counts": dict(
                sorted(non_d4_dual_reachable_arm_counts.items())
            ),
            "non_d4_pick_dual_reachable_arm_imbalance": abs(
                non_d4_dual_reachable_arm_counts.get("left", 0)
                - non_d4_dual_reachable_arm_counts.get("right", 0)
            ),
            "degeneracy_risks": risks,
        },
    }
    shortcut["report_hash"] = artifact_hash(shortcut)
    return report, unsat, shortcut


def _source_set_hashes(
    realized_sources: Sequence[Mapping[str, Any]],
    unit_sources: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    return {
        "realized_families": artifact_hash(
            [
                {
                    "id": row["realized_family_id"],
                    "hash": row["artifact_hash"],
                }
                for row in realized_sources
            ]
        ),
        "units": artifact_hash(
            [
                {
                    "id": row["realized_family_id"],
                    "hash": row["artifact_hash"],
                }
                for row in unit_sources
            ]
        ),
    }


def _source_set_hashes_from_paths(
    source_paths: Sequence[Path], unit_dir: Path
) -> dict[str, str]:
    def header(path: Path) -> dict[str, str]:
        wanted = {"realized_family_id", "artifact_hash"}
        found: dict[str, str] = {}
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line[:1].isspace() or ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    if key in wanted:
                        found[key] = value.strip()
                        if wanted <= found.keys():
                            break
        except OSError:
            return {}
        return found

    realized_rows: list[dict[str, str]] = []
    unit_rows: list[dict[str, str]] = []
    for source_path in source_paths:
        realized = header(source_path)
        realized_id = realized.get("realized_family_id", source_path.stem)
        unit = header(unit_dir / source_path.name)
        realized_rows.append(
            {
                "realized_family_id": realized_id,
                "artifact_hash": realized.get("artifact_hash", "missing"),
            }
        )
        unit_rows.append(
            {
                "realized_family_id": realized_id,
                "artifact_hash": unit.get("artifact_hash", "missing"),
            }
        )
    return _source_set_hashes(realized_rows, unit_rows)


def _realized_source_path_index(
    source_paths: Sequence[Path],
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in source_paths:
        realized_id = ""
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.startswith("realized_family_id:"):
                        realized_id = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
        realized_id = realized_id or path.stem
        if realized_id in result:
            raise ValueError(f"duplicate realized family id: {realized_id}")
        result[realized_id] = path
    return result


def _bootstrap_unsat_failures(
    output_dir: Path,
    *,
    config: LayoutConfig,
    source_set_hashes: Mapping[str, str],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Load exact legacy aggregate failures as *candidates* for verification.

    The old aggregate does not bind the layout source-code manifest introduced
    by the sidecar.  Consequently these rows are never trusted directly: each
    worker reproduces CPU unsat before promoting a row into a sidecar.  Legacy
    planner-unreachable rows are ineligible because the aggregate did not bind
    the provider/runtime identity that originally produced their no-path
    witness; current identity must never be retroactively attached to them.
    """

    unsat = _try_load_yaml(output_dir / UNSAT_FILE)
    if (
        unsat is None
        or unsat.get("schema_version") != SCHEMA_VERSION
        or unsat.get("artifact_kind") != "layout_unsat_report"
        or unsat.get("layout_solver_version") != LAYOUT_SOLVER_VERSION
        or unsat.get("config_hash") != config.hash
        or canonical_data(unsat.get("source_set_hashes"))
        != canonical_data(source_set_hashes)
        or unsat.get("implementation_complete") is not True
        or unsat.get("publication_policy") != PUBLICATION_POLICY
        or unsat.get("publication_scope") != "layout_unsat_audit"
        or not _report_hash_valid(unsat)
    ):
        return {}
    failures = unsat.get("failures")
    if not isinstance(failures, list) or unsat.get("failure_count") != len(
        failures
    ):
        return {}
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in failures:
        if not isinstance(raw, Mapping):
            return {}
        try:
            key = (
                str(raw["realized_family_id"]),
                int(raw["layout_seed"]),
            )
        except (KeyError, TypeError, ValueError):
            return {}
        if key in result:
            return {}
        try:
            failure = canonical_data(dict(raw))
            if str(failure.get("code") or "") == "pick_planning_unreachable":
                continue
            binding = _unsat_failure_binding(
                failure, provider_identity=None
            )
        except Exception:
            return {}
        if binding is not None:
            result[key] = failure
    return result


_LAYOUT_WORKER_CONTEXT: Mapping[str, Any] = {}
_LAYOUT_PLANNING_SEMAPHORE: Any | None = None
_LAYOUT_PLANNING_DEVICE_QUEUE: Any | None = None
_LAYOUT_FALLBACK_CUDA_DEVICE: str | None = None
_RUNTIME_CUDA_DEVICE_CONTEXT_KEY = "__m12_runtime_cuda_device"


def _init_layout_worker(
    context: Mapping[str, Any],
    planning_semaphore: Any | None,
    planning_device_queue: Any | None = None,
    fallback_cuda_device: str | None = None,
) -> None:
    global _LAYOUT_WORKER_CONTEXT
    global _LAYOUT_PLANNING_SEMAPHORE
    global _LAYOUT_PLANNING_DEVICE_QUEUE
    global _LAYOUT_FALLBACK_CUDA_DEVICE
    _LAYOUT_WORKER_CONTEXT = context
    _LAYOUT_PLANNING_SEMAPHORE = planning_semaphore
    _LAYOUT_PLANNING_DEVICE_QUEUE = planning_device_queue
    _LAYOUT_FALLBACK_CUDA_DEVICE = fallback_cuda_device
    _cached_attempt_sources.cache_clear()


def _validated_worker_count(workers: int, *, name: str = "workers") -> int:
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError(f"{name} must be a positive integer")
    return worker_count


def _validated_layout_seeds(
    layout_seeds: Sequence[int],
) -> tuple[int, ...]:
    seeds = tuple(sorted(set(int(seed) for seed in layout_seeds)))
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("layout seeds must be a non-empty non-negative set")
    return seeds


def _validated_layout_seed_search(
    layout_seeds: Sequence[int],
    *,
    success_count: int,
    max_attempts: int,
) -> tuple[tuple[int, ...], int]:
    seeds = _validated_layout_seeds(layout_seeds)
    quota = int(success_count)
    attempt_limit = int(max_attempts)
    if quota < 1:
        raise ValueError("layout success count must be a positive integer")
    if attempt_limit < quota:
        raise ValueError(
            "max layout seed attempts must be at least layout success count"
        )
    candidates = seeds[:attempt_limit]
    if len(candidates) < quota:
        raise ValueError(
            "layout seed stream contains fewer candidates than the success "
            "quota"
        )
    return candidates, quota


def _validate_family_outcomes(
    raw_outcomes: Any,
    *,
    realized_ids: Sequence[str],
    seed_candidates: Sequence[int],
    success_quota: int,
    failures: Sequence[Mapping[str, Any]],
    search_budget: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    search_budget = normalize_search_budget(search_budget)
    if not isinstance(raw_outcomes, list):
        return {}, ["family_outcomes is not a list"]
    outcome_by_id: dict[str, dict[str, Any]] = {}
    failure_by_attempt: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in failures:
        if not isinstance(row, Mapping):
            errors.append("failure audit contains a non-mapping row")
            continue
        try:
            key = (
                str(row.get("realized_family_id") or ""),
                int(row.get("layout_seed", -1)),
            )
        except (TypeError, ValueError):
            errors.append("failure audit contains an invalid attempt key")
            continue
        if key in failure_by_attempt:
            errors.append(f"duplicate failure audit row: {key[0]}:{key[1]}")
        failure_by_attempt[key] = row
    candidates = tuple(int(seed) for seed in seed_candidates)
    candidate_set = set(candidates)
    for raw in raw_outcomes:
        if not isinstance(raw, Mapping):
            errors.append("family_outcomes contains a non-mapping row")
            continue
        outcome = canonical_data(dict(raw))
        realized_id = str(outcome.get("realized_family_id") or "")
        if not realized_id or realized_id in outcome_by_id:
            errors.append(f"duplicate or empty family outcome: {realized_id!r}")
            continue
        outcome_by_id[realized_id] = outcome
        try:
            attempted = [int(seed) for seed in outcome["attempted_seeds"]]
            successful = [int(seed) for seed in outcome["successful_seeds"]]
            failed = [int(seed) for seed in outcome["failed_seeds"]]
            published = [int(seed) for seed in outcome["published_seeds"]]
            quota = int(outcome["success_quota"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{realized_id}: malformed family outcome")
            continue
        if quota != success_quota:
            errors.append(f"{realized_id}: success quota mismatch")
        if attempted != list(candidates[: len(attempted)]) or not attempted:
            errors.append(f"{realized_id}: attempted seeds are not a seed-stream prefix")
        if not attempted:
            continue
        if (
            len(attempted) != len(set(attempted))
            or len(successful) != len(set(successful))
            or len(failed) != len(set(failed))
            or set(successful) & set(failed)
            or set(successful) | set(failed) != set(attempted)
            or not set(attempted) <= candidate_set
            or successful
            != [seed for seed in attempted if seed in set(successful)]
            or failed != [seed for seed in attempted if seed in set(failed)]
        ):
            errors.append(f"{realized_id}: inconsistent attempted seed partition")
        skipped_failures = [
            seed for seed in failed
            if (failure_by_attempt.get((realized_id, seed)) or {}).get("code") == LIVE_SEED_BUDGET_EXHAUSTED
        ]
        if search_budget is not None:
            live = outcome.get("live_seeds")
            skipped = outcome.get("budget_skipped_seeds")
            if (
                not isinstance(live, list)
                or not isinstance(skipped, list)
                or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in [*live, *skipped])
                or live != [seed for seed in attempted if seed in live]
                or skipped != skipped_failures
                or len(live) > search_budget["new_live_seed_limit"]
                or set(live) & set(skipped)
                or (skipped and (len(live) != search_budget["new_live_seed_limit"] or min(skipped) <= max(live)))
            ):
                errors.append(f"{realized_id}: invalid live search-budget accounting")
            else:
                for seed in skipped:
                    metrics = (failure_by_attempt[(realized_id, seed)].get("metrics") or {})
                    if (
                        not isinstance(metrics, Mapping)
                        or metrics.get("search_budget") != search_budget
                        or metrics.get("new_live_seed_count") != len(live)
                        or metrics.get("live_seeds") != live
                        or metrics.get("physical_feasibility") != "not_evaluated"
                    ):
                        errors.append(f"{realized_id}:{seed}: invalid budget-skipped evidence")
        elif skipped_failures or "live_seeds" in outcome or "budget_skipped_seeds" in outcome:
            errors.append(f"{realized_id}: budget outcome lacks a report search_budget")
        quota_met = outcome.get("success_quota_met") is True
        terminal_status = str(outcome.get("terminal_status") or "")
        terminal_code = str(outcome.get("terminal_code") or "")
        if quota_met:
            if (
                len(successful) != success_quota
                or published != successful
                or terminal_status != "published"
                or terminal_code != "success_quota_met"
                or attempted[-1] not in set(successful)
            ):
                errors.append(f"{realized_id}: invalid published family outcome")
        else:
            if published or len(successful) >= success_quota:
                errors.append(f"{realized_id}: invalid unpublished family outcome")
            if terminal_status == "shortfall":
                last_failure = failure_by_attempt.get(
                    (realized_id, attempted[-1])
                )
                last_metrics = (
                    last_failure.get("metrics")
                    if isinstance(last_failure, Mapping)
                    else None
                )
                structural_stop = bool(
                    isinstance(last_metrics, Mapping)
                    and last_metrics.get("seed_invariant_proof") is True
                    and terminal_code
                    == str(last_failure.get("code") or "")
                )
                exhausted = bool(
                    attempted == list(candidates)
                    and terminal_code == "layout_seed_search_exhausted"
                )
                if not (structural_stop or exhausted):
                    errors.append(f"{realized_id}: invalid exhausted family outcome")
            elif terminal_status == "blocked":
                last_failure = failure_by_attempt.get(
                    (realized_id, attempted[-1])
                )
                if (
                    last_failure is None
                    or _layout_seed_failure_disposition(last_failure)
                    != "blocked"
                    or terminal_code != str(last_failure.get("code") or "")
                ):
                    errors.append(f"{realized_id}: invalid blocked family outcome")
            else:
                errors.append(f"{realized_id}: unknown terminal status")
        for seed in failed:
            if (realized_id, seed) not in failure_by_attempt:
                errors.append(f"{realized_id}:{seed}: missing failure audit row")
    expected_ids = set(str(value) for value in realized_ids)
    if set(outcome_by_id) != expected_ids:
        errors.append("family outcome source set mismatch")
    expected_failed = {
        (realized_id, seed)
        for realized_id, outcome in outcome_by_id.items()
        for seed in outcome.get("failed_seeds") or []
    }
    if set(failure_by_attempt) != expected_failed:
        errors.append("failure audit rows do not match family outcomes")
    return outcome_by_id, errors


def _validated_cuda_devices(
    cuda_devices: Sequence[str] | None,
) -> tuple[str, ...]:
    if cuda_devices is None:
        return ()
    return normalize_cuda_device_selectors(cuda_devices)


def _completion_driven_ordered_submit(
    executor: Any,
    function: Callable[[Any], Any],
    tasks: Iterable[Any],
    *,
    max_in_flight: int,
) -> Iterable[Any]:
    """Keep workers fed by completion while yielding canonical results.

    ``max_in_flight`` bounds unfinished futures.  Completed out-of-order
    futures are buffered by input index so their completion order can never
    affect artifacts, progress, or exception ordering.  Replenishment pauses
    once that reorder buffer reaches the same bound; consequently at most
    ``2 * max_in_flight`` submitted tasks can sit behind a blocked canonical
    head.
    """

    window = _validated_worker_count(
        max_in_flight, name="max_in_flight"
    )
    task_iterator = enumerate(tasks)
    pending: dict[Any, int] = {}
    ready: dict[int, Any] = {}
    source_exhausted = False
    next_index = 0

    def submit_until_full() -> None:
        nonlocal source_exhausted
        while (
            not source_exhausted
            and len(pending) < window
            and len(ready) < window
        ):
            try:
                index, task = next(task_iterator)
            except StopIteration:
                source_exhausted = True
                return
            pending[executor.submit(function, task)] = index

    try:
        submit_until_full()
        while pending or ready:
            while next_index in ready:
                future = ready.pop(next_index)
                result = future.result()
                next_index += 1
                # Refill before yielding to the parent aggregator.  This
                # prevents its deterministic bookkeeping from starving the
                # executor between consecutive results.
                submit_until_full()
                yield result

            submit_until_full()
            if not pending:
                if ready:
                    raise RuntimeError(
                        "layout scheduler lost a canonical task result"
                    )
                break

            completed, _ = wait(
                tuple(pending), return_when=FIRST_COMPLETED
            )
            for future in sorted(completed, key=pending.__getitem__):
                ready[pending.pop(future)] = future
    finally:
        # Preserve deterministic first-error semantics while ensuring early
        # consumer exit does not leave queued work running unnecessarily.
        for future in tuple(pending):
            future.cancel()
        for future in tuple(ready.values()):
            future.cancel()


def _run_layout_attempt_workers(
    function: Callable[[Any], Any],
    tasks: Sequence[Any],
    *,
    workers: int,
    pick_planning_workers: int,
    context: Mapping[str, Any],
    cuda_devices: Sequence[str] | None = None,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
) -> Iterable[Any]:
    """Run independent attempts and yield results in canonical source order."""

    worker_count = min(
        _validated_worker_count(workers), max(1, len(tasks))
    )
    planning_worker_count = min(
        _validated_worker_count(
            pick_planning_workers, name="pick_planning_workers"
        ),
        worker_count,
    )
    devices = _validated_cuda_devices(cuda_devices)
    submission_window = _validated_worker_count(
        max_in_flight, name="max_in_flight"
    )
    if worker_count == 1:
        _init_layout_worker(
            context,
            None,
            fallback_cuda_device=(devices[0] if devices else None),
        )
        for task in tasks:
            yield function(task)
        return

    try:
        process_context = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - non-POSIX compatibility fallback
        # Simulator providers are not guaranteed to be spawn-pickleable.  Keep
        # correctness on such hosts and retain process parallelism on Linux.
        _init_layout_worker(
            context,
            None,
            fallback_cuda_device=(devices[0] if devices else None),
        )
        for task in tasks:
            yield function(task)
        return
    planning_device_queue = None
    planning_semaphore = None
    if devices:
        planning_device_queue = process_context.SimpleQueue()
        for index in range(planning_worker_count):
            planning_device_queue.put(devices[index % len(devices)])
    else:
        planning_semaphore = process_context.BoundedSemaphore(
            planning_worker_count
        )
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=process_context,
        initializer=_init_layout_worker,
        initargs=(
            context,
            planning_semaphore,
            planning_device_queue,
            None,
        ),
    ) as executor:
        # ``ProcessPoolExecutor.map`` eagerly submits every input on Python
        # 3.10.  M12 commonly has tens of thousands of layout attempts; the
        # wakeup pipe can then fill before the parent starts draining worker
        # results.  A large but bounded completion-driven window feeds sparse
        # planning work without allowing completion order into canonical artifacts.
        yield from _completion_driven_ordered_submit(
            executor,
            function,
            tasks,
            max_in_flight=submission_window,
        )


def _invoke_worker_pick_witness_provider(
    request: Mapping[str, Any] | None,
    *,
    context: Mapping[str, Any],
) -> dict[str, Any] | None:
    provider = _LAYOUT_WORKER_CONTEXT.get("pick_planning_witness_provider")
    required = bool(
        _LAYOUT_WORKER_CONTEXT.get("require_pick_planning_witness", False)
    )
    semaphore = _LAYOUT_PLANNING_SEMAPHORE if request is not None else None
    device_queue = (
        _LAYOUT_PLANNING_DEVICE_QUEUE if request is not None else None
    )
    cuda_device = _LAYOUT_FALLBACK_CUDA_DEVICE
    if device_queue is not None:
        cuda_device = str(device_queue.get())
    elif semaphore is not None:
        semaphore.acquire()
    try:
        provider_context = dict(context)
        if cuda_device is not None:
            provider_context[_RUNTIME_CUDA_DEVICE_CONTEXT_KEY] = cuda_device
        return _invoke_pick_witness_provider(
            provider,
            request,
            required=required,
            context=provider_context,
        )
    finally:
        if device_queue is not None:
            device_queue.put(cuda_device)
        elif semaphore is not None:
            semaphore.release()


@lru_cache(maxsize=4)
def _cached_attempt_sources(
    source_path_text: str, unit_path_text: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse seed-invariant M10/M11 inputs once per layout worker."""

    source_path = Path(source_path_text)
    unit_path = Path(unit_path_text)
    realized_source = _load_yaml(source_path)
    unit_source = _try_load_yaml(unit_path) or {
        "realized_family_id": realized_source["realized_family_id"],
        "artifact_hash": "missing",
    }
    return realized_source, unit_source


def _attempt_inputs(
    task: tuple[str, int],
) -> tuple[Path, int, dict[str, Any], dict[str, Any], str, LayoutConfig]:
    source_path = Path(task[0])
    seed = int(task[1])
    unit_dir = Path(str(_LAYOUT_WORKER_CONTEXT["unit_dir"]))
    realized_source, unit_source = _cached_attempt_sources(
        str(source_path), str(unit_dir / source_path.name)
    )
    realized_id = str(realized_source["realized_family_id"])
    config = _LAYOUT_WORKER_CONTEXT["config"]
    if not isinstance(config, LayoutConfig):
        raise TypeError("layout worker config is invalid")
    return (
        source_path,
        seed,
        realized_source,
        unit_source,
        realized_id,
        config,
    )


def _freeze_layout_attempt(
    task: tuple[str, int],
    *,
    live_search_budget: FamilyLiveSearchBudget | None = None,
) -> dict[str, Any]:
    (
        _source_path,
        seed,
        realized_source,
        unit_source,
        realized_id,
        config,
    ) = _attempt_inputs(task)
    output_dir = Path(str(_LAYOUT_WORKER_CONTEXT["output_dir"]))
    output_path = output_dir / realized_id / f"{seed}.yml"
    success_cache_path = _success_attempt_cache_path(
        output_dir, realized_id, seed
    )
    existing = _try_load_yaml(output_path)
    require_witness = bool(
        _LAYOUT_WORKER_CONTEXT.get("require_pick_planning_witness", False)
    )
    cache_enabled = bool(
        _LAYOUT_WORKER_CONTEXT.get("reuse_unsat_cache", False)
    )
    provider_identity = _LAYOUT_WORKER_CONTEXT.get(
        "pick_planning_provider_identity"
    )
    attempt_identity = _unsat_attempt_identity(
        realized_source,
        unit_source,
        seed,
        config,
        require_pick_planning_witness=require_witness,
        solver_identity=_LAYOUT_WORKER_CONTEXT.get(
            "unsat_cache_solver_identity"
        ),
    )
    cache_path = _unsat_attempt_cache_path(
        output_dir, realized_id, seed
    )

    def failed_result(failure: Mapping[str, Any]) -> dict[str, Any]:
        success_cache_path.unlink(missing_ok=True)
        return {
            "realized_family_id": realized_id,
            "layout_seed": seed,
            "expected_path": str(output_path),
            "row": None,
            "failure": canonical_data(dict(failure)),
            "shortcut": None,
        }

    try:
        existing_reusable = bool(
            existing is not None
            and _reusable(
                existing,
                realized_source,
                unit_source,
                seed,
                config,
                require_pick_planning_witness=require_witness,
            )
        )
        cached_success = (
            None if existing_reusable else _try_load_yaml(success_cache_path)
        )
        cached_success_reusable = bool(
            cached_success is not None
            and _reusable(
                cached_success,
                realized_source,
                unit_source,
                seed,
                config,
                require_pick_planning_witness=require_witness,
            )
        )
        if existing_reusable:
            artifact = existing
        elif cached_success_reusable:
            artifact = cached_success
        else:
            cached_failure = None
            if cache_enabled:
                cached_failure = _load_reusable_unsat_attempt(
                    cache_path,
                    realized_source,
                    unit_source,
                    seed,
                    config,
                    attempt_identity=attempt_identity,
                    provider_identity=provider_identity,
                )
                if cached_failure is None:
                    bootstrap_failures = _LAYOUT_WORKER_CONTEXT.get(
                        "bootstrap_unsat_failures"
                    )
                    bootstrap_failure = (
                        bootstrap_failures.get((realized_id, seed))
                        if isinstance(bootstrap_failures, Mapping)
                        else None
                    )
                    bootstrap_verified = False
                    if isinstance(bootstrap_failure, Mapping):
                        try:
                            bootstrap_verified = (
                                _verify_unsat_failure_against_current_solver(
                                    bootstrap_failure,
                                    realized_source,
                                    unit_source,
                                    seed,
                                    config,
                                    verify_cpu_unsat=True,
                                )
                            )
                        except Exception:
                            bootstrap_verified = False
                    if bootstrap_verified:
                        cached_failure = canonical_data(
                            dict(bootstrap_failure)
                        )
                        _write_unsat_attempt_cache(
                            cache_path,
                            cached_failure,
                            attempt_identity,
                            provider_identity=provider_identity,
                        )
            if cached_failure is not None:
                return failed_result(cached_failure)
            if live_search_budget is not None and not live_search_budget.start_live_seed(seed):
                raise LayoutSolveError(
                    LIVE_SEED_BUDGET_EXHAUSTED,
                    "new live-seed search allowance exhausted; cached later seeds remain eligible",
                    realized_family_id=realized_id,
                    layout_seed=seed,
                    owner="layout_search_budget",
                    metrics=live_search_budget.exhausted_metrics(),
                )
            solved = _solve_scene_spec_with_pick_planning(
                realized_source,
                unit_source,
                seed,
                config,
                invoke_pick_planning=lambda request, context: (
                    _invoke_worker_pick_witness_provider(
                        request, context=context
                    )
                ),
                require_pick_planning_witness=require_witness,
                **(
                    {"new_planning_layout_limit": live_search_budget.policy["new_planning_layout_limit"]}
                    if live_search_budget is not None else {}
                ),
            )
            scene_spec = solved["scene_spec"]
            preflight = solved["preflight"]
            pick_request = solved["pick_planning_request"]
            pick_witness = solved["pick_planning_witness"]
            artifact = _artifact(
                realized_source,
                unit_source,
                scene_spec,
                preflight,
                config,
                pick_request,
                pick_witness,
                pick_planning_required=pick_request is not None,
                pick_planning_rejections=solved["pick_planning_rejections"],
                pick_planning_arm_selection_retries=solved[
                    "pick_planning_arm_selection_retries"
                ],
            )
        if cache_enabled:
            _discard_unsat_cache_file(cache_path)
        # Every successful attempt has a private, resumable copy regardless
        # of whether this family ultimately reaches its publication quota.
        _write_yaml(success_cache_path, artifact)
        return {
            "realized_family_id": realized_id,
            "layout_seed": seed,
            "expected_path": str(output_path),
            "success_cache_path": str(success_cache_path),
            "artifact": artifact,
            "row": {
                "realized_family_id": realized_id,
                "layout_seed": seed,
                "artifact_hash": artifact["artifact_hash"],
                "pick_planning_witness_required": bool(
                    artifact.get("pick_planning_witness_required")
                ),
                "pick_planning_witness_status": str(
                    artifact.get("pick_planning_witness_status") or ""
                ),
            },
            "failure": None,
            "shortcut": _shortcut_record(
                realized_source["realized_scene_family"],
                artifact["scene_spec"],
            ),
        }
    except LayoutSolveError as exc:
        if exc.realized_family_id is None:
            exc.realized_family_id = realized_id
        if exc.layout_seed is None:
            exc.layout_seed = seed
        failure = exc.to_record()
        if cache_enabled:
            _write_unsat_attempt_cache(
                cache_path,
                failure,
                attempt_identity,
                provider_identity=provider_identity,
            )
        return failed_result(failure)


def _check_layout_attempt(task: tuple[str, int]) -> dict[str, Any]:
    (
        _source_path,
        seed,
        realized_source,
        unit_source,
        realized_id,
        config,
    ) = _attempt_inputs(task)
    output_dir = Path(str(_LAYOUT_WORKER_CONTEXT["output_dir"]))
    private_cache_attempts = _LAYOUT_WORKER_CONTEXT.get(
        "success_cache_attempts"
    ) or set()
    private_cache = (realized_id, seed) in private_cache_attempts
    path = (
        _success_attempt_cache_path(output_dir, realized_id, seed)
        if private_cache
        else output_dir / realized_id / f"{seed}.yml"
    )
    artifact = _try_load_yaml(path)
    reusable = bool(
        artifact is not None
        and _reusable(
            artifact,
            realized_source,
            unit_source,
            seed,
            config,
            require_pick_planning_witness=bool(
                _LAYOUT_WORKER_CONTEXT.get(
                    "require_pick_planning_witness", False
                )
            ),
        )
    )
    return {
        "realized_family_id": realized_id,
        "layout_seed": seed,
        "path": str(path),
        "artifact_present": artifact is not None,
        "reusable": reusable,
        "private_cache": private_cache,
        "artifact_hash": (
            str(artifact.get("artifact_hash") or "") if reusable else ""
        ),
        "missing_witness": bool(
            reusable
            and artifact.get("pick_planning_witness_required") is True
            and artifact.get("pick_planning_witness_status") == "missing"
        ),
    }


def _validate_layout_attempt(task: tuple[str, int]) -> dict[str, Any]:
    (
        _source_path,
        seed,
        realized_source,
        unit_source,
        realized_id,
        config,
    ) = _attempt_inputs(task)
    output_dir = Path(str(_LAYOUT_WORKER_CONTEXT["output_dir"]))
    private_cache_attempts = _LAYOUT_WORKER_CONTEXT.get(
        "success_cache_attempts"
    ) or set()
    path = (
        _success_attempt_cache_path(output_dir, realized_id, seed)
        if (realized_id, seed) in private_cache_attempts
        else output_dir / realized_id / f"{seed}.yml"
    )
    actual = _try_load_yaml(path)
    frozen_failures = _LAYOUT_WORKER_CONTEXT.get("frozen_failures") or {}
    frozen = frozen_failures.get((realized_id, seed))
    errors: list[str] = []
    if isinstance(frozen, Mapping) and frozen.get("code") == LIVE_SEED_BUDGET_EXHAUSTED:
        # This record makes no physical no-path claim. Its counted budget and
        # seed partition are checked by the mandatory family lifecycle audit.
        return {
            "realized_family_id": realized_id,
            "layout_seed": seed,
            "errors": ([f"{path}: skipped budget seed has a solved artifact"] if actual is not None else []),
            "budget_excluded": True,
        }
    replay: _FrozenPickPlanningReplay | None = None
    try:
        if (
            _LAYOUT_WORKER_CONTEXT.get("pick_planning_witness_provider")
            is not None
        ):

            def live_invoker(
                request: Mapping[str, Any], context: Mapping[str, Any]
            ) -> Mapping[str, Any] | None:
                return _invoke_worker_pick_witness_provider(
                    request, context=context
                )

            invoker: PickPlanningAttemptInvoker | None = live_invoker
        else:
            if (
                actual is not None
                and actual.get("pick_planning_witness") is not None
            ):
                replay = _FrozenPickPlanningReplay(
                    actual.get("pick_planning_rejections") or [],
                    final_arm_selection_retries=actual.get(
                        "pick_planning_arm_selection_retries"
                    )
                    or [],
                    final_request=actual.get("pick_planning_request"),
                    final_witness=actual.get("pick_planning_witness"),
                )
                invoker = replay
            elif isinstance(frozen, Mapping):
                metrics = frozen.get("metrics")
                metrics = metrics if isinstance(metrics, Mapping) else {}
                raw_rejections = metrics.get("planning_rejections")
                if isinstance(raw_rejections, list):
                    replay = _FrozenPickPlanningReplay(raw_rejections)
                    invoker = replay
                else:
                    invoker = None
            else:
                invoker = None
        solved = _solve_scene_spec_with_pick_planning(
            realized_source,
            unit_source,
            seed,
            config,
            invoke_pick_planning=invoker,
            require_pick_planning_witness=bool(
                _LAYOUT_WORKER_CONTEXT.get(
                    "require_pick_planning_witness", False
                )
            ),
            **(
                {"new_planning_layout_limit": int(frozen["metrics"]["max_pick_planning_layout_attempts_per_seed"])}
                if isinstance(frozen, Mapping)
                and frozen.get("code") == "pick_planning_search_budget_exhausted"
                and isinstance(frozen.get("metrics"), Mapping)
                and frozen["metrics"].get("max_pick_planning_layout_attempts_per_seed") is not None
                else {}
            ),
        )
        if replay is not None:
            replay.assert_consumed()
        scene_spec = solved["scene_spec"]
        preflight = solved["preflight"]
        pick_request = solved["pick_planning_request"]
        pick_witness = solved["pick_planning_witness"]
        expected = _artifact(
            realized_source,
            unit_source,
            scene_spec,
            preflight,
            config,
            pick_request,
            pick_witness,
            pick_planning_required=pick_request is not None,
            pick_planning_rejections=solved["pick_planning_rejections"],
            pick_planning_arm_selection_retries=solved[
                "pick_planning_arm_selection_retries"
            ],
        )
        if actual != expected:
            errors.append(f"{path}: content mismatch")
        if frozen is not None:
            errors.append(f"{path}: frozen as unsat but now solved")
    except LayoutSolveError as exc:
        if replay is not None:
            try:
                replay.assert_consumed()
            except ValueError as replay_exc:
                errors.append(f"{path}: {replay_exc}")
        if actual is not None:
            errors.append(f"{path}: frozen solved artifact is now unsat")
        if exc.realized_family_id is None:
            exc.realized_family_id = realized_id
        if exc.layout_seed is None:
            exc.layout_seed = seed
        if frozen != exc.to_record():
            errors.append(f"{path}: unsat record mismatch ({exc.code})")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"{path}: validation replay failed ({exc})")
    return {
        "realized_family_id": realized_id,
        "layout_seed": seed,
        "errors": errors,
    }


def _freeze_layout_seed_batch(
    task: tuple[str, tuple[int, ...]],
) -> list[dict[str, Any]]:
    source_path, seeds = task
    return [
        _freeze_layout_attempt((source_path, seed)) for seed in seeds
    ]


def _freeze_layout_family_batch(
    task: tuple[str, tuple[int, ...], int],
) -> dict[str, Any]:
    """Search one family until its successful-layout quota is complete.

    Geometric failures consume a candidate seed and search continues.  A
    non-geometric failure blocks the family immediately so an infrastructure
    problem can never be hidden by burning later seeds.  Successful attempts
    are first written to the private success cache and become visible release
    artifacts only after the whole family reaches its quota.
    """

    source_path, seeds, success_quota = task
    results: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    terminal_code = "layout_seed_search_exhausted"
    terminal_status = "shortfall"
    policy = normalize_search_budget(_LAYOUT_WORKER_CONTEXT.get("search_budget"))
    live_budget = FamilyLiveSearchBudget(policy) if policy is not None else None
    for seed in seeds:
        result = _freeze_layout_attempt(
            (source_path, seed),
            **({"live_search_budget": live_budget} if live_budget is not None else {}),
        )
        results.append(result)
        if result["row"] is not None:
            successful.append(result)
            if len(successful) == success_quota:
                terminal_code = "success_quota_met"
                terminal_status = "published"
                break
            continue
        failure = result["failure"]
        disposition = _layout_seed_failure_disposition(failure)
        metrics = failure.get("metrics")
        if (
            disposition == "seed_unsolved"
            and isinstance(metrics, Mapping)
            and metrics.get("seed_invariant_proof") is True
        ):
            terminal_code = str(failure.get("code") or "layout_unsat")
            terminal_status = "shortfall"
            break
        if disposition == "blocked":
            terminal_code = str(failure.get("code") or "layout_seed_blocked")
            terminal_status = "blocked"
            break

    quota_met = len(successful) == success_quota
    if quota_met:
        _publish_layout_family_artifacts(
            successful,
            success_quota=success_quota,
        )
        for result in successful:
            Path(result["success_cache_path"]).unlink(missing_ok=True)

    realized_id = str(results[0]["realized_family_id"])
    successful_seeds = [
        int(result["layout_seed"]) for result in successful
    ]
    failed_seeds = [
        int(result["layout_seed"])
        for result in results
        if result["failure"] is not None
    ]
    return {
        "realized_family_id": realized_id,
        "results": results,
        "family_outcome": {
            "realized_family_id": realized_id,
            "attempted_seeds": [
                int(result["layout_seed"]) for result in results
            ],
            "successful_seeds": successful_seeds,
            "failed_seeds": failed_seeds,
            "published_seeds": successful_seeds if quota_met else [],
            "success_quota": int(success_quota),
            "success_quota_met": quota_met,
            "terminal_status": terminal_status,
            "terminal_code": terminal_code,
            **(
                {
                    "live_seeds": list(live_budget.live_seeds),
                    "budget_skipped_seeds": list(live_budget.skipped_seeds),
                }
                if live_budget is not None else {}
            ),
        },
    }


def _check_layout_seed_batch(
    task: tuple[str, tuple[int, ...]],
) -> list[dict[str, Any]]:
    source_path, seeds = task
    return [_check_layout_attempt((source_path, seed)) for seed in seeds]


def _validate_layout_seed_batch(
    task: tuple[str, tuple[int, ...]],
) -> list[dict[str, Any]]:
    source_path, seeds = task
    return [
        _validate_layout_attempt((source_path, seed)) for seed in seeds
    ]


def _layout_seed_batches(
    source_paths: Sequence[Path],
    seeds: tuple[int, ...],
    *,
    workers: int,
    max_in_flight: int,
) -> tuple[list[tuple[str, tuple[int, ...]]], int]:
    """Batch full runs by family without starving small multi-GPU pilots."""

    worker_count = _validated_worker_count(workers)
    family_batching = len(seeds) > 1 and len(source_paths) >= worker_count
    if family_batching:
        tasks = [(str(path), seeds) for path in source_paths]
        batch_size = len(seeds)
    else:
        tasks = [
            (str(path), (seed,))
            for path in source_paths
            for seed in seeds
        ]
        batch_size = 1
    return tasks, max(1, int(max_in_flight) // batch_size)


def freeze_scene_specs(
    *,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    unit_dir: Path = DEFAULT_UNIT_DIR,
    layout_seeds: Sequence[int] = DEFAULT_LAYOUT_SEEDS,
    layout_success_count: int | None = None,
    max_layout_seed_attempts: int | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    workers: int = 1,
    d4_planning_workers: int = 1,
    cuda_devices: Sequence[str] | None = None,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    progress: Callable[[int, int, str], None] | None = None,
    d4_planning_witness_provider: PickPlanningWitnessProvider | None = None,
    require_d4_planning_witness: bool = False,
    pick_planning_workers: int | None = None,
    pick_planning_witness_provider: PickPlanningWitnessProvider | None = None,
    require_pick_planning_witness: bool | None = None,
    reuse_unsat_cache: bool = True,
    clear_unsat_cache: bool = False,
    search_budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze solved SceneSpecs and attributed unsat reports."""

    planning_workers = (
        d4_planning_workers
        if pick_planning_workers is None
        else int(pick_planning_workers)
    )
    planning_provider = (
        d4_planning_witness_provider
        if pick_planning_witness_provider is None
        else pick_planning_witness_provider
    )
    planning_required = (
        bool(require_d4_planning_witness)
        if require_pick_planning_witness is None
        else bool(require_pick_planning_witness)
    )
    requested_seeds = _validated_layout_seeds(layout_seeds)
    seed_candidates, success_quota = _validated_layout_seed_search(
        requested_seeds,
        success_count=(
            min(DEFAULT_LAYOUT_SUCCESS_COUNT, len(requested_seeds))
            if layout_success_count is None
            else int(layout_success_count)
        ),
        max_attempts=(
            len(requested_seeds)
            if max_layout_seed_attempts is None
            else int(max_layout_seed_attempts)
        ),
    )
    if clear_unsat_cache:
        _clear_unsat_attempt_cache(output_dir)
    config = LayoutConfig.load(config_path)
    search_budget = normalize_search_budget(search_budget)
    if search_budget is not None and search_budget["new_planning_layout_limit"] > config.max_pick_planning_layout_attempts_per_seed:
        raise ValueError("new planning budget exceeds the frozen validation ceiling")
    source_paths = sorted(realization_dir.glob("rsf_*.yml"))
    source_hashes = _source_set_hashes_from_paths(source_paths, unit_dir)
    solver_identity = (
        _unsat_cache_solver_identity() if reuse_unsat_cache else None
    )
    provider_identity = (
        _provider_semantic_identity(planning_provider)
        if reuse_unsat_cache
        else None
    )
    bootstrap_failures = (
        _bootstrap_unsat_failures(
            output_dir,
            config=config,
            source_set_hashes=source_hashes,
        )
        if (
            reuse_unsat_cache
            and not clear_unsat_cache
            and not _unsat_cache_has_entries(output_dir)
        )
        else {}
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    shortcuts: list[dict[str, Any]] = []
    family_outcomes: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    expected_success_cache_paths: set[Path] = set()
    attempt_budget = len(source_paths) * len(seed_candidates)
    tasks = [
        (str(path), seed_candidates, success_quota)
        for path in source_paths
    ]
    worker_context = {
        "unit_dir": str(unit_dir),
        "output_dir": str(output_dir),
        "config": config,
        "pick_planning_witness_provider": planning_provider,
        "require_pick_planning_witness": planning_required,
        "reuse_unsat_cache": bool(reuse_unsat_cache),
        "unsat_cache_solver_identity": solver_identity,
        "pick_planning_provider_identity": provider_identity,
        "bootstrap_unsat_failures": bootstrap_failures,
        "search_budget": search_budget,
    }
    completed_attempts = 0
    for family_result in _run_layout_attempt_workers(
        _freeze_layout_family_batch,
        tasks,
        workers=workers,
        pick_planning_workers=planning_workers,
        context=worker_context,
        cuda_devices=cuda_devices,
        max_in_flight=max_in_flight,
    ):
        outcome = family_result["family_outcome"]
        family_outcomes.append(outcome)
        quota_met = bool(outcome["success_quota_met"])
        for result in family_result["results"]:
            completed_attempts += 1
            realized_id = str(result["realized_family_id"])
            if quota_met and result["row"] is not None:
                expected_paths.add(Path(str(result["expected_path"])))
                rows.append(result["row"])
                shortcuts.append(result["shortcut"])
            elif result["row"] is not None:
                expected_success_cache_paths.add(
                    Path(str(result["success_cache_path"]))
                )
            elif result["failure"] is not None:
                failures.append(result["failure"])
            if progress is not None:
                progress(completed_attempts, attempt_budget, realized_id)
    if output_dir.is_dir():
        for extra in sorted(output_dir.glob("rsf_*/*.yml")):
            if extra not in expected_paths:
                extra.unlink()
        for directory in sorted(output_dir.glob("rsf_*")):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        success_cache_root = output_dir / SUCCESS_ATTEMPT_CACHE_DIR
        if success_cache_root.is_dir():
            for extra in success_cache_root.glob("**/*.yml"):
                if extra not in expected_success_cache_paths:
                    extra.unlink()
            for directory in sorted(
                success_cache_root.glob("**/*"), reverse=True
            ):
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
    report, unsat, shortcut = _reports(
        rows,
        failures,
        shortcuts,
        family_outcomes,
        expected_family_count=len(source_paths),
        layout_seed_candidates=seed_candidates,
        success_quota=success_quota,
        config=config,
        source_set_hashes=source_hashes,
        search_budget=search_budget,
    )
    _write_yaml(output_dir / REPORT_FILE, report)
    _write_yaml(output_dir / UNSAT_FILE, unsat)
    _write_yaml(output_dir / SHORTCUT_FILE, shortcut)
    return report


def check_scene_specs(
    *,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    unit_dir: Path = DEFAULT_UNIT_DIR,
    layout_seeds: Sequence[int] = DEFAULT_LAYOUT_SEEDS,
    layout_success_count: int | None = None,
    max_layout_seed_attempts: int | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    workers: int = 1,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    progress: Callable[[int, int, str], None] | None = None,
    require_d4_planning_witness: bool = False,
    require_pick_planning_witness: bool | None = None,
) -> dict[str, Any]:
    """Check source joins, solved artifacts, and aggregate lifecycle reports."""

    planning_required = (
        bool(require_d4_planning_witness)
        if require_pick_planning_witness is None
        else bool(require_pick_planning_witness)
    )
    requested_seeds = _validated_layout_seeds(layout_seeds)
    seed_candidates, success_quota = _validated_layout_seed_search(
        requested_seeds,
        success_count=(
            min(DEFAULT_LAYOUT_SUCCESS_COUNT, len(requested_seeds))
            if layout_success_count is None
            else int(layout_success_count)
        ),
        max_attempts=(
            len(requested_seeds)
            if max_layout_seed_attempts is None
            else int(max_layout_seed_attempts)
        ),
    )
    config = LayoutConfig.load(config_path)
    source_paths = sorted(realization_dir.glob("rsf_*.yml"))
    source_index = _realized_source_path_index(source_paths)
    source_hashes = _source_set_hashes_from_paths(source_paths, unit_dir)
    stale_paths: list[str] = []
    valid_attempts: set[tuple[str, int]] = set()
    missing_witness_attempts: set[tuple[str, int]] = set()
    artifact_hashes: dict[tuple[str, int], str] = {}
    report = _try_load_yaml(output_dir / REPORT_FILE)
    unsat = _try_load_yaml(output_dir / UNSAT_FILE)
    shortcut = _try_load_yaml(output_dir / SHORTCUT_FILE)
    try:
        search_budget = normalize_search_budget((report or {}).get("search_budget"))
        if search_budget != normalize_search_budget((unsat or {}).get("search_budget")):
            raise ValueError("report/unsat search budgets differ")
        if search_budget is not None and search_budget["new_planning_layout_limit"] > config.max_pick_planning_layout_attempts_per_seed:
            raise ValueError("new planning budget exceeds the validation ceiling")
    except ValueError:
        search_budget = None
        stale_paths.extend([str(output_dir / REPORT_FILE), str(output_dir / UNSAT_FILE)])
    failure_rows = (unsat or {}).get("failures", [])
    if not isinstance(failure_rows, list):
        failure_rows = []
        stale_paths.append(str(output_dir / UNSAT_FILE))
    raw_outcomes = (report or {}).get("family_outcomes")
    outcome_by_id, outcome_errors = _validate_family_outcomes(
        raw_outcomes,
        realized_ids=tuple(source_index),
        seed_candidates=seed_candidates,
        success_quota=success_quota,
        failures=failure_rows,
        search_budget=search_budget,
    )
    if outcome_errors or canonical_data((unsat or {}).get("family_outcomes")) != canonical_data(raw_outcomes):
        stale_paths.extend(
            [str(output_dir / REPORT_FILE), str(output_dir / UNSAT_FILE)]
        )
    published_attempts = {
        (realized_id, int(seed))
        for realized_id, outcome in outcome_by_id.items()
        for seed in outcome.get("published_seeds") or []
    }
    successful_attempts = {
        (realized_id, int(seed))
        for realized_id, outcome in outcome_by_id.items()
        for seed in outcome.get("successful_seeds") or []
    }
    private_success_attempts = successful_attempts - published_attempts
    tasks = [
        (str(source_index[realized_id]), (seed,))
        for realized_id, seed in sorted(successful_attempts)
        if realized_id in source_index
    ]
    worker_context = {
        "unit_dir": str(unit_dir),
        "output_dir": str(output_dir),
        "config": config,
        "require_pick_planning_witness": planning_required,
        "success_cache_attempts": private_success_attempts,
    }
    completed_attempts = 0
    for result_batch in _run_layout_attempt_workers(
        _check_layout_seed_batch,
        tasks,
        workers=workers,
        pick_planning_workers=1,
        context=worker_context,
        max_in_flight=max_in_flight,
    ):
        for result in result_batch:
            completed_attempts += 1
            realized_id = str(result["realized_family_id"])
            seed = int(result["layout_seed"])
            if result["reusable"]:
                valid_attempts.add((realized_id, seed))
                artifact_hashes[(realized_id, seed)] = str(
                    result["artifact_hash"]
                )
                if result["missing_witness"]:
                    missing_witness_attempts.add((realized_id, seed))
            elif result["artifact_present"]:
                stale_paths.append(str(result["path"]))
            else:
                stale_paths.append(str(result["path"]))
            if progress is not None:
                progress(
                    completed_attempts, len(successful_attempts), realized_id
                )
    if valid_attempts != successful_attempts:
        stale_paths.append(str(output_dir / REPORT_FILE))
    solved_count = len(published_attempts)
    unsat_count, budget_exhausted_count, input_error_count = failure_summary_counts(
        failure_rows, budgeted=search_budget is not None
    )
    missing_witness_count = len(missing_witness_attempts)
    family_outcomes = list(outcome_by_id.values())
    attempt_count = sum(
        len(outcome.get("attempted_seeds") or [])
        for outcome in family_outcomes
    )
    quota_met_family_count = sum(
        outcome.get("success_quota_met") is True
        for outcome in family_outcomes
    )
    shortfall_family_count = sum(
        outcome.get("terminal_status") == "shortfall"
        for outcome in family_outcomes
    )
    blocked_family_count = sum(
        outcome.get("terminal_status") == "blocked"
        for outcome in family_outcomes
    )
    discarded_success_count = sum(
        len(outcome.get("successful_seeds") or [])
        for outcome in family_outcomes
        if outcome.get("success_quota_met") is not True
    )
    expected_summary = {
        "expected_realized_family_count": len(source_paths),
        "layout_seed_candidates": list(seed_candidates),
        "max_layout_seed_attempts": len(seed_candidates),
        "success_quota_per_family": success_quota,
        "attempt_count": attempt_count,
        "solved_count": solved_count,
        "published_family_count": quota_met_family_count,
        "success_quota_met_family_count": quota_met_family_count,
        "shortfall_family_count": shortfall_family_count,
        "blocked_family_count": blocked_family_count,
        "discarded_success_count": discarded_success_count,
        "unsat_count": unsat_count,
        "input_error_count": input_error_count,
        "missing_witness_count": missing_witness_count,
    }
    if search_budget is not None:
        expected_summary["search_budget_exhausted_count"] = budget_exhausted_count
    implementation_complete = (
        len(family_outcomes) == len(source_paths)
        and solved_count + len(failure_rows) + discarded_success_count
        == attempt_count
        and input_error_count == 0
        and blocked_family_count == 0
    )
    coverage_complete = (
        implementation_complete
        and success_quota == DEFAULT_LAYOUT_SUCCESS_COUNT
        and shortfall_family_count == 0
        and missing_witness_count == 0
    )
    publication_ready = (
        implementation_complete
        and success_quota == DEFAULT_LAYOUT_SUCCESS_COUNT
        and solved_count > 0
        and missing_witness_count == 0
    )
    expected_artifact_set_hash = artifact_hash(
        [
            {"id": f"{realized_id}:{seed}", "hash": artifact_hashes[(realized_id, seed)]}
            for realized_id, seed in sorted(published_attempts)
            if (realized_id, seed) in artifact_hashes
        ]
    )
    report_common_valid = (
        report is not None
        and report.get("artifact_kind") == "layout_report"
        and report.get("layout_solver_version") == LAYOUT_SOLVER_VERSION
        and report.get("config_hash") == config.hash
        and report.get("source_set_hashes") == source_hashes
        and implementation_complete
        and report.get("implementation_complete") is True
        and report.get("publication_policy") == PUBLICATION_POLICY
        and report.get("publication_scope") == "solved_scene_specs"
        and report.get("coverage_complete") is coverage_complete
        and report.get("publication_ready") is publication_ready
        and report.get("acceptance_ready") is publication_ready
        and report.get("summary") == expected_summary
        and report.get("artifact_set_hash") == expected_artifact_set_hash
        and _report_hash_valid(report)
    )
    if not report_common_valid:
        stale_paths.append(str(output_dir / REPORT_FILE))
    if (
        unsat is None
        or unsat.get("artifact_kind") != "layout_unsat_report"
        or unsat.get("layout_solver_version") != LAYOUT_SOLVER_VERSION
        or unsat.get("config_hash") != config.hash
        or unsat.get("source_set_hashes") != source_hashes
        or unsat.get("failure_count") != len(failure_rows)
        or not implementation_complete
        or unsat.get("implementation_complete") is not True
        or unsat.get("publication_policy") != PUBLICATION_POLICY
        or unsat.get("publication_scope") != "layout_unsat_audit"
        or unsat.get("coverage_complete") is not coverage_complete
        or unsat.get("publication_ready") is not publication_ready
        or not _report_hash_valid(unsat)
    ):
        stale_paths.append(str(output_dir / UNSAT_FILE))
    if (
        shortcut is None
        or shortcut.get("artifact_kind") != "layout_shortcut_features"
        or shortcut.get("layout_solver_version") != LAYOUT_SOLVER_VERSION
        or shortcut.get("config_hash") != config.hash
        or shortcut.get("source_set_hashes") != source_hashes
        or not _report_hash_valid(shortcut)
    ):
        stale_paths.append(str(output_dir / SHORTCUT_FILE))
    expected_files = {
        output_dir / realized_id / f"{seed}.yml"
        for realized_id, seed in published_attempts
    }
    for realized_id, seed in published_attempts:
        duplicate_cache = _success_attempt_cache_path(
            output_dir, realized_id, seed
        )
        if duplicate_cache.exists():
            stale_paths.append(str(duplicate_cache))
    expected_private_cache_files = {
        _success_attempt_cache_path(output_dir, realized_id, seed)
        for realized_id, seed in private_success_attempts
    }
    success_cache_root = output_dir / SUCCESS_ATTEMPT_CACHE_DIR
    if success_cache_root.is_dir():
        for extra in success_cache_root.glob("**/*.yml"):
            if extra not in expected_private_cache_files:
                stale_paths.append(str(extra))
    if output_dir.is_dir():
        for extra in output_dir.glob("rsf_*/*.yml"):
            if extra not in expected_files:
                stale_paths.append(str(extra))
    return {
        "stale": bool(stale_paths),
        "stale_paths": sorted(set(stale_paths)),
        "attempt_count": attempt_count,
    }


def validate_scene_specs(
    *,
    realization_dir: Path = DEFAULT_REALIZATION_DIR,
    unit_dir: Path = DEFAULT_UNIT_DIR,
    layout_seeds: Sequence[int] = DEFAULT_LAYOUT_SEEDS,
    layout_success_count: int | None = None,
    max_layout_seed_attempts: int | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    workers: int = 1,
    d4_planning_workers: int = 1,
    cuda_devices: Sequence[str] | None = None,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    progress: Callable[[int, int, str], None] | None = None,
    d4_planning_witness_provider: PickPlanningWitnessProvider | None = None,
    require_d4_planning_witness: bool = False,
    pick_planning_workers: int | None = None,
    pick_planning_witness_provider: PickPlanningWitnessProvider | None = None,
    require_pick_planning_witness: bool | None = None,
) -> dict[str, Any]:
    """Re-solve every attempt and compare solved/unsat classification."""

    planning_workers = (
        d4_planning_workers
        if pick_planning_workers is None
        else int(pick_planning_workers)
    )
    planning_provider = (
        d4_planning_witness_provider
        if pick_planning_witness_provider is None
        else pick_planning_witness_provider
    )
    planning_required = (
        bool(require_d4_planning_witness)
        if require_pick_planning_witness is None
        else bool(require_pick_planning_witness)
    )
    requested_seeds = _validated_layout_seeds(layout_seeds)
    seed_candidates, success_quota = _validated_layout_seed_search(
        requested_seeds,
        success_count=(
            min(DEFAULT_LAYOUT_SUCCESS_COUNT, len(requested_seeds))
            if layout_success_count is None
            else int(layout_success_count)
        ),
        max_attempts=(
            len(requested_seeds)
            if max_layout_seed_attempts is None
            else int(max_layout_seed_attempts)
        ),
    )
    config = LayoutConfig.load(config_path)
    source_paths = sorted(realization_dir.glob("rsf_*.yml"))
    source_index = _realized_source_path_index(source_paths)
    report = _try_load_yaml(output_dir / REPORT_FILE) or {}
    unsat = _try_load_yaml(output_dir / UNSAT_FILE) or {}
    budget_errors: list[str] = []
    try:
        search_budget = normalize_search_budget(report.get("search_budget"))
        if search_budget != normalize_search_budget(unsat.get("search_budget")):
            raise ValueError("report/unsat search budgets differ")
    except ValueError as exc:
        search_budget = None
        budget_errors.append(str(exc))
    failure_rows = unsat.get("failures", [])
    if not isinstance(failure_rows, list):
        failure_rows = []
    frozen_failures = {
        (str(row["realized_family_id"]), int(row["layout_seed"])): row
        for row in failure_rows
        if isinstance(row, Mapping)
    }
    outcome_by_id, outcome_errors = _validate_family_outcomes(
        report.get("family_outcomes"),
        realized_ids=tuple(source_index),
        seed_candidates=seed_candidates,
        success_quota=success_quota,
        failures=failure_rows,
        search_budget=search_budget,
    )
    errors: list[str] = [*outcome_errors, *budget_errors]
    if canonical_data(unsat.get("family_outcomes")) != canonical_data(
        report.get("family_outcomes")
    ):
        errors.append("report/unsat family_outcomes mismatch")
    attempted = {
        (realized_id, int(seed))
        for realized_id, outcome in outcome_by_id.items()
        for seed in outcome.get("attempted_seeds") or []
    }
    successful = {
        (realized_id, int(seed))
        for realized_id, outcome in outcome_by_id.items()
        for seed in outcome.get("successful_seeds") or []
    }
    published = {
        (realized_id, int(seed))
        for realized_id, outcome in outcome_by_id.items()
        for seed in outcome.get("published_seeds") or []
    }
    private_success_attempts = successful - published
    attempt_count = len(attempted)
    tasks = [
        (str(source_index[realized_id]), (seed,))
        for realized_id, seed in sorted(attempted)
        if realized_id in source_index
    ]
    worker_context = {
        "unit_dir": str(unit_dir),
        "output_dir": str(output_dir),
        "config": config,
        "frozen_failures": frozen_failures,
        "pick_planning_witness_provider": planning_provider,
        "require_pick_planning_witness": planning_required,
        "success_cache_attempts": private_success_attempts,
    }
    completed_attempts = 0
    for result_batch in _run_layout_attempt_workers(
        _validate_layout_seed_batch,
        tasks,
        workers=workers,
        pick_planning_workers=planning_workers,
        context=worker_context,
        cuda_devices=cuda_devices,
        max_in_flight=max_in_flight,
    ):
        for result in result_batch:
            completed_attempts += 1
            errors.extend(result["errors"])
            if progress is not None:
                progress(
                    completed_attempts,
                    attempt_count,
                    str(result["realized_family_id"]),
                )
    lifecycle = check_scene_specs(
        realization_dir=realization_dir,
        unit_dir=unit_dir,
        layout_seeds=seed_candidates,
        layout_success_count=success_quota,
        max_layout_seed_attempts=len(seed_candidates),
        config_path=config_path,
        output_dir=output_dir,
        workers=workers,
        max_in_flight=max_in_flight,
        require_pick_planning_witness=planning_required,
    )
    errors.extend(
        f"{path}: stale lifecycle artifact" for path in lifecycle["stale_paths"]
    )
    return {
        "valid": not errors,
        "errors": errors,
        "attempt_count": attempt_count,
    }
