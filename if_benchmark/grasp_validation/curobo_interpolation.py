"""Install the reviewed stable interpolator before constructing CuRobo graphs.

No simulation, Torch or Warp imports at module import time. The installation
affects this RoboTwin process only; the external CuRobo installation is never
edited. Both CuRobo aliases must agree, so an unknown override fails closed.
"""
from hashlib import sha256
from importlib import import_module
from pathlib import Path

INTERPOLATION_VERSION = "curobo_stable_linear_equal_endpoints_v1"
# This is the reviewed timing/indexing ABI, not an arbitrary version whitelist.
# A dependency update needs a fresh timing/tail/batch compatibility regression.
UPSTREAM_SOURCE_SHA256 = "c560e6cd7e627ead29ab31c29cc7ec4aa5eddf9c2be4d11d45e1b6070439b775"


def install_stable_interpolation():
    """Idempotently replace the interpolation producer, before path validation.

    CuRobo's batch helper resolves its module's interpolator at each call;
    single, batch and graph-solver paths therefore consume the same producer.
    The producer retains the original output tensor buffers and timing rule.
    """
    trajectory = import_module("curobo.util.trajectory")
    upstream = import_module("curobo.util.warp_interpolation")
    from ._curobo_interpolation_warp import stable_linear_interpolation

    functions = (trajectory.get_cuda_linear_interpolation, upstream.get_cuda_linear_interpolation)
    if all(function is stable_linear_interpolation for function in functions):
        return _receipt()
    if functions[0] is not functions[1]:
        raise RuntimeError("curobo_interpolation_alias_conflict")
    if (getattr(functions[0], "__module__", None) != upstream.__name__
            or getattr(functions[0], "__name__", None) != "get_cuda_linear_interpolation"):
        raise RuntimeError("curobo_interpolation_unknown_override")
    source_hash = sha256(Path(upstream.__file__).read_bytes()).hexdigest()
    if source_hash != UPSTREAM_SOURCE_SHA256:
        raise RuntimeError(f"curobo_interpolation_unreviewed_timing_implementation:{source_hash}")
    trajectory.get_cuda_linear_interpolation = stable_linear_interpolation
    upstream.get_cuda_linear_interpolation = stable_linear_interpolation
    return _receipt()


def _receipt():
    return {"version": INTERPOLATION_VERSION, "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
            "timing_and_buffer_layout": "unchanged", "equal_endpoints": "exact",
            "collision_thresholds_changed": False}
