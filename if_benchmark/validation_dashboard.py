"""Static HTML dashboard for Milestone 4 asset validation results."""

from __future__ import annotations

import argparse
import html
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DATA_DIR = Path("if_benchmark/data/asset_validation")
DEFAULT_REPORTS_DIR = DEFAULT_DATA_DIR / "reports"
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_DIR / "validation_dashboard.html"


def write_validation_dashboard(
    *,
    report_path: Path | None = None,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    records_path: Path | None = None,
    attempts_path: Path | None = None,
    timing_summary_path: Path | None = None,
    validated_projection_path: Path | None = None,
    rejected_projection_path: Path | None = None,
    missing_evidence_path: Path | None = None,
    title: str = "Validation Statistics",
    eyebrow: str = "Milestone 4 Asset Validation",
) -> dict[str, Any]:
    """Build and write a self-contained validation dashboard HTML file."""

    model = build_dashboard_model(
        report_path=report_path,
        reports_dir=reports_dir,
        data_dir=data_dir,
        records_path=records_path,
        attempts_path=attempts_path,
        timing_summary_path=timing_summary_path,
        validated_projection_path=validated_projection_path,
        rejected_projection_path=rejected_projection_path,
        missing_evidence_path=missing_evidence_path,
        title=title,
        eyebrow=eyebrow,
    )
    html_text = render_dashboard_html(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return {
        "output_path": _path_string(output_path),
        "report_path": model["sources"]["report_path"],
        "evidence_id": model["report"].get("evidence_id"),
        "record_count": model["overview"]["record_count"],
        "attempt_count": model["overview"]["attempt_count"],
        "projection_stale": model["projection_comparison"]["stale"],
    }


def build_dashboard_model(
    *,
    report_path: Path | None = None,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
    records_path: Path | None = None,
    attempts_path: Path | None = None,
    timing_summary_path: Path | None = None,
    validated_projection_path: Path | None = None,
    rejected_projection_path: Path | None = None,
    missing_evidence_path: Path | None = None,
    title: str = "Validation Statistics",
    eyebrow: str = "Milestone 4 Asset Validation",
) -> dict[str, Any]:
    """Return the compact data model embedded by the static dashboard."""

    selected_report_path = report_path or latest_canonical_report(reports_dir)
    report = _load_json_required(selected_report_path)
    records_checkpoint_path = records_path or data_dir / "runs" / "checkpoints" / "records.jsonl"
    attempts_checkpoint_path = attempts_path or data_dir / "runs" / "checkpoints" / "attempts.jsonl"
    timing_path = timing_summary_path or data_dir / "runs" / "timing_summary.yml"

    records = _clean_records(report.get("records") or [])
    attempts_source = "report"
    attempts = _clean_attempts(report.get("attempts") or [])
    if not attempts and attempts_checkpoint_path.is_file():
        attempts = _clean_attempts(_load_jsonl(attempts_checkpoint_path))
        attempts_source = "checkpoint"

    timing_source = "report"
    timing_summary = report.get("timing_summary") or {}
    if not timing_summary and timing_path.is_file():
        timing_summary = _load_yaml(timing_path) or {}
        timing_source = "file"

    checkpoint_records = _load_jsonl(records_checkpoint_path) if records_checkpoint_path.is_file() else []
    checkpoint_attempts = _load_jsonl(attempts_checkpoint_path) if attempts_checkpoint_path.is_file() else []
    projections = _projection_summaries(
        data_dir,
        validated_path=validated_projection_path,
        rejected_path=rejected_projection_path,
        missing_path=missing_evidence_path,
    )
    overview = _overview(records, attempts, report.get("summary") or {})

    model = {
        "schema_version": 1,
        "title": title,
        "eyebrow": eyebrow,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {
            "report_path": _path_string(selected_report_path),
            "records_checkpoint_path": _path_string(records_checkpoint_path) if records_checkpoint_path.is_file() else None,
            "attempts_checkpoint_path": _path_string(attempts_checkpoint_path) if attempts_checkpoint_path.is_file() else None,
            "timing_summary_path": _path_string(timing_path) if timing_path.is_file() else None,
            "records_source": "report",
            "attempts_source": attempts_source,
            "timing_source": timing_source,
        },
        "report": _report_identity(report),
        "overview": overview,
        "by_use": _by_use(records, attempts),
        "failure_diagnostics": _failure_diagnostics(records, attempts),
        "probe_behavior": _probe_behavior(attempts),
        "timing": _timing_model(timing_summary, attempts),
        "records": records,
        "worst_assets": _worst_assets(records),
        "projections": projections,
        "projection_comparison": _projection_comparison(overview, projections),
        "checkpoint_comparison": {
            "records_count": len(checkpoint_records),
            "attempts_count": len(checkpoint_attempts),
            "records_match_report": len(checkpoint_records) == len(records) if checkpoint_records else None,
            "attempts_match_report": len(checkpoint_attempts) == len(attempts) if checkpoint_attempts else None,
        },
    }
    return model


def latest_canonical_report(reports_dir: Path = DEFAULT_REPORTS_DIR) -> Path:
    reports = sorted(reports_dir.glob("canonical_skill_*.json"))
    if not reports:
        raise FileNotFoundError(f"no canonical_skill_*.json reports found under {reports_dir}")
    return max(reports, key=lambda path: (path.stat().st_mtime, path.name))


def render_dashboard_html(model: dict[str, Any]) -> str:
    data = json.dumps(model, ensure_ascii=True, sort_keys=True, separators=(",", ":")).replace("</", "<\\/")
    title = str(model.get("title") or "Validation Statistics")
    eyebrow = html.escape(str(model.get("eyebrow") or "Milestone 4 Asset Validation"))
    subtitle = html.escape(str(model.get("report", {}).get("evidence_id") or "canonical skill report"))
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{title}</title>",
            "<style>",
            _DASHBOARD_CSS,
            "</style>",
            "</head>",
            "<body>",
            '<main class="shell">',
            '<section class="hero">',
            '<div class="hero-copy">',
            f'<p class="eyebrow">{eyebrow}</p>',
            f"<h1>{title}</h1>",
            f'<p class="lede">{subtitle}</p>',
            "</div>",
            '<div id="stale-banner" class="banner"></div>',
            "</section>",
            '<section id="overview" class="section"></section>',
            '<section id="charts" class="section"></section>',
            '<section id="diagnostics" class="section"></section>',
            '<section id="timing" class="section"></section>',
            '<section id="projection" class="section"></section>',
            '<section id="records-section" class="section"></section>',
            "</main>",
            f'<script id="dashboard-data" type="application/json">{data}</script>',
            "<script>",
            _DASHBOARD_JS,
            "</script>",
            "</body>",
            "</html>",
        ]
    )


def _report_identity(report: dict[str, Any]) -> dict[str, Any]:
    runner = report.get("runner") or {}
    thresholds = runner.get("thresholds") or _thresholds_from_records(report.get("records") or [])
    seeds = runner.get("seeds") or _seeds_from_records(report.get("records") or [])
    return {
        "schema_version": report.get("schema_version"),
        "kind": report.get("kind"),
        "scope": report.get("scope"),
        "stage": report.get("stage"),
        "source_kind": report.get("source_kind"),
        "evidence_id": report.get("evidence_id"),
        "report_hash": report.get("report_hash"),
        "source_probe_plan_path": report.get("source_probe_plan_path"),
        "source_probe_plan_hash": report.get("source_probe_plan_hash"),
        "source_asset_catalog_path": report.get("source_asset_catalog_path"),
        "source_asset_catalog_sha256": report.get("source_asset_catalog_sha256"),
        "runner": {
            "task_config": runner.get("task_config"),
            "seeds": seeds,
            "thresholds": thresholds,
            "batch_size": runner.get("batch_size"),
            "reuse_task_env": runner.get("reuse_task_env"),
            "parallel_workers": runner.get("parallel_workers"),
            "cuda_devices": runner.get("cuda_devices") or [],
            "started_at": runner.get("started_at"),
            "finished_at": runner.get("finished_at"),
        },
    }


def _overview(records: list[dict[str, Any]], attempts: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    record_count = len(records) or int(summary.get("record_count") or 0)
    validated_count = sum(1 for row in records if row.get("pass") is True)
    rejected_count = sum(1 for row in records if row.get("pass") is False)
    if not records:
        validated_count = int(summary.get("validated_count") or 0)
        rejected_count = int(summary.get("rejected_count") or 0)
    attempt_count = len(attempts) or int(summary.get("attempt_count") or 0)
    success_count = sum(1 for row in attempts if row.get("success") is True)
    plan_success_count = sum(1 for row in attempts if row.get("plan_success") is True)
    return {
        "record_count": record_count,
        "attempt_count": attempt_count,
        "validated_count": validated_count,
        "rejected_count": rejected_count,
        "validated_rate": _rate(validated_count, record_count),
        "rejected_rate": _rate(rejected_count, record_count),
        "attempt_success_count": success_count,
        "attempt_success_rate": _rate(success_count, attempt_count),
        "plan_success_count": plan_success_count,
        "plan_success_rate": _rate(plan_success_count, attempt_count),
    }


def _by_use(records: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attempt_counts: dict[str, Counter] = defaultdict(Counter)
    for attempt in attempts:
        use = str(attempt.get("use") or "unknown")
        attempt_counts[use]["attempts"] += 1
        if attempt.get("success") is True:
            attempt_counts[use]["success"] += 1
        if attempt.get("plan_success") is True:
            attempt_counts[use]["plan_success"] += 1
        reason = str(attempt.get("failure_reason") or "unknown")
        attempt_counts[use][f"reason:{reason}"] += 1

    rows: list[dict[str, Any]] = []
    for use in sorted({str(row.get("use")) for row in records} | set(attempt_counts)):
        use_records = [row for row in records if str(row.get("use")) == use]
        passed = sum(1 for row in use_records if row.get("pass") is True)
        failed = sum(1 for row in use_records if row.get("pass") is False)
        attempts_n = int(attempt_counts[use]["attempts"])
        rows.append(
            {
                "use": use,
                "records": len(use_records),
                "validated": passed,
                "rejected": failed,
                "pass_rate": _rate(passed, len(use_records)),
                "attempts": attempts_n,
                "attempt_success": int(attempt_counts[use]["success"]),
                "attempt_success_rate": _rate(int(attempt_counts[use]["success"]), attempts_n),
                "plan_success": int(attempt_counts[use]["plan_success"]),
                "plan_success_rate": _rate(int(attempt_counts[use]["plan_success"]), attempts_n),
            }
        )
    return rows


def _failure_diagnostics(records: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    record_verdicts = Counter(str(row.get("verdict") or "unknown") for row in records)
    attempt_reasons = Counter(str(row.get("failure_reason") or "unknown") for row in attempts)
    fail_reason_counts: Counter[str] = Counter()
    for row in records:
        for reason, count in (row.get("fail_reasons") or {}).items():
            fail_reason_counts[str(reason)] += int(count)
    return {
        "record_verdicts": _counter_rows(record_verdicts),
        "attempt_failure_reasons": _counter_rows(attempt_reasons),
        "record_fail_reasons": _counter_rows(fail_reason_counts),
    }


def _probe_behavior(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    cuda = Counter()
    root_sources = Counter()
    final_planner_sources = Counter()
    fallback_outcomes: dict[str, Counter] = {
        "fallback_used": Counter(),
        "fallback_not_used": Counter(),
        "planner_fallback_used": Counter(),
        "planner_fallback_not_used": Counter(),
    }
    root_fallback_count = 0
    planner_fallback_count = 0
    fallback_exhausted_count = 0
    warmup_count = 0
    for attempt in attempts:
        cuda[str(attempt.get("cuda_device") or "unspecified")] += 1
        if attempt.get("warmup"):
            warmup_count += 1
        diagnostics = attempt.get("diagnostics") or {}
        root_pose = diagnostics.get("root_pose") or {}
        root_source = str(root_pose.get("source") or "unknown")
        root_sources[root_source] += 1
        planner_attempts = diagnostics.get("planner_pose_attempts") or []
        if planner_attempts:
            final_planner_sources[str((planner_attempts[-1] or {}).get("source") or "unknown")] += 1
        root_pose_attempts = diagnostics.get("root_pose_attempts") or []
        root_fallback = len(root_pose_attempts) > 1 or "fallback:" in root_source
        planner_fallback = len(planner_attempts) > 1
        if root_fallback:
            root_fallback_count += 1
        if planner_fallback:
            planner_fallback_count += 1
        if diagnostics.get("fallback_exhausted"):
            fallback_exhausted_count += 1
        _outcome_bucket(fallback_outcomes["fallback_used" if root_fallback else "fallback_not_used"], attempt)
        _outcome_bucket(
            fallback_outcomes["planner_fallback_used" if planner_fallback else "planner_fallback_not_used"],
            attempt,
        )
    return {
        "cuda_distribution": _counter_rows(cuda),
        "root_pose_sources": _counter_rows(root_sources, limit=20),
        "final_planner_pose_sources": _counter_rows(final_planner_sources, limit=20),
        "root_fallback_count": root_fallback_count,
        "planner_fallback_count": planner_fallback_count,
        "fallback_exhausted_count": fallback_exhausted_count,
        "warmup_count": warmup_count,
        "fallback_outcomes": {key: dict(counter) for key, counter in fallback_outcomes.items()},
    }


def _timing_model(timing_summary: dict[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    keys = timing_summary.get("timing_keys") or {}
    selected_keys = [
        "total_worker_sec",
        "setup_demo_sec",
        "setup_scene_sec",
        "load_robot_sec",
        "planner_reset_sec",
        "play_once_sec",
        "check_stable_sec",
        "check_success_sec",
    ]
    rows = []
    for key in selected_keys:
        stats = keys.get(key)
        if not isinstance(stats, dict):
            continue
        row = {"key": key}
        row.update({name: stats.get(name) for name in ("count", "p50", "p90", "max", "mean")})
        rows.append(row)
    return {
        "summary": {
            "attempt_count": timing_summary.get("attempt_count"),
            "record_count": timing_summary.get("record_count"),
            "batch_size": timing_summary.get("batch_size"),
            "reuse_task_env": timing_summary.get("reuse_task_env"),
            "parallel_workers": timing_summary.get("parallel_workers"),
            "cuda_devices": timing_summary.get("cuda_devices") or [],
        },
        "timing_keys": rows,
        "by_use": timing_summary.get("by_use") or {},
        "slowest_attempts": timing_summary.get("slowest_attempts") or _slowest_attempts(attempts),
    }


def _projection_summaries(
    data_dir: Path,
    *,
    validated_path: Path | None = None,
    rejected_path: Path | None = None,
    missing_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "validated": _projection_summary(validated_path or data_dir / "validated_assets.yml"),
        "rejected": _projection_summary(rejected_path or data_dir / "rejected_assets.yml"),
        "missing_evidence": _missing_summary(missing_path or data_dir / "missing_evidence_report.yml"),
    }


def _projection_summary(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    if not data:
        return {"path": _path_string(path), "present": False}
    records = data.get("records") or []
    by_use = Counter(str(row.get("use") or "unknown") for row in records)
    by_stage = Counter(str(row.get("stage") or "unknown") for row in records)
    executable_records = [row for row in records if row.get("stage") == "canonical_skill"]
    return {
        "path": _path_string(path),
        "present": True,
        "status": data.get("status"),
        "source_ledger_hash": data.get("source_ledger_hash"),
        "projection_hash": data.get("projection_hash"),
        "record_count": len(records),
        "canonical_executable_count": len(executable_records),
        "by_use": dict(sorted(by_use.items())),
        "by_stage": dict(sorted(by_stage.items())),
        "summary": data.get("summary") or {},
    }


def _missing_summary(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    if not data:
        return {"path": _path_string(path), "present": False}
    missing = data.get("missing_canonical_skill_evidence") or []
    return {
        "path": _path_string(path),
        "present": True,
        "source_ledger_hash": data.get("source_ledger_hash"),
        "report_hash": data.get("report_hash"),
        "missing_canonical_skill_evidence_count": len(missing),
        "deferred_pair_candidate_count": (data.get("summary") or {}).get("deferred_pair_candidate_count", 0),
        "summary": data.get("summary") or {},
    }


def _projection_comparison(overview: dict[str, Any], projections: dict[str, Any]) -> dict[str, Any]:
    projection_validated = int((projections.get("validated") or {}).get("canonical_executable_count") or 0)
    projection_rejected = int((projections.get("rejected") or {}).get("canonical_executable_count") or 0)
    canonical_validated = int(overview.get("validated_count") or 0)
    canonical_rejected = int(overview.get("rejected_count") or 0)
    canonical_total = canonical_validated + canonical_rejected
    projection_total = projection_validated + projection_rejected
    missing_evidence_count = int(
        (projections.get("missing_evidence") or {}).get("missing_canonical_skill_evidence_count") or 0
    )
    comparable = projection_total == canonical_total
    stale = missing_evidence_count > 0 or (
        comparable and (projection_validated != canonical_validated or projection_rejected != canonical_rejected)
    )
    reason = "projection counts match canonical evidence"
    if missing_evidence_count > 0:
        reason = "missing canonical skill evidence remains"
    elif not comparable:
        reason = "latest report is a partial canonical retry; projections are full ledger totals"
    elif stale:
        reason = "evidence collected but not frozen into projections"
    return {
        "stale": stale,
        "comparable": comparable,
        "reason": reason,
        "canonical_validated_count": canonical_validated,
        "canonical_rejected_count": canonical_rejected,
        "canonical_total_count": canonical_total,
        "projection_canonical_validated_count": projection_validated,
        "projection_canonical_rejected_count": projection_rejected,
        "projection_canonical_total_count": projection_total,
        "missing_evidence_count": missing_evidence_count,
    }


def _worst_assets(records: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row.get("asset")), str(row.get("model_id")))].append(row)
    rows: list[dict[str, Any]] = []
    for (asset, model_id), asset_records in grouped.items():
        failed = [row for row in asset_records if row.get("pass") is False]
        if not failed:
            continue
        verdicts = Counter(str(row.get("verdict") or "unknown") for row in failed)
        rows.append(
            {
                "asset": asset,
                "model_id": model_id,
                "records": len(asset_records),
                "failed": len(failed),
                "failure_rate": _rate(len(failed), len(asset_records)),
                "top_verdict": verdicts.most_common(1)[0][0],
                "uses": ", ".join(sorted({str(row.get("use")) for row in failed})),
            }
        )
    return sorted(rows, key=lambda row: (-row["failure_rate"], -row["failed"], row["asset"], row["model_id"]))[:limit]


def _clean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in records:
        rows.append(
            {
                "asset": row.get("asset"),
                "model_id": str(row.get("model_id")),
                "use": row.get("use"),
                "pass": row.get("pass"),
                "verdict": row.get("verdict"),
                "min_rate": row.get("min_rate"),
                "attempts": row.get("attempts"),
                "plan_success": row.get("plan_success"),
                "success": row.get("success"),
                "threshold": row.get("threshold"),
                "fail_reasons": row.get("fail_reasons") or {},
            }
        )
    return rows


def _clean_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in attempts:
        rows.append(
            {
                "attempt_id": row.get("attempt_id"),
                "job_id": row.get("job_id"),
                "asset": row.get("asset"),
                "model_id": str(row.get("model_id")),
                "use": row.get("use"),
                "seed": row.get("seed"),
                "success": row.get("success"),
                "plan_success": row.get("plan_success"),
                "failure_reason": row.get("failure_reason"),
                "duration_sec": row.get("duration_sec"),
                "cuda_device": row.get("cuda_device"),
                "warmup": row.get("warmup"),
                "diagnostics": _clean_diagnostics(row.get("diagnostics") or {}),
                "timings": row.get("timings") or {},
            }
        )
    return rows


def _clean_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "root_pose": diagnostics.get("root_pose") or {},
        "root_pose_attempts": diagnostics.get("root_pose_attempts") or [],
        "planner_pose_attempts": diagnostics.get("planner_pose_attempts") or [],
        "fallback_exhausted": diagnostics.get("fallback_exhausted"),
    }


def _slowest_attempts(attempts: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    def duration(row: dict[str, Any]) -> float:
        try:
            return float(row.get("duration_sec") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return [
        {
            "asset": row.get("asset"),
            "model_id": row.get("model_id"),
            "use": row.get("use"),
            "seed": row.get("seed"),
            "success": row.get("success"),
            "failure_reason": row.get("failure_reason"),
            "duration_sec": round(duration(row), 6),
        }
        for row in sorted(attempts, key=duration, reverse=True)[:limit]
    ]


def _thresholds_from_records(records: list[dict[str, Any]]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for row in records:
        use = row.get("use")
        threshold = row.get("threshold")
        if use is None or threshold is None:
            continue
        try:
            thresholds[str(use)] = float(threshold)
        except (TypeError, ValueError):
            continue
    return thresholds


def _seeds_from_records(records: list[dict[str, Any]]) -> list[int]:
    seeds: set[int] = set()
    for row in records:
        for seed in row.get("seeds") or []:
            try:
                seeds.add(int(seed))
            except (TypeError, ValueError):
                continue
    return sorted(seeds)


def _outcome_bucket(counter: Counter, attempt: dict[str, Any]) -> None:
    counter["attempts"] += 1
    if attempt.get("plan_success") is True:
        counter["plan_success"] += 1
    if attempt.get("success") is True:
        counter["success"] += 1


def _counter_rows(counter: Counter, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [{"name": name, "count": count} for name, count in counter.most_common(limit)]
    return rows


def _rate(part: int, whole: int) -> float:
    if not whole:
        return 0.0
    return round(part / whole, 6)


def _load_json_required(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                rows.append(data)
    return rows


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else None


def _path_string(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path).replace("\\", "/")


_DASHBOARD_CSS = r"""
:root {
  color-scheme: light;
  --paper: #f7f3ea;
  --surface: #fffaf2;
  --surface-strong: #ffffff;
  --ink: #211f1b;
  --muted: #6e675c;
  --line: #ded4c6;
  --line-strong: #c9bbaa;
  --clay: #c86f4a;
  --clay-dark: #8f452e;
  --green: #4f7f5b;
  --red: #b8574f;
  --gold: #a8792c;
  --blue: #4d6f85;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--sans);
  line-height: 1.45;
}
.shell {
  width: min(1440px, calc(100vw - 40px));
  margin: 0 auto;
  padding: 28px 0 48px;
}
.hero {
  min-height: 210px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 460px);
  gap: 28px;
  align-items: end;
  border-bottom: 1px solid var(--line-strong);
  padding-bottom: 26px;
}
.eyebrow {
  margin: 0 0 10px;
  color: var(--clay-dark);
  font: 700 12px/1 var(--mono);
  text-transform: uppercase;
  letter-spacing: .08em;
}
h1 {
  margin: 0;
  font-size: clamp(44px, 6vw, 86px);
  line-height: .95;
  letter-spacing: 0;
  font-weight: 740;
}
.lede {
  margin: 18px 0 0;
  color: var(--muted);
  font-family: var(--mono);
  overflow-wrap: anywhere;
}
.banner {
  border: 1px solid var(--line-strong);
  background: var(--surface);
  padding: 16px;
  border-radius: 8px;
  font-size: 14px;
}
.banner.warning {
  border-color: #d6a35e;
  background: #fff3dc;
}
.section {
  padding: 26px 0 0;
}
.section-head {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 14px;
}
h2 {
  margin: 0;
  font-size: 23px;
  letter-spacing: 0;
}
.hint {
  margin: 4px 0 0;
  color: var(--muted);
  font-size: 14px;
}
.grid {
  display: grid;
  gap: 12px;
}
.kpi-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}
.two-col {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.three-col {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}
.metric, .panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
}
.metric {
  padding: 14px 15px;
  min-height: 116px;
}
.metric-label {
  color: var(--muted);
  font: 700 11px/1.2 var(--mono);
  text-transform: uppercase;
  letter-spacing: .06em;
}
.metric-value {
  margin-top: 15px;
  font-size: 34px;
  line-height: 1;
  font-weight: 730;
}
.metric-detail {
  margin-top: 10px;
  color: var(--muted);
  font-size: 13px;
  overflow-wrap: anywhere;
}
.panel {
  padding: 16px;
  min-width: 0;
}
.panel h3 {
  margin: 0 0 14px;
  font-size: 15px;
}
.bars {
  display: grid;
  gap: 10px;
}
.bar-row {
  display: grid;
  grid-template-columns: minmax(90px, 170px) minmax(0, 1fr) 76px;
  gap: 10px;
  align-items: center;
  font-size: 13px;
}
.bar-track {
  height: 12px;
  border: 1px solid var(--line);
  background: #f0e7da;
  border-radius: 999px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  background: var(--clay);
}
.bar-fill.green { background: var(--green); }
.bar-fill.red { background: var(--red); }
.bar-fill.gold { background: var(--gold); }
.bar-fill.blue { background: var(--blue); }
.controls {
  display: grid;
  grid-template-columns: minmax(220px, 1fr) 180px 180px;
  gap: 10px;
  margin-bottom: 12px;
}
input, select {
  width: 100%;
  height: 38px;
  border: 1px solid var(--line-strong);
  border-radius: 6px;
  background: var(--surface-strong);
  color: var(--ink);
  padding: 0 10px;
  font: 13px/1 var(--sans);
}
.table-wrap {
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: auto;
  background: var(--surface);
}
table {
  width: 100%;
  border-collapse: collapse;
  min-width: 850px;
  font-size: 13px;
}
th, td {
  padding: 9px 10px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}
th {
  position: sticky;
  top: 0;
  background: #f2eadf;
  color: #4d463e;
  font: 700 11px/1.2 var(--mono);
  text-transform: uppercase;
  letter-spacing: .05em;
}
tbody tr:hover {
  background: #fff7eb;
}
.pill {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  border: 1px solid var(--line-strong);
  border-radius: 999px;
  padding: 2px 8px;
  font: 700 11px/1 var(--mono);
}
.pill.ok { color: #315f41; background: #e8f2e8; border-color: #bed5c0; }
.pill.bad { color: #8d342f; background: #f9e8e4; border-color: #e0b9b2; }
.mono {
  font-family: var(--mono);
  font-size: 12px;
}
.small {
  color: var(--muted);
  font-size: 12px;
}
@media (max-width: 980px) {
  .hero, .kpi-grid, .two-col, .three-col, .controls {
    grid-template-columns: 1fr;
  }
  .shell {
    width: min(100vw - 24px, 720px);
  }
}
"""


_DASHBOARD_JS = r"""
const dashboard = JSON.parse(document.getElementById("dashboard-data").textContent);

function pct(value) {
  return `${Math.round((Number(value || 0)) * 1000) / 10}%`;
}

function num(value) {
  return Number(value || 0).toLocaleString();
}

function sec(value) {
  if (value === null || value === undefined) return "n/a";
  return `${Number(value).toFixed(Number(value) >= 10 ? 1 : 3)}s`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function fileName(path) {
  const value = String(path ?? "");
  return value.split(/[\\/]/).filter(Boolean).pop() || "missing";
}

function metric(label, value, detail) {
  return `<article class="metric"><div class="metric-label">${escapeHtml(label)}</div><div class="metric-value">${value}</div><div class="metric-detail">${escapeHtml(detail || "")}</div></article>`;
}

function sectionHead(title, hint) {
  return `<div class="section-head"><div><h2>${escapeHtml(title)}</h2><p class="hint">${escapeHtml(hint || "")}</p></div></div>`;
}

function barRows(rows, options = {}) {
  const total = options.total || Math.max(1, ...rows.map((row) => Number(row.count || row.value || 0)));
  const color = options.color || "clay";
  return `<div class="bars">${rows.map((row) => {
    const value = Number(row.count ?? row.value ?? 0);
    const width = Math.max(0, Math.min(100, (value / total) * 100));
    const label = row.name ?? row.use ?? row.key;
    const suffix = row.suffix ?? num(value);
    return `<div class="bar-row"><div class="mono">${escapeHtml(label)}</div><div class="bar-track"><div class="bar-fill ${color}" style="width:${width}%"></div></div><div class="small">${escapeHtml(suffix)}</div></div>`;
  }).join("")}</div>`;
}

function renderBanner() {
  const comparison = dashboard.projection_comparison;
  const el = document.getElementById("stale-banner");
  el.className = `banner${comparison.stale ? " warning" : ""}`;
  const report = dashboard.report;
  const runner = report.runner || {};
  el.innerHTML = `
    <div class="metric-label">${comparison.stale ? "Projection warning" : "Projection status"}</div>
    <p>${escapeHtml(comparison.reason)}</p>
    <div class="small">Report ${escapeHtml(report.evidence_id)} | source ${escapeHtml(report.source_kind)} | GPUs ${(runner.cuda_devices || []).map(escapeHtml).join(", ") || "n/a"} | workers ${escapeHtml(runner.parallel_workers ?? "n/a")}</div>
  `;
}

function renderOverview() {
  const o = dashboard.overview;
  const r = dashboard.report;
  const runner = r.runner || {};
  document.getElementById("overview").innerHTML = `
    ${sectionHead("Overview", "Canonical skill report is the primary evidence source.")}
    <div class="grid kpi-grid">
      ${metric("Canonical validated", `${num(o.validated_count)} / ${num(o.record_count)}`, pct(o.validated_rate))}
      ${metric("Canonical rejected", `${num(o.rejected_count)} / ${num(o.record_count)}`, pct(o.rejected_rate))}
      ${metric("Attempt success", `${num(o.attempt_success_count)} / ${num(o.attempt_count)}`, pct(o.attempt_success_rate))}
      ${metric("Plan success", `${num(o.plan_success_count)} / ${num(o.attempt_count)}`, pct(o.plan_success_rate))}
    </div>
    <div class="grid three-col" style="margin-top:12px">
      ${metric("Evidence id", escapeHtml(r.evidence_id || "n/a"), r.report_hash || "")}
      ${metric("Seeds", escapeHtml((runner.seeds || []).join(", ") || "n/a"), `thresholds: ${JSON.stringify(runner.thresholds || {})}`)}
      ${metric("Run", escapeHtml(runner.finished_at || "n/a"), `${runner.batch_size || "n/a"} batch size, reuse=${runner.reuse_task_env}`)}
    </div>
  `;
}

function renderCharts() {
  const byUse = dashboard.by_use || [];
  const passRows = byUse.map((row) => ({ name: row.use, value: row.validated, suffix: `${num(row.validated)} / ${num(row.records)} (${pct(row.pass_rate)})` }));
  const attemptRows = byUse.map((row) => ({ name: row.use, value: row.attempt_success, suffix: `${num(row.attempt_success)} / ${num(row.attempts)} (${pct(row.attempt_success_rate)})` }));
  const failures = (dashboard.failure_diagnostics.attempt_failure_reasons || []).filter((row) => row.name !== "ok");
  document.getElementById("charts").innerHTML = `
    ${sectionHead("Use Breakdown", "Pass rates and failure reasons by canonical skill.")}
    <div class="grid three-col">
      <article class="panel"><h3>Validated by use</h3>${barRows(passRows, { color: "green" })}</article>
      <article class="panel"><h3>Attempt success by use</h3>${barRows(attemptRows, { color: "blue" })}</article>
      <article class="panel"><h3>Attempt failure reasons</h3>${barRows(failures, { color: "red" })}</article>
    </div>
  `;
}

function renderDiagnostics() {
  const behavior = dashboard.probe_behavior;
  const fallback = behavior.fallback_outcomes || {};
  const fallbackRows = Object.entries(fallback).map(([name, row]) => ({
    name,
    count: row.attempts || 0,
    suffix: `${num(row.success || 0)} success / ${num(row.attempts || 0)}`,
  }));
  document.getElementById("diagnostics").innerHTML = `
    ${sectionHead("Probe Behavior", "Seed-level execution diagnostics from the canonical report attempts.")}
    <div class="grid three-col">
      <article class="panel"><h3>GPU distribution</h3>${barRows(behavior.cuda_distribution || [], { color: "blue" })}</article>
      <article class="panel"><h3>Root pose sources</h3>${barRows((behavior.root_pose_sources || []).slice(0, 8), { color: "gold" })}</article>
      <article class="panel"><h3>Fallback outcomes</h3>${barRows(fallbackRows, { color: "clay" })}</article>
    </div>
    <div class="grid three-col" style="margin-top:12px">
      ${metric("Root fallback attempts", num(behavior.root_fallback_count), "root pose alternatives used")}
      ${metric("Planner fallback attempts", num(behavior.planner_fallback_count), "planner retried with another pose")}
      ${metric("Fallback exhausted", num(behavior.fallback_exhausted_count), "no viable fallback remained")}
    </div>
  `;
}

function renderTiming() {
  const rows = dashboard.timing.timing_keys || [];
  const slowest = dashboard.timing.slowest_attempts || [];
  document.getElementById("timing").innerHTML = `
    ${sectionHead("Timing", "Probe runtime distribution from report timing summary.")}
    <div class="grid two-col">
      <article class="panel"><h3>Phase p90</h3>${barRows(rows.map((row) => ({ name: row.key, value: row.p90, suffix: sec(row.p90) })), { color: "gold" })}</article>
      <article class="panel"><h3>Slowest attempts</h3>${simpleTable(slowest, ["asset", "model_id", "use", "seed", "failure_reason", "duration_sec"], { duration_sec: sec })}</article>
    </div>
  `;
}

function renderProjection() {
  const c = dashboard.projection_comparison;
  const p = dashboard.projections;
  const rows = [
    { name: "canonical validated", count: c.canonical_validated_count },
    { name: "projection canonical validated", count: c.projection_canonical_validated_count },
    { name: "canonical rejected", count: c.canonical_rejected_count },
    { name: "projection canonical rejected", count: c.projection_canonical_rejected_count },
    { name: "missing evidence projection", count: c.missing_evidence_count },
  ];
  document.getElementById("projection").innerHTML = `
    ${sectionHead("Projection Comparison", "Current projection files are shown separately from the canonical evidence report.")}
    <div class="grid two-col">
      <article class="panel"><h3>Evidence vs projection</h3>${barRows(rows, { color: c.stale ? "red" : "green" })}</article>
      <article class="panel"><h3>Projection files</h3>${simpleTable([
        { file: fileName(p.validated.path), present: p.validated.present, records: p.validated.record_count, canonical: p.validated.canonical_executable_count },
        { file: fileName(p.rejected.path), present: p.rejected.present, records: p.rejected.record_count, canonical: p.rejected.canonical_executable_count },
        { file: fileName(p.missing_evidence.path), present: p.missing_evidence.present, records: p.missing_evidence.missing_canonical_skill_evidence_count, canonical: "" },
      ], ["file", "present", "records", "canonical"])}</article>
    </div>
  `;
}

function simpleTable(rows, columns, formatters = {}) {
  return `<div class="table-wrap"><table><thead><tr>${columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${columns.map((col) => {
    const formatter = formatters[col] || ((x) => x);
    return `<td>${escapeHtml(formatter(row[col]))}</td>`;
  }).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function renderRecords() {
  const records = dashboard.records || [];
  const uses = [...new Set(records.map((row) => row.use))].sort();
  const verdicts = [...new Set(records.map((row) => row.verdict))].sort();
  document.getElementById("records-section").innerHTML = `
    ${sectionHead("Canonical Records", "Asset/model/use rows from the canonical report.")}
    <div class="controls">
      <input id="record-search" placeholder="Search asset, model, verdict">
      <select id="record-use"><option value="">All uses</option>${uses.map((use) => `<option value="${escapeHtml(use)}">${escapeHtml(use)}</option>`).join("")}</select>
      <select id="record-verdict"><option value="">All verdicts</option>${verdicts.map((verdict) => `<option value="${escapeHtml(verdict)}">${escapeHtml(verdict)}</option>`).join("")}</select>
    </div>
    <div id="record-count" class="small" style="margin-bottom:8px"></div>
    <div class="table-wrap"><table><thead><tr><th>asset</th><th>model</th><th>use</th><th>status</th><th>verdict</th><th>min rate</th><th>plan</th><th>success</th><th>fail reasons</th></tr></thead><tbody id="records-body"></tbody></table></div>
  `;
  const search = document.getElementById("record-search");
  const use = document.getElementById("record-use");
  const verdict = document.getElementById("record-verdict");
  [search, use, verdict].forEach((el) => el.addEventListener("input", updateRecords));
  updateRecords();
}

function updateRecords() {
  const records = dashboard.records || [];
  const query = document.getElementById("record-search").value.trim().toLowerCase();
  const use = document.getElementById("record-use").value;
  const verdict = document.getElementById("record-verdict").value;
  const filtered = records.filter((row) => {
    if (use && row.use !== use) return false;
    if (verdict && row.verdict !== verdict) return false;
    const text = `${row.asset} ${row.model_id} ${row.use} ${row.verdict}`.toLowerCase();
    return !query || text.includes(query);
  });
  document.getElementById("record-count").textContent = `${num(filtered.length)} shown of ${num(records.length)} records`;
  document.getElementById("records-body").innerHTML = filtered.slice(0, 500).map((row) => `
    <tr>
      <td class="mono">${escapeHtml(row.asset)}</td>
      <td class="mono">${escapeHtml(row.model_id)}</td>
      <td>${escapeHtml(row.use)}</td>
      <td><span class="pill ${row.pass ? "ok" : "bad"}">${row.pass ? "validated" : "rejected"}</span></td>
      <td>${escapeHtml(row.verdict)}</td>
      <td>${pct(row.min_rate)}</td>
      <td>${num(row.plan_success)} / ${num(row.attempts)}</td>
      <td>${num(row.success)} / ${num(row.attempts)}</td>
      <td class="small">${escapeHtml(JSON.stringify(row.fail_reasons || {}))}</td>
    </tr>
  `).join("");
}

renderBanner();
renderOverview();
renderCharts();
renderDiagnostics();
renderTiming();
renderProjection();
renderRecords();
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a static asset validation statistics dashboard.")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--attempts", type=Path)
    parser.add_argument("--timing-summary", type=Path)
    parser.add_argument("--validated-projection", type=Path)
    parser.add_argument("--rejected-projection", type=Path)
    parser.add_argument("--missing-evidence", type=Path)
    parser.add_argument("--title", default="Validation Statistics")
    parser.add_argument("--eyebrow", default="Milestone 4 Asset Validation")
    args = parser.parse_args(argv)
    result = write_validation_dashboard(
        report_path=args.report,
        reports_dir=args.reports_dir,
        data_dir=args.data_dir,
        output_path=args.output,
        records_path=args.records,
        attempts_path=args.attempts,
        timing_summary_path=args.timing_summary,
        validated_projection_path=args.validated_projection,
        rejected_projection_path=args.rejected_projection,
        missing_evidence_path=args.missing_evidence,
        title=args.title,
        eyebrow=args.eyebrow,
    )
    stale = " stale projections" if result["projection_stale"] else ""
    print(
        f"dashboard: wrote {result['output_path']} from {result['report_path']} "
        f"({result['record_count']} records / {result['attempt_count']} attempts){stale}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
