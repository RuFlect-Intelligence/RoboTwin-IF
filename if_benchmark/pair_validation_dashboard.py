"""Self-contained HTML dashboard for Milestone 5 pair validation."""

from __future__ import annotations

import html
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DATA_DIR = Path("if_benchmark/data/pair_validation")
DEFAULT_REPORTS_DIR = DEFAULT_DATA_DIR / "reports"
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_DIR / "pair_validation_dashboard.html"
DEFAULT_MANIFEST_PATH = DEFAULT_DATA_DIR / "m5_dashboard_manifest.yml"


def write_pair_validation_dashboard(
    *,
    report_path: Path | None = None,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    attempts_path: Path | None = None,
    progress_path: Path | None = None,
    audit_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    model = build_pair_validation_dashboard_model(
        report_path=report_path,
        reports_dir=reports_dir,
        data_dir=data_dir,
        attempts_path=attempts_path,
        progress_path=progress_path,
        audit_path=audit_path,
        manifest_path=manifest_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_pair_validation_dashboard_html(model), encoding="utf-8")
    frozen_pair_count = sum(int(value) for value in model["frozen_status_counts"].values())
    return {
        "output_path": _path_string(output_path),
        "report_path": model["sources"]["manifest_path"] or model["sources"]["report_path"],
        "record_count": frozen_pair_count or model["overview"]["record_count"],
        "attempt_count": model["evidence_attempt_count"] or model["overview"]["attempt_count"],
        "skill_pair_count": model["overview"]["validated_skill_pair_count"],
    }


def build_pair_validation_dashboard_model(
    *,
    report_path: Path | None = None,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
    attempts_path: Path | None = None,
    progress_path: Path | None = None,
    audit_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    selected_manifest_path = manifest_path
    if selected_manifest_path is None:
        default_manifest = data_dir / DEFAULT_MANIFEST_PATH.name
        selected_manifest_path = default_manifest if default_manifest.is_file() else None
    manifest = _load_yaml(selected_manifest_path) if selected_manifest_path else {}

    selected_attempts_path = attempts_path
    if selected_attempts_path is None and not manifest:
        selected_attempts_path = data_dir / "runs" / "canonical_skill" / "checkpoints" / "attempts.jsonl"
    metadata_path = (
        selected_attempts_path.parent / "run_metadata.json"
        if selected_attempts_path
        else None
    )
    run_metadata = _load_json(metadata_path) if metadata_path and metadata_path.is_file() else {}
    selected_report_path = report_path
    if selected_report_path is None and not manifest:
        selected_report_path = latest_skill_report(
            reports_dir,
            expected_plan_hash=run_metadata.get("source_probe_plan_hash"),
            expected_policy_hash=run_metadata.get("skill_probe_policy_hash"),
        )
    report = _load_json(selected_report_path) if selected_report_path else {}
    records = list(report.get("records") or [])

    attempts = (
        _load_jsonl(selected_attempts_path)
        if selected_attempts_path and selected_attempts_path.is_file()
        else []
    )
    selected_progress_path = progress_path
    if selected_progress_path is None and not manifest:
        selected_progress_path = data_dir / "runs" / "canonical_skill" / "progress.json"
    progress = (
        _load_json(selected_progress_path)
        if selected_progress_path and selected_progress_path.is_file()
        else {}
    )
    selected_audit_path = audit_path
    if selected_audit_path is None and not manifest:
        selected_audit_path = data_dir / "runs" / "canonical_skill" / "final_audit.yml"
    audit = (
        _load_yaml(selected_audit_path)
        if selected_audit_path and selected_audit_path.is_file()
        else {}
    )

    terminal = _load_yaml(data_dir / "validated_terminal_pairs.yml")
    skill = _load_yaml(data_dir / "validated_skill_pairs.yml")
    combined = _load_yaml(data_dir / "validated_pairs.yml")
    gaps = _load_yaml(data_dir / "pair_gap_report.yml")

    relation_rows = _relation_rows(records, attempts, terminal, skill)
    profile_rows = _profile_rows(records, attempts)
    failure_rows = [
        {"reason": reason, "count": count}
        for reason, count in Counter(
            str(row.get("failure_reason") or "unknown")
            for row in attempts
            if not bool(row.get("success"))
        ).most_common()
    ]

    report_summary = report.get("summary") or {}
    pass_count = sum(bool(row.get("pass")) for row in records)
    blocked_count = sum(str(row.get("outcome")) == "probe_run_blocked" for row in records)
    terminal_count = int((terminal.get("summary") or {}).get("pair_count", 0))
    skill_count = int((skill.get("summary") or {}).get("pair_count", 0))
    gap_count = int((gaps.get("summary") or {}).get("gap_count", 0))
    projection_consistency = _projection_consistency(terminal, skill, combined, gaps)
    review_evidence = _review_evidence_rows(data_dir)
    run_rows = _dashboard_run_rows(
        manifest,
        manifest_path=selected_manifest_path,
        data_dir=data_dir,
    )
    acceptance = dict(manifest.get("acceptance") or {})
    status_counts = {
        str(key): int(value)
        for key, value in (manifest.get("frozen_status_counts") or {}).items()
    }
    if manifest:
        relation_rows = _projection_relation_rows(terminal, skill)
        profile_rows = _articulated_projection_rows(
            terminal,
            skill,
            profiles_path=_resolve_dashboard_path(
                manifest.get("articulated_profiles"),
                manifest_path=selected_manifest_path,
                data_dir=data_dir,
            ),
        )
        failure_rows = _frozen_failure_rows(gaps)

    return {
        "schema_version": 2 if manifest else 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "title": str(manifest.get("title") or "Milestone 5 Pair Validation"),
        "sources": {
            "manifest_path": _path_string(selected_manifest_path),
            "report_path": _path_string(selected_report_path),
            "attempts_path": (
                _path_string(selected_attempts_path)
                if selected_attempts_path and selected_attempts_path.is_file()
                else None
            ),
            "run_metadata_path": (
                _path_string(metadata_path) if metadata_path and metadata_path.is_file() else None
            ),
            "progress_path": (
                _path_string(selected_progress_path)
                if selected_progress_path and selected_progress_path.is_file()
                else None
            ),
            "audit_path": (
                _path_string(selected_audit_path)
                if selected_audit_path and selected_audit_path.is_file()
                else None
            ),
        },
        "report": {
            "evidence_id": report.get("evidence_id"),
            "report_hash": report.get("report_hash"),
            "skill_probe_policy_hash": report.get("skill_probe_policy_hash"),
            "source_probe_plan_hash": report.get("source_probe_plan_hash"),
            "summary": report_summary,
        },
        "overview": {
            "record_count": len(records),
            "attempt_count": len(attempts),
            "attempt_success_count": sum(bool(row.get("success")) for row in attempts),
            "pass_record_count": pass_count,
            "failed_record_count": len(records) - pass_count - blocked_count,
            "blocked_record_count": blocked_count,
            "validated_terminal_pair_count": terminal_count,
            "validated_skill_pair_count": skill_count,
            "combined_pair_count": int((combined.get("summary") or {}).get("pair_count", 0)),
            "gap_count": gap_count,
        },
        "relations": relation_rows,
        "profiles": profile_rows,
        "failures": failure_rows,
        "progress": progress,
        "audit": audit.get("summary") or {},
        "acceptance": acceptance,
        "frozen_status_counts": status_counts,
        "runs": run_rows,
        "evidence_attempt_count": sum(int(row.get("attempt_count", 0)) for row in run_rows),
        "retry_queue": dict(manifest.get("retry_queue") or {}),
        "projection_consistency": projection_consistency,
        "review_evidence": review_evidence,
    }


def latest_skill_report(
    reports_dir: Path,
    *,
    expected_plan_hash: str | None = None,
    expected_policy_hash: str | None = None,
) -> Path | None:
    candidates = list(reports_dir.glob("canonical_skill_pair_*.json"))
    if not candidates:
        return None
    if expected_plan_hash or expected_policy_hash:
        matching: list[Path] = []
        for path in candidates:
            report = _load_json(path)
            if expected_plan_hash and report.get("source_probe_plan_hash") != expected_plan_hash:
                continue
            if expected_policy_hash and report.get("skill_probe_policy_hash") != expected_policy_hash:
                continue
            matching.append(path)
        if not matching:
            return None
        candidates = matching
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _dashboard_run_rows(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None,
    data_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in manifest.get("runs") or []:
        report_path = _resolve_dashboard_path(
            spec.get("report"),
            manifest_path=manifest_path,
            data_dir=data_dir,
        )
        progress_path = _resolve_dashboard_path(
            spec.get("progress"),
            manifest_path=manifest_path,
            data_dir=data_dir,
        )
        audit_path = _resolve_dashboard_path(
            spec.get("audit"),
            manifest_path=manifest_path,
            data_dir=data_dir,
        )
        report = _load_json(report_path) if report_path and report_path.is_file() else {}
        progress = _load_json(progress_path) if progress_path and progress_path.is_file() else {}
        audit = _load_yaml(audit_path) if audit_path and audit_path.is_file() else {}
        records = list(report.get("records") or [])
        report_summary = report.get("summary") or {}
        audit_summary = audit.get("summary") or {}
        outcomes = report_summary.get("outcomes") or {}

        pass_count = spec.get("pass_count", outcomes.get("pass"))
        if pass_count is None:
            pass_count = sum(
                bool(row.get("pass"))
                or str(row.get("verdict")) in {"validated", "validated_terminal", "validated_skill"}
                for row in records
            )
        record_count = int(spec.get("record_count", report_summary.get("record_count", len(records))))
        fail_count = spec.get("fail_count", outcomes.get("fail"))
        if fail_count is None:
            fail_count = max(0, record_count - int(pass_count))
        attempt_count = int(
            spec.get(
                "attempt_count",
                progress.get(
                    "total_attempt_count",
                    report_summary.get("attempt_count", 0),
                ),
            )
        )
        job_count = int(
            spec.get(
                "job_count",
                progress.get(
                    "total_job_count",
                    audit_summary.get("plan_job_count", record_count),
                ),
            )
        )
        infrastructure_count = int(
            audit_summary.get(
                "infrastructure_attempt_count",
                spec.get("infrastructure_attempt_count", 0),
            )
        )
        rows.append(
            {
                "id": str(spec.get("id") or report.get("evidence_id") or "run"),
                "label": str(spec.get("label") or spec.get("id") or "Run"),
                "stage": str(spec.get("stage") or report.get("stage") or "unknown"),
                "scope": str(spec.get("scope") or "all"),
                "status": str(spec.get("status") or progress.get("state") or "recorded"),
                "record_count": record_count,
                "pass_count": int(pass_count),
                "fail_count": int(fail_count),
                "job_count": job_count,
                "attempt_count": attempt_count,
                "infrastructure_attempt_count": infrastructure_count,
                "parallel_workers": int(progress.get("parallel_workers", spec.get("parallel_workers", 0))),
                "elapsed_seconds": round(float(progress.get("elapsed_seconds", 0.0)), 3),
                "evidence_id": report.get("evidence_id") or spec.get("evidence_id"),
                "report_path": _path_string(report_path),
                "report_href": _dashboard_href(report_path, data_dir),
                "audit_path": _path_string(audit_path),
                "audit_href": _dashboard_href(audit_path, data_dir),
                "note": str(spec.get("note") or ""),
            }
        )
    return rows


def _projection_relation_rows(
    terminal: dict[str, Any],
    skill: dict[str, Any],
) -> list[dict[str, Any]]:
    terminal_counts = (terminal.get("summary") or {}).get("pair_count_by_relation") or {}
    skill_counts = (skill.get("summary") or {}).get("pair_count_by_relation") or {}
    return [
        {
            "relation": relation,
            "record_count": int(terminal_counts.get(relation, 0)),
            "pass_count": int(skill_counts.get(relation, 0)),
            "blocked_count": 0,
            "attempt_count": 0,
            "attempt_success_count": 0,
            "terminal_pair_count": int(terminal_counts.get(relation, 0)),
            "skill_pair_count": int(skill_counts.get(relation, 0)),
            "skill_rate_percent": round(
                100.0 * int(skill_counts.get(relation, 0)) / int(terminal_counts.get(relation, 1)),
                1,
            ),
        }
        for relation in sorted(set(terminal_counts) | set(skill_counts))
    ]


def _articulated_projection_rows(
    terminal: dict[str, Any],
    skill: dict[str, Any],
    *,
    profiles_path: Path | None,
) -> list[dict[str, Any]]:
    terminal_counts: Counter[str] = Counter(
        str(row.get("receiver_state_profile_id"))
        for row in terminal.get("records") or []
        if row.get("receiver_state_profile_id")
    )
    skill_counts: Counter[str] = Counter(
        str(row.get("receiver_state_profile_id"))
        for row in skill.get("records") or []
        if row.get("receiver_state_profile_id")
    )
    profiles_payload = _load_yaml(profiles_path) if profiles_path and profiles_path.is_file() else {}
    profiles = profiles_payload.get("profiles") or {}
    profile_ids = sorted(set(profiles) | set(terminal_counts) | set(skill_counts))
    rows: list[dict[str, Any]] = []
    for profile_id in profile_ids:
        profile = profiles.get(profile_id) or {}
        required = profile.get("acceptance_required") is not False
        skill_count = int(skill_counts[profile_id])
        if not required:
            disposition = "deferred"
        elif skill_count:
            disposition = "validated"
        else:
            disposition = "acceptance_blocked"
        rows.append(
            {
                "profile_id": profile_id,
                "review_status": str(profile.get("review_status") or "unknown"),
                "acceptance_required": required,
                "disposition": disposition,
                "terminal_pair_count": int(terminal_counts[profile_id]),
                "skill_pair_count": skill_count,
                "entry_mode": str((profile.get("skill") or {}).get("entry_mode") or ""),
                "reason": str(profile.get("abandonment_reason") or ""),
            }
        )
    return rows


def _frozen_failure_rows(gaps: dict[str, Any]) -> list[dict[str, Any]]:
    reasons = (gaps.get("summary") or {}).get("gap_count_by_reason") or {}
    skill_reasons = {
        str(reason).removeprefix("canonical_skill_fail: "): int(count)
        for reason, count in reasons.items()
        if str(reason).startswith("canonical_skill_fail: ")
    }
    return [
        {"reason": reason, "count": count}
        for reason, count in Counter(skill_reasons).most_common()
    ]


def _resolve_dashboard_path(
    value: Any,
    *,
    manifest_path: Path | None,
    data_dir: Path,
) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    candidates = [path]
    if manifest_path:
        candidates.append(manifest_path.parent / path)
    candidates.append(data_dir / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return path


def _dashboard_href(path: Path | None, data_dir: Path) -> str | None:
    if path is None or not path.is_file():
        return None
    return _path_string(Path(os.path.relpath(path, start=data_dir)))


def render_pair_validation_dashboard_html(model: dict[str, Any]) -> str:
    data = json.dumps(model, sort_keys=True, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(str(model["title"]))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: dark; --bg:#090e1a; --panel:#141c2d; --panel2:#101827; --line:#2a3653; --text:#eef3ff; --muted:#9caaca; --ok:#55d69e; --warn:#f6c85f; --bad:#ff7a8a; --accent:#78a9ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 ui-sans-serif,system-ui,sans-serif; }}
main {{ max-width:1480px; margin:auto; padding:28px; }}
h1 {{ margin:0 0 4px; font-size:28px; }} h2 {{ margin:0 0 14px; font-size:18px; }}
.muted {{ color:var(--muted); }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:22px 0; }}
.card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }}
.value {{ display:block; margin-top:4px; font-size:25px; font-weight:700; }}
.panels {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:16px; }}
.wide {{ grid-column:1/-1; }} .acceptance {{ margin:18px 0 0; display:flex; align-items:center; gap:12px; background:var(--panel2); }}
.badge {{ display:inline-block; border:1px solid currentColor; border-radius:999px; padding:4px 10px; font-weight:700; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:8px 7px; text-align:left; border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-weight:600; }} .ok {{ color:var(--ok); }} .warn {{ color:var(--warn); }} .bad {{ color:var(--bad); }}
code {{ color:var(--accent); overflow-wrap:anywhere; }}
a {{ color:var(--accent); }}
@media(max-width:600px) {{ main {{ padding:16px; }} .panels {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>
<h1>{title}</h1><div class="muted" id="generated"></div>
<section class="panel acceptance" id="acceptance"></section>
<section class="grid" id="overview"></section>
<section class="panels">
  <div class="panel"><h2>Frozen relation capabilities</h2><table><thead><tr><th>Relation</th><th>Terminal</th><th>Skill</th><th>Skill / terminal</th></tr></thead><tbody id="relations"></tbody></table></div>
  <div class="panel"><h2>Failures</h2><table><thead><tr><th>Reason</th><th>Count</th></tr></thead><tbody id="failures"></tbody></table></div>
  <div class="panel wide" id="runs-panel"><h2>Execution evidence</h2><table><thead><tr><th>Run</th><th>Stage</th><th>Scope</th><th>Pairs</th><th>Pass</th><th>Fail</th><th>Jobs</th><th>Attempts</th><th>Infra</th><th>Workers</th><th>Status</th><th>Evidence</th></tr></thead><tbody id="runs"></tbody></table></div>
  <div class="panel wide"><h2>Articulated profiles</h2><table><thead><tr><th>Profile</th><th>Required</th><th>Disposition</th><th>Entry</th><th>Terminal</th><th>Skill</th><th>Reason</th></tr></thead><tbody id="profiles"></tbody></table></div>
  <div class="panel"><h2>Frozen ledger statuses</h2><table><thead><tr><th>Status</th><th>Count</th></tr></thead><tbody id="statuses"></tbody></table></div>
  <div class="panel"><h2>Run integrity</h2><div id="integrity"></div></div>
  <div class="panel"><h2>Review evidence</h2><table><thead><tr><th>Evidence</th><th>Scope</th><th>Classification</th></tr></thead><tbody id="evidence"></tbody></table></div>
</section>
<script id="pair-dashboard-data" type="application/json">{data}</script>
<script>
const d=JSON.parse(document.getElementById("pair-dashboard-data").textContent);
const esc=x=>String(x??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
document.getElementById("generated").textContent=`Generated ${{d.generated_at}} · ${{d.sources.manifest_path||d.sources.report_path||"no manifest"}}`;
const ac=d.acceptance||{{}}, accepted=ac.acceptance_ready===true;
document.getElementById("acceptance").innerHTML=`<span class="badge ${{accepted?"ok":"bad"}}">${{accepted?"ACCEPTED":"NOT ACCEPTED"}}</span><div><b>${{esc(ac.status||"Run-scoped dashboard")}}</b><div class="muted">${{esc(ac.note||"")}}</div></div>`;
const cards=[["Frozen pairs",Object.values(d.frozen_status_counts||{{}}).reduce((a,b)=>a+Number(b||0),0)||d.overview.combined_pair_count],["Terminal pairs",d.overview.validated_terminal_pair_count],["Skill pairs",d.overview.validated_skill_pair_count],["Evidence attempts",d.evidence_attempt_count||d.overview.attempt_count],["Retry queue",Number((d.retry_queue||{{}}).job_count||0)],["Gaps",d.overview.gap_count]];
document.getElementById("overview").innerHTML=cards.map(([k,v])=>`<div class="card"><span class="muted">${{esc(k)}}</span><span class="value">${{esc(v)}}</span></div>`).join("");
const rows=(id,data,cols)=>document.getElementById(id).innerHTML=(data.length?data:[{{empty:"—"}}]).map(r=>`<tr>${{cols.map(c=>`<td>${{esc(r[c]??(c==="relation"||c==="reason"||c==="profile_id"?r.empty:""))}}</td>`).join("")}}</tr>`).join("");
rows("relations",d.relations,["relation","terminal_pair_count","skill_pair_count","skill_rate_percent"]);
rows("failures",d.failures,["reason","count"]);
rows("profiles",d.profiles,["profile_id","acceptance_required","disposition","entry_mode","terminal_pair_count","skill_pair_count","reason"]);
rows("statuses",Object.entries(d.frozen_status_counts||{{}}).map(([status,count])=>({{status,count}})),["status","count"]);
const runRows=d.runs||[];
document.getElementById("runs-panel").style.display=runRows.length?"block":"none";
document.getElementById("runs").innerHTML=runRows.map(r=>`<tr><td><b>${{esc(r.label)}}</b><div class="muted">${{esc(r.note)}}</div></td><td>${{esc(r.stage)}}</td><td>${{esc(r.scope)}}</td><td>${{esc(r.record_count)}}</td><td>${{esc(r.pass_count)}}</td><td>${{esc(r.fail_count)}}</td><td>${{esc(r.job_count)}}</td><td>${{esc(r.attempt_count)}}</td><td>${{esc(r.infrastructure_attempt_count)}}</td><td>${{esc(r.parallel_workers||"—")}}</td><td>${{esc(r.status)}}</td><td>${{r.report_href?`<a href="${{esc(r.report_href)}}">report</a>`:"—"}}${{r.audit_href?` · <a href="${{esc(r.audit_href)}}">audit</a>`:""}}</td></tr>`).join("");
document.getElementById("evidence").innerHTML=(d.review_evidence.length?d.review_evidence:[{{label:"—",scope:"",classification:"",href:""}}]).map(r=>`<tr><td>${{r.href?`<a href="${{esc(r.href)}}">${{esc(r.label)}}</a>`:esc(r.label)}}</td><td>${{esc(r.scope)}}</td><td>${{esc(r.classification)}}</td></tr>`).join("");
const p=d.progress||{{}}, a=d.audit||{{}}, c=d.projection_consistency||{{}};
document.getElementById("integrity").innerHTML=`<p>Artifact check: <b class="${{ac.artifact_check==="current"?"ok":"bad"}}">${{esc(ac.artifact_check||"not recorded")}}</b></p><p>Tests: <b>${{esc(ac.tests||"not recorded")}}</b></p><p>Projection sources: <b class="${{c.consistent?"ok":"bad"}}">${{c.consistent?"consistent":"mismatch"}}</b></p><p>Infrastructure blocked: <b class="${{ac.infrastructure_blocked_attempts===0?"ok":"bad"}}">${{esc(ac.infrastructure_blocked_attempts??"not recorded")}}</b></p><p class="muted"><code>${{esc(d.report.evidence_id||"no primary evidence")}}</code></p>`;
</script>
</main></body></html>
"""


def _relation_rows(
    records: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    terminal: dict[str, Any],
    skill: dict[str, Any],
) -> list[dict[str, Any]]:
    report_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        counter = report_counts[str(row.get("relation") or "unknown")]
        counter["records"] += 1
        counter["pass"] += int(bool(row.get("pass")))
        counter["blocked"] += int(str(row.get("outcome")) == "probe_run_blocked")
    for row in attempts:
        counter = report_counts[str(row.get("relation") or "unknown")]
        counter["attempts"] += 1
        counter["attempt_success"] += int(bool(row.get("success")))
    terminal_counts = (terminal.get("summary") or {}).get("pair_count_by_relation") or {}
    skill_counts = (skill.get("summary") or {}).get("pair_count_by_relation") or {}
    relations = sorted(set(report_counts) | set(terminal_counts) | set(skill_counts))
    return [
        {
            "relation": relation,
            "record_count": report_counts[relation]["records"],
            "pass_count": report_counts[relation]["pass"],
            "blocked_count": report_counts[relation]["blocked"],
            "attempt_count": report_counts[relation]["attempts"],
            "attempt_success_count": report_counts[relation]["attempt_success"],
            "terminal_pair_count": int(terminal_counts.get(relation, 0)),
            "skill_pair_count": int(skill_counts.get(relation, 0)),
        }
        for relation in relations
    ]


def _profile_rows(records: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        profile_id = row.get("receiver_state_profile_id")
        if not profile_id:
            continue
        counts[str(profile_id)]["records"] += 1
        counts[str(profile_id)]["pass"] += int(bool(row.get("pass")))
    for row in attempts:
        profile_id = row.get("receiver_state_profile_id")
        if not profile_id:
            continue
        counts[str(profile_id)]["attempts"] += 1
        counts[str(profile_id)]["attempt_success"] += int(bool(row.get("success")))
    return [
        {
            "profile_id": profile_id,
            "record_count": counter["records"],
            "pass_count": counter["pass"],
            "attempt_count": counter["attempts"],
            "attempt_success_count": counter["attempt_success"],
        }
        for profile_id, counter in sorted(counts.items())
    ]


def _projection_consistency(
    terminal: dict[str, Any],
    skill: dict[str, Any],
    combined: dict[str, Any],
    gaps: dict[str, Any],
) -> dict[str, Any]:
    source_hashes = {
        str(value)
        for value in (
            terminal.get("source_ledger_hash"),
            skill.get("source_ledger_hash"),
            gaps.get("source_ledger_hash"),
        )
        if value
    }
    hash_links_match = (
        not terminal
        or combined.get("terminal_projection_hash") == terminal.get("projection_hash")
    ) and (
        not skill
        or combined.get("skill_projection_hash") == skill.get("projection_hash")
    )
    return {
        "consistent": len(source_hashes) <= 1 and bool(hash_links_match),
        "source_ledger_hashes": sorted(source_hashes),
        "projection_hash_links_match": bool(hash_links_match),
    }


def _review_evidence_rows(data_dir: Path) -> list[dict[str, Any]]:
    specs = [
        (
            "Cabinet top-down clearance",
            "cabinet_46653_drawer_0_open",
            [
                data_dir.parent / "articulated_pair_validation" / "cabinet_top_down_clearance_audit.yml",
                data_dir / "articulated_pair_validation" / "cabinet_top_down_clearance_audit.yml",
            ],
        ),
        (
            "Microwave collision/support",
            "microwave_7167_door_open, microwave_7310_door_open",
            [
                data_dir.parent / "articulated_pair_validation" / "microwave_collision_support_audit.yml",
                data_dir / "articulated_pair_validation" / "microwave_collision_support_audit.yml",
            ],
        ),
        (
            "dump_into phase Render",
            "dump_into|063_tabletrashbin:8|110_basket:0",
            [data_dir / "runs" / "dump_into_render_audit.yml"],
        ),
    ]
    rows = []
    for label, scope, candidates in specs:
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            continue
        payload = _load_yaml(path)
        decision = payload.get("decision") or {}
        rows.append(
            {
                "label": label,
                "scope": scope,
                "classification": str(
                    decision.get("classification") or payload.get("kind") or "reviewed"
                ),
                "path": _path_string(path),
                "href": _path_string(Path(os.path.relpath(path, start=data_dir))),
            }
        )
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.CSafeLoader) or {}


def _path_string(path: Path | None) -> str | None:
    return None if path is None else str(path).replace("\\", "/")
