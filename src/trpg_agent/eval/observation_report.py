from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from trpg_agent.memory.store import SqliteStore

ObservationSource = Literal["reports", "store", "both"]


class ObservationReport(BaseModel):
    source: str
    report_files: int = 0
    store_eval_runs: int = 0
    online_runs: int = 0
    online_full_pass: int = 0
    runtime_profile_count: int = 0
    runtime_profile_missing: int = 0
    advisor_event_count: int = 0
    advisor_fallback_count: int = 0
    advisor_timeout_count: int = 0
    max_advisor_elapsed_ms: int = 0
    weighted_avg_prompt_chars: float = 0.0
    prompt_breakdown_missing: int = 0
    retrieval_diagnostics_missing: int = 0
    scenario_fast_path_count: int = 0
    scenario_full_director_count: int = 0
    slowest_nodes: list[dict[str, Any]] = Field(default_factory=list)
    advisor_roles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    findings_by_case: dict[str, int] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            "# TRPG Agent Observation Report",
            "",
            f"- Source: `{self.source}`",
            f"- Report files: {self.report_files}",
            f"- Store eval runs: {self.store_eval_runs}",
            f"- Online runs: {self.online_full_pass}/{self.online_runs} full pass",
            f"- Runtime profiles: {self.runtime_profile_count} present, "
            f"{self.runtime_profile_missing} missing",
            f"- Advisor events: {self.advisor_event_count}",
            f"- Advisor fallback/timeout: {self.advisor_fallback_count}/"
            f"{self.advisor_timeout_count}",
            f"- Max advisor elapsed: {self.max_advisor_elapsed_ms}ms",
            f"- Weighted avg prompt chars: {self.weighted_avg_prompt_chars:.1f}",
            f"- Prompt breakdown missing: {self.prompt_breakdown_missing}",
            f"- Retrieval diagnostics missing: {self.retrieval_diagnostics_missing}",
            f"- Scenario fast/full director: {self.scenario_fast_path_count}/"
            f"{self.scenario_full_director_count}",
            "",
            "## Advisor Roles",
        ]
        if not self.advisor_roles:
            lines.append("No advisor metrics found.")
        else:
            for role, summary in sorted(self.advisor_roles.items()):
                lines.append(
                    f"- {role}: count={summary['count']}, "
                    f"p50={summary['elapsed_ms_p50']}ms, "
                    f"p95={summary['elapsed_ms_p95']}ms, "
                    f"avg_prompt={summary['avg_prompt_chars']}, "
                    f"fallbacks={summary['fallbacks']}, timeouts={summary['timeouts']}"
                )
        lines.extend(["", "## Slowest Nodes"])
        if not self.slowest_nodes:
            lines.append("No runtime node profiles found.")
        else:
            for node in self.slowest_nodes[:10]:
                lines.append(
                    f"- {node.get('run', '')} turn={node.get('turn', '')}: "
                    f"{node.get('node')} ({node.get('category', 'unknown')}) "
                    f"{node.get('elapsed_ms')}ms"
                )
        lines.extend(["", "## Findings"])
        if not self.findings_by_case:
            lines.append("No persisted findings found in selected sources.")
        else:
            for case_id, count in sorted(self.findings_by_case.items()):
                lines.append(f"- {case_id}: {count}")
        return "\n".join(lines)


def build_observation_report(
    *,
    store: SqliteStore,
    reports_dir: Path,
    source: ObservationSource = "both",
    limit: int = 50,
) -> ObservationReport:
    report_payloads = _load_report_payloads(reports_dir, limit=limit) if source != "store" else []
    store_payloads = _load_store_eval_payloads(store, limit=limit) if source != "reports" else []
    advisor_events: list[dict[str, Any]] = []
    slowest_nodes: list[dict[str, Any]] = []
    findings: Counter[str] = Counter()
    runtime_profile_count = 0
    runtime_profile_missing = 0
    retrieval_diagnostics_missing = 0
    scenario_fast_path_count = 0
    scenario_full_director_count = 0
    online_runs = 0
    online_full_pass = 0
    seen_result_ids: set[str] = set()

    for payload in report_payloads:
        result = payload.get("result", {})
        run_id = str(result.get("run_id") or "")
        is_new_result = not run_id or run_id not in seen_result_ids
        if run_id:
            seen_result_ids.add(run_id)
        if is_new_result and result.get("kind") == "online_playtest":
            online_runs += 1
            if result.get("passed") == result.get("total"):
                online_full_pass += 1
        if is_new_result:
            for finding in result.get("findings", []):
                findings[str(finding.get("case_id", "unknown"))] += 1
        runtime = payload.get("runtime_profile")
        if isinstance(runtime, dict) and runtime.get("turn_count"):
            runtime_profile_count += 1
            slowest_nodes.extend(_slow_nodes_from_runtime(result.get("run_id", ""), runtime))
            scenario_fast_path_count += int(runtime.get("scenario_fast_path_count", 0))
            scenario_full_director_count += int(runtime.get("scenario_full_director_count", 0))
        else:
            runtime_profile_missing += 1
        trace_sample = payload.get("trace_sample", [])
        advisor_events.extend(_advisor_events_from_trace(trace_sample))
        retrieval_diagnostics_missing += _missing_retrieval_diagnostics(trace_sample)

    for payload in store_payloads:
        run_id = str(payload.get("run_id") or "")
        is_new_result = not run_id or run_id not in seen_result_ids
        if run_id:
            seen_result_ids.add(run_id)
        if is_new_result and payload.get("kind") == "online_playtest":
            online_runs += 1
            if payload.get("passed") == payload.get("total"):
                online_full_pass += 1
        if is_new_result:
            for finding in payload.get("findings", []):
                findings[str(finding.get("case_id", "unknown"))] += 1

    store_advisor_events = _advisor_events_from_store(store) if source != "reports" else []
    advisor_events.extend(store_advisor_events)
    advisor_summary = _advisor_summary(advisor_events)
    prompt_total = sum(int(event.get("estimated_prompt_chars", 0)) for event in advisor_events)
    prompt_count = sum(1 for event in advisor_events if int(event.get("estimated_prompt_chars", 0)))
    return ObservationReport(
        source=source,
        report_files=len(report_payloads),
        store_eval_runs=len(store_payloads),
        online_runs=online_runs,
        online_full_pass=online_full_pass,
        runtime_profile_count=runtime_profile_count,
        runtime_profile_missing=runtime_profile_missing,
        advisor_event_count=len(advisor_events),
        advisor_fallback_count=sum(1 for event in advisor_events if event.get("fallback")),
        advisor_timeout_count=sum(1 for event in advisor_events if event.get("timeout")),
        max_advisor_elapsed_ms=max(
            [int(event.get("elapsed_ms", 0)) for event in advisor_events] or [0]
        ),
        weighted_avg_prompt_chars=round(prompt_total / max(1, prompt_count), 1),
        prompt_breakdown_missing=sum(
            1
            for event in advisor_events
            if event.get("estimated_prompt_chars") and not event.get("context_chars")
        ),
        retrieval_diagnostics_missing=retrieval_diagnostics_missing,
        scenario_fast_path_count=scenario_fast_path_count,
        scenario_full_director_count=scenario_full_director_count,
        slowest_nodes=sorted(
            slowest_nodes,
            key=lambda item: int(item.get("elapsed_ms", 0)),
            reverse=True,
        )[:10],
        advisor_roles=advisor_summary,
        findings_by_case=dict(findings),
    )


def _load_report_payloads(reports_dir: Path, *, limit: int) -> list[dict[str, Any]]:
    if not reports_dir.exists():
        return []
    paths = sorted(reports_dir.glob("*-report.json"), key=lambda path: path.stat().st_mtime)
    selected = paths[-limit:] if limit > 0 else paths
    payloads = []
    for path in selected:
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return payloads


def _load_store_eval_payloads(store: SqliteStore, *, limit: int) -> list[dict[str, Any]]:
    rows = store.list_eval_runs()
    selected = rows[-limit:] if limit > 0 else rows
    return [row["payload"] for row in selected if isinstance(row.get("payload"), dict)]


def _advisor_events_from_trace(trace_sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for turn in trace_sample:
        for event in turn.get("trace_events", []):
            advisor = event.get("advisor")
            if isinstance(advisor, dict):
                events.append(_normalized_advisor_event(event.get("node"), advisor))
    return events


def _advisor_events_from_store(store: SqliteStore) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT role, attempts_json
            FROM advisor_runs
            ORDER BY created_at, id
            """
        ).fetchall()
    for row in rows:
        try:
            attempts = json.loads(row["attempts_json"])
        except Exception:
            attempts = []
        metrics = next(
            (
                attempt
                for attempt in attempts
                if isinstance(attempt, dict) and attempt.get("phase") == "metrics"
            ),
            {},
        )
        if metrics:
            events.append(_normalized_advisor_event(row["role"], metrics))
    return events


def _normalized_advisor_event(node: object, advisor: dict[str, Any]) -> dict[str, Any]:
    error = str(advisor.get("error", ""))
    fallback = advisor.get("fallback") in {True, "true", "True", "1"}
    timeout = "timeout" in error.lower() or "timed out" in error.lower()
    return {
        "role": str(advisor.get("advisor_role") or node or "unknown"),
        "elapsed_ms": _int(advisor.get("elapsed_ms")),
        "estimated_prompt_chars": _int(advisor.get("estimated_prompt_chars")),
        "estimated_response_chars": _int(advisor.get("estimated_response_chars")),
        "context_chars": _int(advisor.get("context_chars")),
        "schema_chars": _int(advisor.get("schema_chars")),
        "fallback": fallback,
        "timeout": timeout,
    }


def _advisor_summary(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(str(event.get("role", "unknown")), []).append(event)
    return {role: _summarize_role(rows) for role, rows in grouped.items()}


def _summarize_role(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [int(row.get("elapsed_ms", 0)) for row in rows if int(row.get("elapsed_ms", 0))]
    prompt = [int(row.get("estimated_prompt_chars", 0)) for row in rows]
    response = [int(row.get("estimated_response_chars", 0)) for row in rows]
    return {
        "count": len(rows),
        "elapsed_ms_p50": _percentile(elapsed, 0.50),
        "elapsed_ms_p95": _percentile(elapsed, 0.95),
        "avg_prompt_chars": round(sum(prompt) / max(1, len(prompt)), 1),
        "avg_response_chars": round(sum(response) / max(1, len(response)), 1),
        "fallbacks": sum(1 for row in rows if row.get("fallback")),
        "timeouts": sum(1 for row in rows if row.get("timeout")),
    }


def _slow_nodes_from_runtime(run_id: str, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = []
    for node in runtime.get("slowest_nodes", []):
        if isinstance(node, dict):
            nodes.append({"run": run_id, **node})
    return nodes


def _missing_retrieval_diagnostics(trace_sample: list[dict[str, Any]]) -> int:
    missing = 0
    for turn in trace_sample:
        for event in turn.get("trace_events", []):
            if event.get("node") == "retrieve_content_spans" and "diagnostics" not in event:
                missing += 1
    return missing


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def _int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
