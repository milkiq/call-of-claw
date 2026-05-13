from __future__ import annotations

from typing import Any

from trpg_agent.memory.store import SqliteStore


def summarize_advisor_metrics(store: SqliteStore, session_id: str) -> dict[str, Any]:
    runs = store.list_advisor_runs_for_session(session_id)
    by_role: dict[str, list[dict[str, int]]] = {}
    for run in runs:
        metrics = _metrics_from_attempts(run.get("attempts", []))
        if not metrics:
            continue
        role = str(run.get("role") or "unknown")
        by_role.setdefault(role, []).append(metrics)

    role_summaries = {
        role: _summarize_metric_rows(rows)
        for role, rows in sorted(by_role.items())
    }
    totals = _summarize_metric_rows([row for rows in by_role.values() for row in rows])
    return {
        "session_id": session_id,
        "advisor_run_count": sum(len(rows) for rows in by_role.values()),
        "roles": role_summaries,
        "totals": totals,
    }


def compare_advisor_metrics(
    store: SqliteStore,
    baseline_session_id: str,
    candidate_session_id: str,
) -> dict[str, Any]:
    baseline = summarize_advisor_metrics(store, baseline_session_id)
    candidate = summarize_advisor_metrics(store, candidate_session_id)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": _metric_delta(baseline["totals"], candidate["totals"]),
    }


def _metrics_from_attempts(attempts: list[dict[str, Any]]) -> dict[str, int] | None:
    metrics = next(
        (
            attempt
            for attempt in attempts
            if isinstance(attempt, dict) and attempt.get("phase") == "metrics"
        ),
        None,
    )
    if not metrics:
        return None
    return {
        "elapsed_ms": _int_metric(metrics.get("elapsed_ms")),
        "estimated_prompt_chars": _int_metric(metrics.get("estimated_prompt_chars")),
        "player_input_chars": _int_metric(metrics.get("player_input_chars")),
        "context_chars": _int_metric(metrics.get("context_chars")),
        "schema_chars": _int_metric(metrics.get("schema_chars")),
        "estimated_response_chars": _int_metric(metrics.get("estimated_response_chars")),
        "attempt_count": _int_metric(metrics.get("attempt_count"), default=1),
        "repair_count": 1
        if any(
            isinstance(attempt, dict) and attempt.get("phase") == "repair"
            for attempt in attempts
        )
        else 0,
    }


def _summarize_metric_rows(rows: list[dict[str, int]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "elapsed_ms_p50": 0,
            "elapsed_ms_p95": 0,
            "avg_prompt_chars": 0,
            "avg_response_chars": 0,
            "avg_context_chars": 0,
            "avg_schema_chars": 0,
            "repair_rate": 0.0,
        }
    return {
        "count": len(rows),
        "elapsed_ms_p50": _percentile([row["elapsed_ms"] for row in rows], 0.50),
        "elapsed_ms_p95": _percentile([row["elapsed_ms"] for row in rows], 0.95),
        "avg_prompt_chars": round(
            sum(row["estimated_prompt_chars"] for row in rows) / len(rows),
            1,
        ),
        "avg_response_chars": round(
            sum(row["estimated_response_chars"] for row in rows) / len(rows),
            1,
        ),
        "avg_context_chars": round(
            sum(row["context_chars"] for row in rows) / len(rows),
            1,
        ),
        "avg_schema_chars": round(
            sum(row["schema_chars"] for row in rows) / len(rows),
            1,
        ),
        "repair_rate": round(sum(row["repair_count"] for row in rows) / len(rows), 4),
    }


def _metric_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "elapsed_ms_p50_reduction_pct": _reduction_pct(
            baseline.get("elapsed_ms_p50", 0),
            candidate.get("elapsed_ms_p50", 0),
        ),
        "avg_prompt_chars_reduction_pct": _reduction_pct(
            baseline.get("avg_prompt_chars", 0),
            candidate.get("avg_prompt_chars", 0),
        ),
        "avg_response_chars_reduction_pct": _reduction_pct(
            baseline.get("avg_response_chars", 0),
            candidate.get("avg_response_chars", 0),
        ),
        "repair_rate_delta": round(
            float(candidate.get("repair_rate", 0.0)) - float(baseline.get("repair_rate", 0.0)),
            4,
        ),
    }


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def _reduction_pct(baseline: float | int, candidate: float | int) -> float:
    base = float(baseline or 0)
    if base <= 0:
        return 0.0
    return round(((base - float(candidate or 0)) / base) * 100, 2)


def _int_metric(value: object, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
