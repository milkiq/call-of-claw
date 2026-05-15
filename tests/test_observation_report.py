import json

from coc.eval.observation_report import build_observation_report
from coc.memory.store import SqliteStore


def test_observation_report_reads_old_and_new_online_reports(tmp_path) -> None:
    reports_dir = tmp_path / "online-playtests"
    reports_dir.mkdir()
    old_report = {
        "result": {
            "run_id": "old",
            "kind": "online_playtest",
            "passed": 2,
            "total": 2,
            "findings": [],
        },
        "trace_sample": [
            {
                "turn": 1,
                "trace_events": [
                    {
                        "node": "critic_guardrail_with_llm",
                        "advisor": {
                            "fallback": "true",
                            "error": "The read operation timed out",
                        },
                    },
                    {"node": "retrieve_content_spans"},
                ],
            }
        ],
    }
    new_report = {
        "result": {
            "run_id": "new",
            "kind": "online_playtest",
            "passed": 0,
            "total": 2,
            "findings": [{"case_id": "runtime-timeout"}],
        },
        "runtime_profile": {
            "turn_count": 1,
            "slowest_nodes": [
                {
                    "turn": 1,
                    "node": "narrate_with_llm",
                    "category": "provider_wait",
                    "elapsed_ms": 123,
                }
            ],
        },
        "trace_sample": [
            {
                "turn": 1,
                "trace_events": [
                    {
                        "node": "route_with_intent_arbiter",
                        "advisor": {
                            "advisor_role": "intent_arbiter",
                            "elapsed_ms": "42",
                            "estimated_prompt_chars": "1000",
                            "context_chars": "700",
                            "schema_chars": "200",
                            "estimated_response_chars": "100",
                        },
                    },
                    {"node": "retrieve_content_spans", "diagnostics": {"search_backend": "scan"}},
                ],
            }
        ],
    }
    (reports_dir / "old-report.json").write_text(json.dumps(old_report), encoding="utf-8")
    (reports_dir / "new-report.json").write_text(json.dumps(new_report), encoding="utf-8")

    report = build_observation_report(
        store=SqliteStore(tmp_path / "app.sqlite"),
        reports_dir=reports_dir,
        source="reports",
    )

    assert report.online_runs == 2
    assert report.online_full_pass == 1
    assert report.runtime_profile_count == 1
    assert report.runtime_profile_missing == 1
    assert report.advisor_fallback_count == 1
    assert report.advisor_timeout_count == 1
    assert report.retrieval_diagnostics_missing == 1
    assert report.slowest_nodes[0]["node"] == "narrate_with_llm"
    assert report.advisor_roles["intent_arbiter"]["avg_prompt_chars"] == 1000


def test_observation_report_reads_store_advisor_metrics(tmp_path) -> None:
    store = SqliteStore(tmp_path / "app.sqlite")
    store.migrate()
    store.insert_advisor_run_once(
        run_id="advisor-1",
        turn_id="turn-1",
        role="rules_adjudicator",
        prompt_version="v1",
        input_hash="hash",
        output={},
        attempts=[
            {"phase": "initial", "raw_output": "{}"},
            {
                "phase": "metrics",
                "elapsed_ms": "55",
                "estimated_prompt_chars": "1200",
                "context_chars": "800",
                "schema_chars": "300",
                "estimated_response_chars": "20",
                "attempt_count": "1",
            },
        ],
    )

    report = build_observation_report(
        store=store,
        reports_dir=tmp_path / "missing",
        source="store",
    )

    assert report.advisor_event_count == 1
    assert report.max_advisor_elapsed_ms == 55
    assert report.advisor_roles["rules_adjudicator"]["elapsed_ms_p50"] == 55
