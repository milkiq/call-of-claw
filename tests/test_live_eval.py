from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from coc.app.config import load_config
from coc.eval.live import run_live_eval
from coc.memory.store import SqliteStore


def test_live_eval_runs_with_fake_model(tmp_path: Path) -> None:
    config = load_config(Path.cwd())
    config = config.__class__(
        root_dir=config.root_dir,
        content_dir=config.content_dir,
        seeds_dir=config.seeds_dir,
        data_dir=tmp_path,
        sqlite_path=tmp_path / "live.sqlite",
        langsmith_tracing=False,
        langsmith_project=config.langsmith_project,
    )
    model = FakeListChatModel(
        responses=[
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "player acts"},
              "route": "free_action",
              "needs_rules_resolution": false,
              "needs_scenario_director": true,
              "needs_memory_recall": false,
              "allow_direct_answer": false,
              "reasoning_summary": "Generic routing decision.",
              "uncertainty": null,
              "citations": []
            }
            """,
            """
            {
              "intent": {"kind": "action", "confidence": 0.9, "reason": "player acts"},
              "authority": {"ok": true, "reason": "grounded"},
              "decision": "free_action",
              "tool_requests": [],
              "narration_brief": "The character inspects the door.",
              "citations": []
            }
            """,
            """
            {
              "decision": "no_change",
              "proposed_patches": [],
              "player_visible_context": "No scene change.",
              "gm_only_reason": "No scenario patch needed.",
              "citations": []
            }
            """,
            """
            {
              "final_text": "门很沉，门缝里有冷光。",
              "canon_event_draft": null,
              "memory_candidates": []
            }
            """,
            """
            {
              "ok": true,
              "blocks_output": false,
              "findings": [],
              "revised_final_text": null,
              "reasoning_summary": "ok"
            }
            """,
            """
            {
              "canon_event_draft": null,
              "memory_candidates": [],
              "contradictions": [],
              "should_write": false
            }
            """,
            """
            {
              "summary": "ok",
              "scorecard": {
                "rules_correctness": 5,
                "fictional_authority": 5,
                "continuity": 5,
                "player_agency": 5,
                "pacing": 5,
                "progressive_disclosure": 5,
                "memory_behavior": 5,
                "narration_quality": 5,
                "trace_explainability": 5,
                "generic_architecture_compliance": 5
              },
              "findings": []
            }
            """,
        ]
    )

    result = run_live_eval(config, model, limit=1)

    assert result.failed == 0
    assert result.total == 2
    assert result.kind == "live_llm_eval"
    run_session_prefix = result.metadata["run_session_prefix"]
    assert run_session_prefix.startswith("live-eval-live-")
    store = SqliteStore(config.sqlite_path)
    assert len(store.list_turns(f"{run_session_prefix}-1")) == 1
