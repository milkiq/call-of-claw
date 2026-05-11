from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any

import typer
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from rich.console import Console

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    PromptSession = None
    InMemoryHistory = None

from trpg_agent.app.config import load_config
from trpg_agent.content.compiled import load_compiled_ruleset
from trpg_agent.content.packages import PackageKind
from trpg_agent.content.registry import ContentRegistry
from trpg_agent.eval.judge import run_llm_judge, run_static_quality_gate
from trpg_agent.eval.live import run_live_eval
from trpg_agent.eval.online_playtest import run_online_playtest
from trpg_agent.eval.playtest import build_session_quality_summary, run_scripted_long_play
from trpg_agent.eval.regression import run_regression
from trpg_agent.eval.report import build_quality_report_from_store
from trpg_agent.eval.roadmap import derive_roadmap_from_store, write_roadmap_yaml
from trpg_agent.eval.runner import run_eval_cases
from trpg_agent.eval.scorecard import EvalFinding, EvalResult, score_from_findings
from trpg_agent.graph.runtime import durable_turn_graph, invoke_turn_graph, stream_turn_graph
from trpg_agent.langchain.models import build_chat_model, describe_model, load_model_config
from trpg_agent.langchain.structured import invoke_structured_with_repair
from trpg_agent.langchain.tracing import configure_langsmith
from trpg_agent.memory.canon import import_canon_jsonl
from trpg_agent.memory.store import SqliteStore
from trpg_agent.scenario.runtime import start_session

app = typer.Typer(no_args_is_help=True)
content_app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True)
memory_app = typer.Typer(no_args_is_help=True)
roadmap_app = typer.Typer(no_args_is_help=True)
session_app = typer.Typer(no_args_is_help=True)
console = Console()

GRAPH_PROGRESS_LABELS = {
    "receive_input": "接收行动",
    "load_runtime_context": "读取规则、剧本与 session",
    "retrieve_context_parallel": "检索上下文与记忆",
    "classify_player_intent": "判断玩家意图",
    "route_with_intent_arbiter": "判断玩家意图",
    "run_micro_gates": "执行权限与风险小检查",
    "advise_turn_with_single_llm": "综合判断本回合",
    "advise_rules_with_llm": "查规则与准备裁定",
    "adjudicate_with_llm": "生成回合计划",
    "plan_turn_locally": "生成回合计划",
    "ensure_resolution_tools": "准备规则工具",
    "execute_deterministic_tools": "执行规则工具 / 掷骰",
    "direct_scenario_locally": "判断场景推进",
    "direct_scenario_with_llm": "判断场景推进",
    "apply_world_patch_results": "应用已验证场景变化",
    "emit_turn_output": "生成 GM 回复",
    "narrate_with_llm": "生成 GM 回复",
    "critic_guardrail_locally": "质检与修正",
    "critic_guardrail_with_llm": "质检与修正",
    "review_and_curate_parallel": "质检与整理记忆",
    "curate_memory_locally": "整理长期记忆",
    "curate_memory_with_llm": "整理长期记忆",
    "persist_memory_curation": "保存长期记忆",
    "persist_turn": "保存回合",
    "replay_persisted_turn": "读取已保存回合",
}


class InteractiveInput:
    def __init__(self) -> None:
        self._session: Any | None = None
        if (
            PromptSession is not None
            and InMemoryHistory is not None
            and sys.stdin.isatty()
            and sys.stdout.isatty()
        ):
            self._session = PromptSession(history=InMemoryHistory())

    def prompt(
        self,
        text: str,
        *,
        prompt_suffix: str = ": ",
        default: str = "",
        show_default: bool = False,
    ) -> str:
        if self._session is not None:
            message = f"{text}{prompt_suffix}"
            return str(self._session.prompt(message, default=default))
        return str(
            typer.prompt(
                text,
                default=default,
                show_default=show_default,
                prompt_suffix=prompt_suffix,
            )
        )


class GraphProgressReporter:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled and console.is_terminal
        self._status: Any | None = None
        self._current = ""

    def __enter__(self) -> GraphProgressReporter:
        if not self.enabled:
            return self
        self._manager = console.status("接收行动", spinner="dots")
        self._status = self._manager.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._status is not None:
            self._manager.__exit__(*exc_info)
        self._status = None

    def update(self, node: str, _state: dict[str, Any]) -> None:
        if not self.enabled or self._status is None:
            return
        label = GRAPH_PROGRESS_LABELS.get(node)
        if not label or label == self._current:
            return
        self._current = label
        self._status.update(label)


class CharacterCreationExtraction(BaseModel):
    fields: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


CHARACTER_CREATION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract a tabletop roleplaying character sheet from setup answers. "
            "Use only explicit player answers and the provided ruleset character creation spec. "
            "Do not invent mechanical bonuses, secrets, inventory, or backstory facts. "
            "Return only the requested JSON object.",
        ),
        (
            "human",
            "Character creation spec:\n{spec}\n\n"
            "Player answers:\n{answers}\n\n"
            "Return a JSON object matching this schema:\n{schema}",
        ),
    ]
)


@content_app.command("check")
def content_check() -> None:
    """Validate content package manifests and references."""
    config = load_config()
    registry = ContentRegistry.load(config.content_dir, config.root_dir)
    issues = registry.validate()
    for kind, packages in registry.group_by_kind().items():
        console.print(f"{kind}: {', '.join(pkg.id for pkg in packages) or '(none)'}")
    if issues:
        for issue in issues:
            console.print(f"[red]content issue:[/red] {issue}")
        raise typer.Exit(1)
    console.print(f"[green]content OK[/green]: {len(registry.packages)} package(s)")


@memory_app.command("import-canon")
def memory_import_canon(
    path: Annotated[Path, typer.Argument()] = Path("seeds/canon-log.jsonl"),
) -> None:
    """Import a JSONL canon log into SQLite as append-only canon events."""
    config = load_config()
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    count = import_canon_jsonl(store, path)
    console.print(f"[green]imported[/green] {count} canon event(s)")


@eval_app.command("regression")
def eval_regression() -> None:
    """Run all local regression cases."""
    config = load_config()
    result = run_regression(config)
    console.print(result.to_console_text())
    if result.failed:
        raise typer.Exit(1)


@eval_app.command("deterministic")
def eval_deterministic() -> None:
    """Run YAML-backed deterministic and bootstrap turn eval cases."""
    config = load_config()
    result = run_eval_cases(config, kind="deterministic", case_kinds={"deterministic"})
    console.print(result.to_console_text())
    if result.failed:
        raise typer.Exit(1)


@eval_app.command("transcript")
def eval_transcript() -> None:
    """Run local transcript/turn eval cases through the current graph."""
    config = load_config()
    result = run_eval_cases(config, kind="transcript", case_kinds={"turn"})
    console.print(result.to_console_text())
    if result.failed:
        raise typer.Exit(1)


@eval_app.command("judge")
def eval_judge(
    output: Annotated[str, typer.Option("--output", "-o")] = "",
    forbidden_term: Annotated[list[str] | None, typer.Option("--forbidden-term")] = None,
    use_llm: Annotated[bool, typer.Option("--use-llm")] = False,
    transcript_json: Annotated[Path | None, typer.Option("--transcript-json")] = None,
    trace_json: Annotated[Path | None, typer.Option("--trace-json")] = None,
    evidence_file: Annotated[list[Path] | None, typer.Option("--evidence-file")] = None,
) -> None:
    """Run static quality gate, or LLM-as-judge when --use-llm is set."""
    config = load_config()
    if use_llm:
        if not transcript_json:
            console.print("[red]--transcript-json is required with --use-llm[/red]")
            raise typer.Exit(2)

        transcript = json.loads(transcript_json.read_text(encoding="utf-8"))
        trace = json.loads(trace_json.read_text(encoding="utf-8")) if trace_json else []
        evidence = [
            path.read_text(encoding="utf-8")
            for path in evidence_file or []
        ]
        model_config = load_model_config(config.root_dir / "llm.config.json")
        configure_langsmith(
            config,
            {
                **{key: str(value) for key, value in describe_model(model_config).items()},
                "eval_kind": "judge",
            },
        )
        model = build_chat_model(model_config)
        result = run_llm_judge(model, transcript=transcript, trace=trace, evidence=evidence)
        store = SqliteStore(config.sqlite_path)
        store.migrate()
        store.insert_eval_run(run_id=result.run_id, kind=result.kind, payload=result.model_dump())
    else:
        result = run_static_quality_gate(output=output, forbidden_terms=forbidden_term or [])
    console.print(result.to_console_text())
    if result.failed:
        raise typer.Exit(1)


@eval_app.command("live")
def eval_live(
    limit: Annotated[int, typer.Option("--limit")] = 3,
    min_score: Annotated[int, typer.Option("--min-score")] = 3,
    session_prefix: Annotated[str, typer.Option("--session-prefix")] = "live-eval",
    case_path: Annotated[Path | None, typer.Option("--case-path")] = None,
) -> None:
    """Run live play turns and LLM-as-judge quality evaluation."""
    config = load_config()
    try:
        model_config = load_model_config(config.root_dir / "llm.config.json")
        configure_langsmith(
            config,
            {
                **{key: str(value) for key, value in describe_model(model_config).items()},
                "eval_kind": "live",
            },
        )
        model = build_chat_model(model_config)
        result = run_live_eval(
            config,
            model,
            limit=limit,
            min_score=min_score,
            session_prefix=session_prefix,
            case_path=case_path,
            model_metadata={key: str(value) for key, value in describe_model(model_config).items()},
        )
    except Exception as error:
        console.print(f"[red]live eval failed:[/red] {error}")
        console.print("Use `trpg eval all --offline` to run only local checks.")
        raise typer.Exit(1) from error
    console.print(result.to_console_text())
    if result.failed:
        raise typer.Exit(1)


@eval_app.command("all")
def eval_all(
    offline: Annotated[bool, typer.Option("--offline")] = False,
    live_limit: Annotated[int, typer.Option("--live-limit")] = 3,
    min_score: Annotated[int, typer.Option("--min-score")] = 3,
) -> None:
    """Run local eval suites and, by default, live LLM quality evaluation."""
    config = load_config()
    local_result = run_eval_cases(config, kind="all")
    console.print(local_result.to_console_text())
    if local_result.failed:
        raise typer.Exit(1)
    if offline:
        return

    try:
        model_config = load_model_config(config.root_dir / "llm.config.json")
        configure_langsmith(
            config,
            {
                **{key: str(value) for key, value in describe_model(model_config).items()},
                "eval_kind": "all-live",
            },
        )
        model = build_chat_model(model_config)
        live_result = run_live_eval(
            config,
            model,
            limit=live_limit,
            min_score=min_score,
            model_metadata={key: str(value) for key, value in describe_model(model_config).items()},
        )
    except Exception as error:
        console.print(f"[red]live eval failed:[/red] {error}")
        console.print("Use `trpg eval all --offline` to run only local checks.")
        raise typer.Exit(1) from error
    console.print(live_result.to_console_text())
    if live_result.failed:
        raise typer.Exit(1)


@eval_app.command("report")
def eval_report(
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """Summarize persisted eval runs as a quality report."""
    config = load_config()
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    report = build_quality_report_from_store(store, limit=limit)
    console.print(report.to_markdown(), markup=False)


@eval_app.command("quality-report")
def eval_quality_report(
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """Summarize quality trends, failures, and roadmap input from persisted eval runs."""
    eval_report(limit=limit)


@eval_app.command("long-play")
def eval_long_play(
    turns: Annotated[int, typer.Option("--turns")] = 50,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    ruleset_id: Annotated[str | None, typer.Option("--ruleset-id")] = None,
    scenario_id: Annotated[str | None, typer.Option("--scenario-id")] = None,
) -> None:
    """Run a deterministic long-play durability and coherence smoke test."""
    config = load_config()
    result = run_scripted_long_play(
        config,
        turns=turns,
        session_id=session_id,
        ruleset_id=ruleset_id,
        scenario_id=scenario_id,
    )
    console.print(result.to_console_text())
    if result.failed:
        raise typer.Exit(1)


@eval_app.command("online-playtest")
def eval_online_playtest(
    turns: Annotated[int, typer.Option("--turns")] = 100,
    min_score: Annotated[int, typer.Option("--min-score")] = 4,
    player_mode: Annotated[str, typer.Option("--player-mode")] = "policy",
    judge_mode: Annotated[str, typer.Option("--judge-mode")] = "auto",
    per_call_timeout_seconds: Annotated[int, typer.Option("--per-call-timeout-seconds")] = 90,
    single_turn_advisor: Annotated[bool, typer.Option("--single-turn-advisor")] = False,
    micro_gates: Annotated[bool, typer.Option("--micro-gates")] = False,
    parallel_review: Annotated[bool, typer.Option("--parallel-review")] = False,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    ruleset_id: Annotated[str | None, typer.Option("--ruleset-id")] = None,
    scenario_id: Annotated[str | None, typer.Option("--scenario-id")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Run an online GM long-play test and write the full transcript."""
    config = load_config()
    try:
        model_config = load_model_config(config.root_dir / "llm.config.json")
        configure_langsmith(
            config,
            {
                **{key: str(value) for key, value in describe_model(model_config).items()},
                "eval_kind": "online-playtest",
                "turns": str(turns),
                "judge_mode": judge_mode,
                "per_call_timeout_seconds": str(per_call_timeout_seconds),
                "single_turn_advisor": str(single_turn_advisor),
                "micro_gates": str(micro_gates),
                "parallel_review": str(parallel_review),
            },
        )
        model = build_chat_model(model_config)
        result = run_online_playtest(
            config,
            model,
            turns=turns,
            min_score=min_score,
            player_mode="llm" if player_mode == "llm" else "policy",
            judge_mode=judge_mode if judge_mode in {"auto", "llm", "static"} else "auto",
            per_call_timeout_seconds=(
                per_call_timeout_seconds if per_call_timeout_seconds > 0 else None
            ),
            single_turn_advisor=single_turn_advisor,
            micro_gates=micro_gates,
            parallel_review=parallel_review,
            session_id=session_id,
            ruleset_id=ruleset_id,
            scenario_id=scenario_id,
            output_dir=output_dir,
            model_metadata={key: str(value) for key, value in describe_model(model_config).items()},
        )
    except Exception as error:
        console.print(f"[red]online playtest failed:[/red] {error}")
        raise typer.Exit(1) from error
    console.print(result.to_console_text())
    console.print(f"transcript: {result.metadata.get('transcript_path', '')}", markup=False)
    console.print(f"report: {result.metadata.get('report_path', '')}", markup=False)
    if result.failed:
        raise typer.Exit(1)


@eval_app.command("release-gates")
def eval_release_gates(
    long_play_turns: Annotated[int, typer.Option("--long-play-turns")] = 50,
) -> None:
    """Run offline release gates: content, eval suite, durable replay, and long play."""
    config = load_config()
    registry = ContentRegistry.load(config.content_dir, config.root_dir)
    findings: list[EvalFinding] = []
    content_issues = registry.validate()
    for issue in content_issues:
        findings.append(
            EvalFinding(
                case_id="release-content-check",
                dimension="infrastructure",
                severity="critical",
                message=issue,
                suggested_area="content.registry",
            )
        )

    local_result = run_eval_cases(config, kind="release-offline", persist=False)
    long_play = run_scripted_long_play(config, turns=long_play_turns, persist=False)
    findings.extend(local_result.findings)
    findings.extend(long_play.findings)
    total = 1 + local_result.total + long_play.total
    passed = (0 if content_issues else 1) + local_result.passed + long_play.passed
    result = EvalResult(
        run_id=f"release-gates-{uuid.uuid4().hex[:12]}",
        kind="release_gates",
        total=total,
        passed=passed,
        findings=findings,
        scorecard=score_from_findings(findings),
        metadata={
            "content_packages": str(len(registry.packages)),
            "long_play_turns": str(long_play_turns),
        },
    )
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    store.insert_eval_run(run_id=result.run_id, kind=result.kind, payload=result.model_dump())
    console.print(result.to_console_text())
    if result.failed:
        raise typer.Exit(1)


@roadmap_app.command("derive")
def roadmap_derive(
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Derive a development roadmap from stored eval findings."""
    config = load_config()
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    roadmap = derive_roadmap_from_store(store)
    text = roadmap.to_markdown()
    if output:
        write_roadmap_yaml(roadmap, output)
    console.print(text, markup=False)


@session_app.command("start")
def session_start(
    session_id: Annotated[str, typer.Option("--session-id")] = "default",
    ruleset_id: Annotated[str | None, typer.Option("--ruleset-id")] = None,
    scenario_id: Annotated[str | None, typer.Option("--scenario-id")] = None,
    reset: Annotated[bool, typer.Option("--reset")] = False,
) -> None:
    """Initialize a playable scenario session and print the public opening."""
    config = load_config()
    registry = ContentRegistry.load(config.content_dir, config.root_dir)
    selected_ruleset = ruleset_id
    selected_scenario = scenario_id
    if not selected_ruleset:
        rulesets = registry.by_kind(PackageKind.RULESET)
        selected_ruleset = rulesets[0].id if rulesets else None
    if not selected_scenario:
        scenarios = registry.by_kind(PackageKind.SCENARIO)
        selected_scenario = scenarios[0].id if scenarios else None
    if not selected_ruleset or not selected_scenario:
        console.print("[red]A ruleset and scenario package are required.[/red]")
        raise typer.Exit(2)

    store = SqliteStore(config.sqlite_path)
    state, opening = start_session(
        store=store,
        session_id=session_id,
        content_dir=config.content_dir,
        ruleset_id=selected_ruleset,
        scenario_id=selected_scenario,
        reset=reset,
    )
    console.print(f"session: {session_id}", markup=False)
    console.print(f"ruleset: {selected_ruleset}", markup=False)
    console.print(f"scenario: {selected_scenario}", markup=False)
    console.print(opening.strip(), markup=False)
    console.print(
        json.dumps({"world_projection": state}, ensure_ascii=False, indent=2),
        markup=False,
    )


@session_app.command("list")
def session_list() -> None:
    """List known playable sessions."""
    config = load_config()
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    sessions = store.list_sessions()
    if not sessions:
        console.print("No sessions.")
        return
    for session in sessions:
        console.print(
            (
                f"{session['id']} "
                f"ruleset={session.get('ruleset_id') or '-'} "
                f"scenario={session.get('scenario_id') or '-'} "
                f"updated={session.get('updated_at')}"
            ),
            markup=False,
        )


@session_app.command("recap")
def session_recap(
    session_id: Annotated[str, typer.Option("--session-id")] = "default",
    limit: Annotated[int, typer.Option("--limit")] = 5,
) -> None:
    """Print a concise recap from persisted turns and public state."""
    config = load_config()
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    turns = store.list_turns(session_id)[-max(1, limit) :]
    state = _public_world_state(store.get_session_state(session_id) or {})
    if not turns:
        console.print(f"No turns for session: {session_id}", markup=False)
        return
    console.print(f"session: {session_id}", markup=False)
    console.print("recent turns:", markup=False)
    for turn in turns:
        console.print(f"- player: {turn['input']}", markup=False)
        console.print(f"  gm: {turn['output']}", markup=False)
    console.print("public state:", markup=False)
    console.print(json.dumps(state, ensure_ascii=False, indent=2), markup=False)


@session_app.command("inspect")
def session_inspect(
    session_id: Annotated[str, typer.Option("--session-id")] = "default",
    gm_trace: Annotated[bool, typer.Option("--gm-trace")] = False,
) -> None:
    """Inspect public state, or full GM trace data when --gm-trace is set."""
    config = load_config()
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    payload = _session_payload(store, session_id=session_id, include_gm=gm_trace)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@session_app.command("export")
def session_export(
    session_id: Annotated[str, typer.Option("--session-id")] = "default",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    include_gm: Annotated[bool, typer.Option("--include-gm")] = False,
) -> None:
    """Export a session transcript and state as JSON."""
    config = load_config()
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    payload = _session_payload(store, session_id=session_id, include_gm=include_gm)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        console.print(str(output), markup=False)
        return
    typer.echo(text)


@session_app.command("quality-report")
def session_quality_report(
    session_id: Annotated[str, typer.Option("--session-id")] = "default",
) -> None:
    """Summarize persisted quality and continuity metrics for one session."""
    config = load_config()
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    summary = build_session_quality_summary(store, session_id)
    console.print("# Session Quality Report", markup=False)
    for key, value in summary.items():
        console.print(f"- {key}: {value}", markup=False)


def _session_payload(
    store: SqliteStore,
    *,
    session_id: str,
    include_gm: bool,
) -> dict:
    session = store.get_session(session_id)
    turns = store.list_turns(session_id)
    public_turns = [
        {
            "id": turn["id"],
            "player": turn["input"],
            "gm": turn["output"],
            "created_at": turn["created_at"],
        }
        for turn in turns
    ]
    payload = {
        "session": session or {"id": session_id},
        "world_projection": _public_world_state(store.get_session_state(session_id) or {}),
        "turns": public_turns,
    }
    if include_gm:
        payload.update(
            {
                "gm_traces": [
                    {
                        "id": turn["id"],
                        "trace": turn["trace"],
                    }
                    for turn in turns
                ],
                "memories": store.list_memories(scope=session_id),
                "critic_reports": store.list_critic_reports(session_id),
                "world_patch_applications": store.list_world_patch_applications(session_id),
            }
        )
    return payload


def _public_world_state(state: dict) -> dict:
    scene = state.get("scene")
    public_scene = None
    if isinstance(scene, dict):
        public_scene = {
            key: scene.get(key)
            for key in ["id", "title", "public_summary"]
            if key in scene
        }
    return {
        key: state.get(key)
        for key in [
            "active_scene",
            "clock",
            "revealed_facts",
            "known_clues",
            "npc_stance",
            "character_context",
        ]
        if key in state
    } | ({"scene": public_scene} if public_scene else {})


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _resolve_play_package_ids(
    registry: ContentRegistry,
    *,
    ruleset_id: str | None,
    scenario_id: str | None,
) -> tuple[str, str]:
    selected_ruleset = ruleset_id
    selected_scenario = scenario_id
    if not selected_ruleset:
        rulesets = registry.by_kind(PackageKind.RULESET)
        selected_ruleset = rulesets[0].id if rulesets else None
    if not selected_scenario:
        scenarios = registry.by_kind(PackageKind.SCENARIO)
        selected_scenario = scenarios[0].id if scenarios else None
    if not selected_ruleset or not selected_scenario:
        console.print("[red]A ruleset and scenario package are required.[/red]")
        raise typer.Exit(2)
    return selected_ruleset, selected_scenario


def _build_play_model(
    *,
    config: Any,
    use_llm: bool,
    session_id: str,
    ruleset_id: str,
    scenario_id: str,
    micro_gates: bool,
    single_turn_advisor: bool = False,
    parallel_review: bool = False,
) -> tuple[BaseChatModel | None, dict[str, str]]:
    if not use_llm:
        return None, {}
    model_config = load_model_config(config.root_dir / "llm.config.json")
    model_metadata = {key: str(value) for key, value in describe_model(model_config).items()}
    configure_langsmith(
        config,
        {
            **model_metadata,
            "session_id": session_id,
            "ruleset_id": ruleset_id,
            "scenario_id": scenario_id,
            "micro_gates": str(micro_gates),
            "single_turn_advisor": str(single_turn_advisor),
            "parallel_review": str(parallel_review),
        },
    )
    return build_chat_model(model_config), model_metadata


def _run_play_turn(
    graph: Any,
    *,
    player_input: str,
    session_id: str,
    ruleset_id: str,
    scenario_id: str,
    content_dir: Path,
    sqlite_path: Path,
    micro_gates: bool,
    single_turn_advisor: bool,
    parallel_review: bool,
    model_metadata: dict[str, str],
    progress: GraphProgressReporter | None = None,
) -> dict[str, Any]:
    state = {
        "player_input": player_input,
        "session_id": session_id,
        "thread_id": session_id,
        "turn_id": f"turn-{uuid.uuid4().hex[:12]}",
        "content_dir": str(content_dir),
        "sqlite_path": str(sqlite_path),
        "ruleset_id": ruleset_id,
        "scenario_id": scenario_id,
        "micro_gates_mode": micro_gates,
        "single_turn_advisor_mode": single_turn_advisor,
        "parallel_review_mode": parallel_review,
        "checkpoint_mode": "sqlite",
        "model_metadata": model_metadata,
    }
    if progress is not None and progress.enabled:
        return stream_turn_graph(graph, state, on_node=progress.update)
    return invoke_turn_graph(graph, state)


def _play_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_output": result.get("final_output", ""),
        "turn_plan": result.get("turn_plan", {}),
        "narration_plan": result.get("narration_plan", {}),
        "tool_results": result.get("tool_results", []),
        "trace_events": result.get("trace_events", []),
    }


def _print_play_result(result: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(_play_result_payload(result), ensure_ascii=False, indent=2))
        return
    console.print(result.get("final_output", ""), markup=False)


def _run_interactive_play_loop(
    *,
    session_id: str | None,
    use_llm: bool,
    micro_gates: bool,
    single_turn_advisor: bool,
    parallel_review: bool,
    progress_enabled: bool,
    json_output: bool,
    ruleset_id: str | None,
    scenario_id: str | None,
) -> None:
    config = load_config()
    registry = ContentRegistry.load(config.content_dir, config.root_dir)
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    resolved_session_id = session_id or f"play-{uuid.uuid4().hex[:8]}"
    existing_session = store.get_session(resolved_session_id) or {}
    selected_ruleset, selected_scenario = _resolve_play_package_ids(
        registry,
        ruleset_id=ruleset_id or existing_session.get("ruleset_id"),
        scenario_id=scenario_id or existing_session.get("scenario_id"),
    )
    model, model_metadata = _build_play_model(
        config=config,
        use_llm=use_llm,
        session_id=resolved_session_id,
        ruleset_id=selected_ruleset,
        scenario_id=selected_scenario,
        micro_gates=micro_gates,
        single_turn_advisor=single_turn_advisor,
        parallel_review=parallel_review,
    )

    had_state = store.get_session_state(resolved_session_id) is not None
    had_turns = bool(store.list_turns(resolved_session_id))
    state, opening = start_session(
        store=store,
        session_id=resolved_session_id,
        content_dir=config.content_dir,
        ruleset_id=selected_ruleset,
        scenario_id=selected_scenario,
        reset=False,
    )
    console.print(f"session: {resolved_session_id}", markup=False)
    console.print(f"ruleset: {selected_ruleset}", markup=False)
    console.print(f"scenario: {selected_scenario}", markup=False)
    if not had_state or not had_turns:
        console.print(opening.strip(), markup=False)
    if not _has_player_character(state):
        prompt_input = InteractiveInput()
        state = _ensure_character_created(
            store=store,
            registry=registry,
            session_id=resolved_session_id,
            ruleset_id=selected_ruleset,
            state=state,
            model=model,
            prompt_input=prompt_input,
        )
    else:
        prompt_input = InteractiveInput()
        character = _player_character(state)
        if character and character.get("name"):
            console.print(f"resuming character: {character['name']}", markup=False)
        console.print("resuming session. Type /help for commands.", markup=False)

    with durable_turn_graph(sqlite_path=config.sqlite_path, model=model) as graph:
        while True:
            try:
                player_text = prompt_input.prompt("你", prompt_suffix="> ")
            except (EOFError, KeyboardInterrupt):
                console.print("")
                break
            text = player_text.strip()
            if not text:
                continue
            command = text.lower()
            if command in {"/quit", "/exit"}:
                break
            if command == "/help":
                _print_interactive_help()
                continue
            if command == "/recap":
                _print_session_recap(store, resolved_session_id)
                continue
            if command == "/session":
                _print_interactive_session(store, resolved_session_id)
                continue
            try:
                progress_active = progress_enabled and not json_output
                with GraphProgressReporter(enabled=progress_active) as reporter:
                    result = _run_play_turn(
                        graph,
                        player_input=text,
                        session_id=resolved_session_id,
                        ruleset_id=selected_ruleset,
                        scenario_id=selected_scenario,
                        content_dir=config.content_dir,
                        sqlite_path=config.sqlite_path,
                        micro_gates=micro_gates,
                        single_turn_advisor=single_turn_advisor,
                        parallel_review=parallel_review,
                        model_metadata=model_metadata,
                        progress=reporter,
                    )
            except Exception as error:
                console.print(f"[red]turn failed:[/red] {error}")
                continue
            _print_play_result(result, json_output=json_output)

    console.print(f"session-id: {resolved_session_id}", markup=False)
    console.print(f"resume: trpg play --session-id {resolved_session_id}", markup=False)


def _print_interactive_help() -> None:
    console.print("Commands: /help, /recap, /session, /quit", markup=False)
    console.print("Type any other text as your character's action.", markup=False)


def _print_session_recap(store: SqliteStore, session_id: str) -> None:
    turns = store.list_turns(session_id)[-5:]
    if not turns:
        console.print("No turns yet.", markup=False)
        return
    for turn in turns:
        console.print(f"- player: {turn['input']}", markup=False)
        console.print(f"  gm: {turn['output']}", markup=False)


def _print_interactive_session(store: SqliteStore, session_id: str) -> None:
    payload = _session_payload(store, session_id=session_id, include_gm=False)
    console.print(json.dumps(payload["session"], ensure_ascii=False, indent=2), markup=False)
    character = payload.get("world_projection", {}).get("character_context", {})
    if character:
        console.print(json.dumps(character, ensure_ascii=False, indent=2), markup=False)


def _ensure_character_created(
    *,
    store: SqliteStore,
    registry: ContentRegistry,
    session_id: str,
    ruleset_id: str,
    state: dict[str, Any],
    model: BaseChatModel | None,
    prompt_input: InteractiveInput,
) -> dict[str, Any]:
    ruleset = load_compiled_ruleset(registry, ruleset_id)
    spec = ruleset.character_creation
    character_context = dict(ruleset.default_character_context)
    existing_context = state.get("character_context")
    if isinstance(existing_context, dict):
        character_context.update(existing_context)
    if isinstance(character_context.get("player_character"), dict):
        state["character_context"] = character_context
        store.set_session_state(session_id=session_id, state=state)
        return state
    if not spec.enabled:
        state["character_context"] = character_context
        store.set_session_state(session_id=session_id, state=state)
        return state

    console.print(spec.intro.strip(), markup=False)
    answers: dict[str, str] = {}
    for question in spec.questions:
        answers[question.id] = _ask_character_question(question, prompt_input)
    extraction = _extract_character_creation(spec=spec, answers=answers, model=model)
    player_character, character_context = _apply_character_creation(
        spec=spec,
        answers=answers,
        extraction=extraction,
        base_character_context=character_context,
    )
    state["character_context"] = character_context
    store.set_session_state(session_id=session_id, state=state)
    summary = _render_character_summary(spec.summary_template, player_character, character_context)
    console.print("角色创建完成：", markup=False)
    console.print(summary, markup=False)
    return state


def _has_player_character(state: dict[str, Any]) -> bool:
    return bool(_player_character(state))


def _player_character(state: dict[str, Any]) -> dict[str, Any]:
    character_context = state.get("character_context")
    if not isinstance(character_context, dict):
        return {}
    character = character_context.get("player_character")
    return character if isinstance(character, dict) else {}


def _ask_character_question(question: Any, prompt_input: InteractiveInput) -> str:
    prompt = question.prompt
    if question.choices:
        prompt = f"{prompt} ({'/'.join(question.choices)})"
    if question.numeric_range:
        prompt = f"{prompt} [{question.numeric_range[0]}-{question.numeric_range[1]}]"
    while True:
        answer = prompt_input.prompt(prompt, default="", show_default=False).strip()
        if not answer and question.required:
            console.print("这个问题需要回答。", markup=False)
            continue
        if question.numeric_range and answer:
            value = _parse_first_int(answer)
            lower, upper = question.numeric_range[0], question.numeric_range[1]
            if value is None or value < lower or value > upper:
                console.print(f"请输入 {lower}-{upper} 范围内的数字。", markup=False)
                continue
        return answer


def _extract_character_creation(
    *,
    spec: Any,
    answers: dict[str, str],
    model: BaseChatModel | None,
) -> CharacterCreationExtraction:
    local = CharacterCreationExtraction(fields=_local_character_fields(spec, answers))
    if model is None:
        return local
    try:
        extracted, _attempts = invoke_structured_with_repair(
            model=model,
            prompt=CHARACTER_CREATION_EXTRACTION_PROMPT,
            schema=CharacterCreationExtraction,
            payload={
                "spec": spec.model_dump(),
                "answers": answers,
                "schema": CharacterCreationExtraction.model_json_schema(),
            },
            model_kwargs={"max_tokens": 500},
        )
    except Exception:
        return local
    fields = dict(local.fields)
    for key, value in extracted.fields.items():
        if value not in (None, ""):
            fields[key] = value
    return CharacterCreationExtraction(fields=fields, notes=extracted.notes)


def _local_character_fields(spec: Any, answers: dict[str, str]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for question in spec.questions:
        answer = answers.get(question.id, "").strip()
        if not answer:
            continue
        if question.numeric_range:
            value = _parse_first_int(answer)
            fields[question.field] = value if value is not None else answer
        else:
            fields[question.field] = answer
    return fields


def _apply_character_creation(
    *,
    spec: Any,
    answers: dict[str, str],
    extraction: CharacterCreationExtraction,
    base_character_context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fields = dict(extraction.fields)
    player_character = {
        question.field: fields.get(question.field, answers.get(question.id, "")).strip()
        if isinstance(fields.get(question.field, answers.get(question.id, "")), str)
        else fields.get(question.field)
        for question in spec.questions
        if fields.get(question.field, answers.get(question.id, "")) not in (None, "")
    }
    if extraction.notes:
        player_character["notes"] = extraction.notes

    character_context = dict(base_character_context)
    for assignment in spec.mechanical_assignments:
        source_value = _mechanical_source_value(assignment, spec, fields, answers)
        value = _coerce_mechanical_value(assignment, source_value)
        if value is not None:
            character_context[assignment.field] = value
            player_character[assignment.field] = value
    character_context["player_character"] = player_character
    return player_character, character_context


def _mechanical_source_value(
    assignment: Any,
    spec: Any,
    fields: dict[str, Any],
    answers: dict[str, str],
) -> Any:
    if assignment.source_question:
        for question in spec.questions:
            if question.id == assignment.source_question:
                return fields.get(question.field, answers.get(question.id, assignment.default))
    return fields.get(assignment.field, assignment.default)


def _coerce_mechanical_value(assignment: Any, value: Any) -> Any:
    if value in (None, ""):
        value = assignment.default
    if assignment.min is not None or assignment.max is not None:
        numeric = _parse_first_int(str(value))
        if numeric is None:
            numeric = _parse_first_int(str(assignment.default))
        if numeric is None:
            return None
        if assignment.min is not None and numeric < assignment.min:
            return assignment.default
        if assignment.max is not None and numeric > assignment.max:
            return assignment.default
        return numeric
    if assignment.allowed_values and value not in assignment.allowed_values:
        return assignment.default if assignment.default in assignment.allowed_values else None
    return value


def _parse_first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    return int(match.group(0)) if match else None


def _render_character_summary(
    template: str,
    player_character: dict[str, Any],
    character_context: dict[str, Any],
) -> str:
    data = _SafeFormatDict({**character_context, **player_character})
    return template.format_map(data).strip()


app.add_typer(content_app, name="content")
app.add_typer(eval_app, name="eval")
app.add_typer(memory_app, name="memory")
app.add_typer(roadmap_app, name="roadmap")
app.add_typer(session_app, name="session")


@app.command("play")
def play(
    player_input: Annotated[str | None, typer.Option("--input", "-i")] = None,
    use_llm: Annotated[bool, typer.Option("--use-llm")] = False,
    micro_gates: Annotated[bool, typer.Option("--micro-gates")] = False,
    single_turn_advisor: Annotated[bool, typer.Option("--single-turn-advisor")] = False,
    parallel_review: Annotated[bool, typer.Option("--parallel-review")] = False,
    local: Annotated[bool, typer.Option("--local")] = False,
    progress: Annotated[bool, typer.Option("--progress/--no-progress")] = True,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    ruleset_id: Annotated[str | None, typer.Option("--ruleset-id")] = None,
    scenario_id: Annotated[str | None, typer.Option("--scenario-id")] = None,
) -> None:
    """Run one turn, or enter an interactive play loop when --input is omitted."""
    if player_input is None:
        _run_interactive_play_loop(
            session_id=session_id,
            use_llm=not local,
            micro_gates=True,
            single_turn_advisor=single_turn_advisor,
            parallel_review=parallel_review,
            progress_enabled=progress,
            json_output=json_output,
            ruleset_id=ruleset_id,
            scenario_id=scenario_id,
        )
        return

    config = load_config()
    registry = ContentRegistry.load(config.content_dir, config.root_dir)
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    resolved_session_id = session_id or "default"
    existing_session = store.get_session(resolved_session_id) or {}
    selected_ruleset, selected_scenario = _resolve_play_package_ids(
        registry,
        ruleset_id=ruleset_id or existing_session.get("ruleset_id"),
        scenario_id=scenario_id or existing_session.get("scenario_id"),
    )
    model, model_metadata = _build_play_model(
        config=config,
        use_llm=use_llm and not local,
        session_id=resolved_session_id,
        ruleset_id=selected_ruleset,
        scenario_id=selected_scenario,
        micro_gates=micro_gates,
        single_turn_advisor=single_turn_advisor,
        parallel_review=parallel_review,
    )
    try:
        with durable_turn_graph(sqlite_path=config.sqlite_path, model=model) as graph:
            with GraphProgressReporter(enabled=progress and not json_output) as reporter:
                result = _run_play_turn(
                    graph,
                    player_input=player_input,
                    session_id=resolved_session_id,
                    ruleset_id=selected_ruleset,
                    scenario_id=selected_scenario,
                    content_dir=config.content_dir,
                    sqlite_path=config.sqlite_path,
                    micro_gates=micro_gates,
                    single_turn_advisor=single_turn_advisor,
                    parallel_review=parallel_review,
                    model_metadata=model_metadata,
                    progress=reporter,
                )
    except Exception as error:
        console.print(f"[red]play failed:[/red] {error}")
        if use_llm and not local:
            console.print("Retry without `--use-llm` to run the local fallback graph.")
        raise typer.Exit(1) from error
    _print_play_result(result, json_output=json_output)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
