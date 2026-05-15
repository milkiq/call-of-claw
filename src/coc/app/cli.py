from __future__ import annotations

import json
import re
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any

import typer
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.styles import Style
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    PromptSession = None
    InMemoryHistory = None
    Style = None

from coc.app.config import AppConfig, load_config
from coc.content.compiled import load_compiled_ruleset, load_compiled_scenario
from coc.content.compiler import (
    compile_content_draft,
    draft_to_json,
    write_content_draft,
)
from coc.content.packages import PackageKind
from coc.content.registry import ContentRegistry
from coc.content.retrieval import warm_content_index
from coc.eval.advisor_metrics import compare_advisor_metrics, summarize_advisor_metrics
from coc.eval.judge import run_llm_judge, run_static_quality_gate
from coc.eval.live import run_live_eval
from coc.eval.observation_report import build_observation_report
from coc.eval.online_playtest import run_online_playtest
from coc.eval.playtest import build_session_quality_summary, run_scripted_long_play
from coc.eval.regression import run_regression
from coc.eval.report import build_quality_report_from_store
from coc.eval.roadmap import derive_roadmap_from_store, write_roadmap_yaml
from coc.eval.runner import run_eval_cases
from coc.eval.scorecard import EvalFinding, EvalResult, score_from_findings
from coc.eval.session_cleanup import cleanup_known_test_sessions, is_test_session_id
from coc.graph.runtime import (
    delete_turn_graph_checkpoints,
    durable_turn_graph,
    invoke_turn_graph,
    stream_turn_graph,
)
from coc.langchain.models import build_chat_model, describe_model, load_model_config
from coc.langchain.structured import invoke_structured_with_repair
from coc.langchain.tracing import configure_langsmith
from coc.memory.canon import import_canon_jsonl
from coc.memory.store import SqliteStore
from coc.release.builder import ReleaseBuildSpec, build_release_bundle
from coc.rules.plugin_runtime import load_rules_plugin
from coc.scenario.runtime import start_session
from coc.tools.dice import roll_dice_once

app = typer.Typer(no_args_is_help=True)
content_app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True)
memory_app = typer.Typer(no_args_is_help=True)
roadmap_app = typer.Typer(no_args_is_help=True)
release_app = typer.Typer(no_args_is_help=True)
session_app = typer.Typer(no_args_is_help=True)
console = Console()


@dataclass(frozen=True)
class PlayProfileConfig:
    name: str
    use_llm: bool
    micro_gates: bool
    single_turn_advisor: bool
    conditional_advisors: bool
    parallel_review: bool
    advisor_contracts: str
    runtime_budget_profile: str
    context_budget_mode: str


PLAY_PROFILE_DEFAULTS: dict[str, PlayProfileConfig] = {
    "balanced": PlayProfileConfig(
        name="balanced",
        use_llm=True,
        micro_gates=False,
        single_turn_advisor=False,
        conditional_advisors=False,
        parallel_review=False,
        advisor_contracts="legacy",
        runtime_budget_profile="balanced",
        context_budget_mode="shadow",
    ),
    "fast": PlayProfileConfig(
        name="fast",
        use_llm=True,
        micro_gates=False,
        single_turn_advisor=False,
        conditional_advisors=True,
        parallel_review=True,
        advisor_contracts="legacy",
        runtime_budget_profile="fast",
        context_budget_mode="enforced",
    ),
    "theatrical": PlayProfileConfig(
        name="theatrical",
        use_llm=True,
        micro_gates=False,
        single_turn_advisor=False,
        conditional_advisors=False,
        parallel_review=False,
        advisor_contracts="legacy",
        runtime_budget_profile="theatrical",
        context_budget_mode="shadow",
    ),
}

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
    "select_scenario_surface_with_llm": "选择可见场景信息",
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
        self._style: Any | None = None
        if (
            PromptSession is not None
            and InMemoryHistory is not None
            and sys.stdin.isatty()
            and sys.stdout.isatty()
        ):
            if Style is not None:
                self._style = Style.from_dict(
                    {
                        "prompt.role": "ansicyan bold",
                        "prompt.suffix": "ansiwhite",
                    }
                )
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
            message = [
                ("class:prompt.role", text),
                ("class:prompt.suffix", prompt_suffix),
            ]
            return str(self._session.prompt(message, default=default, style=self._style))
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


@content_app.command("compile")
def content_compile(
    kind: Annotated[str, typer.Option("--kind")] = "ruleset",
    source: Annotated[Path, typer.Option("--source")] = Path("rules.md"),
    package_id: Annotated[str, typer.Option("--package-id")] = "compiled_package",
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    write: Annotated[bool, typer.Option("--write")] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Compile a ruleset or scenario source document into package files."""
    if kind not in {"ruleset", "scenario"}:
        console.print("[red]--kind must be ruleset or scenario.[/red]")
        raise typer.Exit(2)
    if not source.exists():
        console.print(f"[red]source not found:[/red] {source}")
        raise typer.Exit(2)
    config = load_config()
    model_config = load_model_config(config.root_dir / "llm.config.json")
    configure_langsmith(
        config,
        {
            **{key: str(value) for key, value in describe_model(model_config).items()},
            "content_compile_kind": kind,
            "package_id": package_id,
        },
    )
    model = build_chat_model(model_config)
    try:
        draft = compile_content_draft(
            model=model,
            kind="ruleset" if kind == "ruleset" else "scenario",
            package_id=package_id,
            source=source.read_text(encoding="utf-8"),
        )
        package_dir = None
        if write:
            base_dir = output_dir or (
                config.content_dir / ("rulesets" if kind == "ruleset" else "scenarios")
            )
            package_dir = write_content_draft(
                kind="ruleset" if kind == "ruleset" else "scenario",
                package_id=package_id,
                draft=draft,
                output_dir=base_dir,
                force=force,
            )
    except Exception as error:
        console.print(f"[red]content compile failed:[/red] {error}")
        raise typer.Exit(1) from error
    if json_output:
        typer.echo(draft_to_json(draft))
        return
    if package_dir:
        console.print(f"wrote package: {package_dir}", markup=False)
    else:
        console.print(draft_to_json(draft), markup=False)


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
    keep_session: Annotated[
        bool,
        typer.Option("--keep-session", help="Keep live eval sessions for manual inspection."),
    ] = False,
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
            cleanup_session=not keep_session,
            model_metadata={key: str(value) for key, value in describe_model(model_config).items()},
        )
    except Exception as error:
        console.print(f"[red]live eval failed:[/red] {error}")
        console.print("Use `coc eval all --offline` to run only local checks.")
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
            cleanup_session=True,
            model_metadata={key: str(value) for key, value in describe_model(model_config).items()},
        )
    except Exception as error:
        console.print(f"[red]live eval failed:[/red] {error}")
        console.print("Use `coc eval all --offline` to run only local checks.")
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


@eval_app.command("advisor-metrics")
def eval_advisor_metrics(
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    compare: Annotated[list[str] | None, typer.Option("--compare")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Summarize advisor latency and prompt/response character diagnostics."""
    config = load_config()
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    compare_ids = list(compare or [])
    if compare_ids:
        if len(compare_ids) != 2:
            console.print("[red]--compare requires exactly two session ids.[/red]")
            raise typer.Exit(2)
        payload = compare_advisor_metrics(store, compare_ids[0], compare_ids[1])
    else:
        if not session_id:
            console.print("[red]Provide --session-id or --compare twice.[/red]")
            raise typer.Exit(2)
        payload = summarize_advisor_metrics(store, session_id)
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(json.dumps(payload, ensure_ascii=False, indent=2), markup=False)


@eval_app.command("observation-report")
def eval_observation_report(
    source: Annotated[str, typer.Option("--source")] = "both",
    limit: Annotated[int, typer.Option("--limit")] = 50,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Summarize runtime, advisor, context, fallback, and timeout diagnostics."""
    if source not in {"reports", "store", "both"}:
        console.print("[red]--source must be reports, store, or both.[/red]")
        raise typer.Exit(2)
    config = load_config()
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    report = build_observation_report(
        store=store,
        reports_dir=config.data_dir / "online-playtests",
        source=source,  # type: ignore[arg-type]
        limit=limit,
    )
    rendered = (
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2)
        if json_output
        else report.to_markdown()
    )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    typer.echo(rendered)


@eval_app.command("long-play")
def eval_long_play(
    turns: Annotated[int, typer.Option("--turns")] = 50,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    ruleset_id: Annotated[str | None, typer.Option("--ruleset-id")] = None,
    scenario_id: Annotated[str | None, typer.Option("--scenario-id")] = None,
    keep_session: Annotated[
        bool,
        typer.Option("--keep-session", help="Keep the generated long-play session."),
    ] = False,
) -> None:
    """Run a deterministic long-play durability and coherence smoke test."""
    config = load_config()
    result = run_scripted_long_play(
        config,
        turns=turns,
        session_id=session_id,
        ruleset_id=ruleset_id,
        scenario_id=scenario_id,
        cleanup_session=not keep_session,
    )
    console.print(result.to_console_text())
    if result.failed:
        raise typer.Exit(1)


@eval_app.command("online-playtest")
def eval_online_playtest(
    turns: Annotated[int, typer.Option("--turns")] = 100,
    min_score: Annotated[int, typer.Option("--min-score")] = 4,
    profile: Annotated[str, typer.Option("--profile")] = "balanced",
    player_mode: Annotated[str, typer.Option("--player-mode")] = "policy",
    judge_mode: Annotated[str, typer.Option("--judge-mode")] = "auto",
    per_call_timeout_seconds: Annotated[int, typer.Option("--per-call-timeout-seconds")] = 90,
    single_turn_advisor: Annotated[
        bool | None,
        typer.Option("--single-turn-advisor/--no-single-turn-advisor"),
    ] = None,
    micro_gates: Annotated[
        bool | None,
        typer.Option("--micro-gates/--no-micro-gates"),
    ] = None,
    parallel_review: Annotated[
        bool | None,
        typer.Option("--parallel-review/--no-parallel-review"),
    ] = None,
    advisor_contracts: Annotated[str | None, typer.Option("--advisor-contracts")] = None,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    ruleset_id: Annotated[str | None, typer.Option("--ruleset-id")] = None,
    scenario_id: Annotated[str | None, typer.Option("--scenario-id")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    keep_session: Annotated[
        bool,
        typer.Option("--keep-session", help="Keep the generated online playtest session."),
    ] = False,
) -> None:
    """Run an online GM long-play test and write the full transcript."""
    config = load_config()
    profile_config = _resolve_play_profile(
        profile,
        micro_gates=micro_gates,
        single_turn_advisor=single_turn_advisor,
        parallel_review=parallel_review,
        advisor_contracts=advisor_contracts,
    )
    model_metadata: dict[str, str] = {}
    try:
        model_config = load_model_config(config.root_dir / "llm.config.json")
        model_metadata = {
            **{key: str(value) for key, value in describe_model(model_config).items()},
            **_profile_metadata(profile_config),
        }
        configure_langsmith(
            config,
            {
                **model_metadata,
                "eval_kind": "online-playtest",
                "turns": str(turns),
                "judge_mode": judge_mode,
                "per_call_timeout_seconds": str(per_call_timeout_seconds),
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
            profile=profile_config.name,
            runtime_budget_profile=profile_config.runtime_budget_profile,
            single_turn_advisor=profile_config.single_turn_advisor,
            micro_gates=profile_config.micro_gates,
            conditional_advisors=profile_config.conditional_advisors,
            parallel_review=profile_config.parallel_review,
            advisor_contracts=profile_config.advisor_contracts,
            session_id=session_id,
            ruleset_id=ruleset_id,
            scenario_id=scenario_id,
            output_dir=output_dir,
            model_metadata=model_metadata,
            cleanup_session=not keep_session,
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
    long_play = run_scripted_long_play(
        config,
        turns=long_play_turns,
        persist=False,
        cleanup_session=True,
    )
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


@release_app.command("build")
def release_build(
    name: Annotated[str, typer.Option("--name")] = "call-of-claw",
    ruleset_id: Annotated[str, typer.Option("--ruleset-id")] = "coc7_light_investigation",
    scenario_id: Annotated[str, typer.Option("--scenario-id")] = "black_tide_beacon",
    profile: Annotated[str, typer.Option("--profile")] = "balanced",
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("dist/releases"),
    force: Annotated[bool, typer.Option("--force")] = False,
    skip_executable: Annotated[
        bool,
        typer.Option(
            "--skip-executable",
            help="Build only the bundle files. Intended for packaging tests.",
        ),
    ] = False,
) -> None:
    """Build a shareable folder bundle with an executable, content, and config template."""
    config = load_config()
    if profile.strip().lower() not in PLAY_PROFILE_DEFAULTS:
        console.print("[red]--profile must be fast, balanced, or theatrical.[/red]")
        raise typer.Exit(2)
    spec = ReleaseBuildSpec(
        name=name,
        ruleset_id=ruleset_id,
        scenario_id=scenario_id,
        default_profile=profile.strip().lower(),
        output_dir=output_dir,
        force=force,
        build_executable=not skip_executable,
    )
    try:
        result = build_release_bundle(project_root=config.root_dir, spec=spec)
    except Exception as error:
        console.print(f"[red]release build failed:[/red] {error}")
        raise typer.Exit(1) from error
    console.print(f"bundle: {result.bundle_dir}", markup=False)
    console.print(f"archive: {result.archive_path}", markup=False)
    if result.executable_path:
        console.print(f"executable: {result.executable_path}", markup=False)
    console.print(
        "packages: " + ", ".join(result.included_package_ids),
        markup=False,
    )


@app.command("doctor")
def doctor(json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Check release/runtime configuration without calling the LLM provider."""
    config = load_config()
    checks = _doctor_checks(config)
    payload = {
        "root_dir": str(config.root_dir),
        "content_dir": str(config.content_dir),
        "data_dir": str(config.data_dir),
        "checks": checks,
    }
    has_error = any(check["status"] == "error" for check in checks)
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        table = Table("Status", "Check", "Detail")
        for check in checks:
            style = {
                "ok": "green",
                "warning": "yellow",
                "error": "red",
            }.get(str(check["status"]), "white")
            table.add_row(
                str(check["status"]).upper(),
                str(check["name"]),
                str(check["message"]),
                style=style,
            )
        console.print(table)
    if has_error:
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
    selected_ruleset = ruleset_id or _release_default_ruleset_id(config)
    selected_scenario = scenario_id or _release_default_scenario_id(config)
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
def session_list(
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List known playable sessions."""
    config = load_config()
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    sessions = store.list_session_summaries()
    if json_output:
        typer.echo(json.dumps({"sessions": sessions}, ensure_ascii=False, indent=2))
        return
    if not sessions:
        console.print("No sessions.")
        return
    for session in sessions:
        console.print(
            (
                f"{session['id']} "
                f"ruleset={session.get('ruleset_id') or '-'} "
                f"scenario={session.get('scenario_id') or '-'} "
                f"turns={session.get('turn_count', 0)} "
                f"memories={session.get('memory_count', 0)} "
                f"state={'yes' if session.get('has_state') else 'no'} "
                f"updated={session.get('updated_at')}"
            ),
            markup=False,
        )


@session_app.command("delete")
def session_delete(
    session_ids: Annotated[
        list[str] | None,
        typer.Option("--session-id", help="Session id to delete. May be repeated."),
    ] = None,
    all_sessions: Annotated[bool, typer.Option("--all")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Delete playable session data while preserving eval runs and exported files."""
    ids = list(dict.fromkeys(session_ids or []))
    if all_sessions and ids:
        console.print("[red]Use either --all or --session-id, not both.[/red]")
        raise typer.Exit(2)
    if not all_sessions and not ids:
        console.print("[red]Provide --session-id or --all.[/red]")
        raise typer.Exit(2)

    config = load_config()
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    existing = store.list_session_summaries()
    existing_ids = {str(session["id"]) for session in existing}
    targets = existing if all_sessions else [
        session for session in existing if str(session["id"]) in ids
    ]
    missing = (
        []
        if all_sessions
        else [session_id for session_id in ids if session_id not in existing_ids]
    )
    if missing and not json_output:
        console.print(f"Missing sessions: {', '.join(missing)}", markup=False)
    if not targets:
        payload = {"deleted": False, "reason": "no matching sessions", "missing": missing}
        if json_output:
            typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            console.print("No matching sessions to delete.", markup=False)
        return

    target_ids = [str(session["id"]) for session in targets]
    if not yes:
        label = "all sessions" if all_sessions else ", ".join(target_ids)
        confirmed = typer.confirm(
            f"Delete {label} and associated turns, memory, state, dice, advisor, critic, "
            "world patch, canon, and checkpoint data?",
            default=False,
        )
        if not confirmed:
            raise typer.Exit(1)

    delete_counts = (
        store.delete_all_sessions() if all_sessions else store.delete_sessions(target_ids)
    )
    checkpoint_counts = delete_turn_graph_checkpoints(
        config.sqlite_path,
        session_ids=None if all_sessions else target_ids,
        all_sessions=all_sessions,
    )
    payload = {
        "deleted": True,
        "all": all_sessions,
        "session_ids": target_ids,
        "missing": missing,
        "database": delete_counts,
        "checkpoints": checkpoint_counts,
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(f"Deleted sessions: {', '.join(target_ids)}", markup=False)
    console.print(
        "Database rows: "
        + ", ".join(f"{key}={value}" for key, value in delete_counts.items()),
        markup=False,
    )
    console.print(
        "Checkpoint rows: "
        + ", ".join(f"{key}={value}" for key, value in checkpoint_counts.items()),
        markup=False,
    )


@session_app.command("cleanup-tests")
def session_cleanup_tests(
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Delete known eval/playtest sessions while preserving eval runs and exports."""
    config = load_config()
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    targets = [
        str(session["id"])
        for session in store.list_session_summaries()
        if is_test_session_id(str(session["id"]))
    ]
    if not targets:
        payload = {"deleted": False, "reason": "no known test sessions"}
        if json_output:
            typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            console.print("No known test sessions to delete.", markup=False)
        return
    if not yes:
        confirmed = typer.confirm(
            f"Delete {len(targets)} known test sessions and checkpoint data?",
            default=False,
        )
        if not confirmed:
            raise typer.Exit(1)
    cleanup = cleanup_known_test_sessions(store=store, sqlite_path=config.sqlite_path)
    payload = {"deleted": True, **cleanup}
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(
        f"Deleted {len(cleanup.get('session_ids', []))} known test sessions.",
        markup=False,
    )
    console.print(json.dumps(cleanup, ensure_ascii=False, indent=2), markup=False)


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
    default_ruleset_id: str | None = None,
    default_scenario_id: str | None = None,
) -> tuple[str, str]:
    selected_ruleset = ruleset_id or default_ruleset_id
    selected_scenario = scenario_id or default_scenario_id
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


def _default_profile_name(config: AppConfig, explicit_profile: str | None) -> str:
    if explicit_profile:
        return explicit_profile
    if config.release_defaults and config.release_defaults.default_profile:
        return config.release_defaults.default_profile
    return "balanced"


def _release_default_ruleset_id(config: AppConfig) -> str | None:
    if config.release_defaults:
        return config.release_defaults.default_ruleset_id
    return None


def _release_default_scenario_id(config: AppConfig) -> str | None:
    if config.release_defaults:
        return config.release_defaults.default_scenario_id
    return None


def _build_play_model(
    *,
    config: AppConfig,
    profile_config: PlayProfileConfig,
    session_id: str,
    ruleset_id: str,
    scenario_id: str,
) -> tuple[BaseChatModel | None, dict[str, str]]:
    profile_metadata = _profile_metadata(profile_config)
    if not profile_config.use_llm:
        return None, profile_metadata
    try:
        model_config = load_model_config(config.root_dir / "llm.config.json")
    except (FileNotFoundError, ValueError) as error:
        _print_llm_config_error(config, error)
        raise typer.Exit(2) from error
    model_metadata = {
        **{key: str(value) for key, value in describe_model(model_config).items()},
        **profile_metadata,
    }
    configure_langsmith(
        config,
        {
            **model_metadata,
            "session_id": session_id,
            "ruleset_id": ruleset_id,
            "scenario_id": scenario_id,
        },
    )
    return build_chat_model(model_config), model_metadata


def _print_llm_config_error(config: AppConfig, error: Exception) -> None:
    console.print(f"[red]LLM config is not ready:[/red] {error}")
    console.print(
        "Create `llm.config.json` in the runtime root. In a release bundle, copy "
        "`llm.config.example.json` to `llm.config.json` and fill apiKey, baseURL, and model.",
        markup=False,
    )
    console.print(f"runtime root: {config.root_dir}", markup=False)


def _build_play_runtime_preload(
    *,
    registry: ContentRegistry,
    sqlite_path: Path,
    ruleset_id: str,
    scenario_id: str,
) -> dict[str, Any]:
    active_ids = registry.resolve_active_package_ids([ruleset_id, scenario_id])
    ruleset = load_compiled_ruleset(registry, ruleset_id)
    scenario = load_compiled_scenario(registry, scenario_id)
    plugin = load_rules_plugin(registry, ruleset_id)
    index_status = warm_content_index(registry, sqlite_path=sqlite_path)
    return {
        "ruleset_id": ruleset_id,
        "scenario_id": scenario_id,
        "active_package_ids": active_ids,
        "package_profiles": registry.package_profiles(active_ids),
        "compiled_ruleset": ruleset.model_dump(),
        "compiled_scenario": scenario.model_dump(),
        "rules_plugin": plugin.model_dump() if plugin is not None else None,
        "content_index": index_status,
    }


def _run_play_turn(
    graph: Any,
    *,
    player_input: str,
    session_id: str,
    ruleset_id: str,
    scenario_id: str,
    content_dir: Path,
    sqlite_path: Path,
    profile_config: PlayProfileConfig,
    model_metadata: dict[str, str],
    runtime_preload: dict[str, Any] | None = None,
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
        "play_profile": profile_config.name,
        "runtime_budget_profile": profile_config.runtime_budget_profile,
        "context_budget_mode": profile_config.context_budget_mode,
        "micro_gates_mode": profile_config.micro_gates,
        "single_turn_advisor_mode": profile_config.single_turn_advisor,
        "conditional_advisors_mode": profile_config.conditional_advisors,
        "parallel_review_mode": profile_config.parallel_review,
        "advisor_contract_mode": profile_config.advisor_contracts,
        "checkpoint_mode": "sqlite",
        "model_metadata": model_metadata,
    }
    if runtime_preload:
        state["runtime_preload"] = runtime_preload
    if progress is not None and progress.enabled:
        return stream_turn_graph(graph, state, on_node=progress.update)
    return invoke_turn_graph(graph, state)


def _play_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "kind": result.get("kind", "gm_turn"),
        "final_output": result.get("final_output", ""),
        "turn_plan": result.get("turn_plan", {}),
        "narration_plan": result.get("narration_plan", {}),
        "tool_results": result.get("tool_results", []),
        "trace_events": result.get("trace_events", []),
        "runtime_profile": result.get("runtime_profile", {}),
        "runtime_metadata": result.get("runtime_metadata", {}),
    }
    if result.get("manual_roll"):
        payload["manual_roll"] = result["manual_roll"]
    return payload


def _print_play_result(result: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(_play_result_payload(result), ensure_ascii=False, indent=2))
        return
    if result.get("manual_roll"):
        _print_manual_roll_result(result)
        return
    _print_gm_message(result.get("final_output", ""))


def _plain_text(value: Any) -> Text:
    text = str(value or "").strip()
    return Text(text if text else "(empty)")


def _print_panel(
    title: str,
    body: Any,
    *,
    border_style: str = "cyan",
    title_style: str = "bold cyan",
) -> None:
    console.print(
        Panel(
            _plain_text(body),
            title=Text(title, style=title_style),
            title_align="left",
            border_style=border_style,
            style="white",
            padding=(1, 2),
        )
    )


def _print_play_header(
    *,
    session_id: str,
    ruleset_id: str,
    scenario_id: str,
    profile_name: str,
) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(style="white")
    table.add_row("session", session_id)
    table.add_row("ruleset", ruleset_id)
    table.add_row("scenario", scenario_id)
    table.add_row("profile", profile_name)
    console.print(
        Panel(
            table,
            title="Call of Claw",
            title_align="left",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )


def _print_opening(text: str) -> None:
    _print_panel("开场", text, border_style="green", title_style="bold green")


def _print_gm_message(text: str) -> None:
    _print_panel("GM", text, border_style="cyan", title_style="bold cyan")


def _print_system_notice(text: str) -> None:
    console.print(Text(str(text), style="dim"))


def _print_warning(text: str) -> None:
    console.print(Text(str(text), style="yellow"))


def _print_manual_roll_result(result: dict[str, Any]) -> None:
    manual_roll = result.get("manual_roll") or {}
    lines = [
        f"表达式: {manual_roll.get('expression', '')}",
        f"结果: {manual_roll.get('rolls', [])}",
        f"总计: {manual_roll.get('total', '')}",
        "性质: 非规则判定",
    ]
    reason = manual_roll.get("reason")
    if reason:
        lines.insert(1, f"说明: {reason}")
    _print_panel("手动掷骰", "\n".join(lines), border_style="magenta", title_style="bold magenta")


def _print_exit_resume(session_id: str) -> None:
    body = Text(f"session-id: {session_id}\nresume: coc play --session-id {session_id}")
    console.print(Panel(body, title="退出", title_align="left", border_style="bright_black"))


def _run_interactive_play_loop(
    *,
    config: AppConfig,
    session_id: str | None,
    profile_config: PlayProfileConfig,
    progress_enabled: bool,
    json_output: bool,
    ruleset_id: str | None,
    scenario_id: str | None,
) -> None:
    registry = ContentRegistry.load(config.content_dir, config.root_dir)
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    resolved_session_id = session_id or f"play-{uuid.uuid4().hex[:8]}"
    existing_session = store.get_session(resolved_session_id) or {}
    selected_ruleset, selected_scenario = _resolve_play_package_ids(
        registry,
        ruleset_id=ruleset_id or existing_session.get("ruleset_id"),
        scenario_id=scenario_id or existing_session.get("scenario_id"),
        default_ruleset_id=_release_default_ruleset_id(config),
        default_scenario_id=_release_default_scenario_id(config),
    )
    model, model_metadata = _build_play_model(
        config=config,
        profile_config=profile_config,
        session_id=resolved_session_id,
        ruleset_id=selected_ruleset,
        scenario_id=selected_scenario,
    )
    runtime_preload = _build_play_runtime_preload(
        registry=registry,
        sqlite_path=config.sqlite_path,
        ruleset_id=selected_ruleset,
        scenario_id=selected_scenario,
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
    _print_play_header(
        session_id=resolved_session_id,
        ruleset_id=selected_ruleset,
        scenario_id=selected_scenario,
        profile_name=profile_config.name,
    )
    if not had_state or not had_turns:
        _print_opening(opening)
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
            _print_system_notice(f"resuming character: {character['name']}")
        _print_system_notice("resuming session. Type /help for commands.")

    with durable_turn_graph(sqlite_path=config.sqlite_path, model=model) as graph:
        while True:
            try:
                player_text = prompt_input.prompt("玩家", prompt_suffix=" > ")
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
            if command == "/roll" or command.startswith("/roll "):
                try:
                    result = _run_manual_roll_command(
                        store=store,
                        session_id=resolved_session_id,
                        ruleset_id=selected_ruleset,
                        scenario_id=selected_scenario,
                        command_text=text,
                    )
                except ValueError as error:
                    _print_warning(f"roll failed: {error}")
                    continue
                _print_play_result(result, json_output=json_output)
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
                        profile_config=profile_config,
                        model_metadata=model_metadata,
                        runtime_preload=runtime_preload,
                        progress=reporter,
                    )
            except Exception as error:
                console.print(f"[red]turn failed:[/red] {error}")
                continue
            _print_play_result(result, json_output=json_output)

    _print_exit_resume(resolved_session_id)


def _print_interactive_help() -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column(style="white")
    table.add_row("/help", "显示命令")
    table.add_row("/recap", "查看最近回合")
    table.add_row("/session", "查看当前 session 与角色")
    table.add_row("/roll <NdM>", "手动掷骰，不作为规则判定")
    table.add_row("/quit", "退出并显示恢复命令")
    console.print(Panel(table, title="命令", title_align="left", border_style="bright_black"))
    _print_system_notice("Type any other text as your character's action.")


def _run_manual_roll_command(
    *,
    store: SqliteStore,
    session_id: str,
    ruleset_id: str | None,
    scenario_id: str | None,
    command_text: str,
) -> dict[str, Any]:
    parts = command_text.strip().split(maxsplit=2)
    if len(parts) < 2 or parts[0].lower() != "/roll":
        raise ValueError("use /roll <NdM> [reason]")
    expression = parts[1]
    reason = parts[2].strip() if len(parts) > 2 else ""
    turn_index = len(store.list_turns(session_id)) + 1
    turn_id = f"{session_id}-manual-roll-{turn_index:03d}"
    roll_id = f"{turn_id}:roll:1"
    result = roll_dice_once(expression=expression, roll_id=roll_id, seed=session_id)
    manual_roll = {
        **result,
        "reason": reason,
        "authoritative": False,
    }
    reason_text = f"（{reason}）" if reason else ""
    final_output = (
        f"手动掷骰{reason_text}：{result['expression']} -> {result['rolls']}，"
        f"总计 {result['total']}。这不是规则判定结果。"
    )
    trace = {
        "kind": "manual_roll",
        "authoritative": False,
        "manual_roll": manual_roll,
        "trace_events": [
            {
                "node": "manual_roll_command",
                "expression": result["expression"],
                "roll_id": roll_id,
                "authoritative": False,
            }
        ],
    }
    store.upsert_session(
        session_id=session_id,
        ruleset_id=ruleset_id,
        scenario_id=scenario_id,
    )
    store.insert_turn(
        turn_id=turn_id,
        session_id=session_id,
        player_input=command_text,
        output=final_output,
        trace=trace,
    )
    store.insert_dice_roll(
        roll_id=roll_id,
        turn_id=turn_id,
        expression=result["expression"],
        result=manual_roll,
    )
    return {
        "kind": "manual_roll",
        "final_output": final_output,
        "turn_plan": {"decision": "answer", "tool_requests": []},
        "narration_plan": {},
        "tool_results": [],
        "trace_events": trace["trace_events"],
        "runtime_profile": {},
        "runtime_metadata": {},
        "manual_roll": manual_roll,
    }


def _normalize_advisor_contracts(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"legacy", "compact"}:
        console.print("[red]--advisor-contracts must be 'legacy' or 'compact'.[/red]")
        raise typer.Exit(2)
    return normalized


def _resolve_play_profile(
    profile: str,
    *,
    local: bool = False,
    use_llm: bool | None = None,
    micro_gates: bool | None = None,
    single_turn_advisor: bool | None = None,
    parallel_review: bool | None = None,
    advisor_contracts: str | None = None,
) -> PlayProfileConfig:
    normalized = profile.strip().lower()
    if normalized not in PLAY_PROFILE_DEFAULTS:
        console.print("[red]--profile must be fast, balanced, or theatrical.[/red]")
        raise typer.Exit(2)
    resolved = PLAY_PROFILE_DEFAULTS[normalized]
    if advisor_contracts is not None:
        resolved = replace(
            resolved,
            advisor_contracts=_normalize_advisor_contracts(advisor_contracts),
        )
    if use_llm is not None:
        resolved = replace(resolved, use_llm=use_llm)
    if micro_gates is not None:
        resolved = replace(resolved, micro_gates=micro_gates)
    if single_turn_advisor is not None:
        resolved = replace(resolved, single_turn_advisor=single_turn_advisor)
    if parallel_review is not None:
        resolved = replace(resolved, parallel_review=parallel_review)
    if local:
        resolved = replace(
            resolved,
            use_llm=False,
            micro_gates=False,
            single_turn_advisor=False,
            conditional_advisors=False,
            parallel_review=False,
            advisor_contracts="legacy",
            context_budget_mode="shadow",
        )
    return resolved


def _profile_metadata(profile_config: PlayProfileConfig) -> dict[str, str]:
    return {
        "profile": profile_config.name,
        "play_profile": profile_config.name,
        "runtime_budget_profile": profile_config.runtime_budget_profile,
        "context_budget_mode": profile_config.context_budget_mode,
        "use_llm": str(profile_config.use_llm),
        "micro_gates": str(profile_config.micro_gates),
        "single_turn_advisor": str(profile_config.single_turn_advisor),
        "conditional_advisors": str(profile_config.conditional_advisors),
        "parallel_review": str(profile_config.parallel_review),
        "advisor_contracts": profile_config.advisor_contracts,
    }


def _doctor_checks(config: AppConfig) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def add(status: str, name: str, message: str) -> None:
        checks.append({"status": status, "name": name, "message": message})

    add("ok", "runtime root", str(config.root_dir))

    if not config.content_dir.exists():
        add("error", "content directory", f"missing: {config.content_dir}")
        registry = None
    else:
        try:
            registry = ContentRegistry.load(config.content_dir, config.root_dir)
            issues = registry.validate()
        except Exception as error:
            add("error", "content registry", str(error))
            registry = None
        else:
            if issues:
                add("error", "content registry", "; ".join(issues))
            else:
                add("ok", "content registry", f"{len(registry.packages)} package(s)")

    try:
        config.data_dir.mkdir(parents=True, exist_ok=True)
        probe = config.data_dir / ".doctor-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception as error:
        add("error", "data directory", f"not writable: {error}")
    else:
        add("ok", "data directory", str(config.data_dir))

    if config.release_defaults is None:
        add("warning", "release config", "release.json not found; using CLI/default package order")
    else:
        release = config.release_defaults
        add("ok", "release config", str(release.path))
        if release.default_profile and release.default_profile not in PLAY_PROFILE_DEFAULTS:
            add("error", "release default profile", release.default_profile)
        elif release.default_profile:
            add("ok", "release default profile", release.default_profile)
        if registry is not None:
            if release.default_ruleset_id and release.default_ruleset_id not in registry.by_id:
                add("error", "release default ruleset", release.default_ruleset_id)
            elif release.default_ruleset_id:
                add("ok", "release default ruleset", release.default_ruleset_id)
            if release.default_scenario_id and release.default_scenario_id not in registry.by_id:
                add("error", "release default scenario", release.default_scenario_id)
            elif release.default_scenario_id:
                add("ok", "release default scenario", release.default_scenario_id)

    llm_config_path = config.root_dir / "llm.config.json"
    try:
        model_config = load_model_config(llm_config_path)
    except FileNotFoundError:
        add(
            "error",
            "llm config",
            "missing llm.config.json; copy llm.config.example.json and fill provider settings",
        )
    except Exception as error:
        add("error", "llm config", str(error))
    else:
        add(
            "ok",
            "llm config",
            f"{model_config.provider} model={model_config.model}",
        )

    return checks


def _print_session_recap(store: SqliteStore, session_id: str) -> None:
    turns = store.list_turns(session_id)[-5:]
    if not turns:
        _print_system_notice("No turns yet.")
        return
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan")
    table.add_column(style="white")
    for turn in turns:
        table.add_row("玩家", str(turn["input"]))
        table.add_row("GM", str(turn["output"]))
        table.add_row("", "")
    console.print(Panel(table, title="最近回合", title_align="left", border_style="cyan"))


def _print_interactive_session(store: SqliteStore, session_id: str) -> None:
    payload = _session_payload(store, session_id=session_id, include_gm=False)
    session_body = json.dumps(payload["session"], ensure_ascii=False, indent=2)
    _print_panel("Session", session_body, border_style="bright_black")
    character = payload.get("world_projection", {}).get("character_context", {})
    if character:
        character_body = json.dumps(character, ensure_ascii=False, indent=2)
        _print_panel("角色", character_body, border_style="green")


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

    _print_panel("角色创建", spec.intro.strip(), border_style="green", title_style="bold green")
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
    _print_panel("角色", summary, border_style="green", title_style="bold green")
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
            _print_warning("这个问题需要回答。")
            continue
        if question.numeric_range and answer:
            value = _parse_first_int(answer)
            lower, upper = question.numeric_range[0], question.numeric_range[1]
            if value is None or value < lower or value > upper:
                _print_warning(f"请输入 {lower}-{upper} 范围内的数字。")
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
            _set_nested_character_value(character_context, str(assignment.field), value)
            player_character[assignment.field] = value
    character_context["player_character"] = player_character
    return player_character, character_context


def _set_nested_character_value(target: dict[str, Any], field: str, value: Any) -> None:
    parts = [part for part in field.split(".") if part]
    if not parts:
        return
    cursor = target
    for part in parts[:-1]:
        nested = cursor.setdefault(part, {})
        if not isinstance(nested, dict):
            nested = {}
            cursor[part] = nested
        cursor = nested
    cursor[parts[-1]] = value


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
app.add_typer(release_app, name="release")
app.add_typer(session_app, name="session")


@app.command("play")
def play(
    player_input: Annotated[str | None, typer.Option("--input", "-i")] = None,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            help=(
                "fast, balanced, or theatrical. Defaults to release.json defaultProfile "
                "or balanced."
            ),
        ),
    ] = None,
    use_llm: Annotated[bool | None, typer.Option("--use-llm/--no-use-llm", hidden=True)] = None,
    micro_gates: Annotated[
        bool | None,
        typer.Option("--micro-gates/--no-micro-gates", hidden=True),
    ] = None,
    single_turn_advisor: Annotated[
        bool | None,
        typer.Option("--single-turn-advisor/--no-single-turn-advisor", hidden=True),
    ] = None,
    parallel_review: Annotated[
        bool | None,
        typer.Option("--parallel-review/--no-parallel-review", hidden=True),
    ] = None,
    advisor_contracts: Annotated[
        str | None,
        typer.Option("--advisor-contracts", hidden=True),
    ] = None,
    local: Annotated[bool, typer.Option("--local")] = False,
    progress: Annotated[bool, typer.Option("--progress/--no-progress")] = True,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    ruleset_id: Annotated[str | None, typer.Option("--ruleset-id")] = None,
    scenario_id: Annotated[str | None, typer.Option("--scenario-id")] = None,
) -> None:
    """Run one turn, or enter an interactive play loop when --input is omitted."""
    config = load_config()
    profile_config = _resolve_play_profile(
        _default_profile_name(config, profile),
        local=local,
        use_llm=use_llm,
        micro_gates=micro_gates,
        single_turn_advisor=single_turn_advisor,
        parallel_review=parallel_review,
        advisor_contracts=advisor_contracts,
    )
    if player_input is None:
        _run_interactive_play_loop(
            config=config,
            session_id=session_id,
            profile_config=profile_config,
            progress_enabled=progress,
            json_output=json_output,
            ruleset_id=ruleset_id,
            scenario_id=scenario_id,
        )
        return

    registry = ContentRegistry.load(config.content_dir, config.root_dir)
    store = SqliteStore(config.sqlite_path)
    store.migrate()
    resolved_session_id = session_id or "default"
    existing_session = store.get_session(resolved_session_id) or {}
    selected_ruleset, selected_scenario = _resolve_play_package_ids(
        registry,
        ruleset_id=ruleset_id or existing_session.get("ruleset_id"),
        scenario_id=scenario_id or existing_session.get("scenario_id"),
        default_ruleset_id=_release_default_ruleset_id(config),
        default_scenario_id=_release_default_scenario_id(config),
    )
    command_text = player_input.strip()
    if command_text.lower() == "/roll" or command_text.lower().startswith("/roll "):
        try:
            result = _run_manual_roll_command(
                store=store,
                session_id=resolved_session_id,
                ruleset_id=selected_ruleset,
                scenario_id=selected_scenario,
                command_text=command_text,
            )
        except ValueError as error:
            console.print(f"[red]roll failed:[/red] {error}")
            raise typer.Exit(2) from error
        _print_play_result(result, json_output=json_output)
        return
    model, model_metadata = _build_play_model(
        config=config,
        profile_config=profile_config,
        session_id=resolved_session_id,
        ruleset_id=selected_ruleset,
        scenario_id=selected_scenario,
    )
    runtime_preload = _build_play_runtime_preload(
        registry=registry,
        sqlite_path=config.sqlite_path,
        ruleset_id=selected_ruleset,
        scenario_id=selected_scenario,
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
                    profile_config=profile_config,
                    model_metadata=model_metadata,
                    runtime_preload=runtime_preload,
                    progress=reporter,
                )
    except Exception as error:
        console.print(f"[red]play failed:[/red] {error}")
        if profile_config.use_llm:
            console.print("Retry with `--local` to run the local fallback graph.")
        raise typer.Exit(1) from error
    _print_play_result(result, json_output=json_output)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
