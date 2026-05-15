# 通用 Call of Claw 从零重建计划

## 1. Summary

从零重建一个 **通用 TRPG 系统**，不是 Lasers & Feelings 系统，也不是《水晶别唱了》专用 GM。

Lasers & Feelings 和《水晶别唱了》只作为最小 smoke test content package，用来验证接口、编译、评测和游玩闭环。系统内部提示词、agent 能力、工具协议、记忆系统和评测系统都不能与这两个测试用例耦合。

技术栈默认使用：

- Python 3.12+
- LangChain
- LangGraph
- LangSmith
- Pydantic
- SQLite
- pytest
- uv

核心原则：

- GM agent 的核心提示词只描述通用 GM 规范。
- 任何具体 TRPG 规则、剧本要求、世界观约束，都通过 content package 编译成 extension 后接入。
- LLM 做语义判断和叙事计划；程序做确定性动作。
- 所有状态变化必须来自 deterministic tool 和 append-only canon event。
- 所有长期质量判断必须能通过 LangSmith trace、dataset、experiment、evaluator 复现。

## 2. Non-Goals

- 不迁移旧 demo 架构。
- 不把当前 `data/rules.json`、`data/scenario.json` 当作新系统事实来源。
- 不把 Lasers & Feelings 的规则写进核心 GM prompt。
- 不把《水晶别唱了》的 NPC、调频器、水晶、姆姆等写进核心 GM prompt。
- 第一版不做 UI。
- 第一版不接向量库；先用 SQLite FTS，保留接口。

## 3. Project Shape

新项目结构建议：

```text
coc/
  pyproject.toml
  README.md
  docs/
    rebuild-from-scratch-plan.md
    architecture.md
    quality-system.md
  src/coc/
    app/
      cli.py
      config.py
    langchain/
      models.py
      prompts.py
      tools.py
      tracing.py
      structured.py
    graph/
      state.py
      build_turn_graph.py
      build_compile_graph.py
      build_eval_graph.py
      nodes/
        input.py
        retrieval.py
        adjudication.py
        tools.py
        narration.py
        memory.py
        persistence.py
    content/
      registry.py
      packages.py
      retrieval.py
      compiler.py
      visibility.py
    rules/
      base.py
      resolver_runtime.py
    scenario/
      runtime.py
      disclosure.py
    memory/
      store.py
      projection.py
      canon.py
    tools/
      dice.py
      patches.py
      content.py
      memory.py
    eval/
      datasets.py
      regression.py
      playtest.py
      judge.py
      scorecard.py
  content/
    agent_skills/
    rulesets/
    scenarios/
    evaluators/
  seeds/
    rule.md
    scenario.md
    canon-log.jsonl
  tests/
```

最小依赖：

- `langchain`
- `langgraph`
- `langsmith`
- `pydantic`
- `typer`
- `rich`
- `pytest`
- `ruff`
- LLM provider package, through a provider-neutral adapter

## 4. LangChain Responsibilities

全面使用 LangChain 作为应用层，而不是只把它当 LLM 调用器。

LangChain 负责：

- Provider adapter:
  - 统一 Anthropic、OpenAI、兼容 OpenAI API、兼容 Anthropic API。
  - 配置模型名、base URL、temperature、timeout、retry。
- Prompt management:
  - 通用 GM prompt。
  - rules compiler prompt。
  - scenario compiler prompt。
  - extension compiler prompt。
  - evaluator prompt。
- Tool interface:
  - 所有 deterministic tools 用 LangChain tool 包装。
  - tool 输入输出使用 Pydantic schema。
- Structured output:
  - `TurnPlan`
  - `NarrationPlan`
  - `MemoryExtraction`
  - `RulesetCompilation`
  - `ScenarioCompilation`
  - `EvalScorecard`
- Callbacks and tracing:
  - 所有 chain、tool、graph node 都接 LangSmith tracing。
  - 每个 trace 写入 session、turn、package、ruleset、scenario、prompt version、graph node metadata。
- Runnable composition:
  - 编译链、检索链、judge 链用 LangChain Runnable 组合。
  - graph node 内部可以调用 Runnable，但副作用必须进入 deterministic tool。

## 5. LangGraph Responsibilities

LangGraph 负责长流程、有状态、多节点、可恢复执行。

使用三个 graph：

- Turn graph:
  - 单轮玩家输入到 GM 输出。
- Compile graph:
  - 规则/剧本/extension 编译。
- Eval graph:
  - 回归测试、agent playtest、judge review、scorecard 归档。

必须启用 checkpoint，并使用 `thread_id`。

遵守 durable execution 约束：

- 随机性必须封装在 task/tool：骰子不能在 LLM 节点或普通 Python 函数里临时生成后丢失。
- 副作用必须封装在 task/tool：写 DB、写 canon、写 eval result、外部 API 调用不能散落在 node 内。
- resume/replay 时不能重复掷骰、重复写 canon、重复写 memory。

Turn graph:

```text
receive_input
-> load_session
-> retrieve_memory
-> discover_relevant_packages
-> retrieve_content_spans
-> classify_player_intent
-> adjudicate_fictional_authority
-> plan_resolution
-> execute_deterministic_tools
-> narrate_result
-> extract_canon_and_memory
-> persist_turn
-> emit_trace
```

Compile graph:

```text
load_source_documents
-> classify_package_kind
-> extract_public_structure
-> extract_private_structure
-> compile_extensions
-> validate_references
-> generate_regression_examples
-> persist_compiled_package
```

Eval graph:

```text
load_dataset
-> run_subject_agent
-> collect_transcript_and_trace
-> run_deterministic_evaluators
-> run_llm_judge
-> aggregate_scorecard
-> persist_experiment_result
```

## 6. LangSmith Quality System

LangSmith 是开发过程中的一等系统，不是后期可选项。

使用方式：

- Tracing:
  - 每次 CLI play、test、eval 都开启 trace。
  - trace 必须展示 prompt、retrieved spans、structured output、tool calls、state changes。
- Datasets:
  - `scripted_regression`
  - `canon_log_replay`
  - `agent_playtest_seed`
  - `compiler_examples`
  - `boundary_cases`
- Experiments:
  - 每次 prompt 改动跑 regression experiment。
  - 每次 rules compiler 改动跑 compiler experiment。
  - 每次 retrieval/disclosure 改动跑 leakage experiment。
- Evaluators:
  - deterministic evaluator：schema、tool call、state diff、visibility。
  - LLM judge evaluator：GM 质量、叙事、能动性、连贯性。
  - paired comparison：新旧版本 transcript 对比。
- Metadata:
  - `prompt_version`
  - `ruleset_id`
  - `scenario_id`
  - `extension_ids`
  - `retrieval_policy`
  - `model_provider`
  - `model_name`
  - `git_sha` when available

质量门槛：

- regression dataset 必须通过。
- judge scorecard 关键维度不得低于阈值。
- 任一 privacy/disclosure 泄露测试失败，阻断合并。
- 任一 deterministic replay 不一致，阻断合并。

## 7. Generic GM Core Prompt Policy

核心 GM prompt 只包含通用职责：

- 维护已建立事实。
- 不替玩家做选择。
- 不泄露未公开秘密。
- 不把玩家提案误判成已经成立的事实。
- 不自行发放规则奖励、线索、物品或状态变化。
- 风险且不确定时请求已加载 ruleset resolver。
- 明显行动不掷骰。
- 越权声明要 boundary，并给出可玩改写。
- 叙事必须服从 tool result、dice result、canon、loaded rules。
- 如果当前上下文不足，提出澄清问题或请求更多 content span。

核心 GM prompt 禁止包含：

- 特定规则术语，例如 Lasers、Feelings、Sanity、Luck、DC、技能等级。
- 特定剧本实体，例如 姆姆、水晶、调频器、维加。
- 特定世界观默认事实。
- 特定游戏系统的掷骰算法。

任何具体规则或剧本要求必须来自：

- compiled ruleset extension
- compiled scenario extension
- retrieved content spans
- deterministic tool result
- canon projection

## 8. Content And Extension Model

四种 package：

- `agent_skill`
  - 通用 GM 能力。
  - 例如 adjudication、memory management、scenario running、playtest judging。
- `ruleset`
  - 具体规则文本和编译产物。
  - 例如 Lasers & Feelings、Call of Cthulhu、Fate。
- `scenario`
  - 剧本、场景、NPC、线索、秘密、GM 指令。
- `extension`
  - 编译后接入系统的能力。
  - 例如某规则的 sanity resolver、某剧本的特殊 GM move。

每个 package 有 manifest：

```yaml
schema_version: 1
id: string
kind: agent_skill | ruleset | scenario | extension | evaluator
name: string
description: string
version: string
entrypoint: string
visibility:
  default: public | gm_only | tool_only
dependencies: []
extensions: []
references: []
tests: []
```

Ruleset 编译产物必须提供：

- 规则摘要。
- 术语索引。
- 判定接口定义。
- deterministic resolver hooks。
- 可调用 tools。
- GM constraints。
- 示例裁判。
- regression examples。

Scenario 编译产物必须提供：

- public info。
- GM-only secrets。
- scene index。
- NPC index。
- clue index。
- clock/threat index, if present。
- disclosure policy。
- scenario-specific GM extension。
- scenario regression examples。

## 9. Runtime State

`GraphState` 至少包含：

- `session_id`
- `thread_id`
- `turn_id`
- `player_input`
- `ruleset_id`
- `scenario_id`
- `active_extension_ids`
- `world_projection`
- `recent_canon`
- `retrieved_spans`
- `memory_hits`
- `intent`
- `authority_result`
- `turn_plan`
- `tool_requests`
- `tool_results`
- `narration_plan`
- `final_output`
- `trace_refs`

关键 structured schemas：

- `IntentClassification`
- `AuthorityResult`
- `TurnPlan`
- `ToolRequest`
- `ToolResult`
- `NarrationPlan`
- `CanonEventDraft`
- `MemoryExtraction`
- `EvalScorecard`

## 10. Deterministic Tools

所有 tools 必须 schema-first：

- `search_content`
- `load_content_span`
- `roll_dice`
- `run_ruleset_resolver`
- `apply_world_patch`
- `write_canon_event`
- `project_world_state`
- `recall_memory`
- `upsert_memory`
- `write_eval_result`

工具规则：

- LLM 只能请求工具，不能模拟工具结果。
- `roll_dice` 必须可 replay。
- `apply_world_patch` 必须校验 patch、版本、引用。
- `load_content_span` 必须执行 visibility policy。
- `write_canon_event` 是状态事实的唯一持久来源。

## 11. Storage

SQLite first：

- `sessions`
- `turns`
- `canon_events`
- `world_projection`
- `memories`
- `packages`
- `compiled_extensions`
- `retrieved_spans`
- `dice_rolls`
- `eval_runs`
- `eval_scores`

Canon 是 append-only。

World state 是 canon projection，不是 LLM 叙事副产品。

Long-term memory 分为：

- episodic：发生过什么。
- semantic：稳定事实。
- procedural：玩家偏好、GM 风格、常用 house rules。

第一版用 SQLite FTS；向量库后续扩展。

## 12. Development Phases

### Phase 1: Scaffold

- 建 Python 项目。
- 配置 `uv`、`pytest`、`ruff`。
- 建 Typer CLI。
- 建配置加载。
- 建 SQLite migration。
- 建 LangSmith tracing 开关。

CLI：

- `coc content check`
- `coc compile ruleset`
- `coc compile scenario`
- `coc play`
- `coc eval regression`
- `coc eval playtest`

### Phase 2: LangChain Foundation

- 实现 model registry。
- 实现 prompt registry。
- 实现 structured output wrapper。
- 实现 tool registry。
- 实现 LangSmith callback metadata。
- 实现 fake model/test model，用于 no-LLM 单测。

### Phase 3: Content Registry

- 实现 package manifest。
- 实现引用校验。
- 实现 visibility policy。
- 实现 package dependency resolution。
- 实现 content span loader。

### Phase 4: Rule Compiler

- 输入 rules Markdown。
- 输出 ruleset package 和 compiled extension。
- 编译结果包括 resolver interface、术语索引、GM constraints、test examples。
- 编译过程本身进入 LangSmith trace。
- 编译结果必须通过 schema validation 和 regression examples。

### Phase 5: Scenario Compiler

- 输入 scenario Markdown。
- 输出 scenario package。
- 明确区分 public、GM-only、tool-only。
- 剧本特定 GM 要求编译为 scenario extension。
- 编译结果必须通过 disclosure tests。

### Phase 6: Generic GM Turn Graph

- 实现通用 turn graph。
- 接入 content retrieval。
- 接入 ruleset resolver extension。
- 接入 deterministic tools。
- 接入 canon projection。
- 输出完整 trace。

### Phase 7: Minimal Smoke Test Content

- 把 Lasers & Feelings 作为第一个 ruleset package。
- 把《水晶别唱了》作为第一个 scenario package。
- 只用于验证最小闭环，不污染 core agent。

### Phase 8: Evaluation System

- 建 LangSmith datasets。
- 实现 deterministic evaluators。
- 实现 LLM judge evaluator。
- 实现 scorecard 和 turn-level findings。
- 实现 experiment runner。

### Phase 9: Agent Playtest

- player agent 风格：
  - 谨慎型。
  - 破坏型。
  - 规则询问型。
  - 叙事型。
  - 边界挑战型。
- judge agent 读取 transcript、trace、loaded spans、tool results 后评分。

### Phase 10: Hardening

- checkpoint resume 不重复 tool side effects。
- replay 可复现骰点。
- prompt/version/package 变更可比较 eval 分数。
- regression failure 必须定位到 node、prompt、tool 或 package。

## 13. Test Plan

Unit tests：

- package manifest validation。
- visibility policy。
- structured output schema。
- tool idempotency。
- canon projection。
- dice replay。
- memory namespace。

Compiler tests：

- rules compiler 不把规则塞进 core prompt。
- scenario compiler 能区分 public 和 GM-only。
- extension 注册后可被检索和调用。
- 编译产物能生成 regression examples。

Runtime tests：

- 普通查询不掷骰。
- 计划/询问不误判为越权声明。
- 越权声明 boundary。
- 风险行动调用 resolver。
- 叙事不改 tool result。
- GM-only secret 不被泄露。

Evaluation tests：

- scripted regression 全过。
- LangSmith experiment 自动上传 traces。
- judge finding 必须绑定 turn id。
- paired comparison 可比较两个版本。

Smoke test content：

- Lasers & Feelings +《水晶别唱了》验证最小闭环。
- 后续至少再加一个不同结构的规则/剧本，验证没有隐式耦合。

## 14. Initial Regression Cases

第一批测试用例应包括：

- “维加告诉我这个调频器怎么用”不能误判规则查询。
- “询问姆姆能发出抵消广播的声音吗”不能 boundary。
- “我会带它离开”应视为承诺/提案，不因“我会”直接 boundary。
- “刚刚维加说什么”应从 canon 回答，不掷骰。
- “使用魔法控制全宇宙”必须 boundary。
- 普通装备/状态/规则查询不掷骰。
- 等待/什么都不做应触发 GM move 或压力推进。
- 创造性可行方案应进入行动裁判。
- LLM 叙事不得改骰点、改 band、私自发奖励或状态。
- 核心 prompt 搜索不到任何测试规则或测试剧本专名。
- 未公开 GM-only secret 不得出现在玩家输出。

## 15. Acceptance Criteria

- 删除旧 demo 后，系统可从 `content/` 和 `seeds/` 重建。
- 核心 GM prompt 搜索不到任何测试规则或测试剧本专名。
- `coc content check` 通过。
- `coc compile ruleset` 和 `coc compile scenario` 可生成 extensions。
- `coc play` 可运行任意绑定 ruleset/scenario 的 session。
- `coc eval regression` 通过第一批测试。
- LangSmith 中能看到每个 turn 的完整 trace、tool call、retrieved span 和 structured decision。
- 每个状态变化都能追溯到 deterministic tool 和 canon event。
- checkpoint replay 不重复掷骰、不重复写事件。

## 16. Assumptions

- 默认 Python 实现；若以后改 Node.js，架构不变。
- LangGraph 负责长流程和持久执行。
- LangChain 负责模型、prompt、tool、structured output、callbacks。
- LangSmith 负责 tracing、dataset、experiment、evaluation。
- 第一版不做 UI。
- 第一版不做向量库，先用 SQLite FTS。
- 旧 demo 可以删除；只保留 `rule.md`、`scenario.md`、`data/canon-log.jsonl` 作为 seeds。

## 17. References

- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph durable execution: https://docs.langchain.com/oss/python/langgraph/durable-execution
- LangChain structured output: https://docs.langchain.com/oss/python/langchain/structured-output
- LangSmith observability: https://docs.langchain.com/oss/python/langchain/observability
