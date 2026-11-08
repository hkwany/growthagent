# growthagent — 自主增长 Agent

一个**依赖极轻、可离线运行**的 LLM Agent：给定一个产品/任务，Agent 自主完成
「规划 → 调研 → 生成创意 → 建 Campaign → 读指标 → 给出预算/优化建议 → 输出报告」；
以"自主数字员工(Autonomous Digital Employee / AI Agent)"的形式进行工作；覆盖 LLM / Agent / Tool Use / Memory /
RAG / Evaluation / ML(bandit) / 云原生

**核心运行时依赖只有 `numpy`**，其余（OpenAI / Ollama / FastAPI / gRPC /
tiktoken / hnswlib）都是可选、惰性加载。默认 `scripted` provider 不需要任何 API key，
下面的快速开始和测试是**零配置、零网络**的，之后可以自主接入真实的 LLM

---

## 快速开始

```bash
cd growthagent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 只有 numpy

# 端到端 demo（agent 全流程 + 评测 + bandit + RAG/记忆）
python examples/run_growth_agent.py

# 全部单元测试
python -m unittest discover -s tests -v
```

demo 会依次输出：
① 一次完整 agent 运行的 trace 和最终报告；
② 5 个 golden task的评测结果；
③ bandit 离线模拟（随机 vs epsilon-greedy）；
④ RAG 检索与长期记忆召回。

## 接入真实 LLM

`growthagent/llm.py` 直接用 `urllib` 调 OpenAI 兼容接口和 Ollama，**不依赖 SDK**。

OpenAI 兼容（含 deepseek / moonshot 等任何 `/chat/completions` 端点）：

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # 可换成兼容端点
export OPENAI_MODEL=gpt-4o-mini
```

本地 Ollama：

```bash
export LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen2.5:7b
```

然后：

```python
from growthagent.app.service import GrowthAgentService
svc = GrowthAgentService()
run = svc.run("为 FocusApp 启动一个增长投放活动", write_trace=True, write_ledger=True)
print(run.final_answer)
```

## 启动 HTTP 服务

```bash
pip install -r requirements-optional.txt
uvicorn growthagent.app.main:app --reload
```

端点：`GET /health`、`POST /agent/run`（JSON）、`POST /agent/run/stream`（SSE 流式）。

## 项目结构

```
growthagent/
├── growthagent/            # Python 包
│   ├── llm.py              # provider 抽象：Scripted / OpenAI / Ollama
│   ├── schema.py           # Message / ToolCall / LLMResponse / TraceEvent
│   ├── config.py           # env 配置
│   ├── tokenizer.py        # token 估算（tiktoken 可选）
│   ├── vector.py           # HashEmbedder + 余弦 VectorStore（hnswlib 可选）
│   ├── tools/              # @tool 装饰器 + 5 个工具（含 mock 广告平台）
│   ├── agent/              # loop（plan→act→observe→reflect）+ context + planner/reflector
│   ├── memory/             # scratchpad + 长期向量记忆
│   ├── rag/                # 检索器
│   ├── optimizer/          # epsilon-greedy bandit + 预算分配 + 离线模拟
│   ├── eval/               # golden task + 工具调用精确匹配 + judge
│   ├── observability/      # JSONL trace + token/cost 账本
│   ├── app/                # service 核心 + FastAPI
│   └── rpc/                # 工具 RPC（proto + local + gRPC 桩）
├── tests/                  # unittest（全离线）
├── examples/               # 端到端 demo
├── deploy/                 # Dockerfile + k8s manifests
└── .github/workflows/      # CI（compile + unittest + demo）
```

## 本地端到端测试结果
python examples/run_growth_agent.py
python -m unittest discover -s tests -v
```bash
$ python examples/run_growth_agent.py

========================================================================
1) Autonomous growth agent run (scripted LLM)
========================================================================
task: Launch a growth campaign for FocusApp (deep-work productivity timer).
terminated: completed  steps: 5  model: scripted
tokens: 2890  latency: 0.6ms

trace:
  [llm_call] tools=['fetch_url'] usage={'prompt_tokens': 120, 'completion_tokens': 20, 'total_tokens': 140}
  [tool_call] {'name': 'fetch_url', 'arguments': {'url': 'https://example.com/landing/focusapp'}}
  [tool_result] fetch_url -> {"url": "https://example.com/landing/focusapp", "title": "FocusApp landing page", "content": "Hero: 'Get 2x more deep wo
  [llm_call] tools=['search'] usage={'prompt_tokens': 400, 'completion_tokens': 25, 'total_tokens': 425}
  [tool_call] {'name': 'search', 'arguments': {'query': 'productivity app ads'}}
  [tool_result] search -> {"query": "productivity app ads", "results": [{"title": "Top-performing productivity app ad creatives", "snippet": "UGC-
  [llm_call] tools=['create_campaign'] usage={'prompt_tokens': 600, 'completion_tokens': 40, 'total_tokens': 640}
  [tool_call] {'name': 'create_campaign', 'arguments': {'objective': 'app_installs', 'budget': 1000.0, 'creative': 'Get 2x more deep work — 15s vertical demo, CTA in first 3s.'}}
  [tool_result] create_campaign -> {"campaign_id": "camp_1", "status": "active"}
  [llm_call] tools=['get_metrics'] usage={'prompt_tokens': 700, 'completion_tokens': 25, 'total_tokens': 725}
  [tool_call] {'name': 'get_metrics', 'arguments': {'campaign_id': 'camp_1', 'spend': 500.0}}
  [tool_result] get_metrics -> {"campaign_id": "camp_1", "spend": 500.0, "impressions": 781250, "clicks": 8125, "ctr": 1.04, "cpc": 0.64, "conversions"
  [llm_call] tools=[] usage={'prompt_tokens': 900, 'completion_tokens': 60, 'total_tokens': 960}
  [finish] {'terminated': 'completed', 'steps': 5, 'tool_calls': 4}

final answer:
Created campaign camp_1 (app_installs, $1000 budget) with a benefit-led creative. After $500 spend the campaign is live and generating metrics. Recommendation: keep the benefit-led creative, hold budget, and re-evaluate after 50 conversions for statistical significance.

========================================================================
2) Eval harness over golden tasks
========================================================================
tasks: 5  success_rate: 1.000  avg_cost_tokens: 0.0  avg_latency_ms: 0.1

  calc                 success=True  tools=['calculator'] judge_score=100
  search               success=True  tools=['search'] judge_score=100
  create_campaign      success=True  tools=['create_campaign'] judge_score=100
  campaign_and_metrics success=True  tools=['create_campaign', 'get_metrics'] judge_score=100
  calc_then_search     success=True  tools=['calculator', 'search'] judge_score=100

========================================================================
3) Bandit optimizer (random vs epsilon-greedy)
========================================================================
horizon=500 arms=4 trials=200
  random (equal split)  final cumulative reward: 213.02
  epsilon-greedy        final cumulative reward: 365.36
  gap vs optimal: 38.09%

========================================================================
4) RAG retrieval + long-term memory
========================================================================
RAG query: 'how to make an ad creative for a productivity app'
  -> focusapp (score=0.449): FocusApp is a deep-work productivity timer for developers. B
  -> budget (score=0.325): Allocate ad budget toward campaigns with the highest recent
memory recall: 'what worked for the productivity app'
  -> mem_0 (score=0.456): Campaign for FocusApp used benefit-led creative and got ROAS

========================================================================
DONE
========================================================================
All components ran successfully (no API key required).
```

## 本地全部单元测试结果
```bash
$ python -m unittest discover -s tests -v
test_completes_with_tool_then_answer (test_agent.AgentLoopTest.test_completes_with_tool_then_answer) ... ok
test_error_termination (test_agent.AgentLoopTest.test_error_termination) ... ok
test_max_steps_termination (test_agent.AgentLoopTest.test_max_steps_termination) ... ok
test_on_event_callback (test_agent.AgentLoopTest.test_on_event_callback) ... ok
test_tool_result_fed_back_to_model (test_agent.AgentLoopTest.test_tool_result_fed_back_to_model) ... ok
test_trace_event_types (test_agent.AgentLoopTest.test_trace_event_types) ... ok
test_usage_accumulates (test_agent.AgentLoopTest.test_usage_accumulates) ... ok
test_invalid_arms (test_bandit.BanditTest.test_invalid_arms) ... ok
test_select_and_update (test_bandit.BanditTest.test_select_and_update) ... ok
test_observe_shifts_budget (test_bandit.BudgetOptimizerTest.test_observe_shifts_budget) ... ok
test_recommend_splits_total (test_bandit.BudgetOptimizerTest.test_recommend_splits_total) ... ok
test_bandit_beats_random (test_bandit.SimulateTest.test_bandit_beats_random) ... ok
test_over_budget (test_context.ContextManagerTest.test_over_budget) ... ok
test_trim_keeps_system_task_recent (test_context.ContextManagerTest.test_trim_keeps_system_task_recent) ... ok
test_long_truncated (test_context.SummarizeTest.test_long_truncated) ... ok
test_short_unchanged (test_context.SummarizeTest.test_short_unchanged) ... ok
test_fail_wrong_tools (test_eval.EvaluateTest.test_fail_wrong_tools) ... ok
test_success (test_eval.EvaluateTest.test_success) ... ok
test_all_golden_tasks_pass (test_eval.RunEvalTest.test_all_golden_tasks_pass) ... ok
test_precision (test_eval.ToolCallPrecisionTest.test_precision) ... ok
test_messages_to_openai (test_llm.MessageFormatTest.test_messages_to_openai) ... ok
test_parse_tool_calls (test_llm.MessageFormatTest.test_parse_tool_calls) ... ok
test_openai_requires_key (test_llm.ProviderFactoryTest.test_openai_requires_key) ... ok
test_unknown_provider_raises (test_llm.ProviderFactoryTest.test_unknown_provider_raises) ... ok
test_exhausted_returns_empty (test_llm.ScriptedLLMTest.test_exhausted_returns_empty) ... ok
test_pops_in_order (test_llm.ScriptedLLMTest.test_pops_in_order) ... ok
test_remember_and_recall (test_memory_rag.LongTermMemoryTest.test_remember_and_recall) ... ok
test_index_and_search (test_memory_rag.RetrieverTest.test_index_and_search) ... ok
test_json_roundtrip (test_memory_rag.ScratchpadTest.test_json_roundtrip) ... ok
test_set_get_delete (test_memory_rag.ScratchpadTest.test_set_get_delete) ... ok
test_serializable (test_observability.RunToJsonlTest.test_serializable) ... ok
test_ledger_file (test_observability.TokenLedgerTest.test_ledger_file) ... ok
test_report_totals (test_observability.TokenLedgerTest.test_report_totals) ... ok
test_write_and_parse (test_observability.TraceWriterTest.test_write_and_parse) ... ok
test_call_json (test_service_rpc.LocalToolServiceTest.test_call_json) ... ok
test_unknown_tool (test_service_rpc.LocalToolServiceTest.test_unknown_tool) ... ok
test_run_summary (test_service_rpc.ServiceTest.test_run_summary) ... ok
test_run_with_scripted_llm (test_service_rpc.ServiceTest.test_run_with_scripted_llm) ... ok
test_cjk (test_tokenizer.TokenizerTest.test_cjk) ... ok
test_empty (test_tokenizer.TokenizerTest.test_empty) ... ok
test_latin (test_tokenizer.TokenizerTest.test_latin) ... ok
test_messages (test_tokenizer.TokenizerTest.test_messages) ... ok
test_basic (test_tools.CalculatorTest.test_basic) ... ok
test_rejects_attributes (test_tools.CalculatorTest.test_rejects_attributes) ... ok
test_rejects_import (test_tools.CalculatorTest.test_rejects_import) ... ok
test_tool_returns_result (test_tools.CalculatorTest.test_tool_returns_result) ... ok
test_ad_platform_flow (test_tools.MockToolsTest.test_ad_platform_flow) ... ok
test_analytics_deterministic (test_tools.MockToolsTest.test_analytics_deterministic) ... ok
test_fetch_known_and_unknown (test_tools.MockToolsTest.test_fetch_known_and_unknown) ... ok
test_search_returns_results (test_tools.MockToolsTest.test_search_returns_results) ... ok
test_default_registry_has_tools (test_tools.RegistryTest.test_default_registry_has_tools) ... ok
test_execute_calculator (test_tools.RegistryTest.test_execute_calculator) ... ok
test_execute_unknown_tool_returns_error (test_tools.RegistryTest.test_execute_unknown_tool_returns_error) ... ok
test_shape_and_normalization (test_vector.HashEmbedderTest.test_shape_and_normalization) ... ok
test_similar_texts_closer (test_vector.HashEmbedderTest.test_similar_texts_closer) ... ok
test_add_and_search (test_vector.VectorStoreTest.test_add_and_search) ... ok
test_dim_mismatch_raises (test_vector.VectorStoreTest.test_dim_mismatch_raises) ... ok
test_empty_store (test_vector.VectorStoreTest.test_empty_store) ... ok

----------------------------------------------------------------------
Ran 58 tests in 0.072s

OK
```