# nano_jev — a copy-anywhere Jev stand-in (deterministic coarse classifier)

> English | [中文](README.zh-CN.md)

This directory turns the idea of "give a quick coarse verdict first, escalate the hard
cases later" into a **zero-dependency, copy-anywhere** standalone service. It drops
straight into [jev-arena](https://github.com/NanmiCoder/jev-arena) as a **drop-in
replacement for Jev** on the left lane.

## What it is, and what it isn't (read this first)

| | Description |
| --- | --- |
| **Is** | A deterministic lexicon + heuristic **coarse classifier**: text and structured questions in, labels on a controlled vocabulary out. A single judgment runs in the micro-to-millisecond range, is **reproducible**, and never touches the network or a model. |
| **Isn't** | A semantic model. It scores by keyword hits, so its **accuracy ceiling is clearly below a real model** (long sentences, sarcasm, and implicit meaning in English/Chinese will be misjudged). It is positioned as "screen cheaply first" and must **not** be used as a curated conclusion. |

It shares the same philosophy as the sibling project `nano/` (a **geometric numeric
kernel**, a float32 C++ acceleration layer) but sits on a **different side and is not
interchangeable**:

* `nano/` answers "how large is this quantity, roughly?";
* `nano_jev` answers "what category does this content roughly belong to?".

This project is **zero-dependency**: no third-party packages, no dependence on `nano/`'s
C++ kernel. It can be copied wholesale onto any machine with **Python 3.9+** and run
standalone (verified on 3.9).

## Quick start

Two ways, pick one:

**A. Run directly as a script** (copy/clone the whole folder first, then run from its
**parent directory** so `nano_jev` becomes an importable package):

```bash
cd ..            # so the current directory can see the nano_jev/ folder
python3 -m nano_jev.nj_server serve --port 8765

# health check
curl -s http://127.0.0.1:8765/health

# single coarse classification
python3 -m nano_jev.nj_server label "这个模型回答准确，速度快，但价格偏贵"

# replay a decisions request (from a file or stdin)
python3 -m nano_jev.nj_server decide --file req.json
```

**B. Install as a package** (`pyproject.toml` is configured; installing adds a
`nano-jev` command):

```bash
pip install -U "setuptools>=61"   # the [project] table needs setuptools>=61; older versions silently build an empty package
pip install .                     # or pip install -e . for an editable install
nano-jev --version                # confirm which version is installed
nano-jev serve --port 8765        # equivalent to python -m nano_jev.nj_server
```

The wheel ships only the package body (14 modules + `py.typed`), not `tests/`.


## Wiring into jev-arena (two ways)

### Way 1: the native Jev protocol (left lane)

jev-arena's Jev backend reads the environment variable `OPENROUTER_BASE_URL` and appends
`/alpha/decisions` to it. Point the base URL at this service and **jev-arena needs zero
code changes**:

```bash
export OPENROUTER_BASE_URL=http://127.0.0.1:8765   # no longer point at openrouter.ai
```

It derives labels itself by calling `normalizeLabel` on the `answers` in the response.

### Way 2: the OpenAI-compatible protocol (right lane or a custom lane)

In the web UI's "model settings", configure one side as `openai` protocol, set `baseUrl`
to `http://127.0.0.1:8765/v1`, and `model` to `nano-jev`. The service extracts comments
from the user messages and returns a standard chat completion whose `content` is
`{"labels": [...]}`.

This side also aligns with official OpenAI in two ways, so most clients plug in directly:

* **Streaming**: with `"stream": true`, it switches to `text/event-stream`, pushing
  `chat.completion.chunk` frames, ending with `data: [DONE]`.
* **Candidate count `n`**: accepts `n` (1–8). Since this service is deterministic, the
  `n` candidates carry identical content; this is shape compatibility only.
* **Model discovery**: `GET /v1/models` lists models, `GET /v1/models/nano-jev` returns a single one.

## Cascading to an LLM (Hybrid)

`nano_jev` remains an independent, zero-dependency coarse classifier. The cascade
orchestrator accepts an injected LLM callback and does not bind to any provider or SDK.
By default, it escalates labels with heuristic confidence below `0.65` or `mixed`
sentiment. This score is **not a calibrated probability**; treat the default threshold as
a starting point and tune it against labeled examples from the target domain.

```python
from nano_jev import HybridPolicy, Label, judge_hybrid

def llm_judge(text: str, nano_label: Label) -> Label:
    # Call the LLM here and map its controlled structured output to Label.
    # The coarse label is a hint; the returned model label is authoritative.
    ...

result = judge_hybrid(
    "c0",
    "This model answers well at times, but it also makes things up.",
    llm_judge=llm_judge,
    policy=HybridPolicy(minimum_confidence=0.65),
)
print(result.source, result.escalation_reasons, result.label.to_raw())
```

If escalation is required but no callback was supplied, `judge_hybrid` raises
`RuntimeError` rather than presenting an unreviewed coarse result as an LLM judgment.
Model-call errors propagate by default; set `HybridPolicy(on_llm_error="nano")` to fall
back to the coarse result and record the exception type. Each result records its source,
whether it was escalated, the coarse label, LLM latency, and the escalation reasons,
making accuracy, escalation rate, latency, and token cost measurable.

Model output is not trusted as-is: `sentiment` / `intent` / `aspects` / `emotion` are
normalized to the controlled vocabulary, out-of-range `confidence` / `sentiment_score`
values are clamped, and an `evidence_quote` that is not a verbatim substring of the source
falls back to the coarse extractor's quote. Every rewrite is recorded in
`HybridResult.normalizations` and `label.meta["hybrid"]`.

Cost and batching:

```python
from nano_jev import HybridBudget, judge_many

budget = HybridBudget(limit=100)          # shared across requests
results = judge_many(
    [{"comment_id": "c0", "content": "..."}, {"comment_id": "c1", "content": "..."}],
    llm_judge=llm_judge,
    budget=budget,
)
# On exhaustion the default is to return the coarse result with a
# "budget_exhausted" reason; use HybridPolicy(on_budget_exhausted="raise")
# to fail loudly instead.
```

The default route escalates on low confidence or `mixed` sentiment. Enable
`minimum_content_length`, `escalate_irrelevant`, and `escalate_without_aspects` to widen
the escalation surface for higher-risk domains.

```text
Input → nano_jev coarse judgment → lightweight routing policy
                                    ├─ low risk → return coarse label
                                    └─ ambiguous/conflicting → injected LLM → structured label
```

### Input guard (PII / injection)

```python
from nano_jev import GuardPolicy, judge_hybrid

result = judge_hybrid(
    "c0", text,
    llm_judge=judge,
    guard=GuardPolicy(max_chars=4000, injection_action="block"),  # flag / block / raise
)
```

| Capability | Behavior |
| --- | --- |
| PII redaction | Emails, phone numbers, national IDs, bank cards, and common API keys become `[EMAIL]`-style placeholders before the model sees them |
| Checksum filtering | Chinese IDs are validated per GB 11643-1999 **including the birth-date segment**, bank cards by Luhn; **numbers that fail validation are left untouched** |
| Number precision | Mainland mobiles only (`1[3-9]`, optional `86`); international numbers require an explicit `+`; landlines require an area-code separator |
| Truncation | With `max_chars > 0`, the text is truncated and flagged as `truncated` |
| Injection detection | Matches "ignore previous instructions", "reveal your system prompt", and Chinese equivalents; `flag` records, `block` skips escalation, `raise` raises `PromptInjectionError` |
| Wrapping | Content is wrapped in a `<content>` block with delimiters escaped |
| Configurable | `pii_kinds` selects built-in rules; `extra_rules` adds `CustomRule` (regex + optional validator) |

Checksums buy **fewer false positives**: `2024010112345678` (an order id),
`1700000000` (a timestamp), and IDs with a bad check digit are no longer redacted.
The same applies to IDs whose birth date cannot exist (month 13, Feb 30,
Feb 29 in a non-leap year): those can still produce a valid check digit, so a
checksum alone would have redacted them.
The cost is **misses**: real data with a wrong check digit or an international number
without `+` is not detected. For "redact before sending to a model", a miss is safer
than corrupting the text.

```python
from nano_jev import CustomRule, GuardPolicy
import re

policy = GuardPolicy(
    pii_kinds=("email", "phone", "id_number"),
    extra_rules=(
        CustomRule("member", re.compile(r"M\d{8}"), "[MEMBER]",
                   validator=lambda v: v.endswith("7")),
    ),
)
```

### Label contract validation

`nj_validate` answers "is this label valid?", which is deliberately separate from
normalization: normalization runs at request time (converge and continue), validation
runs at **boundaries** (ingress, datasets, cross-service responses) and **never rewrites
data**.

```python
from nano_jev import validate_label, validate_label_mapping

issues = validate_label(label, source_text=original_text, strict_quote=True)
issues = validate_label_mapping(raw_dict)      # raw dict, no normalization
```

It checks enum membership, finite in-range scores, duplicate or excess aspects/emotions,
`is_relevant` being a bool, the quote being a verbatim substring, and `meta` being a
mapping. `require_valid_label` raises `ValueError` listing every problem.

Findings are recorded under `label.meta["hybrid"]["guard"]`. This is **not a security
boundary**: it only catches known phrasing. The real defenses remain controlled-vocabulary
output validation and never executing model output.

### Field-level escalation and arbitration

Escalating a whole label wastes budget when only one column needs review.
`review_fields` adopts only those model fields and keeps the rest coarse;
`conflict_policy` decides who wins on disagreement:

```python
from nano_jev import HybridPolicy

policy = HybridPolicy(
    review_fields=("sentiment", "aspects"),
    conflict_policy="prefer_model",   # or "prefer_coarse"
)
```

Each result records `fields` / `adopted` / `conflicts` under `label.meta["arbitration"]`.
There is no confidence averaging: neither side's score is a calibrated probability.

### Cache and concurrency

```python
from nano_jev import HybridCache, HybridBudget, judge_many

cache = HybridCache(namespace="local-model/prompt-v1", maxsize=2048, ttl_s=3600)
results = judge_many(
    comments, llm_judge=judge, cache=cache, budget=HybridBudget(limit=100), workers=8,
)
```

* `namespace` is caller-supplied (prefer `<model>/<prompt version>`); an empty namespace
  disables caching so a model or prompt change cannot silently reuse stale verdicts.
* Only escalated results are cached; hits return deep copies. `stats()` reports hit rate,
  entry count, and evictions.
* `judge_many(workers=N)` calls the model concurrently while preserving input order;
  `HybridBudget.consume()` is atomic, so concurrency never overspends.
* **Identical concurrent requests are coalesced** (single-flight): the first one calls
  the model and the rest wait for and reuse its result; `stats()["coalesced"]` counts them.

## Hosting the cascade (optional)

```bash
python3 -m nano_jev.nj_server serve \
    --hybrid-endpoint http://127.0.0.1:8000/v1 \
    --hybrid-model local-model \
    --hybrid-guard block \
    --hybrid-cache-namespace local-model/v1
```

* Without an endpoint the service stays **fully deterministic**: no network, no model.
* With one, `/alpha/decisions` and `/v1/chat/completions` (including streaming) both
  coarse-screen first and escalate by policy.
* `/health` reports the `hybrid` switch and cache statistics.
* `NANO_JV_LLM_ENDPOINT` / `NANO_JV_LLM_MODEL` / `NANO_JV_LLM_API_KEY` are also accepted;
  `--hybrid-timeout` controls the per-call timeout.
* Error semantics: injection blocked under `raise` → **400**; escalation budget
  exhausted → **429**; model output violating the contract → **502**; anything else → 500.
* `/health` exposes `metrics` with label count, escalation count, and model errors.
* Rate limits: `--max-concurrent` (in-flight model calls), `--budget` (global escalation
  quota), `--per-client-budget` (by `X-Client-Id` or source address). Over-limit requests
  get an immediate 429 + `Retry-After` instead of queueing; `/health` reports `limits`.

See [`examples/llm_judge_openai.py`](examples/llm_judge_openai.py) for library wiring:
`OpenAICompatJudge` uses only stdlib `urllib` and plugs straight into `judge_hybrid`.

## Evaluation: threshold sweep and guard impact

```bash
# Sweep confidence thresholds; runs hybrid / brain only
.venv/bin/python scripts/eval_hybrid.py --sweep 0.3,0.5,0.65,0.8,0.9

# Enable the input guard too (block: injected samples are not escalated)
.venv/bin/python scripts/eval_hybrid.py --guard block
```

Rows are labelled `mode@threshold`, so you can read the trade-off directly. On the sample
dataset the controller route (brain) stays at a 30% escalation rate across 0.3–0.9, while
the fixed threshold climbs from 20% to 100% — the payoff of routing by evidence and value.

## HTTP API

| Method | Path | Description |
| --- | --- | --- |
| POST | `/alpha/decisions` | Native Jev contract: `{model, state:{comments}, questions}` → `{answers, usage, …}` |
| POST | `/v1/chat/completions` | OpenAI-compatible: `{model, messages}` → standard chat completion; `"stream": true` returns `text/event-stream` incremental frames |
| GET | `/health` | Health check, version, hybrid switch, cache stats, and cascade counters |
| GET | `/v1/models` | OpenAI-style model list |
| GET | `/v1/models/{model}` | A single model object; unknown model returns 404 |

Contract errors always return `{"error": {"message", "code"}}` (same shape as Jev's
errors). The `usage.cost` in responses is always `0.0` — a local service incurs no
provider fees, so `0` means "free", not "unknown".

Error codes at a glance:

| Status | When |
| --- | --- |
| 400 | Body is not valid JSON; `Content-Length` is not an integer or is negative |
| 401 | A token is set but `Authorization` does not match |
| 404 | Unknown path |
| 405 | Known path but wrong method; response carries an `Allow` header. **No longer** falls back to Python's default HTML 501 |
| 413 | Body exceeds 32 MiB |
| 429 | Escalation budget exhausted (`--hybrid-endpoint` mode only) |
| 500 | Internal error; details go only to server stderr, the client gets a generic message — no leaking exception text |
| 502 | Model output violated the label contract (bad JSON, out-of-range field) |

`OPTIONS` returns 204 + `Allow` for method discovery.

The request body is read according to `Content-Length`; an invalid length (non-integer or
negative) returns 400, and over 32 MiB returns 413. In both cases the body is left unread,
so the response carries `Connection: close` to avoid residual bytes polluting the next
request on the same keep-alive connection. A chunked streaming response writes its
terminating chunk whether it ends normally or fails midway, so clients never hang waiting
for the stream to finish. Binding to a non-loopback address (e.g. `--host 0.0.0.0`)
without a token prints a warning to stderr on startup — otherwise it would silently expose
an unauthenticated service to the LAN.

## Command line

```bash
python -m nano_jev.nj_server serve [--host 127.0.0.1] [--port 8765]
                                   [--topics 模型,产品] [--token <access token>]
                                   [--hybrid-endpoint URL] [--hybrid-model ID]
                                   [--hybrid-guard off|flag|block|raise]
                                   [--hybrid-cache-namespace NS] [--hybrid-cache-size N]
                                   [--hybrid-cache-ttl SECONDS]
                                   [--max-concurrent N] [--budget N]
                                   [--per-client-budget N] [--retry-after SECONDS]
python -m nano_jev.nj_server label "一段文本" [--id c0] [--stdin] [--topics ...]
python -m nano_jev.nj_server decide [--file req.json] [--topics ...]
python -m nano_jev.nj_server --version
```

`label`'s positional argument and `--stdin` are mutually exclusive; passing both errors
out (exit code 2) instead of silently preferring stdin.

`--topics` (or the environment variable `NANO_JV_TOPICS`) swaps the topic words used for
the "relevance" criterion, moving the coarse screen to another comment domain (default:
AI models/products).

`--token` (or the environment variable `NANO_JV_TOKEN`) enables authentication: once set,
every request must carry `Authorization: Bearer <token>`, and a mismatch returns 401.
If neither is set, there is no authentication (local use). Command-line arguments take
precedence over environment variables.

## Module overview

| File | Responsibility |
| --- | --- |
| `nj_vocab.py` | Controlled vocabulary and normalization, aligned item-by-item with jev-arena `vocab.mjs` |
| `nj_lexicon.py` | Bilingual (Chinese/English) polarity/intent/aspect/emotion/topic cues — the only place to extend |
| `nj_engine.py` | Coarse classifier: one text → one set of structured labels (pure function, deterministic) |
| `nj_hybrid.py` | Cascade orchestration: coarse screen → lightweight routing → optional LLM review (provider-agnostic) |
| `nj_guard.py` | Input guard: PII redaction, injection detection, untrusted-content wrapping |
| `nj_arbitrate.py` | Field-level escalation and conflict arbitration |
| `nj_cache.py` | Thread-safe LRU cache (namespace + TTL) |
| `nj_llm.py` | OpenAI-compatible judge (stdlib urllib) and label construction |
| `nj_limits.py` | Serving limits: concurrency slots, global and per-client quotas |
| `nj_validate.py` | Label contract validation (raw dicts and `Label` objects) |
| `examples/` | Vendor adapter example (not shipped in the wheel) |
| `nj_contract.py` | Parsing and response construction for the Jev `/alpha/decisions` contract |
| `nj_openai.py` | OpenAI `/chat/completions` adapter layer (including streaming SSE) |
| `nj_server.py` | Zero-dependency HTTP service and command-line entry point |
| `tests/` | pytest tests: pure-function layer + a real-server HTTP layer |
| `pyproject.toml` | Packaging metadata + pytest / ruff / mypy configuration |

## Architecture and data flow

### Module relationships

Dependencies are one-way: **entry point → contract/adapter → engine → vocabulary**. The
two serving contract chains (native Jev, OpenAI-compatible) don't reference each other;
they share the same deterministic core `nj_engine.classify()`. `nj_hybrid` is an optional
library-level orchestrator: it uses that core and calls an injected LLM only when routing
rules request escalation. The cascade modules participate only when explicitly invoked:
`nj_guard` (input guard) → `nj_hybrid` (routing) → `nj_arbitrate` (field arbitration),
with `nj_cache` and `nj_llm` as optional cross-cutting pieces. Serving stays deterministic
unless `--hybrid-endpoint` is configured.

```mermaid
graph LR
    subgraph entry["Entry point"]
        SERVER["nj_server.py<br/>HTTP service + CLI"]
        INIT["__init__.py<br/>VERSION"]
    end

    subgraph contract["Contract & adapter"]
        CONTRACT["nj_contract.py<br/>Jev /alpha/decisions contract"]
        OPENAI["nj_openai.py<br/>OpenAI /chat/completions adapter"]
    end

    ENGINE["nj_engine.py<br/>coarse classifier classify()"]
    HYBRID["nj_hybrid.py<br/>cascade routing"]
    GUARD["nj_guard.py<br/>PII / injection guard"]
    ARB["nj_arbitrate.py<br/>field arbitration"]
    CACHE["nj_cache.py<br/>result cache"]
    LLMADAPTER["nj_llm.py<br/>OpenAI-compatible callback"]
    LLM["Caller-provided LLM"]

    subgraph vocab["Vocabulary & constants"]
        VOCAB["nj_vocab.py<br/>controlled vocabulary & normalization"]
        LEXICON["nj_lexicon.py<br/>Chinese/English cue lexicon"]
    end

    SERVER --> CONTRACT
    SERVER --> OPENAI
    SERVER --> ENGINE
    SERVER --> INIT
    CONTRACT --> ENGINE
    CONTRACT --> VOCAB
    CONTRACT --> LEXICON
    OPENAI --> ENGINE
    HYBRID --> ENGINE
    HYBRID --> GUARD
    HYBRID --> ARB
    HYBRID --> CACHE
    LLMADAPTER --> HYBRID
    HYBRID -. "escalated cases only" .-> LLM
    ENGINE --> VOCAB
    ENGINE --> LEXICON
```

### Request handling flow

One `POST` from arrival to response: authenticate, parse JSON, route by path to one of the
two contract chains; both chains hand every comment to the same `classify()` for per-comment
coarse classification, then assemble JSON in each contract's shape. **No network, no model
call anywhere**, so identical input always yields an identical response.

```mermaid
flowchart TD
    START(["POST enters _Handler.do_POST"]) --> AUTH{"NANO_JV_TOKEN set?"}
    AUTH -- "yes & token mismatch" --> E401["401 unauthorized"]
    AUTH -- "passed / no token" --> JSON{"body valid JSON?"}
    JSON -- "no" --> E400["400 contract error"]
    JSON -- "yes" --> ROUTE{"path"}

    ROUTE -- "/alpha/decisions" --> HD["nj_contract.handle_decisions"]
    ROUTE -- "/v1/chat/completions" --> HC["nj_openai.handle_chat"]
    ROUTE -- "other" --> E404["404 unknown path"]

    subgraph P1["native Jev chain"]
        HD --> PD["parse_decisions_request()"]
        PD --> LC["label_comments() → classify()"]
        LC --> BA["build_answers() translate by question role"]
        BA --> BR["build_decisions_response()"]
    end

    subgraph P2["OpenAI-compatible chain"]
        HC --> EX["extract_comments() pull comments"]
        EX --> CL["classify()"]
        CL --> BC["build_chat_completion()"]
    end

    BR --> RESP(["JSON response (cost always 0.0)"])
    BC --> RESP
```

## Testing

Tests live in `nano_jev/tests/` (**not** the package root: the root is the repo root, and
leaving tests there would get them bundled into the wheel by setuptools). As with running
the service, execute from the **parent directory** of `nano_jev/`:

```bash
python3 -m pytest nano_jev/ -q
```

Or run layer by layer: engine/contract/adapter are pure-function tests (no socket), while
the server tests **really start a server and make real HTTP requests**:

```bash
python3 -m pytest nano_jev/tests/test_nj_engine.py nano_jev/tests/test_nj_contract.py \
                nano_jev/tests/test_nj_openai.py nano_jev/tests/test_nj_hybrid.py \
                nano_jev/tests/test_nj_guard.py nano_jev/tests/test_nj_arbitrate.py \
                nano_jev/tests/test_nj_cache.py nano_jev/tests/test_nj_llm.py -q
python3 -m pytest nano_jev/tests/test_nj_server.py -q
```

After installing, a plain `pytest` also works (configured in `pyproject.toml`).

## Development and quality configuration

`pyproject.toml` holds the packaging and tooling configuration (all targeting the same
`py39`):

```bash
pip install -e ".[dev]"     # installs pytest / ruff / mypy

ruff check .                # static checks (select E,F,W,I,N,UP,B,SIM,RUF,A)
mypy                        # type checks (no third-party deps, near-zero noise)
pytest                      # unit + end-to-end (real-server HTTP tests)
```

`.github/workflows/ci.yml` runs the same set on every push / PR (3.9–3.13 matrix), plus
two gate tasks: **build the wheel and assert its contents** (no test files in the wheel,
`py.typed` must be present) and **install from the wheel and hit the live server once**
(auth 401, method 405, decisions produce answers).

## Limitations (stated plainly)

1. **The criterion is a lexicon, not understanding**: sarcasm, long sentences, implicit
   meaning, and new words not in the lexicon will all be misjudged.
2. **Chinese substring matching has false hits**: "好" matches "好久不见" and "卡" matches
   "卡片". Chinese has no word boundaries and this subsystem deliberately avoids
   tokenization, so this cost is **kept on purpose** — acceptable for a coarse screen, not
   for a curated conclusion.
3. **Chinese first, English second**: the lexicon is predominantly Chinese; English cues
   are coarser.
4. **English matches word boundaries, but inflection coverage is limited**: pure-ASCII
   terms match on English word boundaries (`ai` no longer matches `chairman`/`rain`, `app`
   no longer matches `apple`), and common inflections like `models`/`crashed`/`slower` are
   tolerated; but irregular forms like `like`→`liking` (drop-e + ing) and new abbreviations
   not in the lexicon will still be missed.
5. **Keep the vocabulary in sync with upstream**: when jev-arena changes `vocab.mjs`,
   update `nj_vocab.py` accordingly, otherwise
   `tests/test_nj_contract.py::test_vocab_matches_jev_arena` will turn red.
