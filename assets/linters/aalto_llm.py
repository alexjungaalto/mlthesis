#!/usr/bin/env python3
"""Shared LLM client for the thesis linters.

Default gateway: the Aalto LLM Gateway (LLMGW_BASE_URL below; OpenAI-style
chat completions, ``Authorization: Bearer $AALTO_LLM_KEY``; Aalto
network/VPN only; default model ``zai-org/GLM-5.3-Flash``). Switched
2026-09-22, the day Aalto Scientific Computing announced GLM-5.3-Flash on
the gateway: the models run on Aalto hardware, nothing but token counts is
logged, the service is free, and GLM-5.3-Flash stays warm and honors JSON
mode — the best fit for an unpublished draft. Caveat: the gateway is a
PILOT until 2026-10-15, so its quota and model list can change at any time.

Every alternative stays one ``--base-url`` / ``LLM_BASE_URL`` away, and
LLMClient picks protocol and key from the URL: the Aalto AI API v2
(AIAPI_V2_URL; chat completions, Bearer $AALTO_API_KEY, gemini-2.5-flash,
multimodal — the suite default 2026-09-17 to 2026-09-22 and the fallback
if the pilot ends), the Responses API of that endpoint (AIAPI_V2_URL +
"/responses"; the GPT-5 family), the deprecating Azure gateway
(AALTO_AZURE_URL, Ocp-Apim-Subscription-Key auth) and any OpenAI-style
endpoint such as OpenRouter ($OPENROUTER_API_KEY).

TWO THINGS MAKE A PAPER-SIZED PROMPT WORK ON THIS ENDPOINT, both measured
on 2026-09-22 against a 99k-char linter prompt that otherwise failed:

* Chat-completions requests are STREAMED. The model can go quiet for
  minutes on a long prompt, and the gateway's ingress then kills the idle
  connection with a 504 — every retry died the same way. An SSE response
  keeps bytes moving and the request survives. ``_read_sse`` reassembles
  the deltas into the shape a non-streamed answer has, so callers see no
  difference. This also covers the v2 gateway's matching 504 on a long
  gemini-2.5-pro call (2026-09-17).
* The reasoning pass is OFF by default for THINKING_MODELS (see
  ``_thinking_on``). On a 61k-char paper GLM-5.3-Flash streamed ~8.7k
  reasoning deltas in 65s and spent its whole 9k-token budget without
  emitting one character of answer. With thinking off the same linter
  finishes in ~9s with sound, specific findings. ``LLM_THINKING=on``
  restores it for short prompts, where it does converge.

A CUT-OFF answer is still re-requested with a larger budget: a reasoning
model's thinking tokens count against max_tokens (on ai-api-v2 a 50-token
smoke test already returned finish_reason "length"), so a long JSON report
can be truncated and become unparseable. With thinking on, models in
THINKING_MODELS get a fixed headroom on top of the caller's budget the way
the Responses branch does for the GPT-5 family, and complete() retries with
twice, then four times the budget and says so on stderr (ported from the
dictionary suite's llm_client, fb15e37).

Every ``*_llm`` linter builds its client with ``make_client()`` and picks a
model with ``default_model()`` / ``default_vision_model()`` from this one
module, so a user configures the whole suite through a small set of
environment variables (each can also be overridden by a CLI flag):

  LLM_BASE_URL      which endpoint to talk to; decides everything below.
                    Unset -> the Aalto LLM Gateway (LLMGW_BASE_URL).
  LLM_MODEL         text model id; unset -> a per-endpoint default.
  LLM_VISION_MODEL  model id for image requests (figure_lint_llm.py).
  LLM_VISION_BASE_URL  endpoint for image requests; unset -> the Aalto AI
                    API v2, because the gateway's vision pod is unreliable
                    (see default_vision_base_url).
  LLM_THINKING      "on"/"off" for a reasoning model's thinking pass on the
                    chat-completions protocol. Unset -> off for
                    THINKING_MODELS (it does not converge on a linter-sized
                    prompt; see above).
  AALTO_API_KEY     key for the Aalto AI API, v2 (default) and the
                    deprecating Azure gateway (also accepts
                    AALTO_OPENAI_API_KEY).
  AALTO_LLM_KEY     key for the Aalto LLM Gateway.
  OPENROUTER_API_KEY  key for any other OpenAI-style endpoint.

Which KEY is used follows from LLM_BASE_URL, not the other way round: the
helpers below inspect the URL, then read the matching variable (falling back
to reading it out of the user's shell profile, see ``_key_from_shell_profile``).

CHANGING THE MODEL. Pass a model id with ``--model`` on any linter (or
``run_all_linters.py --model <id>``), or export ``LLM_MODEL``; ``--vision-model``
/ ``LLM_VISION_MODEL`` sets the model figure_lint_llm.py uses for images. The
valid ids depend on the endpoint (``--base-url`` / ``LLM_BASE_URL``). Which ids
each Aalto service offers changes over time — these are the authoritative
sources, and the values below were verified on 2026-09-22 (gateway) and
2026-09-17 (AI API v2):

  * Aalto LLM Gateway (default endpoint,
    ``https://llm-gateway.k8s.aalto.fi/api/v1``). Open-weight models served
    on Aalto hardware; the live list is a GET on ``/models`` (the id string
    is what you pass to ``--model``). Sep-2026 set:
    ``zai-org/GLM-5.3-Flash`` (the suite default: strongest of the set, and
    the one kept warm), ``Qwen/Qwen3.8-27B-FP8``, ``openai/gpt-oss-120b``,
    ``RedHatAI/gemma-4-31B-it-FP8-Dynamic``,
    ``Qwen/Qwen3-30B-A3B-Instruct-2507-FP8``,
    ``Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8``,
    ``Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`` (vision, the suite's vision
    default here), ``Qwen/Qwen3-VL-30B-A3B-Thinking-FP8``,
    ``google/gemma-4-E4B-it``, ``google/codegemma-7b-it``. Everything but
    GLM-5.3-Flash scales to zero and answers 503 while its pod spins up —
    the client retries those. Docs + key self-service UI (Aalto VPN):
    https://scicomp.aalto.fi/aalto/llm-web-apis/
  * Aalto AI API v2 (``--base-url`` AIAPI_V2_URL,
    ``https://ai-api-v2.aalto.fi/v1``).
    Chat completions serve the Gemini family — ``gemini-2.5-flash`` (the
    suite default, text and vision), ``gemini-2.5-pro``,
    ``gemini-3.x-flash`` ids — plus ``gpt-4.1-*`` dated ids; a GET on
    ``/models`` lists the live set (verified 2026-09-17). The GPT-5 family
    sits behind ``/responses`` and was 429 quota-blocked at migration.
    Catalogue + keys (Aalto login):
    https://www.aalto.fi/en/services/aalto-ai-apis
  * Aalto AI API, old Azure gateway (``--base-url`` AALTO_AZURE_URL;
    deprecating). GPT-5 family, dated ids: ``gpt-5-2025-08-07``,
    ``gpt-5-mini-2025-08-07``, ``gpt-5-nano-2025-08-07``.

Both Aalto services require the Aalto network / VPN. Nothing beyond a key is
needed for the default endpoint:

  export AALTO_LLM_KEY=...    # made at https://llm-gateway.k8s.aalto.fi/
  python3 run_all_linters.py thesis.pdf --llm

Example — fall back to the Aalto AI API v2 (e.g. once the pilot ends):

  export AALTO_API_KEY=...    # https://www.aalto.fi/en/services/aalto-ai-apis
  python3 run_all_linters.py thesis.pdf --llm \
      --base-url https://ai-api-v2.aalto.fi/v1 --model gemini-2.5-flash

Stdlib urllib only — no SDK needed.
"""

import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Optional, Tuple

LLMGW_BASE_URL = "https://llm-gateway.k8s.aalto.fi/api/v1"  # default: Aalto-hosted open weights
AIAPI_V2_URL = "https://ai-api-v2.aalto.fi/v1"   # chat completions; + "/responses" for GPT-5 / gpt-oss
AALTO_AZURE_URL = "https://aalto-openai-apigw.azure-api.net/v1/openai/responses"  # deprecating
BASE_URL = os.environ.get("LLM_BASE_URL", LLMGW_BASE_URL)

BASE_URL_HELP = ("LLM endpoint (default: the Aalto LLM Gateway, "
                 f"{LLMGW_BASE_URL}). Alternatives: the Aalto AI API v2 "
                 f"({AIAPI_V2_URL}), its Responses API "
                 f"({AIAPI_V2_URL}/responses, GPT-5 family), the deprecating "
                 f"Azure gateway ({AALTO_AZURE_URL}) or any OpenAI-style "
                 "endpoint such as https://openrouter.ai/api/v1.")
API_KEY_HELP = ("API key (default: $AALTO_LLM_KEY for the Aalto LLM Gateway, "
                "$AALTO_API_KEY for the Aalto AI API v2 and the old Azure "
                "gateway, $OPENROUTER_API_KEY otherwise).")

# Models that reason before answering on the chat-completions protocol, where
# those thinking tokens are billed against max_tokens (unlike the Responses
# API, which budgets them separately). They get THINKING_HEADROOM extra tokens
# so a report-sized JSON answer is not truncated before the model starts
# writing it; complete()'s escalation loop still covers the rest.
THINKING_MODELS = ("GLM-", "Qwen3-VL-30B-A3B-Thinking", "gpt-oss")
THINKING_HEADROOM = 8000


_ANNOT_CACHE: Optional[str] = None

# Only the SEMANTIC linters — the ones judging content, clarity, and framing —
# benefit from the reviewer's margin comments. Injecting the annotation block
# into the structural/mechanical linters (forward-ref, type-consistency, cross-
# ref, acronym, caption, …) just inflates every one of their (often batched)
# calls for no signal, which dominated the rerun's cost. Gate by the running
# script's basename; override with $LINT_ANNOTATE_ONLY (comma-separated names,
# or "*" for all).
_SEMANTIC_LINTERS = {
    "problem_clarity_lint_llm.py",
    "central_concept_citation_lint_llm.py",
    "abstract_selfcontained_lint_llm.py",
    "prose_lint_llm.py",
    "contribution_faithfulness_lint_llm.py",
    "contribution_support_lint_llm.py",
}


def _annotations_wanted() -> bool:
    """True if the currently-running linter is one that should receive the
    reviewer annotations. Determined from sys.argv[0] (each linter runs as its
    own process, incl. under run_all_linters.py, so the basename is the linter
    module). $LINT_ANNOTATE_ONLY overrides the default semantic allowlist."""
    override = os.environ.get("LINT_ANNOTATE_ONLY", "").strip()
    if override == "*":
        return True
    allow = ({s.strip() for s in override.split(",") if s.strip()}
             if override else _SEMANTIC_LINTERS)
    script = os.path.basename(sys.argv[0] or "")
    return script in allow


def annotation_context() -> str:
    """Reviewer margin comments to fold into a SEMANTIC LLM linter's system
    prompt (see _SEMANTIC_LINTERS / $LINT_ANNOTATE_ONLY).

    When $LINT_ANNOTATIONS_FILE points at a JSON list of annotations (each
    {page, type, comment, quoted}, as extracted from the draft PDF), return a
    delimited block instructing the linter to treat them as high-priority
    signal WITHIN ITS OWN REMIT. Empty string when the var is unset/unreadable
    or the running linter is not in the semantic allowlist, so those runs are
    unaffected. Cached: the file is read once per process."""
    global _ANNOT_CACHE
    if _ANNOT_CACHE is not None:
        return _ANNOT_CACHE
    path = os.environ.get("LINT_ANNOTATIONS_FILE")
    if not path or not os.path.exists(path) or not _annotations_wanted():
        _ANNOT_CACHE = ""
        return _ANNOT_CACHE
    try:
        rows = json.load(open(path, encoding="utf-8"))
    except Exception:
        _ANNOT_CACHE = ""
        return _ANNOT_CACHE
    lines = []
    for i, r in enumerate(rows, 1):
        pg = r.get("page", "?")
        cm = (r.get("comment") or "").strip()
        q = (r.get("quoted") or "").strip()
        if not cm and not q:
            continue
        on = f' on "{q}"' if q else ""
        lines.append(f"[A{i} p{pg}]{on}: {cm}" if cm
                     else f"[A{i} p{pg}] (highlight){on}")
    if not lines:
        _ANNOT_CACHE = ""
        return _ANNOT_CACHE
    _ANNOT_CACHE = (
        "\n\n----- REVIEWER ANNOTATIONS -----\n"
        "The following are margin comments written by the human expert "
        "reviewer on THIS exact draft. Treat them as high-priority signal. "
        "Where an annotation raises a concern that falls within your specific "
        "linting remit, corroborate it with located evidence or refute it, and "
        "fold that verdict into your findings — do not merely echo the "
        "comment, and do not step outside your remit to chase annotations that "
        "do not apply to your check.\n" + "\n".join(lines)
        + "\n----- END ANNOTATIONS -----")
    return _ANNOT_CACHE


def is_responses_api(base_url: str) -> bool:
    """True if the URL speaks the OpenAI *Responses* API (the Aalto AI API
    gateway) rather than OpenAI-style chat completions. This one flag drives
    the protocol choice throughout the module."""
    return ("aalto-openai-apigw" in base_url
            or base_url.rstrip("/").endswith("/responses"))


def is_local_endpoint(base_url: str) -> bool:
    """True for a server running on this machine (e.g. `mlx_lm.server`, an
    Ollama/llama.cpp OpenAI shim). Thesis text never leaves the device, so
    it is at least as safe as an Aalto tenant."""
    host = urllib.parse.urlparse(base_url).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def is_aalto_endpoint(base_url: str) -> bool:
    """True for endpoints that keep thesis text off public third-party
    services: the Aalto-hosted gateways that keep it within Aalto's tenant
    (the Aalto AI API v2, its deprecating Azure predecessor and the Aalto
    LLM Gateway), plus a local on-device
    server (see is_local_endpoint). These are the only endpoints suitable
    for unpublished manuscripts; anything else is a public third-party
    service."""
    return ("aalto-openai-apigw" in base_url
            or "ai-api-v2.aalto.fi" in base_url
            or "llm-gateway.k8s.aalto.fi" in base_url
            or is_local_endpoint(base_url))


def default_model(base_url: str = BASE_URL) -> str:
    """Text model id to use: ``$LLM_MODEL`` if set, otherwise a sensible
    default for the given endpoint (GLM-5.3-Flash on the Aalto LLM Gateway,
    Gemini 2.5 Flash on the Aalto AI API v2, GPT-5-mini on the old Azure
    gateway, Gemini Flash elsewhere). Override per run with ``--model`` /
    ``$LLM_MODEL``; see the module docstring for the current valid ids on
    each Aalto endpoint."""
    if os.environ.get("LLM_MODEL"):
        return os.environ["LLM_MODEL"]
    if "llm-gateway" in base_url:
        # the strongest model of the gateway set, and the one kept warm;
        # everything else there scales to zero and 503s while spinning up
        return "zai-org/GLM-5.3-Flash"
    if "ai-api-v2.aalto.fi" in base_url:
        return "gemini-2.5-flash"       # both protocols of the v2 endpoint
    if is_responses_api(base_url):
        return "gpt-5-mini-2025-08-07"  # the old Azure gateway
    return "google/gemini-2.5-flash"


def default_strong_model(base_url: str = BASE_URL) -> str:
    """The endpoint's stronger text model, for the linters whose judgment
    is subtler than most (section intros, captions, flow, RQ quality, the
    Bertsekas composition review): the mini/flash tier lets near-misses
    pass. ``$LLM_MODEL`` still wins. gemini-2.5-pro on the Aalto AI API
    v2, gpt-5 on the old Azure gateway, else the endpoint default. On the
    Aalto LLM Gateway there is no stronger tier to reach for: GLM-5.3-Flash
    is both the default and the strongest model hosted there, and it is a
    reasoning model, so it already deliberates on these calls."""
    if os.environ.get("LLM_MODEL"):
        return os.environ["LLM_MODEL"]
    if "ai-api-v2.aalto.fi" in base_url and not is_responses_api(base_url):
        return "gemini-2.5-pro"
    if is_responses_api(base_url):
        return "gpt-5-2025-08-07"
    return default_model(base_url)


def default_vision_base_url(base_url: str = BASE_URL) -> str:
    """Endpoint for requests that attach images — NOT necessarily the text
    endpoint. The Aalto LLM Gateway (the text default) hosts Qwen3-VL, but
    that pod scales to zero and would not come back up at all on 2026-09-22:
    twelve minutes of demand returned an unbroken run of
    ``503 Model not available yet``, which is longer than the client's retry
    window and far longer than a figure lint can wait. So image requests go
    to the Aalto AI API v2 instead, where gemini-2.5-flash is multimodal and
    always warm. Still an Aalto-hosted tenant, so the privacy property the
    suite relies on is unchanged; it just needs $AALTO_API_KEY alongside
    $AALTO_LLM_KEY. Override with $LLM_VISION_BASE_URL (point it back at the
    gateway once the vision pod is dependable), or per run with
    figure_lint_llm.py --base-url."""
    if os.environ.get("LLM_VISION_BASE_URL"):
        return os.environ["LLM_VISION_BASE_URL"]
    if "llm-gateway" in base_url:
        return AIAPI_V2_URL
    return base_url


def default_vision_model(base_url: str = BASE_URL) -> str:
    """Model for requests that attach images, given the VISION endpoint
    (see default_vision_base_url — by default that is the AI API v2, not the
    text default). gemini-2.5-flash is itself multimodal on the v2 endpoint,
    as is the GPT-5 family on the old gateway; on the Aalto LLM Gateway,
    where GLM-5.3-Flash is text-only, this names the separate Qwen3-VL model
    (which scales to zero — see default_vision_base_url)."""
    if os.environ.get("LLM_VISION_MODEL"):
        return os.environ["LLM_VISION_MODEL"]
    if "llm-gateway" in base_url:
        return "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    return default_model(base_url)


def _key_from_shell_profile(*names: str) -> Optional[str]:
    """Fallback when the variable is not in the environment: read the
    `export NAME=value` line from ~/.zshenv / ~/.zshrc (a shell whose
    snapshot predates the export line, cron, IDE runners)."""
    found: Dict[str, str] = {}
    for path in ("~/.zshenv", "~/.zshrc"):
        try:
            text = open(os.path.expanduser(path)).read()
        except OSError:
            continue
        for name in names:
            for m in re.finditer(
                    rf"^\s*export\s+{name}=[\"']?([^\"'\n]+)[\"']?\s*$",
                    text, re.M):
                found[name] = m.group(1)
    for name in names:
        if name in found:
            return found[name]
    return None


def _thinking_on(model: str) -> bool:
    """Whether to let the model reason before answering, on the chat
    completions protocol. $LLM_THINKING decides when set ("on"/"off"); with
    it unset the answer is NO for THINKING_MODELS and irrelevant for the
    rest, which do not reason anyway.

    Off by default because on a linter-sized prompt the reasoning pass does
    not converge: measured 2026-09-22 against the gateway, GLM-5.3-Flash on
    a 61k-char paper streamed ~8.7k reasoning deltas in 65s and spent the
    whole 9k-token budget without emitting one character of answer, and the
    same linter run non-streamed drew repeated 504s from the gateway's
    ingress while it thought. With thinking off the run finishes in seconds
    and the findings are sound. Set LLM_THINKING=on to get it back on a
    short prompt, where it does converge and does sharpen the judgment."""
    v = os.environ.get("LLM_THINKING", "").strip().lower()
    if v in ("off", "0", "false", "no", "none"):
        return False
    if v in ("on", "1", "true", "yes"):
        return True
    return not any(tag in model for tag in THINKING_MODELS)


def _thinking_budget(model: str, max_tokens: int) -> int:
    """Token budget for a chat-completions request. A reasoning model spends
    its thinking tokens out of max_tokens on this protocol, so when thinking
    is on give the ones we know about a fixed headroom (THINKING_HEADROOM)
    rather than let the report be truncated and force a re-request."""
    if _thinking_on(model) and any(tag in model for tag in THINKING_MODELS):
        return max_tokens + THINKING_HEADROOM
    return max_tokens


def default_api_key(base_url: str = BASE_URL) -> Optional[str]:
    """Pick the API key that matches the endpoint: AALTO_LLM_KEY for the
    Aalto LLM Gateway, AALTO_API_KEY (or AALTO_OPENAI_API_KEY) for the Aalto
    AI API, OPENROUTER_API_KEY otherwise. Each falls back to the user's shell
    profile; returns None if nothing is found."""
    if "llm-gateway" in base_url:              # Aalto LLM Gateway
        return (os.environ.get("AALTO_LLM_KEY")
                or _key_from_shell_profile("AALTO_LLM_KEY"))
    if "ai-api-v2.aalto.fi" in base_url or is_responses_api(base_url):
        # Aalto AI API: v2 (either protocol) or the old Azure gateway
        return (os.environ.get("AALTO_API_KEY")
                or os.environ.get("AALTO_OPENAI_API_KEY")
                or _key_from_shell_profile("AALTO_API_KEY",
                                           "AALTO_OPENAI_API_KEY"))
    return (os.environ.get("OPENROUTER_API_KEY")
            or _key_from_shell_profile("OPENROUTER_API_KEY"))


class LLMClient:
    """One JSON-mode request per complete() call; speaks two protocols,
    chosen from the URL: the OpenAI Responses API (Aalto AI API gateway)
    and OpenAI-style chat completions (Aalto LLM Gateway, OpenRouter)."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.responses = is_responses_api(base_url)

    def complete(self, model: str, system: str, user: str,
                 timeout: float = 900.0, max_tokens: int = 8000,
                 images: Optional[list] = None) -> Tuple[str, dict]:
        """Returns (raw_text, usage_dict). Raises RuntimeError on failure.
        `images` is an optional list of PNG bytes attached to the user
        message (use a vision-capable model, see default_vision_model).

        The token budget is ESCALATED when the answer is cut off: on the
        Aalto AI API v2 a reasoning model's thinking tokens count against
        max_tokens, so a long JSON report can be truncated (finish_reason
        "length" / status "incomplete") and fail to parse. A cut-off answer
        whose text is not complete JSON is re-requested with twice, then
        four times the budget, and the escalation is printed on stderr so a
        slow linter says why (ported from the dictionary suite's
        llm_client, 2026-09-17)."""
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0,
                       "total_tokens": 0}
        for budget in (max_tokens, 2 * max_tokens, 4 * max_tokens):
            text, usage, cut_off = self._complete_once(
                model, system, user, timeout, budget, images)
            for k in usage_total:
                usage_total[k] += usage.get(k, 0)
            if not cut_off or extract_json(text) is not None:
                return text, usage_total
            if budget == 4 * max_tokens:
                break
            print(f"[llm] answer cut off at {budget} tokens (reasoning "
                  f"counts against the budget) — retrying with "
                  f"{2 * budget}", file=sys.stderr)
        return text, usage_total

    def _complete_once(self, model: str, system: str, user: str,
                       timeout: float, max_tokens: int,
                       images: Optional[list]) -> Tuple[str, dict, bool]:
        """One request. Returns (raw_text, usage_dict, cut_off)."""
        import base64
        # Fold in reviewer annotations (no-op unless $LINT_ANNOTATIONS_FILE set).
        system = system + annotation_context()
        if self.responses:
            content = [{"type": "input_text", "text": user}]
            for png in images or []:
                b64 = base64.standard_b64encode(png).decode("ascii")
                content.append({"type": "input_image",
                                "image_url": f"data:image/png;base64,{b64}"})
            body = {
                "model": model,
                "instructions": system,
                "input": [{"role": "user", "content": content}],
                # headroom: gpt-5-family reasoning tokens count against this
                "max_output_tokens": max_tokens + 8000,
                "text": {"format": {"type": "json_object"}},
            }
            fmt_key = "text"
            url = self.base_url
            # the deprecating Azure APIM gateway wants its subscription
            # header; the v2 endpoint (and any other) uses Bearer auth
            auth = ({"Ocp-Apim-Subscription-Key": self.api_key}
                    if "aalto-openai-apigw" in self.base_url
                    else {"Authorization": f"Bearer {self.api_key}"})
            headers = {**auth, "Content-Type": "application/json"}
        else:
            if images:
                content = [{"type": "text", "text": user}]
                for png in images:
                    b64 = base64.standard_b64encode(png).decode("ascii")
                    content.append(
                        {"type": "image_url",
                         "image_url": {"url":
                                       f"data:image/png;base64,{b64}"}})
            else:
                content = user
            body = {
                "model": model,
                "max_tokens": _thinking_budget(model, max_tokens),
                "temperature": 0,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": content}],
                "response_format": {"type": "json_object"},
            }
            if not _thinking_on(model):
                # vLLM-style switch (GLM, Qwen) plus the OpenAI-style one;
                # a server that knows neither ignores both.
                body["chat_template_kwargs"] = {"enable_thinking": False}
                body["reasoning_effort"] = "none"
            # STREAM. Not for show: a long prompt makes the model go quiet
            # for minutes (reasoning, or just a slow prefill) and the Aalto
            # gateway's ingress cuts the idle connection with a 504 —
            # reproduced 2026-09-22 on a 99k-char linter prompt, where every
            # retry died the same way. An SSE response keeps bytes moving,
            # so the request survives; it also covers the v2 gateway's
            # matching 504 on a long gemini-2.5-pro call (2026-09-17).
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            fmt_key = "response_format"
            url = self.base_url.rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}",
                       "Content-Type": "application/json",
                       "X-Title": "msc-thesis-linters"}

        data = None
        for attempt in range(11):
            req = urllib.request.Request(
                url, data=json.dumps(body).encode("utf-8"), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = (_read_sse(resp) if body.get("stream")
                            else json.loads(resp.read().decode("utf-8")))
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:400]
                if e.code == 401:
                    raise RuntimeError(
                        "The LLM gateway rejected the key (401) — check "
                        "AALTO_API_KEY / AALTO_LLM_KEY / OPENROUTER_API_KEY "
                        "/ --api-key.") from e
                if e.code == 400 and fmt_key in body:
                    # Endpoint rejects the JSON-mode format: drop it and
                    # retry (the system prompt demands bare JSON anyway),
                    # and SAY so — a silent drop leaves the caller to
                    # report an unparseable answer with no hint of the cause.
                    print(f"[llm] gateway rejected the JSON response format "
                          f"({detail[:120]}) — retrying without it",
                          file=sys.stderr)
                    body.pop(fmt_key)
                    continue
                if e.code in (502, 503, 504, 429) and attempt < 10:
                    # The Aalto LLM gateway scales models to zero; a cold
                    # model answers 503 while its pod spins up. ai-api-v2
                    # answers 504 when a long gemini-2.5-pro call outlives
                    # its upstream timeout (seen 2026-09-17).
                    wait = 45
                    print(f"[llm] {e.code} from gateway ({detail[:80]}) — "
                          f"retrying in {wait}s ({attempt + 1}/10)",
                          file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"LLM gateway error {e.code}: {detail}") from e
            except (TimeoutError, urllib.error.URLError, ConnectionError,
                    http.client.HTTPException) as e:
                # Transient network/read timeout, or the gateway closing
                # the connection without a response (RemoteDisconnected,
                # seen 2026-09-17 on a 99k-char prompt) — common on large
                # vision payloads or slow reasoning. Retry with backoff
                # instead of aborting the whole linter run. (HTTPError is a
                # URLError subclass but is handled above.)
                reason = getattr(e, "reason", e)
                if attempt < 10:
                    wait = 15
                    print(f"[llm] network error ({reason}) — retrying in "
                          f"{wait}s ({attempt + 1}/10)", file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"LLM gateway unreachable: {reason}") from e
        if data is None:
            raise RuntimeError("LLM gateway unreachable after retries.")
        if isinstance(data.get("error"), dict) and data["error"]:
            raise RuntimeError(f"LLM gateway error: {data['error']}")

        if self.responses:
            text = "".join(c.get("text", "")
                           for item in data.get("output", [])
                           if item.get("type") == "message"
                           for c in item.get("content", [])
                           if c.get("type") == "output_text")
            u = data.get("usage", {}) or {}
            usage = {"prompt_tokens": u.get("input_tokens", 0),
                     "completion_tokens": u.get("output_tokens", 0),
                     "total_tokens": u.get("total_tokens", 0)}
            cut_off = (data.get("status") == "incomplete"
                       and (data.get("incomplete_details") or {}).get(
                           "reason") == "max_output_tokens")
        else:
            choices = data.get("choices") or [{}]
            choice = choices[0] or {}
            # a budget spent entirely on reasoning comes back with no
            # message at all on ai-api-v2 (seen 2026-09-17 at 20 tokens)
            msg = choice.get("message") or {}
            text = msg.get("content") or ""
            u = data.get("usage", {}) or {}
            usage = {"prompt_tokens": u.get("prompt_tokens", 0),
                     "completion_tokens": u.get("completion_tokens", 0),
                     "total_tokens": u.get("total_tokens", 0)}
            cut_off = choice.get("finish_reason") == "length"
        return text, usage, cut_off


def _read_sse(resp) -> dict:
    """Collect a streamed chat-completions response into the same dict shape
    the non-streamed call returns, so the caller sees no difference.

    Deltas arrive as ``data: {...}`` lines terminated by ``data: [DONE]``.
    Only ``delta.content`` is kept: a reasoning model also streams
    ``reasoning_content`` (GLM-5.3-Flash emitted ~8.7k such deltas and no
    content at all on a 61k-char prompt, 2026-09-22), which is not part of
    the answer. usage arrives in a final chunk with an empty choices list,
    courtesy of stream_options.include_usage."""
    parts, finish, usage = [], None, {}
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except ValueError:                 # keep-alive or partial frame
            continue
        if isinstance(chunk.get("error"), dict) and chunk["error"]:
            return {"error": chunk["error"]}
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices") or []:
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                parts.append(piece)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return {"choices": [{"message": {"content": "".join(parts)},
                         "finish_reason": finish}],
            "usage": usage}


def make_client(base_url: str = BASE_URL,
                api_key: Optional[str] = None) -> LLMClient:
    """Build an LLMClient for the endpoint, resolving the key (explicit arg,
    else ``default_api_key``) and raising RuntimeError if none is found.
    Prints a privacy warning to stderr when the endpoint is not Aalto-hosted,
    since a thesis draft is unpublished material."""
    key = api_key or default_api_key(base_url)
    if not key:
        raise RuntimeError(
            "No API key — set AALTO_API_KEY (Aalto AI API v2, the default "
            "gateway) or AALTO_LLM_KEY / OPENROUTER_API_KEY (with "
            "--base-url), or pass --api-key.")
    if not is_aalto_endpoint(base_url):
        print(
            f"[llm] WARNING: {base_url} is not an Aalto-hosted endpoint. "
            "A thesis draft is unpublished material — Aalto policy says not "
            "to send unpublished, confidential, or personal data to public "
            "AI services. Use the Aalto AI API (default) or the Aalto LLM "
            "Gateway instead.",
            file=sys.stderr)
    return LLMClient(base_url, key)


def _repair_json(s: str) -> str:
    r"""Fix the invalid escapes LLMs most often emit. JSON permits only
    \" \\ \/ \b \f \n \r \t \uXXXX; anything else after a backslash is a
    parse error. Models frequently write \' (e.g. "confidence\'s"), so drop
    the backslash before any character that is not a legal escape."""
    return re.sub(r'\\([^"\\/bfnrtu])', r"\1", s)


def extract_json(text: str) -> Optional[dict]:
    """Parse strict JSON, with fallbacks: strip code fences, repair the
    invalid escapes LLMs emit (e.g. \\'), and finally the first {...} block."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    for candidate in (text, _repair_json(text)):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        for candidate in (m.group(0), _repair_json(m.group(0))):
            try:
                return json.loads(candidate)
            except Exception:
                pass
    return None
