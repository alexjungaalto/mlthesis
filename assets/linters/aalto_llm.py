#!/usr/bin/env python3
"""Shared LLM client for the thesis linters.

Default gateway: the Aalto AI API (Azure OpenAI gateway; Responses API at
AALTO_AZURE_URL, Ocp-Apim-Subscription-Key auth from $AALTO_API_KEY,
GPT-5-family models with dated IDs; Aalto network/VPN only). Alternatives
via base_url: the self-hosted Aalto LLM Gateway (LLMGW_BASE_URL below;
OpenAI-style chat completions, Bearer $AALTO_LLM_KEY, Qwen models such as
Qwen/Qwen3-30B-A3B-Instruct-2507-FP8; models scale to zero and answer 503
while spinning up) and any OpenAI-style endpoint such as OpenRouter
($OPENROUTER_API_KEY). LLMClient picks protocol and key from the URL.

Every ``*_llm`` linter builds its client with ``make_client()`` and picks a
model with ``default_model()`` / ``default_vision_model()`` from this one
module, so a user configures the whole suite through a small set of
environment variables (each can also be overridden by a CLI flag):

  LLM_BASE_URL      which endpoint to talk to; decides everything below.
                    Unset -> the Aalto AI API (AALTO_AZURE_URL).
  LLM_MODEL         text model id; unset -> a per-endpoint default.
  LLM_VISION_MODEL  model id for image requests (figure_lint_llm.py).
  AALTO_API_KEY     key for the default Aalto AI API (also accepts
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
sources, and the values below were current in Aug 2026:

  * Aalto AI API (default endpoint). GPT-5 family, dated ids:
    ``gpt-5-2025-08-07``, ``gpt-5-mini-2025-08-07`` (the suite default),
    ``gpt-5-nano-2025-08-07``. Catalogue + keys (Aalto login):
    https://www.aalto.fi/en/services/aalto-ai-apis
  * Aalto LLM Gateway (``--base-url https://llm-gateway.k8s.aalto.fi/api/v1``).
    Open-weight models served on Aalto hardware; the live list is a GET on
    ``/models`` (the id string is what you pass to ``--model``). Aug-2026 set:
    ``RedHatAI/gemma-4-31B-it-FP8-Dynamic`` (recommended starter),
    ``openai/gpt-oss-120b``, ``Qwen/Qwen3-30B-A3B-Instruct-2507-FP8``,
    ``Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8``,
    ``Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`` (vision),
    ``Qwen/Qwen3-VL-30B-A3B-Thinking-FP8``, ``Qwen/Qwen3.8-27B-FP8``,
    ``google/gemma-4-E4B-it``, ``google/codegemma-7b-it``. Docs + key
    self-service UI (Aalto VPN): https://scicomp.aalto.fi/aalto/llm-web-apis/

Both Aalto services require the Aalto network / VPN. Example — run the whole
suite through the gateway's Gemma model:

  export AALTO_LLM_KEY=...    # made at https://llm-gateway.k8s.aalto.fi/
  python3 run_all_linters.py thesis.pdf --llm \
      --base-url https://llm-gateway.k8s.aalto.fi/api/v1 \
      --model RedHatAI/gemma-4-31B-it-FP8-Dynamic

Stdlib urllib only — no SDK needed.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Optional, Tuple

AALTO_AZURE_URL = "https://aalto-openai-apigw.azure-api.net/v1/openai/responses"
LLMGW_BASE_URL = "https://llm-gateway.k8s.aalto.fi/api/v1"
BASE_URL = os.environ.get("LLM_BASE_URL", AALTO_AZURE_URL)

BASE_URL_HELP = ("LLM endpoint (default: the Aalto AI API, "
                 f"{AALTO_AZURE_URL}). Alternatives: the Aalto LLM Gateway "
                 f"({LLMGW_BASE_URL}) or any OpenAI-style endpoint such as "
                 "https://openrouter.ai/api/v1.")
API_KEY_HELP = ("API key (default: $AALTO_API_KEY for the Aalto AI API, "
                "$AALTO_LLM_KEY for the Aalto LLM gateway, "
                "$OPENROUTER_API_KEY otherwise).")


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
    (the Aalto AI API and the Aalto LLM Gateway), plus a local on-device
    server (see is_local_endpoint). These are the only endpoints suitable
    for unpublished manuscripts; anything else is a public third-party
    service."""
    return ("aalto-openai-apigw" in base_url
            or "llm-gateway.k8s.aalto.fi" in base_url
            or is_local_endpoint(base_url))


def default_model(base_url: str = BASE_URL) -> str:
    """Text model id to use: ``$LLM_MODEL`` if set, otherwise a sensible
    default for the given endpoint (GPT-5-mini on the Aalto AI API, Qwen3 on
    the Aalto LLM Gateway, Gemini Flash elsewhere). Override per run with
    ``--model`` / ``$LLM_MODEL``; see the module docstring for the current
    valid ids on each Aalto endpoint."""
    if os.environ.get("LLM_MODEL"):
        return os.environ["LLM_MODEL"]
    if is_responses_api(base_url):
        return "gpt-5-mini-2025-08-07"
    if "llm-gateway" in base_url:
        return "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    return "google/gemini-2.5-flash"


def default_vision_model(base_url: str = BASE_URL) -> str:
    """Model for requests that attach images. On the Aalto AI API the
    GPT-5 family is multimodal; the Aalto LLM Gateway hosts a separate
    vision model."""
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


def default_api_key(base_url: str = BASE_URL) -> Optional[str]:
    """Pick the API key that matches the endpoint: AALTO_LLM_KEY for the
    Aalto LLM Gateway, AALTO_API_KEY (or AALTO_OPENAI_API_KEY) for the Aalto
    AI API, OPENROUTER_API_KEY otherwise. Each falls back to the user's shell
    profile; returns None if nothing is found."""
    if "llm-gateway" in base_url:              # Aalto LLM Gateway
        return (os.environ.get("AALTO_LLM_KEY")
                or _key_from_shell_profile("AALTO_LLM_KEY"))
    if is_responses_api(base_url):             # Aalto AI API (Azure gateway)
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
        message (use a vision-capable model, see default_vision_model)."""
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
            headers = {"Ocp-Apim-Subscription-Key": self.api_key,
                       "Content-Type": "application/json"}
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
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": content}],
                "response_format": {"type": "json_object"},
            }
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
                    data = json.loads(resp.read().decode("utf-8"))
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
                    # retry (the system prompt demands bare JSON anyway).
                    body.pop(fmt_key)
                    continue
                if e.code in (503, 429) and attempt < 10:
                    # The Aalto LLM gateway scales models to zero; a cold
                    # model answers 503 while its pod spins up.
                    wait = 45
                    print(f"[llm] {e.code} from gateway ({detail[:80]}) — "
                          f"retrying in {wait}s ({attempt + 1}/10)",
                          file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"LLM gateway error {e.code}: {detail}") from e
            except (TimeoutError, urllib.error.URLError) as e:
                # Transient network/read timeout — common on large vision
                # payloads or slow reasoning. Retry with backoff instead of
                # aborting the whole linter run. (HTTPError is a URLError
                # subclass but is handled above, so it never reaches here.)
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
        else:
            text = data["choices"][0]["message"]["content"] or ""
            u = data.get("usage", {}) or {}
            usage = {"prompt_tokens": u.get("prompt_tokens", 0),
                     "completion_tokens": u.get("completion_tokens", 0),
                     "total_tokens": u.get("total_tokens", 0)}
        return text, usage


def make_client(base_url: str = BASE_URL,
                api_key: Optional[str] = None) -> LLMClient:
    """Build an LLMClient for the endpoint, resolving the key (explicit arg,
    else ``default_api_key``) and raising RuntimeError if none is found.
    Prints a privacy warning to stderr when the endpoint is not Aalto-hosted,
    since a thesis draft is unpublished material."""
    key = api_key or default_api_key(base_url)
    if not key:
        raise RuntimeError(
            "No API key — set AALTO_API_KEY (Aalto AI API, the default "
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
