#!/usr/bin/env python3
"""LLM linter (LOCAL): what is the ACTUAL main contribution, and is it
faithfully presented?

A paper's stated contribution (abstract, intro, the "our contributions are"
bullets) is a CLAIM. This linter separates that claim from what the paper
actually delivers -- judged from the method, the experiments, and the results
-- and reports whether the two agree. It answers two questions a reviewer asks
first:

  1. What is the actual main contribution?  (one the body genuinely supports:
     a method, an analysis/finding, a benchmark/dataset, a theoretical result,
     a negative result, ...)
  2. Is it faithfully presented?  Does the framing over- or under-sell what
     was done?

Verdicts:
  FAITHFUL     the stated contribution matches what the paper delivers.
  OVERSTATED   the framing claims more than the evidence supports -- a
               "framework/theory/guarantee" that is really a heuristic; "SOTA"
               on thin baselines; generality claimed but only one setting
               shown; a formal-sounding claim backed only by an informal
               argument.
  UNDERSTATED  the paper undersells its real contribution (e.g. buries a solid
               empirical finding behind a modest method claim).
  MISALIGNED   the headline contribution is not the one the paper best
               supports (the abstract sells X; the evidence is really about Y).
  UNCLEAR      no contribution is stated clearly enough to assess, or the text
               is too garbled.

This is a reviewer aid, not a verdict on acceptance. It is deliberately
critical about the gap between claim and delivery. Exit 1 if the verdict is
not FAITHFUL; 0 if FAITHFUL; 2 usage error.

Gateway: same as the other LLM linters (aalto_llm.py); defaults to the Aalto
AI API (--base-url can point at another OpenAI-style endpoint if ever needed).

Usage:
  python3 contribution_faithfulness_lint_llm.py submission.pdf
  python3 contribution_faithfulness_lint_llm.py submission.pdf --format markdown --out c.md
"""

import argparse
import sys
from typing import List, Optional

from aalto_llm import (API_KEY_HELP, BASE_URL, BASE_URL_HELP, default_model,
                       extract_json, make_client)
from lintutil import load_lines

SYSTEM_PROMPT = (
    "You are an experienced, critical reviewer for a top ML venue. You are "
    "given the extracted text of a submission (page markers '[[page N]]' "
    "included). Your job is to separate the paper's CLAIMED contribution from "
    "what it ACTUALLY delivers, and judge whether the claim is faithful.\n\n"
    "STEP 1 -- CLAIMED. Extract the main contribution AS THE PAPER FRAMES IT: "
    "the abstract, the introduction, and any explicit 'our contributions "
    "are' list. Quote or tightly paraphrase it.\n\n"
    "STEP 2 -- ACTUAL. Determine the main contribution the paper GENUINELY "
    "SUPPORTS, judged from the method section, the experiments, and the "
    "results/tables -- not from the framing. Decide its TYPE (new method / "
    "empirical finding or analysis / benchmark or dataset / theoretical "
    "result / negative result / system) and state it in one or two sentences, "
    "grounded in what is actually demonstrated. If the real strength is an "
    "analysis or a negative result rather than the proposed method, say so.\n\n"
    "STEP 3 -- FAITHFULNESS. Compare claimed vs actual and assign ONE verdict:\n"
    "  FAITHFUL    the stated contribution matches what is delivered.\n"
    "  OVERSTATED  the framing claims more than the evidence supports: a "
    "'framework/theory/principled/guarantee' that is really a heuristic; "
    "'state-of-the-art' on thin or non-matched baselines; generality or "
    "robustness claimed but only one dataset/setting shown; a formal claim "
    "backed only by an informal argument; 'first to' without support.\n"
    "  UNDERSTATED the paper undersells its real contribution.\n"
    "  MISALIGNED  the headline is not the contribution the paper best "
    "supports (sells X; the evidence is really about Y).\n"
    "  UNCLEAR     no clearly assessable contribution, or text too garbled.\n\n"
    "Be specific and conservative: only flag OVERSTATED/MISALIGNED with a "
    "concrete, nameable gap you can point to in the text (a claimed property "
    "not demonstrated, a missing baseline, a scope word like 'general/robust/"
    "provable' the evidence does not earn). Do not moralise; a modest, "
    "accurately-framed contribution is FAITHFUL.\n\n"
    "Give: 'claimed' (the contribution as framed), 'actual' (what the paper "
    "supports, with its type), 'verdict', a list of specific 'discrepancies' "
    "(each: the claim word/phrase, why the evidence falls short, and the page "
    "of a supporting quote), and 'fix' -- one restatement of the contribution "
    "that the paper's own evidence would justify.\n\n"
    "Keep every field CONCISE: at most 2-3 sentences each; paraphrase, do "
    "NOT paste long verbatim passages or enumerate every bullet. Output "
    "MUST be a single valid JSON object and nothing else.\n\n"
    "Respond with STRICT JSON:\n"
    '{"claimed": "...", "actual": "...", "actual_type": "...", '
    '"verdict": "FAITHFUL|OVERSTATED|UNDERSTATED|MISALIGNED|UNCLEAR", '
    '"discrepancies": [{"claim": "...", "shortfall": "...", "page": "N"}], '
    '"fix": "...", "notes": "..."}'
)

FAIL_VERDICTS = {"OVERSTATED", "UNDERSTATED", "MISALIGNED"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="LLM linter: actual main contribution and whether it is "
                    "faithfully presented.")
    ap.add_argument("pdf", help="Path to the submission PDF (or .txt extract).")
    ap.add_argument("--base-url", default=BASE_URL, help=BASE_URL_HELP)
    ap.add_argument("--api-key", default=None, help=API_KEY_HELP)
    ap.add_argument("--model", default=None,
                    help="Model id (default depends on the gateway).")
    ap.add_argument("--max-chars", type=int, default=400_000,
                    help="Truncate extracted text beyond this many chars.")
    ap.add_argument("--out", help="Write report to this file.")
    ap.add_argument("--format", choices=["text", "markdown"], default="text")
    args = ap.parse_args(argv)

    lines, mode = load_lines([args.pdf])
    if mode != "pdf":
        print("ERROR: this linter takes a compiled PDF or a .txt extract.",
              file=sys.stderr)
        return 2

    chunks, cur = [], None
    for where, t in lines:
        if where != cur:
            cur = where
            chunks.append(f"\n[[page {where[1:]}]]\n")
        chunks.append(t + "\n")
    text = "".join(chunks)
    truncated = len(text) > args.max_chars
    if truncated:
        text = text[: args.max_chars]
        print(f"[warn] text truncated to {args.max_chars} chars",
              file=sys.stderr)

    model = args.model or default_model(args.base_url)
    client = make_client(args.base_url, args.api_key)
    print(f"[info] gateway={args.base_url}\n[info] model={model}  "
          f"chars={len(text)}", file=sys.stderr)

    user = (f"submission text{' (TRUNCATED)' if truncated else ''}:\n"
            f'"""\n{text}\n"""')
    raw, usage = client.complete(model=model, system=SYSTEM_PROMPT,
                                 user=user, timeout=600, max_tokens=4000)
    parsed = extract_json(raw) or {}
    claimed = str(parsed.get("claimed", "")).strip()
    actual = str(parsed.get("actual", "")).strip()
    atype = str(parsed.get("actual_type", "")).strip()
    verdict = str(parsed.get("verdict", "UNCLEAR")).strip().upper()
    discs = [d for d in parsed.get("discrepancies", []) if isinstance(d, dict)]
    fix = str(parsed.get("fix", "")).strip()
    notes = str(parsed.get("notes", "")).strip()

    md = args.format == "markdown"
    out = []
    if md:
        out += [f"# Contribution faithfulness — {args.pdf}", "",
                f"_model: {model}_", "", f"**Verdict: {verdict}**", ""]
    else:
        out += [f"== Contribution-faithfulness lint (LLM, {model})",
                f"File: {args.pdf}", "", f"VERDICT: {verdict}", ""]

    def block(label, body):
        if not body:
            return
        if md:
            out.append(f"**{label}:** {body}\n")
        else:
            out.append(f"{label}:")
            out.append(f"  {body}")
            out.append("")

    block("Actual main contribution" + (f" ({atype})" if atype else ""), actual)
    block("As presented (claimed)", claimed)

    if discs:
        if md:
            out += ["## Discrepancies", ""]
        else:
            out.append("Discrepancies (claim vs. evidence):")
        for d in discs:
            claim = str(d.get("claim", "")).strip()
            short = str(d.get("shortfall", "")).strip()
            page = str(d.get("page", "?")).strip()
            if md:
                out += [f"- **{claim}** — {short} _(p{page})_"]
            else:
                out.append(f"  - “{claim}”  (p{page})")
                out.append(f"      shortfall: {short}")
        out.append("")

    block("Faithful restatement", fix)
    if notes:
        block("Notes", notes)

    out += ["", f"verdict={verdict}, {len(discs)} discrepancy(ies).  "
            f"(tokens: {usage.get('total_tokens', '?')})"]
    report = "\n".join(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(report)
    return 1 if verdict in FAIL_VERDICTS else 0


if __name__ == "__main__":
    raise SystemExit(main())
