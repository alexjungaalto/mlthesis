#!/usr/bin/env python3
"""LLM linter: does the automated suite catch what the human reviewer flagged?

Cross-checks the reviewer's own margin annotations on a draft against the
findings the rest of the linter suite already produced. For each annotation it
decides whether some automated linter surfaces the same concern:

  COVERED   a linter clearly flags the same issue (name it, and where).
  PARTIAL   a linter touches the area but misses the specific point.
  UNCAUGHT  no linter addresses it — a gap in automated coverage, or a
            concern outside any linter's remit (still worth surfacing).

The point is twofold: (a) corroborate the reviewer — where a linter and the
human agree, the finding is solid; and (b) expose blind spots — annotations no
linter caught are the checks the suite is missing.

Inputs:
  * an annotations JSON list (as extracted from the PDF's margin comments;
    each item {page, type, comment, quoted}), via --annotations
  * the concatenated linter output for the same paper (the full-sweep report
    lintout/<stem>_llm.txt is the comprehensive source), given as the
    positional argument.

Whole-document, single LLM call. Findings are heuristic LLM judgements.

Gateway: the Aalto AI API by default (see aalto_llm.py; $AALTO_API_KEY,
Aalto network/VPN only); --base-url switches gateways.

Usage:
  python3 annotation_coverage_lint_llm.py lintout/foo_llm.txt \
      --annotations lintout/foo_annotations.json --out lintout/foo_annocov.txt
Exit status: 0 every annotation COVERED, 1 any PARTIAL/UNCAUGHT, 2 usage error.
"""

import argparse
import json
import os
import sys
from typing import List, Optional

from aalto_llm import (API_KEY_HELP, BASE_URL, BASE_URL_HELP, default_model,
                       extract_json, make_client)

STATUSES = ["COVERED", "PARTIAL", "UNCAUGHT"]

SYSTEM_PROMPT = (
    "You audit whether an automated paper-linter suite catches the same "
    "issues a human expert reviewer flagged by hand. You are given (1) the "
    "reviewer's margin ANNOTATIONS on a draft (each with an id A1, A2, …, the "
    "page, the highlighted text, and the reviewer's comment) and (2) the "
    "concatenated OUTPUT of every automated linter that ran on that same "
    "draft.\n\n"
    "For EACH annotation, decide whether the automated linters surface the "
    "same concern:\n"
    "  COVERED   at least one linter clearly flags the same issue. Name the "
    "linter(s) (use the .py names that head the linter sections) and quote the "
    "shortest phrase from their output that matches.\n"
    "  PARTIAL   a linter touches the same area but misses the reviewer's "
    "specific point (e.g. flags jargon generally but not this term).\n"
    "  UNCAUGHT  no linter addresses it. This is either a coverage gap in the "
    "suite or a concern no linter is designed to check (say which).\n\n"
    "Judge by MEANING, not string overlap: the reviewer asking 'what is one "
    "data point?' is COVERED by a problem-clarity linter grading the data point "
    "PARTIAL/UNCLEAR, even with no shared words. Be strict about COVERED — the "
    "linter must actually name the same problem, not merely lint the same "
    "sentence for a different reason.\n\n"
    "Respond with STRICT JSON:\n"
    '{"annotations": [{"id": "A1", "status": "COVERED|PARTIAL|UNCAUGHT", '
    '"linters": ["problem_clarity_lint_llm.py"], "note": "one line: what the '
    'linter caught, or why nothing did"}], '
    '"suite_gaps": ["one line per class of concern the suite systematically '
    'misses"], '
    '"overall": "one-sentence verdict on human/automated agreement"}'
)


def load_annotations(path: str) -> List[dict]:
    rows = json.load(open(path, encoding="utf-8"))
    out = []
    for i, r in enumerate(rows, 1):
        cm = (r.get("comment") or "").strip()
        q = (r.get("quoted") or "").strip()
        if not cm and not q:
            continue
        out.append({"id": f"A{i}", "page": r.get("page", "?"),
                    "comment": cm, "quoted": q,
                    "type": r.get("type", "")})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="LLM linter: reviewer-annotation coverage by the suite.")
    ap.add_argument("linters", help="Concatenated linter output "
                    "(e.g. lintout/<stem>_llm.txt).")
    ap.add_argument("--annotations", required=True,
                    help="Annotations JSON (list of {page,type,comment,quoted}).")
    ap.add_argument("--base-url", default=BASE_URL, help=BASE_URL_HELP)
    ap.add_argument("--api-key", default=None, help=API_KEY_HELP)
    ap.add_argument("--model", default=None,
                    help="Model id (default depends on the gateway).")
    ap.add_argument("--max-chars", type=int, default=200_000,
                    help="Truncate linter output beyond this many chars.")
    ap.add_argument("--out", help="Write report to this file.")
    args = ap.parse_args(argv)

    if not os.path.exists(args.annotations):
        print(f"ERROR: annotations file not found: {args.annotations}",
              file=sys.stderr)
        return 2
    annots = load_annotations(args.annotations)
    if not annots:
        print("ERROR: no usable annotations (all empty).", file=sys.stderr)
        return 2
    try:
        lint_text = open(args.linters, encoding="utf-8",
                         errors="replace").read()
    except OSError as e:
        print(f"ERROR: cannot read linter output {args.linters}: {e}",
              file=sys.stderr)
        return 2

    truncated = len(lint_text) > args.max_chars
    if truncated:
        lint_text = lint_text[: args.max_chars]
        print(f"[warn] linter output truncated to {args.max_chars} chars",
              file=sys.stderr)

    annot_block = "\n".join(
        f'{a["id"]} [p{a["page"]}] '
        + (f'on "{a["quoted"]}" ' if a["quoted"] else "")
        + (f'— {a["comment"]}' if a["comment"] else "(highlight, no comment)")
        for a in annots)

    # This linter must NOT get the annotations re-injected into its system
    # prompt (they are already its explicit input); neutralise the hook.
    os.environ.pop("LINT_ANNOTATIONS_FILE", None)

    model = args.model or default_model(args.base_url)
    client = make_client(args.base_url, args.api_key)
    print(f"[info] gateway={args.base_url}\n[info] model={model}  "
          f"annotations={len(annots)}  lint_chars={len(lint_text)}",
          file=sys.stderr)

    user = (f"REVIEWER ANNOTATIONS ({len(annots)}):\n{annot_block}\n\n"
            f"AUTOMATED LINTER OUTPUT{' (TRUNCATED)' if truncated else ''}:\n"
            f'"""\n{lint_text}\n"""')
    raw, usage = client.complete(model=model, system=SYSTEM_PROMPT,
                                 user=user, timeout=600, max_tokens=6000)
    parsed = extract_json(raw) or {}

    by_id = {}
    for r in parsed.get("annotations", []):
        if not isinstance(r, dict):
            continue
        st = str(r.get("status", "UNCAUGHT")).strip().upper()
        if st not in STATUSES:
            st = "UNCAUGHT"
        lints = r.get("linters", [])
        if isinstance(lints, str):
            lints = [lints]
        lints = [str(x).strip() for x in lints if str(x).strip()]
        by_id[str(r.get("id", "")).strip()] = {
            "status": st, "linters": lints,
            "note": str(r.get("note", "")).strip() or "—"}

    rows = []
    for a in annots:
        v = by_id.get(a["id"], {"status": "UNCAUGHT", "linters": [],
                                "note": "not assessed by the model"})
        rows.append({**a, **v})

    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
    overall = str(parsed.get("overall", "")).strip()
    gaps = [str(g).strip() for g in parsed.get("suite_gaps", [])
            if str(g).strip()]

    summary = (f"{len(rows)} annotations — {counts['COVERED']} covered, "
               f"{counts['PARTIAL']} partial, {counts['UNCAUGHT']} uncaught")

    out = [f"== Annotation-coverage lint (LLM, {model})",
           f"File: {args.linters}",
           f"Annotations: {args.annotations}", "",
           f"SUMMARY: {summary}", "",
           "How to read: each reviewer margin annotation, whether the "
           "automated suite already flags the same concern, and which linter.",
           "COVERED a linter flags it · PARTIAL a linter is near but misses "
           "the point · UNCAUGHT no linter addresses it.", ""]

    order = {"UNCAUGHT": 0, "PARTIAL": 1, "COVERED": 2}
    for r in sorted(rows, key=lambda x: (order[x["status"]], x["id"])):
        q = f'  "{r["quoted"][:90]}"' if r["quoted"] else ""
        out.append(f'[{r["status"]:<8}] {r["id"]:<4} p{r["page"]}{q}')
        if r["comment"]:
            out.append(f"    comment: {r['comment']}")
        if r["linters"]:
            out.append(f"    linters: {', '.join(r['linters'])}")
        out.append(f"    note:    {r['note']}")
        out.append("")

    if gaps:
        out.append("-- Suite coverage gaps --")
        for g in gaps:
            out.append(f"  * {g}")
        out.append("")
    if overall:
        out += [f"Overall: {overall}", ""]
    out += [summary + f".  (tokens: {usage.get('total_tokens', '?')})"]
    report = "\n".join(out)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(report)

    return 0 if counts["COVERED"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
