#!/usr/bin/env python3
r"""LLM linter: Bertsekas's ten COMPOSITION rules for mathematical writing.

D. P. Bertsekas, "Ten Simple Rules for Mathematical Writing" (MIT, April
2002; www.mit.edu/~dimitrib/Ten_Rules.pdf). The slides sort writing rules
into SMALL rules (one sentence, checkable — bertsekas_lint.py screens those
deterministically on the LaTeX sources), BROAD rules (unverifiable) and
COMPOSITION rules, which relate the parts of a document and are the ten:

  Structure     1. Organize in segments
                2. Write segments linearly
                3. Consider a hierarchical development
  Consistency   4. Use consistent notation and nomenclature
                5. State results consistently
                6. Don't underexplain — don't overexplain
  Readability   7. Tell them what you'll tell them
                8. Use suggestive references
                9. Consider examples and counterexamples
               10. Use visualization when possible

One call reviews the paper's MAIN TEXT (title through the paragraph before
the references; appendices are left out) with the SECTION as the segment:
Bertsekas's segment runs half a page to three pages, is read comfortably
start to end, and has an identifiable start and end with transition
material — a conference paper's section. The model also reports the small
rules a regex cannot see (three-verbs, strange-symbols, grading-words,
mathspeak outside the "Let $k>0$ be" form) and is told what the gate
already screens so it does not report those twice.

Verdicts:
  [WARN] RULE-<n>        a composition rule the paper fails, with a quote
                         and a concrete edit
  [INFO] SMALL-RULE      a small-rule hit (quote + suggestion)
  [INFO] RULE-<n>-PASS   (--verbose) a rule that passes, with the comment

Ported 2026-09-17 from the dictionary suite's check_bertsekas.py (LLM pass;
~/dictionaryappliedml, fb15e37), re-cut from a glossary entry to a paper:
the segment is the section, first-person voice is fine, theorem/proof
environments exist, and the audience is the venue's reviewer.

Gateway: the Aalto AI API v2 by default (see aalto_llm.py; $AALTO_API_KEY,
Aalto network/VPN only), on the endpoint's stronger model since ten
verdicts with quotes are a subtle judgment. The answer is long; the client
escalates the token budget if it is cut off.

Usage:
  python3 bertsekas_lint_llm.py paper.pdf
  python3 bertsekas_lint_llm.py paper.pdf --verbose --out report.txt
  python3 bertsekas_lint_llm.py paper.pdf --dry-run      # prompt only
Exit status: 0 clean, 1 findings, 2 usage error.
"""

import argparse
import re
import sys
from typing import List

from aalto_llm import (API_KEY_HELP, BASE_URL, BASE_URL_HELP,
                       default_strong_model, extract_json, make_client)
from lintutil import load_lines

MAX_CHARS = 90_000     # ~15k words; an ICLR main text is a third of that
REFS_RE = re.compile(r"^\s*(References|Bibliography)\s*$", re.I)

SYSTEM = """\
You review one machine-learning research paper against the ten COMPOSITION
rules of D. P. Bertsekas's "Ten Simple Rules for Mathematical Writing"
(MIT, 2002). You get the paper's main text as extracted from its PDF
(title, abstract, sections, displayed equations flattened to text, figure
and table captions; appendices omitted). Mathematics is set in LaTeX-like
notation or garbled by extraction — judge composition, not typesetting.

Judge each rule as it applies to a CONFERENCE PAPER. The SECTION is the
segment: Bertsekas's segment runs half a page to three pages, is read
comfortably start to end, stands alone, and has an identifiable start and
end with transition material. Its paragraphs are the blocks of that
segment.

 1. Organize in segments — does each section have a clear start (what it
    is about) and a clear end (it delivers something and does not trail
    off)? Does each paragraph have one topic? Flag a section that mixes
    unrelated topics, or a segment that is too long to read start to end.
 2. Write segments linearly — definitions, notation and auxiliary facts
    sit next to where they are first used ("minimize the thinking
    strain"; depth-first order). Flag a symbol or concept used before it
    is introduced, or a lemma or aside placed far from the argument that
    needs it.
 3. Consider a hierarchical development — background and technical
    machinery are separated from the main line (a preliminaries section,
    an appendix, a cited source) and reached by a pointer. Flag inline
    background that belongs in a separate segment, or a silent assumption
    of another paper's content with no pointer.
 4. Use consistent notation and nomenclature — one symbol per object and
    one name per concept across the paper; no unnecessary notation (not
    "Let X be a compact subset ... f attains a minimum over X" but "A
    continuous function attains a minimum over a compact set"); no loose
    terms ("random values" for random variables, "chance" for
    probability). Flag two names for one object, two symbols for one
    quantity, a symbol whose meaning changes, a symbol introduced for
    nothing.
 5. State results consistently — parallel statements (theorems,
    assumptions, algorithm variants, experimental conditions) in parallel
    form: same order of hypothesis and conclusion, same format. Flag
    needless variation between sibling statements.
 6. Don't underexplain — don't overexplain. The audience is the venue's
    expert reviewer. Flag a load-bearing step that skips too much for that
    reader (a proof sketch that is a claim, a design choice with no
    reason), and belabored trivia.
 7. Tell them what you'll tell them — the introduction says what the paper
    shows before developing it; each section opens on its topic; a result
    is announced by STATING it, then supporting it, never a string of
    statements that surprises the reader with the conclusion at the end
    ("we have thus shown ..."). Flag a section or paragraph whose point
    arrives only at its end.
 8. Use suggestive references — frequent numbered referencing is the
    cardinal sin. Judge the FREQUENCY of number-only references ("(3)
    implies (5)", "by Eq. (7)"); a name beside the number ("the coupled
    update (3)") is what he asks for. Flag also notation used long after
    its introduction without a reminder, and a simple formula referred to
    by number where repeating it would be shorter.
 9. Consider examples and counterexamples — an example should clarify the
    boundary of applicability; a counterexample shows why an assumption is
    needed. Flag a definition, assumption (convexity, Lipschitz, i.i.d.,
    connectivity, ...) or result whose boundary or necessity a one-line
    example or failure case would fix and the paper never gives.
10. Use visualization when possible — keep figures simple; captions are
    substantial and AUGMENT the text rather than repeat it; a figure
    carries the main idea where a picture can; prefer graphs over tables.
    Judge from the captions shown and any tabular material.

SMALL RULES: report only what a regex cannot see, each with a quote:
  three-verbs      a sentence with more than three verbs (the 2-3-4 rule)
  strange-symbols  symbols crowding running text — a sentence that reads
                   as a chain of inline math with connective words
  grading-words    "very", "trivial", "easy", "nice", "fundamental",
                   "novel" and the like used to grade a claim instead of
                   making it
  mathspeak        unreadable mathspeak OUTSIDE the "Let $k>0$ be" form
Already screened deterministically on the sources, so do NOT report: a
sentence starting with math, a display after a full stop, "Let $k>0$ be
...", sentences past two lines, paragraphs with more than four long
sentences, number-only equation references (judge their frequency under
rule 8 instead), which vs that, and commas after i.e./e.g.

First-person voice ("we") is normal in a paper; never flag it. Do not
flag PDF extraction artifacts (broken ligatures, hyphenation, garbled
math).

Be sparing: a well-edited paper passes most rules. Reserve findings for
defects a careful editor would act on; put nitpicks in the comment of a
"pass" verdict instead. Answer with a single JSON object, nothing else:
{"rules": [{"rule": 1, "name": "Organize in segments",
            "verdict": "pass|finding",
            "quote": "verbatim words from the paper (an EMPTY string when pass)",
            "where": "section or page the quote is from",
            "comment": "one or two short sentences",
            "suggestion": "concrete edit (an EMPTY string when pass)"},
           ... exactly one object per rule, 1 through 10 ...],
 "small_rule_findings": [{"small_rule": "three-verbs|strange-symbols|grading-words|mathspeak",
                          "quote": "...", "where": "...", "suggestion": "..."}],
 "summary": "two or three sentences on the paper's overall composition"}"""

RULE_NAMES = ["Organize in segments", "Write segments linearly",
              "Consider a hierarchical development",
              "Use consistent notation and nomenclature",
              "State results consistently",
              "Don't underexplain — don't overexplain",
              "Tell them what you'll tell them", "Use suggestive references",
              "Consider examples and counterexamples",
              "Use visualization when possible"]


def main_text(paths: List[str]) -> str:
    """The paper's main text: every line up to the References heading,
    page-tagged so the model can say where a quote is."""
    lines, mode = load_lines(paths)
    out, page = [], None
    for where, text in lines:
        if REFS_RE.match(text):
            break
        if mode == "pdf" and where != page:
            page = where
            out.append(f"\n[{page}]")
        text = re.sub(r"[ \t]{2,}", " ", text).strip()   # layout padding
        if text:
            out.append(text)
    body = "\n".join(out).strip()
    if len(body) > MAX_CHARS:
        body = body[:MAX_CHARS] + "\n[... truncated for length ...]"
    return body


def _text(field) -> str:
    v = (field or "").strip()
    return "" if v in ("''", '""') else v


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(
        description="LLM linter: Bertsekas's ten composition rules for "
                    "mathematical writing, over the paper's main text.")
    ap.add_argument("inputs", nargs="+", help="paper.pdf (or .tex sources)")
    ap.add_argument("--base-url", default=BASE_URL, help=BASE_URL_HELP)
    ap.add_argument("--api-key", default=None, help=API_KEY_HELP)
    ap.add_argument("--model", default=None,
                    help="Model id (default: the gateway's stronger model).")
    ap.add_argument("--verbose", action="store_true",
                    help="print all ten verdicts, not just findings")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt instead of calling the API")
    ap.add_argument("--out", help="Write report to this file.")
    ap.add_argument("--profile", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    body = main_text(args.inputs)
    if not body:
        print("ERROR: no main text found.", file=sys.stderr)
        return 2
    user = (f"Paper main text ({' '.join(args.inputs)}):\n\n{body}\n\n"
            "Review the paper against the ten composition rules and the "
            "small rules.")
    if args.dry_run:
        print(SYSTEM + "\n\n-----\n\n" + user)
        return 0

    model = args.model or default_strong_model(args.base_url)
    client = make_client(args.base_url, args.api_key)
    print(f"[info] gateway={args.base_url}\n[info] model={model}  "
          f"chars={len(body)}", file=sys.stderr)
    raw, usage = client.complete(model=model, system=SYSTEM, user=user,
                                 timeout=900, max_tokens=16000)
    v = extract_json(raw)

    out = [f"== Bertsekas composition report (LLM, {model})",
           f"File: {' '.join(args.inputs)}",
           "Ten composition rules of 'Ten Simple Rules for Mathematical "
           "Writing', judged over the main text with the section as the "
           "segment; advisory.", ""]
    n_find = n_small = 0
    if not v or not isinstance(v.get("rules"), list):
        out.append("[WARN] NO-VERDICT       -            no parseable "
                   "verdict from the model")
        n_find = 1
    else:
        rules = sorted(v["rules"], key=lambda r: r.get("rule", 0))
        if len(rules) != 10:
            print(f"[warn] model returned {len(rules)} rule verdicts "
                  f"(expected 10)", file=sys.stderr)
        for r in rules:
            n = r.get("rule", "?")
            name = r.get("name") or (RULE_NAMES[n - 1]
                                     if isinstance(n, int) and 1 <= n <= 10
                                     else "")
            where = _text(r.get("where")) or "-"
            if r.get("verdict") == "finding":
                n_find += 1
                out.append(f"[WARN] RULE-{n:<13} {where:<12} {name}: "
                           f"{_text(r.get('comment'))}")
                if _text(r.get("quote")):
                    out.append(f"        \"{_text(r['quote'])[:160]}\"")
                if _text(r.get("suggestion")):
                    out.append(f"        fix: {_text(r['suggestion'])}")
            elif args.verbose:
                out.append(f"[INFO] RULE-{n}-PASS{'':<7} {where:<12} {name}: "
                           f"{_text(r.get('comment'))}")
        for s in v.get("small_rule_findings") or []:
            n_small += 1
            where = _text(s.get("where")) or "-"
            out.append(f"[INFO] SMALL-RULE       {where:<12} "
                       f"{s.get('small_rule', '?')}: "
                       f"\"{_text(s.get('quote'))[:120]}\"")
            if _text(s.get("suggestion")):
                out.append(f"        fix: {_text(s['suggestion'])}")
        out += ["", _text(v.get("summary"))]
    out += ["", f"10 rule(s) judged: {10 - n_find} pass, {n_find} "
                f"finding(s), {n_small} small-rule hit(s).  "
                f"(tokens: {usage.get('total_tokens', '?')})"]
    report = "\n".join(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(report)
    return 1 if n_find else 0


if __name__ == "__main__":
    raise SystemExit(main())
