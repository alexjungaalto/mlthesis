#!/usr/bin/env python3
"""LLM linter (LOCAL): are the paper's CENTRAL concepts given a SOURCE?

The existing checks catch two neighbouring problems but not this one:
  * ``prose_lint_llm`` UNCITED-CLAIM flags uncited *sentences* (empirical or
    literature claims), not the concepts a paper is built on.
  * ``abstract_selfcontained_lint_llm`` flags *undefined* terms in the abstract,
    i.e. a reader cannot resolve them -- a definition problem, not a provenance
    one. A term can be perfectly well defined inline yet still be lifted from
    prior work with no citation to whoever introduced it.

This linter targets provenance of the load-bearing ideas. It first identifies
the CENTRAL concepts -- the notions the paper's problem, method or contribution
actually rests on (those in the title, abstract, research questions and the
"our contributions" list, or used pervasively throughout) -- and then, for each,
asks whether the paper points to a SOURCE for it. A central concept that the
paper borrows from prior work but introduces with neither a citation nor an
attribution of origin is the defect (UNCITED).

Per central concept it reports one of:
  CITED         introduced/backed by a citation to an originating or defining
                source -- fine, not a finding.
  OWN-COINAGE   the paper's OWN coinage/framework/notation, defined inline; it
                originates here, so no external source is owed -- fine.
  ELEMENTARY    textbook/standard ML vocabulary (grounded in the Aalto
                Dictionary of ML, Jung) that needs no citation -- fine.
  UNCITED       central, non-elementary, evidently borrowed from prior work, yet
                introduced with no citation and no origin attribution     [gap]
  ATTR-VAGUE    attributed only vaguely ("prior work", "it is known") with no
                actual reference the reader can follow                     [gap]

For every UNCITED / ATTR-VAGUE concept it gives: why the concept is central,
the page of first substantive use, whether it is defined inline (definition and
provenance are independent), and the concrete fix (cite the origin, or -- if the
concept really is novel here -- state it as the paper's own and define it).

This is a reviewer aid, not an acceptance verdict. Exit 1 if any central concept
is UNCITED/ATTR-VAGUE; 0 if all are sourced/own/elementary; 2 usage error.

Elementary vocabulary is pinned to the Aalto Dictionary of ML snapshot
(--dict, default /Users/junga1/dictionaryappliedml/terms.txt); if it is missing
the linter still runs and leans on the model's own sense of "textbook standard".

Gateway: same as the other LLM linters (aalto_llm.py); defaults to the Aalto
AI API.

Usage:
  python3 central_concept_citation_lint_llm.py submission.pdf
  python3 central_concept_citation_lint_llm.py submission.pdf --out s.txt
  python3 central_concept_citation_lint_llm.py submission.pdf --format markdown
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

from aalto_llm import (API_KEY_HELP, BASE_URL, BASE_URL_HELP, default_model,
                       extract_json, make_client)
from lintutil import load_lines

DEFAULT_DICT = "/Users/junga1/dictionaryappliedml/terms.txt"

# A term line in terms.txt looks like:
#   [IN DRAFT]  [100/100]  federated learning (FL) (fed_learning)
_TERM_RE = re.compile(r"^\[[^\]]*\]\s*\[[^\]]*\]\s+(.*?)\s*\(([^()]+)\)\s*$")


def load_dictionary(path: str) -> List[str]:
    """Return the dictionary's elementary vocabulary as display names.
    Empty list if the snapshot is unavailable."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    names, seen = [], set()
    for ln in raw.splitlines():
        m = _TERM_RE.match(ln)
        if not m:
            continue
        name = re.sub(r"\s*\([^()]*\)\s*$", "", m.group(1).strip()).strip()
        k = name.lower()
        if name and k not in seen:
            seen.add(k)
            names.append(name)
    return names


SYSTEM_PROMPT = (
    "You are an experienced, critical reviewer for a top ML venue. You are "
    "given the extracted text of a submission (page markers '[[page N]]' "
    "included) and a list of ELEMENTARY machine-learning terms (the Aalto "
    "Dictionary of ML). Your job is to check the PROVENANCE of the paper's "
    "CENTRAL concepts: does the paper point to a SOURCE for each idea it is "
    "built on?\n\n"
    "STEP 1 -- FIND THE CENTRAL CONCEPTS. A concept is CENTRAL only if the "
    "paper's problem, method, or contribution genuinely rests on it: it appears "
    "in the title, abstract, research questions, or the 'our contributions' "
    "list, OR it is used pervasively as load-bearing machinery. Name 4-8 such "
    "concepts. Do NOT list peripheral terms, passing mentions, or every "
    "acronym. A named method/metric/dataset/framework the paper adopts counts; "
    "a word used once in related work does not.\n\n"
    "STEP 2 -- CLASSIFY EACH CONCEPT'S SOURCE. Assign exactly one status:\n"
    "  CITED        introduced or backed by a citation to an originating or "
    "defining source (a [n] / (Author, year) near where the concept is "
    "introduced or first used substantively). Fine.\n"
    "  OWN-COINAGE  the paper's OWN coinage, framework, notation, or system "
    "name, defined inline -- it originates in THIS paper, so no external source "
    "is owed. Fine. (Be careful: a fancy name is not automatically the "
    "authors' invention; it is OWN-COINAGE only if the text presents it as "
    "theirs.)\n"
    "  ELEMENTARY   standard textbook vocabulary -- in the elementary list "
    "provided, an obvious synonym of one, or uncontroversially textbook-"
    "standard. Needs no citation. Fine.\n"
    "  UNCITED      central, NOT elementary, and evidently drawn from prior "
    "work (a named technique/metric/model/theory that exists in the "
    "literature), yet introduced with NO citation and NO attribution of "
    "origin. This is the primary defect.\n"
    "  ATTR-VAGUE   attributed only vaguely -- 'prior work', 'it is known', "
    "'the standard X' -- with no actual reference the reader can follow.\n\n"
    "Definition and provenance are INDEPENDENT: a concept can be defined inline "
    "yet still be UNCITED (defined but not sourced), and it can be CITED yet "
    "undefined. Judge only provenance here. Set 'defined_inline' true/false so "
    "the fix can be precise. When unsure whether a concept is the authors' own "
    "versus borrowed, prefer UNCITED and say so in the rationale -- a missing "
    "citation is cheap to add and the safer default.\n\n"
    "For the 'central' field, say in one clause WHY the concept is load-bearing "
    "(e.g. 'names the paper's core method', 'the metric all results report'). "
    "For UNCITED/ATTR-VAGUE, 'fix' must be concrete: name what to cite (the "
    "originating work for concept X), or -- if it truly is novel here -- say to "
    "state it as the paper's own and define it. Give 'first_use' as the page "
    "marker of first substantive use, e.g. 'p1'.\n\n"
    "Keep every field CONCISE: at most 1-2 sentences; paraphrase, never paste "
    "long passages. Output MUST be a single valid JSON object and nothing "
    "else.\n\n"
    "Respond with STRICT JSON:\n"
    '{"concepts": [{"id": "K1", "concept": "...", "central": "...", '
    '"status": "CITED|OWN-COINAGE|ELEMENTARY|UNCITED|ATTR-VAGUE", '
    '"first_use": "p1", "defined_inline": true, "rationale": "...", '
    '"fix": "..."}], "overall": "..."}'
)

STATUSES = ("CITED", "OWN-COINAGE", "ELEMENTARY", "UNCITED", "ATTR-VAGUE")
FAIL_STATUSES = {"UNCITED", "ATTR-VAGUE"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="LLM linter: are the paper's central concepts given a "
                    "source (citation), or borrowed silently?")
    ap.add_argument("pdf", help="Path to the submission PDF (or .txt extract).")
    ap.add_argument("--dict", default=DEFAULT_DICT,
                    help="Aalto Dictionary terms.txt snapshot (elementary "
                         "vocabulary that needs no citation).")
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

    vocab = load_dictionary(args.dict)
    model = args.model or default_model(args.base_url)
    client = make_client(args.base_url, args.api_key)
    print(f"[info] gateway={args.base_url}\n[info] model={model}  "
          f"chars={len(text)}  dict_terms={len(vocab)}", file=sys.stderr)

    elem = ", ".join(vocab) if vocab else "(dictionary snapshot unavailable)"
    user = (f"ELEMENTARY VOCABULARY (needs no citation):\n{elem}\n\n"
            f"submission text{' (TRUNCATED)' if truncated else ''}:\n"
            f'"""\n{text}\n"""')
    raw, usage = client.complete(model=model, system=SYSTEM_PROMPT,
                                 user=user, timeout=600, max_tokens=4000)
    parsed = extract_json(raw) or {}
    concepts = [c for c in parsed.get("concepts", []) if isinstance(c, dict)]
    overall = str(parsed.get("overall", "")).strip()

    norm = []
    for i, c in enumerate(concepts, 1):
        status = str(c.get("status", "UNCITED")).strip().upper()
        if status not in STATUSES:
            status = "UNCITED"
        norm.append({
            "id": str(c.get("id") or f"K{i}").strip(),
            "concept": str(c.get("concept", "")).strip(),
            "central": str(c.get("central", "")).strip(),
            "status": status,
            "first_use": str(c.get("first_use", "")).strip(),
            "defined_inline": bool(c.get("defined_inline", False)),
            "rationale": str(c.get("rationale", "")).strip(),
            "fix": str(c.get("fix", "")).strip(),
        })
    tally = {s: sum(1 for c in norm if c["status"] == s) for s in STATUSES}
    gaps = sum(tally[s] for s in FAIL_STATUSES)

    md = args.format == "markdown"
    out = []
    summary = (f"{len(norm)} central concept(s): {tally['CITED']} cited, "
               f"{tally['OWN-COINAGE']} own-coinage, {tally['ELEMENTARY']} "
               f"elementary, {tally['UNCITED']} uncited, "
               f"{tally['ATTR-VAGUE']} vaguely-attributed")
    if md:
        out += [f"# Central-concept sourcing — {args.pdf}", "",
                f"_model: {model}_", "", f"**{summary}**", ""]
    else:
        out += [f"== Central-concept citation lint (LLM, {model})",
                f"File: {args.pdf}", "", f"SUMMARY: {summary}", "",
                "How to read: each concept the paper is built on, and whether "
                "it points to a source. UNCITED / ATTR-VAGUE are the gaps; "
                "own-coinage and elementary vocabulary owe no citation.", ""]

    # order: gaps first (UNCITED, ATTR-VAGUE), then the rest.
    order = {"UNCITED": 0, "ATTR-VAGUE": 1, "CITED": 2, "OWN-COINAGE": 3,
             "ELEMENTARY": 4}
    for c in sorted(norm, key=lambda x: order.get(x["status"], 0)):
        loc = f" ({c['first_use']})" if c["first_use"] else ""
        dfn = "defined inline" if c["defined_inline"] else "not defined inline"
        if md:
            out.append(f"### `{c['status']}` — {c['id']} {c['concept']}{loc}")
            out.append("")
            out.append(f"- **central:** {c['central'] or '—'}")
            out.append(f"- **{dfn}**")
            if c["rationale"]:
                out.append(f"- **why:** {c['rationale']}")
            if c["status"] in FAIL_STATUSES and c["fix"]:
                out.append(f"- **fix:** {c['fix']}")
            out.append("")
        else:
            out.append(f"[{c['status']}]  {c['id']}  {c['concept']}{loc}")
            out.append(f"    central:  {c['central'] or '—'}")
            out.append(f"    ({dfn})")
            if c["rationale"]:
                out.append(f"    why:      {c['rationale']}")
            if c["status"] in FAIL_STATUSES and c["fix"]:
                out.append(f"    fix:      {c['fix']}")
            out.append("")

    if overall:
        out += [(f"**Overall:** {overall}" if md else f"Overall: {overall}"), ""]

    out += [summary + f".  (tokens: {usage.get('total_tokens', '?')})"]
    report = "\n".join(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(report)
    return 1 if gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
