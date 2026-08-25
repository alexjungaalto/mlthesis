#!/usr/bin/env python3
"""LLM linter (LOCAL): is the ABSTRACT self-contained from elementary concepts?

Motivation. An abstract should be readable by someone who knows only the
ELEMENTARY vocabulary of machine learning -- and nothing about THIS paper. In
practice abstracts leak the authors' internal jargon: a bare "topology" with no
antecedent (what topology -- of a graph? a network?), or novel compounds like
"shared-encoder aggregation" and "graph-propagation dependence" that read as
defined terms but never are. The fast/terminology linters do not catch this:
they check consistency and dictionary spelling, not whether a first-time reader
can actually follow the abstract.

What "elementary" means here is not a matter of taste: it is pinned to the
Aalto Dictionary of Machine Learning (Jung). Every term the dictionary defines
(federated learning, graph, encoder, GNN, message passing, clustering, ...) is
fair game to use unexplained. Anything NOT in the dictionary must be either (a)
built transparently from elementary parts, or (b) defined inline at first use.
A term that is neither is a self-containment gap: a leap the reader cannot make.

The linter extracts the abstract, hands the model the dictionary's term list as
the elementary vocabulary, and asks it to classify every technical term / multi-
word construct in the abstract:

  ELEMENTARY      in the dictionary (or a standard synonym) -- fine, not reported
  DEFINED-INLINE  not elementary, but the abstract unpacks it at/near first use
  UNDEFINED       not in the dictionary, not defined -- a reader with only
                  elementary concepts cannot resolve it            [gap]
  AMBIGUOUS       a common/dictionary word used in a special, unstated sense,
                  e.g. "topology" (graph topology? network topology?)  [gap]
  COMPOUND-JARGON a novel multiword coinage that looks like a defined term but
                  is not, e.g. "shared-encoder aggregation",
                  "prediction-head collaboration"                   [gap]

Each gap comes with the naive-reader question it raises ("what topology?") and a
minimal fix (define inline in <=12 words, or swap for a dictionary term). The
abstract gets an overall grade GOOD / FAIR / POOR.

Heuristic aid, not a proofreader -- skim the report. Exit 1 if grade is POOR or
any gap is found; 0 if the abstract is self-contained; 2 usage error.

Grounding vocabulary: the Aalto Dictionary term list, parsed from a local
terms.txt snapshot (--dict, default
/Users/junga1/dictionaryappliedml/terms.txt). If it is missing the linter still
runs, but tells the model to fall back on general ML-basics knowledge and marks
the grounding as reduced.

Gateway: same as the other LLM linters (aalto_llm.py); defaults to the Aalto AI
API (--base-url can point at another OpenAI-style endpoint if ever needed).

Usage:
  python3 abstract_selfcontained_lint_llm.py submission.pdf
  python3 abstract_selfcontained_lint_llm.py submission.pdf --dict path/to/terms.txt
  python3 abstract_selfcontained_lint_llm.py submission.pdf --format markdown --out rep.md
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from aalto_llm import (API_KEY_HELP, BASE_URL, BASE_URL_HELP, default_model,
                       extract_json, make_client)

DEFAULT_DICT = "/Users/junga1/dictionaryappliedml/terms.txt"

# A term line in terms.txt looks like:
#   [IN DRAFT]  [100/100]  federated learning (FL) (fed_learning)
# i.e. "[status] [score] <display name> (<slug>)". The display name may itself
# contain parenthesised acronyms, so take the LAST parenthesised group as slug.
_TERM_RE = re.compile(r"^\[[^\]]*\]\s*\[[^\]]*\]\s+(.*?)\s*\(([^()]+)\)\s*$")


def load_dictionary(path: str) -> List[str]:
    """Return the dictionary's elementary vocabulary as display names.
    Empty list if the snapshot is unavailable."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    names = []
    for ln in raw.splitlines():
        m = _TERM_RE.match(ln)
        if m:
            name = m.group(1).strip()
            # Drop a trailing "(ACRONYM)" the display name may carry so the
            # vocabulary lists the phrase itself, e.g. "federated learning".
            name = re.sub(r"\s*\([^()]*\)\s*$", "", name).strip()
            if name:
                names.append(name)
    # De-duplicate, preserve order.
    seen, out = set(), []
    for n in names:
        k = n.lower()
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out


def load_text(path: str) -> List[str]:
    """Load a manuscript as reading-order text lines.

    A .txt extract is read directly. For a .pdf we deliberately do NOT use
    lintutil's `pdftotext -layout`: on a two-column paper -layout interleaves
    the abstract with the neighbouring column, so the 'Abstract' heading never
    lands on its own line. We use PyMuPDF's reading-order stream (as pdf_to_txt
    does), falling back to `pdftotext` WITHOUT -layout (reflowed, still in
    reading order)."""
    p = path.lower()
    if p.endswith(".txt"):
        return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if p.endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
            parts = []
            with fitz.open(path) as doc:
                for page in doc:
                    parts.append(page.get_text("text"))
            return "\n".join(parts).splitlines()
        except ImportError:
            pass
        try:
            out = subprocess.run(["pdftotext", path, "-"],  # no -layout
                                 capture_output=True, timeout=120)
            if out.returncode == 0:
                return out.stdout.decode("utf-8", errors="replace").splitlines()
        except (OSError, subprocess.TimeoutExpired):
            pass
        sys.exit("Need PyMuPDF or pdftotext (poppler) for PDF input.")
    sys.exit(f"Unsupported input: {path} (want .pdf or .txt).")


def extract_abstract(texts: List[str]) -> Tuple[str, str]:
    """Pull the abstract out of reading-order text lines.
    Returns (abstract_text, how) where how is 'abstract' if the heading was
    found, else 'fallback-head' (first ~250 words)."""
    joined = "\n".join(texts)
    # Locate an "Abstract" heading (its own line, case-insensitive), then read
    # until the Introduction heading (optionally numbered) or a blank-gap.
    start = None
    for i, t in enumerate(texts):
        if re.match(r"^\s*abstract\s*$", t, re.I):
            start = i + 1
            break
    if start is not None:
        body = []
        for t in texts[start:]:
            if re.match(r"^\s*(\d+[.\s]*)?introduction\s*$", t, re.I):
                break
            # Stop at the anonymised-submission boilerplate AAAI stamps in.
            if re.search(r"anonymi[sz]ed submission for review", t, re.I):
                continue
            body.append(t)
        abstract = "\n".join(body).strip()
        if len(abstract) > 60:
            return abstract, "abstract"
    # Fallback: first ~250 words of the document.
    words = joined.split()
    return " ".join(words[:250]).strip(), "fallback-head"


SYSTEM_PROMPT = (
    "You assess whether a paper's ABSTRACT is SELF-CONTAINED for a reader who "
    "knows only ELEMENTARY machine-learning concepts and NOTHING about this "
    "specific paper. 'Elementary' is defined precisely: the terms listed in "
    "the Aalto Dictionary of Machine Learning, given to you below as the "
    "ELEMENTARY VOCABULARY. A term in that vocabulary (or an obvious standard "
    "synonym) may be used with no explanation. Any other technical term or "
    "multiword construct must be either transparently built from elementary "
    "parts OR defined inline at first use; otherwise it is a self-containment "
    "gap -- a leap the intended reader cannot make.\n\n"
    "Scan the abstract and, for EVERY technical term or multiword construct, "
    "classify it:\n"
    "  ELEMENTARY      -- in the vocabulary (or a standard synonym). Do NOT "
    "report these.\n"
    "  DEFINED-INLINE  -- not elementary, but the abstract unpacks its meaning "
    "at or near first use. Do NOT report these.\n"
    "  UNDEFINED       -- not in the vocabulary and not defined; the reader "
    "cannot resolve it. REPORT.\n"
    "  AMBIGUOUS       -- a common or dictionary word used in a specialized, "
    "unstated sense (classic example: a bare 'topology' -- of a graph? a "
    "network? -- with no antecedent). REPORT.\n"
    "  COMPOUND-JARGON -- a novel multiword coinage that reads like a defined "
    "term but is not (e.g. 'shared-encoder aggregation', 'prediction-head "
    "collaboration', 'graph-propagation dependence'). REPORT.\n\n"
    "Be conservative and specific. Do NOT flag ordinary English, and do NOT "
    "flag a term merely because it is advanced IF the abstract defines it "
    "inline. The target is jargon that silently ASSUMES the reader already "
    "knows this paper. For each REPORTED gap give: the term verbatim; the "
    "code; the one-line question a first-time reader is left asking (e.g. "
    "'what topology?'); whether it appears to be the paper's own coinage "
    "(true/false); and a minimal fix -- either an inline gloss of <=12 words "
    "or a replacement using an elementary/dictionary term.\n\n"
    "Then assign an overall grade:\n"
    "  GOOD -- readable start-to-finish from elementary concepts; 0-1 minor "
    "gaps.\n"
    "  FAIR -- mostly readable but 2-4 gaps force guessing.\n"
    "  POOR -- several undefined/coined terms; the abstract assumes the "
    "reader already knows the paper.\n"
    "and a one-sentence verdict naming the single worst offender.\n\n"
    "Respond with STRICT JSON:\n"
    '{"grade": "GOOD|FAIR|POOR", "gaps": [{"term": "...", "code": '
    '"UNDEFINED|AMBIGUOUS|COMPOUND-JARGON", "question": "...", '
    '"coinage": true, "fix": "..."}], "verdict": "..."}'
)


def _vocab_block(vocab: List[str]) -> str:
    if not vocab:
        return ("ELEMENTARY VOCABULARY: (dictionary snapshot unavailable) -- "
                "fall back on general ML-basics knowledge for what counts as "
                "elementary, and note that grounding is reduced.")
    return ("ELEMENTARY VOCABULARY (Aalto Dictionary of ML, "
            f"{len(vocab)} terms; any of these may be used unexplained):\n"
            + ", ".join(vocab))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="LLM linter: is the abstract self-contained from "
                    "elementary (Aalto Dictionary) concepts?")
    ap.add_argument("pdf", help="Path to the submission PDF (or .txt extract).")
    ap.add_argument("--dict", default=DEFAULT_DICT,
                    help=f"Aalto Dictionary terms.txt (default {DEFAULT_DICT}).")
    ap.add_argument("--base-url", default=BASE_URL, help=BASE_URL_HELP)
    ap.add_argument("--api-key", default=None, help=API_KEY_HELP)
    ap.add_argument("--model", default=None,
                    help="Model id (default depends on the gateway).")
    ap.add_argument("--out", help="Write report to this file.")
    ap.add_argument("--format", choices=["text", "markdown"], default="text")
    args = ap.parse_args(argv)

    if not (args.pdf.lower().endswith(".pdf")
            or args.pdf.lower().endswith(".txt")):
        print("ERROR: this linter takes a compiled PDF or a .txt extract.",
              file=sys.stderr)
        return 2

    abstract, how = extract_abstract(load_text(args.pdf))
    if not abstract:
        print("ERROR: could not locate an abstract.", file=sys.stderr)
        return 2
    if how == "fallback-head":
        print("[warn] no 'Abstract' heading found; using the first ~250 words.",
              file=sys.stderr)

    vocab = load_dictionary(args.dict)
    if not vocab:
        print(f"[warn] dictionary snapshot not found at {args.dict}; grounding "
              "reduced.", file=sys.stderr)

    model = args.model or default_model(args.base_url)
    client = make_client(args.base_url, args.api_key)
    print(f"[info] gateway={args.base_url}\n[info] model={model}  "
          f"abstract_chars={len(abstract)}  dict_terms={len(vocab)}",
          file=sys.stderr)

    user = (f"{_vocab_block(vocab)}\n\n"
            f"ABSTRACT to assess:\n\"\"\"\n{abstract}\n\"\"\"")
    raw, usage = client.complete(model=model, system=SYSTEM_PROMPT,
                                 user=user, timeout=300, max_tokens=4000)
    parsed = extract_json(raw) or {}
    grade = str(parsed.get("grade", "")).upper() or "UNCLEAR"
    gaps = [g for g in parsed.get("gaps", []) if isinstance(g, dict)]
    verdict = str(parsed.get("verdict", "")).strip()

    md = args.format == "markdown"
    out = []
    if md:
        out += [f"# Abstract self-containment — {args.pdf}", "",
                f"_model: {model} · grounding: Aalto Dictionary "
                f"({len(vocab)} terms)_", "",
                f"**Grade: {grade}** — {len(gaps)} gap(s)", ""]
    else:
        out += [f"== Abstract self-containment lint (LLM, {model})",
                f"File: {args.pdf}",
                "Elementary = terms in the Aalto Dictionary of ML "
                f"({len(vocab)} terms). A gap = a term neither elementary "
                "nor defined inline.",
                "", f"GRADE: {grade}   ({len(gaps)} gap(s))", ""]

    if not gaps:
        out.append(("> " if md else "  ")
                   + (verdict or "Abstract is self-contained from elementary "
                      "concepts."))
    else:
        if md:
            out += ["## Gaps", ""]
        for g in gaps:
            term = str(g.get("term", "")).strip()
            code = str(g.get("code", "")).upper()
            q = str(g.get("question", "")).strip()
            coin = g.get("coinage")
            coin_s = "own coinage" if coin is True else (
                "" if coin is None else "standard-ish")
            fix = str(g.get("fix", "")).strip()
            if md:
                out += [f"### [{code}] `{term}`" + (f"  ·  _{coin_s}_"
                                                    if coin_s else ""),
                        f"- **reader asks:** {q}" if q else "",
                        f"- **fix:** {fix}" if fix else "", ""]
            else:
                tag = f"[{code}]"
                out.append(f"  {tag:<17} {term}"
                           + (f"   ({coin_s})" if coin_s else ""))
                if q:
                    out.append(f"        reader asks: {q}")
                if fix:
                    out.append(f"        fix:         {fix}")
        if verdict:
            out += (["", "## Verdict", "", verdict] if md
                    else ["", f"Verdict: {verdict}"])

    out = [x for x in out if x is not None]
    out += ["", f"grade={grade}, {len(gaps)} gap(s).  "
            f"(tokens: {usage.get('total_tokens', '?')})"]
    report = "\n".join(out)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(report)
    return 1 if (grade == "POOR" or gaps) else 0


if __name__ == "__main__":
    raise SystemExit(main())
