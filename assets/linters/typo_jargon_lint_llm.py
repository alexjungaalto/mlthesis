#!/usr/bin/env python3
"""LLM linter: typos, weak grammar, and unexplained jargon in the thesis
SUMMARY section only.

It walks the manuscript PARAGRAPH BY PARAGRAPH, sends each paragraph to the
Aalto LLM API (see aalto_llm.py), and asks for three kinds of defect:

  typo     a misspelling, a doubled word ("the the"), a wrong/missing
           inflection, or a mangled token that is clearly a slip rather
           than a term of art
  grammar  a genuine grammatical error a copy-editor would fix: subject/verb
           disagreement, wrong or missing article, broken tense, a run-on or
           a sentence fragment, a mangled idiom
  jargon   a technical or colloquial term used without being defined or
           expanded on first use in this paragraph, OR an evaluative bit of
           ML-speak ("performs well", "smooth convergence") whose property
           is never measured -- i.e. wording a non-specialist reader could
           not follow

Every finding carries the PAGE NUMBER of the paragraph it came from (PDF
"p<n>" locations), so the author can jump straight to it.

WHY ONLY THE SUMMARY: an article-based dissertation reprints its published
papers verbatim at the back. Those are already peer-reviewed and typeset by
the publisher and are not the candidate's to re-edit, so linting them is
noise. This tool lints only the summary section (front matter, chapters,
bibliography, appendices) and STOPS at the first reprinted-paper cover page
("Paper I" / "Publication I" ...). Override the cutoff with --summary-end.

Gateway: the Aalto AI API by default ($AALTO_API_KEY, Aalto network/VPN
only), which keeps the unpublished manuscript inside Aalto's tenant.
--base-url switches gateways; sending a draft to a public service warns.

Usage:
  python3 typo_jargon_lint_llm.py thesis.pdf
  python3 typo_jargon_lint_llm.py thesis.pdf --summary-end 164
  python3 typo_jargon_lint_llm.py thesis.pdf --checks typo,grammar
  python3 typo_jargon_lint_llm.py thesis.pdf --concurrency 6 --limit 20
Exit status: 0 clean, 1 findings, 2 usage error.
"""

import argparse
import concurrent.futures
import json
import re
import sys
from typing import List, Optional, Tuple

from aalto_llm import (API_KEY_HELP, BASE_URL, BASE_URL_HELP, default_model,
                       extract_json, make_client)
from lintutil import (Report, is_toc_line, load_lines, paragraphs_from_lines)

CHECKS = ["typo", "grammar", "jargon"]

# A reprinted-paper cover page in an article-based dissertation: it starts
# with "Paper I" / "Publication I" ... + a roman numeral, and it is SHORT (a
# cover carries only the label and the paper's citation). The shortness gate
# is what separates a real cover from a same-worded heading inside the summary
# -- e.g. Chapter 7's "PUBLICATION III" section, which is a full page of text.
PAPER_COVER_RE = re.compile(r"^\s*(Paper|Publication|Article)\s+"
                            r"(I{1,3}|IV|V|VI{0,3}|IX|X)\b", re.I)
COVER_MAX_WORDS = 80   # a cover page has at most this many words

SYSTEM_PROMPT = (
    "You are a meticulous academic copy-editor proofreading paragraphs "
    "from the summary section of a doctoral dissertation in machine learning "
    "for the physical layer of wireless communications. The input contains "
    "one or more paragraphs, each introduced by a label of the form [P<n>] "
    "on its own (e.g. [P3]). Treat each labelled paragraph independently, and "
    "in EVERY finding you MUST report which paragraph it came from via a "
    "\"para\" field holding that exact label (e.g. \"P3\"). The text was "
    "extracted from a PDF, so ignore artefacts of extraction: hyphenation at "
    "line breaks, isolated numbers, table/figure fragments, math symbols "
    "rendered oddly, and stray spacing. Do NOT report those as errors.\n\n"
    "Report ONLY these defect categories:\n"
    "  typo: a real misspelling, a doubled word (e.g. 'the the'), a wrong or "
    "missing plural/inflection, or a garbled token that is plainly a slip. "
    "Do NOT flag British spelling, proper nouns, author names, or domain "
    "terms of art (e.g. 'Sionna', 'DeepRx', 'Mahalanobis', 'LMMSE').\n"
    "  grammar: a genuine grammatical error an editor would certainly fix -- "
    "subject/verb disagreement, a wrong or missing article, a broken or "
    "inconsistent tense within the sentence, a run-on, a sentence fragment, "
    "or a mangled English idiom/collocation (e.g. 'in a daily basis'). Do "
    "NOT flag stylistic preferences, the serial comma, or acceptable "
    "academic phrasings.\n"
    "  jargon: a technical or colloquial term used WITHOUT being defined or "
    "expanded on first use in this paragraph, in a way a careful "
    "non-specialist reader could not follow; OR an evaluative/hand-wavy "
    "claim of a PROPERTY (stability, smoothness, speed, robustness, good "
    "performance) that is asserted but never quantified in the paragraph "
    "('performs well', 'smooth convergence', 'the model struggles'). An "
    "acronym expanded in this paragraph, or an established term the summary "
    "clearly defines, is NOT jargon here.\n\n"
    "Be conservative: report only defects a human editor would definitely "
    "mark, and prefer FEWER, sharper findings over a long noisy list. It is "
    "correct and common to return an empty list for a clean paragraph. Each "
    "finding needs: the paragraph label (\"para\"), the category, an EXACT "
    "short quote (<=20 words) copied verbatim from that paragraph, a "
    "suggested correction (the fixed text, or for jargon a brief note on what "
    "to define/measure), and a severity (high | medium | low). Respond with "
    "STRICT JSON, no prose:\n"
    '{"findings": [{"para": "P<n>", "category": "typo|grammar|jargon", '
    '"severity": "high|medium|low", "quote": "...", '
    '"correction": "...", "explanation": "..."}]}'
)


def parse_page_range(s: str) -> Optional[Tuple[int, int]]:
    """Parse a --pages value 'lo-hi' (ordered) or single 'n' into (lo, hi)."""
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s or "")
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (min(lo, hi), max(lo, hi))
    m = re.match(r"^\s*(\d+)\s*$", s or "")
    if m:
        return (int(m.group(1)), int(m.group(1)))
    raise ValueError(f"Bad --pages value: {s!r}")


def page_of(loc: str) -> Optional[int]:
    """Page number from a 'p<n>' location, or None if not PDF-style."""
    m = re.match(r"^p(\d+)$", loc)
    return int(m.group(1)) if m else None


def batch_paras(paras: List[Tuple[str, str]],
                n: int) -> List[List[Tuple[str, str]]]:
    """Group the (loc, paragraph) list into batches of up to n paragraphs.
    n<=1 yields one paragraph per batch (strict paragraph-by-paragraph)."""
    n = max(1, n)
    return [paras[i:i + n] for i in range(0, len(paras), n)]


def _norm(s: str) -> str:
    """Loose normalisation for matching a model quote back to its source
    paragraph: collapse whitespace and drop non-alphanumerics, lowercased."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def locate_finding(batch: List[Tuple[str, str]], label: str,
                   quote: str) -> str:
    """Map a finding to the location of the paragraph it belongs to. Prefer
    the model's [P<n>] label; if that is missing or out of range, fall back to
    the batch paragraph whose text actually contains the quote; failing that,
    use the first paragraph of the batch. This keeps the page correct even
    when the model mislabels a finding."""
    m = re.match(r"\s*P?(\d+)\s*$", label or "")
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(batch):
            return batch[idx][0]
    q = _norm(quote)
    if len(q) >= 8:
        for loc, text in batch:
            if q in _norm(text):
                return loc
    return batch[0][0]


def detect_summary_end(lines: List[Tuple[str, str]]) -> Optional[int]:
    """Return the last PDF page that still belongs to the summary section:
    the page just before the first reprinted-paper cover ("Paper I" /
    "Publication I" ...). A cover both starts with the marker AND is short
    (<= COVER_MAX_WORDS), so a same-worded summary heading (e.g. Chapter 7's
    "PUBLICATION III", a full text page) is not mistaken for one. None if no
    cover is found (then the whole document is treated as summary). Only pages
    past the front matter (p >= 10) are considered."""
    first_line_by_page, words_by_page = {}, {}
    for loc, text in lines:
        pno = page_of(loc)
        if pno is None:
            continue
        words_by_page[pno] = words_by_page.get(pno, 0) + len(text.split())
        if text.strip():
            first_line_by_page.setdefault(pno, text.strip())
    for pno in sorted(first_line_by_page):
        if (pno >= 10
                and words_by_page.get(pno, 0) <= COVER_MAX_WORDS
                and PAPER_COVER_RE.match(first_line_by_page[pno])):
            return pno - 1
    return None


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(
        description="LLM typo/grammar/jargon linter for the thesis summary "
                    "(skips reprinted published articles).")
    ap.add_argument("inputs", nargs="+", help="thesis.pdf (PDF input expected)")
    ap.add_argument("--base-url", default=BASE_URL, help=BASE_URL_HELP)
    ap.add_argument("--api-key", default=None, help=API_KEY_HELP)
    ap.add_argument("--model", default=None,
                    help="Model id (default depends on the gateway).")
    ap.add_argument("--summary-end", type=int, default=None,
                    help="Last PDF page of the summary section to lint. "
                         "Default: auto-detect (stop before the first "
                         "reprinted 'Paper I'/'Publication I' cover page).")
    ap.add_argument("--pages", default=None,
                    help="Restrict to a page range, e.g. '9-120' (further "
                         "narrows the summary section).")
    ap.add_argument("--checks", default=",".join(CHECKS),
                    help=f"Comma-separated subset of: {', '.join(CHECKS)}")
    ap.add_argument("--min-severity", choices=["high", "medium", "low"],
                    default="low",
                    help="Drop findings below this severity "
                         "(high->ERROR, medium->WARN, low->INFO).")
    ap.add_argument("--min-chars", type=int, default=120,
                    help="Skip paragraphs shorter than this many characters "
                         "(default 120; drops headings and stray fragments).")
    ap.add_argument("--batch", type=int, default=1,
                    help="Paragraphs per LLM call (default 1 = strict "
                         "paragraph-by-paragraph). Higher values cut the "
                         "number of calls; each finding is still mapped back "
                         "to its own paragraph's page. Try --batch 10.")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Concurrent LLM calls (default 4).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many paragraphs (quick tests).")
    args = ap.parse_args(argv)

    checks = [c.strip() for c in args.checks.split(",") if c.strip()]
    bad = [c for c in checks if c not in CHECKS]
    if bad:
        print(f"ERROR: unknown check(s): {', '.join(bad)}", file=sys.stderr)
        return 2

    lines, mode = load_lines(args.inputs)
    if mode != "pdf":
        print("ERROR: this linter expects a PDF (page numbers come from the "
              "PDF); pass thesis.pdf.", file=sys.stderr)
        return 2

    # 1) Restrict to the summary section (skip the reprinted articles).
    end = args.summary_end if args.summary_end else detect_summary_end(lines)
    if end:
        kept = [(w, t) for (w, t) in lines
                if (page_of(w) or 0) <= end]
        how = ("given" if args.summary_end else "auto-detected")
        print(f"[info] summary section: pages 1-{end} ({how}); "
              f"reprinted articles after p{end} are skipped.", file=sys.stderr)
        lines = kept
    else:
        print("[info] no reprinted-paper cover found; linting the whole "
              "document. Use --summary-end N to set the cutoff.",
              file=sys.stderr)

    # 2) Optional further page-range narrowing.
    if args.pages:
        lo, hi = parse_page_range(args.pages)
        lines = [(w, t) for (w, t) in lines
                 if lo <= (page_of(w) or -1) <= hi]

    # 3) Drop TOC dot-leader lines and the bibliography (reference entries
    #    are not prose to proofread and only generate noise).
    lines = [(w, t) for (w, t) in lines if not is_toc_line(t)]
    body, in_refs = [], False
    for w, t in lines:
        if re.match(r"^\s*(References|Bibliography)\s*$", t.strip(), re.I):
            in_refs = True
        if not in_refs:
            body.append((w, t))

    # 4) Paragraph by paragraph.
    paras = [(w, p) for (w, p) in paragraphs_from_lines(body)
             if len(p) >= args.min_chars]
    if args.limit:
        paras = paras[: args.limit]
    if not paras:
        print("ERROR: no paragraphs extracted from the summary section.",
              file=sys.stderr)
        return 2

    batches = batch_paras(paras, args.batch)

    model = args.model or default_model(args.base_url)
    client = make_client(args.base_url, args.api_key)
    print(f"[info] gateway={args.base_url}\n"
          f"[info] model={model}  paragraphs={len(paras)}  "
          f"batch={max(1, args.batch)}  calls={len(batches)}  "
          f"checks={','.join(checks)}", file=sys.stderr)

    def judge(batch: List[Tuple[str, str]]):
        # One LLM call per batch of paragraphs. Each paragraph is labelled
        # [P<n>] so the model can attribute every finding; locate_finding then
        # maps it back to that paragraph's own page (with a quote-match
        # fallback), so the page stays exact regardless of batch size.
        blocks = []
        for i, (loc, text) in enumerate(batch, start=1):
            blocks.append(f"[P{i}] (from {loc})\n{text}")
        user = (f"requested categories: {json.dumps(checks)}\n\n"
                f"{len(batch)} paragraph(s) follow, each labelled [P<n>]:\n\n"
                + "\n\n".join(blocks))
        raw, usage = client.complete(model=model, system=SYSTEM_PROMPT,
                                     user=user, timeout=300,
                                     max_tokens=max(2000, 400 * len(batch)))
        parsed = extract_json(raw) or {}
        finds = parsed.get("findings", [])
        return batch, (finds if isinstance(finds, list) else []), usage

    rep = Report(f"Typo / grammar / jargon lint (summary only, {model})",
                 " ".join(args.inputs),
                 about="Paragraph-by-paragraph LLM proofread of the "
                       "dissertation summary section (reprinted published "
                       "articles are not linted). Findings are heuristic — "
                       "review each.")
    SEV_TAG = {"high": "ERROR", "medium": "WARN", "low": "INFO"}
    SEV_RANK = {"high": 2, "medium": 1, "low": 0}
    floor = SEV_RANK[args.min_severity]
    total_tokens = 0
    done = 0
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.concurrency)) as ex:
        for batch, finds, usage in ex.map(judge, batches):
            done += 1
            total_tokens += usage.get("total_tokens", 0)
            if finds:
                span = (batch[0][0] if len(batch) == 1
                        else f"{batch[0][0]}-{batch[-1][0]}")
                print(f"[progress] call {done}/{len(batches)} {span}: "
                      f"{len(finds)} finding(s)", file=sys.stderr)
            for f in finds:
                if not isinstance(f, dict):
                    continue
                cat = str(f.get("category", "")).strip().lower()
                if cat not in checks:
                    continue
                sev = str(f.get("severity", "medium")).strip().lower()
                if sev not in SEV_RANK:
                    sev = "medium"
                if SEV_RANK[sev] < floor:
                    continue
                quote = str(f.get("quote", "")).strip()
                corr = str(f.get("correction", "")).strip()
                expl = str(f.get("explanation", "")).strip()
                where = locate_finding(batch, str(f.get("para", "")), quote)
                tail = f" [-> {corr[:50]}]" if corr else ""
                rep.add(SEV_TAG[sev], cat.upper(), where,
                        f"\"{quote[:100]}\" — {expl[:140]}{tail}")

    print(rep.render())
    print(f"\nTotal tokens used: {total_tokens}")
    return rep.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
