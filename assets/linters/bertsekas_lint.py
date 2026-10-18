#!/usr/bin/env python3
r"""Linter: Bertsekas's SMALL RULES for mathematical writing (LaTeX sources).

D. P. Bertsekas, "Ten Simple Rules for Mathematical Writing" (MIT, April
2002; www.mit.edu/~dimitrib/Ten_Rules.pdf) sorts writing rules into three
kinds (slide 7): SMALL rules apply to a single sentence and are checkable;
BROAD rules ("be clear") are not verifiable; COMPOSITION rules relate the
parts of a document — the ten. This linter is the deterministic gate for
the small rules; the ten composition rules are the LLM review in
bertsekas_lint_llm.py. Ported 2026-09-17 from the dictionary suite's
check_bertsekas.py (~/dictionaryappliedml, fb15e37), re-cut from glossary
entries to a paper's .tex sources: the whole document after
\begin{document} is read, floats are blanked and displays masked, and
paragraphs are blank-line blocks.

Codes (advisory: every one is a candidate edit, not a defect by itself):
  [WARN] SENTSTART       a sentence that begins with mathematics (slide 8:
                         BAD "$f$ is continuous", GOOD "The function $f$ is
                         continuous"); also a DISPLAY placed after a full
                         stop, which starts a sentence with math — give it
                         a lead-in ending in a comma or colon. A compound
                         term ("$k$-means", "$\ell_1$-norm") is not math.
  [INFO] MATHSPEAK       "Let $k > 0$ be an integer" (slide 8: "Let $k$ be
                         a positive integer" or "Consider an integer
                         $k > 0$"); a definitional \defeq / = is not a
                         relation in this sense.
  [INFO] FOURLONG        a paragraph with more than four long sentences —
                         the third clause of the 2-3-4 rule (slide 8);
                         "long" = past two printed lines = --long-sent
                         words (default 30; a two-column paper sets ~12
                         words per line, so pass --long-sent 24 there).
  [INFO] TWOLINES        (--hints only) a sentence past --long-sent words,
                         the first clause of 2-3-4; a hint tier, since a
                         fifth of any technical text trips it.
  [INFO] BAREREF         an equation referenced by number alone — "the
                         solution of \eqref{eq:erm}", "Eq.~\eqref{..}", a
                         sentence opening on \eqref (slide 25: "frequent
                         numbered referencing is a cardinal sin ... refer
                         to equations by content/name, in addition to
                         number"). A named reference ("the update
                         \eqref{..}", "the SVM problem \eqref{..}") passes;
                         a generic noun ("Equation \eqref", "the equation
                         \eqref") does not. One is fine; the LLM review
                         judges the frequency.
  [INFO] WHICHTHAT       "which" opening a restrictive clause with no comma
                         ("a statistic $T$ which is ..."; slide 9): either
                         "that", or ", which". Not after a comma, a
                         preposition, a dash, a display, "and"/"or", and
                         not the interrogative "which devices".
  [INFO] ABBRCOMMA       "i.e." or "e.g." with no comma after it (slide 9:
                         "use abbreviations correctly"; American style).
  [WARN] LOOSETERM       the loose terms of slide 12: "random value(s)" /
                         "random sample(s)" (say random variable, or
                         realization), "chance(s)" (say probability).
  [WARN] UNUSEDNOTATION  a symbol introduced with "Let $x$", "denoted by
                         $x$" or "written $x$" and never used again in the
                         document's math (slide 21, "avoid unnecessary
                         notation"). Only a bare atom with decoration is
                         read ("\mA^{\top}", "t_{\text{obs}}"), never a
                         relation or a tuple; the count runs over every
                         math span of the document, boundary-aware so that
                         "\sigma\Phi" is a use of \Phi and "x_k" of k.

Usage:
  python3 bertsekas_lint.py paper.tex
  python3 bertsekas_lint.py main.tex chapters/ --hints --long-sent 24
  python3 bertsekas_lint.py --rule SENTSTART --rule BAREREF paper.tex
  python3 bertsekas_lint.py --selftest
Exit status: 0 clean, 1 findings (WARN or worse), 2 usage error.
"""

import argparse
import re
import sys
from typing import Iterator, List, Optional, Tuple

from lintutil import Report, sentences, strip_tex_comments, tex_files

CODES = ["SENTSTART", "MATHSPEAK", "FOURLONG", "TWOLINES", "BAREREF",
         "WHICHTHAT", "ABBRCOMMA", "LOOSETERM", "UNUSEDNOTATION"]
SEVERITY = {"SENTSTART": "WARN", "MATHSPEAK": "INFO", "FOURLONG": "INFO",
            "TWOLINES": "INFO", "BAREREF": "INFO", "WHICHTHAT": "INFO",
            "ABBRCOMMA": "INFO", "LOOSETERM": "WARN",
            "UNUSEDNOTATION": "WARN"}

LONG_SENT = 30        # words past two printed lines (single column)
FOUR_LONG = 4         # ... and a paragraph may carry this many of them

# ── masking (length-preserving, so offsets still map to source lines) ─────
_FLOAT = re.compile(r"\\begin\{(figure\*?|tikzpicture|table\*?|tabular|"
                    r"algorithm\*?|algorithmic|lstlisting|verbatim|"
                    r"minipage|wrapfigure)\}.*?\\end\{\1\}", re.S)
_DISPLAY = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]|"
                      r"\\begin\{(equation|align|gather|multline|eqnarray|"
                      r"alignat|flalign)\*?\}.*?\\end\{\1\*?\}", re.S)
_DISPLAY_CLOSER = re.compile(r"(\$\$|\\\]|\\end\{[a-z*]+\})\s*$")
_ALL_MATH = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]|"
                       r"\\begin\{(?:equation|align|gather|multline|eqnarray|"
                       r"alignat|flalign)\*?\}.*?\\end\{(?:equation|align|"
                       r"gather|multline|eqnarray|alignat|flalign)\*?\}|"
                       r"\$[^$]+\$", re.S)
DISP, DISP_END = "\x01", "\x02"     # a display; its last char when it ends a sentence


def _blank(s: str) -> str:
    return "".join("\n" if c == "\n" else " " for c in s)


def _display_ends_sentence(span: str) -> bool:
    inner = _DISPLAY_CLOSER.sub("", span.rstrip())
    inner = re.sub(r"(\\label\{[^}]*\}|\\nonumber|\\notag)\s*$", "",
                   inner).rstrip()
    return bool(re.search(r"(\\text\{\s*\.\s*\}|(?<!\\)\.)\s*$", inner))


def mask_prose(body: str) -> str:
    """Floats blanked, displays turned into DISP runs (DISP_END last when
    the display closes its sentence), everything else untouched. Length is
    preserved character for character."""
    t = _FLOAT.sub(lambda m: _blank(m.group(0)), body)

    def disp(m):
        s = m.group(0)
        run = "".join("\n" if c == "\n" else DISP for c in s)
        if _display_ends_sentence(s):
            run = run[:-1] + DISP_END
        return run
    return _DISPLAY.sub(disp, t)


# ── the small rules ───────────────────────────────────────────────────────
_ABBR_END = re.compile(
    r"(?<![A-Za-z])(?:e\.g|i\.e|cf|etc|vs|resp|Fig|Figs|Eq|Eqs|Sect|Sec|Ch|"
    r"Thm|Th|Prop|Lem|Cor|Def|Ex|No|al|St|w\.r\.t|s\.t|i\.i\.d)\.$|\d\.$")
_SENT_START = re.compile(
    r"(?:\A\s*|(?<=[.!?])\s+|(?<=" + DISP_END + r")\s*|\\\\\s*(?:\[[^\]]*\])?\s*|"
    r"\n\s*\n\s*)"
    r"(\$[^$]+\$|[" + DISP + DISP_END + r"]+)")


def find_sentstart(t: str) -> Iterator[Tuple[int, str, str, str]]:
    for m in _SENT_START.finditer(t):
        pre = t[max(0, m.start() - 16):m.start()].rstrip()
        if _ABBR_END.search(pre):
            continue
        tok = m.group(1)
        if tok.startswith("$") and t[m.end():m.end() + 1] == "-":
            continue                    # $k$-means: a compound term
        if tok.startswith("$"):
            hint = ("a sentence must not start with math: name the object "
                    "first (\"The function $f$ is continuous\", not \"$f$ "
                    "is continuous\")")
            shown = tok
        else:
            hint = ("a display after a full stop begins a sentence with "
                    "math: give it a lead-in that ends in a comma or colon, "
                    "so the display continues the sentence")
            shown = "[display]"
        yield m.start(), "SENTSTART", shown, hint


_MATHSPEAK = re.compile(r"\b[Ll]et\s+\$([^$]+)\$\s+(?:be|denote)\b")
_RELATION = re.compile(r"\\(?:in|notin|subset|subseteq|supset|supseteq|geq|"
                       r"leq|ge|le|neq|ne|succ|prec|succeq|preceq|perp|sim)"
                       r"\b|[<>]")


def find_mathspeak(t: str):
    for m in _MATHSPEAK.finditer(t):
        inner = m.group(1)
        if "\\defeq" in inner or "\\coloneqq" in inner or "=" in inner:
            continue        # "Let $d^{\star} \defeq \max ...$ be" defines
        if _RELATION.search(inner):
            yield (m.start(), "MATHSPEAK", m.group(0),
                   "readable mathspeak: \"Let $k$ be a positive integer\" or "
                   "\"Consider an integer $k > 0$\", not \"Let $k > 0$ be an "
                   "integer\"")


_EQREF = re.compile(r"\\eqref\{[^}]*\}|\(\\ref\{eq:[^}]*\}\)")
_BARE_BEFORE = set("""
of in to for by from via with into and or that as then see using use solving
solve solves satisfies satisfy satisfying becomes become minimizing
maximizing apply applying is are cf the a an this
""".split())
# a generic noun names nothing by itself ("Equation \eqref", "the equation
# \eqref") but is named when qualified ("the gradient-flow equation \eqref",
# "the SVM problem \eqref"), so it is bare only when the word before it is a
# function word or a sentence boundary
_GENERIC_NOUN = set("equation equations eq eqs expression expressions formula "
                    "formulas problem problems".split())


def find_bareref(t: str):
    for m in _EQREF.finditer(t):
        pre = t[max(0, m.start() - 60):m.start()]
        pre = re.sub(r"[~\s(]+$", "", pre)
        words = re.findall(r"[A-Za-z.-]+|[.!?]", pre)
        prev = words[-1] if words else ""
        pw = prev.rstrip(".").lower()
        bare = not prev or prev in ".!?" or pw in _BARE_BEFORE
        if pw in _GENERIC_NOUN:
            before = words[-2] if len(words) > 1 else ""
            bw = before.rstrip(".").lower()
            bare = (not before or before in ".!?" or bw in _BARE_BEFORE)
        if bare:
            yield (m.start(), "BAREREF", (prev + " " + m.group(0)).strip(),
                   "refer to the equation by content, not number alone: "
                   "\"the normal equations \\eqref{..}\", \"the update "
                   "\\eqref{..}\" (rule 8)")


_WHICH = re.compile(r"(\S+)\s+which\s+(?:is|are|was|were|has|have|had|does|"
                    r"do|did|can|cannot|may|must|should|will|would|could)\b")
_PREP = set("""
in on for of at to from by with under over about through into upon within
without after before during against among between along across via per
beyond towards toward
""".split())


def find_whichthat(t: str):
    for m in _WHICH.finditer(t):
        prev = m.group(1)
        if prev[-1] in ",(\u2014" or prev.endswith("---") \
                or prev[-1] in (DISP, DISP_END) or prev.endswith("\\]"):
            continue
        pw = re.sub(r"[^a-z]", "", prev.lower())
        if pw in _PREP or pw in ("and", "or"):
            continue            # "which are typical and which are rare"
        yield (m.start(), "WHICHTHAT", m.group(0),
               "a restrictive clause takes \"that\" and no comma; a "
               "nonrestrictive one \", which\" (the which/that rule)")


_ABBR_NOCOMMA = re.compile(r"\b(?:i\.e\.|e\.g\.)(?=[\s~\\])")


def find_abbrcomma(t: str):
    for m in _ABBR_NOCOMMA.finditer(t):
        yield (m.start(), "ABBRCOMMA", m.group(0),
               "a comma follows i.e. and e.g. (American style)")


_LOOSE = re.compile(r"\b(?:random\s+(?:values?|samples?)|chances?)\b", re.I)


def find_looseterm(t: str):
    for m in _LOOSE.finditer(t):
        w = m.group(0).lower()
        fix = ("say \"random variable\" (or \"realization\")"
               if w.startswith("random") else "say \"probability\"")
        yield (m.start(), "LOOSETERM", m.group(0),
               f"loose term (slide 12): {fix}")


_INTRO = re.compile(r"(?:\b[Ll]et\s+|\bdenoted?\s+(?:by\s+|as\s+)?|"
                    r"\bwrit(?:e|ten)\s+(?:as\s+)?)\$([^$]+)\$")
_ATOM = re.compile(r"^(?:\\[A-Za-z]+|[A-Za-z])"
                   r"(?:[\^_](?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\\?[A-Za-z0-9]+))*$")


def find_unusednotation(t: str, all_math: str):
    """all_math: every math span of the document, whitespace removed, in
    which the introduced symbol must occur at least twice (its introduction
    plus one use)."""
    for m in _INTRO.finditer(t):
        sym = re.sub(r"\s+", "", m.group(1))
        if not _ATOM.match(sym):
            continue
        lead = "" if sym.startswith("\\") else r"(?<![A-Za-z\\])"
        uses = re.findall(lead + re.escape(sym) + r"(?![A-Za-z])", all_math)
        if len(uses) <= 1:
            yield (m.start(), "UNUSEDNOTATION", m.group(0),
                   f"{sym} is introduced and never used again: drop the "
                   "symbol and state the fact in words (rule 4, "
                   "\"avoid unnecessary notation\")")


_PARA_SPLIT = re.compile(r"\n\s*\n")
_SECTION_CMD = re.compile(r"\\(?:sub)*section\*?\{[^}]*\}|\\paragraph\{[^}]*\}|"
                          r"\\(?:label|caption|cite[pt]?\*?|ref|eqref|autoref|"
                          r"cref|Cref)\{[^}]*\}")


def _clean_words(text: str) -> str:
    """Prose with structural commands removed, for word counts: math stays
    (macro names inside it count as words, as they roughly do on the page)."""
    t = _SECTION_CMD.sub(" ", text)
    t = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", " ", t)
    t = t.replace(DISP, " ").replace(DISP_END, ". ")
    return t


def find_long_paragraphs(t: str, long_sent: int, hints: bool):
    """FOURLONG per paragraph, TWOLINES per sentence (hints only), over the
    masked text. Yields (offset of paragraph, code, shown, hint)."""
    pos = 0
    for part in _PARA_SPLIT.split(t):
        off = t.find(part, pos) if part else pos
        pos = off + len(part)
        clean = _clean_words(part)
        sents = [s for s in sentences(clean) if s.strip()]
        long_ones = [s for s in sents if len(s.split()) > long_sent]
        if len(long_ones) > FOUR_LONG:
            yield (off, "FOURLONG",
                   f"{len(long_ones)} of {len(sents)} sentences over "
                   f"{long_sent} words — \"{sents[0][:50]}\"",
                   "2-3-4 rule: consider splitting a paragraph with more "
                   "than four long sentences")
        if hints:
            for s in long_ones:
                yield (off, "TWOLINES",
                       f"{len(s.split())} words — \"{s[:60]}\"",
                       "2-3-4 rule: consider splitting a sentence past two "
                       "printed lines")


def gate(body: str, hints: bool = False, only: Optional[set] = None,
         long_sent: int = LONG_SENT):
    """Run the deterministic gate over one document body.
    Yields (offset within body, code, shown, hint)."""
    all_math = re.sub(r"\s+", "", " ".join(
        m.group(0) for m in _ALL_MATH.finditer(body)))
    t = mask_prose(body)
    finders = [find_sentstart(t), find_mathspeak(t), find_bareref(t),
               find_whichthat(t), find_abbrcomma(t), find_looseterm(t),
               find_unusednotation(t, all_math),
               find_long_paragraphs(t, long_sent, hints)]
    for finder in finders:
        for off, code, shown, hint in finder:
            if only and code not in only:
                continue
            yield off, code, shown, hint


def document_body(raw: str) -> Tuple[str, int]:
    """The text after \\begin{document} (comments stripped line by line so
    line numbers survive), plus the number of lines skipped before it."""
    lines = [strip_tex_comments(ln) for ln in raw.splitlines()]
    start = 0
    for i, ln in enumerate(lines):
        if "\\begin{document}" in ln:
            start = i + 1
            break
    body = "\n".join(lines[start:])
    m = re.search(r"\\end\{document\}", body)
    if m:
        body = body[:m.start()]
    return body, start


# ── selftest ──────────────────────────────────────────────────────────────
_LONG = "Word " + " ".join(["word"] * 30) + "."
SELFTEST = [
    ("The trainset is fixed. $R^2 = 0$ means no improvement.", ["SENTSTART"]),
    ("$\\probdist$ is never observed.", ["SENTSTART"]),
    ("First paragraph.\n\n$R^2$ is widely reported.", ["SENTSTART"]),
    ("The set is closed. \\[ x = y \\text{.} \\] The next fact.", ["SENTSTART"]),
    ("The set is closed. $$ x = y. $$ The next fact.", ["SENTSTART"]),
    ("The error is orthogonal to $x$, \\[ 0 = a \\text{,} \\] which holds.",
     []),
    ("It reads: \\begin{equation} a = b \\text{.} \\end{equation} The next.",
     []),
    ("It reads \\[ a = b \\text{.} \\] $c$ is the constant.", ["SENTSTART"]),
    ("Consider the data. $k$-means partitions it.", []),
    ("A norm, e.g. $\\ell_1$, is convex.", ["ABBRCOMMA"]),
    ("The realizations of i.i.d. $\\datapoint^{(r)}$ are averaged.", []),
    ("See Fig. $2$ for the plot.", []),
    ("Let $k > 0$ be an integer.", ["MATHSPEAK"]),
    ("Let $\\featurevec \\in \\reals^{d}$ denote the feature vector.",
     ["MATHSPEAK"]),
    ("Let $k$ be a positive integer. The sum has $k$ terms.", []),
    ("Consider an integer $k > 0$.", []),
    ("Let $k$ be a positive integer.", ["UNUSEDNOTATION"]),
    ("Let $d^{\\star} \\defeq \\max_{x} q(x)$ be the dual value.", []),
    ("The solution of \\eqref{eq:erm} is unique.", ["BAREREF"]),
    ("Eq.~\\eqref{eq:x} shows the update.", ["BAREREF"]),
    ("\\eqref{eq:x} implies \\eqref{eq:y}.", ["BAREREF"]),
    ("The update \\eqref{eq:gd} is a contraction.", []),
    ("the gradient-flow equation \\eqref{eq:gf} with step $\\lrate$", []),
    ("every solution of the SVM problem \\eqref{eq:svm} has", []),
    ("Equation \\eqref{eq:x} delivers one number.", ["BAREREF"]),
    ("the equation \\eqref{eq:x} shows the update", ["BAREREF"]),
    ("The solution of (\\ref{eq:erm}) is unique.", ["BAREREF"]),
    ("a test statistic $T$ which is a function of the data", ["WHICHTHAT"]),
    ("the set, which is closed, contains the limit", []),
    ("the sense in which the bound holds", []),
    ("The network specifies which devices can interact.", []),
    ("which agents are typical and which are rare", []),
    ("It holds that \\[ a = b \\text{,} \\] which is immediate.", []),
    ("convex --- which is why the likelihood is concave", []),
    ("the label, i.e. the outcome, is observed", ["ABBRCOMMA"]),
    ("the label, i.e., the outcome, is observed", []),
    ("a penalty (e.g., ridge) restores uniqueness", []),
    ("labels replaced by random values carry no signal", ["LOOSETERM"]),
    ("a $10\\%$ chance of rain", ["LOOSETERM"]),
    ("the probability of rain, and the likelihood function", []),
    ("Its transpose is denoted by $\\mA^{\\top}$ and satisfies "
     "$(A^{\\top})_{i,j} = A_{j,i}$.", ["UNUSEDNOTATION"]),
    ("Let $\\weights$ be the parameters. Then $\\weights$ minimizes $f$.", []),
    ("Let $t_{\\text{obs}}$ be the observed value; reject if "
     "$t_{\\text{obs}} > c$.", []),
    ("Let $\\Phi$ be the cdf. Then $\\mu + \\sigma\\Phi^{-1}(U)$ is Gaussian.",
     []),
    ("Let $(\\paramspace, d)$ be a complete metric space.", []),
    # a float is blanked: math inside a table never starts a sentence
    ("The results follow. \\begin{table}\\centering $x$ is here\\end{table} "
     "Then more.", []),
    (" ".join([_LONG] * 5), ["FOURLONG"]),
    (" ".join([_LONG] * 4), []),
]
SELFTEST_HINTS = [
    (_LONG, ["TWOLINES"]),
]


def selftest() -> int:
    bad = 0
    for text, want in SELFTEST:
        got = sorted({c for _, c, _, _ in gate(text)})
        if got != sorted(want):
            bad += 1
            print(f"FAIL want={want!r} got={got!r}  {text[:60]}")
    for text, want in SELFTEST_HINTS:
        got = sorted({c for _, c, _, _ in gate(text, hints=True)})
        if got != sorted(want):
            bad += 1
            print(f"FAIL (hints) want={want!r} got={got!r}  {text[:60]}")
        got = sorted({c for _, c, _, _ in gate(text)})
        if got:
            bad += 1
            print(f"FAIL (no hints) want=[] got={got!r}  {text[:60]}")
    total = len(SELFTEST) + len(SELFTEST_HINTS)
    print(f"\n{total - bad}/{total} self-test case(s) pass.")
    return bad


# ── main ──────────────────────────────────────────────────────────────────
def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Bertsekas's small rules for mathematical writing: a "
                    "deterministic gate over LaTeX sources (advisory).")
    ap.add_argument("inputs", nargs="*", help=".tex files or directories")
    ap.add_argument("--rule", action="append", metavar="CODE",
                    help=f"only this code (repeatable): {', '.join(CODES)}")
    ap.add_argument("--hints", action="store_true",
                    help="also list sentences past two printed lines "
                         "(TWOLINES)")
    ap.add_argument("--long-sent", type=int, default=LONG_SENT,
                    help=f"words per 'two printed lines' (default "
                         f"{LONG_SENT}; ~24 for a two-column layout)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the pinned cases instead of the inputs")
    ap.add_argument("--profile", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.selftest:
        return 1 if selftest() else 0
    if not args.inputs:
        ap.error("give .tex inputs or --selftest")
    only = {r.upper() for r in args.rule} if args.rule else None
    if only and not only <= set(CODES):
        ap.error(f"unknown rule code(s): {sorted(only - set(CODES))}")

    files = tex_files(args.inputs)
    if not files:
        sys.exit("No .tex input found (this linter needs LaTeX sources).")
    rep = Report("Bertsekas small-rules report (mathematical writing)",
                 " ".join(args.inputs),
                 about="Deterministic gate for the checkable rules of "
                       "Bertsekas's 'Ten Simple Rules for Mathematical "
                       "Writing': sentences opening on math, unreadable "
                       "mathspeak, number-only equation references, "
                       "which/that, i.e./e.g. commas, loose terms, unused "
                       "notation, and over-long paragraphs. Advisory; the "
                       "ten composition rules are bertsekas_lint_llm.py.")
    for f in files:
        raw = f.read_text(encoding="utf-8", errors="replace")
        body, skipped = document_body(raw)
        for off, code, shown, hint in gate(body, args.hints, only,
                                           args.long_sent):
            line = body.count("\n", 0, off) + 1 + skipped
            flat = re.sub(r"\s+", " ", shown)[:90]
            rep.add(SEVERITY[code], code, f"{f.name}:{line}",
                    f"{flat!r} — {hint}")
    print(rep.render())
    return rep.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
