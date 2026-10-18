#!/usr/bin/env python3
"""LLM linter: quality of a thesis's research scope and research questions.

Judges HOW GOOD the stated research questions and scope are --- not whether
they are answered (that is research_questions_lint_llm.py). The criteria
follow authoritative university guidance:

  * University research-question checklists (e.g. Monash University
    Library, "Developing research questions"): a good question is focused
    on a single problem, researchable with available sources/methods,
    feasible within the time frame, specific enough to answer thoroughly,
    complex enough to require analysis, and relevant to the field.
  * George Mason University Writing Center, "How to Write a Research
    Question": clear ("enough specifics that one's audience can easily
    understand its purpose"), focused ("narrow enough that it can be
    answered thoroughly"), concise, complex ("not answerable with a simple
    'yes' or 'no'"), and arguable ("potential answers are open to debate").
  * The FINER criteria (Hulley et al., Designing Clinical Research):
    Feasible, Interesting, Novel, Ethical, Relevant.
  * Self-containment (an ml-theses.org addition): every technical term or
    acronym the question uses is actually DEFINED -- glossed, expanded, or
    given a "let X denote ..." -- not merely mentioned, at or before the
    point where the question is stated, so the question is understandable
    where it appears rather than only after later chapters define its terms.

Per research question: verdict STRONG / ADEQUATE / WEAK with per-criterion
flags and a suggested reformulation for anything less than STRONG.
The self-containment criterion above is READER-POSITION: a term counts as
defined if the paper defines it anywhere before the RQ. A second, STANDALONE
pass (on by default; --no-standalone to skip) grades each extracted RQ the
way abstract_selfcontained_lint_llm.py grades the abstract: against the
ELEMENTARY vocabulary of the Aalto Dictionary of ML only, ignoring the rest
of the paper. It catches the anaphora and own coinages that pass the
reader-position test ('this platform', 'T4 (Decision Explainability)',
'tamper-evident local attribution records') -- an RQ should be readable by
someone who has seen nothing but the dictionary.
  [WARN] NOT-STANDALONE   RQ graded POOR/FAIR standalone, with the gaps
  [INFO] STANDALONE       RQ graded GOOD standalone
Scope-level checks: research gap identified, delimitations stated,
alignment between questions and stated objectives/contributions, and joint
coverage of the thesis aim.

  [WARN] WEAK-RQ / NO-RQS / scope-level gaps
  [INFO] STRONG-RQ / ADEQUATE-RQ and satisfied scope checks

Uses the full GPT-5 model by default on the Aalto AI API (the judgement is
calibration-sensitive). --base-url switches gateways (see aalto_llm.py).

Usage:
  python3 rq_quality_lint_llm.py thesis.pdf
  python3 rq_quality_lint_llm.py thesis.pdf --format markdown --out rq.md
Exit status: 0 no warnings, 1 findings, 2 usage error.
"""

import argparse
import sys
from typing import List

from aalto_llm import (API_KEY_HELP, BASE_URL, BASE_URL_HELP, default_model,
                       default_strong_model, extract_json, is_responses_api,
                       make_client)
from lintutil import load_lines
from abstract_selfcontained_lint_llm import (DEFAULT_DICT, _vocab_block,
                                             load_dictionary)

STANDALONE_PROMPT = (
    "You assess whether each RESEARCH QUESTION below is SELF-CONTAINED for a "
    "reader who knows only ELEMENTARY machine-learning concepts and NOTHING "
    "about this paper -- not its title, abstract, platform, or taxonomy. "
    "'Elementary' is defined precisely: the terms in the ELEMENTARY "
    "VOCABULARY (the Aalto Dictionary of Machine Learning). A term in that "
    "vocabulary (or an obvious standard synonym) may be used unexplained. "
    "Any other technical term, label code, named entity, or multiword "
    "construct must be either transparently built from elementary parts OR "
    "defined inside the question itself; otherwise it is a gap.\n\n"
    "Judge each question IN ISOLATION. Classify every technical term or "
    "construct it uses:\n"
    "  ELEMENTARY      -- in the vocabulary (or standard synonym). Do NOT "
    "report.\n"
    "  DEFINED-INLINE  -- unpacked inside the question. Do NOT report.\n"
    "  UNDEFINED       -- not elementary, not defined in the question. "
    "REPORT.\n"
    "  AMBIGUOUS       -- a demonstrative or common word with no antecedent "
    "inside the question ('this platform', 'the system', 'the mechanism', "
    "a bare 'topology'). REPORT.\n"
    "  COMPOUND-JARGON -- a coinage or label code that reads like a defined "
    "term but is not ('T4 (Decision Explainability)', 'tamper-evident local "
    "attribution records'). REPORT.\n\n"
    "Be conservative: do not flag ordinary English or a term the question "
    "glosses itself. For each gap give the term verbatim, the code, the "
    "one-line question a first-time reader is left asking, whether it looks "
    "like the paper's own coinage (true/false), and a minimal fix (an "
    "inline gloss of <=12 words or an elementary replacement). Grade each "
    "question GOOD (0-1 minor gaps), FAIR (2-3 gaps force guessing) or POOR "
    "(the question presupposes the paper), with a one-sentence verdict "
    "naming the worst offender.\n\n"
    "Respond with STRICT JSON:\n"
    '{"questions": [{"id": "RQ1", "grade": "GOOD|FAIR|POOR", "gaps": '
    '[{"term": "...", "code": "UNDEFINED|AMBIGUOUS|COMPOUND-JARGON", '
    '"question": "...", "coinage": true, "fix": "..."}], '
    '"verdict": "..."}]}'
)

SYSTEM_PROMPT = (
    "You are an experienced examiner judging the QUALITY of the research "
    "scope and research questions of a master's thesis in machine "
    "learning -- not whether they are answered, but whether they are "
    "well-posed. You receive the thesis text with '[[page N]]' markers.\n\n"
    "STEP 1 -- EXTRACT. Quote verbatim (lightly shortened is fine), with "
    "pages: the main aim; every explicitly stated research question or "
    "hypothesis; stated objectives/contributions; stated delimitations; "
    "the research-gap statement. Do not invent items; empty lists are "
    "valid.\n\n"
    "STEP 2 -- JUDGE EACH RESEARCH QUESTION against this checklist, "
    "compiled from university guidance (Monash University Library; George "
    "Mason University Writing Center; the FINER criteria):\n"
    "  clear: enough specifics that a reader understands its purpose "
    "without extra explanation (weak: 'How should social networking "
    "sites address the harm they cause?'; strong: names the actors, the "
    "harm, and the aspect of interest).\n"
    "  focused: a SINGLE problem, answerable thoroughly in the space of "
    "a thesis; flag double-barreled questions that bundle two askables.\n"
    "  concise: expressed without filler.\n"
    "  complex: not answerable with a bare 'yes'/'no' or a lookup; "
    "requires synthesis or quantified analysis. An engineering question "
    "in yes/no FORM ('Can X beat Y?') that plainly demands quantified "
    "evidence is a minor form issue -- suggest 'To what extent ...' "
    "phrasing -- whereas a question that a single fact settles is a "
    "substantive defect.\n"
    "  specific: key concepts are precise and measurable, not vague "
    "('effective', 'good performance') without stated metrics.\n"
    "  self-contained: every non-generic technical term, acronym, or "
    "named entity IN THE QUESTION must be actually DEFINED -- not merely "
    "mentioned -- at or BEFORE the page where the question is stated. A "
    "DEFINITION is a parenthetical gloss ('the agent harness (the "
    "non-model infrastructure between the LLM and the target system)'), a "
    "'let X denote ...' / 'X is defined as ...' / 'we call X ...' "
    "sentence, an acronym expansion, a Definition environment, or a "
    "notation/glossary entry. A bare earlier OCCURRENCE of the term with "
    "no explanation is 'mentioned, NOT defined' and does NOT satisfy this "
    "criterion. Check strictly by page order using the '[[page N]]' "
    "markers: the defining page must be <= the question's page; a "
    "definition that appears only in a LATER section (even one page later) "
    "does NOT count -- never justify self-containment by citing a section "
    "after the question. When you flag it, say whether the term is "
    "undefined entirely or only mentioned-not-defined before the RQ, and "
    "name the term, the RQ page, and the page where it is actually first "
    "defined (e.g. \"'agent harness' appears in the RQ on p10 but is only "
    "mentioned before then; first defined in 2.6 on p14\"). A term defined "
    "in the question's own sentence, or a genuinely common-ML term "
    "(accuracy, dataset, neural network), counts as satisfied. This is "
    "reader-POSITION self-containment -- distinct from 'clear' (rubric "
    "well-posedness in isolation): a question can be clear to an expert "
    "yet not self-contained. Mentioned-but-not-defined, or defined just a "
    "page or two later, is a minor issue; a term whose definition depends "
    "on a whole later chapter is substantive.\n"
    "  researchable-feasible: answerable with the data, methods, and "
    "time an MSc allows; not requiring inaccessible populations or "
    "unbounded experiments.\n"
    "  relevant-novel: tied to an identified gap; the answer matters to "
    "the field (FINER: interesting, novel, relevant).\n"
    "  arguable: the answer is genuinely open, not a foregone "
    "conclusion or definitional truth.\n\n"
    "Verdicts: STRONG (no criterion meaningfully violated), ADEQUATE "
    "(minor issues, e.g. yes/no form with obvious quantitative intent), "
    "WEAK (one or more substantive violations). For anything below "
    "STRONG give a concrete reformulation.\n\n"
    "STEP 3 -- JUDGE THE SCOPE as a whole: gap_identified (explicit "
    "research-gap statement exists), delimitations_stated (what is out "
    "of scope is said), aligned (each question maps to stated "
    "objectives/contributions and vice versa), covers_aim (the questions "
    "jointly operationalize the stated aim).\n\n"
    "Respond with STRICT JSON:\n"
    '{"aim": "...", "questions": [{"id": "RQ1", "question": "...", '
    '"page": 6, "verdict": "STRONG|ADEQUATE|WEAK", '
    '"criteria_violated": ["focused", "..."], "comment": "...", '
    '"reformulation": "..."}], '
    '"scope": {"gap_identified": true, "delimitations_stated": true, '
    '"aligned": true, "covers_aim": true, "comment": "..."}}'
)


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(
        description="LLM linter: quality of research scope and questions "
                    "(university guideline criteria).")
    ap.add_argument("pdf", help="Path to the thesis PDF.")
    ap.add_argument("--base-url", default=BASE_URL, help=BASE_URL_HELP)
    ap.add_argument("--api-key", default=None, help=API_KEY_HELP)
    ap.add_argument("--model", default=None,
                    help="Model id (default: full GPT-5 on the Aalto AI "
                         "API; gateway default otherwise).")
    ap.add_argument("--max-chars", type=int, default=400_000,
                    help="Truncate text beyond this (default 400000).")
    ap.add_argument("--out", help="Write report to this file.")
    ap.add_argument("--format", choices=["text", "markdown"],
                    default="text")
    ap.add_argument("--profile", choices=["thesis", "paper"], default="thesis",
                    help="'paper' judges the problem statement + contributions "
                         "when no explicit research questions are stated.")
    ap.add_argument("--no-standalone", action="store_true",
                    help="Skip the dictionary-grounded STANDALONE "
                         "self-containment pass on each extracted RQ.")
    ap.add_argument("--dict", default=DEFAULT_DICT,
                    help=f"Aalto Dictionary terms.txt for the standalone "
                         f"pass (default {DEFAULT_DICT}).")
    args = ap.parse_args(argv)

    lines, mode = load_lines([args.pdf])
    if mode != "pdf":
        print("ERROR: this linter takes a compiled PDF.", file=sys.stderr)
        return 2
    chunks, cur = [], None
    for where, t in lines:
        if where != cur:
            cur = where
            chunks.append(f"\n[[page {where[1:]}]]\n")
        chunks.append(t + "\n")
    text = "".join(chunks)[: args.max_chars]

    model = args.model or default_strong_model(args.base_url)
    client = make_client(args.base_url, args.api_key)
    print(f"[info] gateway={args.base_url}\n[info] model={model}  "
          f"chars={len(text)}", file=sys.stderr)

    paper_note = ""
    if args.profile == "paper":
        paper_note = (
            "NOTE: this is a conference/journal paper, not a thesis. If it "
            "states no explicit research questions, judge the PROBLEM "
            "STATEMENT and enumerated CONTRIBUTIONS by the same criteria "
            "(focused, specific, feasible, complex, relevant, self-"
            "contained). Do not emit NO-RQS when contributions play that "
            "role.\n\n")
    raw, usage = client.complete(
        model=model, system=SYSTEM_PROMPT,
        user=f'{paper_note}paper text:\n"""\n{text}\n"""',
        timeout=600, max_tokens=10000)
    parsed = extract_json(raw) or {}
    questions = [q for q in parsed.get("questions", [])
                 if isinstance(q, dict)]
    scope = parsed.get("scope", {}) if isinstance(parsed.get("scope"),
                                                  dict) else {}

    md = args.format == "markdown"
    out = []
    header = f"Research-scope quality report (LLM, {model})"
    out += [f"# {header}", "", f"File: `{args.pdf}`", ""] if md else \
           [f"== {header}", f"File: {args.pdf}", ""]
    aim = str(parsed.get("aim", "")).strip()
    if aim:
        out.append(f"Aim: {aim}")
        out.append("")

    n_warn = 0
    if not questions:
        n_warn += 1
        out.append("[WARN] NO-RQS            no explicitly stated research "
                   "questions found — state them in the introduction.")
    for q in questions:
        verdict = str(q.get("verdict", "WEAK")).upper()
        tag = {"STRONG": ("INFO", "STRONG-RQ"),
               "ADEQUATE": ("INFO", "ADEQUATE-RQ")}.get(
                   verdict, ("WARN", "WEAK-RQ"))
        if tag[0] == "WARN":
            n_warn += 1
        rid = str(q.get("id", "RQ?"))
        violated = ", ".join(str(c) for c in q.get("criteria_violated", []))
        comment = str(q.get("comment", "")).strip()
        reform = str(q.get("reformulation", "")).strip()
        if md:
            out += [f"## {rid} — {verdict}",
                    f"> {q.get('question', '')} *(p. {q.get('page', '?')})*",
                    ""]
            if violated:
                out.append(f"- **Criteria violated:** {violated}")
            out.append(f"- {comment}")
            if reform:
                out.append(f"- **Reformulation:** {reform}")
            out.append("")
        else:
            out.append(f"[{tag[0]}] {tag[1]:<12} {rid}: "
                       f"{str(q.get('question', ''))[:90]}… "
                       f"(p.{q.get('page', '?')})")
            if violated:
                out.append(f"        violates: {violated}")
            out.append(f"        {comment}")
            if reform:
                out.append(f"        reformulate: {reform}")
            out.append("")

    # ---- STANDALONE self-containment pass (dictionary-grounded) ----------
    # The main pass judges self-containment at the reader's position in the
    # paper. This pass asks the stricter question an abstract is held to:
    # can someone who has seen only the ELEMENTARY vocabulary read the RQ?
    sa_usage = {}
    if questions and not args.no_standalone:
        vocab = load_dictionary(args.dict)
        if not vocab:
            print(f"[warn] dictionary snapshot not found at {args.dict}; "
                  "standalone grounding reduced.", file=sys.stderr)
        qlist = "\n".join(
            f"{q.get('id', 'RQ?')} (p.{q.get('page', '?')}): "
            f"{str(q.get('question', '')).strip()}" for q in questions)
        sa_model = default_model(args.base_url) if args.model is None \
            else args.model
        sa_raw, sa_usage = client.complete(
            model=sa_model, system=STANDALONE_PROMPT,
            user=f"{_vocab_block(vocab)}\n\nRESEARCH QUESTIONS to assess, "
                 f"each in isolation:\n\"\"\"\n{qlist}\n\"\"\"",
            timeout=300, max_tokens=4000)
        sa = extract_json(sa_raw) or {}
        sa_q = {str(x.get("id", "")).strip(): x
                for x in sa.get("questions", []) if isinstance(x, dict)}
        if md:
            out += ["## Standalone self-containment (Aalto Dictionary, "
                    f"{len(vocab)} elementary terms)", ""]
        else:
            out.append("-- Standalone self-containment: each RQ read in "
                       "isolation, elementary = Aalto Dictionary of ML "
                       f"({len(vocab)} terms) --")
        for q in questions:
            rid = str(q.get("id", "RQ?"))
            r = sa_q.get(rid, {})
            grade = str(r.get("grade", "")).upper() or "UNCLEAR"
            gaps = [g for g in r.get("gaps", []) if isinstance(g, dict)]
            verdict = str(r.get("verdict", "")).strip()
            bad = grade in ("POOR", "FAIR")
            if bad:
                n_warn += 1
            lvl, code = ("WARN", "NOT-STANDALONE") if bad else \
                ("INFO", "STANDALONE")
            if md:
                out.append(f"### {rid} — {grade} ({len(gaps)} gap(s))")
                for g in gaps:
                    out.append(f"- **[{str(g.get('code', '')).upper()}]** "
                               f"`{g.get('term', '')}` — "
                               f"{str(g.get('question', '')).strip()} "
                               f"_fix:_ {str(g.get('fix', '')).strip()}")
                if verdict:
                    out.append(f"> {verdict}")
                out.append("")
            else:
                out.append(f"[{lvl}] {code:<14} {rid}: grade {grade} "
                           f"({len(gaps)} gap(s))")
                for g in gaps:
                    coin = g.get("coinage")
                    coin_s = "own coinage" if coin is True else (
                        "" if coin is None else "standard-ish")
                    tag = f"[{str(g.get('code', '')).upper()}]"
                    out.append(f"        {tag:<17} {g.get('term', '')}"
                               + (f"   ({coin_s})" if coin_s else ""))
                    qq = str(g.get("question", "")).strip()
                    fx = str(g.get("fix", "")).strip()
                    if qq:
                        out.append(f"              reader asks: {qq}")
                    if fx:
                        out.append(f"              fix:         {fx}")
                if verdict:
                    out.append(f"        {verdict}")
                out.append("")

    labels = [("gap_identified", "GAP-IDENTIFIED",
               "no explicit research-gap statement found"),
              ("delimitations_stated", "DELIMITATIONS",
               "what is out of scope is never stated"),
              ("aligned", "RQ-ALIGNMENT",
               "questions and stated objectives/contributions do not "
               "map onto each other"),
              ("covers_aim", "AIM-COVERAGE",
               "the questions do not jointly operationalize the aim")]
    for key, code, missing_msg in labels:
        ok = bool(scope.get(key, False))
        if ok:
            out.append(f"[INFO] {code:<16} satisfied")
        else:
            n_warn += 1
            out.append(f"[WARN] {code:<16} {missing_msg}")
    sc = str(scope.get("comment", "")).strip()
    if sc:
        out.append(f"        scope: {sc}")

    n_strong = sum(1 for q in questions
                   if str(q.get("verdict", "")).upper() == "STRONG")
    out += ["", f"{len(questions)} question(s): {n_strong} strong, "
                f"{len(questions) - n_strong} adequate/weak; "
                f"{n_warn} warning(s).  "
                f"(tokens: {usage.get('total_tokens', '?')}"
                + (f" + standalone {sa_usage.get('total_tokens', '?')}"
                   if sa_usage else "") + ")"]

    report = "\n".join(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(report)
    return 1 if n_warn else 0


if __name__ == "__main__":
    raise SystemExit(main())
