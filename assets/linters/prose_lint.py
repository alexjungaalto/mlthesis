#!/usr/bin/env python3
"""Linter: recurring prose defects from the ml-theses.org self-editing pass.

Guidelines enforced (heuristically):
  * Vague quantifiers — "significantly", "very", "a lot" without a number.
    -> [WARN] VAGUE-QUANTIFIER when a sentence contains such a word but no
       digit, percentage, or citation.
  * Jargon / undefined evaluative claims — "smoothest convergence", "the
    model struggles", "performs well": colloquial ML-speak that misuses
    defined terms (in the Aalto Dictionary "smooth" means differentiable)
    or claims a property with no defined metric. A nearby number does not
    excuse these.
    -> [WARN] JARGON per occurrence.
  * Empty buzzwords — words near-universally replaceable by a plainer
    one: "leverage"/"utilize" -> "use", "in order to" -> "to", plus
    unearned hype ("seamless", "cutting-edge", "holistic", "synergy").
    Legitimate senses ("financial leverage", measured "utilization")
    are carved out.
    -> [WARN] BUZZWORD per occurrence.
  * Dead metaphors — figurative stock phrases standing in for a literal
    statement: "blast radius", "silver bullet", "low-hanging fruit",
    "under the hood", "shed light on", "bridge the gap", "pave the way",
    "deep dive", "in the wild", "battle-tested", "north star", "paradigm
    shift", "the landscape of". Sharpened 2026-10-05.
    -> [WARN] METAPHOR per occurrence. Tightened 2026-10-06: the list
       now covers the long tail ("headroom", "scalability ceiling",
       "operating envelope", "migration path", "closing the gap",
       "substrate", "fabric", "anchor", "lens", "heavy lifting", ...).
  * Hype adjectives — "novel", "robust", "scalable", "lightweight",
    "efficient", "principled", "comprehensive", "meaningful", "smart",
    "sovereign", "canonical", ... with nothing in the sentence that could
    earn the word (no number, citation, or statistical claim).
    -> [WARN] HYPE per occurrence (tightened 2026-10-06).
  * Hyphenated coinages — "X-aware", "X-driven", "X-centric",
    "cloud-native", "fault-containing", "data-sovereign": insider
    vocabulary unless defined; fixed terms of art ("fault-tolerant",
    "privacy-preserving", "human-in-the-loop") are carved out.
    -> [WARN] COINAGE per occurrence (tightened 2026-10-06).
  * Two or more em-dashes in one sentence (paired interpolation / dash
    chain, the LLM-signature construction).
    -> [WARN] EMDASH per sentence.
  * Dangling references — "this shows", "it follows" where "this"/"it" has
    no clear antecedent.
    -> [INFO] DANGLING-REFERENCE for sentences STARTING with a bare
       "This/These/It/Those" + verb (no noun in between). INFO because many
       such sentences are fine; scan the list quickly by eye.
  * Excessive forward referencing — forward-looking cue phrases such as
    "as we will see in Chapter 5" scattered throughout.
    -> [WARN] FORWARD-CUE per occurrence; [WARN] FORWARD-CUE-HABIT summary
       when the count exceeds --max-forward-cues (default 3).

For structural forward references to figures/equations use
crossref_forward_lint.py; for conceptual ones use forward_ref_lint(_llm).py.
The LLM cousin of this linter is prose_lint_llm.py.

Input: thesis PDF or LaTeX sources.
Usage:
  python3 prose_lint.py thesis.pdf
  python3 prose_lint.py main.tex chapters/
Exit status: 0 clean, 1 findings (WARN or worse), 2 usage error.
"""

import argparse
import re
from typing import List

from lintutil import (Report, is_toc_line, load_lines, paragraphs_from_lines,
                      sentences)

VAGUE_WORDS = [
    "significantly", "significant", "very", "a lot", "lots of", "huge",
    "hugely", "extremely", "considerably", "considerable", "substantially",
    "substantial", "drastically", "dramatically", "vastly", "greatly",
    "enormous", "massive", "immense", "much better", "much worse",
    "much faster", "much slower", "quite", "fairly", "really", "clearly",
    "obviously", "many", "few",
]
# "many"/"few"/"significant" are noisy; only the adverb forms and the worst
# offenders are on by default. The full list is enabled with --strict.
DEFAULT_VAGUE = [
    "significantly", "very", "a lot", "lots of", "hugely", "extremely",
    "drastically", "dramatically", "vastly", "enormous", "massive",
    "much better", "much worse", "much faster", "much slower",
]

# A vague quantifier is excused when the same sentence carries any of these:
# a digit/percentage (HAS_NUMBER_RE), a citation ([n] or \cite, CITATION_RE),
# or an explicit statistical-significance claim (STAT_SIG_RE).
HAS_NUMBER_RE = re.compile(r"\d|%|\\%")
CITATION_RE = re.compile(r"\[\d+(?:,\s*\d+)*\]|\\cite[pt]?\*?\{")
STAT_SIG_RE = re.compile(r"statistically\s+significant|p\s*[<=>]", re.I)

DANGLING_RE = re.compile(
    r"^(This|These|It|Those)\s+"
    r"(is|are|was|were|shows?|showed|means?|meant|implies|implied|"
    r"follows?|followed|suggests?|suggested|demonstrates?|demonstrated|"
    r"indicates?|indicated|leads?|led|results?|resulted|allows?|allowed|"
    r"makes?|made|can|could|will|would|may|might|does|did|has|have|had)\b")

# Colloquial/evaluative ML-speak: either misuses a defined technical term
# in an informal sense ("smooth" means differentiable in the Aalto
# Dictionary) or claims a property with no defined metric. A nearby number
# does not cure these — flag regardless.
JARGON_RE = re.compile(
    r"\b(smooth(?:est|er)?\s+convergence|converge[sd]?\s+(?:very\s+)?"
    r"smoothly|(?:model|training|network|method)\s+struggl\w+|"
    r"degrades?\s+gracefully|graceful\s+degradation|"
    r"(?:works?|performs?)\s+(?:very\s+|quite\s+|really\s+)?well|"
    r"(?:good|great|strong|excellent|impressive|decent|competitive)\s+"
    r"(?:performance|results|accuracy)|"
    r"achieves?\s+(?:strong|impressive|competitive|superior)\s+"
    r"(?:performance|results|accuracy)|"
    r"(?:training|convergence)\s+is\s+(?:very\s+)?(?:stable|smooth)|"
    r"state[- ]of[- ]the[- ]art\s+(?:performance|results|accuracy)|"
    r"(?:fast(?:est)?|smooth(?:est)?|(?:most\s+)?stable)\s+"
    r"(?:and\s+(?:fast(?:est)?|smooth(?:est)?|(?:most\s+)?stable)\s+)?"
    r"convergence)\b", re.I)

# Empty buzzwords: near-universally replaceable by a plainer word, high
# enough precision for a regex pass. "leverage"/"utilize" -> "use";
# "in order to" -> "to"; hype adjectives ("novel", "seamless", ...) that
# read as unearned praise. Carve-outs below drop the legitimate senses.
BUZZWORD_RE = re.compile(
    r"\b(leverages?|leveraging|leveraged|utili[sz]es?|utili[sz]ing|"
    r"utili[sz]ed|in\s+order\s+to|seamless(?:ly)?|cutting[- ]edge|"
    r"holistic(?:ally)?|synerg\w+|"
    # sharpened 2026-10-05: LLM-signature and hype vocabulary
    r"delv(?:e|es|ing|ed)|tapestry|(?:a|is\s+a)\s+testament\s+to|"
    r"underscor(?:e|es|ing|ed)|myriad|plethora|realm|paradigm|"
    r"ecosystem|showcas(?:e|es|ing|ed)|unlock(?:s|ing|ed)?|"
    r"empower(?:s|ing|ed)?|harness(?:es|ing|ed)?|foster(?:s|ing|ed)?|"
    r"streamlin(?:e|es|ing|ed)|bolster(?:s|ing|ed)?|"
    r"facilitat(?:e|es|ing|ed)|actionable|game[- ]chang(?:er|ing)|"
    r"next[- ]generation|best[- ]in[- ]class|world[- ]class|"
    r"transformative|holy\s+grail|robustif\w+|"
    r"genuinely|materially|truly|arguably|it\s+is\s+worth\s+noting|"
    r"in\s+our\s+view|needless\s+to\s+say|"
    # tightened 2026-10-06: filler phrases and LLM-signature verbs
    r"it\s+should\s+be\s+noted|the\s+fact\s+that|in\s+the\s+context\s+of|"
    r"plays?\s+an?\s+(?:\w+\s+)?role|a\s+wide\s+range\s+of|"
    r"operationali[sz](?:e|es|ed|ing)|navigat(?:e|es|ing|ed)\s+the|"
    r"landscape|roadmap|synergi\w*|cutting[- ]edge|"
    r"state[- ]of[- ]the[- ]art|end[- ]to[- ]end|real[- ]world|"
    r"production[- ](?:grade|ready|adjacent|like)|"
    r"at\s+scale|first[- ]class(?:\s+citizen)?)\b", re.I)

# Hype adjectives (tightened 2026-10-06): praise words that assert a
# property. Flagged as BUZZWORD "hype" UNLESS the same sentence carries a
# number, a citation, or a statistical claim that could earn the word
# (HAS_NUMBER_RE / CITATION_RE / STAT_SIG_RE, the vague-quantifier gates).
HYPE_RE = re.compile(
    r"\b(novel|robust(?:ly|ness)?|seamless|holistic|powerful|efficient(?:ly)?|"
    r"lightweight|scalable|scalability|principled|comprehensive|rigorous|"
    r"systematic|trustworthy|resilien(?:t|ce)|smart|intelligent|advanced|"
    r"sophisticated|enhanced|optimal(?:ly)?|optimi[sz]ed|effective(?:ly)?|"
    r"flexible|flexibility|modular|unified|integrated|dynamic(?:ally)?|"
    r"adaptive|innovative|promising|emerging|crucial(?:ly)?|vital|"
    r"essential|meaningful(?:ly)?|non[- ]?trivial|informative|practical|"
    r"mature|elegant|sovereign(?:ty)?|commodity|compact|inherent(?:ly)?|"
    r"canonical|natural\s+fit|acceptable|appropriate|suitable|"
    r"fine[- ]grained|high[- ]quality|high[- ]performance|rich(?:er|est)?)\b",
    re.I)

# Hyphenated coinages (tightened 2026-10-06): "X-aware", "X-driven",
# "X-centric", "cloud-native", "fault-containing" ... read as insider
# vocabulary unless defined. Flagged as COINAGE per occurrence.
COINAGE_RE = re.compile(
    r"\b([A-Za-z]+-(?:aware|driven|centric|native|first|grade|ready|"
    r"adjacent|powered|enabled|enhanced|oriented|agnostic|friendly|"
    r"sovereign|containing|preserving|tolerant|resilient|hardened|"
    r"as-code|in-the-loop|at-rest|in-transit|of-the-art))\b", re.I)
# Coinages that ARE fixed terms of art in ML/systems — not flagged.
COINAGE_OK_RE = re.compile(
    r"\b(fault-tolerant|privacy-preserving|human-in-the-loop|"
    r"infrastructure-as-code|state-of-the-art|model-agnostic|"
    r"shift-invariant|scale-invariant|encryption-at-rest|"
    r"encryption-in-transit|context-aware)\b", re.I)

# Dead metaphors: figurative stock phrases with a literal equivalent. High
# precision list; the LLM pass (prose_lint_llm.py dead-metaphor) covers the
# long tail. 'bottleneck', 'backbone', 'pipeline', 'footprint' are terms of
# art and deliberately absent.
METAPHOR_RE = re.compile(
    r"\b(blast[- ]radius|silver\s+bullet|low[- ]hanging\s+fruit|"
    r"under\s+the\s+hood|at\s+the\s+end\s+of\s+the\s+day|"
    r"sheds?\s+(?:new\s+)?light\s+on|shedding\s+light\s+on|"
    r"bridg(?:e|es|ing)\s+the\s+gap|pav(?:e|es|ing|ed)\s+the\s+way|"
    r"deep[- ]dive|in\s+the\s+wild|battle[- ]tested|north\s+star|"
    r"paradigm\s+shift|the\s+landscape\s+of|sweet\s+spot|"
    r"double[- ]edged\s+sword|achilles(?:'|’)?\s+heel|"
    r"tip\s+of\s+the\s+iceberg|elephant\s+in\s+the\s+room|"
    r"paint(?:s|ed|ing)?\s+a\s+picture|moving\s+the\s+needle|"
    r"secret\s+sauce|silicon\s+boundary|the\s+last\s+mile|"
    r"hook\s+point|through\s+the\s+lens\s+of|"
    r"(?:security|compliance|resilience|R\d)\s+posture|"
    r"journey|guardrails?|stepping[- ]stone|cornerstone|"
    r"a\s+recipe\s+for|the\s+bar\s+(?:is|for)|raise(?:s|d)?\s+the\s+bar|"
    # tightened 2026-10-06: the long tail seen in systems/FL drafts
    r"clos(?:e|es|ing|ed)\s+the\s+gap|headroom|(?:scalability|scaling|"
    r"capacity|performance)\s+ceiling|operating\s+envelope|"
    r"(?:migration|scaling|upgrade)\s+path|path\s+forward|"
    r"escap(?:e|es|ed|ing)\s+(?:the|its|our|all)\s+\w+\s*(?:checks?|"
    r"monitoring|alerts?|detection|metrics)|heavy\s+lifting|"
    r"moving\s+parts|pain\s+points?|friction|ripple\s+effects?|"
    r"snowball\w*|spin(?:s|ning)?\s+up|tear(?:s|ing)?\s+down|"
    r"out\s+of\s+the\s+box|plug[- ]and[- ]play|the\s+big\s+picture|"
    r"takeaways?|boils?\s+down\s+to|in\s+a\s+nutshell|sprawl|"
    r"scaffold(?:ing)?|footguns?|substrate|fabric|umbrella|pillars?|"
    r"anchor(?:s|ed|ing)?|glue(?:s|d)?|knobs?|levers?|toolbox|"
    r"swiss[- ]army[- ]knife|rabbit\s+hole|(?:a|the)\s+lens|"
    r"weav(?:e|es|ing)|unlock(?:s|ed|ing)?|surfac(?:es|ed|ing)\s+(?:the|"
    r"a|an|this|these)|lean(?:s|ed|ing)?\s+on|shrugs?\s+off|bleeds?|"
    r"breath(?:es|ing)\s+room|wiggle\s+room|sanity\s+check|"
    r"a\s+(?:first|second)\s+line\s+of\s+defen[cs]e|attack\s+surface|"
    r"single\s+pane\s+of\s+glass|source\s+of\s+truth|"
    r"the\s+(?:gold|golden)\s+standard|a\s+(?:step|leap)\s+forward|"
    r"stands?\s+on\s+the\s+shoulders)\b",
    re.I)
# "leverage" as the physics/finance noun and "utilization" (a measured
# quantity, e.g. GPU utilization) are legitimate — do not flag those.
BUZZWORD_OK_RE = re.compile(
    r"\b(financial\s+leverage|utili[sz]ation)\b", re.I)
# A hype word inside a cited paper title is the cited authors' wording, not
# the author's prose. Skip sentences that look like a bibliographic entry
# (an "et al." or an author-initial list run together with a 4-digit year).
CITATIONISH_RE = re.compile(
    r"et\s+al\.|[A-Z][a-z]+,\s+[A-Z]\.[^.]*\b(19|20)\d\d\b|"
    r"[\u201c\u201d]|\b(?:vol|no|pp)\.\s*\d|"
    r"\b(?:Proc\.|Proceedings\s+of|arXiv|Trans\.\s+on|Conf\.\s+on|"
    r"Journal\s+of|Symposium\s+on|Workshop\s+on)\b|"
    r"\[\d+\]\s+[A-Z]\.(?:\s|$)")

# Forward-looking cue phrases: "as we will see", "we will discuss ... later",
# "will be defined in ...", "more on this later", "later in this chapter".
# Two or more em-dashes in one sentence: the paired interpolation ("— though
# loose —") and the dash chain are the signature of machine-generated prose.
# A single em-dash stays a judgment call. Ported 2026-09-17 from the
# dictionary suite's check_house_rules EMDASH screen (d90264a); a PDF
# extract carries U+2014, LaTeX sources "---".
EMDASH_RE = re.compile(r"\u2014|(?<!-)---(?!-)")

FORWARD_CUE_RE = re.compile(
    r"\b(as\s+(?:we\s+(?:will|shall)\s+see|will\s+be\s+(?:seen|shown|"
    r"discussed|explained|described))|"
    r"we\s+(?:will|shall)\s+(?:see|show|discuss|explain|describe|return\s+to|"
    r"come\s+back\s+to)|"
    r"(?:will|shall)\s+be\s+(?:discussed|described|explained|introduced|"
    r"presented|defined)\s+(?:in|later)|"
    r"more\s+on\s+this\s+(?:later|in)|"
    r"later\s+(?:in\s+this\s+(?:thesis|chapter|section)|chapters?|sections?))\b",
    re.I)


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Prose linter: vague quantifiers, dangling references, "
                    "forward-looking cue phrases.")
    ap.add_argument("inputs", nargs="+", help="thesis.pdf or .tex files/dirs")
    ap.add_argument("--strict", action="store_true",
                    help="Enable the full vague-word list (noisier: adds "
                         "'many', 'few', 'clearly', 'quite', ...).")
    ap.add_argument("--max-forward-cues", type=int, default=3,
                    help="Threshold for the FORWARD-CUE-HABIT summary "
                         "finding (default 3).")
    ap.add_argument("--no-dangling", action="store_true",
                    help="Skip the DANGLING-REFERENCE check.")
    args = ap.parse_args(argv)

    lines, mode = load_lines(args.inputs)
    lines = [(w, t) for (w, t) in lines if not is_toc_line(t)]
    rep = Report("Prose lint report (self-editing pass)",
                 " ".join(args.inputs),
                 about="Flags self-editing prose issues: vague quantifiers "
                       "without a number, colloquial/undefined claims, "
                       "forward-looking cue phrases, and dangling references.")

    vague = args.strict and VAGUE_WORDS or DEFAULT_VAGUE
    vague_re = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in vague) + r")\b", re.I)

    n_cues = 0
    in_references = False
    for where, para in paragraphs_from_lines(lines):
        # Reference-list start: the bare heading, or the heading run into
        # the first numbered entry — but NOT a table-of-contents entry
        # like "References 96".
        if re.match(r"^(References|Bibliography)\s*(\[1\]|$)",
                    para.strip(), re.I):
            in_references = True
        if in_references:
            continue
        for sent in sentences(para):
            m = vague_re.search(sent)
            if m and not (HAS_NUMBER_RE.search(sent)
                          or CITATION_RE.search(sent)
                          or STAT_SIG_RE.search(sent)):
                rep.add("WARN", "VAGUE-QUANTIFIER", where,
                        f"'{m.group(1)}' without a number: "
                        f"\"{sent[:110]}{'…' if len(sent) > 110 else ''}\"")
            m = JARGON_RE.search(sent)
            if m:
                rep.add("WARN", "JARGON", where,
                        f"colloquial/undefined claim '{m.group(1)}' — "
                        f"define and measure the property or drop it: "
                        f"\"{sent[:95]}{'…' if len(sent) > 95 else ''}\"")
            m = BUZZWORD_RE.search(sent)
            if m and not BUZZWORD_OK_RE.search(sent) \
                    and not CITATIONISH_RE.search(sent):
                word = m.group(1)
                plain = {"in order to": "to"}.get(word.lower(), "'use' / plain wording")
                if re.match(r"(?i)leverage|utili|employ|harness", word):
                    plain = "use"
                elif re.match(r"(?i)genuinely|materially|truly|arguably|"
                              r"it\s+is\s+worth|in\s+our\s+view|needless",
                              word):
                    plain = "(delete) — rhetorical intensifier, adds no information"
                elif re.match(r"(?i)delv|showcas|underscor", word):
                    plain = "'examine' / 'show'"
                elif re.match(r"(?i)myriad|plethora", word):
                    plain = "'many' or the number"
                elif re.match(r"(?i)paradigm|ecosystem|realm|tapestry", word):
                    plain = "the concrete noun it stands for"
                rep.add("WARN", "BUZZWORD", where,
                        f"empty buzzword '{word}' — prefer {plain}: "
                        f"\"{sent[:95]}{'…' if len(sent) > 95 else ''}\"")

            if not CITATIONISH_RE.search(sent):
                # Hype adjectives: flagged unless the sentence carries
                # something that could earn the word (number / citation /
                # statistical claim) — the same excuses as vague words.
                earned = (HAS_NUMBER_RE.search(sent)
                          or CITATION_RE.search(sent)
                          or STAT_SIG_RE.search(sent))
                for hm in HYPE_RE.finditer(sent):
                    if earned:
                        break
                    rep.add("WARN", "HYPE", where,
                            f"hype word '{hm.group(1)}' with nothing in the "
                            f"sentence to earn it — delete it, or replace "
                            f"it with the measured property: "
                            f"\"{sent[:95]}{'…' if len(sent) > 95 else ''}\"")
                # Hyphenated coinages: insider vocabulary unless a fixed
                # term of art (COINAGE_OK_RE).
                for cm in COINAGE_RE.finditer(sent):
                    if COINAGE_OK_RE.search(cm.group(1)):
                        continue
                    rep.add("WARN", "COINAGE", where,
                            f"hyphenated coinage '{cm.group(1)}' — define "
                            f"it in plain words on first use or replace "
                            f"it: \"{sent[:95]}{'…' if len(sent) > 95 else ''}\"")
            for m in METAPHOR_RE.finditer(sent):
                if CITATIONISH_RE.search(sent):
                    break
                rep.add("WARN", "METAPHOR", where,
                        f"metaphor '{m.group(0)}' — state the literal "
                        f"fact instead: \"{sent[:110]}\"")
            if len(EMDASH_RE.findall(sent)) >= 2:
                rep.add("WARN", "EMDASH", where,
                        f"two or more em-dashes in one sentence (the "
                        f"LLM-signature interpolation) — rewrite as "
                        f"separate sentences or commas: "
                        f"\"{sent[:95]}{'…' if len(sent) > 95 else ''}\"")
            m = FORWARD_CUE_RE.search(sent)
            if m:
                n_cues += 1
                rep.add("WARN", "FORWARD-CUE", where,
                        f"forward-looking phrase '{m.group(0)}': "
                        f"\"{sent[:100]}{'…' if len(sent) > 100 else ''}\"")
            if not args.no_dangling:
                m = DANGLING_RE.match(sent)
                if m:
                    rep.add("INFO", "DANGLING-REFERENCE", where,
                            f"sentence opens with bare '{m.group(1)} "
                            f"{m.group(2)}' — check the antecedent: "
                            f"\"{sent[:90]}{'…' if len(sent) > 90 else ''}\"")

    if n_cues > args.max_forward_cues:
        rep.add("WARN", "FORWARD-CUE-HABIT", "-",
                f"{n_cues} forward-looking cue phrases in total — 'a few "
                f"pointers are fine, but a habit of them signals "
                f"disorganised structure' (threshold "
                f"{args.max_forward_cues}).")

    print(rep.render())
    return rep.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
