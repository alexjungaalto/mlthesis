#!/usr/bin/env python3
"""LLM linter: clarity of presentation of the learning problem.

A machine-learning manuscript should let the reader reconstruct, without
guessing, the objects that define its learning problem (Jung, "Machine
Learning: The Basics", Springer). WHICH objects those are depends on the
learning technique, so the linter first identifies the paradigm and then
grades the paradigm's defining objects:

  supervised (ERM)       data point, features, label
  unsupervised           data point, features, objective (structure sought
                         / loss minimised)
  reinforcement          state (observation), action, reward, objective
                         (return / policy goal)
  generative /           data point, features (representation /
  self-supervised        conditioning), training objective
  other / hybrid         the 3-5 objects the manuscript's own setting
                         requires (named by the linter)

Each object is graded for clarity of presentation:

  CLEAR    stated explicitly and unambiguously, with enough detail that a
           reader could reconstruct the object (shape/space/units located).
  PARTIAL  named but under-specified — a reader must infer shape, units,
           channel count, or which quantities play which role.
  UNCLEAR  mentioned only implicitly or inconsistently; genuinely ambiguous.
  MISSING  not defined anywhere.

It also flags the two classic confusions, phrased for the paradigm at hand:
(a) interface boundary — is it unambiguous what the learner consumes vs.
what it produces or controls (features vs. label; observation vs. action)?
(b) unit identity — is the granularity of ONE training unit stated (one
data point, one transition, one episode, one trajectory)?

Whole-document, single LLM call. Findings are heuristic LLM judgements.

Gateway: the Aalto AI API by default (see aalto_llm.py; $AALTO_API_KEY,
Aalto network/VPN only); --base-url switches gateways.

Usage:
  python3 problem_clarity_lint_llm.py submission.pdf
  python3 problem_clarity_lint_llm.py submission.pdf --out lintout/foo_problem.txt
Exit status: 0 all defining objects CLEAR, 1 any gap, 2 usage error.
"""

import argparse
import sys
from typing import List, Optional

from aalto_llm import (API_KEY_HELP, BASE_URL, BASE_URL_HELP, default_model,
                       extract_json, make_client)
from lintutil import load_lines

LEVELS = ["CLEAR", "PARTIAL", "UNCLEAR", "MISSING"]

# Canonical defining objects per paradigm. For a paradigm not listed here
# ("other"), the model names the objects itself and they are kept verbatim.
PARADIGM_ELEMENTS = {
    "supervised":    ["data_point", "features", "label"],
    "unsupervised":  ["data_point", "features", "objective"],
    "reinforcement": ["state", "action", "reward", "objective"],
    "generative":    ["data_point", "features", "objective"],
}

SYSTEM_PROMPT = (
    "You are a meticulous machine-learning reviewer assessing CLARITY OF "
    "PRESENTATION. You judge whether the manuscript lets a reader "
    "reconstruct, without guessing, the objects that define its learning "
    "problem (Jung, 'Machine Learning: The Basics').\n\n"
    "STEP 1 — identify the learning paradigm actually used, one of:\n"
    "  supervised | unsupervised | reinforcement | generative | other\n"
    "(generative covers self-supervised/foundation-model pretraining; use "
    "'other' for hybrids or settings none of the labels fit, and say what "
    "it is). Quote the evidence for your identification.\n\n"
    "STEP 2 — grade the paradigm's defining objects:\n"
    "  supervised:    data_point (what is ONE example the model reasons "
    "about — one grid cell? one frame? one trajectory?), features (the "
    "quantities FED INTO the model for one data point: shape, channels, "
    "which fields), label (the quantity the model is trained to predict: "
    "its space, range, units).\n"
    "  unsupervised:  data_point, features, objective (what structure is "
    "sought / what loss is minimised).\n"
    "  reinforcement: state (what the agent observes, its space), action "
    "(what the agent controls, its space), reward (the scalar signal: "
    "definition, range, source), objective (return/discounting/horizon — "
    "what is being maximised).\n"
    "  generative:    data_point, features (representation/conditioning), "
    "objective (the training objective).\n"
    "  other:         the 3-5 objects a reader must reconstruct to "
    "understand THIS problem; name each yourself.\n\n"
    "For EACH object, assign a clarity level:\n"
    "  CLEAR   explicit and unambiguous, enough to reconstruct the object; "
    "shape/space/units locatable in the text.\n"
    "  PARTIAL named but under-specified — the reader must infer shape, "
    "units, ranges, or which quantity plays which role.\n"
    "  UNCLEAR only implicit or internally inconsistent; genuinely "
    "ambiguous.\n"
    "  MISSING not defined anywhere.\n\n"
    "Be strict: listing variables in a table is NOT the same as stating the "
    "input tensor shape; naming a target product is NOT the same as stating "
    "its space/range; calling something 'the reward' is NOT the same as "
    "defining how it is computed. Reward explicit reconstructable "
    "definitions only.\n\n"
    "STEP 3 — judge two boundary questions, phrased for the paradigm:\n"
    "  interface_boundary: is it unambiguous what the learner CONSUMES vs. "
    "what it PRODUCES or CONTROLS (features vs. label, incl. prescribed "
    "forcings; observation vs. action)? level CLEAR/PARTIAL/UNCLEAR.\n"
    "  unit_identity: is the granularity of ONE training unit stated "
    "explicitly (one data point, one transition, one episode, one "
    "trajectory), not left for the reader to infer? level "
    "CLEAR/PARTIAL/UNCLEAR.\n\n"
    "For every judgement, quote the SHORTEST located evidence with a page "
    "marker if present (the text carries [[page N]] markers), and give a "
    "one-line gap describing exactly what a reader cannot reconstruct.\n\n"
    "Respond with STRICT JSON:\n"
    '{"paradigm": "supervised|unsupervised|reinforcement|generative|other", '
    '"paradigm_evidence": "short quote (p.N)", '
    '"elements": [{"name": "<object name>", '
    '"level": "CLEAR|PARTIAL|UNCLEAR|MISSING", "evidence": "short quote '
    '(p.N)", "gap": "what cannot be reconstructed"}], '
    '"boundaries": [{"name": "interface_boundary|unit_identity", '
    '"level": "CLEAR|PARTIAL|UNCLEAR", "evidence": "...", "gap": "..."}], '
    '"overall": "one-sentence clarity-of-presentation verdict"}'
)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="LLM linter: clarity of presentation of the learning "
                    "problem (paradigm-aware: ERM triple, RL tuple, ...).")
    ap.add_argument("pdf", help="Path to the submission PDF (or .txt extract).")
    ap.add_argument("--base-url", default=BASE_URL, help=BASE_URL_HELP)
    ap.add_argument("--api-key", default=None, help=API_KEY_HELP)
    ap.add_argument("--model", default=None,
                    help="Model id (default depends on the gateway).")
    ap.add_argument("--max-chars", type=int, default=400_000,
                    help="Truncate extracted text beyond this many chars.")
    ap.add_argument("--out", help="Write report to this file.")
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

    # Subtle judgement: prefer the full GPT-5 model on the Aalto AI API.
    model = args.model or default_model(args.base_url)
    client = make_client(args.base_url, args.api_key)
    print(f"[info] gateway={args.base_url}\n[info] model={model}  "
          f"chars={len(text)}", file=sys.stderr)

    user = (f"paper text{' (TRUNCATED)' if truncated else ''}:\n"
            f'"""\n{text}\n"""')
    raw, usage = client.complete(model=model, system=SYSTEM_PROMPT,
                                 user=user, timeout=600, max_tokens=3000)
    parsed = extract_json(raw) or {}

    def norm(rec, allowed):
        name = str(rec.get("name", "")).strip()
        level = str(rec.get("level", "MISSING")).strip().upper()
        if level not in allowed:
            level = "UNCLEAR" if "UNCLEAR" in allowed else "MISSING"
        return {"name": name,
                "level": level,
                "evidence": str(rec.get("evidence", "")).strip() or "—",
                "gap": str(rec.get("gap", "")).strip() or "—"}

    paradigm = str(parsed.get("paradigm", "other")).strip().lower()
    paradigm_evidence = str(parsed.get("paradigm_evidence", "")).strip()

    elems = [norm(r, LEVELS) for r in parsed.get("elements", [])
             if isinstance(r, dict)]
    canonical = PARADIGM_ELEMENTS.get(paradigm)
    if canonical:
        # keep the canonical objects first and in order, even if the model
        # reordered them; anything extra the model graded follows
        by_name = {e["name"]: e for e in elems}
        ordered = [by_name.pop(n, {"name": n, "level": "MISSING",
                                   "evidence": "—", "gap": "not addressed"})
                   for n in canonical]
        elems = ordered + [e for e in elems if e["name"] in by_name]
    elif not elems:
        elems = [{"name": "learning problem", "level": "MISSING",
                  "evidence": "—", "gap": "no defining objects identified"}]
    bounds = [norm(r, ["CLEAR", "PARTIAL", "UNCLEAR"])
              for r in parsed.get("boundaries", []) if isinstance(r, dict)]
    overall = str(parsed.get("overall", "")).strip()

    n_clear = sum(1 for e in elems if e["level"] == "CLEAR")
    detail = ", ".join(f"{e['name']}={e['level']}" for e in elems)
    summary = (f"Learning problem ({paradigm}): {n_clear}/{len(elems)} CLEAR "
               f"({detail})")

    role = {"data_point": "one example the model reasons about",
            "features":   "what the model consumes",
            "label":      "what the model predicts",
            "objective":  "what is optimised",
            "state":      "what the agent observes",
            "action":     "what the agent controls",
            "reward":     "the training signal"}
    blabel = {"interface_boundary": "INTERFACE BOUNDARY (input vs. output)",
              "unit_identity":      "UNIT IDENTITY (one training unit)"}
    width = max(len(e["name"]) for e in elems)

    out = [f"== Learning-problem clarity lint (LLM, {model})",
           f"File: {args.pdf}", "",
           f"SUMMARY: {summary}", "",
           f"Paradigm: {paradigm}"
           + (f" — evidence: {paradigm_evidence}" if paradigm_evidence else ""),
           "",
           "How to read: for each object defining the learning problem — "
           "the located evidence and what a reader cannot reconstruct.",
           "CLEAR reconstructable · PARTIAL under-specified · UNCLEAR "
           "ambiguous · MISSING absent.", ""]

    for e in elems:
        tag = role.get(e["name"])
        head = e["name"].upper().ljust(width) + (f"  ({tag})" if tag else "")
        out.append(f"[{e['level']:<7}] {head}")
        out.append(f"    evidence: {e['evidence']}")
        out.append(f"    gap:      {e['gap']}")
        out.append("")

    if bounds:
        out.append("-- Boundary checks --")
        for b in bounds:
            out.append(f"[{b['level']:<7}] {blabel.get(b['name'], b['name'])}")
            out.append(f"    evidence: {b['evidence']}")
            out.append(f"    gap:      {b['gap']}")
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

    return 0 if n_clear == len(elems) else 1


if __name__ == "__main__":
    raise SystemExit(main())
