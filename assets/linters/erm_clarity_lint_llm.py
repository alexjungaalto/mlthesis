#!/usr/bin/env python3
"""LLM linter: clarity of presentation of the ERM triple.

A machine-learning paper should let the reader reconstruct, without
guessing, the three objects that define its learning problem (Jung, "Machine
Learning: The Basics", Springer):

  data point   the unit the model reasons about — one training/test example.
               What is ONE data point here? (e.g. "one initialization time:
               a pair of consecutive cloud-cover frames on the 5 km grid").
  features     the quantities FED INTO the model for one data point — the
               input representation (dimension, channels, what each is).
  label        the quantity the model OUTPUTS / is trained to predict for one
               data point — the target (its space, range, units).

This linter reads the whole paper and grades each of the three for clarity
of presentation:

  CLEAR    stated explicitly and unambiguously, with enough detail that a
           reader could reconstruct the object (shape/space/units located).
  PARTIAL  named but under-specified — a reader must infer shape, units,
           channel count, or which fields are input vs. output.
  UNCLEAR  mentioned only implicitly or inconsistently; genuinely ambiguous.
  MISSING  not defined anywhere.

It also flags the two classic confusions: (a) input/output boundary — is it
unambiguous which fields are features and which are the label (forcings that
are prescribed inputs vs. the predicted field)? and (b) data-point identity —
is one data point a single grid cell, one frame, one initialization, or one
forecast trajectory?

Whole-document, single LLM call. Findings are heuristic LLM judgements.

Gateway: the Aalto AI API by default (see aalto_llm.py; $AALTO_API_KEY,
Aalto network/VPN only); --base-url switches gateways.

Usage:
  python3 erm_clarity_lint_llm.py submission.pdf
  python3 erm_clarity_lint_llm.py submission.pdf --out lintout/foo_erm.txt
Exit status: 0 all three CLEAR, 1 any gap, 2 usage error.
"""

import argparse
import sys
from typing import List, Optional

from aalto_llm import (API_KEY_HELP, BASE_URL, BASE_URL_HELP, default_model,
                       extract_json, make_client)
from lintutil import load_lines

LEVELS = ["CLEAR", "PARTIAL", "UNCLEAR", "MISSING"]
ELEMENTS = ["data_point", "features", "label"]

SYSTEM_PROMPT = (
    "You are a meticulous machine-learning reviewer assessing CLARITY OF "
    "PRESENTATION. You judge whether the paper lets a reader reconstruct, "
    "without guessing, the three objects that define its learning problem, "
    "in the sense of empirical risk minimization (Jung, 'Machine Learning: "
    "The Basics'):\n"
    "  * data_point: what constitutes ONE example the model reasons about "
    "(one grid cell? one frame? one initialization time with a pair of "
    "input frames? one forecast trajectory?). The identity must be "
    "unambiguous.\n"
    "  * features: the quantities FED INTO the model for one data point — "
    "the input representation. A reader should be able to state its shape / "
    "channels / which physical fields, and whether each is input.\n"
    "  * label: the quantity the model OUTPUTS / is trained to predict for "
    "one data point — the target, its space, range, and units.\n\n"
    "For EACH of the three, assign a clarity level:\n"
    "  CLEAR   explicit and unambiguous, enough to reconstruct the object; "
    "shape/space/units locatable in the text.\n"
    "  PARTIAL named but under-specified — the reader must infer shape, "
    "units, channel count, or the input/output split.\n"
    "  UNCLEAR only implicit or internally inconsistent; genuinely "
    "ambiguous.\n"
    "  MISSING not defined anywhere.\n\n"
    "Be strict: listing variables in a table is NOT the same as stating the "
    "input tensor shape; naming a target product is NOT the same as stating "
    "its space/range. Reward explicit reconstructable definitions only.\n\n"
    "Also judge two boundary questions:\n"
    "  input_output_boundary: is it unambiguous which fields are FEATURES "
    "(inputs, incl. prescribed forcings) and which are the LABEL (the "
    "predicted field)? level CLEAR/PARTIAL/UNCLEAR.\n"
    "  data_point_identity: is the granularity of one data point stated "
    "explicitly (not left for the reader to infer)? level "
    "CLEAR/PARTIAL/UNCLEAR.\n\n"
    "For every judgement, quote the SHORTEST located evidence with a page "
    "marker if present (the text carries [[page N]] markers), and give a "
    "one-line gap describing exactly what a reader cannot reconstruct.\n\n"
    "Respond with STRICT JSON:\n"
    '{"elements": [{"name": "data_point|features|label", '
    '"level": "CLEAR|PARTIAL|UNCLEAR|MISSING", "evidence": "short quote '
    '(p.N)", "gap": "what cannot be reconstructed"}], '
    '"boundaries": [{"name": "input_output_boundary|data_point_identity", '
    '"level": "CLEAR|PARTIAL|UNCLEAR", "evidence": "...", "gap": "..."}], '
    '"overall": "one-sentence clarity-of-presentation verdict"}'
)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="LLM linter: clarity of presentation of the ERM triple "
                    "(data point, features, label).")
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

    elems = [norm(r, LEVELS) for r in parsed.get("elements", [])
             if isinstance(r, dict)]
    # keep the canonical three in order, even if the model reordered them
    by_name = {e["name"]: e for e in elems}
    elems = [by_name.get(n, {"name": n, "level": "MISSING",
                             "evidence": "—", "gap": "not addressed"})
             for n in ELEMENTS]
    bounds = [norm(r, ["CLEAR", "PARTIAL", "UNCLEAR"])
              for r in parsed.get("boundaries", []) if isinstance(r, dict)]
    overall = str(parsed.get("overall", "")).strip()

    n_clear = sum(1 for e in elems if e["level"] == "CLEAR")
    summary = (f"ERM triple: {n_clear}/3 CLEAR "
               f"(data_point={elems[0]['level']}, "
               f"features={elems[1]['level']}, label={elems[2]['level']})")

    label = {"data_point": "DATA POINT (one example)",
             "features":   "FEATURES  (model input)",
             "label":      "LABEL     (model output)"}
    blabel = {"input_output_boundary": "INPUT/OUTPUT BOUNDARY",
              "data_point_identity":   "DATA-POINT IDENTITY  "}

    out = [f"== ERM clarity-of-presentation lint (LLM, {model})",
           f"File: {args.pdf}", "",
           f"SUMMARY: {summary}", "",
           "How to read: for each object defining the learning problem — "
           "the located evidence and what a reader cannot reconstruct.",
           "CLEAR reconstructable · PARTIAL under-specified · UNCLEAR "
           "ambiguous · MISSING absent.", ""]

    for e in elems:
        out.append(f"[{e['level']:<7}] {label.get(e['name'], e['name'])}")
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

    return 0 if n_clear == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
