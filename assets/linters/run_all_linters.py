#!/usr/bin/env python3
"""Run the whole ml-theses.org linter suite on a thesis.

Runs every applicable linter (PDF or LaTeX input), prints each report, and
ends with a one-line-per-linter summary. Fast heuristic linters run always;
the LLM-based ones (Aalto AI API; need $AALTO_API_KEY and the Aalto
network/VPN) and the network-based bibliography check are opt-in.

Annotated-PDF augmentation (--annotations): point this at a manuscript PDF
carrying the reviewer's own margin highlights and comments (or at a
pre-extracted annotations JSON). The margin comments are then (a) folded
into every semantic LLM linter's context so it weighs what the human
flagged, and (b) cross-checked at the end by annotation_coverage_lint_llm.py,
which reports whether the automated suite caught each hand-written note.

Usage:
  python3 run_all_linters.py thesis.pdf
  python3 run_all_linters.py thesis.pdf --llm            # + LLM linters
  python3 run_all_linters.py thesis.pdf --bib            # + bibliography
  python3 run_all_linters.py main.tex chapters/          # LaTeX sources
  python3 run_all_linters.py paper.pdf --profile paper --llm   # conf/journal paper
  python3 run_all_linters.py thesis.pdf --dashboard      # + HTML report, opened
  python3 run_all_linters.py thesis.pdf --llm \
      --annotations thesis_annotated.pdf   # fold in reviewer margin comments
  python3 run_all_linters.py thesis.pdf --llm \
      --base-url http://localhost:8080/v1 --model my-model   # local LLM endpoint
Exit status: 0 if every linter passed, 1 otherwise.
"""

from __future__ import annotations  # PEP 604 (X | None) hints on Python 3.9

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def _resolve(script: str) -> Path | None:
    """Locate a helper/linter script. Looks in this directory first (the flat
    published layout), then one level up (the private suite keeps
    extract_annotations.py at its root). Returns None if not found."""
    for cand in (HERE / script, HERE.parent / script):
        if cand.exists():
            return cand
    return None


def _dashboard_out_path(inputs, explicit):
    """Where to write the HTML dashboard: --dashboard-out if given, else
    ``<first-input-stem>_lint_dashboard.html`` in the current directory."""
    if explicit:
        return Path(explicit)
    stem = Path(inputs[0]).stem or "thesis"
    return Path(f"{stem}_lint_dashboard.html")

THESIS, PAPER = "thesis", "paper"
BOTH = (THESIS, PAPER)

# (script, pdf-mode, tex-mode, profiles, passes_profile)
#   profiles       -- manuscript types this linter runs for
#   passes_profile -- forward --profile=<p> to the linter (it adapts itself)
# A registered script whose file is absent is skipped silently, so the same
# registry drives both the full private suite and the published subset (which
# ships without the reviewer-only linters).
FAST = [
    ("structure_lint.py", True, True, BOTH, True),
    ("unreferenced_entity_linter.py", True, True, BOTH, False),
    ("crossref_forward_lint.py", True, False, BOTH, False),
    ("forward_ref_lint.py", True, False, BOTH, False),
    ("acronym_lint.py", True, True, BOTH, False),
    ("prose_lint.py", True, True, BOTH, False),
    ("unresolved_reference_lint.py", True, True, BOTH, False),
    ("terminology_lint.py", True, True, BOTH, False),
    ("math_typeset_lint.py", False, True, BOTH, False),
    ("citation_style_lint.py", True, True, BOTH, True),
    ("caption_lint.py", True, True, BOTH, False),
    ("ai_disclosure_lint.py", True, True, (THESIS,), False),
]
LLM = [
    ("thesis_checklist_llm.py", True, False, (THESIS,), False),
    ("paper_checklist_llm.py", True, False, (PAPER,), False),
    ("data_split_lint_llm.py", True, False, BOTH, True),
    ("research_questions_lint_llm.py", True, False, BOTH, True),
    ("rq_quality_lint_llm.py", True, False, BOTH, True),
    ("related_work_faithfulness_llm.py", True, False, BOTH, True),
    ("figure_lint_llm.py", True, False, BOTH, False),
    ("section_intro_lint_llm.py", True, False, BOTH, False),
    ("flow_lint_llm.py", True, False, BOTH, False),
    ("prose_lint_llm.py", True, True, BOTH, False),
    ("caption_lint_llm.py", True, True, BOTH, False),
    ("forward_ref_lint_llm.py", True, False, BOTH, False),
    ("type_consistency_lint_llm.py", True, False, BOTH, False),
    ("contribution_support_lint_llm.py", True, False, BOTH, False),
    # Aalto-Dictionary-grounded linters. own_work_relation_lint_llm.py is
    # reviewer-only and not shipped in the published subset; it is skipped
    # automatically wherever its file is absent (see _resolve).
    ("contribution_faithfulness_lint_llm.py", True, False, BOTH, False),
    ("abstract_selfcontained_lint_llm.py", True, False, (PAPER,), False),
    ("central_concept_citation_lint_llm.py", True, False, (PAPER,), False),
    ("erm_clarity_lint_llm.py", True, False, (PAPER,), False),
    ("own_work_relation_lint_llm.py", True, False, (PAPER,), False),
]
BIB = [("bibliography_linter.py", True, False, BOTH, False)]


def _extract_annotations(src: str) -> str | None:
    """Turn --annotations input into an annotations JSON path. A .json is used
    as-is; a PDF is run through extract_annotations.py into a sidecar
    ``<stem>_annotations.json`` next to the current directory. Returns the JSON
    path, or None on failure (a warning is printed and the run continues)."""
    if src.lower().endswith(".json"):
        return src if os.path.exists(src) else None
    extractor = _resolve("extract_annotations.py")
    if extractor is None:
        print("[annotations] extract_annotations.py not found; skipping "
              "annotation augmentation.", file=sys.stderr)
        return None
    out = Path(f"{Path(src).stem}_annotations.json")
    rc = subprocess.run([sys.executable, str(extractor), src,
                         "--out", str(out)]).returncode
    if rc != 0 or not out.exists():
        print(f"[annotations] extraction failed for {src}; continuing without.",
              file=sys.stderr)
        return None
    return str(out)


def main(argv=None) -> int:
    """Parse the CLI, run each applicable linter as a subprocess, print its
    report plus a final per-linter summary, optionally fold in reviewer PDF
    annotations, and optionally build the HTML dashboard. Returns the worst
    status (0 clean, 1 any findings/error)."""
    ap = argparse.ArgumentParser(description="Run the full linter suite.")
    ap.add_argument("inputs", nargs="+", help="thesis.pdf or .tex files/dirs")
    ap.add_argument("--profile", choices=[THESIS, PAPER], default=THESIS,
                    help="Manuscript type: 'thesis' (default, ml-theses.org "
                         "rubric) or 'paper' (IEEE/ACM conference/journal "
                         "draft; skips thesis-only checks, adapts the rest).")
    ap.add_argument("--llm", action="store_true",
                    help="Also run the LLM linters (Aalto AI API).")
    ap.add_argument("--annotations", default=None, metavar="PDF_OR_JSON",
                    help="An annotated manuscript PDF (margin highlights + "
                         "comments) or a pre-extracted annotations JSON. The "
                         "comments are folded into every semantic LLM linter, "
                         "and annotation_coverage_lint_llm.py reports which "
                         "notes the suite caught. Most useful with --llm.")
    ap.add_argument("--base-url", default=None, metavar="URL",
                    help="Send the LLM linters to this OpenAI-style endpoint "
                         "instead of the Aalto AI API (e.g. a local "
                         "http://localhost:8080/v1 server). Applies to every "
                         "LLM linter in the run.")
    ap.add_argument("--model", default=None, metavar="NAME",
                    help="Chat/completions model the LLM linters use (default: "
                         "per-endpoint default in aalto_llm.py).")
    ap.add_argument("--vision-model", default=None, metavar="NAME",
                    help="Vision model for figure_lint_llm.py (default: "
                         "per-endpoint default in aalto_llm.py).")
    ap.add_argument("--bib", action="store_true",
                    help="Also run the bibliography existence check "
                         "(network queries to Crossref/arXiv/DBLP).")
    ap.add_argument("--dashboard", action="store_true",
                    help="Also render the run as a self-contained HTML "
                         "dashboard and open it in the default browser.")
    ap.add_argument("--dashboard-out", default=None, metavar="FILE",
                    help="Dashboard path (implies --dashboard; default "
                         "<input>_lint_dashboard.html in the current dir).")
    ap.add_argument("--no-open", action="store_true",
                    help="With --dashboard, write the HTML but do not open a "
                         "browser (for headless/CI runs).")
    args = ap.parse_args(argv)

    # The endpoint/model overrides are forwarded to the LLM linters through the
    # environment: each subprocess inherits this process's env, and every
    # *_llm linter reads these variables via aalto_llm.py. Setting them here is
    # uniform across the suite regardless of which CLI flags a given linter
    # exposes. A flag, when passed, wins over a pre-existing env var; omit it
    # and any env var the user already exported still stands.
    for flag, var in (("base_url", "LLM_BASE_URL"),
                      ("model", "LLM_MODEL"),
                      ("vision_model", "LLM_VISION_MODEL")):
        val = getattr(args, flag)
        if val:
            os.environ[var] = val

    # Annotated-PDF augmentation: extract the reviewer's margin comments and
    # expose them to the semantic LLM linters via $LINT_ANNOTATIONS_FILE (read
    # by aalto_llm.py). We also cross-check coverage at the end, which needs the
    # concatenated linter output -- so capturing the run is forced on.
    annotations_json = None
    if args.annotations:
        annotations_json = _extract_annotations(args.annotations)
        if annotations_json:
            os.environ["LINT_ANNOTATIONS_FILE"] = annotations_json
            print(f"[annotations] using {annotations_json}", file=sys.stderr)

    want_dashboard = args.dashboard or bool(args.dashboard_out)
    # Capture the run text when a dashboard OR annotation-coverage needs it.
    capture = want_dashboard or bool(annotations_json)
    buf = []

    def emit(text="", end="\n"):
        sys.stdout.write(text + end)
        sys.stdout.flush()
        if capture:
            buf.append(text + end)

    pdf_mode = args.inputs[0].lower().endswith(".pdf")
    todo = FAST + (LLM if args.llm else []) + (BIB if args.bib else [])
    results = []
    for script, pdf_ok, tex_ok, profiles, passes in todo:
        if args.profile not in profiles:
            continue
        if (pdf_mode and not pdf_ok) or (not pdf_mode and not tex_ok):
            continue
        resolved = _resolve(script)
        if resolved is None:
            continue  # registered but not shipped in this layout -- skip.
        extra = [f"--profile={args.profile}"] if passes else []
        emit(f"\n{'=' * 74}\n>>> {script}\n{'=' * 74}")
        cmd = [sys.executable, str(resolved), *args.inputs, *extra]
        if capture:
            # Capture stdout (the report the dashboard/coverage parses); let
            # stderr inherit so progress lines still stream to the terminal.
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            emit(proc.stdout, end="")
            rc = proc.returncode
        else:
            rc = subprocess.run(cmd).returncode
        results.append((script, rc))

    # Annotation-coverage pass: audit the run we just captured against the
    # reviewer's own annotations. Appended to the report (and the dashboard).
    if annotations_json:
        results.append(("annotation_coverage_lint_llm.py",
                        _run_annotation_coverage(buf, annotations_json, emit)))

    emit(f"\n{'=' * 74}\nSummary\n{'=' * 74}")
    worst = 0
    for script, rc in results:
        status = {0: "clean", 1: "findings"}.get(rc, f"error ({rc})")
        emit(f"  {script:<32} {status}")
        worst = max(worst, min(rc, 1))

    if want_dashboard:
        _build_dashboard(args, "".join(buf))
    return worst


def _run_annotation_coverage(buf, annotations_json, emit) -> int:
    """Feed the concatenated run so far to annotation_coverage_lint_llm.py and
    echo its report. Returns the linter's exit status (2 if unavailable)."""
    cov = _resolve("annotation_coverage_lint_llm.py")
    if cov is None:
        emit("\n[annotations] annotation_coverage_lint_llm.py not found; "
             "skipping coverage audit.")
        return 2
    # This pass must NOT see the annotations re-injected into its own context
    # (they are its explicit input); the linter clears the hook itself, but we
    # also avoid confusing it by passing the annotations only as an argument.
    with tempfile.NamedTemporaryFile("w", suffix="_lintout.txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("".join(buf))
        lintout_path = fh.name
    emit(f"\n{'=' * 74}\n>>> annotation_coverage_lint_llm.py\n{'=' * 74}")
    proc = subprocess.run(
        [sys.executable, str(cov), lintout_path,
         "--annotations", annotations_json],
        stdout=subprocess.PIPE, text=True)
    emit(proc.stdout, end="")
    try:
        os.unlink(lintout_path)
    except OSError:
        pass
    return proc.returncode


def _build_dashboard(args, run_text):
    """Render the captured run as HTML via dashboard.py and open it."""
    sys.path.insert(0, str(HERE))
    try:
        import dashboard
    except Exception as e:  # pragma: no cover - defensive
        print(f"[dashboard] could not import dashboard.py: {e}",
              file=sys.stderr)
        return
    sections, summary = dashboard.parse_run(run_text)
    if not sections:
        print("[dashboard] no linter sections captured; skipping HTML.",
              file=sys.stderr)
        return
    out = _dashboard_out_path(args.inputs, args.dashboard_out)
    title = Path(args.inputs[0]).stem or "Master's thesis"
    flags = [f"--profile {args.profile}"]
    if args.llm:
        flags.append("--llm")
    if args.bib:
        flags.append("--bib")
    if args.annotations:
        flags.append("--annotations")
    meta = f"{Path(args.inputs[0]).name} · run_all_linters.py {' '.join(flags)}"
    doc, n_flag = dashboard.render(sections, summary, title, "&nbsp;",
                                   meta, body_only=False)
    out.write_text(doc, encoding="utf-8")
    print(f"[dashboard] wrote {out} ({len(sections)} linters, {n_flag} with "
          f"findings/errors)", file=sys.stderr)
    if args.no_open:
        return
    if dashboard.open_in_browser(out):
        print(f"[dashboard] opened {out} in your browser", file=sys.stderr)
    else:
        print(f"[dashboard] could not open a browser; open {out} manually",
              file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
