#!/usr/bin/env python3
"""Extract a PDF's margin annotations (highlights, notes) to JSON.

Each annotation becomes {page, type, comment, quoted}, where `comment` is the
reviewer's typed text and `quoted` is the underlying page text the annotation
covers (for highlights). Feeds both the annotation-aware LLM linting (via
$LINT_ANNOTATIONS_FILE) and the annotation-coverage linter.

Usage: python3 extract_annotations.py draft.pdf --out lintout/draft_annotations.json
"""
import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import fitz  # PyMuPDF

    doc = fitz.open(args.pdf)
    rows = []
    for i, page in enumerate(doc):
        for an in (page.annots() or []):
            info = an.info
            comment = (info.get("content", "") or "").replace("\r", " ").strip()
            quoted = ""
            if an.type[1] in ("Highlight", "Underline", "Squiggly", "StrikeOut"):
                try:
                    quoted = page.get_textbox(an.rect).replace("\n", " ").strip()
                except Exception:
                    quoted = ""
            rows.append({"page": i + 1, "type": an.type[1],
                         "comment": comment, "quoted": quoted[:200]})

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1, ensure_ascii=False)
    print(f"{len(rows)} annotation(s) -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
