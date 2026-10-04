#!/usr/bin/env python3
"""
Compile the list of theses supervised by Alex Jung.

The list covers every level (bachelor's, master's, doctoral) and every host
university (Aalto University, IMC Krems, TU Wien, ...). Reads thesis data from
a CSV file (theses.csv) and generates:
  - A summary printed to the console
  - A Markdown file (theses.md) with tables grouped by university
  - Basic statistics (per level, per year, per industry partner, ongoing vs
    completed)

Usage:
    python compile_theses.py                  # print summary
    python compile_theses.py --markdown       # also write theses.md
    python compile_theses.py --stats          # print statistics
    python compile_theses.py --level PhD      # filter by thesis level
"""

import argparse
import csv
import os
import re
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "theses.csv"
DEFAULT_MD = SCRIPT_DIR / "theses.md"
DEFAULT_TEX = SCRIPT_DIR / "theses.tex"

# Markers delimiting the generated thesis list on the alexjung.at supervision
# page (see --supervision). Everything between them is replaced on each sync;
# the hand-written intro above the markers is left untouched.
SUPERVISION_BEGIN = (
    "<!-- theses:begin — generated from theses.csv by"
    " `compile_theses.py --supervision`; do not edit by hand -->"
)
SUPERVISION_END = "<!-- theses:end -->"

# Thesis levels, in the order they are listed when a university has several.
# The CSV `level` column holds the short key; an empty level means MSc (the
# column was added after the list already held 130+ master's theses).
LEVELS = {
    "PhD": {"label": "Doctoral thesis", "citation": "Ph.D. dissertation"},
    "MSc": {"label": "Master's thesis", "citation": "M.Sc. thesis"},
    "BSc": {"label": "Bachelor's thesis", "citation": "B.Sc. thesis"},
}
DEFAULT_LEVEL = "MSc"

# Short university keys used in the CSV -> full names for the published pages.
UNI_DISPLAY = {
    "Aalto": "Aalto University",
    "IMC Krems": "IMC Krems University of Applied Sciences",
    "TU Wien": "TU Wien",
}


def normalize_level(raw: str) -> str:
    """Map a CSV level value onto a LEVELS key (case-insensitive, '' -> MSc)."""
    key = (raw or "").strip()
    if not key:
        return DEFAULT_LEVEL
    for k in LEVELS:
        if k.lower() == key.lower():
            return k
    raise SystemExit(
        f"Error: unknown thesis level {raw!r}; expected one of {', '.join(LEVELS)}"
    )


def uni_display(uni: str) -> str:
    """Full display name for a university key (falls back to the key)."""
    return UNI_DISPLAY.get(uni, uni)


def load_theses(csv_path: Path) -> list[dict]:
    """Load thesis records from a CSV file."""
    theses = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["number"] = int(row["number"])
            row["level"] = normalize_level(row.get("level", ""))
            theses.append(row)
    return theses


def filter_theses(
    theses: list[dict],
    university: str | None = None,
    status: str | None = None,
    year: str | None = None,
    industry: str | None = None,
    level: str | None = None,
) -> list[dict]:
    """Filter thesis records by university, level, status, year, or industry."""
    result = theses
    if university:
        result = [t for t in result if t["university"].lower() == university.lower()]
    if level:
        wanted = normalize_level(level)
        result = [t for t in result if t["level"] == wanted]
    if status:
        result = [t for t in result if t["status"].lower() == status.lower()]
    if year:
        result = [t for t in result if t["date"].endswith(year) or t["date"].startswith(year)]
    if industry:
        result = [t for t in result if industry.lower() in t["industry"].lower()]
    return result


def print_summary(theses: list[dict]) -> None:
    """Print a plain-text summary grouped by university."""
    by_uni = {}
    for t in theses:
        by_uni.setdefault(t["university"], []).append(t)

    for uni, records in by_uni.items():
        print(f"\n{'='*70}")
        print(f"  {uni_display(uni)}  ({len(records)} theses)")
        print(f"{'='*70}")
        for t in records:
            status_tag = " [ongoing]" if t["status"].lower() == "ongoing" else ""
            industry_tag = f"  (industry: {t['industry']})" if t["industry"] else ""
            url_tag = f"  {t['url']}" if t["url"] else ""
            print(
                f"  {t['number']:>3}. [{t['level']}] {t['author']}, "
                f"{t['title']}{status_tag}{industry_tag}"
            )
            if url_tag:
                print(f"       {t['url']}")
    print()


def level_summary(theses: list[dict]) -> str:
    """One sentence with the total and the per-level counts, e.g.
    '137 theses in total: 1 doctoral, 135 master's, 1 bachelor's.'"""
    counts = Counter(t["level"] for t in theses)
    parts = [
        f"{counts[k]} {v['label'].lower().replace(' thesis', '')}"
        for k, v in LEVELS.items()
        if counts[k]
    ]
    return f"{len(theses)} theses in total: {', '.join(parts)}."


def generate_markdown(theses: list[dict], output_path: Path) -> None:
    """Write a Markdown file with thesis tables grouped by university."""
    by_uni = {}
    for t in theses:
        by_uni.setdefault(t["university"], []).append(t)

    lines = [
        "# Supervised Theses",
        "",
        "Bachelor's, master's, and doctoral theses supervised or co-supervised by "
        "[Alex Jung](https://machinelearningforall.github.io/about/), grouped by "
        "host university. " + level_summary(theses),
        "",
        "The **Level** column uses " + ", ".join(
            f"`{k}` for a {v['label'].lower()}" for k, v in LEVELS.items()
        ) + ". Generated from `theses.csv`; do not edit this file by hand.",
        "",
    ]

    for uni, records in by_uni.items():
        ongoing = [r for r in records if r["status"].lower() == "ongoing"]
        completed = [r for r in records if r["status"].lower() != "ongoing"]

        lines.append(f"## {uni_display(uni)} ({len(records)} total)\n")

        if ongoing:
            lines.append(f"### Ongoing ({len(ongoing)})\n")
            lines.append("| # | Level | Author | Title | Industry | Recording |")
            lines.append("|---|-------|--------|-------|----------|-----------|")
            for i, t in enumerate(ongoing, 1):
                rec = f"[video]({t['recording']})" if t.get("recording") else ""
                lines.append(
                    f"| {i} | {t['level']} | {t['author']} | {t['title']} "
                    f"| {t['industry']} | {rec} |"
                )
            lines.append("")

        if completed:
            lines.append(f"### Completed ({len(completed)})\n")
            lines.append("| # | Level | Author | Title | Date | Industry | Recording |")
            lines.append("|---|-------|--------|-------|------|----------|-----------|")
            for i, t in enumerate(completed, 1):
                title = f"[{t['title']}]({t['url']})" if t["url"] else t["title"]
                rec = f"[video]({t['recording']})" if t.get("recording") else ""
                lines.append(
                    f"| {i} | {t['level']} | {t['author']} | {title} "
                    f"| {t['date']} | {t['industry']} | {rec} |"
                )
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown written to {output_path}")


def tex_escape(s: str) -> str:
    """Escape LaTeX special characters in a string."""
    if not s:
        return ""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for a, b in replacements:
        s = s.replace(a, b)
    return s


def generate_tex(theses: list[dict], output_path: Path) -> None:
    """Write a standalone LaTeX file with thesis tables grouped by university."""
    by_uni = {}
    for t in theses:
        by_uni.setdefault(t["university"], []).append(t)

    lines = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{booktabs}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{xurl}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.5em}",
        r"",
        r"\title{Supervised Theses \\[0.3em]"
        r" \large Bachelor's, master's, and doctoral theses supervised by"
        r" Alex Jung, Assoc.\ Prof.\ for Machine Learning}",
        r"\author{}",
        r"\date{\today}",
        r"",
        r"\begin{document}",
        r"\maketitle",
        r"",
    ]

    for uni, records in by_uni.items():
        ongoing = [r for r in records if r["status"].lower() == "ongoing"]
        completed = [r for r in records if r["status"].lower() != "ongoing"]

        lines.append(rf"\section*{{{tex_escape(uni_display(uni))} ({len(records)} total)}}")
        lines.append("")

        if ongoing:
            lines.append(rf"\subsection*{{Ongoing ({len(ongoing)})}}")
            lines.append("")
            lines.append(r"\begin{longtable}{@{}r l p{2.8cm} p{6cm} p{2.6cm} p{1.8cm}@{}}")
            lines.append(r"\toprule")
            lines.append(r"\# & Level & Author & Title & Industry & Recording \\")
            lines.append(r"\midrule")
            lines.append(r"\endfirsthead")
            lines.append(r"\toprule")
            lines.append(r"\# & Level & Author & Title & Industry & Recording \\")
            lines.append(r"\midrule")
            lines.append(r"\endhead")
            lines.append(r"\bottomrule")
            lines.append(r"\endfoot")
            for i, t in enumerate(ongoing, 1):
                rec = rf"\href{{{t['recording']}}}{{video}}" if t.get("recording") else ""
                lines.append(
                    f"{i} & {t['level']} & {tex_escape(t['author'])} & "
                    f"{tex_escape(t['title'])} & {tex_escape(t['industry'])} & {rec} \\\\"
                )
            lines.append(r"\end{longtable}")
            lines.append("")

        if completed:
            lines.append(rf"\subsection*{{Completed ({len(completed)})}}")
            lines.append("")
            lines.append(r"\begin{longtable}{@{}r l p{2.6cm} p{5.8cm} p{1.6cm} p{1.9cm} p{1.6cm}@{}}")
            lines.append(r"\toprule")
            lines.append(r"\# & Level & Author & Title & Date & Industry & Recording \\")
            lines.append(r"\midrule")
            lines.append(r"\endfirsthead")
            lines.append(r"\toprule")
            lines.append(r"\# & Level & Author & Title & Date & Industry & Recording \\")
            lines.append(r"\midrule")
            lines.append(r"\endhead")
            lines.append(r"\bottomrule")
            lines.append(r"\endfoot")
            for i, t in enumerate(completed, 1):
                title_tex = tex_escape(t['title'])
                if t['url']:
                    title_tex = rf"\href{{{t['url']}}}{{{title_tex}}}"
                rec = rf"\href{{{t['recording']}}}{{video}}" if t.get("recording") else ""
                lines.append(
                    f"{i} & {t['level']} & {tex_escape(t['author'])} & {title_tex} & "
                    f"{tex_escape(t['date'])} & {tex_escape(t['industry'])} & {rec} \\\\"
                )
            lines.append(r"\end{longtable}")
            lines.append("")

    lines.append(r"\end{document}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX written to {output_path}")


def thesis_year(t: dict) -> int:
    """Extract the four-digit year from a thesis record's date field."""
    m = re.search(r"\b(19|20)\d{2}\b", t["date"])
    return int(m.group(0)) if m else 0


def render_supervision_entry(t: dict, label: str, display: str) -> str:
    """Render one thesis as an IEEE-style citation line for the Jekyll page."""
    kind = LEVELS[t["level"]]["citation"]
    parts = [f'[{label}]&nbsp;&nbsp;{t["author"]}, "{t["title"]},"']
    if t["status"].lower() == "ongoing":
        parts.append(f"{kind} (in progress), {display}.")
    else:
        parts.append(f"{kind}, {display}, {t['date']}.")
    if t["industry"]:
        parts.append(f"(with {t['industry']})")
    if t["status"].lower() != "ongoing" and t["url"]:
        parts.append(f"[link]({t['url']})")
    if t.get("recording"):
        parts.append(f"[video]({t['recording']})")
    return " ".join(parts)


def generate_supervision(theses: list[dict], page_path: Path) -> None:
    """Regenerate the thesis list on the alexjung.at supervision page.

    Replaces the content between SUPERVISION_BEGIN/SUPERVISION_END markers in
    the Jekyll page, keeping the hand-written front matter and intro intact.
    Ongoing theses are labeled [O1], [O2], ...; completed theses are numbered
    continuously across universities and grouped by year (newest first), with
    everything up to 2017 collected under a single heading.
    """
    by_uni: dict[str, list[dict]] = {}
    for t in theses:
        by_uni.setdefault(t["university"], []).append(t)

    lines: list[str] = []
    completed_no = 0
    for uni, records in by_uni.items():
        display = uni_display(uni)
        ongoing = [r for r in records if r["status"].lower() == "ongoing"]
        completed = [r for r in records if r["status"].lower() != "ongoing"]
        # Newest first; stable, so CSV order is kept within a year.
        completed.sort(key=thesis_year, reverse=True)

        lines += [f"## {display}", ""]

        if ongoing:
            lines += ["### Ongoing", ""]
            for i, t in enumerate(ongoing, 1):
                lines += [render_supervision_entry(t, f"O{i}", display), ""]

        if completed:
            if ongoing:
                lines += ["### Completed", ""]
            # Group by year, folding 2017 and older into one bucket. Skip the
            # year headings entirely when a university has a single group
            # (e.g. TU Wien).
            group = lambda t: max(thesis_year(t), 2017)
            headings = len({group(t) for t in completed}) > 1
            prev = None
            for t in completed:
                if headings and group(t) != prev:
                    prev = group(t)
                    label = str(prev) if prev > 2017 else "2017 and earlier"
                    lines += [f"#### {label}", ""]
                completed_no += 1
                lines += [render_supervision_entry(t, str(completed_no), display), ""]

    page = page_path.read_text(encoding="utf-8")
    if SUPERVISION_BEGIN not in page or SUPERVISION_END not in page:
        raise SystemExit(
            f"Error: markers not found in {page_path}.\n"
            f"Add these two lines around the generated thesis list:\n"
            f"  {SUPERVISION_BEGIN}\n  {SUPERVISION_END}"
        )
    head, rest = page.split(SUPERVISION_BEGIN, 1)
    _, tail = rest.split(SUPERVISION_END, 1)
    body = "\n".join([SUPERVISION_BEGIN, "", *lines]).rstrip() + "\n" + SUPERVISION_END
    page_path.write_text(head + body + tail, encoding="utf-8")
    n_ongoing = sum(1 for t in theses if t["status"].lower() == "ongoing")
    print(
        f"Supervision page updated: {page_path} "
        f"({n_ongoing} ongoing, {completed_no} completed)"
    )


def print_stats(theses: list[dict]) -> None:
    """Print summary statistics."""
    print(f"\nTotal theses: {len(theses)}")

    # By university
    uni_counts = Counter(t["university"] for t in theses)
    print("\nBy university:")
    for uni, count in uni_counts.most_common():
        print(f"  {uni_display(uni)}: {count}")

    # By level
    level_counts = Counter(t["level"] for t in theses)
    print("\nBy level:")
    for key, meta in LEVELS.items():
        if level_counts[key]:
            print(f"  {meta['label']} ({key}): {level_counts[key]}")

    # Ongoing vs completed
    ongoing = sum(1 for t in theses if t["status"].lower() == "ongoing")
    print(f"\nOngoing: {ongoing}")
    print(f"Completed: {len(theses) - ongoing}")

    # By year (completed only)
    completed = [t for t in theses if t["status"].lower() != "ongoing"]
    year_counts = Counter()
    for t in completed:
        parts = t["date"].replace(",", "").split()
        for part in parts:
            if part.isdigit() and len(part) == 4:
                year_counts[part] += 1
                break
    print("\nCompleted by year:")
    for year, count in sorted(year_counts.items()):
        print(f"  {year}: {count}")

    # Top industry partners
    partners = [t["industry"] for t in theses if t["industry"]]
    partner_counts = Counter(partners)
    print(f"\nIndustry-partnered theses: {len(partners)}")
    print("Top industry partners:")
    for partner, count in partner_counts.most_common(10):
        print(f"  {partner}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Compile the list of supervised theses (all levels, all universities)."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to theses.csv")
    parser.add_argument("--markdown", action="store_true", help="Generate theses.md")
    parser.add_argument("--output", type=Path, default=DEFAULT_MD, help="Markdown output path")
    parser.add_argument("--tex", action="store_true", help="Generate theses.tex")
    parser.add_argument("--tex-output", type=Path, default=DEFAULT_TEX, help="LaTeX output path")
    parser.add_argument(
        "--supervision",
        type=Path,
        metavar="PAGE",
        help="Sync the thesis list into the alexjung.at supervision page (Jekyll .md)",
    )
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    parser.add_argument(
        "--university", type=str, help="Filter by university key (e.g. Aalto, 'IMC Krems')"
    )
    parser.add_argument(
        "--level", type=str, help="Filter by thesis level: " + ", ".join(LEVELS)
    )
    parser.add_argument("--status", type=str, help="Filter by status (ongoing/completed)")
    parser.add_argument("--year", type=str, help="Filter by year")
    parser.add_argument("--industry", type=str, help="Filter by industry partner")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Error: CSV file not found at {args.csv}")
        print(f"Please create it first. See theses.csv for the expected format.")
        return 1

    theses = load_theses(args.csv)
    theses = filter_theses(
        theses, args.university, args.status, args.year, args.industry, args.level
    )

    print_summary(theses)

    if args.stats:
        print_stats(theses)

    if args.markdown:
        generate_markdown(theses, args.output)

    if args.tex:
        generate_tex(theses, args.tex_output)

    if args.supervision:
        if not args.supervision.exists():
            # SystemExit so the message reaches stderr even when stdout is
            # redirected (build_site.sh silences the summary output).
            raise SystemExit(f"Error: supervision page not found at {args.supervision}")
        generate_supervision(theses, args.supervision)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
