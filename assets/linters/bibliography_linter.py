#!/usr/bin/env python3
"""Bibliography linter: verify that cited references exist and that their
author list, title, and publication venue are correct.

Sources checked against (no API keys needed, but see OpenAlex):
  - OpenAlex  (api.openalex.org)   - primary; a cited DOI is resolved directly
                                     and treated as authoritative, otherwise a
                                     title search. Without a key OpenAlex
                                     allows only ~100 searches/day per IP;
                                     export OPENALEX_API_KEY for more.
  - Crossref  (api.crossref.org)   - journals + proceedings, title search
  - arXiv     (export.arxiv.org)   - entries citing an arXiv identifier
  - DBLP      (dblp.org)           - secondary for CS venues; best-effort
                                     (DBLP rate-limits aggressively; failures
                                     are tolerated)

Input formats:
  *.bib  - BibTeX file (lightweight built-in parser, no dependencies)
  *.pdf  - compiled manuscript. The References section is located and split
           automatically. Two list styles are recognised:
             IEEE numbered:  "[1] A. Author, ...\"Title,\" Venue, year."
             author-year:    "Surname, I.N., Surname, I.: Title. Venue
                              vol(issue), pages (year) https://doi.org/..."
                             (Springer/Nature "sn-*" styles, APA-like lists);
                             used automatically when no "[n]" markers exist.
           URLs/DOIs broken across lines by the PDF text extraction are
           re-joined before parsing.

Findings (severity in brackets):
  [WARN]  NOT-FOUND         no plausible match in any database -- entry may be
                            garbled or too new to be indexed; verify by hand
                            (absence of a match is not proof of fabrication)
  [ERROR] DOI-MISMATCH      the cited DOI resolves to a different work
  [ERROR] AUTHOR-MISMATCH   matched record's authors differ from the citation
  [WARN]  TITLE-DRIFT       best match found but title similarity is imperfect
  [WARN]  VENUE-MISMATCH    cited venue disagrees with the matched record
  [WARN]  PREPRINT          cited as arXiv/SSRN/URL-only although a published
                            venue exists for the same work
  [WARN]  YEAR-MISMATCH     year differs by more than 1 from the matched record
  [INFO]  INCOMPLETE        entry lacks a parseable title/author/year
  [INFO]  WEB-SOURCE        dataset/website/repository citation -- not checked
                            against publication databases
  [INFO]  UNVERIFIED        all database queries failed (network/rate limit)

Results are cached in a JSON file (--cache, default .bibcheck_cache.json next
to the input) so re-runs are cheap. Queries are rate-limited (--delay).

Usage:
  python3 bibliography_linter.py thesis.pdf
  python3 bibliography_linter.py refs.bib --max 20 --delay 1.0
  python3 bibliography_linter.py paper.pdf --parse-only   # show extracted refs
Exit status: 0 if no ERROR/WARN findings, 1 otherwise, 2 usage error.
"""

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

MAILTO = "alex.jung@aalto.fi"
UA = {"User-Agent": f"bibliography-linter/1.1 (thesis supervision; mailto:{MAILTO})"}
# OpenAlex meters unauthenticated use per IP address (a shared free daily
# budget of ~100 title searches, reset at midnight UTC; a search costs ten
# credits, a DOI lookup one). Export OPENALEX_API_KEY (free key from
# https://openalex.org/settings/api) to get an own, far larger budget.
OPENALEX_KEY = os.environ.get("OPENALEX_API_KEY", "")
OPENALEX_AUTH = f"&mailto={MAILTO}" + (f"&api_key={OPENALEX_KEY}" if OPENALEX_KEY else "")
# A DOI: "10.<registrant>/<suffix>"; trailing sentence punctuation is stripped.
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)
# Accept the inline forms ("arXiv:2308.03688", "abs/2308.03688") AND the DOI
# form used by recent entries ("doi: 10.48550/arXiv.2604.16338", where a dot,
# not a colon, follows "arxiv"). Missing the DOI form meant the arXiv id was
# never extracted, the authoritative arXiv lookup was skipped, and the entry
# fell back to a fuzzy title search that mismatched brand-new papers.
ARXIV_ID_RE = re.compile(
    r"(?:ar[xX]iv[:\s.]*|abs/|10\.48550/ar[xX]iv\.)(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})",
    re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+|doi\.org/\S+", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
WEB_HOST_RE = re.compile(
    r"kaggle|roboflow|zenodo|hugging\s*face|huggingface|github|gitlab|medium\.com|"
    r"towardsdatascience|wikipedia|blog|\bdataset\b", re.IGNORECASE)
PREPRINT_RE = re.compile(r"arxiv|preprint|corr\b|ssrn|techrxiv|biorxiv", re.IGNORECASE)
STOPWORDS = {"the", "a", "an", "of", "for", "and", "with", "via", "using",
             "from", "towards", "toward", "on", "in", "to", "by"}


def title_tokens(s):
    """Return the distinctive lowercase words of a title: punctuation is
    dropped and stopwords plus very short tokens (<=2 chars) are removed.
    Used to build loose search queries and to compare titles word-by-word."""
    return [t for t in re.sub(r"[^a-z0-9]+", " ", s.lower()).split()
            if t not in STOPWORDS and len(t) > 2]


# --------------------------------------------------------------------------
# parsing: BibTeX
# --------------------------------------------------------------------------

def parse_bib(path: Path):
    """Parse a BibTeX file into a list of normalised entry dicts, each with
    keys id/title/authors/venue/year/arxiv/url/raw. Uses a lightweight
    regex/brace-matching parser (no third-party dependency); comment,
    string and preamble blocks are skipped."""
    text = path.read_text(encoding="utf-8", errors="replace")
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        etype, key = m.group(1).lower(), m.group(2)
        if etype in {"comment", "string", "preamble"}:
            continue
        # take the balanced-brace body
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[m.end():i - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|\w+)",
                              body):
            fields[fm.group(1).lower()] = fm.group(2).strip("{}\"").strip()
        title = re.sub(r"[{}]", "", fields.get("title", ""))
        authors = [a.strip() for a in
                   re.split(r"\s+and\s+", fields.get("author", "")) if a.strip()]
        venue = fields.get("journal") or fields.get("booktitle") or \
            fields.get("publisher") or fields.get("howpublished") or ""
        raw = f"@{etype}{{{key}}} {fields.get('author','')} \"{title}\" {venue}"
        entries.append({
            "id": key, "title": title, "authors": authors, "venue": venue,
            "year": fields.get("year", ""), "arxiv": _find_arxiv(body),
            "url": fields.get("url", ""), "raw": raw,
            "doi": _find_doi(fields.get("doi", "") or body),
        })
    return entries


def _find_arxiv(text):
    """Return the first arXiv identifier found in text (inline, abs/, or
    DOI form), or "" if none is present."""
    m = ARXIV_ID_RE.search(text)
    return m.group(1) if m else ""


def _find_doi(text):
    """Return the first DOI found in text (with trailing punctuation
    stripped), or "". arXiv DOIs (10.48550/...) are ignored: the arXiv id is
    extracted separately and resolved against arXiv itself."""
    for m in DOI_RE.finditer(text):
        doi = m.group(1).rstrip(".,;:)]")
        if not doi.lower().startswith("10.48550/"):
            return doi
    return ""


# --------------------------------------------------------------------------
# parsing: PDF with IEEE-style numbered references
# --------------------------------------------------------------------------

def extract_pdf_text(path: Path):
    """Return the full text of a PDF. Prefers the `pdftotext -layout`
    command-line tool (preserves column/right-margin layout, which the
    equation and reference parsing relies on); falls back to PyMuPDF
    (`fitz`) if pdftotext is not installed or fails."""
    try:
        out = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                             capture_output=True, check=True)
        return out.stdout.decode("utf-8", errors="replace")
    except (FileNotFoundError, subprocess.CalledProcessError):
        import fitz
        with fitz.open(path) as doc:
            return "\f".join(p.get_text() for p in doc)


def parse_pdf_references(path: Path):
    """Locate the reference list of a compiled manuscript and parse it into
    entry dicts (same shape as parse_bib). The last 'References'/'Bibliography'
    heading is taken as the start, an appendix heading (if any) as the end,
    and the body is split on line-leading "[n]" markers. Exits the program
    if no reference section can be found."""
    text = extract_pdf_text(path)
    # find the References heading closest to the end
    heads = [m.start() for m in
             re.finditer(r"^\s*(References|Bibliography)\s*$", text, re.MULTILINE)]
    if not heads:
        # Looser fallback: "REFERENCES", "7 References", "References and Notes"
        heads = [m.start() for m in
                 re.finditer(r"^\s*(?:\d+\.?\s+)?(References|Bibliography)\b[^\n]{0,20}$",
                             text, re.MULTILINE | re.IGNORECASE)]
    if not heads:
        sys.exit(f"error: no 'References' section found in {path}")
    refs_text = text[heads[-1]:]
    # stop at an appendix heading if one follows
    stop = re.search(r"^\s*(?:A\s+)?Appendix\b", refs_text, re.MULTILINE)
    if stop and stop.start() > 100:
        refs_text = refs_text[:stop.start()]

    # split on bracketed numbers at line starts, joining wrapped lines
    chunks = re.split(r"(?m)^\s*\[(\d+)\]", refs_text)
    entries = []
    for i in range(1, len(chunks) - 1, 2):
        num, body = chunks[i], _join_lines(chunks[i + 1])
        entries.append(_parse_ieee_entry(num, body))
    entries = [e for e in entries if e]
    if not entries:
        # No "[n]" markers: an author-year list (Springer/Nature "sn-*"
        # styles, APA-like), where entries are blank-line separated
        # paragraphs and continuation lines are indented.
        entries = _parse_authoryear_list(refs_text)
    return entries


def _join_lines(block):
    """Collapse the lines of one reference into a single string. A line that
    ends inside a URL/DOI/arXiv id (pdftotext wraps long identifiers at
    arbitrary points: "https://doi.org/10.1109/TNNLS.2022." + "3217694") is
    joined to the next line WITHOUT a space; everything else with one."""
    out = ""
    for line in block.splitlines():
        line = line.strip()
        # Drop running page numbers that pdftotext leaves between the lines
        # of an entry broken by a page break.
        if not line or re.fullmatch(r"\d{1,4}", line):
            continue
        if out and re.search(r"(https?:|doi\.org|arxiv\.org|\b10\.\d{4,9}/)\S*$",
                             out, re.IGNORECASE) and out[-1] in "./:-_":
            out += line
        elif out and out.endswith("-") and line[:1].islower():
            # End-of-line hyphenation ("impu-" + "tation"): keep the hyphen but
            # no space, so a genuine compound ("time-" + "series") stays
            # intact and a split word differs from the real one by one char.
            out += line
        else:
            out += (" " if out else "") + line
    return re.sub(r"\s+", " ", out).strip()


# --------------------------------------------------------------------------
# parsing: PDF with an author-year reference list (Springer "sn-*", APA-like)
# --------------------------------------------------------------------------

# "Surname, I.N., Surname, I.: Title ..." -- the author block ends at the first
# initial that is followed by a colon.
AUTHORYEAR_HEAD_RE = re.compile(
    r"^(?P<authors>[^:]{3,400}?[A-Za-z]\.)\s*:\s+(?P<rest>.+)$", re.DOTALL)
INITIALS_RE = re.compile(r"^(?:[A-Za-z]\.)(?:-?[A-Za-z]\.)*$|^[A-Z]{1,3}$")
PAREN_YEAR_RE = re.compile(r"\(((?:19|20)\d{2})[a-z]?\)")


def _parse_authoryear_list(refs_text):
    """Split an author-year reference section into entries. Paragraphs that
    do not start like an entry are either junk (page numbers, figure/table
    material that floated into the list) or the continuation of an entry
    broken by a page break; the latter is recognised by carrying a year,
    page range, DOI or URL and is appended to the previous entry."""
    paras = [p for p in re.split(r"\n\s*\n", refs_text) if p.strip()]
    bodies = []
    for p in paras[1:] if re.match(r"\s*(References|Bibliography)", paras[0]) else paras:
        body = _join_lines(p)
        if re.fullmatch(r"\d{1,4}", body) or re.match(r"(Fig(ure)?|Table)\b\.?\s", body):
            continue
        if AUTHORYEAR_HEAD_RE.match(body):
            bodies.append(body)
        elif bodies and re.search(r"\((?:19|20)\d{2}\)|\bpp?\.\s*\d|doi|https?://|\d+[–-]\d+",
                                  body):
            bodies[-1] += " " + body
    entries = []
    for n, body in enumerate(bodies, 1):
        e = _parse_authoryear_entry(n, body)
        if e:
            entries.append(e)
    return entries


def _parse_authoryear_entry(num, body):
    """Parse one "Surname, I., Surname, I.: Title. Venue vol(no), pages (year)
    doi" string into an entry dict. The title runs from the colon to the first
    sentence-ending period (periods after initials or in "vs."/"e.g." do not
    count); the venue is what follows up to the parenthesised year, with
    volume/issue/page fields and "In:" stripped."""
    m = AUTHORYEAR_HEAD_RE.match(body)
    if not m:
        return None
    authors_part, rest = m.group("authors"), m.group("rest").strip()
    authors_part = re.sub(r"\bet\s+al\.?", "", authors_part, flags=re.IGNORECASE)
    # "Surname, I.N., Surname2, I." -> walk the comma-separated tokens; an
    # initials token attaches to the preceding surname.
    authors = []
    for tok in [t.strip() for t in authors_part.split(",") if t.strip()]:
        if INITIALS_RE.match(tok) and authors:
            continue
        authors.append(tok)
    # An edited-volume chapter: "In: Editor, E. (eds.) Title, pp. ..."
    if rest.startswith("In:") and "(eds.)" in rest:
        rest = rest.split("(eds.)", 1)[1].strip()
    cut = re.search(r"(?<![A-Z])(?<!vs)(?<!e\.g)(?<!i\.e)(?<!\bpp)(?<!\bvol)(?<!\bno)"
                    r"\.\s+(?=(?:In:\s|[A-Za-z\d(]))", rest)
    if cut:
        title, after = rest[:cut.start()].strip(), rest[cut.end():].strip()
    else:
        ym = PAREN_YEAR_RE.search(rest)
        title, after = (rest[:ym.start()].strip(" ,."), rest[ym.start():]) if ym else (rest, "")
    # A chapter title may carry its page range ("..., pp. 281–299"); move it out.
    pages = re.search(r",?\s*\bpp?\.\s*[\d–\-]+\s*$", title)
    if pages:
        title, after = title[:pages.start()], title[pages.start():] + " " + after
    title = title.rstrip(".,")
    body_for_year = URL_RE.sub("", ARXIV_ID_RE.sub("", body))
    ym = list(PAREN_YEAR_RE.finditer(body_for_year)) or list(YEAR_RE.finditer(body_for_year))
    year = ym[-1].group(1) if ym and ym[-1].re is PAREN_YEAR_RE else (ym[-1].group(0) if ym else "")
    venue = re.sub(r"^In:\s*", "", after)
    venue = PAREN_YEAR_RE.split(venue)[0]            # drop "(year) doi ..." tail
    venue = URL_RE.sub("", venue)
    venue = re.sub(r",?\s*(?:pp?|vol)\.\s*[\dA-Za-z–\-]+", "", venue)   # pp. 12–34
    venue = re.sub(r",?\s*\d+\s*[–-]\s*\d+\s*$", "", venue)             # bare 1515–1524
    venue = re.sub(r"\s*\b\d+(?:\(\d+\))?,\s*[\dA-Za-z–\-]+\s*$", "", venue)  # 34(2), 1–9
    venue = re.sub(r"\s*\b\d+(?:\(\d+\))?\s*$", "", venue)              # trailing volume
    venue = venue.strip(" ,.;")
    url_m = URL_RE.search(body)
    return {
        "id": f"[{num}]", "title": title, "authors": authors, "venue": venue,
        "year": year, "arxiv": _find_arxiv(body), "doi": _find_doi(body),
        "url": url_m.group(0) if url_m else "", "raw": body[:220],
    }


def _parse_ieee_entry(num, body):
    """Parse one IEEE-style reference string into an entry dict. The quoted
    span is taken as the title, text before it as the author list, and text
    after it as venue/year/identifiers. Returns the entry dict keyed by
    id "[num]"."""
    # Title is the first straight- or curly-quoted span.
    title_m = re.search(r"[\"“]\s*(.+?)[,.]?\s*[\"”]", body)
    title = title_m.group(1).strip() if title_m else ""
    authors_part = body[:title_m.start()].strip(" ,") if title_m else ""
    # Strip "et al." wherever it appears: otherwise "Cheng Qian et al." survives
    # as a single author token whose last name parses to "al", producing spurious
    # AUTHOR-MISMATCH findings against the record's real author list.
    authors_part = re.sub(r"\bet\s+al\.?", "", authors_part, flags=re.IGNORECASE)
    rest = body[title_m.end():].strip(" ,") if title_m else body
    authors = [a.strip() for a in
               re.split(r",\s*(?:and\s+)?|\s+and\s+", authors_part)
               if a.strip() and not a.strip().lower().startswith("et al")]
    # arXiv ids ("arXiv:1905.11946") and URLs contain digit runs that must not
    # be mistaken for years
    body_for_year = URL_RE.sub("", ARXIV_ID_RE.sub("", body))
    year_m = list(YEAR_RE.finditer(body_for_year))
    year = year_m[-1].group(0) if year_m else ""
    # Venue is the text up to the first "vol./no./pp./p." field; the year is
    # stripped out so it does not become part of the venue name.
    venue = re.split(r",\s*(?:vol|no|pp|p)\.\s", rest)[0]
    venue = YEAR_RE.sub("", venue).strip(" ,.")
    return {
        "id": f"[{num}]", "title": title, "authors": authors, "venue": venue,
        "year": year, "arxiv": _find_arxiv(body), "doi": _find_doi(body),
        "url": URL_RE.search(body).group(0)
        if URL_RE.search(body) else "", "raw": body[:220],
    }


# --------------------------------------------------------------------------
# database queries
# --------------------------------------------------------------------------

def _get(url, timeout=20):
    """Perform an HTTP GET (with the linter's User-Agent) and return the
    decoded response body as text."""
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def query_crossref(title):
    """Search Crossref by bibliographic title and return up to five candidate
    records as normalised dicts (title/authors/venue/year/type/doi/source)."""
    q = urllib.parse.quote(title)
    data = json.loads(_get(
        f"https://api.crossref.org/works?query.bibliographic={q}&rows=5"))
    out = []
    for it in data.get("message", {}).get("items", []):
        out.append({
            "title": (it.get("title") or [""])[0],
            "authors": [a.get("family", "") for a in it.get("author", [])],
            "venue": (it.get("container-title") or [""])[0]
            or it.get("publisher", ""),
            "year": str((it.get("issued", {}).get("date-parts") or [[None]])[0][0] or ""),
            "type": it.get("type", ""), "doi": it.get("DOI", ""),
            "source": "crossref",
        })
    return out


def query_dblp(title):
    """Search DBLP by title and return up to five candidate records as
    normalised dicts (same shape as query_crossref). Authors are reduced to
    last names, with DBLP's numeric homonym suffixes stripped."""
    q = urllib.parse.quote(title)
    data = json.loads(_get(f"https://dblp.org/search/publ/api?q={q}&format=json&h=5"))
    hits = data.get("result", {}).get("hits", {}).get("hit", []) or []
    out = []
    for h in hits:
        info = h.get("info", {})
        auth = info.get("authors", {}).get("author", [])
        if isinstance(auth, dict):
            auth = [auth]
        # DBLP disambiguates homonyms with numeric suffixes ("Mark Sandler 0002")
        names = [re.sub(r"\s+\d{4}$", "", a.get("text", "")).split()
                 for a in auth if a.get("text")]
        out.append({
            "title": info.get("title", ""),
            "authors": [n[-1] for n in names if n],
            "venue": info.get("venue", ""), "year": info.get("year", ""),
            "type": info.get("type", ""), "doi": info.get("doi", ""),
            "source": "dblp",
        })
    return out


def _openalex_record(w):
    """Normalise one OpenAlex work object into the candidate dict shape. The
    venue is the primary location's source; if that is a repository (arXiv,
    SSRN, ...) and another location is a journal/conference, that one is
    reported instead, so the published venue is visible."""
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    venue, vtype = src.get("display_name") or "", src.get("type") or ""
    if vtype == "repository":
        for other in w.get("locations") or []:
            s = other.get("source") or {}
            if s.get("display_name") and s.get("type") != "repository":
                venue, vtype = s["display_name"], s.get("type") or ""
                break
    # Keep the full display name: OpenAlex stores some East Asian names in
    # family-given order ("Bai Yu" for Yu Bai), so reducing to the last token
    # would compare the wrong part. The surname comparison in check_entry
    # accepts a cited surname that appears anywhere in the matched name.
    authors = []
    for a in w.get("authorships") or []:
        name = (a.get("author") or {}).get("display_name") or ""
        if name:
            authors.append(re.sub(r"[,.]", " ", name).strip())
    return {
        "title": w.get("title") or w.get("display_name") or "",
        "authors": authors, "venue": venue,
        "year": str(w.get("publication_year") or ""),
        "type": "preprint" if vtype == "repository" else (w.get("type") or ""),
        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
        "source": "openalex",
    }


def query_openalex(title):
    """Search OpenAlex by title (full-text relevance search over
    title/abstract) and return up to five candidates."""
    q = urllib.parse.quote(title)
    data = json.loads(_get(
        f"https://api.openalex.org/works?search={q}&per-page=5{OPENALEX_AUTH}"))
    _openalex_check(data)
    return [_openalex_record(w) for w in data.get("results", [])]


def _openalex_check(data):
    """OpenAlex signals throttling with an HTTP 200 whose body is an error
    object ({"error": "Rate limit exceeded", ...}). Raise so that the caller
    retries with backoff and, above all, never caches the empty answer as
    "no such paper"."""
    err = data.get("error") or data.get("message") if isinstance(data, dict) else None
    if err and "results" not in data and "id" not in data:
        raise RuntimeError(f"HTTP 429: openalex {str(err)[:60]}"
                           if "rate" in str(err).lower() else f"openalex: {str(err)[:80]}")


def query_openalex_doi(doi):
    """Resolve a DOI through OpenAlex. Returns a one-element list, or an
    empty list if the DOI is unknown (HTTP 404)."""
    try:
        data = json.loads(_get(
            f"https://api.openalex.org/works/https://doi.org/"
            f"{urllib.parse.quote(doi, safe='')}?{OPENALEX_AUTH.lstrip('&')}"))
    except urllib.error.HTTPError as ex:
        if ex.code == 404:
            return []
        raise
    _openalex_check(data)
    return [_openalex_record(data)] if data.get("id") else []


def query_s2(arxiv_id):
    """Semantic Scholar: resolves an arXiv id to its canonical (published)
    record even when the title changed between preprint and publication."""
    data = json.loads(_get(
        f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
        f"?fields=title,venue,year,authors,externalIds"))
    if str(data.get("code")) == "429" or "Too Many Requests" in str(data.get("message", "")):
        raise RuntimeError("HTTP 429: semanticscholar rate limit")
    if "title" not in data:
        return []
    return [{
        "title": data.get("title", ""),
        "authors": [(a.get("name", "")).split()[-1]
                    for a in data.get("authors", []) if a.get("name")],
        "venue": data.get("venue", ""), "year": str(data.get("year") or ""),
        "type": "record",
        "doi": (data.get("externalIds") or {}).get("DOI", ""),
        "source": "semanticscholar",
    }]


def query_arxiv(arxiv_id):
    """Fetch the arXiv record for a given identifier and return it as a
    one-element list of normalised dicts (venue fixed to "arXiv"), or an
    empty list if the id does not resolve to a real entry."""
    xml = _get(f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml)
    e = root.find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        return []
    title = re.sub(r"\s+", " ", e.find("a:title", ns).text or "").strip()
    if title.lower() == "error":
        return []
    return [{
        "title": title,
        "authors": [(a.find("a:name", ns).text or "").split()[-1]
                    for a in e.findall("a:author", ns)],
        "venue": "arXiv", "year": (e.find("a:published", ns).text or "")[:4],
        "type": "preprint", "doi": "", "source": "arxiv",
    }]


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def cached_query(cache, key, fn, arg, delay, log, backoff0=1.5):
    """Query with cache; retry with backoff on 429/503; never cache failures."""
    if cache.get(key) is not None:
        return cache[key], None
    backoff, last = max(delay, backoff0), ""
    for _ in range(4):
        try:
            res = fn(arg)
            cache[key] = res
            time.sleep(delay)
            return res, None
        except Exception as ex:
            last = str(ex)
            if "429" in last or "503" in last:
                log(f"    ~ {key.split(':', 1)[0]} rate limited, retrying in {backoff:.0f}s")
                time.sleep(backoff)
                backoff *= 2.5
            else:
                break
    log(f"    ! {key.split(':', 1)[0]} query failed: {last[:100]}")
    return None, last


def norm(s):
    """Lowercase ASCII skeleton of a string: accents are transliterated
    (Márquez -> marquez) rather than deleted, so a citation that lost its
    diacritics in PDF extraction still equals the database's spelling."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("ø", "o").replace("ß", "ss").replace("ı", "i").replace("ł", "l")
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def last_names(authors):
    """Reduce a list of author strings to normalised surnames for comparison.
    Braces/dots are stripped, all-uppercase initials are dropped, and stray
    "et"/"al" tokens left over from "et al." are removed."""
    out = []
    for a in authors:
        a = re.sub(r"[{}\\.]", "", a).strip()
        parts = [p for p in a.split() if len(p) > 1 or not p.isupper()]
        if parts:
            out.append(norm(parts[-1]))
    return [x for x in out if x and x not in {"al", "et"}]


def check_entry(entry, cache, delay, log):
    """Verify a single bibliography entry against the online databases and
    return a list of (severity, code, message) findings. Web/dataset and
    incomplete entries short-circuit early; otherwise the entry's title (and
    arXiv id, if any) is queried, the best match is chosen, and title,
    authors, year, and venue/preprint status are each compared. Returns a
    single ("OK", ...) finding when nothing is wrong."""
    findings = []  # (severity, code, message)
    title, venue = entry["title"], entry["venue"]

    if not title:
        if entry["url"] or WEB_HOST_RE.search(entry["raw"]):
            return [("INFO", "WEB-SOURCE", "no quoted title; looks like a web/"
                     "dataset source -- check the URL manually")]
        return [("INFO", "INCOMPLETE", "no title could be parsed; entry cannot "
                 "be verified (check completeness of the entry)")]
    doi = entry.get("doi", "")
    if WEB_HOST_RE.search(venue) or (not venue and entry["url"]
                                     and not entry["arxiv"] and not doi):
        return [("INFO", "WEB-SOURCE",
                 f"web/dataset source ('{(venue or entry['url'])[:60]}') -- "
                 f"not checked against publication databases")]

    candidates, errors, doi_hits = [], [], []
    for name, fn, arg in (("openalex-doi", query_openalex_doi, doi),
                          ("openalex", query_openalex, title),
                          ("crossref", query_crossref, title),
                          ("dblp", query_dblp, title),
                          ("arxiv", query_arxiv, entry["arxiv"])):
        if not arg:
            continue
        res, err = cached_query(cache, f"{name}:{arg}", fn, arg, delay, log)
        if err:
            errors.append(f"{name}: {err[:60]}")
        if res:
            candidates.extend(res)
            if name == "openalex-doi":
                doi_hits = res

    if not candidates:
        if errors:
            return [("INFO", "UNVERIFIED",
                     f"all database queries failed ({'; '.join(errors)[:120]})")]
        return [("WARN", "NOT-FOUND",
                 f"no database match for title '{title[:70]}'")]

    # A cited identifier is authoritative: a DOI resolved by OpenAlex, or an
    # arXiv id resolved by arXiv, is trusted over a fuzzy title search (which
    # mismatches papers too new to be indexed). A DOI that resolves to a
    # clearly different title is reported and the title search takes over.
    if doi_hits and sim(title, doi_hits[0]["title"]) < 0.55:
        findings.append(("ERROR", "DOI-MISMATCH",
                         f"cited DOI {doi} resolves to "
                         f"'{doi_hits[0]['title'][:70]}' "
                         f"({doi_hits[0]['venue'][:40]} {doi_hits[0]['year']}) "
                         f"[openalex], not to the cited title"))
        doi_hits = []
    arxiv_hit = next((c for c in candidates if c["source"] == "arxiv"), None)
    if doi_hits:
        best = doi_hits[0]
    elif entry["arxiv"] and arxiv_hit:
        best = arxiv_hit
    else:
        best = max(candidates, key=lambda c: sim(title, c["title"]))
    best_sim = sim(title, best["title"])
    # A mediocre title match with no author in common is a different work
    # (typically a book or report the databases do not index), not a drift.
    cited0 = last_names(entry["authors"])
    found0 = [norm(x) for x in best["authors"]]
    no_author_overlap = cited0 and found0 and not any(
        c == f or c in f or f in c for c in cited0 for f in found0)
    if best_sim < 0.55 or (best_sim < 0.75 and no_author_overlap):
        # Absence of a database match is not evidence of fabrication -- recent
        # papers routinely lag the indexes. Flag as a WARN to check, not an
        # ERROR. (A cited id that resolves to a genuinely different title still
        # surfaces here, via the low similarity against that resolved record.)
        return [("WARN", "NOT-FOUND",
                 f"no plausible match for title '{title[:70]}' "
                 f"(closest: '{best['title'][:70]}' [{best['source']}], "
                 f"similarity {best_sim:.2f})")]
    if best_sim < 0.85:
        findings.append(("WARN", "TITLE-DRIFT",
                         f"cited title '{title[:60]}' vs matched "
                         f"'{best['title'][:60]}' [{best['source']}] "
                         f"(similarity {best_sim:.2f})"))

    # authors: compare last names; citation may truncate with 'et al.'
    cited, found = last_names(entry["authors"]), [norm(x) for x in best["authors"]]
    if cited and found:
        def name_eq(c, f):
            # Compare with spaces removed as well ("acevedosandoval" vs
            # "otilio arturo acevedo sandoval").
            cs, fs = c.replace(" ", ""), f.replace(" ", "")
            return c == f or c in f or f in c or cs in fs or fs in cs \
                or sim(c, f) > 0.85
        overlap = sum(1 for c in cited if any(name_eq(c, f) for f in found))
        first_ok = not found or not cited or name_eq(cited[0], found[0]) \
            or any(name_eq(cited[0], f) for f in found)
        # Flag a mismatch only if fewer than half the cited surnames appear in
        # the record, or the (usually reliable) first author is absent -- this
        # tolerates "et al." truncation and author-order noise.
        if overlap < max(1, len(cited) // 2) or not first_ok:
            findings.append(("ERROR", "AUTHOR-MISMATCH",
                             f"cited authors {cited[:6]} vs matched "
                             f"{found[:6]} [{best['source']}]"))

    # year
    if entry["year"] and best["year"]:
        try:
            if abs(int(entry["year"]) - int(best["year"])) > 1:
                findings.append(("WARN", "YEAR-MISMATCH",
                                 f"cited year {entry['year']} vs matched "
                                 f"{best['year']} [{best['source']}]"))
        except ValueError:
            pass

    # venue / preprint status
    cited_preprint = bool(PREPRINT_RE.search(venue)) or (not venue and entry["arxiv"])
    if not venue and not entry["arxiv"] and not entry["url"]:
        findings.append(("WARN", "INCOMPLETE",
                         "no publication venue, arXiv id, or URL given -- the "
                         "entry cannot be located from the citation alone"))
    published = [c for c in candidates
                 if sim(title, c["title"]) > 0.85
                 and c["venue"] and not PREPRINT_RE.search(c["venue"])
                 and c.get("type") not in {"preprint", "posted-content", "Informal and Other Publications"}]
    if cited_preprint and entry["arxiv"] and not published:
        # title may have changed at publication: resolve via Semantic Scholar
        res, err = cached_query(cache, f"s2:{entry['arxiv']}", query_s2,
                                entry["arxiv"], delay, log, backoff0=6.0)
        if res is None and err:
            findings.append(("INFO", "PREPRINT-UNCHECKED",
                             "could not determine whether a published version "
                             f"of arXiv:{entry['arxiv']} exists "
                             f"({err[:60]}) -- re-run to retry"))
        published = [c for c in res or []
                     if c["venue"] and not PREPRINT_RE.search(c["venue"])]
    if cited_preprint and entry["arxiv"] and not published:
        # Semantic Scholar keeps some renamed papers as separate records
        # (e.g. the Krum paper, arXiv:1703.02757 vs. its NIPS 2017 version):
        # search DBLP for first author + distinctive title words and accept a
        # real-venue hit whose title contains (nearly) all cited title words.
        first = (last_names(entry["authors"]) or [""])[0]
        tokens = title_tokens(title)
        if first and tokens:
            q = " ".join([first] + tokens[:3])
            res, _ = cached_query(cache, f"dblp:{q}", query_dblp, q, delay, log)
            cited_tok = set(tokens)
            for c in res or []:
                if not c["venue"] or PREPRINT_RE.search(c["venue"]) \
                        or c.get("type") in {"preprint", "posted-content",
                                             "Informal and Other Publications"}:
                    continue
                if first not in {norm(x) for x in c["authors"]}:
                    continue
                if len(cited_tok - set(title_tokens(c["title"]))) \
                        <= len(cited_tok) // 3:
                    published = [c]
                    break
    if cited_preprint and published:
        p = max(published, key=lambda c: sim(title, c["title"]))
        findings.append(("WARN", "PREPRINT",
                         f"cited as preprint but published version exists: "
                         f"'{p['title'][:60]}', {p['venue']} {p['year']} "
                         f"[{p['source']}"
                         + (f", doi:{p['doi']}]" if p['doi'] else "]")))
    elif venue and not cited_preprint and best["venue"] \
            and not PREPRINT_RE.search(best["venue"]) \
            and best.get("type") != "preprint":
        # (A record known to the databases only as its arXiv version says
        # nothing about the cited proceedings venue, so it is not compared.)
        v_sim = sim(venue, best["venue"])
        tokens_c = set(norm(venue).split()) - {"proceedings", "of", "the", "on",
                                               "in", "conference", "international"}
        tokens_f = set(norm(best["venue"]).split())
        token_overlap = bool(tokens_c & tokens_f)
        abbrev = "".join(w[0] for w in norm(best["venue"]).split() if w)
        if v_sim < 0.5 and not token_overlap and norm(venue).replace(" ", "") \
                not in abbrev and abbrev not in norm(venue).replace(" ", ""):
            findings.append(("WARN", "VENUE-MISMATCH",
                             f"cited venue '{venue[:50]}' vs matched "
                             f"'{best['venue'][:50]}' [{best['source']}]"))

    if not findings:
        findings.append(("OK", "OK", f"verified against {best['source']} "
                        f"('{best['title'][:60]}', {best['venue'][:40]} "
                        f"{best['year']})"))
    elif errors:
        # A finding reached while a database was unreachable may be an
        # artefact of the fallback match; say so and invite a re-run (failed
        # queries are never cached, so the re-run retries them).
        findings.append(("INFO", "PARTIAL-SOURCES",
                         f"some databases failed for this entry "
                         f"({'; '.join(errors)[:100]}) -- re-run to retry"))
    return findings


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify bibliography entries "
                                 "against OpenAlex, Crossref, DBLP, and arXiv.")
    ap.add_argument("path", help=".bib file or .pdf with an IEEE-numbered or "
                                 "author-year reference list")
    ap.add_argument("--parse-only", action="store_true",
                    help="print the extracted entries and exit (no queries)")
    ap.add_argument("--max", type=int, default=0, help="check only first N entries")
    ap.add_argument("--delay", type=float, default=0.7,
                    help="seconds between API queries (default 0.7)")
    ap.add_argument("--cache", default="",
                    help="cache file (default: .bibcheck_cache.json beside input)")
    ap.add_argument("--only", default="",
                    help="comma-separated entry numbers/keys to check, e.g. 2,5,42")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="only print findings, no per-entry progress")
    args = ap.parse_args(argv)

    path = Path(args.path)
    if not path.is_file():
        sys.exit(f"error: no such file: {path}")
    entries = parse_bib(path) if path.suffix.lower() == ".bib" \
        else parse_pdf_references(path)
    if args.only:
        wanted = {w.strip() for w in args.only.split(",")}
        entries = [e for e in entries
                   if e["id"].strip("[]") in wanted or e["id"] in wanted]
    if args.max:
        entries = entries[:args.max]
    if args.parse_only:
        for e in entries:
            print(f"{e['id']} year={e['year'] or '?'} doi={e.get('doi') or '-'} "
                  f"arxiv={e['arxiv'] or '-'}\n    title:   {e['title']}"
                  f"\n    authors: {', '.join(e['authors'])}\n    venue:   {e['venue']}")
        print(f"\n{len(entries)} entr(ies) extracted")
        return 0

    cache_path = Path(args.cache) if args.cache else \
        path.with_name(".bibcheck_cache.json")
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    log = (lambda *a: None) if args.quiet else print
    counts = {"ERROR": 0, "WARN": 0, "INFO": 0, "OK": 0}
    try:
        for e in entries:
            log(f"{e['id']} {e['raw'][:90]}")
            for sev, code, msg in check_entry(e, cache, args.delay, log):
                counts[sev] += 1
                if sev == "OK":
                    log(f"    ok: {msg}")
                else:
                    print(f"{path}:{e['id']}: [{sev}] {code}: {msg}")
    finally:
        cache_path.write_text(json.dumps(cache))

    print(f"\nchecked {len(entries)} entr(ies): {counts['OK']} verified, "
          f"{counts['ERROR']} error(s), {counts['WARN']} warning(s), "
          f"{counts['INFO']} unverifiable/skipped   (cache: {cache_path})")
    return 1 if counts["ERROR"] or counts["WARN"] else 0


if __name__ == "__main__":
    sys.exit(main())
