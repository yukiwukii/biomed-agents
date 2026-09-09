#!/usr/bin/env python3
"""Propose candidate scientific hypotheses for each capsule.

Given a capsule (the raw input data from a single biology study — count
matrices, sample metadata, variant calls, single-cell data, sequences, …), this
script asks an LLM to infer the experimental design and propose a few *specific,
falsifiable* hypotheses that could be tested using only the data in that
capsule. Output is hypothesis text only (one JSON object per capsule).

The generated hypotheses are meant to look like the ones in the
``EdisonScientific/bixbench_hypothesis`` dataset: a single declarative sentence
naming concrete biological entities (genes, mutations, strains, cell types,
processes) with a clear direction/relationship, testable by a standard
bioinformatics analysis on the provided data. See ``HYPGEN_INSTRUCTIONS`` below
for the exact prompt — that prompt was reverse-engineered by comparing capsule
contents against the dataset's hypothesis/protocol/rubric fields.

Two ingestion modes (``--mode``) share the same generation + prompt so their
outputs are directly comparable:

  * ``preview``: a cheap, deterministic, single-shot pass. A text summary of the
    capsule (file tree + header/column previews of csv/xlsx, etc.) is built
    locally and handed to the LLM in one call.
  * ``agent``: the model actively loads/inspects the data in a live Jupyter kernel
    (via the repo's ``Interpreter``) before proposing — it runs up to
    ``--explore-steps`` code cells, and that exploration transcript becomes the
    context. Slower/costlier but grounded in the real data.

Trajectories (explore cells + generation prompt + hypotheses) are saved per capsule
under ``proposer/<mode>/`` by default — i.e. ``proposer/preview/`` or
``proposer/agent/`` — so the two modes stay separated for comparison
(``--save-traj DIR`` to change the base, ``--no-save-traj`` to disable). View them
with ``proposer/inspect_proposer_traj.py``.

IMPORTANT — leakage control: capsules often ship the *original analysis
notebook* (``.ipynb``) and R result objects (``.rds``), which reveal the paper's
actual hypothesis and answer. Those are excluded from context by default (see
``EXCLUDE_PATTERNS``) so the model must propose from the raw data, not paraphrase
the known result. Edit ``EXCLUDE_PATTERNS`` to change what is hidden.

Source paper: when a capsule has a downloaded paper under ``<capsule>/paper/``
(see ``scripts/`` fetch tooling), its text is extracted and shown to the model in
a dedicated ``SOURCE PAPER`` block — NOT as a data-file preview, which is why
``paper*`` sits in ``EXCLUDE_PATTERNS``. The prompt then requires the proposed
hypotheses to be *distinct from* what the paper already established: the paper is
related work to build past, not an answer to paraphrase. Pass ``--no-paper`` to
withhold it. Extraction prefers Europe PMC JATS full text (title/abstract/results)
and falls back to parsing the article PDF, which needs the optional ``pypdf``
package; without it, PDF-only capsules simply get no paper block.

Optionally (``--rubrics``) the script also generates a grading rubric for each
proposed hypothesis, in the style of the dataset's ``rubric``/``max_points``
fields (several 1-point procedure criteria plus one higher-value overall-
correctness criterion). Each rubric is a *separate*, focused LLM call per
hypothesis — the ground-truth accept/reject outcome is unknown here, so the final
criterion credits a sound conclusion in either direction rather than hard-coding a
result. Rubrics land in the output/trajectory as a ``rubrics`` list aligned with
``hypotheses`` (``null`` when the toggle is off). Rubrics may ingest the capsule in
a *different* ``--mode`` than the hypotheses via ``--rubric-mode`` — e.g. cheap
``preview`` hypotheses but an ``agent``-explored rubric grounded in the real data.

Usage:
    .venv/bin/python proposer/generate_hypotheses.py \
        --capsules-dir capsules/ --out hypotheses_generated.json --n 3

    # also generate a grading rubric per hypothesis:
    .venv/bin/python proposer/generate_hypotheses.py --n 3 --rubrics

    # preview hypotheses, but let the rubric explore the data live (agent):
    .venv/bin/python proposer/generate_hypotheses.py --n 3 --rubrics --mode preview --rubric-mode agent --limit 15

    # limit / target specific capsules while iterating:
    .venv/bin/python proposer/generate_hypotheses.py --limit 5
    .venv/bin/python proposer/generate_hypotheses.py --only 0923d260

    # print the exact prompt(s) that would be sent for one capsule, then exit:
    .venv/bin/python proposer/generate_hypotheses.py --only 0923d260 --dry-run --rubrics
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import io
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
from lmi import LiteLLMModel
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent

# ── configuration knobs ──────────────────────────────────────────────────────

# Files hidden from the model's context. Notebooks and R result objects leak the
# paper's real hypothesis/answer; checksums and macOS cruft are noise. Add e.g.
# "*.tiff", "*.png" here to also hide figure outputs.
EXCLUDE_PATTERNS: tuple[str, ...] = (
    "*.ipynb",  # original analysis notebook — states the real hypothesis + method
    "*.rds",  # R result objects — often the computed answer
    # The downloaded source paper is excluded from the *data file* listing on purpose: it is
    # surfaced separately, as extracted text, via extract_paper_text() + the SOURCE PAPER block
    # in build_prompt(). Raw PDFs/JATS XML previewed as data files are unreadable tag soup and
    # would eat the data budget; this way the model still sees the paper, just usefully.
    "paper*",
    "*.checksum",
    "__MACOSX*",
    "._*",
    ".*",
)

# Per-file text preview budget and overall context budget (characters).
MAX_FILE_PREVIEW_CHARS = 1500
MAX_CONTEXT_CHARS = 14000
MAX_PREVIEW_READ_BYTES = 12 * 1024 * 1024

# Budget for the extracted SOURCE PAPER text (characters). Separate from MAX_CONTEXT_CHARS so
# the paper never crowds out the data-file previews (and vice versa).
MAX_PAPER_CHARS = 6000

# Default rubric/generation model (litellm name resolved via lmi + .env keys).
DEFAULT_MODEL = "claude-sonnet-4-6"

# Agent (--mode agent) exploration budget: max code cells the explorer may run,
# per-cell execution timeout (s), and the char cap on each cell's captured output.
DEFAULT_EXPLORE_STEPS = 6

DEFAULT_EXEC_TIMEOUT = 120
MAX_CELL_OUTPUT_CHARS = 1500

DEFAULT_DATASET = "EdisonScientific/bixbench_hypothesis"


# ── .env loading (repo has no dotenv dependency; parse it ourselves) ──────────


def load_env(path: Path = ROOT / ".env") -> None:
    """Populate os.environ from a KEY=VALUE .env file (does not overwrite)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def load_expert_hypotheses(dataset_name: str) -> dict[str, str]:
    """Map capsule id (UUID) → the dataset's ground-truth ``hypothesis``.

    Returns an empty map (with a warning) if the dataset can't be loaded, so the
    generator still runs — records just get ``expert_hypothesis: null``.
    """
    try:
        ds = load_dataset(dataset_name, split="train")
    except Exception as e:  # noqa: BLE001
        print(f"warning: could not load expert hypotheses from {dataset_name}: {type(e).__name__}: {e}")
        return {}
    return {str(row["id"]): row["hypothesis"] for row in ds}


# ── capsule enumeration ──────────────────────────────────────────────────────


@dataclass
class Capsule:
    """A capsule on disk — either an unpacked folder or a .zip archive."""

    capsule_id: str  # the UUID (without the CapsuleData-/CapsuleFolder- prefix)
    path: Path
    is_zip: bool


def _capsule_id(name: str) -> str:
    """Strip CapsuleData-/CapsuleFolder- prefix and .zip suffix down to the UUID."""
    stem = name[:-4] if name.endswith(".zip") else name
    for prefix in ("CapsuleData-", "CapsuleFolder-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    return stem


def discover_capsules(capsules_dir: Path) -> list[Capsule]:
    """Enumerate the raw-data capsules only.

    The directory mixes two entry kinds per study: ``CapsuleData-<uuid>`` /
    ``CapsuleFolder-<uuid>.zip`` (the raw inputs we want) and
    ``CapsuleNotebook-<uuid>`` (the original analysis notebook — pure answer
    leakage). Only the data capsules are returned; notebook folders are skipped.
    Note: the dataset's ``input_data_path`` references ``CapsuleFolder-*.zip``,
    but those zips are typically not downloaded, so the benchmark (and this
    script) fall back to the ``CapsuleData-*`` folders that are present.
    """
    out: list[Capsule] = []
    for entry in sorted(capsules_dir.iterdir()):
        if entry.name.startswith(".") or entry.name.startswith("CapsuleNotebook-"):
            continue
        if entry.is_dir():
            out.append(Capsule(_capsule_id(entry.name), entry, is_zip=False))
        elif entry.suffix == ".zip":
            out.append(Capsule(_capsule_id(entry.name), entry, is_zip=True))
    return out


def _excluded(rel_name: str) -> bool:
    base = Path(rel_name).name
    return any(fnmatch.fnmatch(base, pat) or fnmatch.fnmatch(rel_name, pat) for pat in EXCLUDE_PATTERNS)


# ── per-file preview ─────────────────────────────────────────────────────────


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n / 1:.0f}{unit}" if False else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def _preview_text(data: bytes, max_lines: int = 8) -> str:
    """Head of a text/csv/tsv file: first `max_lines` lines, char-capped."""
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return "(unreadable text)"
    lines = text.splitlines()[:max_lines]
    joined = "\n".join(lines)
    return joined[:MAX_FILE_PREVIEW_CHARS]


def _preview_xlsx(data: bytes) -> str:
    """Sheet names + header row (and first data row) of each sheet."""
    try:
        import openpyxl  # noqa: PLC0415  (lazy: heavy import)

        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as e:  # noqa: BLE001
        return f"(xlsx not previewable: {type(e).__name__})"
    parts: list[str] = []
    for name in wb.sheetnames[:6]:
        ws = wb[name]
        rows = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            rows.append([("" if c is None else str(c))[:40] for c in row][:40])
            if i >= 1:  # header + first data row is enough to reveal columns
                break
        dims = f"{ws.max_row}x{ws.max_column}" if ws.max_row else "?"
        header = " | ".join(rows[0]) if rows else ""
        sample = " | ".join(rows[1]) if len(rows) > 1 else ""
        parts.append(f"  sheet '{name}' ({dims})\n    columns: {header}" + (f"\n    e.g.:    {sample}" if sample else ""))
    wb.close()
    return "\n".join(parts)[:MAX_FILE_PREVIEW_CHARS]


def _preview_sequences(data: bytes, kind: str) -> str:
    """First header of a FASTA/FASTQ-like file (avoid scanning huge files)."""
    head = data[:4096].decode("utf-8", errors="replace").splitlines()
    first = next((ln for ln in head if ln.startswith((">", "@"))), "")
    return f"({kind} sequence file; first record: {first[:120]})"


def preview_file(rel_name: str, size: int, reader) -> str:
    """Return a one-file preview block. `reader()` lazily returns the bytes."""
    suffix = Path(rel_name).suffix.lower()
    header = f"- {rel_name}  [{_human_size(size)}]"

    # Types we cannot meaningfully preview as text — name + note only.
    note = {
        ".h5ad": "(AnnData single-cell matrix, binary — not previewed)",
        ".h5": "(HDF5 binary — not previewed)",
        ".rds": "(R data object — not previewed)",
        ".bam": "(aligned reads, binary — not previewed)",
        ".gz": "(gzip archive — not previewed)",
        ".zip": "(nested zip — not previewed)",
        ".png": "(image — not previewed)",
        ".tiff": "(image — not previewed)",
        ".tif": "(image — not previewed)",
        ".pdf": "(pdf — not previewed)",
    }.get(suffix)
    if note:
        return f"{header} {note}"

    if size > MAX_PREVIEW_READ_BYTES:
        return f"{header} (too large to preview)"

    try:
        data = reader()
    except Exception as e:  # noqa: BLE001
        return f"{header} (unreadable: {type(e).__name__})"

    if suffix in (".xlsx", ".xls"):
        return f"{header}\n{_preview_xlsx(data)}"
    if suffix in (".faa", ".fna", ".fa", ".fasta"):
        return f"{header} {_preview_sequences(data, 'FASTA')}"
    if suffix in (".fastq", ".fq"):
        return f"{header} {_preview_sequences(data, 'FASTQ')}"
    if suffix in (".csv", ".tsv", ".txt", ".json", ".gmt", ".vcf", ".treefile", ".mafft", ".clipkit", ""):
        return f"{header}\n{_preview_text(data)}"
    # Unknown extension: try a short text head; harmless if binary.
    return f"{header}\n{_preview_text(data, max_lines=4)}"


# ── source paper text (the "already published" channel) ──────────────────────

# Body sections worth quoting: what the paper concluded is what a new hypothesis must not
# duplicate. Methods/acknowledgements add length without informing distinctness.
PAPER_BODY_SECTIONS = ("result", "discussion", "conclusion")


def _paper_text_from_jats(path: Path) -> str:
    """Title + abstract + results/discussion prose from a Europe PMC JATS full text."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415  (lazy: only needed when a paper exists)

    root = ET.parse(path).getroot()

    def flat(el) -> str:
        return " ".join("".join(el.itertext()).split()) if el is not None else ""

    parts = []
    if title := flat(root.find(".//article-title")):
        parts.append(f"TITLE: {title}")
    if abstract := flat(root.find(".//abstract")):
        parts.append(f"ABSTRACT: {abstract}")
    for sec in root.findall(".//body/sec"):
        head = flat(sec.find("title"))
        if head and any(k in head.lower() for k in PAPER_BODY_SECTIONS):
            parts.append(f"{head.upper()}: {flat(sec)}")
    return "\n\n".join(parts)


# Journal front matter that survives PDF extraction: funding statements, licences,
# correspondence, submission dates, running heads. It says nothing about the science but
# eats the MAX_PAPER_CHARS budget — on some articles it dominated the extracted window.
PDF_BOILERPLATE_RE = re.compile(
    r"^\s*(?:"
    r"copyright\b|©|\(c\)\s*\d{4}|all rights reserved|"
    r"funding\b|this work was (?:supported|funded)|grant (?:no|number)\b|"
    r"competing interests?\b|conflicts? of interest|declaration of interests?|"
    r"data availability|"
    r"acknowledge?ments?\b|"
    r"\*?\s*correspond(?:ing|ence)\b|e-?mail(?: address)?\b|"
    r"(?:received|accepted|published|submitted|revised)\s*[:\s]\d|"
    r"citation\s*[:]|editor\s*[:]|"
    r"this is an open[- ]access article|creative commons|licen[cs]e\b|"
    r"issn\b|doi\s*[:]|https?://|www\.|"
    r"\d+\s*$"  # bare page numbers
    r")",
    re.IGNORECASE,
)

# Phrases that only ever appear in journal front matter, never in scientific prose. Matched
# against a whitespace-stripped copy of the line, because PDF extraction routinely breaks words
# apart ("Data Availabilit y Statement") and splices sidebars into body lines, which defeats both
# ordinary substring search and the line-anchored patterns above.
PDF_BOILERPLATE_PHRASES = (
    "creativecommons",
    "openaccessarticle",
    "distributedundertheterms",
    "dataavailability",
    "competinginterests",
    "conflictofinterest",
    "allrightsreserved",
    "reproductioninanymedium",
    "sourcearecredited",
    "permitsunrestricted",
    "correspondingauthor",
    "peerreview",
    # Preprint-server watermarks, stamped on every page of bioRxiv/medRxiv PDFs.
    "copyrightholder",
    "biorxivpreprint",
    "medrxivpreprint",
    "thisversionposted",
    "editorialhistory",
)

# Where the scientific content starts and stops.
PDF_ABSTRACT_RE = re.compile(r"^\s*(?:abstract|summary)\b", re.IGNORECASE)
PDF_REFERENCES_RE = re.compile(r"^\s*(?:references|bibliography|literature cited)\s*$", re.IGNORECASE)


def _clean_pdf_text(raw: str) -> str:
    """Strip journal boilerplate from raw PDF text, keeping title + abstract onward.

    Works line-wise (pypdf preserves line breaks) so whole boilerplate lines can be dropped,
    then flattens. Everything from a ``References`` heading on is discarded — citation lists
    are pure budget burn.
    """
    for lig, plain in (("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"), ("ﬃ", "ffi"), ("ﬄ", "ffl")):
        raw = raw.replace(lig, plain)
    raw = re.sub(r"/C\d+", "", raw)  # pdf escape artifacts, e.g. /C211 for ©

    def is_boilerplate(ln: str) -> bool:
        if PDF_BOILERPLATE_RE.match(ln):
            return True
        squashed = re.sub(r"\s+", "", ln).lower()
        return any(p in squashed for p in PDF_BOILERPLATE_PHRASES)

    lines = [ln.strip() for ln in raw.splitlines()]
    abs_i = next((i for i, ln in enumerate(lines) if PDF_ABSTRACT_RE.match(ln)), None)
    ref_i = next((i for i, ln in enumerate(lines) if PDF_REFERENCES_RE.match(ln)), len(lines))

    # Title block: the lines above the abstract, boilerplate removed, kept short.
    title = ""
    if abs_i is not None:
        head = [ln for ln in lines[:abs_i] if ln and not is_boilerplate(ln)]
        title = " ".join(" ".join(head).split())[:300]

    body_start = abs_i if abs_i is not None else 0
    body = [ln for ln in lines[body_start:ref_i] if ln and not is_boilerplate(ln)]
    text = " ".join(" ".join(body).split())
    return f"TITLE: {title}\n\n{text}" if title else text


def _paper_text_from_pdf(path: Path, max_pages: int = 12) -> str:
    """Article PDF text with journal boilerplate stripped — fallback when there is no JATS.

    Reads more pages than the character budget needs: front matter and references are
    discarded afterwards, so the surviving text should be science rather than the first
    N pages verbatim.
    """
    try:
        import pypdf  # noqa: PLC0415  (lazy + optional: not a repo dependency)
    except ImportError:
        return ""
    try:
        reader = pypdf.PdfReader(str(path))
    except Exception:  # noqa: BLE001 — a corrupt PDF must not abort generation
        return ""
    out = []
    for page in reader.pages[:max_pages]:
        try:
            out.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            continue
    return _clean_pdf_text("\n".join(out))


def extract_paper_text(cap: Capsule) -> str:
    """Return readable text of the capsule's source paper, or "" if none is present.

    Prefers the Europe PMC JATS full text (clean title/abstract/results) and falls back to
    parsing the article PDF. Only the *article* is used — supplementary files are ignored,
    since they are figures/tables that add little beyond the abstract and conclusions.
    """
    pdir = cap.path / "paper"
    if not pdir.is_dir():
        return ""

    text = ""
    for xml_path in sorted(pdir.glob("*.fulltext.xml")):
        try:
            text = _paper_text_from_jats(xml_path)
        except Exception:  # noqa: BLE001 — malformed XML falls through to the PDF
            text = ""
        if text:
            break
    if not text:
        # Skip PDFs inside *.supplementary/ — we want the article, not its appendix.
        for pdf_path in sorted(pdir.glob("*.pdf")):
            text = _paper_text_from_pdf(pdf_path)
            if text:
                break

    if len(text) > MAX_PAPER_CHARS:
        text = text[:MAX_PAPER_CHARS] + " … (paper text truncated)"
    return text


# ── context builders (the ingestion toggle) ──────────────────────────────────


@dataclass
class CapsuleContext:
    capsule_id: str
    text: str
    n_files: int = 0
    warnings: list[str] = field(default_factory=list)
    # agent mode only: the exploration trajectory as (code, output, reasoning) cells
    # (reasoning is the model's thinking for that step, None if thinking was off), plus
    # the file listing the explorer started from. Empty for preview mode.
    steps: list[tuple[str, str, str | None]] = field(default_factory=list)
    seed_listing: str = ""
    # Extracted text of the capsule's source paper (see extract_paper_text). Empty when the
    # capsule has no downloaded paper or when --no-paper is passed.
    paper_text: str = ""


def gather_context_preview(cap: Capsule) -> CapsuleContext:
    """MODE=preview — build a static text summary of the capsule's data files."""
    blocks: list[str] = []
    n_files = 0

    if cap.is_zip:
        with zipfile.ZipFile(cap.path) as zf:
            for info in zf.infolist():
                if info.is_dir() or _excluded(info.filename):
                    continue
                n_files += 1
                # Strip the top-level CapsuleData-<id>/ directory for readability.
                rel = info.filename.split("/", 1)[-1] if "/" in info.filename else info.filename
                blocks.append(preview_file(rel, info.file_size, lambda i=info: zf.read(i)))
    else:
        for fp in sorted(cap.path.rglob("*")):
            if not fp.is_file() or _excluded(str(fp.relative_to(cap.path))):
                continue
            n_files += 1
            rel = str(fp.relative_to(cap.path))
            blocks.append(preview_file(rel, fp.stat().st_size, lambda p=fp: p.read_bytes()))

    text = "\n".join(blocks)
    warnings: list[str] = []
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[:MAX_CONTEXT_CHARS] + "\n… (context truncated)"
        warnings.append("context truncated")
    if n_files == 0:
        warnings.append("no non-excluded files — capsule may contain only notebooks/results")
    return CapsuleContext(cap.capsule_id, text, n_files, warnings)


def materialize_capsule(cap: Capsule, dest: Path, include_paper: bool = False) -> int:
    """Copy/extract a capsule's NON-excluded files into `dest`. Returns file count.

    Applies the same ``EXCLUDE_PATTERNS`` as preview mode, so the agent explores
    the same set of files the model would otherwise see summarized — notebooks
    and .rds results stay withheld.

    ``include_paper`` re-admits ``<capsule>/paper/`` despite its ``EXCLUDE_PATTERNS``
    entry, so the exploring agent can open the article and its supplementary files.
    The exclusion exists to keep the paper out of the *preview text* (raw JATS/PDF
    bytes are unreadable there); it is not meant to hide the paper from the agent.
    """
    n = 0
    if include_paper and not cap.is_zip:
        pdir = cap.path / "paper"
        if pdir.is_dir():
            # Deliberately NOT counted in the return value: `n` gates the "no usable input
            # files" skip, and a capsule holding only results plus a paper still has no data.
            shutil.copytree(pdir, dest / "paper", dirs_exist_ok=True)
    if cap.is_zip:
        with zipfile.ZipFile(cap.path) as zf:
            for info in zf.infolist():
                if info.is_dir() or _excluded(info.filename):
                    continue
                rel = info.filename.split("/", 1)[-1] if "/" in info.filename else info.filename
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    else:
        for fp in sorted(cap.path.rglob("*")):
            if not fp.is_file() or _excluded(str(fp.relative_to(cap.path))):
                continue
            out = dest / fp.relative_to(cap.path)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fp, out)
            n += 1
    return n


# The explorer's per-step instruction: emit ONE code cell to inspect the data, or
# stop. It is a data-understanding pass only — it must NOT propose hypotheses (that
# happens later via the shared HYPGEN_INSTRUCTIONS, so both modes are comparable).
EXPLORE_INSTRUCTIONS = """\
You are a bioinformatician exploring the raw data files of ONE study in a live Python kernel.
The working directory contains the files listed below. Your ONLY goal this phase is to UNDERSTAND
the data well enough that someone could later propose hypotheses: the modality, what was measured,
the groups/conditions/genotypes/covariates, sample sizes, and key variable names/values.

Each turn, respond with EXACTLY ONE fenced Python code block that prints what you want to inspect
(e.g. read a table's shape and columns, value_counts of a metadata column, unique conditions, a few
gene/feature names). Keep outputs small — print summaries, not whole files. Do not make plots.
Build on previous outputs. When you have seen enough to characterize the experiment, reply with the
single word DONE and no code block.

Do NOT propose or state any hypotheses. Explore only."""

# Appended to the explore prompt when the capsule's source paper was materialized into the
# workdir (see materialize_capsule(include_paper=True)). The agent is told it may read the
# paper, because knowing what has already been established is what lets the generation step
# propose something new — but reading it must not replace looking at the actual data.
PAPER_EXPLORE_NOTE = """\

The published paper for this study is available in ./paper/ ({files}). You MAY read it to
understand the experimental design and what the authors already established — the JATS
full text (.fulltext.xml) is plain text, and PDFs can be read with `pypdf` if installed.
Do not spend more than one or two cells on it: the point of this phase is still to
characterize the DATA, and hypotheses proposed later must go beyond what the paper reports."""


def _extract_code(text: str) -> str | None:
    """Pull the first fenced code block out of a model reply, if present."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _build_explore_prompt(listing: str, transcript: list[tuple[str, str]], step: int, max_steps: int,
                          paper_note: str = "") -> str:
    parts = [EXPLORE_INSTRUCTIONS + paper_note, f"\n(You have run {step}/{max_steps} cells.)",
             "\n=== FILES IN WORKDIR ===", listing]
    if transcript:
        parts.append("\n=== YOUR EXPLORATION SO FAR ===")
        for i, (code, out) in enumerate(transcript, 1):
            parts.append(f"[cell {i}]\n{code}\n[output]\n{out}")
    parts.append("\nReply with the next code cell, or DONE.")
    return "\n".join(parts)


async def gather_context_agent(
    cap: Capsule,
    model: LiteLLMModel,
    max_steps: int = DEFAULT_EXPLORE_STEPS,
    exec_timeout: int = DEFAULT_EXEC_TIMEOUT,
    include_paper: bool = True,
) -> CapsuleContext:
    """MODE=agent — let the model actively inspect the capsule via a live kernel.

    Materializes the capsule's (non-excluded) files into a temp workspace, starts
    a Python Jupyter kernel there, and runs an explore loop where the model emits
    code cells and sees their outputs. The resulting transcript becomes the
    context string fed to the SAME ``build_prompt`` + generation as preview mode.

    When ``include_paper`` is set (the default) and the capsule has a downloaded paper,
    ``paper/`` is copied into the workdir too, so the agent can open the article and its
    supplementary files during exploration. ``--no-paper`` turns this off, which is what
    makes the two arms of the A/B test differ.
    """
    # Lazy import: pulls in the heavy env/ray stack only when agent mode runs.
    from hypotest.env import utils as env_utils
    from hypotest.env.interpreter import Interpreter

    # Reuse preview's file listing as the seed context the explorer starts from.
    listing = gather_context_preview(cap).text

    workdir = Path(tempfile.mkdtemp(prefix=f"hypgen-{cap.capsule_id[:8]}-"))
    warnings: list[str] = []
    try:
        n_files = materialize_capsule(cap, workdir, include_paper=include_paper)
        if n_files == 0:
            return CapsuleContext(cap.capsule_id, "", 0, ["no usable input files"])

        # Tell the explorer the paper is there — it is absent from `listing` (paper* is in
        # EXCLUDE_PATTERNS to keep raw JATS/PDF bytes out of the previews), so without this
        # note the agent would have no way to know the directory exists.
        paper_note = ""
        paper_dir = workdir / "paper"
        if paper_dir.is_dir():
            names = sorted(p.name for p in paper_dir.iterdir() if p.name != "paper_metadata.json")
            if names:
                paper_note = PAPER_EXPLORE_NOTE.format(files=", ".join(names[:6]))

        interp = Interpreter(work_dir=workdir, language=env_utils.NBLanguage.PYTHON, execution_timeout=exec_timeout)
        await interp.start()
        transcript: list[tuple[str, str, str | None]] = []
        try:
            for step in range(max_steps):
                resp = await model.call_single(
                    _build_explore_prompt(listing, [(c, o) for c, o, _ in transcript], step, max_steps, paper_note),
                    timeout=180,
                )
                reply = resp.text or ""
                code = _extract_code(reply)
                if not code:  # model signalled DONE (or gave no runnable cell)
                    break
                result = await interp.execute_code(code, execution_timeout=exec_timeout)
                out = result.get_combined_text()[:MAX_CELL_OUTPUT_CHARS] or "(no output)"
                transcript.append((code, out, resp.reasoning_content))
        finally:
            await interp.close()

        if not transcript:
            warnings.append("explorer ran no cells")
        body = "\n\n".join(f"[cell {i}]\n{code}\n[output]\n{out}" for i, (code, out, _) in enumerate(transcript, 1))
        text = f"{listing}\n\n=== DATA EXPLORATION TRANSCRIPT ===\n{body}"
        if len(text) > MAX_CONTEXT_CHARS:
            text = text[:MAX_CONTEXT_CHARS] + "\n… (context truncated)"
            warnings.append("context truncated")
        return CapsuleContext(cap.capsule_id, text, n_files, warnings, steps=transcript, seed_listing=listing)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


MODES = ("preview", "agent")


# ── the prompt (reverse-engineered from the dataset's hypothesis style) ───────

# Style exemplars are drawn from real dataset rows in OTHER capsules, so they
# anchor the target style without leaking the current capsule's answer.
HYPGEN_INSTRUCTIONS = """\
You are a computational biologist proposing testable scientific hypotheses for a benchmark.

You are given a description of the raw data files available in ONE "capsule" — the input data from a
single biology study (e.g. RNA-seq counts, sample metadata, variant calls, single-cell data, sequences).
The original analysis, paper, and conclusions are hidden from you.

Your task: from the data alone, infer the experimental design (what was measured; which groups,
conditions, genotypes, or covariates exist; what modality), then propose {n} DISTINCT scientific
hypotheses that could be tested using ONLY the files in this capsule.

What makes a good hypothesis (match this style exactly):
- A single declarative sentence stating a SPECIFIC, FALSIFIABLE claim — a real scientific bet that
  could turn out either true or false once the analysis is run.
- Names concrete biological entities that plausibly appear in the data: specific genes, mutations,
  strains/genotypes, cell types, tissues, conditions/media, or named biological processes/pathways.
- States a DIRECTION or RELATIONSHIP, e.g. "X drives/increases/is required for Y", "the effect of X
  on Y is environment-dependent", or "X is NOT explained by Y". Negative / null-style claims are
  welcome where the data supports testing them.
- Is testable by a standard bioinformatics analysis (differential expression, enrichment,
  association/regression, group comparison, etc.) on the PROVIDED data — no external datasets needed.

Avoid:
- Vague or guaranteed-true statements ("there are differences between groups"), pure method
  descriptions, or meta commentary ("this dataset could be used to ...").
- Hedging words ("may", "might", "could suggest").
- Referencing file names or saying "the data shows".
- Hypotheses that require data not present in this capsule.

The study's PUBLISHED PAPER is provided below, when available. Use it the way a researcher uses
related work: to understand the system, the experimental design, and what has ALREADY been
established — then to go somewhere it did not. Your hypotheses must be NEW RELATIVE TO THAT PAPER.

Specifically, a proposed hypothesis is INVALID if it:
- restates, paraphrases, or narrows the paper's own hypothesis or headline claim;
- asserts a finding the paper already reports (including its supplementary results);
- would be confirmed simply by redoing the paper's main analysis.

A proposed hypothesis is GOOD if testing it would add something the paper did not report — for
example a different comparison among the same groups, an untested covariate or interaction, a
mechanism or pathway the paper only mentions in passing, a subgroup or time point it did not
break out, or a claim that would challenge/qualify its conclusion. It must still be answerable
with ONLY the files in this capsule.

Example hypotheses from OTHER capsules, for style calibration only (do NOT reuse or adapt them):
- "Truncating ASXL1 mutations drive specific blood gene expression changes that reflect altered \
hematological and immune processes, including T cell and neutrophil activation."
- "Quorum sensing-mediated division of labor is environment-dependent: ΔrhlI and ΔlasI \
mutants exhibit altered downstream gene expression that varies across different media."
- "The elevated burden of somatic variants in CHIP genes observed in Bloom syndrome probands and \
carriers is not explained by an increased burden of germline variants in CHIP genes."

Return ONLY JSON of the form {{"hypotheses": ["...", "..."]}} with exactly {n} items, ordered from
most to least confidently testable given the available data. Each item must be distinct from the
published paper's claims and from the other items."""


def build_prompt(ctx: CapsuleContext, n: int) -> str:
    parts = [
        HYPGEN_INSTRUCTIONS.format(n=n),
        "\n\n=== CAPSULE DATA FILES ===\n",
        ctx.text or "(no previewable input files)",
        "\n=== END CAPSULE DATA FILES ===\n",
    ]
    if ctx.paper_text:
        parts += [
            "\n=== SOURCE PAPER (already published — your hypotheses must NOT duplicate it) ===\n",
            ctx.paper_text,
            "\n=== END SOURCE PAPER ===\n",
        ]
    return "".join(parts)


# ── grading rubric (opt-in via --rubrics; mirrors the dataset's `rubric` field) ──

# The dataset ships an expert ``rubric`` per capsule: a bulleted list of
# ``* N point(s): <criterion>`` lines — several 1-point procedure steps (data
# loading, QC, core stat method, multiple-testing correction, reporting) plus ONE
# higher-value final criterion for overall correctness, where ``max_points`` is
# their sum. When --rubrics is set, we generate one such rubric per proposed
# hypothesis in that same style. Unlike the expert rubric we do NOT know the
# ground-truth accept/reject outcome, so the final criterion is written to credit
# a rigorous conclusion in whichever direction the evidence points (no hard-coded
# effect size / p-value / direction).
RUBRIC_INSTRUCTIONS = """\
You are a computational biologist writing a GRADING RUBRIC for a hypothesis-testing benchmark task.

An agent will be given the SAME capsule data described below and asked to test the following hypothesis by
writing and running a Jupyter notebook analysis, then stating an accept/reject conclusion:

    HYPOTHESIS: {hypothesis}

Write a rubric a grader will use to score the agent's notebook + final conclusion. It must be tailored to
THIS hypothesis and checkable using ONLY the files in this capsule.

Structure (match the reference rubrics' style exactly):
- A sequence of procedure criteria, each worth 1 point, walking the correct analysis pipeline in order:
  loading the data and correctly defining the groups / variables / covariates; required QC, preprocessing,
  or normalization; the core statistical method appropriate to the modality (e.g. differential expression,
  enrichment, association/regression, group comparison); multiple-testing correction and significance
  thresholds; and clearly reporting the key result. Aim for 4-8 such criteria — only steps a competent
  analysis of THIS hypothesis genuinely requires.
- Exactly ONE final criterion worth more points (2-5) for overall correctness: the analysis is sound and the
  stated accept/reject conclusion is correctly supported by the agent's OWN results. The ground-truth
  outcome is NOT known to you, so DO NOT hard-code an expected direction, effect size, or p-value — credit a
  rigorous conclusion in whichever direction the evidence supports.

Each criterion names concrete, checkable actions (specific methods/tools, thresholds, variables) rather than
vague goals. Do not reference file names. Do not award partial/half points.

Reference rubric from ANOTHER capsule, for style calibration only (do NOT reuse its content):
* 1 point: Loads count data and metadata; correctly defines disease vs control and includes sex as a covariate.
* 1 point: Runs DE analysis (e.g., DESeq2) controlling for sex; applies LFC shrinkage with apeglm (or equivalent).
* 1 point: Applies multiple testing correction (BH/FDR) and filters DEGs at padj < 0.05.
* 1 point: Performs GO BP enrichment on the DEGs using clusterProfiler (or an equivalent tool) with BH correction.
* 1 point: Clearly reports the enriched processes relevant to the hypothesis if significant.
* 5 points: Overall analysis is correct and the accept/reject conclusion is properly supported by the results.

Return ONLY JSON of the form
{{"criteria": [{{"points": 1, "criterion": "..."}}, {{"points": 5, "criterion": "..."}}]}}
ordered from the first analysis step to the final overall-correctness criterion."""


def build_rubric_prompt(ctx: CapsuleContext, hypothesis: str) -> str:
    return (
        RUBRIC_INSTRUCTIONS.format(hypothesis=hypothesis)
        + "\n\n=== CAPSULE DATA FILES ===\n"
        + (ctx.text or "(no previewable input files)")
        + "\n=== END CAPSULE DATA FILES ===\n"
    )


# ── generation ───────────────────────────────────────────────────────────────


class HypothesisSet(BaseModel):
    hypotheses: list[str]


def _first_json_object(text: str) -> dict:
    """Decode the first complete JSON object in ``text``, ignoring trailing data.

    Models sometimes wrap the object in a ```` ```json ```` fence or append prose after the
    closing brace; ``raw_decode`` stops at the end of the first value so that trailing content
    (which ``json.loads`` would reject as "Extra data") is harmless.
    """
    obj, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
    return obj


def _unwrap_envelope(data: dict, field: str) -> dict:
    """Unwrap a tool-call-style envelope (parameters / input / arguments) around ``field``."""
    if field in data:
        return data
    for key in ("parameters", "input", "arguments"):
        inner = data.get(key)
        if isinstance(inner, str):  # openai-style: arguments is a JSON string
            inner = json.loads(inner)
        if isinstance(inner, dict) and field in inner:
            return inner
    return data


def _parse_hypotheses(text: str) -> HypothesisSet:
    """Parse a generation response, unwrapping any tool-call-style envelope."""
    return HypothesisSet.model_validate(_unwrap_envelope(_first_json_object(text), "hypotheses"))


async def generate_one(model: LiteLLMModel, prompt: str, n: int, retries: int = 3) -> tuple[list[str], str | None]:
    """Return ``(hypotheses, reasoning)`` — reasoning is the model's thinking for the call."""
    last_err: Exception | None = None
    for _ in range(retries):
        resp = await model.call_single(prompt, output_type=HypothesisSet, timeout=3 * 60)
        # Content-level failures (empty text, empty '{}' tool call, malformed/truncated
        # JSON) succeed at the HTTP layer, so litellm's transport retry never sees them —
        # retry them here. Transient API errors are still retried inside litellm.
        try:
            if not resp.text:
                raise ValueError("empty response from model")
            return _parse_hypotheses(resp.text).hypotheses[:n], resp.reasoning_content
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_err = e
    raise last_err  # type: ignore[misc]


class RubricItem(BaseModel):
    points: int
    criterion: str


class GeneratedRubric(BaseModel):
    criteria: list[RubricItem]


def render_rubric(rub: GeneratedRubric) -> str:
    """Render a rubric into the dataset's bulleted ``* N point(s): ...`` string form."""
    return "\n".join(
        f"* {it.points} {'point' if it.points == 1 else 'points'}: {it.criterion}" for it in rub.criteria
    )


def _parse_rubric(text: str) -> GeneratedRubric:
    """Parse a rubric response, unwrapping any tool-call-style envelope (see ``_parse_hypotheses``)."""
    return GeneratedRubric.model_validate(_unwrap_envelope(_first_json_object(text), "criteria"))


async def generate_rubric_one(model: LiteLLMModel, prompt: str, retries: int = 3) -> tuple[GeneratedRubric, str | None]:
    """Generate one grading rubric, retrying content-level failures like ``generate_one``.

    Returns ``(rubric, reasoning)`` — reasoning is the model's thinking for the call.
    """
    last_err: Exception | None = None
    for _ in range(retries):
        resp = await model.call_single(prompt, output_type=GeneratedRubric, timeout=3 * 60)
        try:
            if not resp.text:
                raise ValueError("empty response from model")
            return _parse_rubric(resp.text), resp.reasoning_content
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_err = e
    raise last_err  # type: ignore[misc]


async def generate_rubrics(model: LiteLLMModel, ctx: CapsuleContext, hypotheses: list[str]) -> list[dict]:
    """Generate one rubric per hypothesis (concurrently). Returns dicts aligned with ``hypotheses``.

    Each entry mirrors the dataset's grading fields: the structured ``criteria``, the rendered ``rubric``
    string (dataset ``rubric`` format), ``max_points`` (their point sum, the dataset ``max_points``), and
    ``reasoning`` (the model's thinking for that rubric call, None if thinking was off).
    """
    rubrics = await asyncio.gather(*(generate_rubric_one(model, build_rubric_prompt(ctx, h)) for h in hypotheses))
    return [
        {
            "hypothesis": h,
            "criteria": [it.model_dump() for it in rub.criteria],
            "rubric": render_rubric(rub),
            "max_points": sum(it.points for it in rub.criteria),
            "reasoning": reasoning,
        }
        for h, (rub, reasoning) in zip(hypotheses, rubrics, strict=True)
    ]


def write_trajectory(
    save_dir: Path,
    cap: Capsule,
    ctx: CapsuleContext,
    mode: str,
    model_name: str,
    prompt: str,
    hyps: list[str],
    hyp_reasoning: str | None,
    rubrics: list[dict] | None,
    rubric_mode: str | None,
    rubric_ctx: CapsuleContext | None,
    expert_hypothesis: str | None,
    status: str,
    err: str | None,
    max_turns: int,
) -> None:
    """Persist one capsule's proposer trajectory as JSON for later inspection.

    Captures the agent's exploration cells (agent mode), the exact generation
    prompt fed to the model, and the proposed hypotheses. Rendered by
    ``proposer/inspect_proposer_traj.py``.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    traj = {
        "capsule_id": cap.capsule_id,
        "mode": mode,
        "model": model_name,
        "status": status,
        "error": err,
        "n_input_files": ctx.n_files,
        "warnings": ctx.warnings,
        # Agent exploration turns: how many code cells the agent actually ran vs. the
        # allowed budget. turns < max_turns means it stopped early (signalled DONE);
        # turns == max_turns means it likely hit the cap. None/0 for preview mode.
        "turns": len(ctx.steps),
        "max_turns": max_turns if mode == "agent" else None,
        "expert_hypothesis": expert_hypothesis,
        "seed_listing": ctx.seed_listing,
        # The source-paper text the model was shown (empty when the capsule has no paper or
        # --no-paper was passed). Recorded so a hypothesis can be checked against it later.
        "paper_text": ctx.paper_text,
        # In agent mode each explore step carries the model's thinking for that cell (None if off).
        "explore_steps": [
            {"step": i, "code": code, "output": out, "reasoning": reasoning}
            for i, (code, out, reasoning) in enumerate(ctx.steps, 1)
        ],
        "generation_prompt": prompt,
        "hypotheses": hyps,
        # The model's thinking for the hypothesis-generation call (None if thinking was off).
        "hypotheses_reasoning": hyp_reasoning,
        # Per-hypothesis grading rubrics (only when --rubrics is set; None otherwise). Each aligns
        # with hypotheses[i] and carries the structured criteria + rendered dataset-style rubric.
        "rubric_mode": rubric_mode,
        "rubrics": rubrics,
        # When the rubric used a DIFFERENT ingestion mode than the hypotheses (e.g. an agent
        # exploration for rubrics on top of preview hypotheses), record that separate exploration
        # so the rubric context is inspectable too. None when the rubric reused the hypothesis context.
        "rubric_explore_steps": (
            [
                {"step": i, "code": code, "output": out, "reasoning": reasoning}
                for i, (code, out, reasoning) in enumerate(rubric_ctx.steps, 1)
            ]
            if rubric_ctx is not None and rubric_mode != mode
            else None
        ),
    }
    (save_dir / f"{cap.capsule_id}.json").write_text(json.dumps(traj, indent=2))


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capsules-dir", type=Path, default=ROOT / "capsules")
    ap.add_argument("--out", type=Path, default=ROOT / "hypotheses_generated.json")
    ap.add_argument("--n", type=int, default=3, help="Hypotheses to propose per capsule")
    ap.add_argument("--rubrics", action="store_true",
                    help="Also generate a dataset-style grading rubric per proposed hypothesis "
                         "(an extra LLM call per hypothesis).")
    ap.add_argument("--mode", choices=MODES, default="preview", help="Ingestion mode: static previews or live agent")
    ap.add_argument("--rubric-mode", choices=MODES, default=None,
                    help="Ingestion mode for rubric generation (default: same as --mode). Set to 'agent' to "
                         "ground rubrics in a live data exploration even when hypotheses use 'preview'.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="litellm model name (keys via .env)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--explore-steps", type=int, default=DEFAULT_EXPLORE_STEPS, help="agent mode: max code cells")
    ap.add_argument("--exec-timeout", type=int, default=DEFAULT_EXEC_TIMEOUT, help="agent mode: per-cell timeout (s)")
    ap.add_argument("--save-traj", type=Path, default=ROOT / "proposer",
                    help="Base dir for per-capsule trajectory JSON; written under <dir>/<mode>/ "
                         "(e.g. proposer/agent/, proposer/preview/). Pass --no-save-traj to disable.")
    ap.add_argument("--no-save-traj", action="store_true", help="Disable trajectory saving")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N capsules")
    ap.add_argument("--only", default=None, help="Only capsules whose id contains this substring")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset for expert (ground-truth) hypotheses")
    ap.add_argument("--no-paper", action="store_true",
                    help="Do not show the capsule's downloaded source paper to the model. By default the "
                         "paper is included and the model is asked to propose hypotheses DISTINCT from it.")
    ap.add_argument("--dry-run", action="store_true", help="Print the prompt for the first capsule and exit")
    args = ap.parse_args()

    load_env()
    expert = load_expert_hypotheses(args.dataset)

    capsules = discover_capsules(args.capsules_dir)
    if args.only:
        capsules = [c for c in capsules if args.only in c.capsule_id]
    if args.limit is not None:
        capsules = capsules[: args.limit]
    if not capsules:
        sys.exit("no capsules matched")

    # Enable extended thinking for generation. reasoning_effort MUST live inside
    # litellm_params (like server.yaml): lmi only forwards a whitelist (model/n/temperature/
    # max_tokens) from top-level config, so a bare {"reasoning_effort": ...} is silently
    # dropped and no thinking happens. (temperature is left unset — Anthropic requires
    # temperature=1 when thinking is on.)
    model = LiteLLMModel(
        name=args.model,
        config={
            "model_list": [
                {
                    "model_name": args.model,
                    "litellm_params": {"model": args.model, "reasoning_effort": "high"},
                }
            ],
        },
    )

    # Rubrics may ingest the capsule differently than hypotheses (e.g. cheap preview hypotheses but a
    # live agent-explored rubric). Default the rubric mode to the hypothesis mode.
    rubric_mode = args.rubric_mode or args.mode

    async def build_context(cap: Capsule, mode: str) -> CapsuleContext:
        """Dispatch to the given ingestion mode (preview is sync; agent is async).

        The source paper is attached here rather than inside the mode-specific gatherers, so
        both modes see the same paper text on top of their own view of the data.
        """
        if mode == "agent":
            ctx = await gather_context_agent(cap, model, args.explore_steps, args.exec_timeout,
                                             include_paper=not args.no_paper)
        else:
            ctx = gather_context_preview(cap)
        if not args.no_paper:
            ctx.paper_text = extract_paper_text(cap)
            # Warn only when an article file exists but yielded nothing (e.g. a scanned PDF, or
            # pypdf missing) — not for capsules whose paper/ holds just the metadata stub.
            pdir = cap.path / "paper"
            if not ctx.paper_text and (list(pdir.glob("*.fulltext.xml")) or list(pdir.glob("*.pdf"))):
                ctx.warnings.append("paper present but no text extractable")
        return ctx

    if args.dry_run:
        ctx = await build_context(capsules[0], args.mode)
        print(f"# capsule {ctx.capsule_id} ({ctx.n_files} files, warnings={ctx.warnings})\n")
        print(build_prompt(ctx, args.n))
        if args.rubrics:
            # Reuse the hypothesis context when the modes match; otherwise build the rubric-mode one
            # (this actually runs the agent explorer if --rubric-mode agent).
            rctx = ctx if rubric_mode == args.mode else await build_context(capsules[0], rubric_mode)
            print(f"\n\n########## RUBRIC PROMPT (mode={rubric_mode}, one per generated hypothesis) ##########\n")
            print(build_rubric_prompt(rctx, "<one generated hypothesis will be inserted here>"))
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def run(cap: Capsule) -> dict:
        async with sem:
            prompt = ""
            try:
                ctx = await build_context(cap, args.mode)
                # No usable raw input (e.g. capsules holding only .rds results whose
                # CapsuleFolder-*.zip was never downloaded) → skip rather than prompt
                # the model with nothing, which only yields generic filler hypotheses.
                if ctx.n_files == 0:
                    print(f"[skip] {cap.capsule_id}  (no usable input files — {ctx.warnings})")
                    return {
                        "capsule_id": cap.capsule_id, "n_input_files": 0, "mode": args.mode,
                        "status": "skipped", "error": "no usable input files",
                        "expert_hypothesis": expert.get(cap.capsule_id), "hypotheses": [],
                        "hypotheses_reasoning": None, "rubrics": None,
                    }
                prompt = build_prompt(ctx, args.n)
                hyps, hyp_reasoning = await generate_one(model, prompt, args.n)
                status = "ok"
                err = None
            except Exception as e:  # noqa: BLE001 — record per-capsule failures, keep going
                ctx = CapsuleContext(cap.capsule_id, "")
                hyps, hyp_reasoning, status, err = [], None, "error", f"{type(e).__name__}: {e}"

            # Optional second pass: a grading rubric per hypothesis. Kept separate so a rubric
            # failure records an error but never discards the successfully proposed hypotheses.
            # The rubric may ingest the capsule in a different mode than the hypotheses (--rubric-mode):
            # reuse the same context when the modes match, else build a fresh rubric-mode context once
            # per capsule (e.g. a live agent exploration) and reuse it across all this capsule's rubrics.
            rubrics: list[dict] | None = None
            rubric_ctx: CapsuleContext | None = None
            if args.rubrics and status == "ok" and hyps:
                try:
                    rubric_ctx = ctx if rubric_mode == args.mode else await build_context(cap, rubric_mode)
                    rubrics = await generate_rubrics(model, rubric_ctx, hyps)
                except Exception as e:  # noqa: BLE001
                    err = f"{err}; rubric generation failed: {type(e).__name__}: {e}" if err \
                        else f"rubric generation failed: {type(e).__name__}: {e}"

            print(f"[{status}] {cap.capsule_id}  ({len(hyps)} hypotheses"
                  + (f", {len(rubrics)} rubrics [{rubric_mode}]" if rubrics is not None else "") + ")"
                  + (f"  warnings={ctx.warnings}" if ctx.warnings else "")
                  + (f"  err={err}" if err else ""))
            if not args.no_save_traj:
                # Separate trajectories by mode: proposer/agent/ vs proposer/preview/.
                write_trajectory(args.save_traj / args.mode, cap, ctx, args.mode, args.model, prompt, hyps,
                                 hyp_reasoning, rubrics, rubric_mode if args.rubrics else None, rubric_ctx,
                                 expert.get(cap.capsule_id), status, err, args.explore_steps)
            return {
                "capsule_id": cap.capsule_id,
                "n_input_files": ctx.n_files,
                "mode": args.mode,
                "status": status,
                "error": err,
                "expert_hypothesis": expert.get(cap.capsule_id),
                "hypotheses": hyps,
                "hypotheses_reasoning": hyp_reasoning,
                "rubric_mode": rubric_mode if args.rubrics else None,
                "rubrics": rubrics,
            }

    results = await asyncio.gather(*(run(c) for c in capsules))

    args.out.write_text(json.dumps(results, indent=2))
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nWrote {args.out}  ({ok}/{len(results)} capsules succeeded)")


if __name__ == "__main__":
    asyncio.run(main())
