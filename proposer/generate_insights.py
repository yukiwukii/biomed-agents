#!/usr/bin/env python3
"""Generate a cross-capsule data OVERVIEW (thematic file groups) across a batch of capsules.

This is the sibling of ``generate_hypotheses.py`` — it keeps the exact same agentic
capabilities (a live Jupyter kernel driven by the repo's ``Interpreter``, the same
capsule discovery / leakage-controlled materialization / expert-hypothesis lookup),
but swaps the task: instead of proposing falsifiable hypotheses, the agent explores
the raw data of several studies at once and then ORGANIZES every data file — across
ALL capsules, not per capsule — into a small number of natural groups by what kind of
data it is (e.g. "RNA-seq count matrices", "sample metadata tables"), the way
unsupervised clustering would, but by inspection rather than computation. Each group
gets one plain-language, non-expert explanation of what that kind of file contains.

Unlike the hypothesis proposer (one agent per capsule), this script gives ONE agent a
single working directory that holds 10 capsules at once — each materialized into its
own ``<capsule_id>/`` subfolder — and a shared budget of live data-analysis turns across
all of them. After exploration, a single generation pass produces the cross-capsule
groups.

Leakage control is inherited: the same ``EXCLUDE_PATTERNS`` from the proposer hide the
original analysis notebooks (``.ipynb``) and R result objects (``.rds``), so the agent
summarizes the raw inputs, not the paper's computed answer. The dataset's ground-truth
``hypothesis`` is NEVER shown to the model — it is only attached to the OUTPUT, as a
separate per-capsule list, as requested.

The generation pass also writes one top-level ``summary`` — a short plain-language overview
tying the groups together (what kinds of data are present overall, roughly how many studies
and files, and anything notable spanning the batch) — in addition to the groups themselves.
Both the exploration turns and the generation pass run with extended thinking enabled.

Output (``--out``, default ``insights_generated.json``):

    {
      "summary": "This batch spans 10 studies covering ...",
      "expert_hypotheses": [
        {"capsule_id": "...", "expert_hypothesis": "Truncating ASXL1 mutations drive ..."},
        ...
      ],
      "groups": [
        {
          "label": "RNA-seq count matrices",
          "explanation": "Plain-language description of this kind of file ...",
          "files": ["0f14ffa7-.../ROSMAP_genexp_ad.csv", "15ff11e5-.../counts_raw_unfiltered.csv", ...]
        },
        ...
      ]
    }

A full run trajectory (shared workdir layout, seed listing, every explore cell, the
generation prompt, and the reconciled groups) is saved under ``insights/`` by default
(``--save-traj DIR`` / ``--no-save-traj``).

With ``--classification``, the same exploration phase is reused but the generation pass swaps to a
different prompt: instead of clustering files into cross-capsule groups, it works per capsule. A capsule
can hold more than one distinct DATASET (e.g. two independent cohorts, or a bulk and a single-cell assay
on different sample sets); files describing the same samples under the same design (e.g. an expression
matrix and its matched metadata) are kept as one dataset, and clearly separate assays/cohorts become
separate datasets. For each capsule the model writes a plain-language ``overview`` of the dataset(s)
available, then classifies EACH dataset along a fixed set of axes (``study_design``, ``temporal_structure``,
``biological_context``, ``comparison_axis``), emitting ``"unknown"`` for any axis the exploration transcript
doesn't support. Output becomes:

    {
      "expert_hypotheses": [...],
      "classifications": [
        {
          "capsule_id": "...",
          "overview": "This capsule contains one dataset: bulk RNA-seq expression with matched clinical ...",
          "datasets": [
            {
              "label": "ROSMAP bulk RNA-seq + clinical metadata",
              "files": ["ROSMAP_genexp_ad.csv", "ROSMAP_meta_ad.csv"],
              "study_design": "cross_sectional",
              "temporal_structure": "single_timepoint",
              "biological_context": "cancer",
              "comparison_axis": "tumor_vs_normal"
            },
            ...
          ]
        },
        ...
      ]
    }

Usage:
    .venv/bin/python proposer/generate_insights.py                        # first 10 capsules
    .venv/bin/python proposer/generate_insights.py --n-capsules 10 --explore-steps 5
    .venv/bin/python proposer/generate_insights.py --only 0923d260 --only 0f14ffa7
    .venv/bin/python proposer/generate_insights.py --dry-run             # print prompts, no LLM/kernel
    .venv/bin/python proposer/generate_insights.py --classification      # classify studies instead of grouping files
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

# Reuse the proposer's machinery verbatim so the two scripts stay in sync and share the
# "exact same agentic capabilities". scripts/ is not a package, so add it to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from typing import Any, Literal

from generate_hypotheses import (
    DEFAULT_DATASET,
    DEFAULT_EXEC_TIMEOUT,
    DEFAULT_MODEL,
    Capsule,
    CapsuleContext,
    _extract_code,
    _first_json_object,
    _unwrap_envelope,
    discover_capsules,
    gather_context_preview,
    load_env,
    load_expert_hypotheses,
    materialize_capsule,
)
from lmi import LiteLLMModel
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent

# One agent's working directory holds this many capsules; it gets this many explore turns.
DEFAULT_N_CAPSULES = 10
DEFAULT_EXPLORE_STEPS = 50

# Context budget for the (long) transcript fed to the generation pass.
MAX_TRANSCRIPT_CHARS = 80_000
MAX_CELL_OUTPUT_CHARS = 1500

# Extended-thinking budgets (Anthropic `thinking.budget_tokens`) and matching `max_tokens`
# ceilings (must exceed the thinking budget, with enough headroom left for the actual
# response). Explore turns pick one code cell each turn; generation clusters ~236 files
# across 10 studies and writes prose, so it gets a much larger budget.
EXPLORE_THINKING_BUDGET = 2000
EXPLORE_MAX_TOKENS = 4096
GEN_THINKING_BUDGET = 6000
GEN_MAX_TOKENS = 32000


# ── the shared-workdir agent explore loop (multi-capsule variant of the proposer's) ──

EXPLORE_INSTRUCTIONS = """\
You are a bioinformatician exploring the raw data of SEVERAL independent biology studies in one live
Python kernel. The working directory contains one subfolder per study (per "capsule"), each holding
that study's raw data files. Your goal this phase is to UNDERSTAND every data file well enough to later
group similar files together across studies and describe each group: the modality, what was measured,
the shape/dimensions, the key columns/variables, the groups/conditions/genotypes/covariates, sample
sizes, and notable values or obvious data-quality issues.

Each turn, respond with EXACTLY ONE fenced Python code block that prints what you want to inspect
(e.g. a table's shape and columns, value_counts of a metadata column, unique conditions, a few feature
names, dtypes, NA counts). Keep outputs small — print summaries, not whole files. Do not make plots.
Work through the files systematically, capsule by capsule, building on previous outputs. When you have
inspected enough of every file to characterize it, reply with the single word DONE and no code block.

Do NOT propose hypotheses and do NOT state conclusions about the studies. Characterize the data only."""

INSIGHT_GEN_INSTRUCTIONS = """\
You are a bioinformatician writing a DATA OVERVIEW for a benchmark. You just explored, in a live Python
kernel, the raw data files of several independent biology studies laid out as one subfolder per study.

Instead of summarizing each file separately, ORGANIZE all files across ALL studies into a small number of
natural groups by what kind of data they are — like unsupervised clustering, but by inspection rather than
computation. Group files by modality/content similarity (e.g. "gene expression count matrices", "sample or
clinical metadata tables", "protein or genome sequence files", "phylogenetic trees", "variant call
tables"), NOT by which study or capsule they came from — a group should typically span multiple studies
when their files are of the same kind.

For each group, write:
  - "label": a short name for the group (e.g. "RNA-seq count matrices")
  - "explanation": a plain-language description for a NON-EXPERT reader (avoid jargon, or briefly define
    any unavoidable term) of what this kind of file contains, what it's typically used for, and any
    notable patterns or variation you noticed across the group's members (e.g. differing sample sizes,
    species, conditions, obvious data-quality issues). Ground every statement in what you actually
    observed in the exploration transcript.
  - "files": the files in this group, each identified by its path relative to the working directory,
    i.e. "<capsule_id>/<filename>"

Every file listed below must end up in exactly one group. Do NOT propose hypotheses, state study
conclusions, or reference the original paper.

Also write one top-level "summary": a short plain-language overview (a few sentences) tying the groups
together for a NON-EXPERT reader — what kinds of data are present overall across the studies, roughly how
many studies/files, and anything notable spanning the whole batch. Do not state per-study conclusions or
propose hypotheses here either.

Return ONLY JSON of the form:
{"summary": "...", "groups": [{"label": "...", "explanation": "...", "files": ["<capsule_id>/<filename>", ...]}, ...]}"""

# Alternate generation-phase prompt, used instead of INSIGHT_GEN_INSTRUCTIONS when --classification is
# passed: instead of clustering files into cross-capsule groups, this classifies each DATASET WITHIN each
# capsule (a capsule can hold more than one distinct dataset) along a fixed set of axes, plus a plain-language
# per-capsule overview of what datasets it holds. Shares the same explore phase/transcript as the grouping
# prompt.
CLASSIFICATION_INSTRUCTIONS = """\
You are a bioinformatician characterizing several independent biology studies. You just explored, in a live
Python kernel, the raw data files of these studies, laid out as one subfolder per study ("capsule").

A single capsule can contain more than one distinct DATASET — e.g. two independent cohorts, or a bulk-tissue
assay and a single-cell assay run on different sample sets. Files that jointly describe the same set of samples
under the same design (e.g. an expression matrix and its matched sample-metadata file) belong to the SAME
dataset; files from a clearly separate assay, cohort, or design belong to a DIFFERENT dataset. Most capsules
will have exactly one dataset — only split into more when the files genuinely describe unrelated sample sets.

For EACH capsule below, produce:

  - "overview": a short plain-language description (for a non-expert reader) of the dataset(s) available in
    this capsule — how many distinct datasets it contains and, briefly, what each one is.
  - "datasets": a list of the distinct datasets you identified in this capsule. For each dataset, give:
      - "label": a short name for the dataset
      - "files": the filenames belonging to this dataset, relative to the capsule subfolder (i.e. just
        "<filename>", with no capsule_id prefix)
      - "study_design": one of ["cross_sectional", "longitudinal", "perturbation_stimulation",
        "challenge_infection", "multi_tissue_paired", "developmental_trajectory", "atlas", "unknown"]
      - "temporal_structure": one of ["single_timepoint", "multi_timepoint_same_subject",
        "multi_timepoint_different_subjects", "pseudotime_only", "unknown"]
      - "biological_context": one of ["vaccination", "infection", "cancer", "autoimmune", "drug_treatment",
        "development", "healthy_reference", "other", "unknown"]
      - "comparison_axis": one of ["disease_vs_healthy", "pre_vs_post_treatment", "stimulated_vs_unstimulated",
        "tumor_vs_normal", "across_time", "across_tissue", "none", "unknown"]

Use "unknown" for any axis where the evidence you actually observed is insufficient to decide confidently — do
not guess, and do not rely on outside knowledge of the study or its paper. Classify each dataset independently:
two datasets in the same capsule may end up with different classifications.

Return ONLY JSON of the form:
{"classifications": [{"capsule_id": "...", "overview": "...", "datasets": [{"label": "...",
"files": ["<filename>", ...], "study_design": "...", "temporal_structure": "...", "biological_context": "...",
"comparison_axis": "..."}, ...]}, ...]}

Every capsule id listed below must appear in exactly one classification entry, and every file belonging to a
capsule must appear in exactly one of that capsule's datasets."""


class FileGroup(BaseModel):
    label: str
    explanation: str
    files: list[str]


class GroupedInsights(BaseModel):
    summary: str
    groups: list[FileGroup]


class CapsuleDataset(BaseModel):
    label: str
    files: list[str]
    study_design: Literal[
        "cross_sectional",
        "longitudinal",
        "perturbation_stimulation",
        "challenge_infection",
        "multi_tissue_paired",
        "developmental_trajectory",
        "atlas",
        "unknown",
    ]
    temporal_structure: Literal[
        "single_timepoint",
        "multi_timepoint_same_subject",
        "multi_timepoint_different_subjects",
        "pseudotime_only",
        "unknown",
    ]
    biological_context: Literal[
        "vaccination",
        "infection",
        "cancer",
        "autoimmune",
        "drug_treatment",
        "development",
        "healthy_reference",
        "other",
        "unknown",
    ]
    comparison_axis: Literal[
        "disease_vs_healthy",
        "pre_vs_post_treatment",
        "stimulated_vs_unstimulated",
        "tumor_vs_normal",
        "across_time",
        "across_tissue",
        "none",
        "unknown",
    ]


class CapsuleClassification(BaseModel):
    capsule_id: str
    overview: str
    datasets: list[CapsuleDataset]


class ClassifiedInsights(BaseModel):
    classifications: list[CapsuleClassification]


def _capsule_file_list(cap_root: Path) -> list[str]:
    """Relative paths (posix) of every materialized file under a capsule subfolder."""
    return sorted(str(p.relative_to(cap_root).as_posix()) for p in cap_root.rglob("*") if p.is_file())


def _build_explore_prompt(listing: str, transcript: list[tuple[str, str]], step: int, max_steps: int) -> str:
    parts = [
        EXPLORE_INSTRUCTIONS,
        f"\n(You have run {step}/{max_steps} cells.)",
        "\n=== CAPSULE SUBFOLDERS IN WORKDIR ===",
        listing,
    ]
    if transcript:
        parts.append("\n=== YOUR EXPLORATION SO FAR ===")
        for i, (code, out) in enumerate(transcript, 1):
            parts.append(f"[cell {i}]\n{code}\n[output]\n{out}")
    parts.append("\nReply with the next code cell, or DONE.")
    return "\n".join(parts)


def _render_transcript(transcript: list[tuple[str, str]]) -> str:
    body = "\n\n".join(f"[cell {i}]\n{code}\n[output]\n{out}" for i, (code, out) in enumerate(transcript, 1))
    if len(body) > MAX_TRANSCRIPT_CHARS:
        body = body[:MAX_TRANSCRIPT_CHARS] + "\n… (transcript truncated)"
    return body


def build_gen_prompt(listing: str, file_paths: list[str], transcript: list[tuple[str, str]]) -> str:
    files_block = "\n".join(f"- {p}" for p in file_paths)
    return (
        INSIGHT_GEN_INSTRUCTIONS
        + "\n\n=== FILES TO ORGANIZE INTO GROUPS (every path below must appear in exactly one group) ===\n"
        + files_block
        + "\n\n=== CAPSULE SUBFOLDERS (seed listing) ===\n"
        + listing
        + "\n\n=== YOUR DATA EXPLORATION TRANSCRIPT ===\n"
        + (_render_transcript(transcript) or "(no exploration was run)")
    )


def build_classification_prompt(
    listing: str, capsule_files: dict[str, list[str]], transcript: list[tuple[str, str]]
) -> str:
    caps_block = "\n".join(
        f"### capsule {cid}/  ({len(files)} files)\n" + "\n".join(f"- {f}" for f in files)
        for cid, files in capsule_files.items()
    )
    return (
        CLASSIFICATION_INSTRUCTIONS
        + "\n\n=== CAPSULES AND THEIR FILES (every file below must appear in exactly one dataset of its capsule) "
        "===\n"
        + caps_block
        + "\n\n=== CAPSULE SUBFOLDERS (seed listing) ===\n"
        + listing
        + "\n\n=== YOUR DATA EXPLORATION TRANSCRIPT ===\n"
        + (_render_transcript(transcript) or "(no exploration was run)")
    )


def _parse_groups(text: str) -> GroupedInsights:
    return GroupedInsights.model_validate(_unwrap_envelope(_first_json_object(text), "groups"))


def _parse_classification(text: str) -> ClassifiedInsights:
    return ClassifiedInsights.model_validate(_unwrap_envelope(_first_json_object(text), "classifications"))


async def generate_groups(model: LiteLLMModel, prompt: str, retries: int = 3) -> GroupedInsights:
    """Single generation pass producing the top-level summary and cross-capsule file groups.

    Retries content-level failures (empty text, malformed/truncated JSON). Runs with extended
    thinking, since clustering ~236 files across 10 studies and writing the overview is the
    hardest reasoning step in this pipeline.
    """
    last_err: Exception | None = None
    for _ in range(retries):
        resp = await model.call_single(
            prompt,
            output_type=GroupedInsights,
            timeout=10 * 60,
            temperature=1,
            max_tokens=GEN_MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": GEN_THINKING_BUDGET},
        )
        try:
            if not resp.text:
                raise ValueError("empty response from model")
            return _parse_groups(resp.text)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_err = e
    raise last_err  # type: ignore[misc]


async def generate_classification(model: LiteLLMModel, prompt: str, retries: int = 3) -> ClassifiedInsights:
    """Single generation pass classifying each study along the fixed axes (--classification mode).

    Mirrors ``generate_groups`` (same retry-on-malformed-JSON behavior, same thinking budget) but
    produces a per-capsule classification instead of cross-capsule file groups.
    """
    last_err: Exception | None = None
    for _ in range(retries):
        resp = await model.call_single(
            prompt,
            output_type=ClassifiedInsights,
            timeout=10 * 60,
            temperature=1,
            max_tokens=GEN_MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": GEN_THINKING_BUDGET},
        )
        try:
            if not resp.text:
                raise ValueError("empty response from model")
            return _parse_classification(resp.text)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_err = e
    raise last_err  # type: ignore[misc]


# ── batch layout: one shared workdir with N capsule subfolders ────────────────


class BatchLayout:
    """A shared working directory holding several capsules, one subfolder each."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        # capsule_id -> list of relative file paths under that capsule's subfolder
        self.files: dict[str, list[str]] = {}
        # capsule_id -> per-capsule preview listing text (seed context)
        self.previews: dict[str, str] = {}
        self.warnings: dict[str, list[str]] = {}

    def add(self, cap: Capsule) -> int:
        dest = self.workdir / cap.capsule_id
        dest.mkdir(parents=True, exist_ok=True)
        n = materialize_capsule(cap, dest)
        self.files[cap.capsule_id] = _capsule_file_list(dest)
        preview = gather_context_preview(cap)
        self.previews[cap.capsule_id] = preview.text
        self.warnings[cap.capsule_id] = list(preview.warnings)
        return n

    def seed_listing(self) -> str:
        blocks: list[str] = []
        for cid, files in self.files.items():
            blocks.extend((
                f"### capsule {cid}/  ({len(files)} files)",
                self.previews.get(cid, "") or "(no previewable files)",
            ))
        return "\n".join(blocks)

    def all_file_paths(self) -> list[str]:
        return [f"{cid}/{rel}" for cid, files in self.files.items() for rel in files]


async def explore_batch(
    layout: BatchLayout,
    model: LiteLLMModel,
    max_steps: int,
    exec_timeout: int,
) -> list[tuple[str, str]]:
    """Run one live-kernel explore loop over the shared workdir; return the (code, output) transcript."""
    from hypotest.env import utils as env_utils
    from hypotest.env.interpreter import Interpreter

    listing = layout.seed_listing()
    interp = Interpreter(work_dir=layout.workdir, language=env_utils.NBLanguage.PYTHON, execution_timeout=exec_timeout)
    await interp.start()
    transcript: list[tuple[str, str]] = []
    try:
        for step in range(max_steps):
            resp = await model.call_single(
                _build_explore_prompt(listing, transcript, step, max_steps),
                timeout=300,
                temperature=1,
                max_tokens=EXPLORE_MAX_TOKENS,
                thinking={"type": "enabled", "budget_tokens": EXPLORE_THINKING_BUDGET},
            )
            code = _extract_code(resp.text or "")
            if not code:  # DONE (or no runnable cell)
                break
            result = await interp.execute_code(code, execution_timeout=exec_timeout)
            out = result.get_combined_text()[:MAX_CELL_OUTPUT_CHARS] or "(no output)"
            transcript.append((code, out))
    finally:
        await interp.close()
    return transcript


def reconcile_groups(
    layout: BatchLayout,
    groups: list[FileGroup],
) -> tuple[list[dict], list[str]]:
    """Validate model groups against the known cross-capsule file universe.

    Drops any path the model invented or duplicated across groups (kept in the first group
    it appears in); any file the model failed to place lands in a trailing "(ungrouped)"
    bucket. Warns on every anomaly so nothing silently vanishes.
    """
    known = set(layout.all_file_paths())
    warnings: list[str] = []
    seen: set[str] = set()
    results: list[dict] = []
    for g in groups:
        files: list[str] = []
        for path in g.files:
            if path not in known:
                warnings.append(f"group {g.label!r} references unknown path {path} (dropped)")
                continue
            if path in seen:
                warnings.append(f"path {path} appears in multiple groups (kept in first, dropped from {g.label!r})")
                continue
            seen.add(path)
            files.append(path)
        results.append({"label": g.label, "explanation": g.explanation, "files": files})
    missing = sorted(known - seen)
    warnings.extend(f"missing group assignment for {path}" for path in missing)
    if missing:
        results.append({"label": "(ungrouped)", "explanation": None, "files": missing})
    return results, warnings


_UNKNOWN_AXES = {
    "study_design": "unknown",
    "temporal_structure": "unknown",
    "biological_context": "unknown",
    "comparison_axis": "unknown",
}


def reconcile_classifications(
    layout: BatchLayout,
    classifications: list[CapsuleClassification],
) -> tuple[list[dict], list[str]]:
    """Validate model classifications against the known capsule/file universe (--classification mode).

    Drops any capsule_id the model invented or duplicated (kept in the first entry it appears in).
    Within a capsule, drops any file the model invented or assigned to more than one dataset (kept
    in the first dataset it appears in); any file the model failed to place lands in a trailing
    "(unclassified)" dataset. Any capsule the model failed to classify at all gets a single
    "(unclassified)" dataset covering every one of its files, with "unknown" axes and no overview.
    Warns on every anomaly so nothing silently vanishes.
    """
    known_capsules = set(layout.files.keys())
    warnings: list[str] = []
    seen_capsules: set[str] = set()
    results: list[dict] = []
    for c in classifications:
        if c.capsule_id not in known_capsules:
            warnings.append(f"classification references unknown capsule_id {c.capsule_id!r} (dropped)")
            continue
        if c.capsule_id in seen_capsules:
            warnings.append(f"capsule_id {c.capsule_id!r} classified more than once (kept first)")
            continue
        seen_capsules.add(c.capsule_id)

        known_files = set(layout.files[c.capsule_id])
        seen_files: set[str] = set()
        datasets: list[dict] = []
        for ds in c.datasets:
            files: list[str] = []
            for f in ds.files:
                if f not in known_files:
                    warnings.append(
                        f"capsule {c.capsule_id!r} dataset {ds.label!r} references unknown file {f!r} (dropped)"
                    )
                    continue
                if f in seen_files:
                    warnings.append(
                        f"capsule {c.capsule_id!r}: file {f!r} appears in multiple datasets "
                        f"(kept in first, dropped from {ds.label!r})"
                    )
                    continue
                seen_files.add(f)
                files.append(f)
            datasets.append({
                "label": ds.label,
                "files": files,
                "study_design": ds.study_design,
                "temporal_structure": ds.temporal_structure,
                "biological_context": ds.biological_context,
                "comparison_axis": ds.comparison_axis,
            })
        missing_files = sorted(known_files - seen_files)
        warnings.extend(f"capsule {c.capsule_id!r}: missing dataset assignment for file {f!r}" for f in missing_files)
        if missing_files:
            datasets.append({"label": "(unclassified)", "files": missing_files, **_UNKNOWN_AXES})
        results.append({"capsule_id": c.capsule_id, "overview": c.overview, "datasets": datasets})

    missing_capsules = sorted(known_capsules - seen_capsules)
    for cid in missing_capsules:
        warnings.append(f"missing classification for capsule {cid}")
        results.append({
            "capsule_id": cid,
            "overview": None,
            "datasets": [{"label": "(unclassified)", "files": list(layout.files[cid]), **_UNKNOWN_AXES}],
        })
    return results, warnings


def write_trajectory(
    save_dir: Path,
    layout: BatchLayout,
    model_name: str,
    max_steps: int,
    transcript: list[tuple[str, str]],
    gen_prompt: str,
    output: dict,
    warnings: list[str],
    status: str,
    err: str | None,
) -> None:
    """``output`` is the same mode-specific dict written to ``--out``: ``{"summary", "expert_hypotheses",
    "groups"}`` in the default grouping mode, or ``{"expert_hypotheses", "classifications"}`` under
    ``--classification``.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    traj = {
        "model": model_name,
        "status": status,
        "error": err,
        "capsule_ids": list(layout.files.keys()),
        "n_capsules": len(layout.files),
        "n_files_total": sum(len(f) for f in layout.files.values()),
        "turns": len(transcript),
        "max_turns": max_steps,
        "warnings": warnings,
        "seed_listing": layout.seed_listing(),
        "explore_steps": [{"step": i, "code": c, "output": o} for i, (c, o) in enumerate(transcript, 1)],
        "generation_prompt": gen_prompt,
        **output,
    }
    ts = save_dir / "run.json"
    ts.write_text(json.dumps(traj, indent=2))


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capsules-dir", type=Path, default=ROOT / "capsules")
    ap.add_argument("--out", type=Path, default=ROOT / "insights_generated.json")
    ap.add_argument(
        "--n-capsules",
        type=int,
        default=DEFAULT_N_CAPSULES,
        help="How many capsules to place in the shared working directory",
    )
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        help="Only capsules whose id contains this substring (repeatable). Overrides --n-capsules.",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help="litellm model name (keys via .env)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument(
        "--explore-steps",
        type=int,
        default=DEFAULT_EXPLORE_STEPS,
        help="Max code cells the shared agent may run over all capsules",
    )
    ap.add_argument("--exec-timeout", type=int, default=DEFAULT_EXEC_TIMEOUT, help="Per-cell timeout (s)")
    ap.add_argument(
        "--save-traj",
        type=Path,
        default=ROOT / "insights",
        help="Dir for the run trajectory JSON (run.json). Pass --no-save-traj to disable.",
    )
    ap.add_argument("--no-save-traj", action="store_true", help="Disable trajectory saving")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset for expert (ground-truth) hypotheses")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize the batch and print the explore + generation prompts, then exit (no LLM/kernel)",
    )
    ap.add_argument(
        "--classification",
        action="store_true",
        help="Classify each study along fixed axes (study_design, temporal_structure, "
        "biological_context, comparison_axis) instead of grouping files across capsules",
    )
    args = ap.parse_args()

    load_env()
    expert = load_expert_hypotheses(args.dataset)

    capsules = discover_capsules(args.capsules_dir)
    if args.only:
        capsules = [c for c in capsules if any(s in c.capsule_id for s in args.only)]
    else:
        capsules = capsules[: args.n_capsules]
    if not capsules:
        sys.exit("no capsules matched")

    model = LiteLLMModel(name=args.model, config={"temperature": args.temperature})

    workdir = Path(tempfile.mkdtemp(prefix="insights-batch-"))
    try:
        layout = BatchLayout(workdir)
        for cap in capsules:
            n = layout.add(cap)
            print(f"[load] {cap.capsule_id}  ({n} files)")

        file_paths = layout.all_file_paths()
        if not file_paths:
            sys.exit("no usable input files across the selected capsules")

        if args.dry_run:
            print("\n########## EXPLORE PROMPT (turn 0) ##########\n")
            print(_build_explore_prompt(layout.seed_listing(), [], 0, args.explore_steps))
            print("\n\n########## GENERATION PROMPT (after exploration) ##########\n")
            if args.classification:
                print(build_classification_prompt(layout.seed_listing(), layout.files, []))
            else:
                print(build_gen_prompt(layout.seed_listing(), file_paths, []))
            return

        transcript: list[tuple[str, str]] = []
        status, err = "ok", None
        try:
            transcript = await explore_batch(layout, model, args.explore_steps, args.exec_timeout)
            print(f"[explore] ran {len(transcript)}/{args.explore_steps} cells over {len(capsules)} capsules")
            if args.classification:
                gen_prompt = build_classification_prompt(layout.seed_listing(), layout.files, transcript)
                classified = await generate_classification(model, gen_prompt)
                classifications = classified.classifications
            else:
                gen_prompt = build_gen_prompt(layout.seed_listing(), file_paths, transcript)
                generated = await generate_groups(model, gen_prompt)
                summary, groups = generated.summary, generated.groups
        except Exception as e:
            status, err = "error", f"{type(e).__name__}: {e}"
            if args.classification:
                classifications = []
                gen_prompt = build_classification_prompt(layout.seed_listing(), layout.files, transcript)
            else:
                summary, groups = None, []
                gen_prompt = build_gen_prompt(layout.seed_listing(), file_paths, transcript)

        expert_hypotheses = [{"capsule_id": cid, "expert_hypothesis": expert.get(cid)} for cid in layout.files]

        if args.classification:
            results, warnings = reconcile_classifications(layout, classifications)
            if err:
                warnings.append(err)
            output: dict[str, Any] = {"expert_hypotheses": expert_hypotheses, "classifications": results}
            n_datasets = sum(len(c["datasets"]) for c in results)
            print(
                f"[{status}] classified {len(results)}/{len(layout.files)} capsules into {n_datasets} datasets"
                + (f"  ({len(warnings)} warnings)" if warnings else "")
                + (f"  err={err}" if err else "")
            )
        else:
            results, warnings = reconcile_groups(layout, groups)
            if err:
                warnings.append(err)
            output = {"summary": summary, "expert_hypotheses": expert_hypotheses, "groups": results}
            n_grouped = sum(len(g["files"]) for g in results if g["label"] != "(ungrouped)")
            print(
                f"[{status}] {n_grouped}/{len(file_paths)} files grouped into {len(results)} groups"
                + (f"  ({len(warnings)} warnings)" if warnings else "")
                + (f"  err={err}" if err else "")
            )

        args.out.write_text(json.dumps(output, indent=2))
        print(f"\nWrote {args.out}")

        if not args.no_save_traj:
            write_trajectory(
                args.save_traj,
                layout,
                args.model,
                args.explore_steps,
                transcript,
                gen_prompt,
                output,
                warnings,
                status,
                err,
            )
            print(f"Wrote trajectory {args.save_traj / 'run.json'}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
