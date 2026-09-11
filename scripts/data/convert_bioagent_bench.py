#!/usr/bin/env python3
"""Convert bioagent-bench into hypotest's ProblemInstance jsonl.

Run ``scripts/data/stage_bioagent_capsules.py`` first — this script reads what that staged:

    capsules/bioagent-bench/<task_id>/data|reference/   the agent's inputs
    capsules/bioagent-bench/_truth/<task_id>/           the answer key (never shipped to the agent)
    capsules/bioagent-bench/manifest.json               what actually landed

Why the rubrics are generated rather than hand-written
------------------------------------------------------
bioagent-bench itself ships no rubric. Its reward lives in a *second* repo,
``bioagent-bench/bioagent-experiments``, as ``tasksets/bioagent/bioagent_taskset/scoring.py``:
the agent writes a CSV to ``results/`` and a per-task Python function compares it to the truth
files, returning one bool. The whole reward is ``float(deterministic_match)`` — pass/fail, no
partial credit. (That repo also runs an LLM judge, but its output is recorded as metrics only
and explicitly does not feed the score.)

These checks are now transcribed into hypotest as a deterministic judge,
``hypotest/env/judges/bioagent.py``, registered with ``needs_model=False``. Problems emitted here
carry ``judge: "bioagent"``, so **that judge — not the rubric below — produces the reward**: it
reads whatever the agent wrote to ``results/``, compares it to ``_truth/<task_id>/``, and returns
0 or 1 out of 1. No LLM is called, and a run over this dataset needs no ``rubric_model`` at all.

The rubric is still generated, for two reasons: it predates the deterministic judge and remains
the record of what each task is asking for, and forcing ``judge: hypotest`` in ``server.yaml``
falls back to it, which is how you A/B the graded-/10 signal against the binary one. It restates
each deterministic check as a *question the agent answers in prose* and grades that — which works
because their checks are really facts, not file properties: ``cystic-fibrosis`` does not care that
you produced a CSV, it cares that you found chr7:117227832 G>T in CFTR.

Each rubric is built as:

  - Criterion 1, worth ``UPSTREAM_POINTS`` of ``MAX_SCORE``, all-or-nothing: a literal restatement
    of upstream's ``RESULT_RULES[task_id]`` with the real truth values inlined from ``_truth/``.
    Scored 0 or 6, so this single criterion reproduces their binary metric — recover it post-hoc
    from ``criteria[0]`` in the saved ``score_info.json`` to get a number comparable to their
    published table.
  - Criteria 2-4, worth the remaining points: method, rigour, and clarity. These exist because a
    0-or-1 reward over nine tasks is a very blunt training signal; upstream reached for the same
    thing when its judge counted "how many pipeline steps completed".

Truth values are read from ``_truth/`` at conversion time rather than pasted into this file, so a
re-staged capsule and its rubric can never disagree.

Two deliberate deviations from upstream, both recorded in each problem's ``metadata``. Note these
are properties of the RUBRIC only — ``judges/bioagent.py`` implements upstream's exact condition in
both cases, so it is the stricter of the two graders:

  1. ``single-cell``: upstream requires a matching ``(cluster_id, predicted_cell_type)`` pair, but
     cluster numbering is not stable across runs (reseed the clustering and cluster 3 becomes
     cluster 7). The rubric grades the cell-type identification and its marker-gene support, and
     ignores the cluster number.
  2. ``transcript-quant``: upstream requires all 278 transcript counts to match exactly. An LLM
     judge cannot reliably diff 278 integers, so the rubric checks the total assigned count plus
     a fixed deterministic sample of named transcripts. This is a weaker check than upstream's.

``giab`` is not converted: its score is an F1 produced by ``hap.py`` against a truth VCF the agent
must never see. The deterministic judge could host it now, but three things are still missing —
its truth VCF is not staged, ``hap.py`` must be on PATH, and an F1 is continuous where the reward
is binary (it would need a threshold, or a float ``JudgeResult.raw_score``).

Usage:
    .venv/bin/python scripts/data/convert_bioagent_bench.py
    .venv/bin/python scripts/data/convert_bioagent_bench.py --tasks cystic-fibrosis viral-metagenomics
    .venv/bin/python scripts/data/convert_bioagent_bench.py --print-rubric cystic-fibrosis
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

REPO_ID = "bioagent-bench/bioagent-bench"
METADATA_URL = f"https://raw.githubusercontent.com/{REPO_ID}/master/src/task_metadata.json"
UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"hypotest:{REPO_ID}")

TRUTH_DIR_NAME = "_truth"

# Points split. Criterion 1 carries UPSTREAM_POINTS and is all-or-nothing so it reproduces
# upstream's binary pass/fail; the rest spread over method/rigour/clarity.
UPSTREAM_POINTS = 6
MAX_SCORE = 10

# Verbatim from bioagent-experiments tasksets/bioagent/bioagent_taskset/prompts.py::RESULT_RULES.
# Kept in lockstep so criterion 1 states exactly the condition upstream's scorer applies.
RESULT_RULES = {
    "alzheimer-mouse": (
        "results_match is true only if the generated CSV and expected CSV share at least one Pathway value."
    ),
    "comparative-genomics": (
        "results_match is true only if at least one consensus_annotation value exactly matches the expected results."
    ),
    "cystic-fibrosis": (
        "results_match is true only if the causal CFTR variant is reported exactly once with chromosome 7, "
        "position 117227832, variant_id 7115, reference G, alternate T, gene CFTR, gene_id ENSG00000001626, "
        "annotation stop_gained, impact HIGH, and transcript ENST00000003084."
    ),
    "deseq": "results_match is true only if at least five gene_id values overlap the expected DESeq output.",
    "evolution": (
        "results_match is true only if at least one chromosome/CHROM and position/POS pair matches the "
        "expected variants."
    ),
    "metagenomics": (
        "results_match is true only if the most abundant phylum is Pseudomonadota when comparing JP4D and JC1A "
        "and at least two additional OTUs have the expected Phylum labels."
    ),
    "single-cell": (
        "results_match is true only if at least one (cluster_id, predicted_cell_type) pair matches the "
        "expected results."
    ),
    "transcript-quant": (
        "results_match is true only if the complete transcript_id to count mapping equals the expected mapping; "
        "row order and formatting may differ."
    ),
    "viral-metagenomics": (
        "results_match is true only if Bottlenose dolphin adenovirus 1 is explicitly reported under the Viruses domain."
    ),
}

# Not convertible to a prose question — see module docstring.
SKIP_TASKS = {"giab"}

PREAMBLE = f"""\
Score each criterion independently against the notebook and the agent's final answer.

Criterion 1 restates the benchmark's own deterministic pass condition and is ALL OR NOTHING: award
its full {UPSTREAM_POINTS} points only if the condition holds exactly as written, otherwise 0. Do not
award partial credit on criterion 1 and do not soften it because the surrounding analysis looked
reasonable. A correct value that the agent listed as one possibility among several, or asserted
without supporting analysis in the notebook, does not satisfy it.

The remaining criteria grade how the result was reached. Award those on the evidence visible in the
notebook; claims made only in the final answer, with no corresponding executed cell, score 0.
"""


def load_env_metadata(path: Path | None) -> list[dict[str, Any]]:
    if path is not None:
        return json.loads(path.read_text())
    import requests

    resp = requests.get(METADATA_URL, timeout=60)
    resp.raise_for_status()
    return resp.json()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def bullet_list(items: list[str], indent: str = "     ") -> str:
    return "\n".join(f"{indent}- {item}" for item in items)


def render(task_id: str, upstream_criterion: str, method: str, rigour: str, clarity: str) -> str:
    """Assemble one rubric in hypotest's ``N. (X points)`` criterion format.

    The ``N. (X points)`` shape matters: ``is_biomni_rubric`` keys off the BiomniBench "Levels:
    A=/B=/C=" structure, so staying in this format keeps grading on hypotest's integer-per-criterion
    judge rather than the A/B/C one.
    """
    remaining = MAX_SCORE - UPSTREAM_POINTS
    method_pts, rigour_pts, clarity_pts = remaining - 2, 1, 1
    return (
        f"RUBRIC: {REPO_ID} — {task_id}\n\n"
        f"Total Points: {MAX_SCORE}/{MAX_SCORE}\n\n"
        f"{PREAMBLE}\n"
        f"1. ({UPSTREAM_POINTS} points) BENCHMARK PASS CONDITION (all or nothing).\n"
        f"   Upstream rule: {RESULT_RULES[task_id]}\n"
        f"{upstream_criterion}\n\n"
        f"2. ({method_pts} points) Method.\n{method}\n\n"
        f"3. ({rigour_pts} point) Rigour.\n{rigour}\n\n"
        f"4. ({clarity_pts} point) Clarity of the final answer.\n{clarity}"
    )


# --------------------------------------------------------------------------------------------
# Per-task rubric builders. Each reads its own truth file(s) and returns the rubric text.
# --------------------------------------------------------------------------------------------


def rubric_alzheimer(truth: Path) -> str:
    rows = read_csv(truth / "pathway_comparison.csv")
    pathways = [r["Pathway"] for r in rows]
    return render(
        "alzheimer-mouse",
        "   Award the points only if the agent's answer names at least one of these KEGG pathways as\n"
        "   shared across the three mouse models:\n"
        f"{bullet_list(pathways)}\n"
        "   Pathway naming need not match character for character — the KEGG identifier (hsaNNNNN) or an\n"
        "   unambiguous name match is enough. A pathway mentioned only in passing, not presented as a\n"
        "   cross-model shared result, does not count.",
        "   Differential expression run separately for all three models (5xFAD, 3xTG-AD, PS3O1S), then KEGG\n"
        "   enrichment per model, then an explicit cross-model comparison. Note that PS3O1S ships as\n"
        "   precomputed DE results while the other two ship raw counts; handling that difference correctly\n"
        "   is part of the task.",
        "   Multiple-testing correction applied to the DE results, and per-pathway p-values reported for each\n"
        "   model rather than a bare pathway list.",
        "   The final answer states the shared pathways and their per-model p-values, not just a file path.",
    )


def rubric_comparative_genomics(truth: Path) -> str:
    rows = read_csv(truth / "cluster_annotation_mapping.csv")
    annotations = sorted({r["consensus_annotation"] for r in rows})
    ko_ids = sorted({a.split()[0] for a in annotations})
    return render(
        "comparative-genomics",
        "   Award the points only if the agent reports at least one cluster annotation whose KEGG orthology\n"
        "   identifier appears in this expected set:\n"
        f"     {', '.join(ko_ids)}\n"
        "   Match on the K-number; the trailing description need not match. The agent must present it as a\n"
        "   consensus annotation of a co-evolving cluster it constructed.",
        "   Orthologous cluster construction across all four Micrococcus genomes, phylogenetic reconstruction,\n"
        "   and the three required filters applied: present in all four organisms, coding regions only, and at\n"
        "   least one high-confidence annotation per cluster.",
        "   Clusters are grounded in the actual sequence/annotation data rather than asserted, and the\n"
        "   filtering steps are shown to have been applied rather than merely described.",
        "   The final answer gives the cluster-to-annotation mapping using K-numbers, as the task requires.",
    )


def rubric_cystic_fibrosis(truth: Path) -> str:
    row = read_csv(truth / "cf_variants.csv")[0]
    fields = [
        ("chromosome", row["chromosome"]),
        ("position", row["position"]),
        ("reference allele", row["reference"]),
        ("alternate allele", row["alternate"]),
        ("gene", row["gene_name"]),
        ("annotation", row["annotation"]),
        ("impact", row["impact"]),
        ("transcript", row["transcript_id"]),
        ("HGVS protein", row["hgvs_p"]),
    ]
    return render(
        "cystic-fibrosis",
        "   Award the points only if the agent identifies this single variant as THE cause of the disease:\n"
        f"{bullet_list([f'{k}: {v}' for k, v in fields])}\n"
        "   It must be reported as the causal variant, exactly once — not as one candidate among several, and\n"
        "   not as a shortlist the agent declined to narrow. Chromosome may be written '7' or 'chr7'. The\n"
        "   agent need not report every field above, but nothing it does report may contradict them, and the\n"
        "   gene and position must both be present and correct.",
        "   Filtering follows the recessive inheritance pattern using the supplied pedigree: homozygous in the\n"
        "   three affected siblings (NA12885, NA12886, NA12879), heterozygous in the parents, and absent or\n"
        "   heterozygous in unaffected siblings. Note the input VCF is already snpEff-annotated, so the work is\n"
        "   the genotype-based filtering, not re-annotation.",
        "   Candidate variants are narrowed by evidence rather than by picking a known CF gene a priori; the\n"
        "   ClinVar reference is used to corroborate rather than to substitute for the pedigree filtering.",
        "   The final answer states the variant and its coordinates explicitly.",
    )


def rubric_deseq(truth: Path) -> str:
    rows = read_csv(truth / "up_regulated_genes.csv")
    genes = sorted(r["gene_id"] for r in rows)
    return render(
        "deseq",
        "   Award the points only if at least five genes the agent reports as significantly UP-regulated in\n"
        "   biofilm relative to planktonic appear in this expected set:\n"
        f"     {', '.join(genes)}\n"
        "   Note the direction: the benchmark's truth file contains up-regulated genes only, so\n"
        "   down-regulated hits do not count toward the five.",
        "   Reads aligned to the C. parapsilosis reference, counts summarised per gene, then a negative-binomial\n"
        "   differential expression method (DESeq2 or pydeseq2) applied with condition as the design factor and\n"
        "   the six samples correctly assigned to planktonic vs biofilm.",
        "   Significance judged on adjusted p-values rather than raw ones, and the reported gene list is filtered\n"
        "   by an explicit threshold.",
        "   The final answer lists specific up-regulated gene IDs with their fold changes, not just a count.",
    )


def rubric_evolution(truth: Path) -> str:
    rows = read_csv(truth / "variants_shared.csv")
    variants = [
        f"{r['CHROM']}:{r['POS']} {r['REF']}>{r['ALT']} ({r['GENE']}, {r['IMPACT']} {r['EFFECT']})" for r in rows
    ]
    return render(
        "evolution",
        "   Award the points only if at least one variant the agent reports as shared by both evolved lines\n"
        "   matches one of these on contig and position:\n"
        f"{bullet_list(variants)}\n"
        "   The contig name and position must both match. Reference/alternate alleles and the gene name are\n"
        "   corroborating detail and need not match exactly (assembly-dependent naming varies).",
        "   Both evolved lines called against the ancestor, the two call sets intersected to find shared\n"
        "   variants, and the survivors functionally annotated with a predicted impact.",
        "   Variant calls are quality-filtered before intersection, and the reported set is restricted to\n"
        "   moderate-or-higher predicted severity as the task asks.",
        "   The final answer lists the shared variants with contig, position, gene and impact.",
    )


def rubric_metagenomics(truth: Path) -> str:
    rows = read_csv(truth / "phylum_relative_abundances.csv")
    top = max(rows, key=lambda r: float(r["JP4D"] or 0) + float(r["JC1A"] or 0))
    ranked = [f"{r['Phylum']}: JP4D {float(r['JP4D']):.2f}%, JC1A {float(r['JC1A']):.2f}%" for r in rows[:8]]
    return render(
        "metagenomics",
        f"   Award the points only if BOTH hold: (a) the agent identifies {top['Phylum']} as the most abundant\n"
        "   phylum in both samples, and (b) the agent names at least two further phyla from the expected\n"
        "   profile below with abundances in broadly the right range (same order of magnitude):\n"
        f"{bullet_list(ranked)}\n"
        "   Phylum synonyms are acceptable (e.g. Proteobacteria for Pseudomonadota, Firmicutes for Bacillota).",
        "   Quality control, assembly, and taxonomic classification against the supplied kraken2 database, run\n"
        "   for both the control (JC1A) and fertilized (JP4D) samples, with abundances normalised to relative\n"
        "   frequencies rather than raw read counts.",
        "   The comparison between conditions accounts for the different sequencing depths of the two samples.",
        "   The final answer reports the per-phylum relative abundances for both samples and states which\n"
        "   phylum dominates.",
    )


def rubric_single_cell(truth: Path) -> str:
    rows = read_csv(truth / "all_clusters_de_genes.csv")
    cell_types = sorted({r["predicted_cell_type"] for r in rows})
    return render(
        "single-cell",
        "   Award the points only if the agent identifies at least one of these cell types in the muscle tissue\n"
        "   and supports it with marker genes computed from the data:\n"
        f"{bullet_list(cell_types)}\n"
        "   DEVIATION FROM UPSTREAM: the benchmark's own scorer requires the cluster NUMBER to match too, but\n"
        "   cluster numbering is not reproducible across runs, so grade the cell-type identification and its\n"
        "   marker-gene evidence only, ignoring which cluster index it was assigned.",
        "   Quality control, normalisation, dimensionality reduction and clustering, then marker-based cell type\n"
        "   annotation, then differential expression between pre- and post-exercise within cell types. The six\n"
        "   samples (three subjects × two timepoints) must be combined coherently.",
        "   Cell type calls are justified by marker genes actually computed here, not asserted from prior\n"
        "   knowledge; batch/subject effects are at least acknowledged.",
        "   The final answer names the cell types found and describes the exercise response within them.",
    )


def rubric_transcript_quant(truth: Path) -> str:
    rows = [line.split() for line in (truth / "truth.tsv").read_text().splitlines() if line.strip()]
    pairs: list[tuple[str, int]] = [(t, int(c)) for t, c in rows]
    total = sum(c for _, c in pairs)
    # Fixed deterministic sample (every Nth of the sorted set) so the rubric is reproducible.
    sample = sorted(pairs)[:: max(len(pairs) // 10, 1)][:10]
    return render(
        "transcript-quant",
        "   DEVIATION FROM UPSTREAM: the benchmark requires all "
        f"{len(pairs)} transcript counts to match exactly, which\n"
        "   cannot be checked reliably by reading a notebook. Award the points only if BOTH hold:\n"
        f"     (a) the total number of reads assigned across all transcripts is within 2% of {total:,}; and\n"
        "     (b) the counts reported for these transcripts each match within 5%:\n"
        f"{bullet_list([f'{t}: {c}' for t, c in sample])}\n"
        "   If the agent reports quantification output without these specific values being checkable, award 0.",
        "   Selective-alignment or pseudoalignment quantification (salmon, kallisto or equivalent) against the\n"
        "   supplied transcriptome, with the paired-end reads supplied as a pair rather than independently.",
        "   The index is built from the provided transcriptome.fa, and counts (not TPM) are reported, since the\n"
        "   data is simulated and exact read assignment is the point.",
        "   The final answer reports per-transcript counts and the total assigned.",
    )


def rubric_viral_metagenomics(truth: Path) -> str:
    rows = read_csv(truth / "taxonomy.csv")
    viral = [r for r in rows if (r["domain"] or "").lower() == "viruses"]
    target = max(viral, key=lambda r: int(r["contig_count"]))
    others = [r["species"] for r in viral if r["species"] != target["species"]]
    return render(
        "viral-metagenomics",
        f"   Award the points only if the agent explicitly reports '{target['species']}' as a viral species\n"
        "   present in the sample, and presents it as the likely agent of the gastroenteritis. Naming it only\n"
        "   inside a raw tool dump, without the agent drawing the conclusion, does not count.\n"
        f"   For reference, the expected classification also contains {', '.join(others)} at much lower\n"
        f"   abundance ({target['contig_count']} contigs for the target versus "
        f"{viral[0]['contig_count'] if viral[0]['species'] != target['species'] else 'few'} for the other), so\n"
        "   naming only the low-abundance virus does not earn the points.",
        "   Read QC, removal of dolphin host sequence using the supplied host genome, assembly of the remaining\n"
        "   reads into contigs, and taxonomic classification of the contigs against the supplied viral database.",
        "   Host removal is actually performed rather than skipped, and classification is done at contig level\n"
        "   with counts reported per taxon.",
        "   The final answer names the viral species and gives per-species contig counts.",
    )


RUBRIC_BUILDERS: dict[str, Callable[[Path], str]] = {
    "alzheimer-mouse": rubric_alzheimer,
    "comparative-genomics": rubric_comparative_genomics,
    "cystic-fibrosis": rubric_cystic_fibrosis,
    "deseq": rubric_deseq,
    "evolution": rubric_evolution,
    "metagenomics": rubric_metagenomics,
    "single-cell": rubric_single_cell,
    "transcript-quant": rubric_transcript_quant,
    "viral-metagenomics": rubric_viral_metagenomics,
}

# What the agent must state in its final answer, per task. Derived from the deterministic check, but
# phrased so it never reveals the answer — asking "report chromosome, position, ref, alt" tells the
# agent nothing about WHICH variant is causal.
ANSWER_REQUIREMENTS = {
    "alzheimer-mouse": "the KEGG pathways shared across all three models, with the p-value for each pathway in each model",
    "comparative-genomics": "each co-evolving cluster with its consensus annotation, using KEGG orthology K-numbers",
    "cystic-fibrosis": "the causal variant: chromosome, position, reference and alternate alleles, gene, transcript, predicted effect and impact",
    "deseq": "the genes significantly up-regulated in biofilm relative to planktonic, with log2 fold change and adjusted p-value",
    "evolution": "each variant shared by both evolved lines: contig, position, reference and alternate alleles, gene, predicted effect and impact",
    "metagenomics": "the relative abundance of each phylum in both samples, and which phylum dominates",
    "single-cell": "the cell types identified with the marker genes supporting each, and the differentially expressed genes per cell type between pre- and post-exercise",
    "transcript-quant": "the per-transcript read counts and the total number of reads assigned",
    "viral-metagenomics": "the viral species identified with the number of contigs assigned to each, and which is the likely agent",
}


# Upstream's task_prompt ends with an <example> block showing the required output format. For four
# tasks that example is a VERBATIM ROW FROM THE TRUTH FILE, which hands the agent a free pass: under
# upstream's own scoring.py, echoing the example row alone satisfies the check for alzheimer-mouse,
# comparative-genomics and evolution (each needs only one matching value). These replacements keep
# the column layout and value shapes but use obviously-synthetic content, matching what upstream
# already does correctly for cystic-fibrosis, transcript-quant and viral-metagenomics.
EXAMPLE_OVERRIDES = {
    "alzheimer-mouse": (
        "Pathway,5xFAD_pvalue,3xTG_AD_pvalue,PS3O1S_pvalue\n"
        "Example pathway name Homo sapiens hsa00000,1.2345e-08,0.0234567890123456,0.4567890123456789"
    ),
    "comparative-genomics": (
        "cluster_number,consensus_annotation\n"
        "1,K00000  exmA, example gene product description\n"
        "2,K00001  exmB, exmC, second example product [EC:0.0.0.0]"
    ),
    "evolution": (
        "CHROM,POS,REF,ALT,GENE,IMPACT,EFFECT,STATUS\n"
        "NODE_1_length_50000_cov_5.000000,12345,A,G,EXAMPLE_00001,MODERATE,missense_variant,shared"
    ),
    "metagenomics": ("OTU,Kingdom,Phylum,JP4D,JC1A\n0000000,Bacteria,Examplephylum,12.3456789012345,6.78901234567890"),
    # Upstream relabelled the cell type here ("Endothelial cell" -> "Perivascular cell") but left the
    # gene, fold change and both p-values verbatim from truth row 1.
    "single-cell": (
        "cluster_id,predicted_cell_type,gene_name,logfoldchanges,pvals,pvals_adj,direction,abs_logfc\n"
        "0,Example cell type,EXGENE1,-1.234567,1.234567890123456e-10,2.345678901234567e-08,down,1.234567"
    ),
}

# Categorical vocabulary that legitimately appears in a format example without revealing which
# specific result is correct — snpEff impact/effect classes, taxonomic kingdoms, direction labels.
# Deliberately excludes "stop_gained": that one IS the cystic-fibrosis answer.
GENERIC_TOKENS = {
    "moderate",
    "high",
    "low",
    "modifier",
    "missense_variant",
    "frameshift_variant",
    "bacteria",
    "archaea",
    "eukaryota",
    "viruses",
    "unclassified",
    "shared",
    "up",
    "down",
}


def sanitize_task_prompt(task: dict[str, Any]) -> str:
    """Return the task prompt with any answer-revealing <example> block swapped for a synthetic one."""
    prompt = task["task_prompt"].strip()
    replacement = EXAMPLE_OVERRIDES.get(task["task_id"])
    if replacement is None:
        return prompt
    return re.sub(r"<example>.*?</example>", f"<example>\n{replacement}\n</example>", prompt, flags=re.DOTALL)


def truth_tokens(truth_dir: Path) -> set[str]:
    """Distinctive cell values from a task's truth files, for the leak guard.

    Short and purely numeric values are dropped: they collide with ordinary prose ("7", "Bacteria")
    and would make the guard cry wolf on every task.
    """
    tokens: set[str] = set()
    for path in sorted(truth_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh, delimiter=delimiter))
        # Skip the header row: column names are specified in the task prompt by design, so they are
        # not leaks — flagging them would bury the real hits.
        for row in rows[1:]:
            for cell in row:
                cell = cell.strip().strip('"')
                if cell.casefold() in GENERIC_TOKENS:
                    continue
                if (len(cell) >= 8 and not cell.replace(".", "").isdigit()) or len(cell) >= 10:
                    tokens.add(cell)
    return tokens


def check_leaks(problem: dict[str, Any], truth_dir: Path) -> list[str]:
    """Truth values that reached the agent-facing text. Must stay empty — see EXAMPLE_OVERRIDES.

    Matches on word boundaries so a generic prefix does not trip on a longer synthetic value
    (upstream's "Likely_pathogenic" placeholder must not read as the truth value "Pathogenic").
    """
    agent_facing = f"{problem['hypothesis']}\n{problem['protocol']}"
    return sorted(
        tok
        for tok in truth_tokens(truth_dir)
        if re.search(rf"(?<!\w){re.escape(tok)}(?!\w)", agent_facing, re.IGNORECASE)
    )


def build_protocol(task: dict[str, Any], manifest_entry: dict[str, Any]) -> str:
    """The <objectives> block: background plus the files actually staged in this capsule."""
    parts = [f"## Data Background\n{task['description'].strip()}"]

    sections = []
    for subdir, label in (("data", "Input data (`data/`)"), ("reference", "Reference data (`reference/`)")):
        files = manifest_entry.get("capsule", {}).get(subdir, [])
        if files:
            listed = files[:40]
            more = f"\n- ... and {len(files) - len(listed)} more file(s)" if len(files) > len(listed) else ""
            sections.append(f"### {label}\n" + "\n".join(f"- `{subdir}/{f}`" for f in listed) + more)
    if sections:
        parts.append("## Data Files\nThese files are in your working directory:\n\n" + "\n\n".join(sections))

    return "\n\n".join(parts)


def build_hypothesis(task: dict[str, Any]) -> str:
    """The <question> block: upstream's task prompt plus what the final answer must state."""
    task_id = task["task_id"]
    return (
        f"{sanitize_task_prompt(task)}\n\n"
        f"State in your final answer: {ANSWER_REQUIREMENTS[task_id]}.\n\n"
        "Also save the table described above to `results/` in your working directory."
    )


def build_problem(task: dict[str, Any], manifest_entry: dict[str, Any], truth_dir: Path) -> dict[str, Any]:
    task_id = task["task_id"]
    rubric = RUBRIC_BUILDERS[task_id](truth_dir)
    return {
        "id": str(uuid.uuid5(UUID_NAMESPACE, task_id)),
        "hypothesis": build_hypothesis(task),
        "protocol": build_protocol(task, manifest_entry),
        "answer": None,
        "rubric": rubric,
        "max_points": MAX_SCORE,
        "input_data_path": task_id,
        "task_style": "question",
        # Scored deterministically by hypotest/env/judges/bioagent.py — a transcription of
        # upstream's scoring.py, run against _truth/<task_id>/ with no LLM in the loop.
        "judge": "bioagent",
        "nb_primary_language": "python",
        "metadata": {
            "source": REPO_ID,
            "task_id": task_id,
            "name": task["name"],
            # Upstream's deterministic pass condition, kept alongside so criterion 1 can be audited
            # against the thing it is meant to reproduce.
            "upstream_result_rule": RESULT_RULES[task_id],
            "upstream_points": UPSTREAM_POINTS,
            "truth_files": manifest_entry.get("truth", []),
            "deviates_from_upstream": task_id in {"single-cell", "transcript-quant"},
            "example_sanitized": task_id in EXAMPLE_OVERRIDES,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capsule-dir", type=Path, default=ROOT / "capsules" / "bioagent-bench")
    ap.add_argument("--metadata", type=Path, default=None, help="Local task_metadata.json (default: fetch)")
    ap.add_argument("--out-jsonl", type=Path, default=ROOT / "problems_bioagent_bench.jsonl")
    ap.add_argument("--tasks", nargs="*", default=None, help="Only these task_ids")
    ap.add_argument("--print-rubric", default=None, help="Print one task's rubric and exit")
    args = ap.parse_args()

    capsule_dir: Path = args.capsule_dir
    manifest_path = capsule_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"missing {manifest_path} — run scripts/data/stage_bioagent_capsules.py first")
    manifest = json.loads(manifest_path.read_text())["tasks"]

    tasks = [t for t in load_env_metadata(args.metadata) if t["task_id"] not in SKIP_TASKS]
    if args.tasks:
        wanted = set(args.tasks)
        unknown = wanted - {t["task_id"] for t in tasks}
        if unknown:
            sys.exit(f"unknown or skipped task_id(s): {sorted(unknown)}")
        tasks = [t for t in tasks if t["task_id"] in wanted]

    if args.print_rubric:
        truth_dir = capsule_dir / TRUTH_DIR_NAME / args.print_rubric
        print(RUBRIC_BUILDERS[args.print_rubric](truth_dir))
        return

    problems, skipped = [], []
    for task in tasks:
        task_id = task["task_id"]
        truth_dir = capsule_dir / TRUTH_DIR_NAME / task_id
        if task_id not in manifest:
            skipped.append(f"{task_id} (not staged)")
            continue
        if not truth_dir.is_dir():
            skipped.append(f"{task_id} (no truth files)")
            continue
        try:
            problem = build_problem(task, manifest[task_id], truth_dir)
            leaked = check_leaks(problem, truth_dir)
            if leaked:
                # Never emit a problem whose prompt contains its own answer.
                skipped.append(f"{task_id} (LEAK: {leaked[:3]})")
                print(f"[LEAK] {task_id}: truth values in agent-facing text: {leaked[:3]}")
                continue
            problems.append(problem)
            print(f"[ok] {task_id}" + ("  (example sanitized)" if task_id in EXAMPLE_OVERRIDES else ""))
        except Exception as exc:
            skipped.append(f"{task_id} ({type(exc).__name__}: {exc})")

    with args.out_jsonl.open("w") as fh:
        for problem in problems:
            fh.write(json.dumps(problem) + "\n")

    print(f"\nWrote {len(problems)} problems -> {args.out_jsonl}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    print(f"Not converted (needs the file-based scorer): {', '.join(sorted(SKIP_TASKS))}")
    print(
        "\nserver.yaml:\n"
        f"  capsule_dir: {capsule_dir}/\n"
        f"  problem_jsonl: {args.out_jsonl.name}\n"
        "  include_protocol: true\n"
        "  judge: hypotest\n"
    )


if __name__ == "__main__":
    main()
