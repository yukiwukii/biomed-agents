"""bioagent-bench's judge: a deterministic scorer, no LLM.

bioagent-bench ships no rubric. Its reward lives in a second repo
(``bioagent-experiments``, ``tasksets/bioagent/bioagent_taskset/scoring.py``): the agent writes a
table to ``results/``, a per-task Python function compares it to the truth files, and the reward
is ``float(deterministic_match)`` — binary, no partial credit. This module transcribes those
checks. The pass conditions are recorded verbatim upstream-side in
``scripts/convert_bioagent_bench.py::RESULT_RULES`` and per problem at
``metadata["upstream_result_rule"]``; each check here implements exactly one of them.

Registered with ``needs_model=False``, so a run over this dataset needs no rubric model at all.
The judge returns 0 or 1 out of 1 — ``JudgeResult.max_score`` overrides the problem's own
``max_points`` (still 10, from the LLM-rubric conversion), so reward normalizes to 0.0/1.0 and is
directly comparable to upstream's published pass/fail table.

Truth values are read from ``ctx.truth_dir`` at scoring time rather than pasted in here, so a
re-staged capsule and its scorer can never disagree. Stdlib only (``csv``) — no pandas — which
keeps the judges package importable by the offline grading scripts.

Matching policy: **tolerant on column naming, strict on values.** Agents will not reproduce the
truth files' headers, and upstream's own checks are value comparisons, so most checks intersect
normalized cell values rather than requiring a named column. Where a rule genuinely is about a
particular column (metagenomics' per-sample abundances, transcript-quant's id→count mapping) the
header is matched by alias, case-insensitively.
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from lmi import LiteLLMModel

from .base import JudgeContext, JudgeResult, judge

logger = logging.getLogger(__name__)

# (passed, human-readable detail)
CheckResult = tuple[bool, str]
Check = Callable[[list["Table"], Path], CheckResult]

# Where the agent is told to save its table (convert_bioagent_bench.build_hypothesis).
RESULTS_GLOBS = ("results/**/*.csv", "results/**/*.tsv", "results/**/*.txt")


class Table:
    """A parsed delimited file: normalized headers, rows as dicts, and per-row value sets."""

    def __init__(self, path: Path, headers: list[str], rows: list[list[str]]):
        self.path = path
        self.headers = [norm(h) for h in headers]
        self.rows = rows
        # Every normalized cell value in a row, for column-name-agnostic matching.
        self.row_values: list[set[str]] = [{norm(c) for c in r if norm(c)} for r in rows]

    def column(self, *aliases: str) -> list[str] | None:
        """Values of the first column whose header contains any alias (normalized)."""
        for alias in aliases:
            for i, h in enumerate(self.headers):
                if alias in h:
                    return [r[i].strip().strip('"') if i < len(r) else "" for r in self.rows]
        return None

    def pairs(self, a_aliases: Sequence[str], b_aliases: Sequence[str]) -> list[tuple[str, str]] | None:
        a, b = self.column(*a_aliases), self.column(*b_aliases)
        if a is None or b is None:
            return None
        return [(norm(x), norm(y)) for x, y in zip(a, b, strict=False)]

    def values(self) -> set[str]:
        return {v for s in self.row_values for v in s}


def norm(value: str) -> str:
    """Casefold, strip quotes, collapse internal whitespace. Values compare after this."""
    return re.sub(r"\s+", " ", str(value).strip().strip('"').strip()).casefold()


def read_table(path: Path) -> Table | None:
    """Parse a CSV/TSV. Delimiter is sniffed from the extension, then from the header line."""
    try:
        text = path.read_text(errors="replace")
    except OSError as e:  # unreadable file is not a scoring error, just a non-candidate
        logger.warning("could not read %s: %s", path, e)
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    delim = "\t" if (path.suffix == ".tsv" or lines[0].count("\t") > lines[0].count(",")) else ","
    rows = list(csv.reader(lines, delimiter=delim))
    if not rows:
        return None
    # Headerless files (transcript-quant's truth.tsv) still parse: the "header" is data, and the
    # checks that care read by column alias, which simply misses — so also keep row 0 as a row.
    return Table(path, rows[0], rows[1:])


def read_truth(truth_dir: Path, filename: str) -> Table:
    t = read_table(truth_dir / filename)
    if t is None:
        raise FileNotFoundError(f"missing or empty truth file: {truth_dir / filename}")
    return t


def collect_candidates(work_dir: Path) -> list[Table]:
    """Every table the agent wrote under ``results/``.

    The task prompt only says "save the table to `results/`" — no filename — so a check passes if
    *any* produced table satisfies it, matching upstream's "the generated CSV" leniency without
    coupling to a name the agent was never given.
    """
    paths: list[Path] = []
    for pattern in RESULTS_GLOBS:
        paths.extend(sorted(work_dir.glob(pattern)))
    tables = [read_table(p) for p in dict.fromkeys(paths)]
    return [t for t in tables if t is not None]


def _any_table(tables: Iterable[Table], predicate: Callable[[Table], CheckResult]) -> CheckResult:
    """First table that passes wins; otherwise report the best (last) failure detail."""
    detail = "no table under results/ satisfied the rule"
    for t in tables:
        passed, why = predicate(t)
        if passed:
            return True, f"{t.path.name}: {why}"
        detail = f"{t.path.name}: {why}"
    return False, detail


# ── per-task checks — one per RESULT_RULES entry ─────────────────────────────


def check_alzheimer_mouse(tables: list[Table], truth_dir: Path) -> CheckResult:
    """≥1 shared Pathway value between the generated CSV and the expected CSV."""
    truth = read_truth(truth_dir, "pathway_comparison.csv")
    expected = {norm(v) for v in (truth.column("pathway") or [])}

    def predicate(t: Table) -> CheckResult:
        hits = expected & t.values()
        return len(hits) >= 1, f"{len(hits)} of {len(expected)} expected pathways present (need >=1)"

    return _any_table(tables, predicate)


def check_comparative_genomics(tables: list[Table], truth_dir: Path) -> CheckResult:
    """≥1 consensus_annotation value matching the expected results exactly."""
    truth = read_truth(truth_dir, "cluster_annotation_mapping.csv")
    expected = {norm(v) for v in (truth.column("consensus_annotation", "annotation") or [])}

    def predicate(t: Table) -> CheckResult:
        hits = expected & t.values()
        return len(hits) >= 1, f"{len(hits)} of {len(expected)} expected annotations present (need >=1)"

    return _any_table(tables, predicate)


def check_cystic_fibrosis(tables: list[Table], truth_dir: Path) -> CheckResult:
    """The causal CFTR variant reported exactly once, with all ten fields matching."""
    truth = read_truth(truth_dir, "cf_variants.csv")
    fields = (
        "chromosome",
        "position",
        "variant_id",
        "reference",
        "alternate",
        "gene_name",
        "gene_id",
        "annotation",
        "impact",
        "transcript_id",
    )
    required: set[str] = set()
    for f in fields:
        col = truth.column(f)
        if col:
            required.add(norm(col[0]))
    # "chr7" and "7" are the same chromosome; accept either spelling of that one value.
    chrom = norm((truth.column("chromosome") or ["7"])[0])

    def predicate(t: Table) -> CheckResult:
        matches = 0
        for values in t.row_values:
            enriched = values | {v.removeprefix("chr") for v in values}
            if required - {chrom} <= enriched and (chrom in enriched or f"chr{chrom}" in values):
                matches += 1
        return matches == 1, f"{matches} row(s) carry all {len(required)} variant fields (need exactly 1)"

    return _any_table(tables, predicate)


def check_deseq(tables: list[Table], truth_dir: Path) -> CheckResult:
    """≥5 gene_id values overlapping the expected DESeq output."""
    truth = read_truth(truth_dir, "up_regulated_genes.csv")
    expected = {norm(v) for v in (truth.column("gene_id", "gene") or [])}

    def predicate(t: Table) -> CheckResult:
        hits = expected & t.values()
        return len(hits) >= 5, f"{len(hits)} of {len(expected)} expected gene_ids present (need >=5)"

    return _any_table(tables, predicate)


def check_evolution(tables: list[Table], truth_dir: Path) -> CheckResult:
    """≥1 (CHROM, POS) pair matching the expected shared variants."""
    truth = read_truth(truth_dir, "variants_shared.csv")
    expected = set(truth.pairs(["chrom", "contig"], ["pos"]) or [])

    def predicate(t: Table) -> CheckResult:
        hits = {(c, p) for c, p in expected if {c, p} <= set().union(*t.row_values) and _same_row(t, c, p)}
        return len(hits) >= 1, f"{len(hits)} of {len(expected)} expected (CHROM, POS) pairs present (need >=1)"

    return _any_table(tables, predicate)


def check_metagenomics(tables: list[Table], truth_dir: Path) -> CheckResult:
    """Pseudomonadota most abundant in both samples, plus ≥2 further OTUs with expected Phylum."""
    truth = read_truth(truth_dir, "phylum_relative_abundances.csv")
    expected_pairs = set(truth.pairs(["otu"], ["phylum"]) or [])
    dominant = {}
    for sample in ("jp4d", "jc1a"):
        col, phyla = truth.column(sample), truth.column("phylum")
        if col and phyla:
            dominant[sample] = norm(max(zip(phyla, col, strict=False), key=lambda kv: _as_float(kv[1]))[0])

    def predicate(t: Table) -> CheckResult:
        phyla = t.column("phylum")
        if not phyla:
            return False, "no phylum column"
        for sample, expected_top in dominant.items():
            col = t.column(sample)
            if not col:
                return False, f"no {sample} abundance column"
            top = norm(max(zip(phyla, col, strict=False), key=lambda kv: _as_float(kv[1]))[0])
            if top != expected_top:
                return False, f"{sample} dominated by {top!r}, expected {expected_top!r}"
        extra = {(a, b) for a, b in (t.pairs(["otu"], ["phylum"]) or []) if (a, b) in expected_pairs}
        return len(extra) >= 2, (
            f"dominant phylum correct in both samples; {len(extra)} matching OTU/Phylum rows (need >=2)"
        )

    return _any_table(tables, predicate)


def check_single_cell(tables: list[Table], truth_dir: Path) -> CheckResult:
    """≥1 (cluster_id, predicted_cell_type) pair matching the expected results.

    Upstream requires the cluster NUMBER to match too. The LLM rubric deliberately relaxes this
    (cluster numbering is not stable across reseeds — see convert_bioagent_bench.py:41); this
    scorer restores upstream's exact condition, so it is the stricter of the two.
    """
    truth = read_truth(truth_dir, "all_clusters_de_genes.csv")
    expected = set(truth.pairs(["cluster_id", "cluster"], ["predicted_cell_type", "cell_type"]) or [])

    def predicate(t: Table) -> CheckResult:
        got = set(t.pairs(["cluster_id", "cluster"], ["predicted_cell_type", "cell_type", "celltype"]) or [])
        hits = got & expected
        if not got:
            return False, "no cluster_id / predicted_cell_type columns"
        return len(hits) >= 1, f"{len(hits)} of {len(expected)} expected (cluster, cell type) pairs (need >=1)"

    return _any_table(tables, predicate)


def check_transcript_quant(tables: list[Table], truth_dir: Path) -> CheckResult:
    """The complete transcript_id → count mapping equals the expected mapping.

    Upstream requires all 278 counts; the LLM rubric samples named transcripts instead
    (convert_bioagent_bench.py:46). This scorer restores upstream's exact condition.
    """
    truth = read_truth(truth_dir, "truth.tsv")
    # Headerless: row 0 is data, so fold the parsed header line back in.
    expected = {norm(r[0]): norm(r[1]) for r in [truth.headers, *truth.rows] if len(r) >= 2 and r[0]}

    def predicate(t: Table) -> CheckResult:
        rows = [t.headers, *t.rows]
        ids = t.column("transcript", "target_id", "name")
        counts = t.column("count", "reads", "est_counts", "numreads")
        if ids and counts:
            got = {norm(i): norm(c) for i, c in zip(ids, counts, strict=False) if i}
        else:  # headerless two-column output
            got = {norm(r[0]): norm(r[1]) for r in rows if len(r) >= 2 and r[0]}
        got = {k: _canon_count(v) for k, v in got.items() if k.startswith("enst")}
        want = {k: _canon_count(v) for k, v in expected.items()}
        if got == want:
            return True, f"all {len(want)} transcript counts match"
        missing = len(want.keys() - got.keys())
        wrong = sum(1 for k in want.keys() & got.keys() if want[k] != got[k])
        return False, f"{len(got)}/{len(want)} transcripts; {missing} missing, {wrong} with wrong counts"

    return _any_table(tables, predicate)


def check_viral_metagenomics(tables: list[Table], truth_dir: Path) -> CheckResult:
    """"Bottlenose dolphin adenovirus 1" explicitly reported under the Viruses domain."""
    truth = read_truth(truth_dir, "taxonomy.csv")
    species = [norm(s) for s in (truth.column("species") or [])]
    target = next((s for s in species if "adenovirus" in s), "bottlenose dolphin adenovirus 1")

    def predicate(t: Table) -> CheckResult:
        for values in t.row_values:
            if target in values and any("virus" in v for v in values):
                return True, f"{target!r} reported under the Viruses domain"
        found = any(target in v for v in t.values())
        return False, f"{target!r} {'present but not under Viruses' if found else 'not reported'}"

    return _any_table(tables, predicate)


CHECKS: dict[str, Check] = {
    "alzheimer-mouse": check_alzheimer_mouse,
    "comparative-genomics": check_comparative_genomics,
    "cystic-fibrosis": check_cystic_fibrosis,
    "deseq": check_deseq,
    "evolution": check_evolution,
    "metagenomics": check_metagenomics,
    "single-cell": check_single_cell,
    "transcript-quant": check_transcript_quant,
    "viral-metagenomics": check_viral_metagenomics,
}


def _same_row(t: Table, a: str, b: str) -> bool:
    return any(a in values and b in values for values in t.row_values)


def _as_float(value: str) -> float:
    try:
        return float(str(value).strip().strip('"'))
    except (TypeError, ValueError):
        return float("-inf")


def _canon_count(value: str) -> str:
    """'3278', '3278.0' and '3,278' are the same count."""
    v = value.replace(",", "")
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return value


@judge("bioagent", needs_model=False)
async def bioagent_judge(ctx: JudgeContext, model: LiteLLMModel | None) -> JudgeResult:
    """Deterministic pass/fail: run this task's check over whatever the agent wrote to results/."""
    task_id = str(ctx.problem.metadata.get("task_id", ""))
    rule = str(ctx.problem.metadata.get("upstream_result_rule", ""))

    check = CHECKS.get(task_id)
    if check is None:
        raise ValueError(f"no bioagent-bench check for task_id {task_id!r}; known: {sorted(CHECKS)}")
    if ctx.truth_dir is None or not ctx.truth_dir.exists():
        raise ValueError(f"bioagent judge needs a staged truth dir for {task_id!r}, got {ctx.truth_dir!r}")
    if ctx.work_dir is None:
        raise ValueError("bioagent judge needs work_dir to find the agent's results/")

    tables = collect_candidates(ctx.work_dir)
    if tables:
        passed, detail = check(tables, ctx.truth_dir)
    else:
        passed, detail = False, "the agent wrote no table under results/"

    return JudgeResult(
        raw_score=int(passed),
        max_score=1,
        correct=passed,
        criteria=[
            {
                "criterion": f"bioagent-bench deterministic check ({task_id})",
                "score": int(passed),
                "justification": detail,
                "rule": rule,
            }
        ],
        metadata={
            "task_id": task_id,
            "rule": rule,
            "candidates": [str(t.path) for t in tables],
            "detail": detail,
        },
    )
