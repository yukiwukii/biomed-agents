#!/usr/bin/env python3
"""Stage bioagent-bench/bioagent-bench task data into hypotest's capsule layout.

bioagent-bench keeps no data in git: ``src/task_metadata.json`` lists, per task, a set of
OSF download URLs split into three categories:

  - ``data``            the agent's inputs (FASTQs, count matrices, VCFs, ...)
  - ``reference_data``  large shared reference resources (kraken2 DB, GRCh38, ClinVar, ...)
  - ``results``         the TRUTH FILES the upstream scorer compares against

This script downloads all three and lays them out as::

    capsules/bioagent-bench/
        <task_id>/data/...        <- copied into the agent's work_dir by dataset_server
        <task_id>/reference/...   <-          "
        _truth/<task_id>/...      <- NEVER copied to the agent; source for rubric ground truth

The split matters. ``Dataset.get_new_env_by_idx`` (src/hypotest/dataset_server.py) does
``copytree(capsule_dir / problem.input_data_path, work_dir)``, so *everything* under
``<task_id>/`` reaches the agent. Truth files therefore live in a sibling ``_truth/``
directory that no ``input_data_path`` ever points at. Keep it that way: staging results
inside a task dir silently invalidates every score derived from it.

The ``data/`` + ``reference/`` split is preserved (rather than flattened into the capsule
root) because upstream's own task prompts say "use the files in `data/` and `reference/`",
and its reproduction scripts reference those paths.

Extraction mirrors upstream ``src/dataset.py::_extract_tarfile`` exactly — archive members
are flattened to their basename, except the two prefixes that upstream gives their own
subdirectory (``k2_standard_*``, ``kaiju_db_*``) because their tools expect a DB directory.
Deviating here would produce paths the task prompts do not describe.

Downloads are resumable at file granularity: each category directory gets a ``.staged.json``
marker listing what landed, and a re-run skips anything already recorded. Partial downloads
go to ``<name>.part`` and are only renamed on success, so an interrupted run never leaves a
truncated archive that looks complete.

Usage:
    # everything (~33 GB+ over the wire, smallest tasks first)
    .venv/bin/python scripts/stage_bioagent_capsules.py

    # just the three cheap tasks, to get a pipeline working end to end
    .venv/bin/python scripts/stage_bioagent_capsules.py --tasks transcript-quant single-cell alzheimer-mouse

    # see the plan and per-file sizes without downloading
    .venv/bin/python scripts/stage_bioagent_capsules.py --dry-run

    # skip the 12 GB kraken2 DB and friends
    .venv/bin/python scripts/stage_bioagent_capsules.py --skip-reference
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent

REPO_ID = "bioagent-bench/bioagent-bench"
METADATA_URL = f"https://raw.githubusercontent.com/{REPO_ID}/master/src/task_metadata.json"

# Category -> subdirectory of the capsule. `results` is deliberately absent: it is the answer
# key and is routed to _truth/<task_id>/ instead (see module docstring).
AGENT_VISIBLE = {"data": "data", "reference_data": "reference"}
TRUTH_CATEGORY = "results"
TRUTH_DIR_NAME = "_truth"

# Archives whose members upstream extracts into a named subdirectory rather than flattening
# into the category root, because the tools that read them expect a database directory.
DB_ARCHIVE_PREFIXES = ("k2_standard_16gb_20241228", "kaiju_db_viruses_2024-08-15")

CHUNK = 1 << 20  # 1 MiB
DOWNLOAD_RETRIES = 3


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_metadata(path: Path | None) -> list[dict[str, Any]]:
    if path is not None:
        return json.loads(path.read_text())
    resp = requests.get(METADATA_URL, timeout=60)
    resp.raise_for_status()
    return resp.json()


def probe_size(url: str) -> int:
    """Content-Length for a URL, or 0 when OSF declines to report one (chunked responses)."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=30)
        return int(resp.headers.get("Content-Length") or 0)
    except requests.RequestException:
        return 0


def marker_path(dest: Path) -> Path:
    return dest / ".staged.json"


def read_marker(dest: Path) -> dict[str, Any]:
    path = marker_path(dest)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_marker(dest: Path, filename: str, entry: dict[str, Any]) -> None:
    marker = read_marker(dest)
    marker[filename] = entry
    marker_path(dest).write_text(json.dumps(marker, indent=2, sort_keys=True))


def download(url: str, target: Path) -> None:
    """Stream `url` to `target`, via a .part file so an interrupted run leaves no half-archive."""
    part = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length") or 0)
                written = 0
                start = time.monotonic()
                next_report = start + 15
                with part.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
                        now = time.monotonic()
                        if now >= next_report:
                            rate = written / max(now - start, 1e-6)
                            pct = f" ({100 * written / total:.0f}%)" if total else ""
                            print(f"      {human(written)}{pct} @ {human(rate)}/s", flush=True)
                            next_report = now + 15
            part.rename(target)
            return
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt < DOWNLOAD_RETRIES:
                print(f"      retry {attempt}/{DOWNLOAD_RETRIES - 1} after {type(exc).__name__}: {exc}", flush=True)
                time.sleep(5 * attempt)

    raise RuntimeError(f"failed to download {url}") from last_error


def extract(archive: Path, category_dir: Path) -> list[str]:
    """Extract `archive` the way upstream's dataset.py does; return the member basenames written.

    Members are flattened to their basename (upstream: ``member.name = Path(member.name).name``)
    so task prompts can reference bare filenames. Database archives get their own subdirectory.
    """
    subdir = next((p for p in DB_ARCHIVE_PREFIXES if archive.name.startswith(p)), None)
    out_dir = category_dir / subdir if subdir else category_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    seen: set[str] = set()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            member.name = Path(member.name).name
            if member.name in seen:
                # Upstream flattening is lossy for nested archives; surface it rather than
                # silently letting one member clobber another.
                print(f"      WARNING: duplicate basename in {archive.name}: {member.name}", flush=True)
            seen.add(member.name)
            tar.extract(member, path=out_dir)  # noqa: S202 - names flattened to basenames above
            written.append(str((out_dir / member.name).relative_to(category_dir)))
    return sorted(written)


def stage_item(item: dict[str, str], category_dir: Path, dry_run: bool) -> dict[str, Any] | None:
    """Download + extract one metadata entry into `category_dir`. Returns its marker entry."""
    filename, url = item["filename"], item["url"]
    marker = read_marker(category_dir)
    if filename in marker:
        print(f"    [skip] {filename} (already staged)", flush=True)
        return None

    if dry_run:
        print(f"    [plan] {filename}  {human(probe_size(url))}", flush=True)
        return None

    category_dir.mkdir(parents=True, exist_ok=True)
    archive = category_dir / filename
    print(f"    [get ] {filename}", flush=True)
    download(url, archive)

    entry: dict[str, Any] = {"url": url, "bytes": archive.stat().st_size}
    if tarfile.is_tarfile(archive):
        entry["files"] = extract(archive, category_dir)
        archive.unlink()
        print(f"    [ok  ] {filename} -> {len(entry['files'])} file(s)", flush=True)
    else:
        # Not an archive (e.g. clinvar_*.vcf.gz) — keep as-is.
        entry["files"] = [filename]
        print(f"    [ok  ] {filename} ({human(entry['bytes'])})", flush=True)

    write_marker(category_dir, filename, entry)
    return entry


def stage_task(task: dict[str, Any], capsule_root: Path, args: argparse.Namespace) -> None:
    task_id = task["task_id"]
    urls = task["download_urls"]

    print(f"\n=== {task_id} — {task['name']}", flush=True)

    for category, subdir in AGENT_VISIBLE.items():
        items = urls.get(category) or []
        if category == "reference_data" and args.skip_reference:
            if items:
                print(f"  [skip] reference_data ({len(items)} item(s), --skip-reference)", flush=True)
            continue
        if not items:
            continue
        print(f"  {subdir}/", flush=True)
        for item in items:
            stage_item(item, capsule_root / task_id / subdir, args.dry_run)

    if args.skip_truth:
        return
    items = urls.get(TRUTH_CATEGORY) or []
    if items:
        # Answer key. Sibling of the task capsules, never inside one.
        print(f"  {TRUTH_DIR_NAME}/{task_id}/  (ground truth — not visible to the agent)", flush=True)
        for item in items:
            stage_item(item, capsule_root / TRUTH_DIR_NAME / task_id, args.dry_run)


def write_manifest(capsule_root: Path, tasks: list[dict[str, Any]]) -> Path:
    """Record what actually landed per task — the converter builds its data-file manifest from this.

    Scans the capsule root rather than only the tasks staged in this run: the manifest is rewritten
    every time, so building it from the (possibly ``--tasks``-filtered) run would silently drop
    every task staged by an earlier invocation.
    """
    manifest: dict[str, Any] = {"source": REPO_ID, "tasks": {}}
    for task in tasks:
        task_id = task["task_id"]
        task_dir = capsule_root / task_id
        if not task_dir.is_dir():
            continue
        entry: dict[str, Any] = {"name": task["name"], "capsule": {}, "truth": []}
        for subdir in ("data", "reference"):
            d = task_dir / subdir
            if d.is_dir():
                entry["capsule"][subdir] = sorted(
                    str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() and p.name != ".staged.json"
                )
        truth_dir = capsule_root / TRUTH_DIR_NAME / task_id
        if truth_dir.is_dir():
            entry["truth"] = sorted(
                str(p.relative_to(truth_dir)) for p in truth_dir.rglob("*") if p.is_file() and p.name != ".staged.json"
            )
        manifest["tasks"][task_id] = entry

    path = capsule_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capsule-dir", type=Path, default=ROOT / "capsules" / "bioagent-bench")
    ap.add_argument("--metadata", type=Path, default=None, help="Local task_metadata.json (default: fetch from GitHub)")
    ap.add_argument("--tasks", nargs="*", default=None, help="Only these task_ids (default: all)")
    ap.add_argument("--skip-reference", action="store_true", help="Skip reference_data (the multi-GB shared DBs)")
    ap.add_argument("--skip-truth", action="store_true", help="Skip the results/ truth files")
    ap.add_argument("--dry-run", action="store_true", help="Probe sizes and print the plan, download nothing")
    ap.add_argument("--largest-first", action="store_true", help="Default is smallest-first (quick wins early)")
    ap.add_argument(
        "--manifest-only",
        action="store_true",
        help="Rewrite manifest.json from what is already on disk; download nothing",
    )
    args = ap.parse_args()

    all_tasks = load_metadata(args.metadata)
    tasks = list(all_tasks)

    if args.manifest_only:
        manifest = write_manifest(args.capsule_dir, all_tasks)
        print(f"Manifest -> {manifest} ({len(json.loads(manifest.read_text())['tasks'])} task(s))")
        return

    if args.tasks:
        wanted = set(args.tasks)
        unknown = wanted - {t["task_id"] for t in tasks}
        if unknown:
            sys.exit(f"unknown task_id(s): {sorted(unknown)}")
        tasks = [t for t in tasks if t["task_id"] in wanted]

    capsule_root: Path = args.capsule_dir
    capsule_root.mkdir(parents=True, exist_ok=True)

    print(f"Probing download sizes for {len(tasks)} task(s)...", flush=True)
    sizes: dict[str, int] = {}
    for task in tasks:
        total = 0
        for category, items in task["download_urls"].items():
            if category == "reference_data" and args.skip_reference:
                continue
            if category == TRUTH_CATEGORY and args.skip_truth:
                continue
            total += sum(probe_size(item["url"]) for item in items)
        sizes[task["task_id"]] = total
        print(f"  {task['task_id']:22s} {human(total):>10s}", flush=True)

    known = sum(sizes.values())
    print(f"\nTotal (where Content-Length was reported): {human(known)}", flush=True)
    print(f"Capsules -> {capsule_root}", flush=True)

    tasks.sort(key=lambda t: sizes[t["task_id"]], reverse=args.largest_first)

    failures: list[str] = []
    for task in tasks:
        try:
            stage_task(task, capsule_root, args)
        except Exception as exc:  # noqa: BLE001 — one bad task must not abort the rest
            print(f"  [ERROR] {task['task_id']}: {type(exc).__name__}: {exc}", flush=True)
            failures.append(task["task_id"])

    if not args.dry_run:
        # Full task list, not the filtered one — see write_manifest().
        manifest = write_manifest(capsule_root, all_tasks)
        print(f"\nManifest -> {manifest}", flush=True)
        used = shutil.disk_usage(capsule_root)
        print(f"On disk: {human(sum(p.stat().st_size for p in capsule_root.rglob('*') if p.is_file()))} "
              f"({human(used.free)} free)", flush=True)

    if failures:
        print(f"\nFAILED: {failures}  (re-run to resume; completed files are skipped)", flush=True)
        sys.exit(1)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
