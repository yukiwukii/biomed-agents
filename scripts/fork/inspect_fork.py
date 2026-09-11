#!/usr/bin/env python3
"""Inspect forked trajectories under a forks/ directory as a single HTML page.

Each fork is a sub-directory holding a one-element ``trajectories.pkl`` plus a
``fork_info.json`` with its metadata (fork cell, scores, injected feedback, regenerated
answer). This renders the same step-by-step view as ``inspect_trajectory.py`` but with a
fork-aware sidebar and a metadata header per panel: which steps were replayed verbatim
and are frozen, the evaluator guidance injected at the fork point, and the regenerated
answer.

Forking is single-shot, so ``<traj>-fork_cell<N>/`` directories each render as one panel.
The chain grouping below is retained for the *archived* runs made when forking was
sequential (``<traj>-rN_cell<K>/`` plus a ``<traj>-chain.json`` rollup, under
``archive/``): those nest their rounds beneath a chain header carrying the score trace and
stop reason, with a delta against the previous round as well as against the original run.
A single fork is simply a chain of one, so both layouts render from the same code.

Usage:
    python inspect_fork.py [forks_dir] [--html out.html]

    # the all-feedback ablation run:
    python scripts/fork/inspect_fork.py archive/fork_trial/forks-allfb-sonnet \
        --html forks-allfb-sonnet.html

Defaults: forks_dir=forks, out=forks.html
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

# Reuse the trajectory/rubric rendering from the general trajectory inspector.
from inspect_trajectory import (
    _HTML_CSS,
    _h,
    _html_result_text,
    _html_trajectory_body,
    load_trajectories,
)

# ── extra CSS for the fork metadata header ───────────────────────────────────

_FORK_CSS = """
.fork-meta {
    background: #0d1320;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 24px;
}
.fork-stats { display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 4px; }
.fork-stat { font-size: 0.82rem; color: #94a3b8; }
.fork-stat b { color: #f1f5f9; }
.score-up   { color: #4ade80; font-weight: 700; }
.score-down { color: #f87171; font-weight: 700; }
.score-same { color: #94a3b8; font-weight: 700; }
.fork-feedback {
    margin-top: 10px;
    padding: 10px 12px;
    border-radius: 5px;
    background: #1c1500;
    border: 1px solid #3b2e0a;
    color: #fbbf24;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 0.85rem;
}
.fork-answer-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #4ade80; margin: 14px 0 4px;
}
.fork-answer {
    padding: 10px 12px; border-radius: 5px;
    background: #0a1f12; border: 1px solid #166534;
    white-space: pre-wrap; word-break: break-word; font-size: 0.85rem;
}
.nav-delta { font-weight: 700; }

/* ── [PATCH 27] chain grouping ─────────────────────────────────────────────── */
.chain-head {
    padding: 12px 14px 8px;
    margin-top: 14px;
    border-top: 1px solid #1e293b;
}
.chain-name { font-size: 0.86rem; font-weight: 700; color: #f1f5f9; }
.chain-trace {
    font-size: 0.76rem; color: #94a3b8; margin-top: 4px;
    word-break: break-word; line-height: 1.5;
}
.chain-trace b { color: #f1f5f9; }
.stop-tag {
    display: inline-block; margin-left: 6px; padding: 1px 6px; border-radius: 3px;
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
    vertical-align: middle;
}
.stop-win   { background: #052e16; border: 1px solid #166534; color: #4ade80; }
.stop-done  { background: #0c2a3f; border: 1px solid #1e4f6d; color: #67c8f0; }
.stop-limit { background: #2a1c00; border: 1px solid #4d3708; color: #fbbf24; }
.nav-round { padding-left: 26px; }
.nav-round .nav-main { font-size: 0.8rem; }

/* clickable round strip at the top of each panel */
.round-strip { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; align-items: center; }
.round-pill {
    padding: 3px 9px; border-radius: 12px; font-size: 0.76rem; cursor: pointer;
    background: #0d1320; border: 1px solid #1e293b; color: #94a3b8;
}
.round-pill:hover { border-color: #334155; color: #cbd5e1; }
.round-pill.current { background: #1e293b; border-color: #475569; color: #f1f5f9; font-weight: 700; }
.round-arrow { color: #475569; font-size: 0.76rem; }
.frozen-note {
    margin-top: 10px; padding: 8px 12px; border-radius: 5px;
    background: #0b1220; border: 1px solid #1e293b; color: #94a3b8; font-size: 0.8rem;
}
.frozen-note b { color: #cbd5e1; }
"""

# Round dirs written by scripts/fork/fork_trajectory.py: `<root>-r<N>_cell<C>`.
_ROUND_RE = re.compile(r"^(?P<root>.+)-r(?P<round>\d+)_cell(?P<cell>\d+)$")
# Pre-Patch-27 single forks: `<root>-fork_cell<C>`. Rendered as a one-round chain.
_LEGACY_RE = re.compile(r"^(?P<root>.+)-fork_cell(?P<cell>\d+)$")

# Stop reasons → badge class. Green = the chain achieved something; blue = it ran out of
# things to fix; amber = it hit a limit with work still outstanding.
_STOP_CLASS = {
    "full_reward": "stop-win",
    "submitted": "stop-done",
    "no_wrong_step": "stop-done",
    "truncated": "stop-limit",
    "no_room": "stop-limit",
    "max_rounds": "stop-limit",  # only in runs from the chained-fork era; see archive/
    "cell_never_appended": "stop-limit",
}


def _natural_key(name: str):
    """Sort task_2 before task_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _fmt(v) -> str:
    """Two decimals for scores, plain str otherwise (None → '·')."""
    if v is None:
        return "·"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


@dataclass
class Round:
    """One fork round: its directory, its fork_info.json, and its trajectory."""

    name: str
    info: dict | None
    traj: Any
    root: str
    number: int
    panel_idx: int = -1  # assigned when the page is laid out


@dataclass
class Chain:
    """All rounds forked from one source trajectory, in round order."""

    root: str
    rounds: list[Round] = field(default_factory=list)
    info: dict | None = None  # <root>-chain.json, when fork_trajectory.py wrote one

    @property
    def stop_reason(self) -> str | None:
        if self.info and self.info.get("stop_reason"):
            return self.info["stop_reason"]
        # Fall back to the per-round copy; fork_trajectory.py stamps every round with it.
        for r in reversed(self.rounds):
            if r.info and r.info.get("stop_reason"):
                return r.info["stop_reason"]
        return None

    @property
    def trace(self) -> list:
        """[original score, round 1 score, round 2 score, …]."""
        if self.info and self.info.get("score_trace"):
            return list(self.info["score_trace"])
        first = self.rounds[0].info if self.rounds and self.rounds[0].info else {}
        return [(first or {}).get("old_score"), *((r.info or {}).get("new_score") for r in self.rounds)]


def find_forks(forks_dir: Path) -> list[Round]:
    """Return a Round for each fork sub-dir holding a trajectories.pkl, in name order."""
    rounds: list[Round] = []
    for sub in sorted(forks_dir.iterdir(), key=lambda p: _natural_key(p.name)):
        pkl = sub / "trajectories.pkl"
        if not (sub.is_dir() and pkl.exists()):
            continue
        info = None
        info_path = sub / "fork_info.json"
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text())
            except Exception:
                info = None
        trajs = load_trajectories(pkl)
        traj = trajs[0] if trajs else None
        if traj is None:
            continue
        # Prefer fork_info.json's own fields; fall back to parsing the directory name so a
        # round with an unreadable/absent fork_info.json still lands in the right chain.
        seq = _ROUND_RE.match(sub.name)
        m = seq or _LEGACY_RE.match(sub.name)
        root = (info or {}).get("source_traj_id") or (m.group("root") if m else sub.name)
        # Legacy single-fork dirs carry no round number; they are round 1 of a chain of one.
        number = (info or {}).get("round") or (int(seq.group("round")) if seq else 1)
        rounds.append(Round(name=sub.name, info=info, traj=traj, root=root, number=number))
    return rounds


def group_chains(rounds: list[Round], forks_dir: Path) -> list[Chain]:
    """Group rounds by source trajectory, ordered by round number, newest chain last."""
    chains: dict[str, Chain] = {}
    for r in rounds:
        chain = chains.setdefault(r.root, Chain(root=r.root))
        chain.rounds.append(r)
    for root, chain in chains.items():
        chain.rounds.sort(key=lambda r: r.number)
        path = forks_dir / f"{root}-chain.json"
        if path.exists():
            try:
                chain.info = json.loads(path.read_text())
            except Exception:
                chain.info = None
    return [chains[k] for k in sorted(chains, key=_natural_key)]


def _score_span(old, new, label: str = "") -> str:
    """Render `old → new` with up/down/same coloring."""
    if old is None and new is None:
        return '<span class="score-same">n/a</span>'
    cls = "score-same"
    if old is not None and new is not None:
        if new > old:
            cls = "score-up"
        elif new < old:
            cls = "score-down"
    return f'<span class="{cls}">{_h(_fmt(old))} → {_h(_fmt(new))}{_h(label)}</span>'


def _delta_dot(old, new) -> str:
    """Sidebar dot class: green if improved, red if regressed, gray if same/unknown."""
    if old is None or new is None:
        return "dot-bad"
    if new > old:
        return "dot-good"
    if new < old:
        return "dot-bad"
    return "dot-bad"


def _stop_tag(reason: str | None) -> str:
    if not reason:
        return ""
    return f'<span class="stop-tag {_STOP_CLASS.get(reason, "stop-limit")}">{_h(reason)}</span>'


def _trace_html(trace: list, current: int | None = None) -> str:
    """`0.30 → 0.45 → 1.00`, bolding the entry for round ``current`` (1-based)."""
    if not trace:
        return ""
    parts = []
    for i, v in enumerate(trace):
        s = _h(_fmt(v))
        parts.append(f"<b>{s}</b>" if current is not None and i == current else s)
    return " → ".join(parts)


def _round_strip_html(chain: "Chain", current: Round) -> str:
    """Clickable pills for every round in the chain, with the current one highlighted."""
    if len(chain.rounds) <= 1:
        return ""
    pills = []
    for r in chain.rounds:
        cell = (r.info or {}).get("fork_cell")
        cls = "round-pill current" if r is current else "round-pill"
        pills.append(
            f'<span class="{cls}" onclick="selectTraj({r.panel_idx})">r{r.number} · cell {_h(_fmt(cell))}</span>'
        )
    # Joined outside the f-string: a backslash-escaped quote inside an f-string expression
    # is a SyntaxError before Python 3.12, and CI still builds on 3.11.
    sep = '<span class="round-arrow">→</span>'
    return f'<div class="round-strip">{sep.join(pills)}</div>'


def _fork_meta_html(info, chain: "Chain | None" = None, rnd: "Round | None" = None) -> str:
    """Render the fork metadata header (scores, fork cell, feedback, new answer)."""
    if not info:
        return ""
    old, new = info.get("old_score"), info.get("new_score")
    stats = [
        f'<span class="fork-stat">source: <b>{_h(info.get("source_traj_id", "?"))}</b></span>',
        f'<span class="fork-stat">fork cell: <b>{_h(info.get("fork_cell"))}</b></span>',
        f'<span class="fork-stat">vs. original: {_score_span(old, new)}</span>',
        f'<span class="fork-stat">replayed/generated: <b>{_h(info.get("n_replayed"))}</b>'
        f"+<b>{_h(info.get('n_generated'))}</b> of {_h(info.get('n_steps_total'))}</span>",
    ]
    # [PATCH 27] Chain-specific stats. `fork_floor` is what the previous round's fork cell
    # was — the boundary below which this round's grade could not flag anything.
    if rnd is not None and chain is not None and len(chain.rounds) > 1:
        prev = next((r for r in chain.rounds if r.number == rnd.number - 1), None)
        prev_score = (prev.info or {}).get("new_score") if prev else old
        stats.insert(2, f'<span class="fork-stat">round: <b>{rnd.number}</b> of {len(chain.rounds)}</span>')
        stats.insert(4, f'<span class="fork-stat">vs. previous round: {_score_span(prev_score, new)}</span>')
    floor = info.get("fork_floor")
    if floor is not None:
        stats.append(f'<span class="fork-stat">floor: <b>{_h(floor)}</b></span>')
    if info.get("submitted") is False:
        stats.append('<span class="fork-stat"><b class="score-down">never submitted</b></span>')
    fws_old = info.get("old_first_wrong_step")
    fws_new = info.get("new_first_wrong_step")
    if fws_old is not None or fws_new is not None:
        stats.append(f'<span class="fork-stat">first wrong step: <b>{_h(fws_old)}</b> → <b>{_h(fws_new)}</b></span>')

    parts = []
    if chain is not None and rnd is not None:
        parts.append(_round_strip_html(chain, rnd))
    parts.append(f'<div class="fork-stats">{"".join(stats)}</div>')

    # Spell out what is replayed vs. generated — the single most confusing thing about
    # reading a forked notebook, and the reason first_wrong_step is floored.
    n_replayed, cell = info.get("n_replayed"), info.get("fork_cell")
    if n_replayed:
        parts.append(
            f'<div class="frozen-note">Steps <b>0–{_h(n_replayed - 1)}</b> were replayed verbatim from '
            f"<b>{_h(info.get('parent_traj_id') or info.get('source_traj_id') or 'the parent')}</b> and are frozen. "
            f"The policy took over at step <b>{_h(n_replayed)}</b> (notebook cell <b>{_h(cell)}</b>), "
            f"and this round's grade only flags wrong steps from that cell onward.</div>"
        )

    feedback = (info.get("injected_feedback") or "").strip()
    if feedback:
        parts.append(f'<div class="fork-feedback">{_h(feedback)}</div>')

    answer = (info.get("new_answer") or "").strip()
    if answer:
        parts.extend((
            '<div class="fork-answer-label">Regenerated answer</div>',
            f'<div class="fork-answer">{_html_result_text(answer)}</div>',
        ))

    return f'<div class="fork-meta">{"".join(parts)}</div>'


def write_html(forks_dir: Path, out: Path) -> None:
    rounds = find_forks(forks_dir)
    if not rounds:
        print(f"No forks with trajectories.pkl found under {forks_dir}")
        return
    chains = group_chains(rounds, forks_dir)

    # Panel indices are assigned chain-first so the sidebar order matches the page order.
    ordered = [r for c in chains for r in c.rounds]
    for i, r in enumerate(ordered):
        r.panel_idx = i

    nav_items, panels = [], []
    for chain in chains:
        trace = chain.trace
        nav_items.append(f"""
        <div class="chain-head">
            <div class="chain-name">{_h(chain.root)}{_stop_tag(chain.stop_reason)}</div>
            <div class="chain-trace">{_trace_html(trace)}</div>
        </div>""")

        for r in chain.rounds:
            info = r.info or {}
            i = r.panel_idx
            old, new = info.get("old_score"), info.get("new_score")
            # In a chain, the meaningful delta is against the previous round — that is what
            # this round actually changed. Round 1 falls back to the original run's score.
            prev = next((p for p in chain.rounds if p.number == r.number - 1), None)
            base = (prev.info or {}).get("new_score") if prev else old
            active = " active" if i == 0 else ""

            delta = f'<span class="nav-delta">{_score_span(base, new)}</span>'
            nav_items.append(f"""
        <button class="nav-item nav-round{active}" data-idx="{i}" onclick="selectTraj({i})">
            <span class="nav-dot {_delta_dot(base, new)}"></span>
            <span class="nav-main">r{r.number} · cell {_h(_fmt(info.get("fork_cell")))}</span>
            <span class="nav-sub">{len(r.traj.steps)} steps · {delta}</span>
        </button>""")

            head = (
                f"{_h(chain.root)} · round {r.number} of {len(chain.rounds)}" if len(chain.rounds) > 1 else _h(r.name)
            )
            meta = (
                f"index {i} &nbsp;·&nbsp; {len(r.traj.steps)} steps &nbsp;·&nbsp; {_h(r.name)} "
                f"&nbsp;·&nbsp; chain: {_trace_html(trace, current=r.number)}"
                f"{_stop_tag(chain.stop_reason) if r is chain.rounds[-1] else ''}"
            )
            panels.append(f"""
    <div class="panel{active}" id="panel-{i}">
        <h1>{head}</h1>
        <div class="meta">{meta}</div>
        {_fork_meta_html(r.info, chain, r)}
        {_html_trajectory_body(r.traj, None)}
    </div>""")

    n_chains, n_rounds = len(chains), len(ordered)
    title = f"{n_rounds} rounds / {n_chains} chains" if n_rounds != n_chains else f"{n_chains} forks"
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Forks ({title})</title>
<style>{_HTML_CSS}{_FORK_CSS}</style>
</head>
<body>
<div class="layout">
    <nav class="sidebar">
        <div class="sidebar-title">{title}</div>
        {"".join(nav_items)}
    </nav>
    <main class="content">
        {"".join(panels)}
    </main>
</div>
<script>
function selectTraj(idx) {{
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const panel = document.getElementById('panel-' + idx);
    if (panel) panel.classList.add('active');
    const nav = document.querySelector('.nav-item[data-idx="' + idx + '"]');
    if (nav) nav.classList.add('active');
    window.scrollTo(0, 0);
}}
</script>
</body>
</html>"""

    out.write_text(page, encoding="utf-8")
    # Forks are single-shot now, so n_rounds == n_chains for anything freshly produced; the
    # "across N chains" phrasing only says anything for the chained runs under archive/.
    summary = f"{n_rounds} fork(s)" if n_rounds == n_chains else f"{n_rounds} round(s) across {n_chains} chain(s)"
    print(f"Wrote {out} ({summary})")
    for chain in chains:
        print(
            f"  {chain.root:<22} {' → '.join(_fmt(v) for v in chain.trace):<32} "
            f"cells {[(r.info or {}).get('fork_cell') for r in chain.rounds]} "
            f"stopped: {chain.stop_reason or '?'}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect forked trajectories as HTML")
    parser.add_argument("forks_dir", nargs="?", default="forks", help="Directory of fork sub-dirs (default: forks)")
    parser.add_argument("--html", metavar="FILE", default="forks.html", help="Output HTML path (default: forks.html)")
    args = parser.parse_args()
    write_html(Path(args.forks_dir), Path(args.html))
