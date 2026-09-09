#!/usr/bin/env python3
"""Inspect proposer (hypothesis-generation) trajectories.

Reads the per-capsule JSON files written by
``proposer/generate_hypotheses.py --save-traj DIR`` and renders them the way
``scripts/eval/inspect_trajectory.py`` renders benchmark rollouts: a step-by-step view
of what the proposer did. For ``--mode agent`` that means each data-exploration
cell (code → output); for either mode it shows the final generation prompt (with
``--prompt``), the proposed hypotheses, and the dataset's expert hypothesis for
side-by-side comparison.

Usage:
    # list every trajectory in a dir
    .venv/bin/python proposer/inspect_proposer_traj.py proposer/ --list

    # show one (by index or id substring)
    .venv/bin/python proposer/inspect_proposer_traj.py proposer/ --idx 0
    .venv/bin/python proposer/inspect_proposer_traj.py proposer/ --only 0f14ffa7

    # also print the full generation prompt that produced the hypotheses
    .venv/bin/python proposer/inspect_proposer_traj.py proposer/ --only 0f14ffa7 --prompt

    # a single trajectory file works too
    .venv/bin/python proposer/inspect_proposer_traj.py proposer/0f14ffa7-....json

    # write a self-contained HTML page (all trajectories, sidebar picker)
    .venv/bin/python proposer/inspect_proposer_traj.py proposer/ --html proposer.html
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

# ── ANSI (mirrors inspect_trajectory.py) ─────────────────────────────────────

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, YELLOW, GREEN, RED, MAGENTA = (
    "\033[34m", "\033[36m", "\033[33m", "\033[32m", "\033[31m", "\033[35m",
)
WIDTH = 100
_color = True


def c(*codes: str) -> str:
    return "".join(codes) if _color else ""


def box(title: str, body: str, color: str = "") -> str:
    """A titled box, matching inspect_trajectory.py's _term_box look."""
    line = "─" * WIDTH
    head = f"{c(color, BOLD)}┌─ {title} {'─' * max(0, WIDTH - len(title) - 4)}{c(RESET)}"
    out = [head]
    for raw in body.rstrip("\n").split("\n"):
        for wrapped in textwrap.wrap(raw, WIDTH - 2, drop_whitespace=False, replace_whitespace=False) or [""]:
            out.append(f"{c(color)}│{c(RESET)} {wrapped}")
    out.append(f"{c(color)}└{line}{c(RESET)}")
    return "\n".join(out)


# ── loading ──────────────────────────────────────────────────────────────────


def load_trajectories(path: Path) -> list[dict[str, Any]]:
    """Load one trajectory file or every ``*.json`` in a directory (sorted)."""
    if path.is_file():
        files = [path]
    elif path.is_dir():
        # Recurse so a base dir like proposer/ picks up proposer/agent/ + proposer/preview/.
        files = sorted(path.rglob("*.json"))
    else:
        sys.exit(f"no such file or directory: {path}")
    if not files:
        sys.exit(f"no trajectory JSON files found in {path}")
    trajs = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"warning: skipping {f.name}: {e}", file=sys.stderr)
            continue
        # Per-capsule trajectory files are dicts; skip aggregate --out files (a list
        # of records) that may share the dir, so they don't break rendering.
        if isinstance(data, dict):
            trajs.append(data)
    return trajs


# ── rendering ─────────────────────────────────────────────────────────────────


def _status_color(status: str) -> str:
    return {"ok": GREEN, "skipped": YELLOW, "error": RED}.get(status, "")


def _turns(t: dict[str, Any]) -> tuple[int, int | None, str]:
    """Return (turns_used, max_turns, note). Falls back to cell count for old files."""
    used = t.get("turns", len(t.get("explore_steps") or []))
    mx = t.get("max_turns")
    note = ""
    if mx:
        note = "stopped early" if used < mx else "hit cap"
    return used, mx, note


def _turns_str(t: dict[str, Any]) -> str:
    used, mx, note = _turns(t)
    base = f"{used}/{mx} turns" if mx else f"{used} turns"
    return f"{base} ({note})" if note else base


def print_trajectory(traj: dict[str, Any], show_prompt: bool) -> None:
    status = traj.get("status", "?")
    print()
    print(box(
        f"CAPSULE {traj.get('capsule_id', '?')}",
        f"mode={traj.get('mode')}   model={traj.get('model')}   "
        f"status={c(_status_color(status), BOLD)}{status}{c(RESET)}   "
        f"input_files={traj.get('n_input_files')}\n"
        + (f"warnings: {traj['warnings']}\n" if traj.get("warnings") else "")
        + (f"error: {c(RED)}{traj['error']}{c(RESET)}" if traj.get("error") else ""),
        BLUE,
    ))

    # Exploration trajectory (agent mode).
    steps = traj.get("explore_steps") or []
    if steps:
        print(f"\n{c(MAGENTA, BOLD)}══ DATA EXPLORATION · {_turns_str(traj)} ══{c(RESET)}")
        for s in steps:
            print(box(f"cell {s['step']} · code", s["code"], CYAN))
            print(box(f"cell {s['step']} · output", s["output"], DIM))
    elif traj.get("mode") == "agent":
        print(f"\n{c(YELLOW)}(agent mode but no exploration cells were recorded){c(RESET)}")

    if show_prompt:
        print(f"\n{c(DIM, BOLD)}══ GENERATION PROMPT ══{c(RESET)}")
        print(box("prompt", traj.get("generation_prompt", "(none)"), DIM))

    # Expert (ground-truth) vs generated.
    if traj.get("expert_hypothesis"):
        print(box("EXPERT HYPOTHESIS (dataset ground truth)", traj["expert_hypothesis"], YELLOW))

    hyps = traj.get("hypotheses") or []
    body = "\n\n".join(f"{c(GREEN, BOLD)}{i}.{c(RESET)} {h}" for i, h in enumerate(hyps, 1)) or "(none)"
    print(box(f"GENERATED HYPOTHESES ({len(hyps)})", body, GREEN))


def list_trajectories(trajs: list[dict[str, Any]]) -> None:
    print(f"{c(BOLD)}{'idx':>3}  {'status':<8} {'mode':<8} {'turns':>7}  capsule / expert hypothesis{c(RESET)}")
    for i, t in enumerate(trajs):
        status = t.get("status", "?")
        used, mx, _ = _turns(t)
        turns = f"{used}/{mx}" if mx else str(used)
        expert = (t.get("expert_hypothesis") or "")[:60]
        print(f"{i:>3}  {c(_status_color(status))}{status:<8}{c(RESET)} {t.get('mode', '?'):<8} "
              f"{turns:>7}  {t.get('capsule_id', '?')[:13]}…  {c(DIM)}{expert}{c(RESET)}")


# ── HTML rendering (all trajectories, sidebar picker — mirrors inspect_trajectory.py --html) ──

import html as _html  # noqa: E402


def _h(text: str) -> str:
    return _html.escape(str(text))


def _html_traj_body(t: dict[str, Any], show_prompt: bool) -> str:
    status = t.get("status", "?")
    turns_html = f" · <b>{_h(_turns_str(t))}</b>" if t.get("mode") == "agent" else ""
    parts = [
        f"<h2>Capsule {_h(t.get('capsule_id', '?'))}</h2>",
        f"<p class='meta'>mode=<b>{_h(t.get('mode'))}</b> · model={_h(t.get('model'))} · "
        f"status=<span class='st-{_h(status)}'>{_h(status)}</span> · input_files={_h(t.get('n_input_files'))}"
        + turns_html
        + (f" · warnings: {_h(t['warnings'])}" if t.get("warnings") else "")
        + (f"<br><span class='err'>error: {_h(t['error'])}</span>" if t.get("error") else "") + "</p>",
    ]
    steps = t.get("explore_steps") or []
    if steps:
        parts.append(f"<h3>Data exploration · {_h(_turns_str(t))}</h3>")
        for s in steps:
            parts.append(f"<div class='cell'><div class='lbl'>cell {s['step']} · code</div>"
                         f"<pre class='code'>{_h(s['code'])}</pre>"
                         f"<div class='lbl'>output</div><pre class='out'>{_h(s['output'])}</pre></div>")
    elif t.get("mode") == "agent":
        parts.append("<p class='warn'>(agent mode but no exploration cells were recorded)</p>")

    if show_prompt and t.get("generation_prompt"):
        parts.append("<h3>Generation prompt</h3>"
                     f"<pre class='prompt'>{_h(t['generation_prompt'])}</pre>")

    if t.get("expert_hypothesis"):
        parts.append("<h3>Expert hypothesis <span class='sub'>(dataset ground truth)</span></h3>"
                     f"<div class='expert'>{_h(t['expert_hypothesis'])}</div>")

    hyps = t.get("hypotheses") or []
    parts.append(f"<h3>Generated hypotheses · {len(hyps)}</h3><ol class='gen'>"
                 + "".join(f"<li>{_h(h)}</li>" for h in hyps) + "</ol>")
    return "\n".join(parts)


def write_html(trajs: list[dict[str, Any]], out: Path, show_prompt: bool) -> None:
    panels, links = [], []
    for i, t in enumerate(trajs):
        cid = t.get("capsule_id", "?")
        status = t.get("status", "?")
        used, mx, _ = _turns(t)
        turns = f"{used}/{mx}t" if mx else f"{used}t"
        links.append(
            f"<a href='#' onclick=\"show({i});return false\" id='lnk{i}'>"
            f"<span class='st-{_h(status)}'>●</span> {_h(cid[:13])}… "
            f"<span class='sub'>{_h(t.get('mode'))}·{_h(turns)}</span></a>"
        )
        panels.append(f"<div class='panel' id='p{i}' style='display:none'>{_html_traj_body(t, show_prompt)}</div>")
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>Proposer trajectories</title>
<style>
 body{{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:#1a1a1a;background:#fff}}
 #wrap{{display:flex}} #side{{width:280px;flex:none;height:100vh;overflow:auto;border-right:1px solid #ddd;background:#fafafa}}
 #side h1{{font-size:14px;padding:12px;margin:0;border-bottom:1px solid #ddd}}
 #side a{{display:block;padding:6px 12px;text-decoration:none;color:#333;border-bottom:1px solid #eee;font-size:12px}}
 #side a:hover{{background:#eef}} #side a.active{{background:#dde7ff;font-weight:600}}
 #main{{flex:1;height:100vh;overflow:auto;padding:0 28px 60px}}
 h2{{margin-top:24px}} h3{{margin-top:22px;border-bottom:1px solid #eee;padding-bottom:4px}}
 .meta{{color:#555}} .sub{{color:#999;font-weight:400}} .err{{color:#c00}} .warn{{color:#b60}}
 pre{{white-space:pre-wrap;word-break:break-word;border-radius:6px;padding:10px;font:12px/1.45 ui-monospace,Menlo,monospace}}
 .code{{background:#f3f6fb;border:1px solid #dce3ee}} .out{{background:#f7f7f7;border:1px solid #e5e5e5;color:#333}}
 .prompt{{background:#fbf7ef;border:1px solid #ece0c8;max-height:420px;overflow:auto}}
 .lbl{{font:600 11px sans-serif;color:#777;margin:8px 0 3px}} .cell{{margin-bottom:14px}}
 .expert{{background:#fff7e0;border-left:4px solid #e0a800;padding:10px 14px;border-radius:4px}}
 ol.gen li{{background:#eefaf0;border-left:4px solid #2e9e4f;padding:8px 12px;margin:8px 0;border-radius:4px}}
 .st-ok{{color:#2e9e4f}} .st-skipped{{color:#e0a800}} .st-error{{color:#c00}}
</style></head><body><div id='wrap'>
<div id='side'><h1>Proposer · {len(trajs)} capsules</h1>{''.join(links)}</div>
<div id='main'>{''.join(panels)}</div></div>
<script>
 function show(i){{document.querySelectorAll('.panel').forEach(p=>p.style.display='none');
  document.getElementById('p'+i).style.display='block';
  document.querySelectorAll('#side a').forEach(a=>a.classList.remove('active'));
  document.getElementById('lnk'+i).classList.add('active');
  document.getElementById('main').scrollTop=0;}}
 show(0);
</script></body></html>"""
    out.write_text(doc)
    print(f"Wrote {out}  ({len(trajs)} trajectories)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="Trajectory dir (from --save-traj) or a single .json file")
    ap.add_argument("--idx", type=int, default=None, help="Show trajectory at this index")
    ap.add_argument("--only", default=None, help="Show trajectories whose capsule id contains this substring")
    ap.add_argument("--list", action="store_true", help="List all trajectories and exit")
    ap.add_argument("--prompt", action="store_true", help="Also include the full generation prompt")
    ap.add_argument("--html", metavar="FILE", type=Path, default=None, help="Write an HTML page instead of terminal output")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    args = ap.parse_args()

    global _color
    _color = not args.no_color and sys.stdout.isatty()

    trajs = load_trajectories(args.path)

    if args.only:
        trajs = [t for t in trajs if args.only in t.get("capsule_id", "")]
    if not trajs:
        sys.exit("no trajectories matched")

    if args.html is not None:
        write_html(trajs, args.html, show_prompt=args.prompt)
        return

    if args.list:
        list_trajectories(trajs)
        return

    selected = [trajs[args.idx]] if args.idx is not None else trajs
    for t in selected:
        print_trajectory(t, show_prompt=args.prompt)


if __name__ == "__main__":
    main()
