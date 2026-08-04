#!/usr/bin/env python3
"""Render a per-criterion diff of the ORIGINAL judge vs a re-grade judge as one HTML page.

Reads a ``judge_output.regrade.<model>.json`` produced by ``regrade.py --write``
(each entry carries ``old_criteria`` = original benchmark judge and ``criteria`` =
the new judge, plus ``old_score``/``score``). Criteria are aligned by position
(same rubric, same order) and shown side by side; rows where the per-criterion
score changed are highlighted.

Usage:
    python3 scripts/inspect_judge_diff.py \
        --judge benchmark_results/judge_output.regrade.anthropic_claude-sonnet-4-6.json \
        --out benchmark_results/judge_diff.anthropic_claude-sonnet-4-6.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def task_key(tid: str) -> tuple:
    """Natural sort key by task number, then rep, then fork_cell (if any)."""
    m = re.match(r"task_(\d+)_rep(\d+)(?:-fork_cell(\d+))?", tid)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else -1, tid)
    return (10**9, 0, 0, tid)


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def fws_badge(c: dict) -> str:
    """Small 'first wrong step' cell marker, e.g. cell 20; empty if none flagged."""
    v = c.get("first_wrong_step")
    return f"<div class='fws'>c{esc(v)}</div>" if v is not None else ""


def earliest_fws(criteria: list | None):
    """Earliest (min) first_wrong_step across criteria, or None if none flagged."""
    vals = [c.get("first_wrong_step") for c in (criteria or []) if c.get("first_wrong_step") is not None]
    return min(vals) if vals else None


def crit_rows(old: list | None, new: list | None) -> tuple[str, int]:
    old = old or []
    new = new or []
    n = max(len(old), len(new))
    changed = 0
    rows = []
    for i in range(n):
        o = old[i] if i < len(old) else {}
        c = new[i] if i < len(new) else {}
        os_, ns = o.get("score"), c.get("score")
        label = c.get("criterion") or o.get("criterion") or f"criterion {i + 1}"
        cls = ""
        if os_ != ns:
            changed += 1
            cls = "up" if (ns or 0) > (os_ or 0) else "down"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td class='crit'>{esc(label)}</td>"
            f"<td class='sc'>{esc(os_)}{fws_badge(o)}</td>"
            f"<td class='sc'>{esc(ns)}{fws_badge(c)}</td>"
            f"<td class='just'>{esc(o.get('justification'))}</td>"
            f"<td class='just'>{esc(c.get('justification'))}</td>"
            f"</tr>"
        )
    return "\n".join(rows), changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", type=Path, default=ROOT / "benchmark_results/judge_output.regrade.anthropic_claude-sonnet-4-6.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sort", choices=["task", "delta"], default="task",
                    help="Order cards by task/rep (default) or by absolute score change")
    ap.add_argument("--compare", choices=["old", "parent"], default="old",
                    help="'old' = original judge vs new judge (uses old_criteria); "
                    "'parent' = pre-fork parent vs post-fork, both this model (uses parent_criteria)")
    args = ap.parse_args()

    data = json.loads(args.judge.read_text())
    model = next(iter(data.values())).get("model", "?") if data else "?"
    out = args.out or args.judge.with_suffix("").with_name(args.judge.stem.replace("judge_output", "judge_diff") + ".html")

    if args.compare == "parent":
        title = f"Pre-fork (parent) vs post-fork judge — both {model}"
        before_label, after_label = "pre-fork", "post-fork"
    else:
        title = f"Original judge vs new judge — {model}"
        before_label, after_label = "original", "new"

    def before_of(d: dict):
        """(criteria, normalized_score, raw_score) for the left/'before' side."""
        if args.compare == "parent":
            crit = d.get("parent_criteria")
            raw = sum((c.get("score") or 0) for c in crit) if crit else None
            return crit, d.get("parent_score", 0.0), raw
        return d.get("old_criteria"), d.get("old_score", 0.0), d.get("old_raw_score")

    # Build cards.
    items = []
    for tid, d in data.items():
        before_crit, old_s, before_raw = before_of(d)
        new_s = d.get("score", 0.0)
        rows, changed = crit_rows(before_crit, d.get("criteria"))
        items.append((abs(new_s - old_s), tid, d, old_s, new_s, rows, changed, before_raw, before_crit))
    if args.sort == "delta":
        items.sort(key=lambda x: x[0], reverse=True)
    else:
        items.sort(key=lambda x: task_key(x[1]))

    n = len(items)
    up = sum(1 for _a, _t, _d, o, s, *_ in items if s > o)
    dn = sum(1 for _a, _t, _d, o, s, *_ in items if s < o)
    eq = n - up - dn
    old_mean = sum(x[3] for x in items) / n if n else 0
    new_mean = sum(x[4] for x in items) / n if n else 0

    cards = []
    for _a, tid, d, old_s, new_s, rows, changed, before_raw, before_crit in items:
        arrow = "↑" if new_s > old_s else ("↓" if new_s < old_s else "=")
        hcls = "up" if new_s > old_s else ("down" if new_s < old_s else "same")
        parent_id = f" &nbsp;<span class='rid'>parent {esc(d.get('parent_id'))}</span>" if args.compare == "parent" else ""
        fb, fa = earliest_fws(before_crit), earliest_fws(d.get("criteria"))
        fws_span = (
            f"<span class='fwsagg'>first wrong: "
            f"{'c' + str(fb) if fb is not None else '—'} &rarr; {'c' + str(fa) if fa is not None else '—'}</span>"
        )
        cards.append(f"""
<div class="card" data-changed="{1 if new_s != old_s else 0}">
  <div class="hd {hcls}" onclick="this.parentNode.classList.toggle('open')">
    <span class="tid">{esc(tid)}{parent_id}</span>
    <span class="rid">{esc(d.get('run_id'))}</span>
    <span class="score">{old_s:.2f} ({esc(before_raw)}/{esc(d.get('max_score'))})
      &rarr; {new_s:.2f} ({esc(d.get('raw_score'))}/{esc(d.get('max_score'))}) {arrow}</span>
    {fws_span}
    <span class="chg">{changed} criteria changed</span>
  </div>
  <div class="body">
    <table>
      <thead><tr><th>Criterion</th><th>{esc(before_label)}</th><th>{esc(after_label)}</th>
        <th>{esc(before_label)} justification</th><th>{esc(after_label)} justification</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>""")

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{esc(title)}</title>
<style>
  body {{ font: 14px/1.5 -apple-system, system-ui, sans-serif; margin: 0; background: #f6f7f9; color: #1a1a1a; }}
  header {{ position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd; padding: 12px 20px; z-index: 5; }}
  h1 {{ font-size: 16px; margin: 0 0 6px; }}
  .summary {{ color: #444; font-size: 13px; }}
  .summary b {{ color: #000; }}
  .controls {{ margin-top: 8px; font-size: 13px; }}
  .wrap {{ padding: 16px 20px; max-width: 1200px; margin: 0 auto; }}
  .card {{ background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; margin-bottom: 10px; overflow: hidden; }}
  .hd {{ display: grid; grid-template-columns: 1fr 1fr auto auto auto; gap: 12px; align-items: center;
         padding: 10px 14px; cursor: pointer; border-left: 4px solid #bbb; }}
  .fwsagg {{ font-size: 12px; color: #555; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .fws {{ font-size: 10px; color: #999; font-weight: normal; margin-top: 1px; }}
  .hd.up {{ border-left-color: #2e9c4a; }} .hd.down {{ border-left-color: #d13b3b; }} .hd.same {{ border-left-color: #bbb; }}
  .tid {{ font-weight: 600; }} .rid {{ color: #888; font-family: monospace; }}
  .score {{ font-variant-numeric: tabular-nums; }} .chg {{ color: #777; font-size: 12px; }}
  .body {{ display: none; padding: 0 14px 12px; }}
  .card.open .body {{ display: block; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; vertical-align: top; padding: 6px 8px; border-bottom: 1px solid #eee; }}
  th {{ font-size: 12px; color: #666; }}
  td.sc {{ text-align: center; font-variant-numeric: tabular-nums; width: 40px; font-weight: 600; }}
  td.crit {{ width: 22%; }} td.just {{ width: 30%; color: #333; }}
  tr.up {{ background: #eafaef; }} tr.down {{ background: #fdeceb; }}
  tr.up td.sc {{ color: #2e9c4a; }} tr.down td.sc {{ color: #d13b3b; }}
</style></head><body>
<header>
  <h1>{esc(title)}</h1>
  <div class="summary">
    <b>{n}</b> trajectories &nbsp;|&nbsp; {esc(before_label)} mean <b>{old_mean:.3f}</b> &rarr; {esc(after_label)} mean <b>{new_mean:.3f}</b>
    &nbsp;|&nbsp; <span style="color:#2e9c4a">↑ {up}</span> &nbsp;
    <span style="color:#d13b3b">↓ {dn}</span> &nbsp; = {eq}
    &nbsp;|&nbsp; sorted by {esc(args.sort)}
  </div>
  <div class="controls">
    <label><input type="checkbox" id="only" onchange="filt()"> only trajectories whose score changed</label>
    &nbsp;&nbsp;<a href="#" onclick="toggleAll(true);return false">expand all</a>
    &nbsp;<a href="#" onclick="toggleAll(false);return false">collapse all</a>
  </div>
</header>
<div class="wrap">{''.join(cards)}</div>
<script>
  function filt() {{
    const only = document.getElementById('only').checked;
    document.querySelectorAll('.card').forEach(c => {{
      c.style.display = (only && c.dataset.changed === '0') ? 'none' : '';
    }});
  }}
  function toggleAll(open) {{
    document.querySelectorAll('.card').forEach(c => c.classList.toggle('open', open));
  }}
</script>
</body></html>"""

    out.write_text(doc)
    print(f"Wrote {out}  ({n} trajectories)")


if __name__ == "__main__":
    main()
