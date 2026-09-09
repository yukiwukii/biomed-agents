#!/usr/bin/env python3
"""Inspect trajectories from benchmark_results/trajectories.pkl.

Terminal:  python inspect_trajectory.py [--idx N] [--step N]
HTML:      python inspect_trajectory.py --html out.html   (all trajectories, pick from sidebar)
"""

import argparse
import html
import json
import pickle
import re
import textwrap
from pathlib import Path
from typing import Optional

# ── ANSI terminal ────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[34m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"

WIDTH = 100
_color = True


def c(*codes: str) -> str:
    return "".join(codes) if _color else ""


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


# ── shared data extraction ───────────────────────────────────────────────────


def extract_action_calls(action):
    """Return list of (tool_name, args_dict)."""
    if action is None:
        return []
    inner = getattr(action, "value", action)
    tool_calls = getattr(inner, "tool_calls", None) or []
    result = []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None) or getattr(tc, "name", "?")
        raw_args = getattr(fn, "arguments", None) or getattr(tc, "arguments", {})
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except Exception:
                raw_args = {"raw": raw_args}
        result.append((name, raw_args))
    return result


def extract_thinking(action) -> str:
    """Return the model's thinking text from the action content, or empty string."""
    if action is None:
        return ""
    inner = getattr(action, "value", action)
    content = getattr(inner, "content", None)
    if not content or not isinstance(content, str):
        return ""
    # Strip leading <think> tag if present
    text = content.strip()
    if text.startswith("<think>"):
        text = text[len("<think>") :].lstrip()
    # Everything before </think> is the thinking
    if "</think>" in text:
        return text[: text.index("</think>")].strip()
    # No closing tag — treat whole content as thinking only if no tool calls
    inner_calls = getattr(inner, "tool_calls", None) or []
    if not inner_calls:
        return text
    return ""


def load_trajectories(path: Path):
    with open(path, "rb") as f:
        trajs = pickle.load(f)
    if not isinstance(trajs, (list, tuple)):
        trajs = [trajs]
    return trajs


# ── terminal renderer ────────────────────────────────────────────────────────


def _term_box(title: str, body: str, color: str = "") -> str:
    title_colored = f"{c(color, BOLD)}{title}{c(RESET)}"
    top = f"┌─ {title_colored} {'─' * max(0, WIDTH - 4 - len(title))}┐"
    lines = []
    for raw_line in body.splitlines():
        clean = strip_ansi(raw_line)
        wrapped = textwrap.wrap(clean, width=WIDTH - 4) if clean.strip() else [""]
        lines.extend(f"│ {w:<{WIDTH - 3}}│" for w in wrapped)
    bottom = f"└{'─' * (WIDTH - 1)}┘"
    return "\n".join([top, *lines, bottom])


def _term_fmt_messages(msgs) -> str:
    parts = []
    for msg in msgs:
        role = getattr(msg, "role", "?")
        content = strip_ansi(str(getattr(msg, "content", msg) or ""))
        role_colored = f"{c(DIM)}[{role}]{c(RESET)}"
        lines = content.splitlines() or [""]
        first = f"{role_colored} {lines[0]}"
        rest = [f"{'':>{len(role) + 3}}{ln}" for ln in lines[1:]]
        parts.append("\n".join([first, *rest]))
    return "\n\n".join(parts)


def _term_fmt_action(action) -> str:
    calls = extract_action_calls(action)
    if not calls:
        return strip_ansi(str(action)) if action else "(no action)"
    parts = []
    for name, args in calls:
        lines = [f"{c(YELLOW, BOLD)}{name}{c(RESET)}("]
        for k, v in args.items():
            v_str = str(v)
            if "\n" in v_str or len(v_str) > 80:
                indented = textwrap.indent(v_str, "    ")
                lines.extend((f"  {c(CYAN)}{k}{c(RESET)} =", indented))
            else:
                lines.append(f"  {c(CYAN)}{k}{c(RESET)} = {v_str}")
        lines.append(")")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# Per-criterion pass state → (html_class, mark char, terminal color).
# Biomni criteria are graded by A/B/C level and the serialized judge output carries
# NO numeric score, so grade by level (A=full, B=partial, C=zero), not by score.
_LEVEL_STATUS = {
    "A": ("crit-pass", "✓", GREEN),
    "B": ("crit-partial", "~", YELLOW),
    "C": ("crit-fail", "✗", RED),
}


def _crit_status(cr) -> tuple[str, str, str]:
    """Return (html_class, mark, term_color) for a criterion's pass state."""
    level = str(cr.get("level") or "").strip().upper()
    if level:
        return _LEVEL_STATUS.get(level, ("crit-fail", "✗", RED))
    passed = int(cr.get("score") or 0) > 0
    return ("crit-pass", "✓", GREEN) if passed else ("crit-fail", "✗", RED)


def _rubric_header(criteria) -> str:
    """Level tally for biomni criteria (no numeric score in the observation), else a points total."""
    if any(str(cr.get("level") or "").strip() for cr in criteria):
        tally: dict[str, int] = {}
        for cr in criteria:
            tally[str(cr.get("level") or "?").upper()] = tally.get(str(cr.get("level") or "?").upper(), 0) + 1
        counts = "  ".join(f"{lvl}×{tally[lvl]}" for lvl in sorted(tally))
        return f"{counts}  across {len(criteria)} criteria"
    total = sum(int(cr.get("score") or 0) for cr in criteria)
    return f"{total} pts across {len(criteria)} criteria"


def _term_fmt_rubric(criteria, first_wrong_step) -> str:
    lines = [f"{c(BOLD)}{_rubric_header(criteria)}{c(RESET)}", ""]
    for cr in criteria:
        cls, mark_ch, color = _crit_status(cr)
        mark = f"{c(color)}{mark_ch}{c(RESET)}"
        # Biomni criteria carry an A/B/C level (no numeric score in the observation);
        # show the level. Hypotest criteria show the integer points.
        level = str(cr.get("level") or "").strip().upper()
        badge = level or f"{int(cr.get('score') or 0)} pt"
        lines.append(f"{mark} [{badge}] {c(color, BOLD)}{cr.get('criterion', '')}{c(RESET)}")
        just = str(cr.get("justification", "")).strip()
        if just:
            lines.append(f"{c(DIM)}{just}{c(RESET)}")
        for st in cr.get("relevant_steps") or []:
            if not isinstance(st, dict):
                continue
            step_ok = bool(st.get("correct"))
            step_mark = f"{c(GREEN)}✓{c(RESET)}" if step_ok else f"{c(RED)}✗{c(RESET)}"
            note = str(st.get("note") or "").strip()
            lines.append(f"  {step_mark} cell {st.get('step')}: {c(DIM)}{note}{c(RESET)}")
        cr_fws = cr.get("first_wrong_step")
        if cr_fws is not None:
            lines.append(f"{c(YELLOW)}First wrong step: cell {cr_fws}{_fws_suffix(cr)}{c(RESET)}")
        cr_fb = str(cr.get("feedback") or "").strip()
        if cr_fb:
            lines.append(f"{c(CYAN)}Feedback: {cr_fb}{c(RESET)}")
        lines.append("")
    if first_wrong_step is not None:
        lines.append(f"{c(YELLOW, BOLD)}Earliest wrong step: cell {first_wrong_step}{c(RESET)}")
    return "\n".join(lines).rstrip()


def _print_term_result(role: str, text: str) -> None:
    """Print a RESULT box, upgrading a `Rubric evaluation:` JSON tail into a rubric box."""
    marker = "Rubric evaluation:"
    if marker in text:
        prose, _, rubric_text = text.partition(marker)
        parsed = _extract_rubric(rubric_text)
        if parsed:
            if prose.strip():
                print(_term_box(f"RESULT [{role}]", prose.strip(), GREEN))
                print()
            print(_term_box("RUBRIC", _term_fmt_rubric(*parsed), GREEN))
            return
    print(_term_box(f"RESULT [{role}]", text.strip(), GREEN))


def print_trajectory(path: Path, traj_idx: int, step_filter: int | None) -> None:
    trajs = load_trajectories(path)

    if traj_idx >= len(trajs):
        print(f"Only {len(trajs)} trajectories (0–{len(trajs) - 1})")
        return

    traj = trajs[traj_idx]
    final_reward = traj.steps[-1].reward if traj.steps else 0.0
    reward_color = GREEN if final_reward > 0 else RED

    print()
    print(f"{c(BOLD)}{'═' * WIDTH}{c(RESET)}")
    print(
        f"{c(BOLD)}  Trajectory {traj.traj_id}  •  {len(traj.steps)} steps  "
        f"•  final reward: {c(reward_color)}{final_reward}{c(RESET)}"
    )
    print(f"{c(BOLD)}{'═' * WIDTH}{c(RESET)}")

    steps = traj.steps
    if step_filter is not None:
        if step_filter >= len(steps):
            print(f"Step {step_filter} does not exist (0–{len(steps) - 1})")
            return
        steps = [steps[step_filter]]

    last_step = steps[-1]
    for step in steps:
        done_str = f"{c(GREEN)}done{c(RESET)}" if step.done else f"{c(DIM)}running{c(RESET)}"
        reward_str = f"{c(GREEN)}{step.reward}{c(RESET)}" if step.reward > 0 else f"{c(DIM)}{step.reward}{c(RESET)}"
        print()
        print(f"{c(BOLD)}  ── STEP {step.timestep} ──  reward={reward_str}  {done_str}{c(RESET)}")
        print()

        obs_msgs = getattr(step, "observation", []) or []
        if obs_msgs:
            print(_term_box("OBSERVATION", _term_fmt_messages(obs_msgs), BLUE))
            print()

        thinking = extract_thinking(step.action)
        if thinking:
            print(_term_box("THINKING", thinking, CYAN))
            print()

        print(_term_box("ACTION", _term_fmt_action(step.action), YELLOW))

        if step is last_step:
            next_obs = getattr(step, "next_observation", None) or []
            if next_obs:
                print()
                for msg in next_obs:
                    role = getattr(msg, "role", "?")
                    content = strip_ansi(str(getattr(msg, "content", msg) or ""))
                    remaining = content
                    while "<think>" in remaining and "</think>" in remaining:
                        before = remaining[: remaining.index("<think>")]
                        inner = remaining[remaining.index("<think>") + len("<think>") : remaining.index("</think>")]
                        remaining = remaining[remaining.index("</think>") + len("</think>") :]
                        if before.strip():
                            print(_term_box(f"RESULT [{role}]", before.strip(), GREEN))
                            print()
                        print(_term_box("RUBRIC THINKING", inner.strip(), CYAN))
                        print()
                    if remaining.strip():
                        _print_term_result(role, remaining)

    print()


# ── HTML renderer ────────────────────────────────────────────────────────────

_HTML_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Menlo', 'Consolas', 'DejaVu Sans Mono', monospace;
    font-size: 13px;
    background: #0f1117;
    color: #e2e8f0;
    line-height: 1.6;
}

.layout { display: flex; min-height: 100vh; }

.sidebar {
    width: 220px;
    flex-shrink: 0;
    background: #0b0d13;
    border-right: 1px solid #1e293b;
    padding: 16px 10px;
    position: sticky;
    top: 0;
    align-self: flex-start;
    height: 100vh;
    overflow-y: auto;
}
.sidebar-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    padding: 0 8px 10px;
}
.nav-item {
    display: block;
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    color: #94a3b8;
    font-family: inherit;
    font-size: 0.8rem;
    padding: 8px 10px;
    margin-bottom: 2px;
    cursor: pointer;
}
.nav-item:hover { background: #141821; }
.nav-item.active { background: #1e293b; border-color: #334155; color: #f8fafc; }
.nav-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}
.dot-good { background: #4ade80; }
.dot-bad  { background: #f87171; }
.nav-main { font-weight: 700; }
.nav-sub { display: block; color: #64748b; font-size: 0.72rem; margin-top: 2px; padding-left: 16px; }

.content { flex: 1; padding: 24px; min-width: 0; }
.panel { display: none; }
.panel.active { display: block; }

h1 { font-size: 1.1rem; color: #f8fafc; margin-bottom: 4px; }
.meta { color: #64748b; font-size: 0.85rem; margin-bottom: 24px; }
.reward-good { color: #4ade80; font-weight: 700; }
.reward-bad  { color: #f87171; font-weight: 700; }

.step {
    margin-bottom: 32px;
    border-left: 3px solid #334155;
    padding-left: 16px;
}
.step-header {
    font-weight: 700;
    font-size: 0.85rem;
    color: #94a3b8;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
}
.step-header .ts   { color: #e2e8f0; font-size: 1rem; }
.step-header .done { color: #4ade80; }
.step-header .running { color: #64748b; }

.section { margin-bottom: 10px; }
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 4px;
    padding: 2px 6px;
    border-radius: 3px;
    display: inline-block;
}
.obs-label      { background: #1e3a5f; color: #60a5fa; }
.thinking-label { background: #1e1535; color: #c084fc; }
.action-label   { background: #3b2e0a; color: #fbbf24; }

.box {
    border-radius: 6px;
    padding: 12px 14px;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-x: auto;
}
.obs-box      { background: #0d1f35; border: 1px solid #1e3a5f; }
.thinking-box { background: #120d1f; border: 1px solid #3b1f6b; color: #d8b4fe; font-style: italic; }
.action-box   { background: #1c1500; border: 1px solid #3b2e0a; }

.role-tag {
    color: #475569;
    font-size: 0.8rem;
    margin-bottom: 2px;
    margin-top: 10px;
}
.role-tag:first-child { margin-top: 0; }

.result-label { background: #0f2e1a; color: #4ade80; }
.result-box   { background: #0a1f12; border: 1px solid #166534; }

.tool-name { color: #fbbf24; font-weight: 700; }
.arg-name  { color: #67e8f9; }
.code-block {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 10px 12px;
    margin-top: 4px;
    white-space: pre;
    overflow-x: auto;
    color: #cbd5e1;
}

/* ── rubric ─────────────────────────────────────────── */
.rubric { margin-top: 6px; line-height: 1.35; }
.rubric-top {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin-bottom: 4px;
}
.rubric-top b { color: #f8fafc; }
.criterion {
    border-radius: 5px;
    padding: 4px 10px;
    margin-bottom: 3px;
    border-left: 3px solid #334155;
    background: #0d1320;
}
.criterion.crit-pass { border-left-color: #22c55e; background: #0a1f12; }
.criterion.crit-partial { border-left-color: #f59e0b; background: #1f1600; }
.criterion.crit-fail { border-left-color: #ef4444; background: #1f0d0d; }
.crit-head { display: flex; align-items: baseline; gap: 7px; }
.crit-mark { font-weight: 700; }
.crit-pass .crit-mark { color: #4ade80; }
.crit-partial .crit-mark { color: #fbbf24; }
.crit-fail .crit-mark { color: #f87171; }
.crit-score {
    font-weight: 700;
    font-size: 0.68rem;
    padding: 0 6px;
    border-radius: 9px;
    background: #1e293b;
    color: #e2e8f0;
    flex-shrink: 0;
}
.crit-pass .crit-score { background: #14532d; color: #bbf7d0; }
.crit-fail .crit-score { background: #4c1414; color: #fecaca; }
/* A/B/C level badge (present on biomni-graded criteria) */
.crit-level {
    font-weight: 700;
    font-size: 0.68rem;
    padding: 0 7px;
    border-radius: 9px;
    background: #312e81;
    color: #c7d2fe;
    flex-shrink: 0;
}
.crit-pass .crit-level { background: #14532d; color: #bbf7d0; }
.crit-partial .crit-level { background: #78350f; color: #fde68a; }
.crit-fail .crit-level { background: #4c1414; color: #fecaca; }
.crit-name { font-weight: 700; color: #f1f5f9; }
.crit-just {
    margin-top: 1px;
    font-size: 0.82rem;
    color: #94a3b8;
    white-space: pre-wrap;
    word-break: break-word;
}
.rubric-fw {
    margin-top: 4px;
    padding: 3px 9px;
    border-radius: 5px;
    background: #3b2e0a;
    color: #fbbf24;
    font-weight: 700;
    display: inline-block;
}
.crit-fb {
    margin-top: 4px;
    padding: 3px 9px;
    border-radius: 5px;
    background: #0d2535;
    color: #67e8f9;
    white-space: pre-wrap;
    word-break: break-word;
}
.rsteps { margin-top: 4px; display: flex; flex-direction: column; gap: 2px; }
.rstep {
    display: flex;
    align-items: baseline;
    gap: 7px;
    font-size: 0.8rem;
    padding: 2px 8px;
    border-radius: 4px;
    border-left: 2px solid #334155;
    background: #0d1320;
}
.rstep-ok  { border-left-color: #22c55e; }
.rstep-bad { border-left-color: #ef4444; }
.rstep-mark { font-weight: 700; flex-shrink: 0; }
.rstep-ok  .rstep-mark { color: #4ade80; }
.rstep-bad .rstep-mark { color: #f87171; }
.rstep-cell { color: #94a3b8; font-weight: 700; flex-shrink: 0; }
.rstep-note { color: #cbd5e1; word-break: break-word; }
"""


def _h(text: str) -> str:
    """HTML-escape."""
    return html.escape(str(text))


def _html_messages(msgs) -> str:
    parts = []
    for msg in msgs:
        role = getattr(msg, "role", "?")
        content = strip_ansi(str(getattr(msg, "content", msg) or ""))
        parts.extend((f'<div class="role-tag">[{_h(role)}]</div>', f"<div>{_h(content)}</div>"))
    return "\n".join(parts)


def _derive_first_wrong_steps(criteria: list) -> None:
    """Fill in each criterion's `first_wrong_step` when the judge didn't emit one.

    Mirrors hypotest.env.interpreter_env.derive_first_wrong_step: the earliest
    `relevant_steps` entry with `"correct": false`. The judge no longer emits the field,
    and this viewer reads the raw judge JSON out of the trajectory's observation text
    (not score_info.json), so without this the value renders as absent.

    One known divergence from score_info.json: the env also nulls the field for criteria
    awarded full marks, which needs the rubric's per-criterion maximum. The rubric is not
    carried in the trajectory at all (only the judge's prompt has it, and that lives in
    score_info.json), so this gate cannot be applied here — a full-marks criterion with a
    flagged step shows a step here but `null` in score_info.json. Values derived here are
    therefore tagged `_fws_derived` and rendered as "(derived)" so they are not mistaken
    for the authoritative ones. Criteria that already carry the key (post-Patch-19 runs,
    and older runs graded when the judge emitted it) are left untouched.
    """
    for c in criteria:
        # Presence of the key — not its value — means it's already authoritative: since
        # Patch 19 the env emits derived criteria where an explicit null is a *decision*
        # (full marks), which we must not overwrite by re-deriving.
        if not isinstance(c, dict) or "first_wrong_step" in c:
            continue
        wrong = [
            s.get("step")
            for s in c.get("relevant_steps") or []
            if isinstance(s, dict) and not s.get("correct", True) and isinstance(s.get("step"), int)
        ]
        c["first_wrong_step"] = min(wrong) if wrong else None
        c["_fws_derived"] = True


def _fws_suffix(cr: dict) -> str:
    """ " (derived)" when the viewer computed the step itself — see _derive_first_wrong_steps."""
    return " (derived)" if cr.get("_fws_derived") else ""


def _extract_rubric(text: str):
    """Parse a rubric evaluation JSON (`{criteria: [...], first_wrong_step}`) out of text.

    Returns (criteria_list, first_wrong_step) or None if no rubric JSON is found.
    """
    if "{" not in text or "}" not in text:
        return None
    start = text.index("{")
    end = text.rindex("}")
    if end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return None
    criteria = data.get("criteria")
    if not isinstance(criteria, list):
        return None
    _derive_first_wrong_steps(criteria)
    first_wrong_step = data.get("first_wrong_step")
    if first_wrong_step is None:
        # New schema: first_wrong_step is per-criterion; derive the earliest one.
        cr_steps = [cr.get("first_wrong_step") for cr in criteria if cr.get("first_wrong_step") is not None]
        first_wrong_step = min(cr_steps, default=None)
    return criteria, first_wrong_step


def _html_relevant_steps(steps) -> str:
    """Render a criterion's `relevant_steps` (list of {step, correct, note}) as a per-cell list."""
    rows = []
    for st in steps or []:
        if not isinstance(st, dict):
            continue
        ok = bool(st.get("correct"))
        cls = "rstep-ok" if ok else "rstep-bad"
        mark = "✓" if ok else "✗"
        rows.append(
            f'<div class="rstep {cls}"><span class="rstep-mark">{mark}</span>'
            f'<span class="rstep-cell">cell {_h(st.get("step"))}</span>'
            f'<span class="rstep-note">{_h(st.get("note", ""))}</span></div>'
        )
    return f'<div class="rsteps">{"".join(rows)}</div>' if rows else ""


def _html_rubric(criteria, first_wrong_step) -> str:
    """Render parsed rubric criteria as readable pass/fail cards."""
    rows = []
    for cr in criteria:
        cls, mark, _color = _crit_status(cr)
        # Biomni criteria carry an A/B/C level (no numeric score in the observation);
        # show the level chip. Hypotest criteria show the integer points.
        level = str(cr.get("level") or "").strip().upper()
        grade_badge = (
            f'<span class="crit-level">{_h(level)}</span>'
            if level
            else f'<span class="crit-score">{_h(int(cr.get("score") or 0))} pt</span>'
        )
        rows.append(f"""
        <div class="criterion {cls}">
            <div class="crit-head">
                <span class="crit-mark">{mark}</span>
                {grade_badge}
                <span class="crit-name">{_h(cr.get("criterion", ""))}</span>
            </div>
            <div class="crit-just">{_h(cr.get("justification", ""))}</div>
            {_html_relevant_steps(cr.get("relevant_steps"))}
            {f'<div class="crit-fw">First wrong step: cell {_h(cr.get("first_wrong_step"))}{_fws_suffix(cr)}</div>' if cr.get("first_wrong_step") is not None else ""}
            {f'<div class="crit-fb">Feedback: {_h(cr.get("feedback"))}</div>' if str(cr.get("feedback") or "").strip() else ""}
        </div>""")
    fw = ""
    if first_wrong_step is not None:
        fw = f'<div class="rubric-fw">Earliest wrong step: cell {_h(first_wrong_step)}</div>'
    return f"""
    <div class="rubric">
        <div class="rubric-top">Rubric &nbsp;·&nbsp; <b>{_h(_rubric_header(criteria))}</b></div>
        {"".join(rows)}
        {fw}
    </div>"""


def _html_result_text(text: str) -> str:
    """Render result prose, upgrading a `Rubric evaluation:` JSON tail into rubric cards."""
    marker = "Rubric evaluation:"
    if marker in text:
        prose, _, rubric_text = text.partition(marker)
        parsed = _extract_rubric(rubric_text)
        if parsed:
            head = f"<div>{_h(prose.rstrip())}</div>" if prose.strip() else ""
            return head + _html_rubric(*parsed)
    return f"<div>{_h(text)}</div>"


def _html_messages_with_thinking(msgs) -> str:
    """Like _html_messages but renders <think>…</think> blocks as styled thinking boxes."""
    parts = []
    for msg in msgs:
        role = getattr(msg, "role", "?")
        content = strip_ansi(str(getattr(msg, "content", msg) or ""))
        parts.append(f'<div class="role-tag">[{_h(role)}]</div>')
        rendered = ""
        remaining = content
        while "<think>" in remaining and "</think>" in remaining:
            before = remaining[: remaining.index("<think>")]
            inner = remaining[remaining.index("<think>") + len("<think>") : remaining.index("</think>")]
            remaining = remaining[remaining.index("</think>") + len("</think>") :]
            if before.strip():
                rendered += f"<div>{_h(before)}</div>"
            rendered += f'<div class="section-label thinking-label" style="margin-top:6px">Thinking</div><div class="box thinking-box">{_h(inner.strip())}</div>'
        if remaining.strip():
            rendered += _html_result_text(remaining)
        parts.append(rendered or "<div></div>")
    return "\n".join(parts)


def _html_action(action) -> str:
    calls = extract_action_calls(action)
    if not calls:
        raw = strip_ansi(str(action)) if action else "(no action)"
        return _h(raw)
    parts = []
    for name, args in calls:
        lines = [f'<span class="tool-name">{_h(name)}</span>(']
        for k, v in args.items():
            v_str = str(v)
            if "\n" in v_str or len(v_str) > 80:
                lines.extend((
                    f'  <span class="arg-name">{_h(k)}</span> =',
                    f'<div class="code-block">{_h(v_str)}</div>',
                ))
            else:
                lines.append(f'  <span class="arg-name">{_h(k)}</span> = {_h(v_str)}')
        lines.append(")")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _html_trajectory_body(traj, step_filter: int | None) -> str:
    """Render the inner step-by-step body for a single trajectory."""
    steps = traj.steps
    if step_filter is not None:
        if step_filter >= len(steps):
            return f'<div class="meta">Step {step_filter} does not exist (0–{len(steps) - 1})</div>'
        steps = [steps[step_filter]]

    if not steps:
        return '<div class="meta">(no steps)</div>'

    body_parts = []
    last_step = steps[-1]
    for step in steps:
        done_cls = "done" if step.done else "running"
        done_label = "done" if step.done else "running"
        reward_span = (
            f'<span class="reward-good">{step.reward}</span>'
            if step.reward > 0
            else f'<span style="color:#64748b">{step.reward}</span>'
        )

        obs_msgs = getattr(step, "observation", []) or []

        obs_html = ""
        if obs_msgs:
            obs_html = f"""
            <div class="section">
                <div class="section-label obs-label">Observation</div>
                <div class="box obs-box">{_html_messages(obs_msgs)}</div>
            </div>"""

        thinking_text = extract_thinking(step.action)
        thinking_html = ""
        if thinking_text:
            thinking_html = f"""
            <div class="section">
                <div class="section-label thinking-label">Thinking</div>
                <div class="box thinking-box">{_h(thinking_text)}</div>
            </div>"""

        result_html = ""
        if step is last_step:
            next_obs = getattr(step, "next_observation", None) or []
            if next_obs:
                result_html = f"""
            <div class="section">
                <div class="section-label result-label">Result</div>
                <div class="box result-box">{_html_messages_with_thinking(next_obs)}</div>
            </div>"""

        body_parts.append(f"""
        <div class="step">
            <div class="step-header">
                <span class="ts">Step {step.timestep}</span>
                &nbsp;·&nbsp; reward={reward_span}
                &nbsp;·&nbsp; <span class="{done_cls}">{done_label}</span>
            </div>
            {obs_html}
            {thinking_html}
            <div class="section">
                <div class="section-label action-label">Action</div>
                <div class="box action-box">{_html_action(step.action)}</div>
            </div>
            {result_html}
        </div>""")

    return "".join(body_parts)


def write_html(path: Path, step_filter: int | None, out: Path) -> None:
    """Write all trajectories into a single page with a sidebar picker."""
    trajs = load_trajectories(path)

    if not trajs:
        print("No trajectories found")
        return

    nav_items = []
    panels = []
    for i, traj in enumerate(trajs):
        final_reward = traj.steps[-1].reward if traj.steps else 0.0
        reward_cls = "reward-good" if final_reward > 0 else "reward-bad"
        active = " active" if i == 0 else ""
        dot_cls = "dot-good" if final_reward > 0 else "dot-bad"

        nav_items.append(f"""
        <button class="nav-item{active}" data-idx="{i}" onclick="selectTraj({i})">
            <span class="nav-dot {dot_cls}"></span>
            <span class="nav-main">#{i}</span>
            <span class="nav-sub">{len(traj.steps)} steps · <span class="{reward_cls}">{final_reward}</span></span>
        </button>""")

        body = _html_trajectory_body(traj, step_filter)
        panels.append(f"""
    <div class="panel{active}" id="panel-{i}">
        <h1>Trajectory {_h(str(traj.traj_id))}</h1>
        <div class="meta">
            index {i} &nbsp;·&nbsp; {len(traj.steps)} steps &nbsp;·&nbsp;
            final reward: <span class="{reward_cls}">{final_reward}</span>
        </div>
        {body}
    </div>""")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trajectories ({len(trajs)})</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<div class="layout">
    <nav class="sidebar">
        <div class="sidebar-title">{len(trajs)} trajectories</div>
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
    print(f"Wrote {out} ({len(trajs)} trajectories)")


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect trajectories from a .pkl file")
    parser.add_argument("pkl", nargs="?", default="benchmark_results/trajectories.pkl")
    parser.add_argument("--idx", type=int, default=0, help="Trajectory index (default: 0)")
    parser.add_argument("--step", type=int, default=None, help="Show only this step number")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors (terminal mode)")
    parser.add_argument("--html", metavar="FILE", help="Write output as HTML instead of printing")
    args = parser.parse_args()

    if args.html:
        write_html(Path(args.pkl), args.step, Path(args.html))
    else:
        if args.no_color:
            _color = False
        print_trajectory(Path(args.pkl), args.idx, args.step)
