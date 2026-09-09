#!/usr/bin/env python3
"""ABLATION: fork once, injecting feedback from *every* criterion — not just the driving ones.

The baseline (``scripts/fork/fork_trajectory.py``) forks at the earliest per-criterion
``first_wrong_step`` and injects the ``feedback`` of only the criteria sitting at
*that* cell. This variant changes exactly one thing: the injected note carries the
feedback from **every** criterion that has any. The fork *cell* is untouched.

One further point about what "every criterion" means, since it is not obvious:

* **It means every criterion that lost points.** The judge emits
  ``feedback: null`` at full marks by contract (``hypotest.env.prompts``, the
  ``"feedback"`` bullet: *"null only at full marks"*), verified on the 1023 criteria in
  ``archive/sonnet-judge/results-hypotest-wo-protocol``: of the 509 with no feedback,
  473 scored 1/1 and 33 scored 5/5. A satisfied criterion has no forward-looking
  guidance to give, so there is nothing to inject and nothing is fabricated.

Criteria that lost points but have no located wrong cell (``first_wrong_step`` null —
7 of 514) ARE included, sorted last; see ``_feedback_order``. In practice this never
fires: all 7 sit in a single run that has no ``first_wrong_step`` anywhere and is
therefore skipped outright.

Measured effect on the same 143 forkable runs: bullets per fork rise from mean 1.90
(max 7) to mean 3.55 (max 9).

Implementation: this imports the baseline module and rebinds two module-level
functions, rather than duplicating ~660 lines that would drift out of sync. Drift is
exactly what invalidates an ablation — the two arms must differ in one place only, and
here that place is auditable at a glance. Both rebinds are safe because of how the
baseline calls them:

* ``select_fork_criterion`` is called at two sites; ``fork_round`` uses only element
  ``[0]`` (the fork cell, which this override computes identically), and only
  ``fork_one`` unpacks element ``[1]``.
* That ``[1]`` (``crits``) flows *only* into ``criterion_feedback``. Nothing else reads
  it, and the copies are never serialised, so the transient ``_fork_driver`` key cannot
  reach ``score_info.json`` or ``fork_info.json``.

**Both arms must be run fresh.** No archived fork run is a valid control: every one of
them was graded with the judge-side fork anchoring (``resume_from_step`` /
``parent_criteria`` / the resume + prior-score prompt notes) that has since been removed
from ``src/``, so their grades are not comparable with anything produced now. Run
``fork.bash`` for the driving-criteria arm and ``fork.allfb.bash`` for this one, against
the same ``--pkl`` / ``--results``.

Usage — identical to the baseline, which owns every CLI flag:

    source .venv/bin/activate && set -a && source .env && set +a
    python scripts/fork/fork_trajectory_allfb.py \\
        --server-config server.fork.yaml \\
        --benchmark-config benchmark.fork.gen.yaml \\
        --pkl archive/sonnet-judge/benchmark_results-hypotest-wo-protocol/trajectories.pkl \\
        --results archive/sonnet-judge/results-hypotest-wo-protocol \\
        --out-dir archive/sonnet-judge/forks-allfb-hypotest-wo-protocol/

Point ``--out-dir`` somewhere distinct from the baseline arm's forks: the two arms are
told apart on disk by directory, and each fork's ``fork_info.json`` stores the full
``injected_feedback``, whose "Start here" header is self-identifying.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import fork_trajectory`

import fork_trajectory as ft  # noqa: E402


def _feedback_order(crit: dict) -> int:
    """Sort key: by ``first_wrong_step`` ascending, criteria with no located cell last.

    Criteria that lost points without any cell flagged ``correct: false`` carry no
    ``first_wrong_step``; they still have guidance worth injecting, but it is not
    anchored to a cell, so it goes at the bottom rather than interleaved.
    """
    fws = crit.get("first_wrong_step")
    return fws if isinstance(fws, int) else sys.maxsize


def select_fork_criterion(criteria: list) -> tuple[int | None, list[dict]]:
    """Baseline's fork cell, but every criterion with feedback comes along for the ride.

    Element ``[0]`` — the fork cell — is computed exactly as the baseline does, so the
    fork point, the recorded ``new_first_wrong_step`` and the ``SkipFork`` check are all
    unchanged. Element ``[1]`` widens from "the criteria at that cell" to "every criterion
    carrying feedback", each a *shallow copy* tagged ``_fork_driver`` (True for the ones
    that actually drive the fork cell). Copies, so the tag can never leak into the criteria
    dicts the baseline serialises into ``fork_info.json``.
    """
    candidates = [c for c in criteria if isinstance(c, dict) and isinstance(c.get("first_wrong_step"), int)]
    if not candidates:
        return None, []
    earliest = min(c["first_wrong_step"] for c in candidates)

    with_feedback = [c for c in criteria if isinstance(c, dict) and (c.get("feedback") or "").strip()]
    drivers = [{**c, "_fork_driver": True} for c in with_feedback if c.get("first_wrong_step") == earliest]
    rest = sorted(
        ({**c, "_fork_driver": False} for c in with_feedback if c.get("first_wrong_step") != earliest),
        key=_feedback_order,
    )
    return earliest, [*drivers, *rest]


def criterion_feedback(crits: list[dict]) -> str | None:
    """Build the two-section fork note from every criterion that carried feedback.

    The criteria driving the fork cell head the note under "Start here"; the rest follow
    under "Also address as you continue", in the order the policy will meet them. Criterion
    *names* are deliberately omitted — they are full sentences (median 104 chars, max 288 in
    the archived runs) and would swamp the guidance they label.

    Falls back to a single unlabelled section when nothing but drivers has feedback, so a
    one-criterion fork reads much like the baseline note. Returns ``None`` when no criterion
    has feedback at all, matching the baseline contract (``fork_round`` prints "forking
    without feedback" and injects no message).
    """
    drivers = [fb for c in crits if c.get("_fork_driver") and (fb := (c.get("feedback") or "").strip())]
    others = [fb for c in crits if not c.get("_fork_driver") and (fb := (c.get("feedback") or "").strip())]
    if not drivers and not others:
        return None

    def bullets(notes: list[str]) -> str:
        return "\n".join(f"- {n}" for n in notes)

    # One section only when there is nothing to split: either no criterion drives the fork cell
    # (possible in principle), or nothing beyond the drivers has feedback — the common
    # single-criterion fork, which then reads exactly like the baseline note.
    if not drivers or not others:
        return (
            "Guidance for how to proceed (from an evaluator):\n"
            f"{bullets(drivers or others)}\n\n"
            "Take this into account as you continue."
        )
    return (
        "Guidance for how to proceed (from an evaluator).\n\n"
        f"Start here — the earliest thing to fix:\n{bullets(drivers)}\n\n"
        f"Also address as you continue:\n{bullets(others)}\n\n"
        "Take this into account as you continue."
    )


ft.select_fork_criterion = select_fork_criterion
ft.criterion_feedback = criterion_feedback


if __name__ == "__main__":
    # `main` builds its ArgumentParser with `description=__doc__`, resolved from the baseline
    # module at call time — so without this, `--help` describes the baseline's feedback rule.
    ft.__doc__ = __doc__
    asyncio.run(ft.main())
