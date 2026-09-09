#!/usr/bin/env bash
# rl/scripts/make_rl_runtime.sh
# =============================================================================
# Build RL_RUNTIME: a self-contained, world-readable copy of the hypotest venv
# and its interpreter, for pods to run off the scratch PVC.
#
#   ./rl/scripts/make_rl_runtime.sh            # build (refuses if it exists)
#   ./rl/scripts/make_rl_runtime.sh --force    # rebuild from scratch
#
# WHY THIS EXISTS
# ---------------
# The scratch export is NFS with root_squash. Containers run as root by default,
# root is squashed to `nobody`, so a pod only ever sees world-readable files.
#
#   hypotest/.venv            0/3450 dirs blocked   <- fine
#   hypotest/src              0/12                  <- fine
#   kernel_env_conda          0/20240               <- fine
#   hypotest/capsules         0/582                 <- fine
#   ~/.uv/cpython-3.13.10-*   234/287 blocked       <- the whole problem
#
# .venv/bin/python is a symlink into that last tree, so the env pod died with
# `.venv/bin/python: Permission denied` even though the venv itself is readable.
#
# Two rejected alternatives:
#   * run the pod as the user's uid/gid -- would give a cluster workload write
#     access to the whole home tree as that user. Explicitly declined.
#   * chmod o+rX ~/.uv/cpython-* -- permanent widening of a shared export.
# Copying changes nothing that already exists and grants no pod any identity.
#
# WHAT IT PRODUCES
#   $DST/cpython/   the stock CPython build, copied out of ~/.uv
#   $DST/venv/      a copy of hypotest/.venv, rewired to that interpreter:
#                     - bin/python{,3,3.13} symlinks repointed
#                     - pyvenv.cfg `home` repointed
#                     - console-script shebangs rewritten
#   everything a+rX, so `nobody` can read and execute it.
#
# NOT AUTOMATIC: re-run this after `uv sync` or any dependency change, or the
# pods keep running the old packages.
# =============================================================================
set -euo pipefail

SRC_VENV="${SRC_VENV:-/mlbio_scratch/wangsaja/hypotest/.venv}"
DST="${RL_RUNTIME:-/mlbio_scratch/wangsaja/rl-runtime}"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

log() { printf '\033[36m[make_rl_runtime]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31m[make_rl_runtime] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

[ -d "${SRC_VENV}" ] || die "SRC_VENV=${SRC_VENV} not found."

# The interpreter location is whatever the venv actually points at, not a
# hardcoded version -- a `uv sync` onto a new patch release moves it.
SRC_INTERP="$(readlink -f "${SRC_VENV}/bin/python")" || die "cannot resolve ${SRC_VENV}/bin/python"
SRC_INTERP="${SRC_INTERP%/bin/*}"
[ -x "${SRC_INTERP}/bin/python3" ] || die "resolved interpreter root looks wrong: ${SRC_INTERP}"
log "interpreter: ${SRC_INTERP}"

if [ -e "${DST}" ]; then
  [ "${FORCE}" = 1 ] || die "${DST} already exists. Re-run with --force to rebuild."
  log "removing existing ${DST}"
  rm -rf "${DST}"
fi

mkdir -p "${DST}"
log "copying interpreter -> ${DST}/cpython"
cp -a "${SRC_INTERP}" "${DST}/cpython"
log "copying venv        -> ${DST}/venv"
cp -a "${SRC_VENV}" "${DST}/venv"

log "rewiring the copy to be self-contained"
for l in python python3 python3.13; do
  [ -L "${DST}/venv/bin/${l}" ] && ln -sfn "${DST}/cpython/bin/python3.13" "${DST}/venv/bin/${l}"
done
sed -i "s|^home = .*|home = ${DST}/cpython/bin|" "${DST}/venv/pyvenv.cfg"
# Console scripts keep a `#!<original venv>/bin/python` shebang, which would
# send them straight back into the unreadable tree.
# `|| true` throughout: grep exits 1 on "no matches", which under `set -e` with
# pipefail would abort the script on the success case.
{ grep -rl "^#!${SRC_VENV}" "${DST}/venv/bin" 2>/dev/null || true; } \
  | xargs -r sed -i "1s|^#!${SRC_VENV}|#!${DST}/venv|"

log "chmod a+rX (this is what makes it usable by a root-squashed pod)"
chmod -R a+rX "${DST}"

# ---- verify, loudly: these are the exact failure modes seen on the cluster ---
blocked=$(find "${DST}" -type d ! -perm -o+x 2>/dev/null | wc -l)
[ "${blocked}" = 0 ] || die "${blocked} dirs still not world-traversable."
leaked=$({ grep -rl "${SRC_INTERP}" "${DST}/venv/bin" 2>/dev/null || true; } | wc -l)
[ "${leaked}" = 0 ] || die "${leaked} file(s) still reference ${SRC_INTERP}."
"${DST}/venv/bin/python" -c "import hypotest, sys; assert sys.base_prefix.startswith('${DST}'), sys.base_prefix" \
  || die "the copied interpreter cannot import hypotest."

log "OK -> ${DST}  ($(du -sh "${DST}" | cut -f1))"
