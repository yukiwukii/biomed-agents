#!/usr/bin/env bash
# rl/scripts/with_secrets.sh
# =============================================================================
# Export each file in a mounted Secret volume as an environment variable, then
# exec the rest of the command line.
#
#     with_secrets.sh python .../dataset_server.py /config/server.yaml
#
# WHY THIS EXISTS
# ---------------
# Run:ai's workload CRDs have no way to inject a Secret into the environment:
#
#   kubectl explain interactiveworkloads.spec | grep -i secret
#   -> imagePullSecrets, secretVolumes        (no envSecrets, no envFrom)
#   kubectl explain interactiveworkloads.spec.environment.items.value
#   -> FIELD: value <string>                  (a literal string, nothing else)
#
# The only alternative would be putting API keys as plaintext into the workload
# spec, where anyone with `get trainingworkload` could read them. Mounting the
# Secret as files and exporting them here keeps the keys in a Secret.
#
# Kubernetes projects each Secret key as a file named after the key, so
# /secrets/ANTHROPIC_API_KEY contains the value and nothing else.
#
# Empty values are skipped rather than exported as "" -- an empty
# ANTHROPIC_API_KEY would make litellm fail deep inside the judge call, whereas
# an unset one fails immediately and legibly.
# =============================================================================
set -euo pipefail

SECRETS_DIR="${SECRETS_DIR:-/secrets}"

if [ -d "${SECRETS_DIR}" ]; then
  exported=0
  for f in "${SECRETS_DIR}"/*; do
    [ -f "${f}" ] || continue
    key="$(basename "${f}")"
    # Kubernetes uses ..data/..2024_01_01 symlink dirs for atomic updates.
    case "${key}" in ..*) continue ;; esac
    value="$(cat "${f}")"
    [ -n "${value}" ] || continue
    export "${key}=${value}"
    exported=$((exported + 1))
  done
  echo ">> with_secrets: exported ${exported} variable(s) from ${SECRETS_DIR}" >&2
else
  echo ">> with_secrets: ${SECRETS_DIR} not mounted; continuing with the ambient environment" >&2
fi

[ "$#" -gt 0 ] || { echo "with_secrets.sh: no command given" >&2; exit 2; }
exec "$@"
