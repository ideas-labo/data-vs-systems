#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config: change only here
# =========================
TARGET_STRATEGY="rf"   # e.g., "1NN" | "kNN" | "kNNw" | "random" ...
# =========================

# Usage:
#   ./delete_by_strategy.sh         # dry-run (list only)
#   ./delete_by_strategy.sh --yes   # actually delete
#
# Run it inside the results directory.

CONFIRM="${1:-}"

ROOT_DIR="$(pwd)"

# Safety: refuse to run if we are not inside a directory named 'results'
if [[ "$(basename "${ROOT_DIR}")" != "results" ]]; then
  echo "ERROR: Please run this script inside the 'results' directory."
  echo "Current dir: ${ROOT_DIR}"
  exit 2
fi

if [[ -z "${TARGET_STRATEGY}" ]]; then
  echo "ERROR: TARGET_STRATEGY is empty. Set it at the top of the script."
  exit 3
fi

# Find target directories: results/run*/<TARGET_STRATEGY>/
mapfile -t TARGETS < <(
  find "${ROOT_DIR}" -type d -path "*/run*/${TARGET_STRATEGY}" -print | sort
)

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  echo "No directories found for strategy '${TARGET_STRATEGY}'. Nothing to do."
  exit 0
fi

echo "Found ${#TARGETS[@]} directories to delete (strategy='${TARGET_STRATEGY}'):"
for p in "${TARGETS[@]}"; do
  echo "  ${p}"
done

if [[ "${CONFIRM}" != "--yes" ]]; then
  echo
  echo "Dry-run only. Re-run with '--yes' to actually delete:"
  echo "  $0 --yes"
  exit 0
fi

echo
echo "Deleting..."
for p in "${TARGETS[@]}"; do
  rm -rf -- "${p}"
done

echo "Done. Deleted all '${TARGET_STRATEGY}' folders under run*/."
