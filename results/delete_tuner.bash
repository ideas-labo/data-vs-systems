#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config: change only here
# =========================
TARGET_ALGO="promisetune"   # e.g., "PromiseTune" | "FLASH" | "GA" ...
# =========================

# Usage:
#   ./delete_by_algo.sh         # dry-run (list only)
#   ./delete_by_algo.sh --yes   # actually delete
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

if [[ -z "${TARGET_ALGO}" ]]; then
  echo "ERROR: TARGET_ALGO is empty. Set it at the top of the script."
  exit 3
fi

# Collect directories matching:
#   .../run*/<any_strategy>/<TARGET_ALGO>
TARGETS=()

while IFS= read -r -d '' p; do
  parent_dir="$(basename "$(dirname "${p}")")"
  grandparent_dir="$(basename "$(dirname "$(dirname "${p}")")")"

  # Ensure exact structure: .../run*/strategy/algo
  if [[ "${grandparent_dir}" == run* ]]; then
    TARGETS+=("${p}")
  fi
done < <(find "${ROOT_DIR}" -type d -name "${TARGET_ALGO}" -print0)

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  echo "No directories found for algorithm '${TARGET_ALGO}' under run*/<strategy>/."
  exit 0
fi

# Sort for stable output
IFS=$'\n' TARGETS=($(printf '%s\n' "${TARGETS[@]}" | sort))
unset IFS

echo "Found ${#TARGETS[@]} directories to delete (algorithm='${TARGET_ALGO}'):"
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

echo "Done. Deleted all '${TARGET_ALGO}' folders under run*/<strategy>/."