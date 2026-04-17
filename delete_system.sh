#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./delete_system.sh mysql           # dry-run (list only)
#   ./delete_system.sh mysql --yes     # actually delete
#
# Run it from the results directory, or pass --root to specify.

SYSTEM="${1:-}"
CONFIRM="${2:-}"

if [[ -z "${SYSTEM}" ]]; then
  echo "Usage: $0 <system_name> [--yes]"
  echo "Example: $0 mysql --yes"
  exit 1
fi

ROOT_DIR="$(pwd)"

# Safety: refuse to run if we are not inside a directory named 'results'
if [[ "$(basename "${ROOT_DIR}")" != "results" ]]; then
  echo "ERROR: Please run this script inside the 'results' directory."
  echo "Current dir: ${ROOT_DIR}"
  exit 2
fi

# Find target directories: results/run*/(1NN|kNN|kNNw)/*/<system>/
mapfile -t TARGETS < <(
  find "${ROOT_DIR}" -type d \
    \( -path "*/run*/1NN/*/${SYSTEM}" -o -path "*/run*/kNN/*/${SYSTEM}" -o -path "*/run*/kNNw/*/${SYSTEM}" \) \
    -print | sort
)

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  echo "No directories found for system '${SYSTEM}'. Nothing to do."
  exit 0
fi

echo "Found ${#TARGETS[@]} directories to delete (system='${SYSTEM}'):"
for p in "${TARGETS[@]}"; do
  echo "  ${p}"
done

if [[ "${CONFIRM}" != "--yes" ]]; then
  echo
  echo "Dry-run only. Re-run with '--yes' to actually delete:"
  echo "  $0 ${SYSTEM} --yes"
  exit 0
fi

echo
echo "Deleting..."
for p in "${TARGETS[@]}"; do
  rm -rf -- "${p}"
done

echo "Done. Deleted all '${SYSTEM}' folders under run*/{1NN,kNN,kNNw}/*/."
