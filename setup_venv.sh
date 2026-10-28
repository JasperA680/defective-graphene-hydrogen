#!/usr/bin/env bash
#
# Create the .venv-dos virtual environment the report's commands refer to.
#
#   bash setup_venv.sh
#   .venv-dos/bin/python3 scripts/visualisation/plot_n_membered_rings_all.py
#
# The environment is named .venv-dos because that is the name used in the paper's
# Data and Code Availability section, so the commands quoted there work verbatim.
# Python 3.10 or newer is required. If python3 on PATH is older, point PYTHON at a
# newer interpreter (PYTHON=python3.11 bash setup_venv.sh) or, on an HPC system,
# load one first (on ARCHER2: module load cray-python).

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if [ -d .venv-dos ]; then
    echo ".venv-dos already exists; delete it first to rebuild."
    exit 0
fi

if ! { "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; } 2>/dev/null; then
    found="$({ "$PYTHON" --version; } 2>/dev/null || echo "not found")"
    echo "error: Python 3.10 or newer is required, but $PYTHON is $found." >&2
    echo "Rerun with PYTHON=/path/to/python3.10+ bash setup_venv.sh," \
         "or load a newer Python first (on ARCHER2: module load cray-python)." >&2
    exit 1
fi

# Remove a half-built environment on failure, so a rerun does not stop at
# ".venv-dos already exists" with nothing installed.
trap 'rm -rf .venv-dos' ERR
"$PYTHON" -m venv .venv-dos
.venv-dos/bin/pip install --upgrade pip
.venv-dos/bin/pip install -r requirements.txt
trap - ERR

echo
echo "Done. Reproduce Figure 3 with:"
echo "    .venv-dos/bin/python3 scripts/visualisation/plot_n_membered_rings_all.py"
echo "    .venv-dos/bin/python3 scripts/visualisation/plot_ring_topology_census.py"
