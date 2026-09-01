#!/usr/bin/env bash
#
# Create the .venv-dos virtual environment the report's commands refer to.
#
#   bash setup_venv.sh
#   .venv-dos/bin/python3 scripts/visualisation/plot_n_membered_rings_all.py
#
# The environment is named .venv-dos because that is the name used in the paper's
# Data and Code Availability section, so the commands quoted there work verbatim.
# Python 3.10 or newer is required.

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if [ -d .venv-dos ]; then
    echo ".venv-dos already exists; delete it first to rebuild."
    exit 0
fi

"$PYTHON" -m venv .venv-dos
.venv-dos/bin/pip install --upgrade pip
.venv-dos/bin/pip install -r requirements.txt

echo
echo "Done. Reproduce Figure 3 with:"
echo "    .venv-dos/bin/python3 scripts/visualisation/plot_n_membered_rings_all.py"
echo "    .venv-dos/bin/python3 scripts/visualisation/plot_ring_topology_census.py"
