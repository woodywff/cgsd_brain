#!/usr/bin/env bash
# CGSD-Brain MVP: 1-command reproduction
# Runs the 30-second smoke test that exercises the entire pipeline.
# Phase 2 will add --full mode for the 30-subject sweep.
#
# Python interpreter: we use the system VP312 (torch 2.5.1 + torch_geometric
# 2.7.0 + trimesh + numpy + scipy already installed). We deliberately do NOT
# create a project-local .venv — the project inherits the system interpreter
# to keep the dependency surface small. Override with `PYTHON=/path/to/python`
# if you need a different one.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/woody/vp/vp3.12/bin/python}"
echo "Using interpreter: $PYTHON"
$PYTHON --version

echo "Generating synthetic fsaverage5 surface (one-time)..."
$PYTHON scripts/synthesize_surface.py

echo "Running smoke test..."
PYTHONPATH=. $PYTHON -m cgsd_brain.run_all --smoke
