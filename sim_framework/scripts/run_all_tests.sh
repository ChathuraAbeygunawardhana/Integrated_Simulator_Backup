#!/usr/bin/env bash
# Master test runner: pytest + matrix sweep + timing-closure (if Vivado).
set -uo pipefail

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "${ROOT}"

[[ -f .venv/bin/activate ]] && source .venv/bin/activate

echo "=== [1/4] Lint RTL ==="
if command -v verilator >/dev/null 2>&1; then
  make -C rtl -f Makefile.lint lint || true
else
  echo "[skip] verilator not installed"
fi

echo
echo "=== [2/4] Pytest (unit + integration + timing) ==="
PYTHONPATH="${ROOT}" python -m pytest tb/ -v

echo
echo "=== [3/4] Matrix sweep + reports ==="
PYTHONPATH="${ROOT}" python scripts/iterate_until_pass.py --max-iter 2

echo
echo "=== [4/4] Vivado timing closure (optional) ==="
if command -v vivado >/dev/null 2>&1; then
  vivado -mode batch -source timing/vivado/synth_single_dnn.tcl -tclargs "${ROOT}" || true
  vivado -mode batch -source timing/vivado/synth_multi_dnn.tcl  -tclargs "${ROOT}" || true
else
  echo "[skip] vivado not installed"
fi

echo
echo "=== Done. See docs/verification_matrix.md and build/matrix_results.csv ==="
