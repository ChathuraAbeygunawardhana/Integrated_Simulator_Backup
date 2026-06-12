#!/usr/bin/env bash
# check_env.sh -- verifies the toolchain. Exit 0 on success; non-zero on missing
# required components. Vivado is optional and only warned.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

fail=0
warn=0

say()  { printf "  %-30s %s\n" "$1" "$2"; }
ok()   { say "$1" "OK $2"; }
miss() { say "$1" "MISSING $2"; fail=$((fail+1)); }
note() { say "$1" "WARN $2";    warn=$((warn+1)); }

echo "[check_env] Required tools"
command -v verilator >/dev/null 2>&1 && ok verilator "($(verilator --version 2>/dev/null | head -1))" || miss verilator "(required by cocotb tests)"
command -v gtkwave   >/dev/null 2>&1 && ok gtkwave   "($(gtkwave --version 2>/dev/null | head -1))" || note gtkwave   "(VCD viewer; optional)"
command -v make      >/dev/null 2>&1 && ok make      ""                                            || miss make      "(install build-essential)"
command -v python3   >/dev/null 2>&1 && ok python3   "($(python3 --version))"                       || miss python3   ""

echo "[check_env] Python venv"
VENV="${ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
  ok venv "(${VENV})"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  for mod in cocotb numpy yaml pytest; do
    if python -c "import ${mod}" 2>/dev/null; then ok "py:${mod}" ""
    else miss "py:${mod}" "(pip install ${mod})"; fi
  done
  if python -c "import tensorflow" 2>/dev/null; then
    ok "py:tensorflow" "($(python -c 'import tensorflow as t; print(t.__version__)'))"
  else
    note "py:tensorflow" "(TF golden flow will be skipped)"
  fi
else
  miss venv "(${VENV} missing; run scripts/install_deps.sh)"
fi

echo "[check_env] Optional tools"
command -v vivado >/dev/null 2>&1 && ok vivado "($(vivado -version 2>/dev/null | head -1))" || note vivado "(Phase I STA flow will be skipped)"
command -v yosys  >/dev/null 2>&1 && ok yosys  "($(yosys -V 2>/dev/null | head -1))"         || note yosys  "(unused unless STA fallback enabled)"

echo
if (( fail > 0 )); then
  echo "[check_env] FAIL: ${fail} required tool(s) missing, ${warn} warning(s)."
  exit 1
fi
if (( warn > 0 )); then
  echo "[check_env] OK with ${warn} optional warning(s)."
else
  echo "[check_env] OK -- all tools present."
fi
exit 0
