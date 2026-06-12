#!/usr/bin/env bash
# install_deps.sh -- one-shot dependency install for the simulator framework.
# Idempotent; safe to re-run. Requires sudo for apt steps.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

echo "[install_deps] Detecting OS..."
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  echo "[install_deps]   ${PRETTY_NAME}"
else
  echo "[install_deps]   /etc/os-release missing; assuming Ubuntu."
fi

# ---------------------------------------------------------------------------
# 1. apt packages (verilator, gtkwave, build essentials)
# ---------------------------------------------------------------------------
need_apt=()
command -v verilator >/dev/null 2>&1 || need_apt+=(verilator)
command -v gtkwave   >/dev/null 2>&1 || need_apt+=(gtkwave)
command -v make      >/dev/null 2>&1 || need_apt+=(build-essential)
command -v git       >/dev/null 2>&1 || need_apt+=(git)
python3 -c 'import venv' >/dev/null 2>&1 || need_apt+=(python3-venv python3-full)

if (( ${#need_apt[@]} )); then
  echo "[install_deps] Installing apt packages: ${need_apt[*]}"
  if sudo -n true 2>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y "${need_apt[@]}"
  else
    echo "[install_deps] WARNING: sudo not available. Run manually:"
    echo "  sudo apt-get install -y ${need_apt[*]}"
  fi
else
  echo "[install_deps] All apt packages already present."
fi

# ---------------------------------------------------------------------------
# 2. Verilator >= 5.0 check (cocotb 2.x requires it). Build from source if old.
# ---------------------------------------------------------------------------
if command -v verilator >/dev/null 2>&1; then
  VER_MAJOR=$(verilator --version 2>/dev/null | head -1 | awk '{print $2}' | cut -d. -f1)
  if [[ "${VER_MAJOR}" -lt 5 ]]; then
    echo "[install_deps] WARNING: Verilator version is < 5.0 (got ${VER_MAJOR})."
    echo "[install_deps]   cocotb 2.x needs Verilator >= 5.0. To build from source:"
    cat <<'EOF'
        git clone https://github.com/verilator/verilator /tmp/verilator
        cd /tmp/verilator && git checkout v5.026
        autoconf && ./configure && make -j$(nproc)
        sudo make install
EOF
  fi
fi

# ---------------------------------------------------------------------------
# 3. Python virtualenv + pip packages
# ---------------------------------------------------------------------------
VENV="${ROOT}/.venv"
if [[ ! -d "${VENV}" ]]; then
  echo "[install_deps] Creating venv at ${VENV}"
  python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet \
    cocotb cocotb-bus \
    numpy pyyaml pytest \
    tensorflow-cpu==2.15.0 \
    matplotlib pandas tqdm

echo "[install_deps] Python venv ready: source ${VENV}/bin/activate"

# ---------------------------------------------------------------------------
# 4. Vivado (optional, for the timing-closure phase)
# ---------------------------------------------------------------------------
if command -v vivado >/dev/null 2>&1; then
  echo "[install_deps] Vivado found: $(vivado -version 2>/dev/null | head -1)"
else
  echo "[install_deps] Vivado not found. Phase I (timing closure) will skip the synthesis step."
  echo "[install_deps]   Install Vivado 2024.1+ manually for full timing reports."
fi

echo "[install_deps] Done."
