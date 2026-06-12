#!/usr/bin/env bash
# =============================================================================
# install_deps.sh — one-shot dependency installer for the Quadramind
#                   Integrated Simulator (Linux / macOS).
#
# What it does
# ────────────
#   1.  Checks Python >= 3.10 is present.
#   2.  On Linux: installs system packages (verilator >=5, gtkwave, make, git)
#       via apt (Debian/Ubuntu) or dnf (Fedora/RHEL).  Requires sudo.
#   3.  On macOS: installs via Homebrew (brew install verilator gtkwave).
#   4.  Creates (or reuses) a Python venv at <repo-root>/.venv.
#   5.  Pip-installs every package in requirements.txt.
#   6.  Adds the optimizer source directory to a .pth file so local modules
#       (loop_optimizer, blocking_optimizer) are importable without setting
#       PYTHONPATH manually.
#   7.  Runs a quick smoke test to verify the install.
#   8.  Prints instructions for optional tools (Vivado, GTKWave).
#
# Usage
# ─────
#   chmod +x install_deps.sh
#   ./install_deps.sh                  # full install
#   ./install_deps.sh --no-system-pkg  # skip apt/brew (CI, Docker, etc.)
#   ./install_deps.sh --gpu            # install tensorflow[and-cuda] instead of -cpu
#
# Re-running is safe (idempotent).
# =============================================================================

set -euo pipefail

# ── Parse flags ──────────────────────────────────────────────────────────────
SKIP_SYSTEM_PKG=0
USE_GPU_TF=0
for arg in "$@"; do
  case "$arg" in
    --no-system-pkg) SKIP_SYSTEM_PKG=1 ;;
    --gpu)           USE_GPU_TF=1 ;;
    -h|--help)
      sed -n '/^# Usage/,/^# ─/p' "$0"
      exit 0
      ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
info()    { echo -e "\033[1;34m[install]\033[0m  $*"; }
ok()      { echo -e "\033[1;32m[  ok  ]\033[0m  $*"; }
warn()    { echo -e "\033[1;33m[ warn ]\033[0m  $*"; }
die()     { echo -e "\033[1;31m[ fail ]\033[0m  $*" >&2; exit 1; }
have()    { command -v "$1" >/dev/null 2>&1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"         # repo root = directory containing this script

# ── 1.  Python version check ─────────────────────────────────────────────────
info "Checking Python version..."
PYTHON=""
PYTHON_FALLBACK=""  # 3.14+ found but cocotb-incompatible

# cocotb requires Python 3.10-3.13; try those first.
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if have "$candidate"; then
    major=$("$candidate" -c 'import sys; print(sys.version_info[0])')
    minor=$("$candidate" -c 'import sys; print(sys.version_info[1])')
    ver=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    if [[ "$major" -ge 3 && "$minor" -ge 10 && "$minor" -le 13 ]]; then
      PYTHON="$candidate"
      ok "Found $PYTHON  (Python $ver)"
      break
    elif [[ "$major" -ge 3 && "$minor" -ge 14 && -z "$PYTHON_FALLBACK" ]]; then
      PYTHON_FALLBACK="$candidate"  # record but keep looking
    fi
  fi
done

if [[ -z "$PYTHON" ]]; then
  if [[ -n "$PYTHON_FALLBACK" ]]; then
    fbver=$("$PYTHON_FALLBACK" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    warn "Only Python $fbver found.  cocotb does not yet support Python >= 3.14."
    warn "RTL simulation unavailable; software_ref fallback still works."
    warn "Install Python 3.12: https://www.python.org/downloads/"
    warn "Continuing with Python $fbver for non-RTL packages..."
    PYTHON="$PYTHON_FALLBACK"
  else
    die "Python 3.10-3.13 not found. Install Python 3.12 and re-run."
  fi
fi

# ── 2.  System packages ───────────────────────────────────────────────────────
if [[ "$SKIP_SYSTEM_PKG" -eq 0 ]]; then
  info "Installing system packages..."

  # ── Linux (Debian / Ubuntu) ────────────────────────────────────────────────
  if [[ "$(uname -s)" == "Linux" ]] && have apt-get; then
    NEED_APT=()
    have verilator || NEED_APT+=(verilator)
    have gtkwave   || NEED_APT+=(gtkwave)
    have make      || NEED_APT+=(build-essential)
    have git       || NEED_APT+=(git)
    "$PYTHON" -c 'import venv' 2>/dev/null || NEED_APT+=(python3-venv python3-full)

    if (( ${#NEED_APT[@]} )); then
      if sudo -n true 2>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y "${NEED_APT[@]}"
      else
        warn "sudo not available.  Run manually then re-run this script:"
        warn "  sudo apt-get install -y ${NEED_APT[*]}"
      fi
    else
      ok "All apt packages already present."
    fi

    # Verilator >= 5.0 guard (cocotb 2.x requirement)
    if have verilator; then
      VER_MAJ=$(verilator --version 2>/dev/null | head -1 | awk '{print $2}' | cut -d. -f1)
      if [[ "${VER_MAJ:-0}" -lt 5 ]]; then
        warn "Verilator ${VER_MAJ}.x found — cocotb 2.x needs >= 5.0."
        warn "Build from source:"
        warn "  git clone https://github.com/verilator/verilator /tmp/verilator"
        warn "  cd /tmp/verilator && git checkout v5.026"
        warn "  autoconf && ./configure && make -j\$(nproc) && sudo make install"
      else
        ok "Verilator $(verilator --version 2>/dev/null | head -1)"
      fi
    fi

  # ── Linux (Fedora / RHEL / Rocky) ─────────────────────────────────────────
  elif [[ "$(uname -s)" == "Linux" ]] && have dnf; then
    NEED_DNF=()
    have verilator || NEED_DNF+=(verilator)
    have gtkwave   || NEED_DNF+=(gtkwave)
    have make      || NEED_DNF+=("@development-tools")
    have git       || NEED_DNF+=(git)

    if (( ${#NEED_DNF[@]} )); then
      sudo dnf install -y "${NEED_DNF[@]}" || warn "Some dnf packages failed; continuing."
    else
      ok "All dnf packages already present."
    fi

  # ── macOS (Homebrew) ───────────────────────────────────────────────────────
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    if ! have brew; then
      warn "Homebrew not found. Install it from https://brew.sh then re-run."
    else
      NEED_BREW=()
      have verilator || NEED_BREW+=(verilator)
      have gtkwave   || NEED_BREW+=(gtkwave)
      have make      || NEED_BREW+=(make)
      have git       || NEED_BREW+=(git)

      if (( ${#NEED_BREW[@]} )); then
        brew install "${NEED_BREW[@]}"
      else
        ok "All Homebrew packages already present."
      fi
    fi

  else
    warn "Unsupported OS for automatic system-package install."
    warn "Install manually: verilator>=5.0, gtkwave, make, git."
  fi
else
  info "Skipping system package install (--no-system-pkg)."
fi

# ── 3.  Python virtual environment ───────────────────────────────────────────
VENV="$ROOT/.venv"
info "Setting up Python venv at $VENV ..."
if [[ ! -d "$VENV" ]]; then
  "$PYTHON" -m venv "$VENV"
  ok "Created venv."
else
  ok "Venv already exists — reusing."
fi

PIP="$VENV/bin/pip"
PYEXE="$VENV/bin/python"

"$PIP" install --quiet --upgrade pip setuptools wheel

# ── 4.  Install Python packages ───────────────────────────────────────────────
REQ="$ROOT/requirements.txt"
if [[ ! -f "$REQ" ]]; then
  die "requirements.txt not found at $REQ"
fi

if [[ "$USE_GPU_TF" -eq 1 ]]; then
  info "Installing Python packages (GPU TensorFlow) ..."
  # Replace the cpu variant with the CUDA-enabled one
  TMP_REQ=$(mktemp /tmp/req_XXXXXX.txt)
  sed 's/tensorflow-cpu/tensorflow/g' "$REQ" > "$TMP_REQ"
  "$PIP" install --quiet -r "$TMP_REQ"
  rm -f "$TMP_REQ"
else
  info "Installing Python packages from requirements.txt ..."
  "$PIP" install --quiet -r "$REQ"
fi
ok "Python packages installed."

# ── 5.  Register local optimizer source on sys.path (via .pth) ───────────────
#        This makes `import loop_optimizer` and `import blocking_optimizer`
#        work anywhere inside the venv without setting PYTHONPATH.
SITE_PKG=$("$PYEXE" -c \
  "import sysconfig; print(sysconfig.get_path('purelib'))")

OPT_SRC="$ROOT/systolic_optimizer_codebase/systolic_optimizer/src"
PTH_FILE="$SITE_PKG/quadramind_local.pth"

if [[ -d "$OPT_SRC" ]]; then
  # Also add sim_framework itself (for `import pysim`)
  {
    echo "$OPT_SRC"
    echo "$ROOT/sim_framework"
    echo "$ROOT/vivado_flow/optimizer"
  } > "$PTH_FILE"
  ok "Registered local source paths in $PTH_FILE"
else
  warn "Optimizer src dir not found at $OPT_SRC — skipping .pth registration."
fi

# ── 6.  Smoke test ────────────────────────────────────────────────────────────
info "Running smoke test ..."

SMOKE_PASS=1

# 6a. Core imports
"$PYEXE" - <<'PYSMOKE'
import sys, importlib
failures = []
for mod in ["numpy", "yaml", "pytest", "tensorflow", "cocotb", "matplotlib", "tqdm"]:
    try:
        importlib.import_module(mod)
    except ImportError as e:
        failures.append(f"  {mod}: {e}")
if failures:
    print("FAILED imports:")
    print("\n".join(failures))
    sys.exit(1)
print("  All core Python imports OK.")
PYSMOKE
[[ $? -eq 0 ]] || SMOKE_PASS=0

# 6b. Local optimizer modules
"$PYEXE" - <<'PYLOCAL'
import sys
try:
    from loop_optimizer import LoopOptimizer
    from blocking_optimizer import BlockingOptimizer
    print("  Local optimizer modules importable OK.")
except ImportError as e:
    print(f"  WARNING: local optimizer import failed: {e}")
    print("  Set PYTHONPATH manually if needed (see README).")
PYLOCAL

# 6c. config_knobs.py self-test
"$PYEXE" "$ROOT/vivado_flow/optimizer/config_knobs.py" 2>/dev/null \
  && ok "config_knobs.py self-test passed." \
  || warn "config_knobs.py self-test returned non-zero — check manually."

if [[ "$SMOKE_PASS" -eq 1 ]]; then
  ok "Smoke test passed."
else
  warn "Some smoke tests failed (see above).  The venv may still be usable."
fi

# ── 7.  Optional tools notice ─────────────────────────────────────────────────
echo ""
info "Optional tools:"
if have vivado; then
  ok "  Vivado: $(vivado -version 2>/dev/null | head -1)"
else
  warn "  Vivado not found.  Timing-closure (Phase I) will be skipped."
  warn "  Download Vivado 2024.1+ from https://www.xilinx.com/support/download.html"
fi

if have verilator; then
  ok "  Verilator: $(verilator --version 2>/dev/null | head -1)"
else
  warn "  Verilator not found.  RTL simulation will use software_ref fallback."
fi

if have gtkwave; then
  ok "  GTKWave: $(gtkwave --version 2>/dev/null | head -1)"
else
  warn "  GTKWave not found.  Waveform viewing unavailable."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
ok "Install complete."
echo ""
echo "  To activate the venv:"
echo "    source $VENV/bin/activate"
echo ""
echo "  Quick start:"
echo "    cd $ROOT/sim_framework"
echo "    source ../.venv/bin/activate"
echo "    make smoke            # software-ref regression (no Verilator needed)"
echo "    make full_test        # full pytest suite"
echo ""
echo "  Generate optimal config from a TF model:"
echo "    cd $ROOT/vivado_flow/optimizer"
echo "    python generate_config.py --tf-model mobilenet_v2"
echo "    python config_knobs.py --export-vh"
