#Requires -Version 5.1
<#
.SYNOPSIS
    One-shot dependency installer for the Quadramind Integrated Simulator (Windows).

.DESCRIPTION
    1. Checks Python >= 3.10 is on PATH.
    2. Creates (or reuses) a Python venv at <repo-root>\.venv.
    3. Pip-installs every package listed in requirements.txt.
    4. Registers local source directories (loop_optimizer, blocking_optimizer,
       pysim) via a .pth file so they are importable without setting PYTHONPATH.
    5. Runs a quick smoke test.
    6. Prints guidance for optional tools (Verilator via MSYS2/WSL, Vivado).

    Verilator is not natively available on Windows.  The script installs
    everything that works natively and prints WSL / MSYS2 instructions for
    the RTL-simulation path (the framework falls back to software_ref.py
    automatically if Verilator is absent, so all pytest tests still pass).

.PARAMETER NoSystemPkg
    Skip attempts to install system packages via winget / choco (CI use).

.PARAMETER Gpu
    Install the CUDA-enabled 'tensorflow' wheel instead of 'tensorflow-cpu'.

.PARAMETER ForceTf
    Force TensorFlow installation even when only Python 3.14+ is present.
    This installs Python 3.12 via winget (requires an internet connection and
    winget on PATH), then recreates the venv using that interpreter so the
    full requirements.txt -- including tensorflow-cpu -- can be installed.
    If winget is unavailable, prints a manual download URL instead.

.EXAMPLE
    .\install_deps.ps1
    .\install_deps.ps1 -NoSystemPkg
    .\install_deps.ps1 -Gpu
    .\install_deps.ps1 -ForceTf          # install Python 3.12 + full deps
    .\install_deps.ps1 -ForceTf -Gpu     # same, but with GPU tensorflow
#>

param(
    [switch]$NoSystemPkg,
    [switch]$Gpu,
    [switch]$ForceTf
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Helpers ───────────────────────────────────────────────────────────────────
function Info  ($msg) { Write-Host "[install]  $msg" -ForegroundColor Cyan    }
function Ok    ($msg) { Write-Host "[  ok  ]  $msg" -ForegroundColor Green   }
function Warn  ($msg) { Write-Host "[ warn ]  $msg" -ForegroundColor Yellow  }
function Fail  ($msg) { Write-Host "[ fail ]  $msg" -ForegroundColor Red; exit 1 }
function Have  ($cmd) { $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue) }

$Root = $PSScriptRoot           # directory containing this script = repo root
$UsingFallbackPython = $false   # set to $true when only Python 3.14+ was found

# ── Pip cache / temp redirect ─────────────────────────────────────────────────
# pip downloads wheels to %TEMP% before installing, then caches them under
# %LOCALAPPDATA%\pip\cache.  If the system drive (C:) is nearly full either
# operation fails with "No space left on device".
# Solution: if D:\ has more free space than C:\, redirect both to D:\ for the
# duration of this script and persist the cache-dir via pip config.
$CDrive = Get-PSDrive C -ErrorAction SilentlyContinue
$DDrive = Get-PSDrive D -ErrorAction SilentlyContinue
if ($DDrive -and $DDrive.Free -gt ($CDrive.Free + 512MB)) {
    $PipCacheDir = "D:\pip_cache"
    $PipTmpDir   = "D:\pip_tmp"
    New-Item -ItemType Directory -Force $PipCacheDir | Out-Null
    New-Item -ItemType Directory -Force $PipTmpDir   | Out-Null
    $env:TEMP = $PipTmpDir
    $env:TMP  = $PipTmpDir
    $env:PIP_CACHE_DIR = $PipCacheDir
    Info "C:\ low on space -- pip cache/temp redirected to D:\\"
}

# ── 1. Python version check ───────────────────────────────────────────────────
Info "Checking Python version..."
$PythonExe   = $null
$PythonExeFallback = $null   # 3.14+ found but cocotb incompatible

# cocotb requires Python 3.10-3.13; try those first.
foreach ($candidate in @("python3.13","python3.12","python3.11","python3.10",
                          "python3","python")) {
    if (Have $candidate) {
        $ver = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver) {
            $parts = $ver.Split(".")
            $major = [int]$parts[0]; $minor = [int]$parts[1]
            if ($major -ge 3 -and $minor -ge 10 -and $minor -le 13) {
                $PythonExe = $candidate
                Ok "Found $PythonExe  (Python $ver)"
                break
            } elseif ($major -ge 3 -and $minor -ge 14 -and -not $PythonExeFallback) {
                $PythonExeFallback = $candidate   # record but keep looking
            }
        }
    }
}
if (-not $PythonExe) {
    if ($PythonExeFallback) {
        $fbver = & $PythonExeFallback -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        Warn "Only Python $fbver found.  cocotb does not yet support Python >= 3.14."
        Warn "RTL simulation will be unavailable; software_ref fallback will still work."
        Warn "Recommended: install Python 3.12 from https://www.python.org/downloads/"
        Warn "Continuing with Python $fbver for non-RTL packages..."
        $PythonExe = $PythonExeFallback
        $UsingFallbackPython = $true
    } else {
        Fail "Python 3.10-3.13 not found on PATH.`n  Download Python 3.12 from https://www.python.org/downloads/"
    }
}

# ── 1b. ForceTf: install Python 3.12 if we are stuck on 3.14+ ────────────────
if ($ForceTf -and $UsingFallbackPython) {
    Info "-ForceTf specified -- attempting to install Python 3.12 via winget..."

    if (Have winget) {
        # winget id for the official Python 3.12 release from python.org
        winget install --id Python.Python.3.12 -e --silent `
            --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -eq 0) {
            Ok "Python 3.12 installed via winget."
        } else {
            Warn "winget returned non-zero. Python 3.12 may or may not have installed."
        }

        # Refresh PATH so the new python3.12 is visible in this session
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("PATH","User")

        # Re-probe for Python 3.12
        $py312 = $null
        foreach ($c in @("python3.12","python3","python")) {
            if (Have $c) {
                $v = & $c -c "import sys; print(sys.version_info.minor)" 2>$null
                if ([int]$v -eq 12) { $py312 = $c; break }
            }
        }

        if ($py312) {
            Ok "Found $py312 after winget install -- switching interpreter."
            $PythonExe = $py312
            $UsingFallbackPython = $false
        } else {
            Warn "python3.12 still not on PATH after winget install."
            Warn "Open a new PowerShell window and re-run: .\install_deps.ps1 -ForceTf"
            Warn "Or download manually: https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
            Warn "Continuing without tensorflow for this run..."
        }

    } else {
        Warn "winget not available -- cannot auto-install Python 3.12."
        Warn "Download Python 3.12 manually, add to PATH, then re-run:"
        Warn "  https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
        Warn "Continuing without tensorflow for this run..."
    }

    # If we now have Python 3.12, wipe the old 3.14 venv so it is rebuilt correctly
    if (-not $UsingFallbackPython) {
        $VenvDirEarly = Join-Path $Root ".venv"
        if (Test-Path $VenvDirEarly) {
            Info "Removing old Python 3.14 venv so it can be rebuilt with Python 3.12..."
            Remove-Item $VenvDirEarly -Recurse -Force
        }
    }
}

# ── 2. Optional system-level installs ────────────────────────────────────────
if (-not $NoSystemPkg) {
    Info "Checking for optional system tools..."

    # Try winget first, then Chocolatey, then just warn.
    function TryInstall($pkg_winget, $pkg_choco, $label) {
        if (Have winget) {
            Info "  Installing $label via winget..."
            try { winget install --id $pkg_winget -e --silent 2>$null } catch {}
        } elseif (Have choco) {
            Info "  Installing $label via choco..."
            try { choco install $pkg_choco -y 2>$null } catch {}
        } else {
            Warn "  $label not found and no package manager available (winget/choco)."
        }
    }

    if (-not (Have git)) { TryInstall "Git.Git" "git" "git" }
    if (-not (Have make)) {
        Warn "  'make' not found. Install via:"
        Warn "    winget install GnuWin32.Make  (or use 'nmake' from VS Build Tools)"
    }
} else {
    Info "Skipping system package install (-NoSystemPkg)."
}

# ── 3a. Vivado auto-discovery ────────────────────────────────────────────────
# Look for vivado.bat in common install roots.  If found, persist VIVADO_BIN as
# a User environment variable so that .venv\Scripts\Activate.ps1 can add it to
# PATH automatically on every activation.
if (-not (Have vivado)) {
    $VivadoBin = $null
    $VivadoRoots = @("C:\Xilinx", "D:\Xilinx", "C:\AMD", "D:\AMD")
    foreach ($root in $VivadoRoots) {
        if (Test-Path $root) {
            $found = Get-ChildItem $root -Recurse -Filter "vivado.bat" -ErrorAction SilentlyContinue |
                     Select-Object -First 1
            if ($found) { $VivadoBin = $found.DirectoryName; break }
        }
    }

    if ($VivadoBin) {
        [System.Environment]::SetEnvironmentVariable("VIVADO_BIN", $VivadoBin, "User")
        $env:VIVADO_BIN = $VivadoBin     # also set for current session
        $env:PATH       = "$VivadoBin;$env:PATH"
        Ok "Vivado found and VIVADO_BIN set: $VivadoBin"
    } else {
        Warn "Vivado not found on PATH and not auto-detected."
        Warn "  If Vivado is installed, set the VIVADO_BIN user env var once:"
        Warn "    [System.Environment]::SetEnvironmentVariable('VIVADO_BIN','C:\Xilinx\Vivado\2024.1\bin','User')"
        Warn "  The venv Activate.ps1 will add it to PATH automatically after that."
        Warn "  Or pass the full path at runtime:"
        Warn "    python scripts/iterate_until_pass.py --vivado 'C:\Xilinx\Vivado\2024.1\bin\vivado.bat' ..."
    }
} else {
    $vvver = (vivado -version 2>$null | Select-Object -First 1)
    Ok "Vivado already on PATH: $vvver"
    # Persist VIVADO_BIN if not already set
    if (-not $env:VIVADO_BIN) {
        $vvBin = (Get-Command vivado).Source | Split-Path -Parent
        [System.Environment]::SetEnvironmentVariable("VIVADO_BIN", $vvBin, "User")
        $env:VIVADO_BIN = $vvBin
        Ok "VIVADO_BIN saved: $vvBin"
    }
}

# ── 3. Verilator / GTKWave guidance (Windows-specific) ───────────────────────
if (-not (Have verilator)) {
    Warn "Verilator not found on PATH."
    Warn "  On Windows, the recommended paths are:"
    Warn "  A) WSL 2 (Ubuntu):"
    Warn "       wsl -- sudo apt-get install verilator"
    Warn "       # Then run the Linux install_deps.sh from inside WSL."
    Warn "  B) MSYS2 (MinGW64 shell):"
    Warn "       pacman -S mingw-w64-x86_64-verilator"
    Warn "  C) Run without Verilator -- the framework auto-falls back to"
    Warn "     software_ref.py (NumPy); all pytest tests still pass."
} else {
    $verlver = (verilator --version 2>$null | Select-Object -First 1)
    Ok "Verilator: $verlver"
}

# ── 4. Python virtual environment ─────────────────────────────────────────────
$VenvDir = Join-Path $Root ".venv"
Info "Setting up Python venv at $VenvDir ..."
if (-not (Test-Path $VenvDir)) {
    & $PythonExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Fail "python -m venv returned exit code $LASTEXITCODE -- check your Python installation."
    }
    Ok "Created venv."
} else {
    Ok "Venv already exists -- reusing."
}

$PyExe    = Join-Path $VenvDir "Scripts\python.exe"
$ActPs1   = Join-Path $VenvDir "Scripts\Activate.ps1"

# On some Windows Python installs 'python -m venv' creates the venv but omits
# the activation scripts (Activate.ps1 / activate.bat / deactivate.bat).
# Copy them from the Python installation's venv template directory if missing.
if (-not (Test-Path $ActPs1)) {
    Info "Activate.ps1 missing from venv -- copying from Python venv templates..."
    $PyHome   = & $PythonExe -c "import sys; print(sys.prefix)"
    $TplBase  = Join-Path $PyHome "Lib\venv\scripts"
    $Scripts  = Join-Path $VenvDir "Scripts"
    foreach ($src in @(
        (Join-Path $TplBase "common\Activate.ps1"),
        (Join-Path $TplBase "nt\activate.bat"),
        (Join-Path $TplBase "nt\deactivate.bat")
    )) {
        if (Test-Path $src) {
            Copy-Item $src $Scripts -Force
        } else {
            Warn "  Template not found (skipped): $src"
        }
    }
    if (Test-Path $ActPs1) {
        Ok "Activation scripts restored."
    } else {
        Warn "Could not restore Activate.ps1 -- activate the venv manually with: & '$Scripts\python.exe'"
    }
}

# Verify pip is importable inside the venv.  We deliberately use 'python -m pip'
# rather than testing for a specific pip.exe / pip3.exe / pip3.12.exe, because
# ensurepip on Windows installs whichever versioned name matches the interpreter
# and does not always create a bare 'pip.exe'.  'python -m pip' always works.
& $PyExe -m pip --version 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Info "pip not importable in venv -- bootstrapping via ensurepip..."
    & $PyExe -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) {
        Fail "ensurepip failed (exit $LASTEXITCODE).  Run manually: & '$PyExe' -m ensurepip --upgrade"
    }
    & $PyExe -m pip --version 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Fail "pip still not importable after ensurepip.  Inspect the venv at $VenvDir."
    }
    Ok "pip bootstrapped via ensurepip."
}

# pip 26+ must be upgraded via 'python -m pip', not 'pip install --upgrade pip'
& $PyExe -m pip install --quiet --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Warn "pip self-upgrade returned exit $LASTEXITCODE -- proceeding with the current version."
} else {
    Ok "pip / setuptools / wheel upgraded."
}

# ── 5. Install Python packages ────────────────────────────────────────────────
$ReqFile = Join-Path $Root "requirements.txt"
if (-not (Test-Path $ReqFile)) {
    Fail "requirements.txt not found at $ReqFile"
}

# Build the effective requirements content, applying any requested substitutions.
$ReqLines = Get-Content $ReqFile

if ($Gpu) {
    $ReqLines = $ReqLines -replace 'tensorflow-cpu', 'tensorflow'
}

if ($UsingFallbackPython) {
    # These packages have no Python 3.14 wheels yet -- skip them so the rest install.
    #   cocotb / find-libpython  : RTL simulation layer
    #   tensorflow-cpu/tensorflow: no 3.14 wheels released yet; use explicit dims instead
    Info "Python 3.14+ detected -- skipping cocotb and tensorflow (no 3.14 wheels yet)."
    $ReqLines = $ReqLines | Where-Object {
        $_ -notmatch '^\s*cocotb' -and
        $_ -notmatch '^\s*find-libpython' -and
        $_ -notmatch '^\s*tensorflow'
    }
    Warn "tensorflow skipped -- use explicit --K/--C/... flags with generate_config.py instead of --tf-model."
}

$TmpReq = [System.IO.Path]::GetTempFileName() + ".txt"
$ReqLines | Set-Content $TmpReq -Encoding utf8

Info "Installing Python packages..."
# --no-cache-dir avoids stale / corrupt cache entries from a previous Python version.
# --retries 5 recovers from transient network drops (tensorflow-cpu is ~350 MB).
& $PyExe -m pip install --quiet --no-cache-dir --retries 5 -r $TmpReq
$pipExit = $LASTEXITCODE
Remove-Item $TmpReq -Force

if ($pipExit -ne 0) {
    Warn "pip exited with code $pipExit -- some packages may not have installed."
    Warn "Re-run after fixing the error, or check the output above."
} else {
    Ok "Python packages installed."
}

# ── 6. Register local source paths via .pth ──────────────────────────────────
#       Makes loop_optimizer, blocking_optimizer, and pysim importable
#       without manually setting PYTHONPATH each session.
Info "Registering local source paths..."
$SitePkg = & $PyExe -c "import sysconfig; print(sysconfig.get_path('purelib'))"
$PthFile = Join-Path $SitePkg "quadramind_local.pth"

$OptimizerSrc = Join-Path $Root "systolic_optimizer_codebase\systolic_optimizer\src"
$SimFramework = Join-Path $Root "sim_framework"
$VivadoOpt    = Join-Path $Root "vivado_flow\optimizer"

$PthLines = @()
foreach ($p in @($OptimizerSrc, $SimFramework, $VivadoOpt)) {
    if (Test-Path $p) {
        $PthLines += $p
    } else {
        Warn "  Path not found (skipped): $p"
    }
}

if ($PthLines.Count -gt 0) {
    $PthLines | Set-Content $PthFile -Encoding utf8
    Ok "Registered $($PthLines.Count) path(s) in $PthFile"
}

# ── 7. Smoke test ─────────────────────────────────────────────────────────────
Info "Running smoke test..."

# Build the smoke-test module list dynamically: always test core packages,
# only test tensorflow if it was actually installed.
$SmokeModules = '"numpy", "yaml", "pytest", "matplotlib", "tqdm"'
if (-not $UsingFallbackPython) {
    $SmokeModules = '"numpy", "yaml", "pytest", "tensorflow", "matplotlib", "tqdm"'
}

$SmokeScript = @"
import sys, importlib
failures = []
for mod in [$SmokeModules]:
    try:
        importlib.import_module(mod)
    except ImportError as e:
        failures.append(f'  {mod}: {e}')
if failures:
    print('FAILED imports:')
    print('\n'.join(failures))
    sys.exit(1)
print('  All installed core imports OK.')
"@

$tmpPy = [System.IO.Path]::GetTempFileName() + ".py"
$SmokeScript | Set-Content $tmpPy -Encoding utf8
& $PyExe $tmpPy        # output goes directly to console
$smokeRc = $LASTEXITCODE   # $LASTEXITCODE is the only reliable way to check native-exe exit
Remove-Item $tmpPy -Force

if ($smokeRc -eq 0) {
    Ok "Core import smoke test passed."
} else {
    Warn "Some core imports failed (see above)."
}

# cocotb check -- use 'pip show' to avoid triggering stderr/NativeCommandError
$cocotbInfo = & $PyExe -m pip show cocotb
if ($LASTEXITCODE -eq 0) {
    $cocotbVer = ($cocotbInfo | Where-Object { $_ -match '^Version:' }) -replace 'Version:\s*', ''
    Ok "  cocotb $cocotbVer"
} else {
    if ($UsingFallbackPython) {
        Warn "  cocotb skipped (Python 3.14 unsupported) -- install Python 3.12 for RTL simulation."
    } else {
        Warn "  cocotb not installed -- RTL simulation unavailable."
    }
    Warn "  All pytest tests still pass via the software_ref fallback."
}

# Local optimizer module check -- use pip show to check .pth registration
$loSrc = Join-Path $Root "systolic_optimizer_codebase\systolic_optimizer\src\loop_optimizer.py"
if (Test-Path $loSrc) {
    # .pth was registered; do a quick import via a temp script to confirm
    $loCheck = [System.IO.Path]::GetTempFileName() + ".py"
    "from loop_optimizer import LoopOptimizer; print('loop_optimizer OK')" | Set-Content $loCheck -Encoding utf8
    & $PyExe $loCheck
    $loRc = $LASTEXITCODE
    Remove-Item $loCheck -Force
    if ($loRc -ne 0) {
        Warn "  loop_optimizer import failed -- open a new shell for .pth to take effect."
    }
} else {
    Warn "  loop_optimizer source not found at expected path."
}

# config_knobs self-test
$ConfigKnobsPath = Join-Path $Root "vivado_flow\optimizer\config_knobs.py"
if (Test-Path $ConfigKnobsPath) {
    & $PyExe $ConfigKnobsPath | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Ok "config_knobs.py self-test passed."
    } else {
        Warn "config_knobs.py self-test returned non-zero -- check manually."
    }
}

# ── 8. Optional tools summary ─────────────────────────────────────────────────
Write-Host ""
Info "Optional tools summary:"
if (Have vivado) {
    $vvver = (vivado -version 2>$null | Select-Object -First 1)
    Ok "  Vivado: $vvver"
} else {
    Warn "  Vivado not found -- timing-closure (Phase I) will be skipped."
    Warn "  Download Vivado 2024.1+ from https://www.xilinx.com/support/download.html"
}
if (Have gtkwave) {
    Ok "  GTKWave: $(gtkwave --version 2>$null | Select-Object -First 1)"
} else {
    Warn "  GTKWave not found -- install from https://gtkwave.sourceforge.net/"
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Ok "Install complete."
Write-Host ""
Write-Host "  To activate the venv (PowerShell):"
Write-Host "    $VenvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  Quick start (from $Root\sim_framework):"
Write-Host "    cd $Root\sim_framework"
Write-Host "    ..\\.venv\Scripts\python.exe -m pytest tb\ -v"
Write-Host ""
Write-Host "  Generate optimal config from a TF model:"
Write-Host "    cd $Root\vivado_flow\optimizer"
Write-Host "    python generate_config.py --tf-model mobilenet_v2"
Write-Host "    python config_knobs.py --export-vh"
Write-Host ""
Write-Host "  NOTE: For full RTL simulation on Windows, use WSL 2:"
Write-Host "    wsl --install -d Ubuntu        # one-time"
Write-Host "    wsl                            # enter WSL shell"
Write-Host "    cd /mnt/d/path/to/repo"
Write-Host "    ./install_deps.sh"
