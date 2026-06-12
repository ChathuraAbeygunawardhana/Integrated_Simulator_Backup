# Vivado Flow — Systolic Array Loop-Optimizer Pipeline

End-to-end pipeline that:
1. Extracts a DNN layer from TensorFlow **or** accepts raw dimensions
2. Runs the loop-order + tile-size optimisers
3. Emits Verilog header files the testbench `\`include`s
4. Generates golden reference data
5. Drives a parameterised Vivado testbench
6. Iterates automatically until all tests pass

---

## Directory layout

```
vivado_flow/
├── optimizer/
│   ├── generate_config.py   Stage 1 — TF model / dims → .vh + .json
│   └── config_knobs.py      Stage 2 — user knobs → user_config.vh
│
├── rtl/tb/
│   └── systolic_tb.sv       Parameterised Vivado testbench
│
├── scripts/
│   ├── generate_golden.py   Stage 3 — write tb_data/ hex memory files
│   ├── run_vivado_sim.tcl   Stage 4 — Vivado batch-mode TCL script
│   └── iterate_until_pass.py  Master loop (Stages 1-4, auto-retry)
│
├── configs/generated/       Auto-written — do not commit
│   ├── optimal_config.vh    Verilog defines from the optimiser
│   ├── user_config.vh       Verilog defines from user overrides
│   └── optimal_config.json  Machine-readable optimiser summary
│
├── tb_data/                 Auto-written — do not commit
│   ├── input_hex.mem        $readmemh input activations
│   ├── weight_hex.mem       $readmemh weights
│   └── golden_hex.mem       $readmemh expected output
│
├── logs/                    Per-iteration log files
└── vivado_project/          Vivado project (auto-created)
```

---

## Quick start

### Prerequisites

```powershell
# Python ≥ 3.9 with numpy
pip install numpy

# Optional: TensorFlow (needed for --tf-model)
pip install tensorflow

# Vivado 2022.x or newer on PATH (only needed for RTL simulation)
```

### One command — full dry run (no Vivado needed)

```powershell
cd "Integrated_Simulator (Copy)\vivado_flow"

python scripts\iterate_until_pass.py `
    --K 16 --C 8 --KH 3 --KW 3 --H 30 --W 30 `
    --array-height 4 --array-width 4 `
    --dry-run
```

### From a TensorFlow model

```powershell
python scripts\iterate_until_pass.py `
    --tf-model tiny_cnn `
    --array-height 4 --array-width 4 `
    --dry-run
```

### Full Vivado flow (Vivado on PATH)

```powershell
python scripts\iterate_until_pass.py `
    --K 16 --C 8 --KH 3 --KW 3 --H 30 --W 30 `
    --array-height 4 --array-width 4 `
    --vivado vivado `
    --max-iter 5
```

---

## Running each stage independently

### Stage 1 — Optimiser → Verilog header

```powershell
python optimizer\generate_config.py --K 16 --C 8 --KH 3 --KW 3 --H 30 --W 30
# → configs/generated/optimal_config.vh
# → configs/generated/optimal_config.json
```

### Stage 2 — User config knobs

```powershell
# Print current knobs
python optimizer\config_knobs.py

# Export Verilog header (seeds from optimizer JSON automatically)
python optimizer\config_knobs.py --export-vh

# Override from a specific JSON
python optimizer\config_knobs.py --export-vh --from-json configs\generated\optimal_config.json
```

Edit `optimizer/config_knobs.py` to change any knob:

```python
CFG.hardware.ARRAY_HEIGHT = 8
CFG.hardware.ARRAY_WIDTH  = 8
CFG.tiling.TILE_K         = 8    # 0 = let the optimiser decide
CFG.sim.DUMP_WAVEFORMS    = False
```

### Stage 3 — Golden reference data

```powershell
python scripts\generate_golden.py
# reads configs/generated/optimal_config.json for dims
# → tb_data/input_hex.mem
# → tb_data/weight_hex.mem
# → tb_data/golden_hex.mem
```

### Stage 4 — Vivado simulation (requires Vivado)

```powershell
vivado -mode batch `
    -source scripts\run_vivado_sim.tcl `
    -tclargs "D:\path\to\repo" `
    -nolog -nojournal
# → vivado_project/sim_result.txt
```

---

## Configuration knob reference

| Knob | Default | Description |
|------|---------|-------------|
| `ARRAY_HEIGHT` | 4 | PE array rows (output-channel dimension) |
| `ARRAY_WIDTH` | 4 | PE array columns (input-channel dimension) |
| `DATA_WIDTH` | 32 | Bits per activation / weight |
| `ACCUM_WIDTH` | 48 | Partial-sum accumulator width |
| `LAYER_K` | 16 | Output channels |
| `LAYER_C` | 8 | Input channels |
| `LAYER_KH` / `KW` | 3/3 | Kernel spatial dimensions |
| `LAYER_H_IN` / `W_IN` | 30/30 | Input feature-map spatial dimensions |
| `TILE_K/C/H/W` | 0 | Override tiling (0 = optimizer decides) |
| `SIM_TIMEOUT` | auto | Watchdog cycle limit |
| `GOLDEN_CHECK` | 1 | Enable output vs golden comparison |
| `ERROR_TOLERANCE` | 0 | Allowed |diff| per element |
| `DUMP_WAVEFORMS` | 1 | Write VCD waveform file |

---

## Automatic fixup strategies

If a Vivado simulation fails, `iterate_until_pass.py` applies one of:

| Failure | Detected by | Fixup |
|---------|-------------|-------|
| **TIMEOUT** | `WATCHDOG` in result | Double `TILE_K/C/H/W` → fewer tiles → fewer cycles |
| **MISMATCH** | `errors=N` where N>0 | Halve `val_range` → smaller products → no overflow |
| **RTL_ERROR** | elaboration failure | Abort — requires manual RTL fix |

After fixup the optimizer re-runs with the new settings and a fresh
Vivado simulation is launched. The process repeats up to `--max-iter` times.

---

## Outputs

| File | Description |
|------|-------------|
| `configs/generated/optimal_config.vh` | Optimal `\`define` macros |
| `configs/generated/user_config.vh` | User-override `\`define` macros |
| `configs/generated/optimal_config.json` | Optimiser metrics |
| `vivado_project/sim_result.txt` | `TEST_PASS` or `TEST_FAIL ...` |
| `vivado_project/sim_waveforms.vcd` | VCD waveform (if enabled) |
| `logs/iter_NN/` | Per-stage log files |
| `iteration_report.json` | Full iteration history |
