# Failure triage

When `scripts/iterate_until_pass.py` reports a failure, the failing run's
artifacts land under `build/<sha>/`:

* `sim_metrics.json` -- final pass flag + cycles + stats
* `stdout.log`       -- the cocotb/Verilator log
* `dump.vcd`         -- waveform (if `--trace` was on)

## Common failure patterns

### "verilator-missing" skip reason
* **Cause**: Verilator is not installed; the runner used the software-ref fallback.
* **Fix**: `./scripts/install_deps.sh` (or `apt install verilator`).
* **Note**: the software-ref result is still a valid functional check;
  cycle counts are analytical, not RTL-measured.

### Timeout (cycle count == 50_000)
* **Cause**: the DUT never asserted `done`. Usually an FSM that's stuck in a
  WAIT state because a fetch_done pulse was missed.
* **Fix**: re-check the fetcher's done-pulse latch (sticky latch in
  `systolic_array_os.sv` is the reference pattern).
* **Debug**: dump VCD, watch the FSM `state` signal + `*_fetch_done` pulses.

### Output mismatch (golden_compare.fraction_within < 1.0)
* **Cause**: usually a sign-extension or fixed-point overflow in the MAC.
* **Fix**: ensure `accum_width >= 2 * data_width + log2(reduction_size)`.
  Default ACCUM_WIDTH=48 covers DATA_WIDTH=32 with up to 32 accumulations.
* **Debug**: dump the simulator's fixed-point output (`sim_output.hex`),
  re-run `golden_compare.compare_from_hex` with the per-layer quant.json.

### X-propagation in early cycles
* **Cause**: reset deassert order; some registers initialise to X.
* **Fix**: all flip-flops in the unified RTL use the `if (!rst_n) ... else`
  pattern; verify no module forgets the rst_n branch.

### Package-collision compile errors
* **Cause**: leftover `*_pkg.sv` from the legacy trees got pulled into the
  source list, conflicting with `sim_framework_pkg.sv`'s re-export aliases.
* **Fix**: ensure `rtl/all_sources.f` is the only source list and contains
  only the unified packages.

### Vivado WNS < 0
* **Cause**: combinatorial path through the systolic array exceeds the
  requested 5 ns clock period.
* **Fix**: relax the clock in `timing/sdc/single_dnn.xdc` (e.g. 7 ns =
  ~140 MHz); or add a pipeline register at the worst path.
