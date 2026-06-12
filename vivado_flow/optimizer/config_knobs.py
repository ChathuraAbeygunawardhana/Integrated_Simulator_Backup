#!/usr/bin/env python3
"""
config_knobs.py
===============
User-facing configuration knob file for the Vivado systolic-array flow.

Edit the KNOBS section below, then run:

    python config_knobs.py                  # print current config
    python config_knobs.py --export-vh      # write Verilog header
    python config_knobs.py --export-vh --from-json configs/generated/optimal_config.json
                                            # seed knobs from optimizer output

The exported Verilog header is written to:
    configs/generated/user_config.vh

This file is backtick-included by the testbench AFTER optimal_config.vh so
any knob you override here takes precedence (later define wins in Verilog).

Workflow
--------
1. Run generate_config.py  → configs/generated/optimal_config.vh   (optimizer)
2. Edit KNOBS below as needed
3. Run config_knobs.py --export-vh → configs/generated/user_config.vh
4. The testbench includes BOTH headers; user_config.vh overrides win.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional

# ── Cross-platform UTF-8 stdout/stderr ──────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
# ╔══════════════════════════════════════════════════════════╗
# ║              USER CONFIGURATION KNOBS                   ║
# ║  Modify any value; everything else auto-derives         ║
# ╚══════════════════════════════════════════════════════════╝
# ============================================================

@dataclass
class HardwareKnobs:
    """Physical array properties."""

    # ── Array geometry ───────────────────────────────────────
    ARRAY_HEIGHT: int  = 4      # PE rows   (maps to output-channel dim)
    ARRAY_WIDTH:  int  = 4      # PE cols   (maps to input-channel dim)

    # ── Data widths ──────────────────────────────────────────
    DATA_WIDTH:   int  = 32     # bits per activation / weight element
    ACCUM_WIDTH:  int  = 48     # bits for partial-sum accumulator
    ADDR_WIDTH:   int  = 32     # bits for memory addresses

    # ── Memory layout ────────────────────────────────────────
    # 0 = NHWC / CHANNEL_MAJOR   1 = NCHW / ROW_MAJOR   2 = COLUMN_MAJOR
    MEM_LAYOUT:   int  = 0

    # ── Dataflow flavor ──────────────────────────────────────
    # 0 = Output-Stationary (OS)   1 = Input-Stationary (IS)   2 = Weight-Stationary (WS)
    DATAFLOW:     int  = 0

    # ── Memory backend ───────────────────────────────────────
    # 0 = STAMP (stamp-controller + hash table)   1 = PAGED (page-table)
    MEMORY:       int  = 0


@dataclass
class MemoryKnobs:
    """On-chip scratchpad / AXI memory sizing."""

    # ── Scratchpad ───────────────────────────────────────────
    SPAD_DEPTH:      int = 4096   # depth of on-chip SRAM (words, power-of-2)
    AXI_DATA_WIDTH:  int = 64     # AXI bus width in bits (32 / 64 / 128 / 256)
    N_MEM_PORTS:     int = 4      # number of independent SRAM read/write ports

    # ── STAMP backend (ignored when MEMORY=1) ────────────────
    METADATA_DEPTH:  int = 256    # depth of stamp-controller metadata SRAM

    # ── Paged backend (ignored when MEMORY=0) ────────────────
    NUM_PAGES:       int = 256    # total number of virtual pages
    PAGE_SIZE_BITS:  int = 12     # log2(page size in bytes), e.g. 12 → 4 KiB pages


@dataclass
class WorkloadKnobs:
    """Layer dimensions for the convolution workload."""

    K:  int = 16    # output channels  (number of filters)
    C:  int = 8     # input  channels
    KH: int = 3     # kernel height
    KW: int = 3     # kernel width
    H:  int = 30    # input feature-map height
    W:  int = 30    # input feature-map width

    @property
    def H_OUT(self) -> int: return self.H - self.KH + 1
    @property
    def W_OUT(self) -> int: return self.W - self.KW + 1


@dataclass
class TilingKnobs:
    """
    Tiling / blocking factors.
    Set any to 0 to inherit the value from optimal_config.vh (optimizer picks it).
    Set to a positive integer to override.
    """
    TILE_K: int = 0   # 0 = let optimizer decide
    TILE_C: int = 0
    TILE_H: int = 0
    TILE_W: int = 0

    # ── Loop order (0 = inherit from optimizer) ──────────────
    # Encoded as integer 0-5 mapping the 6-D loop permutation:
    #   0 = K→C→KH→KW→H→W   1 = K→C→H→W→KH→KW
    #   2 = C→K→KH→KW→H→W   3 = H→W→K→C→KH→KW
    #   4 = K→H→W→C→KH→KW   5 = C→H→W→K→KH→KW
    # Set to -1 to inherit from optimal_config.vh.
    LOOP_ORDER: int = -1   # -1 = inherit from optimizer


@dataclass
class AddressKnobs:
    """
    Base addresses (byte offsets into the flat SRAM model).
    Set any to -1 to inherit the value computed by generate_config.py.
    """
    INPUT_BASE_ADDR:  int = -1    # -1 = auto-computed from workload
    WEIGHT_BASE_ADDR: int = -1    # -1 = auto-computed from workload
    OUTPUT_BASE_ADDR: int = -1    # -1 = auto-computed from workload


@dataclass
class MultiDNNKnobs:
    """
    Multi-DNN scheduler and dispatch knobs.
    Only active when the top-level module is multi_dnn_top.
    """
    # ── Scheduler selection ──────────────────────────────────
    # 0=FIFO  1=LIFO  2=SJF  3=RR  4=PRI  5=EDF  6=LRU
    # 7=SRTF  8=HRRN  9=MLQ  10=MLFQ  11=AIMT  12=BATCHDNN  13=BATCHDNN_PP
    SCHEDULER_SELECT: int = 0     # 0 = FIFO (default)

    # ── Dispatch table sizing ────────────────────────────────
    MAX_TASKS:  int = 16   # maximum outstanding tasks in the scheduler queue
    MAX_LAYERS: int = 32   # maximum layers per DNN workload entry
    MAX_DNNS:   int = 4    # maximum concurrent DNN workloads


@dataclass
class QuantKnobs:
    """Fixed-point quantisation parameters."""

    FP_TOTAL_BITS: int   = 32    # total fixed-point word width
    FP_FRAC_BITS:  int   = 16    # fractional bits (Q(TOTAL-FRAC).FRAC format)
    SIM_TOL:       float = 0.02  # max allowed relative error vs golden reference


@dataclass
class SimKnobs:
    """Simulation control."""

    SIM_TIMEOUT:      int  = 0       # 0 = let generate_config.py decide
    DUMP_WAVEFORMS:   bool = True    # write VCD/WDB waveform dump
    VERBOSE:          bool = False   # extra $display logging in testbench
    GOLDEN_CHECK:     bool = True    # compare output vs software reference
    ERROR_TOLERANCE:  int  = 0       # allowed |diff| per element (ULP)


# ============================================================
# Master config (edit here or load from JSON)
# ============================================================

@dataclass
class SimConfig:
    """Aggregates all knob groups into one object."""
    hardware:  HardwareKnobs  = field(default_factory=HardwareKnobs)
    memory:    MemoryKnobs    = field(default_factory=MemoryKnobs)
    workload:  WorkloadKnobs  = field(default_factory=WorkloadKnobs)
    tiling:    TilingKnobs   = field(default_factory=TilingKnobs)
    addresses: AddressKnobs  = field(default_factory=AddressKnobs)
    multi_dnn: MultiDNNKnobs = field(default_factory=MultiDNNKnobs)
    quant:     QuantKnobs    = field(default_factory=QuantKnobs)
    sim:       SimKnobs      = field(default_factory=SimKnobs)

    # ── Convenience passthrough ─────────────────────────────
    @property
    def array_height(self): return self.hardware.ARRAY_HEIGHT
    @property
    def array_width(self):  return self.hardware.ARRAY_WIDTH

    # ── Serialisation ───────────────────────────────────────
    def to_dict(self):
        return {
            "hardware":  asdict(self.hardware),
            "memory":    asdict(self.memory),
            "workload":  asdict(self.workload),
            "tiling":    asdict(self.tiling),
            "addresses": asdict(self.addresses),
            "multi_dnn": asdict(self.multi_dnn),
            "quant":     asdict(self.quant),
            "sim":       asdict(self.sim),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        return cls(
            hardware  = HardwareKnobs(**d.get("hardware",  {})),
            memory    = MemoryKnobs(**d.get("memory",    {})),
            workload  = WorkloadKnobs(**d.get("workload",  {})),
            tiling    = TilingKnobs(**d.get("tiling",    {})),
            addresses = AddressKnobs(**d.get("addresses", {})),
            multi_dnn = MultiDNNKnobs(**d.get("multi_dnn", {})),
            quant     = QuantKnobs(**d.get("quant",     {})),
            sim       = SimKnobs(**d.get("sim",       {})),
        )

    @classmethod
    def from_json(cls, path: Path) -> "SimConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Accept the output of generate_config.py (different schema) or our own
        if "hardware" not in raw and "layer_dims" in raw:
            return cls._from_optimizer_json(raw)
        return cls.from_dict(raw)

    @classmethod
    def _from_optimizer_json(cls, raw: dict) -> "SimConfig":
        """Seed from generate_config.py's output JSON."""
        dims = raw.get("layer_dims", {})
        arr  = raw.get("array",      {})
        blk  = raw.get("blocking",   {})
        mem  = raw.get("memory",     {})
        addr = raw.get("addresses",  {})
        return cls(
            hardware=HardwareKnobs(
                ARRAY_HEIGHT=arr.get("height",   4),
                ARRAY_WIDTH =arr.get("width",    4),
                DATAFLOW    =arr.get("dataflow", 0),
                MEMORY      =arr.get("memory",   0),
            ),
            memory=MemoryKnobs(
                SPAD_DEPTH     =mem.get("spad_depth",      4096),
                AXI_DATA_WIDTH =mem.get("axi_data_width",    64),
                N_MEM_PORTS    =mem.get("n_mem_ports",        4),
                METADATA_DEPTH =mem.get("metadata_depth",   256),
                NUM_PAGES      =mem.get("num_pages",        256),
                PAGE_SIZE_BITS =mem.get("page_size_bits",    12),
            ),
            workload=WorkloadKnobs(
                K=dims.get("K",  16), C=dims.get("C",   8),
                KH=dims.get("KH", 3), KW=dims.get("KW",  3),
                H=dims.get("H",  30), W=dims.get("W",   30),
            ),
            tiling=TilingKnobs(
                TILE_K    =blk.get("tile_K",     0),
                TILE_C    =blk.get("tile_C",     0),
                TILE_H    =blk.get("tile_H",     0),
                TILE_W    =blk.get("tile_W",     0),
                LOOP_ORDER=blk.get("loop_order", -1),
            ),
            addresses=AddressKnobs(
                INPUT_BASE_ADDR  =addr.get("input_base",  -1),
                WEIGHT_BASE_ADDR =addr.get("weight_base", -1),
                OUTPUT_BASE_ADDR =addr.get("output_base", -1),
            ),
        )

    def to_json(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    # ── Verilog export ──────────────────────────────────────
    def export_to_verilog(self, output_path: Path):
        """
        Emit a Verilog header that *overrides* whichever macros the user
        has set.  Values of 0 / -1 mean "inherit from optimal_config.vh".

        Sections emitted:
          1. Hardware (array geometry + data widths + dataflow + memory backend)
          2. On-chip memory sizing
          3. Workload (layer dims, always override)
          4. Tiling + loop order (0 / -1 → inherit from optimizer)
          5. Base addresses (-1 → inherit from optimizer)
          6. Multi-DNN / scheduler knobs
          7. Fixed-point / quantisation
          8. Simulation control
        """
        hw  = self.hardware
        mem = self.memory
        wl  = self.workload
        tl  = self.tiling
        ad  = self.addresses
        md  = self.multi_dnn
        qk  = self.quant
        sm  = self.sim

        def _timeout():
            if sm.SIM_TIMEOUT > 0:
                return sm.SIM_TIMEOUT
            import math
            n_tiles = (math.ceil(wl.K  / max(tl.TILE_K, hw.ARRAY_HEIGHT)) *
                       math.ceil(wl.C  / max(tl.TILE_C, hw.ARRAY_WIDTH))  *
                       math.ceil(wl.H  / max(tl.TILE_H, hw.ARRAY_HEIGHT)) *
                       math.ceil(wl.W  / max(tl.TILE_W, hw.ARRAY_WIDTH))  *
                       wl.KH * wl.KW)
            return max(50_000, n_tiles * (hw.ARRAY_HEIGHT + hw.ARRAY_WIDTH + 10) * 4)

        def _addr_line(macro: str, val: int) -> str:
            if val >= 0:
                return f"`define {macro:<20} 32'h{val:08X}  // user override"
            return f"// `define {macro}  // inherit from optimal_config.vh"

        # ── fixed-point tolerance as integer ULPs if ERROR_TOLERANCE > 0,
        #    otherwise encode float SIM_TOL as a scaled integer (×2^FP_FRAC)
        def _tol_ulp() -> int:
            if sm.ERROR_TOLERANCE > 0:
                return sm.ERROR_TOLERANCE
            # convert relative float tolerance to ULPs (1 ULP = 2^-FP_FRAC)
            import math
            max_val = 2 ** (qk.FP_TOTAL_BITS - qk.FP_FRAC_BITS - 1)
            return max(0, int(math.ceil(qk.SIM_TOL * max_val * (2 ** qk.FP_FRAC_BITS))))

        _DATAFLOW_LABEL = {0: "OS (Output-Stationary)",
                           1: "IS (Input-Stationary)",
                           2: "WS (Weight-Stationary)"}
        _MEMORY_LABEL   = {0: "STAMP", 1: "PAGED"}

        lines = [
            "//============================================================",
            "// USER_CONFIG.vh — generated by config_knobs.py",
            "// Edit config_knobs.py; do not edit this file directly.",
            "//============================================================",
            "",
            "// ---- 1. Hardware — array geometry & datapath widths --------",
            f"`define ARRAY_HEIGHT    {hw.ARRAY_HEIGHT}",
            f"`define ARRAY_WIDTH     {hw.ARRAY_WIDTH}",
            f"`define DATA_WIDTH      {hw.DATA_WIDTH}",
            f"`define ACCUM_WIDTH     {hw.ACCUM_WIDTH}",
            f"`define ADDR_WIDTH      {hw.ADDR_WIDTH}",
            f"`define MEM_LAYOUT      {hw.MEM_LAYOUT}",
            f"// ---- 1b. Dataflow: {_DATAFLOW_LABEL.get(hw.DATAFLOW, hw.DATAFLOW)} ---",
            f"`define DATAFLOW        {hw.DATAFLOW}",
            f"// ---- 1c. Memory backend: {_MEMORY_LABEL.get(hw.MEMORY, hw.MEMORY)} ---",
            f"`define MEMORY          {hw.MEMORY}",
            "",
            "// ---- 2. On-chip memory sizing (always override) -----------",
            f"`define SPAD_DEPTH      {mem.SPAD_DEPTH}",
            f"`define AXI_DATA_WIDTH  {mem.AXI_DATA_WIDTH}",
            f"`define N_MEM_PORTS     {mem.N_MEM_PORTS}",
            f"// STAMP backend only (ignored when MEMORY=1):",
            f"`define METADATA_DEPTH  {mem.METADATA_DEPTH}",
            f"// Paged backend only (ignored when MEMORY=0):",
            f"`define NUM_PAGES       {mem.NUM_PAGES}",
            f"`define PAGE_SIZE_BITS  {mem.PAGE_SIZE_BITS}",
            "",
            "// ---- 3. Workload (always override) ------------------------",
            f"`define LAYER_K         {wl.K}",
            f"`define LAYER_C         {wl.C}",
            f"`define LAYER_KH        {wl.KH}",
            f"`define LAYER_KW        {wl.KW}",
            f"`define LAYER_H_IN      {wl.H}",
            f"`define LAYER_W_IN      {wl.W}",
            f"`define LAYER_H_OUT     {wl.H_OUT}",
            f"`define LAYER_W_OUT     {wl.W_OUT}",
            "",
            "// ---- 4. Tiling (0 → keep optimizer value) -----------------",
        ]

        for macro, val in [("TILE_K", tl.TILE_K), ("TILE_C", tl.TILE_C),
                            ("TILE_H", tl.TILE_H), ("TILE_W", tl.TILE_W)]:
            if val > 0:
                lines.append(f"`define {macro:<14} {val}  // user override")
            else:
                lines.append(f"// `define {macro} 0  // inherit from optimal_config.vh")

        # Loop order: -1 means inherit
        if tl.LOOP_ORDER >= 0:
            lines.append(f"`define LOOP_ORDER      {tl.LOOP_ORDER}  // user override")
        else:
            lines.append("// `define LOOP_ORDER  // inherit from optimal_config.vh")

        lines += [
            "",
            "// ---- 5. Base addresses (-1 → auto from optimizer) ---------",
            _addr_line("INPUT_BASE_ADDR",  ad.INPUT_BASE_ADDR),
            _addr_line("WEIGHT_BASE_ADDR", ad.WEIGHT_BASE_ADDR),
            _addr_line("OUTPUT_BASE_ADDR", ad.OUTPUT_BASE_ADDR),
            "",
            "// ---- 6. Multi-DNN / scheduler (active with multi_dnn_top) -",
            f"// Scheduler IDs: 0=FIFO 1=LIFO 2=SJF 3=RR 4=PRI 5=EDF 6=LRU",
            f"//               7=SRTF 8=HRRN 9=MLQ 10=MLFQ 11=AIMT 12=BatchDNN 13=BatchDNN++",
            f"`define SCHEDULER_SELECT {md.SCHEDULER_SELECT}",
            f"`define MAX_TASKS        {md.MAX_TASKS}",
            f"`define MAX_LAYERS       {md.MAX_LAYERS}",
            f"`define MAX_DNNS         {md.MAX_DNNS}",
            "",
            "// ---- 7. Fixed-point / quantisation ------------------------",
            f"`define FP_TOTAL_BITS    {qk.FP_TOTAL_BITS}",
            f"`define FP_FRAC_BITS     {qk.FP_FRAC_BITS}",
            f"// SIM_TOL={qk.SIM_TOL} → {_tol_ulp()} ULPs at Q{qk.FP_TOTAL_BITS-qk.FP_FRAC_BITS}.{qk.FP_FRAC_BITS}",
            "",
            "// ---- 8. Simulation control --------------------------------",
            f"`define SIM_TIMEOUT     {_timeout()}",
            f"`define DUMP_WAVEFORMS  {'1' if sm.DUMP_WAVEFORMS else '0'}",
            f"`define VERBOSE         {'1' if sm.VERBOSE else '0'}",
            f"`define GOLDEN_CHECK    {'1' if sm.GOLDEN_CHECK else '0'}",
            f"`define ERROR_TOLERANCE {_tol_ulp()}",
            "",
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[config_knobs] Wrote user Verilog header -> {output_path}")

    # ── Pretty print ────────────────────────────────────────
    def print_summary(self):
        hw  = self.hardware
        mem = self.memory
        wl  = self.workload
        tl  = self.tiling
        ad  = self.addresses
        md  = self.multi_dnn
        qk  = self.quant
        sm  = self.sim

        _DF = {0: "OS", 1: "IS", 2: "WS"}
        _MB = {0: "STAMP", 1: "PAGED"}
        _LO = tl.LOOP_ORDER if tl.LOOP_ORDER >= 0 else "(opt)"

        def _addr(v): return f"0x{v:08X}" if v >= 0 else "(auto)"

        print("\n" + "=" * 62)
        print("  SIM CONFIG KNOBS")
        print("=" * 62)
        print(f"  Hardware  : {hw.ARRAY_HEIGHT}×{hw.ARRAY_WIDTH} PEs  "
              f"DATA={hw.DATA_WIDTH}b  ACCUM={hw.ACCUM_WIDTH}b  "
              f"ADDR={hw.ADDR_WIDTH}b")
        print(f"  Dataflow  : {_DF.get(hw.DATAFLOW, hw.DATAFLOW)}  |  "
              f"Memory backend : {_MB.get(hw.MEMORY, hw.MEMORY)}  |  "
              f"Layout : {hw.MEM_LAYOUT}")
        print(f"  On-chip   : SPAD={mem.SPAD_DEPTH}w  AXI={mem.AXI_DATA_WIDTH}b  "
              f"ports={mem.N_MEM_PORTS}  "
              f"meta={mem.METADATA_DEPTH}  pages={mem.NUM_PAGES}×2^{mem.PAGE_SIZE_BITS}B")
        print(f"  Workload  : K={wl.K}  C={wl.C}  KH={wl.KH}  KW={wl.KW}  "
              f"H={wl.H}  W={wl.W}  =>  H_out={wl.H_OUT}  W_out={wl.W_OUT}")
        print(f"  Tiling    : K={tl.TILE_K or '(opt)'}  C={tl.TILE_C or '(opt)'}  "
              f"H={tl.TILE_H or '(opt)'}  W={tl.TILE_W or '(opt)'}  "
              f"loop_order={_LO}")
        print(f"  Addresses : IN={_addr(ad.INPUT_BASE_ADDR)}  "
              f"WT={_addr(ad.WEIGHT_BASE_ADDR)}  "
              f"OUT={_addr(ad.OUTPUT_BASE_ADDR)}")
        print(f"  Multi-DNN : scheduler={md.SCHEDULER_SELECT}  "
              f"max_tasks={md.MAX_TASKS}  max_layers={md.MAX_LAYERS}  "
              f"max_dnns={md.MAX_DNNS}")
        print(f"  Quant     : Q{qk.FP_TOTAL_BITS-qk.FP_FRAC_BITS}.{qk.FP_FRAC_BITS}  "
              f"tol={qk.SIM_TOL}")
        print(f"  Sim       : timeout={sm.SIM_TIMEOUT or '(auto)'}  "
              f"golden={'yes' if sm.GOLDEN_CHECK else 'no'}  "
              f"waveforms={'yes' if sm.DUMP_WAVEFORMS else 'no'}  "
              f"verbose={'yes' if sm.VERBOSE else 'no'}")
        print("=" * 62)


# ============================================================
# Default config instance  ← modify this object in your script
# ============================================================

CFG = SimConfig()

# ─── Example user overrides (edit freely) ───────────────────────────────────
# Hardware / dataflow
# CFG.hardware.ARRAY_HEIGHT  = 8
# CFG.hardware.ARRAY_WIDTH   = 8
# CFG.hardware.DATA_WIDTH    = 16    # 8 / 16 / 32
# CFG.hardware.ACCUM_WIDTH   = 32
# CFG.hardware.DATAFLOW      = 1     # 0=OS  1=IS  2=WS
# CFG.hardware.MEMORY        = 1     # 0=STAMP  1=PAGED
# CFG.hardware.MEM_LAYOUT    = 1     # 0=NHWC  1=NCHW  2=COLUMN_MAJOR

# On-chip memory
# CFG.memory.SPAD_DEPTH      = 8192
# CFG.memory.AXI_DATA_WIDTH  = 128
# CFG.memory.N_MEM_PORTS     = 8
# CFG.memory.METADATA_DEPTH  = 512   # STAMP only
# CFG.memory.NUM_PAGES       = 512   # PAGED only
# CFG.memory.PAGE_SIZE_BITS  = 14    # PAGED only (2^14 = 16 KiB pages)

# Workload
# CFG.workload.K   = 32
# CFG.workload.C   = 16
# CFG.workload.KH  = 5
# CFG.workload.KW  = 5
# CFG.workload.H   = 56
# CFG.workload.W   = 56

# Tiling (0 = let optimizer decide)
# CFG.tiling.TILE_K      = 8
# CFG.tiling.TILE_C      = 4
# CFG.tiling.LOOP_ORDER  = 2  # 0-5; -1 = inherit

# Base addresses (-1 = auto)
# CFG.addresses.INPUT_BASE_ADDR   = 0x0000_0000
# CFG.addresses.WEIGHT_BASE_ADDR  = 0x0001_0000
# CFG.addresses.OUTPUT_BASE_ADDR  = 0x0002_0000

# Multi-DNN scheduler
# CFG.multi_dnn.SCHEDULER_SELECT  = 12  # 12=BATCHDNN
# CFG.multi_dnn.MAX_TASKS         = 32
# CFG.multi_dnn.MAX_DNNS          = 8

# Fixed-point / quantisation
# CFG.quant.FP_TOTAL_BITS = 16
# CFG.quant.FP_FRAC_BITS  = 8
# CFG.quant.SIM_TOL       = 0.01

# Simulation
# CFG.sim.DUMP_WAVEFORMS  = False
# CFG.sim.VERBOSE         = True
# CFG.sim.SIM_TIMEOUT     = 500_000
# ─────────────────────────────────────────────────────────────────────────────


# ============================================================
# CLI
# ============================================================

_HERE     = Path(__file__).resolve().parent
_OUT_DIR  = _HERE.parent / "configs" / "generated"
_VH_PATH  = _OUT_DIR / "user_config.vh"
_OPT_JSON = _OUT_DIR / "optimal_config.json"


def _build_parser():
    p = argparse.ArgumentParser(
        description="Inspect or export the user configuration knobs."
    )
    p.add_argument("--export-vh",    action="store_true",
                   help="Write configs/generated/user_config.vh")
    p.add_argument("--from-json",    type=Path, default=None,
                   metavar="PATH",
                   help="Seed knobs from a JSON file (optimizer output or previous export)")
    p.add_argument("--out-json",     type=Path, default=None,
                   metavar="PATH",
                   help="Also write the current knobs as JSON")
    p.add_argument("--vh-path",      type=Path, default=_VH_PATH,
                   help=f"Override VH output path (default: {_VH_PATH})")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    cfg = CFG  # starts from module-level defaults + any edits above

    if args.from_json:
        json_path = args.from_json
        if not json_path.exists() and not json_path.is_absolute():
            json_path = _OPT_JSON  # default
        print(f"[config_knobs] Seeding knobs from {json_path} …")
        cfg = SimConfig.from_json(json_path)

    cfg.print_summary()

    if args.export_vh:
        cfg.export_to_verilog(args.vh_path)

    if args.out_json:
        cfg.to_json(args.out_json)
        print(f"[config_knobs] Wrote JSON knobs → {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
