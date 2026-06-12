"""SimConfig dataclass exposing every control knob in the framework.

A SimConfig instance fully describes a single cocotb run: which top module
(single_dnn_top or multi_dnn_top), which dataflow flavor, which memory
backend, array geometry, data widths, layer workload, scheduler choice,
fixed-point format, and tolerance.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional

import yaml

# ----------------------------------------------------------------------------
# Per-layer descriptor (single workload entry)
# ----------------------------------------------------------------------------
@dataclass
class LayerConfig:
    name: str = "layer_0"
    # tensor shapes (NHWC)
    input_channels: int = 3
    input_height: int = 8
    input_width: int = 8
    weight_k: int = 8           # output channels
    weight_c: int = 3           # input channels (must match input_channels)
    weight_kh: int = 3
    weight_kw: int = 3
    output_channels: int = 8
    output_height: int = 6
    output_width: int = 6
    # memory base addresses in the host AXI memory model
    input_base_addr: int = 0x00000000
    weight_base_addr: int = 0x00010000
    output_base_addr: int = 0x00020000
    # tile coordinates (drive the per-tile execution)
    tile_row: int = 0
    tile_col_start: int = 0
    tile_ch_start: int = 0
    # WS-specific extras (ignored for OS/IS)
    ws_input_ch: int = 0
    ws_kernel_row: int = 0
    ws_kernel_col_start: int = 0


# ----------------------------------------------------------------------------
# Full simulator configuration
# ----------------------------------------------------------------------------
@dataclass
class SimConfig:
    # -------- top selection --------
    top: Literal["single_dnn_top", "multi_dnn_top"] = "single_dnn_top"

    # -------- dataflow knob --------
    dataflow: Literal["OS", "IS", "WS"] = "OS"

    # -------- memory backend knob --------
    memory: Literal["STAMP", "PAGED"] = "STAMP"

    # -------- array geometry --------
    array_height: int = 4
    array_width: int = 4
    data_width: int = 32
    accum_width: int = 48
    addr_width: int = 32

    # -------- on-chip memory sizing --------
    spad_depth: int = 4096
    axi_data_width: int = 64
    metadata_depth: int = 256     # stamp only
    num_pages: int = 256          # paged only
    page_size_bits: int = 12      # paged only
    n_mem_ports: int = 4

    # -------- layout (data tensor packing in DRAM) --------
    mem_layout: Literal["CHANNEL_MAJOR", "ROW_MAJOR", "COLUMN_MAJOR"] = "CHANNEL_MAJOR"

    # -------- workload --------
    layers: List[LayerConfig] = field(default_factory=lambda: [LayerConfig()])

    # -------- multi-DNN extras --------
    scheduler: Optional[str] = None   # one of SCHEDULER_NAMES; None for single_dnn
    max_tasks: int = 16
    max_layers: int = 32
    max_dnns: int = 4

    # -------- quantisation / tolerance --------
    fp_total_bits: int = 32
    fp_frac_bits: int = 16
    tolerance: float = 0.02

    # -------- cycle budget for timing tests --------
    cycle_budget: Optional[int] = None

    # -------- iteration / sweep --------
    seed: int = 0

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    def hash(self) -> str:
        """Stable SHA-256 over the full config (12 hex chars)."""
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def to_yaml(self, path: str | None = None) -> str:
        text = yaml.safe_dump(asdict(self), sort_keys=False)
        if path:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_yaml(cls, path: str) -> "SimConfig":
        with open(path) as fh:
            data = yaml.safe_load(fh)
        layers = [LayerConfig(**l) for l in data.pop("layers", [])]
        return cls(layers=layers, **data)

    # ------------------------------------------------------------------
    # Mappings to Verilator -G parameters and cocotb env vars
    # ------------------------------------------------------------------
    DATAFLOW_TO_RTL = {"OS": 0, "IS": 1, "WS": 2}
    MEMORY_TO_RTL   = {"STAMP": 0, "PAGED": 1}
    SCHEDULER_NAMES = [
        "FIFO", "LIFO", "SJF", "RR", "PRI", "EDF", "LRU",                 # 0..6
        "SRTF", "HRRN", "MLQ", "MLFQ",                                    # 7..10
        "AIMT", "BATCHDNN", "BATCHDNN_PP",                                # 11..13
    ]

    def scheduler_select(self) -> int:
        if self.scheduler is None:
            return 0
        if self.scheduler not in self.SCHEDULER_NAMES:
            raise ValueError(f"unknown scheduler {self.scheduler}")
        return self.SCHEDULER_NAMES.index(self.scheduler)

    def verilator_params(self) -> List[str]:
        """List of `-GNAME=value` strings for Verilator parameter override."""
        return [
            f"-GDATAFLOW={self.DATAFLOW_TO_RTL[self.dataflow]}",
            f"-GMEMORY={self.MEMORY_TO_RTL[self.memory]}",
            f"-GARRAY_HEIGHT={self.array_height}",
            f"-GARRAY_WIDTH={self.array_width}",
            f"-GDATA_WIDTH={self.data_width}",
            f"-GACCUM_WIDTH={self.accum_width}",
            f"-GADDR_WIDTH={self.addr_width}",
            f"-GSPAD_DEPTH={self.spad_depth}",
            f"-GAXI_DATA_WIDTH={self.axi_data_width}",
            f"-GMETADATA_DEPTH={self.metadata_depth}",
            f"-GNUM_PAGES={self.num_pages}",
            f"-GPAGE_SIZE_BITS={self.page_size_bits}",
            f"-GN_MEM_PORTS={self.n_mem_ports}",
        ]

    def cocotb_env(self) -> dict:
        """Environment variables consumed by the cocotb test runner."""
        return {
            "SIM_DATAFLOW":  self.dataflow,
            "SIM_MEMORY":    self.memory,
            "SIM_LAYOUT":    self.mem_layout,
            "SIM_ARRAY_H":   str(self.array_height),
            "SIM_ARRAY_W":   str(self.array_width),
            "SIM_DATA_W":    str(self.data_width),
            "SIM_FP_TOTAL":  str(self.fp_total_bits),
            "SIM_FP_FRAC":   str(self.fp_frac_bits),
            "SIM_TOL":       str(self.tolerance),
            "SIM_SCHEDULER": str(self.scheduler_select()),
            "SIM_LAYERS_JSON": json.dumps([asdict(l) for l in self.layers]),
        }


# ----------------------------------------------------------------------------
# Convenience presets
# ----------------------------------------------------------------------------
def preset_small_os_stamp() -> SimConfig:
    return SimConfig(
        dataflow="OS", memory="STAMP", array_height=4, array_width=4,
        layers=[LayerConfig(
            input_channels=3, input_height=8, input_width=8,
            weight_k=8, weight_c=3, weight_kh=3, weight_kw=3,
            output_channels=8, output_height=6, output_width=6,
        )],
    )


def preset_medium_ws_paged() -> SimConfig:
    return SimConfig(
        dataflow="WS", memory="PAGED", array_height=8, array_width=8,
        layers=[LayerConfig(
            input_channels=16, input_height=14, input_width=14,
            weight_k=32, weight_c=16, weight_kh=3, weight_kw=3,
            output_channels=32, output_height=12, output_width=12,
        )],
    )
