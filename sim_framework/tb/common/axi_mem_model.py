"""Cocotb-friendly behavioural AXI memory model.

Ported from systolic_optimizer/tests/test_systolic_cocotb.py with the stale
signal references (pt_write_en, spad_hit_*) stripped out -- those reference
an older module name (systolic_spad_top) that is not in the unified tree.
"""
from __future__ import annotations

from typing import Dict, Optional

try:
    import cocotb
    from cocotb.triggers import RisingEdge
except Exception:                       # pragma: no cover -- importable without cocotb
    cocotb = None


class AXIMemoryModel:
    """Word-addressable behavioural memory used for AXI read tests.

    Constructor:
        AXIMemoryModel(dut, dw=32)

    Pre-load data with .write(addr, value). The model snoops the DUT's
    AXI read channel and returns data one beat at a time.
    """

    def __init__(self, dut, dw: int = 32):
        self.dut  = dut
        self.dw   = dw
        self.mem: Dict[int, int] = {}

    def write(self, addr: int, value: int) -> None:
        self.mem[addr] = value & ((1 << self.dw) - 1)

    def read(self, addr: int) -> int:
        return self.mem.get(addr, 0)

    def writes_from_array(self, base_addr: int, words: list[int]) -> None:
        bpw = self.dw // 8
        for i, w in enumerate(words):
            self.write(base_addr + i * bpw, w)

    async def run(self):
        """Coroutine that handles AXI handshakes. Yields one beat per cycle."""
        if cocotb is None:
            raise RuntimeError("cocotb is not importable")
        while True:
            await RisingEdge(self.dut.clk)
            # Always-ready, single-outstanding model.
            try:
                self.dut.axi_arready.value = 1
                if int(self.dut.axi_arvalid.value):
                    addr = int(self.dut.axi_arvalid_addr.value)
                    nbeats = int(self.dut.axi_arlen.value) + 1
                    for b in range(nbeats):
                        await RisingEdge(self.dut.clk)
                        self.dut.axi_rdata.value  = self.read(addr + b * (self.dw // 8))
                        self.dut.axi_rvalid.value = 1
                        self.dut.axi_rlast.value  = 1 if (b == nbeats - 1) else 0
                    await RisingEdge(self.dut.clk)
                    self.dut.axi_rvalid.value = 0
                    self.dut.axi_rlast.value  = 0
            except AttributeError:
                # DUT lacks an AXI port (some unit tests). Just idle.
                pass
