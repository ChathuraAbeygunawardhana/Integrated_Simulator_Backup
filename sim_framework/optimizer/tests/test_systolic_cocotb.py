"""
CocoTB Testbench for Systolic Array with TensorFlow Integration
================================================================
This testbench interfaces the systolic array hardware with TensorFlow
test vectors and validates correctness.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, FallingEdge
from cocotb.binary import BinaryValue
import json
import os
import numpy as np


class AXIMemoryModel:
    """Simple AXI4 memory model for simulation."""
    
    def __init__(self, dut, data_width=64):
        self.dut = dut
        self.data_width = data_width
        self.memory = {}  # addr -> data
        self.read_latency = 5  # cycles
        
    def load_from_hex(self, filename):
        """Load memory image from hex file."""
        with open(filename, 'r') as f:
            current_addr = 0
            for line in f:
                line = line.strip()
                if line.startswith('@'):
                    current_addr = int(line[1:], 16)
                elif line:
                    data = int(line, 16)
                    self.memory[current_addr] = data
                    current_addr += 4
        
        print(f"Loaded {len(self.memory)} words from {filename}")
    
    async def handle_read(self):
        """Handle AXI read transactions."""
        while True:
            await RisingEdge(self.dut.clk)
            
            # Check for read request
            if self.dut.m_axi_arvalid.value == 1:
                self.dut.m_axi_arready.value = 1
                await RisingEdge(self.dut.clk)
                self.dut.m_axi_arready.value = 0
                
                # Get address
                addr = int(self.dut.m_axi_araddr.value)
                burst_len = int(self.dut.m_axi_arlen.value) + 1
                
                # Simulate read latency
                for _ in range(self.read_latency):
                    await RisingEdge(self.dut.clk)
                
                # Send data
                for i in range(burst_len):
                    read_addr = addr + (i * (self.data_width // 8))
                    
                    # Get data from memory (little-endian)
                    if self.data_width == 64:
                        data_low = self.memory.get(read_addr, 0)
                        data_high = self.memory.get(read_addr + 4, 0)
                        data = (data_high << 32) | data_low
                    else:
                        data = self.memory.get(read_addr, 0)
                    
                    self.dut.m_axi_rdata.value = data
                    self.dut.m_axi_rvalid.value = 1
                    self.dut.m_axi_rlast.value = 1 if i == burst_len - 1 else 0
                    self.dut.m_axi_rresp.value = 0  # OKAY
                    
                    # Wait for ready
                    while self.dut.m_axi_rready.value == 0:
                        await RisingEdge(self.dut.clk)
                    
                    await RisingEdge(self.dut.clk)
                    self.dut.m_axi_rvalid.value = 0
    
    async def handle_write(self):
        """Handle AXI write transactions."""
        while True:
            await RisingEdge(self.dut.clk)
            
            # Check for write address
            if self.dut.m_axi_awvalid.value == 1:
                self.dut.m_axi_awready.value = 1
                addr = int(self.dut.m_axi_awaddr.value)
                burst_len = int(self.dut.m_axi_awlen.value) + 1
                
                await RisingEdge(self.dut.clk)
                self.dut.m_axi_awready.value = 0
                
                # Receive data
                for i in range(burst_len):
                    self.dut.m_axi_wready.value = 1
                    
                    while self.dut.m_axi_wvalid.value == 0:
                        await RisingEdge(self.dut.clk)
                    
                    data = int(self.dut.m_axi_wdata.value)
                    write_addr = addr + (i * (self.data_width // 8))
                    
                    # Store in memory (little-endian)
                    if self.data_width == 64:
                        self.memory[write_addr] = data & 0xFFFFFFFF
                        self.memory[write_addr + 4] = (data >> 32) & 0xFFFFFFFF
                    else:
                        self.memory[write_addr] = data
                    
                    await RisingEdge(self.dut.clk)
                    self.dut.m_axi_wready.value = 0
                
                # Send write response
                self.dut.m_axi_bvalid.value = 1
                self.dut.m_axi_bresp.value = 0  # OKAY
                
                while self.dut.m_axi_bready.value == 0:
                    await RisingEdge(self.dut.clk)
                
                await RisingEdge(self.dut.clk)
                self.dut.m_axi_bvalid.value = 0


@cocotb.test()
async def test_conv2d_layer(dut):
    """Test a single Conv2D layer from TensorFlow."""
    
    # Start clock (100 MHz)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Initialize signals
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.tile_row.value = 0
    dut.tile_col_start.value = 0
    dut.tile_ch_start.value = 0
    
    # Page table and scratchpad control
    dut.pt_write_en.value = 0
    dut.pt_write_vpn.value = 0
    dut.pt_write_ppn.value = 0
    dut.pt_write_valid.value = 0
    dut.spad_load_en.value = 0
    dut.spad_load_addr.value = 0
    dut.spad_load_data.value = 0
    
    # AXI initialization
    dut.m_axi_arready.value = 0
    dut.m_axi_rvalid.value = 0
    dut.m_axi_rdata.value = 0
    dut.m_axi_rlast.value = 0
    dut.m_axi_rresp.value = 0
    dut.m_axi_awready.value = 0
    dut.m_axi_wready.value = 0
    dut.m_axi_bvalid.value = 0
    dut.m_axi_bresp.value = 0
    
    # Reset
    await Timer(100, units="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    # Load configuration
    config_file = os.environ.get('CONFIG_FILE', 'config.json')
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    dut._log.info(f"Configuration loaded: {config}")
    
    # Create AXI memory model
    mem_model = AXIMemoryModel(dut, data_width=64)
    
    # Load memory images
    input_file = os.environ.get('INPUT_FILE', 'input.hex')
    weight_file = os.environ.get('WEIGHT_FILE', 'weights.hex')
    
    mem_model.load_from_hex(input_file)
    mem_model.load_from_hex(weight_file)
    
    # Start AXI handlers
    cocotb.start_soon(mem_model.handle_read())
    cocotb.start_soon(mem_model.handle_write())
    
    # Configure DUT
    dut.input_base_addr.value = config['input_base_addr']
    dut.weight_base_addr.value = config['weight_base_addr']
    dut.output_base_addr.value = config['output_base_addr']
    dut.input_channels.value = config['input_channels']
    dut.input_height.value = config['input_height']
    dut.input_width.value = config['input_width']
    dut.output_channels.value = config['output_channels']
    dut.output_height.value = config['output_height']
    dut.output_width.value = config['output_width']
    dut.weight_k.value = config['weight_k']
    dut.weight_c.value = config['weight_c']
    dut.weight_kh.value = config['weight_kh']
    dut.weight_kw.value = config['weight_kw']
    
    # Set memory layout (assuming NHWC)
    dut.mem_layout.value = 0  # NHWC
    
    await Timer(100, units="ns")
    
    # Start computation
    dut._log.info("Starting computation...")
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for completion (with timeout)
    timeout_cycles = 100000
    cycle_count = 0
    
    while dut.done.value == 0 and cycle_count < timeout_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        if cycle_count % 1000 == 0:
            dut._log.info(f"Cycle {cycle_count}, busy={dut.busy.value}, stalled={dut.compute_stalled.value}")
    
    if cycle_count >= timeout_cycles:
        dut._log.error("Timeout waiting for computation!")
        assert False, "Timeout"
    
    dut._log.info(f"Computation completed in {cycle_count} cycles")
    
    # Extract output from memory
    output_addr = config['output_base_addr']
    output_size = config['output_height'] * config['output_width'] * config['output_channels']
    
    hw_output = []
    for i in range(output_size):
        addr = output_addr + (i * 4)
        data = mem_model.memory.get(addr, 0)
        # Convert from fixed-point (assuming 16 fractional bits)
        fp_value = np.int32(data).astype(np.float32) / (2**16)
        hw_output.append(fp_value)
    
    hw_output = np.array(hw_output).reshape(
        (config['output_height'], config['output_width'], config['output_channels'])
    )
    
    # Load expected output
    expected_file = os.environ.get('EXPECTED_FILE', 'expected_output.hex')
    if os.path.exists(expected_file):
        expected_output = []
        with open(expected_file, 'r') as f:
            current_addr = 0
            for line in f:
                line = line.strip()
                if line.startswith('@'):
                    current_addr = int(line[1:], 16)
                elif line:
                    data = int(line, 16)
                    # Convert from fixed-point
                    fp_value = np.int32(data).astype(np.float32) / (2**16)
                    expected_output.append(fp_value)
        
        expected_output = np.array(expected_output).reshape(hw_output.shape)
        
        # Compare
        error = np.abs(hw_output - expected_output)
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        dut._log.info(f"Verification results:")
        dut._log.info(f"  Max error:  {max_error:.6f}")
        dut._log.info(f"  Mean error: {mean_error:.6f}")
        dut._log.info(f"  HW output range:  [{np.min(hw_output):.4f}, {np.max(hw_output):.4f}]")
        dut._log.info(f"  Expected range:   [{np.min(expected_output):.4f}, {np.max(expected_output):.4f}]")
        
        tolerance = 0.01
        if max_error > tolerance:
            dut._log.error(f"Verification FAILED! Max error {max_error} > {tolerance}")
            # Print first few mismatches
            diff_indices = np.where(error > tolerance)
            for i in range(min(10, len(diff_indices[0]))):
                h, w, c = diff_indices[0][i], diff_indices[1][i], diff_indices[2][i]
                dut._log.error(
                    f"  Mismatch at [{h},{w},{c}]: HW={hw_output[h,w,c]:.6f}, "
                    f"Expected={expected_output[h,w,c]:.6f}, Error={error[h,w,c]:.6f}"
                )
            assert False, "Output verification failed"
        else:
            dut._log.info("Verification PASSED!")
    else:
        dut._log.warning(f"No expected output file found at {expected_file}")
        dut._log.info(f"Hardware output shape: {hw_output.shape}")
        dut._log.info(f"Hardware output range: [{np.min(hw_output):.4f}, {np.max(hw_output):.4f}]")
    
    # Additional checks
    dut._log.info(f"Hit/Miss statistics:")
    dut._log.info(f"  Input hits:  {dut.spad_hit_input.value}")
    dut._log.info(f"  Weight hits: {dut.spad_hit_weight.value}")
    dut._log.info(f"  Input misses:  {dut.spad_miss_input.value}")
    dut._log.info(f"  Weight misses: {dut.spad_miss_weight.value}")
