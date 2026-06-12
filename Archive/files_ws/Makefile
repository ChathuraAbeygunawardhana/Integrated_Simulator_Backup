#=============================================================================
# Makefile for Systolic Array RTL
# Supports simulation with various tools
#=============================================================================

# Design parameters
DESIGN = weight_stationary_top
TB = weight_stationary_tb

# Directories
RTL_DIR = .
SIM_DIR = sim
SYN_DIR = syn

# File lists
FILELIST = filelist.f
RTL_FILES = $(shell cat $(FILELIST) | grep -v "^\#" | grep "\.sv")

# Simulator selection (default: iverilog)
SIM ?= iverilog

#=============================================================================
# Simulation Targets
#=============================================================================

.PHONY: all clean sim wave help

all: sim

# Icarus Verilog simulation
ifeq ($(SIM),iverilog)
sim:
	@echo "=== Compiling with Icarus Verilog ==="
	mkdir -p $(SIM_DIR)
	iverilog -g2012 -o $(SIM_DIR)/$(TB).vvp -f $(FILELIST)
	@echo "=== Running simulation ==="
	cd $(SIM_DIR) && vvp $(TB).vvp
	@echo "=== Simulation complete ==="

wave:
	@echo "Opening waveform viewer..."
	gtkwave $(SIM_DIR)/weight_stationary.vcd &
endif

# Verilator simulation
ifeq ($(SIM),verilator)
sim:
	@echo "=== Compiling with Verilator ==="
	mkdir -p $(SIM_DIR)
	verilator --lint-only --timing -Wall -Wno-fatal \
		--top-module $(TB) -f $(FILELIST)
	@echo "=== Lint check complete ==="
endif

# Questa/ModelSim simulation
ifeq ($(SIM),questa)
sim:
	@echo "=== Compiling with Questa ==="
	mkdir -p $(SIM_DIR)
	cd $(SIM_DIR) && vlib work
	cd $(SIM_DIR) && vlog -sv -f ../$(FILELIST)
	@echo "=== Running simulation ==="
	cd $(SIM_DIR) && vsim -c -do "run -all; quit" $(TB)
	@echo "=== Simulation complete ==="

wave:
	@echo "Opening waveform viewer..."
	cd $(SIM_DIR) && vsim -view vsim.wlf &
endif

# VCS simulation
ifeq ($(SIM),vcs)
sim:
	@echo "=== Compiling with VCS ==="
	mkdir -p $(SIM_DIR)
	cd $(SIM_DIR) && vcs -sverilog -full64 -timescale=1ns/1ps \
		-f ../$(FILELIST) -o $(TB).simv
	@echo "=== Running simulation ==="
	cd $(SIM_DIR) && ./$(TB).simv
	@echo "=== Simulation complete ==="

wave:
	@echo "Opening waveform viewer..."
	dve -vpd $(SIM_DIR)/vcdplus.vpd &
endif

#=============================================================================
# Synthesis Targets (Example with Yosys)
#=============================================================================

synthesize:
	@echo "=== Synthesizing with Yosys ==="
	mkdir -p $(SYN_DIR)
	yosys -p "read_verilog -sv $(RTL_FILES); \
		synth -top $(DESIGN); \
		write_verilog $(SYN_DIR)/$(DESIGN)_synth.v; \
		stat"

#=============================================================================
# Utility Targets
#=============================================================================

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(SIM_DIR) $(SYN_DIR)
	rm -f *.vcd *.vvp *.log
	@echo "Clean complete"

help:
	@echo "Systolic Array RTL Makefile"
	@echo "============================"
	@echo ""
	@echo "Usage:"
	@echo "  make sim SIM=<simulator>  - Run simulation"
	@echo "  make wave                 - Open waveform viewer"
	@echo "  make synthesize           - Synthesize design"
	@echo "  make clean                - Clean build artifacts"
	@echo ""
	@echo "Supported simulators (SIM=):"
	@echo "  iverilog  - Icarus Verilog (default)"
	@echo "  verilator - Verilator (lint only)"
	@echo "  questa    - Questa/ModelSim"
	@echo "  vcs       - Synopsys VCS"
	@echo ""
	@echo "Examples:"
	@echo "  make sim                  - Simulate with Icarus Verilog"
	@echo "  make sim SIM=questa       - Simulate with Questa"
	@echo "  make wave                 - View waveforms"
	@echo ""

#=============================================================================
# Parameter Sweep (for testing different configurations)
#=============================================================================

sweep:
	@echo "Running parameter sweep..."
	@for layout in CHANNEL_MAJOR ROW_MAJOR COLUMN_MAJOR; do \
		echo "Testing layout: $$layout"; \
		make sim SIM=iverilog; \
	done

.DEFAULT_GOAL := help
