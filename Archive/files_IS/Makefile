#=============================================================================
# Makefile for Input Stationary Systolic Array RTL
# Mirrors the OS and WS Makefiles in structure
#=============================================================================

DESIGN  = input_stationary_top
TB      = input_stationary_tb

RTL_DIR = .
SIM_DIR = sim
SYN_DIR = syn

FILELIST = filelist.f
RTL_FILES = $(shell grep -v "^#" $(FILELIST) | grep "\.sv")

SIM ?= iverilog

#=============================================================================
.PHONY: all clean sim wave synthesize help

all: sim

# ---- Icarus Verilog ----------------------------------------------------------
ifeq ($(SIM),iverilog)
sim:
	@echo "=== Compiling with Icarus Verilog ==="
	mkdir -p $(SIM_DIR)
	iverilog -g2012 -o $(SIM_DIR)/$(TB).vvp -f $(FILELIST)
	@echo "=== Running simulation ==="
	cd $(SIM_DIR) && vvp $(TB).vvp
	@echo "=== Simulation complete ==="

wave:
	gtkwave $(SIM_DIR)/input_stationary.vcd &
endif

# ---- Verilator ---------------------------------------------------------------
ifeq ($(SIM),verilator)
sim:
	@echo "=== Lint check with Verilator ==="
	mkdir -p $(SIM_DIR)
	verilator --lint-only --timing -Wall -Wno-fatal \
		--top-module $(TB) -f $(FILELIST)
	@echo "=== Lint check complete ==="
endif

# ---- Questa / ModelSim -------------------------------------------------------
ifeq ($(SIM),questa)
sim:
	@echo "=== Compiling with Questa ==="
	mkdir -p $(SIM_DIR)
	cd $(SIM_DIR) && vlib work
	cd $(SIM_DIR) && vlog -sv -f ../$(FILELIST)
	@echo "=== Running simulation ==="
	cd $(SIM_DIR) && vsim -c -do "run -all; quit" $(TB)

wave:
	cd $(SIM_DIR) && vsim -view vsim.wlf &
endif

# ---- Synopsys VCS ------------------------------------------------------------
ifeq ($(SIM),vcs)
sim:
	@echo "=== Compiling with VCS ==="
	mkdir -p $(SIM_DIR)
	cd $(SIM_DIR) && vcs -sverilog -full64 -timescale=1ns/1ps \
		-f ../$(FILELIST) -o $(TB).simv
	@echo "=== Running simulation ==="
	cd $(SIM_DIR) && ./$(TB).simv
endif

# ---- Synthesis (Yosys) -------------------------------------------------------
synthesize:
	@echo "=== Synthesizing with Yosys ==="
	mkdir -p $(SYN_DIR)
	yosys -p "read_verilog -sv $(RTL_FILES); \
		synth -top $(DESIGN); \
		write_verilog $(SYN_DIR)/$(DESIGN)_synth.v; \
		stat"

# ---- Utilities ---------------------------------------------------------------
clean:
	rm -rf $(SIM_DIR) $(SYN_DIR)
	rm -f *.vcd *.vvp *.log

help:
	@echo "Input Stationary Systolic Array Makefile"
	@echo "Usage:"
	@echo "  make sim [SIM=iverilog|verilator|questa|vcs]"
	@echo "  make wave"
	@echo "  make synthesize"
	@echo "  make clean"

.DEFAULT_GOAL := help
