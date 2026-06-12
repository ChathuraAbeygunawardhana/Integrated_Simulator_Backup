# ============================================================================
# Makefile – Hardware Task Scheduler System
# Supports: basic schedulers, advanced schedulers,
#           AI-MT, BATCH-DNN, BATCH-DNN++
# ============================================================================

# Tool selection: iverilog (default) or verilator
TOOL ?= iverilog

# Source files
BASIC_SRC   = task_scheduler.sv advanced_task_scheduler.sv
BASIC_TB    = task_scheduler_tb.sv

DNN_SRC     = aimt_scheduler.sv batchdnn_scheduler.sv batchdnn_pp_scheduler.sv
DNN_TB      = dnn_schedulers_tb.sv

WRAPPER_SRC = multi_scheduler_wrapper.sv

# -----------------------------------------------------------------------
# Icarus Verilog targets
# -----------------------------------------------------------------------
.PHONY: all sim_basic sim_dnn sim_all clean

all: sim_basic sim_dnn

## Compile + run the original basic/advanced scheduler testbench
sim_basic:
	@echo "=== Compiling basic schedulers ==="
	iverilog -g2012 -Wall \
		-o sim_basic \
		$(BASIC_SRC) $(BASIC_TB)
	@echo "=== Running basic scheduler simulation ==="
	vvp sim_basic
	@echo "=== Done. Waveform: task_scheduler.vcd ==="

## Compile + run the AI-MT / BATCH-DNN / BATCH-DNN++ testbench
sim_dnn:
	@echo "=== Compiling DNN schedulers (AI-MT, BATCH-DNN, BATCH-DNN++) ==="
	iverilog -g2012 -Wall \
		-o sim_dnn \
		$(DNN_SRC) $(DNN_TB)
	@echo "=== Running DNN scheduler simulation ==="
	vvp sim_dnn
	@echo "=== Done. Waveform: dnn_schedulers_tb.vcd ==="

## Run both testbenches
sim_all: sim_basic sim_dnn

# -----------------------------------------------------------------------
# Verilator targets (binary mode, produces faster simulation)
# -----------------------------------------------------------------------
.PHONY: verilate_basic verilate_dnn

verilate_basic:
	@echo "=== Verilating basic schedulers ==="
	verilator --binary --trace -Wall \
		$(BASIC_SRC) $(BASIC_TB) \
		--top-module task_scheduler_tb \
		-Mdir obj_basic
	./obj_basic/Vtask_scheduler_tb

verilate_dnn:
	@echo "=== Verilating DNN schedulers ==="
	verilator --binary --trace -Wall \
		$(DNN_SRC) $(DNN_TB) \
		--top-module dnn_schedulers_tb \
		-Mdir obj_dnn
	./obj_dnn/Vdnn_schedulers_tb

# -----------------------------------------------------------------------
# ModelSim / QuestaSim targets
# -----------------------------------------------------------------------
.PHONY: modelsim_basic modelsim_dnn

modelsim_basic:
	vlog $(BASIC_SRC) $(BASIC_TB)
	vsim -c task_scheduler_tb -do "run -all; quit"

modelsim_dnn:
	vlog $(DNN_SRC) $(DNN_TB)
	vsim -c dnn_schedulers_tb -do "run -all; quit"

# -----------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------
clean:
	rm -f sim_basic sim_dnn *.vcd
	rm -rf obj_basic obj_dnn
