# Verilator file list for the unified simulator framework.
# Compile packages first.

# --- packages (must compile first) ---
rtl/pkg/sim_framework_pkg.sv
rtl/pkg/stamp_memory_pkg.sv

# --- interfaces ---
rtl/interfaces/mem_if.sv

# --- processing elements ---
rtl/pe/processing_element_os.sv
rtl/pe/processing_element_is.sv
rtl/pe/processing_element_ws.sv

# --- fetchers / address generators ---
rtl/fetchers/address_generator_os.sv
rtl/fetchers/address_generator_is.sv
rtl/fetchers/address_generator_ws.sv
rtl/fetchers/input_fetcher_os.sv
rtl/fetchers/weight_fetcher_os.sv
rtl/fetchers/output_fetcher_os.sv
rtl/fetchers/input_fetcher_is.sv
rtl/fetchers/weight_fetcher_is.sv
rtl/fetchers/output_fetcher_is.sv
rtl/fetchers/input_fetcher_ws.sv
rtl/fetchers/weight_fetcher_ws.sv
rtl/fetchers/output_fetcher_ws.sv

# --- systolic array tops (per-flavor) ---
rtl/array/systolic_array_os.sv
rtl/array/systolic_array_is.sv
rtl/array/systolic_array_ws.sv
rtl/array/systolic_array_wrap.sv

# --- memory subsystem ---
rtl/memory/paged_table.v
rtl/memory/stamp_based_memory_controller.sv
rtl/memory/scratchpad_ram.sv
rtl/memory/axi_read_adapter.sv
rtl/memory/stamp_memory_backend.sv
rtl/memory/paged_memory_backend.sv
rtl/memory/mem_backend_wrap.sv

# --- schedulers ---
rtl/schedulers/task_scheduler.sv
rtl/schedulers/advanced_task_scheduler.sv
rtl/schedulers/aimt_scheduler.sv
rtl/schedulers/batchdnn_scheduler.sv
rtl/schedulers/batchdnn_pp_scheduler.sv
rtl/schedulers/multi_scheduler_wrapper.sv
rtl/schedulers/dnn_scheduler_wrapper.sv
rtl/schedulers/unified_scheduler_wrapper.sv

# --- top-level modules ---
rtl/tops/single_dnn_top.sv
rtl/tops/multi_dnn_top.sv
