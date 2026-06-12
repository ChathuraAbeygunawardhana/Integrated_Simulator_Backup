# Vivado timing constraints for single_dnn_top.
# 200 MHz default clock; relax via --period if synth fails timing.
create_clock -name clk -period 5.000 [get_ports clk]
set_input_delay  -clock clk 1.0 [all_inputs]
set_output_delay -clock clk 1.0 [all_outputs]
