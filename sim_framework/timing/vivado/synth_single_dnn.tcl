# Vivado synthesis + report_timing for single_dnn_top.
# Usage: vivado -mode batch -source synth_single_dnn.tcl -tclargs <FRAMEWORK_ROOT>

set FRAMEWORK [lindex $argv 0]
if {$FRAMEWORK eq ""} { set FRAMEWORK [pwd]/.. }

# FPGA part (Artix-7, common dev board); override as needed.
set PART "xc7a100tcsg324-1"
set TOP  "single_dnn_top"

# Source list (rtl/all_sources.f, # comments stripped).
set src_fh [open "$FRAMEWORK/rtl/all_sources.f" r]
set sources {}
foreach line [split [read $src_fh] "\n"] {
    set t [string trim $line]
    if {$t eq "" || [string match "#*" $t]} continue
    lappend sources "$FRAMEWORK/$t"
}
close $src_fh

create_project -part $PART -in_memory single_dnn_synth

foreach s $sources {
    if {[string match "*.v" $s]} {
        read_verilog $s
    } elseif {[string match "*.sv" $s]} {
        read_verilog -sv $s
    }
}
read_xdc "$FRAMEWORK/timing/sdc/single_dnn.xdc"

synth_design -top $TOP -part $PART
opt_design

set rpt_dir "$FRAMEWORK/timing/reports"
file mkdir $rpt_dir
report_timing_summary -file "$rpt_dir/single_dnn_timing_summary.rpt"
report_utilization     -file "$rpt_dir/single_dnn_utilization.rpt"

# Slack check: assert WNS >= 0 (timing closed). Returns nonzero exit if not.
set wns [get_property SLACK [get_timing_paths -max_paths 1 -delay_type max]]
puts "[INFO] Worst negative slack (setup): $wns ns"
if {$wns < 0} {
    puts "[ERROR] Timing not met at requested clock period"
    exit 1
}
puts "[OK] Timing met"
exit 0
