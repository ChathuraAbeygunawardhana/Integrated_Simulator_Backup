# Vivado synthesis for multi_dnn_top. Same pattern as synth_single_dnn.tcl.
set FRAMEWORK [lindex $argv 0]
if {$FRAMEWORK eq ""} { set FRAMEWORK [pwd]/.. }
set PART "xc7a100tcsg324-1"
set TOP  "multi_dnn_top"

set src_fh [open "$FRAMEWORK/rtl/all_sources.f" r]
set sources {}
foreach line [split [read $src_fh] "\n"] {
    set t [string trim $line]
    if {$t eq "" || [string match "#*" $t]} continue
    lappend sources "$FRAMEWORK/$t"
}
close $src_fh

create_project -part $PART -in_memory multi_dnn_synth
foreach s $sources {
    if {[string match "*.v" $s]}  { read_verilog $s }
    if {[string match "*.sv" $s]} { read_verilog -sv $s }
}
read_xdc "$FRAMEWORK/timing/sdc/single_dnn.xdc"
synth_design -top $TOP -part $PART
opt_design

set rpt_dir "$FRAMEWORK/timing/reports"
file mkdir $rpt_dir
report_timing_summary -file "$rpt_dir/multi_dnn_timing_summary.rpt"
report_utilization     -file "$rpt_dir/multi_dnn_utilization.rpt"

set wns [get_property SLACK [get_timing_paths -max_paths 1 -delay_type max]]
puts "[INFO] WNS: $wns ns"
if {$wns < 0} { puts "[ERROR] Timing not met"; exit 1 }
puts "[OK] Timing met"; exit 0
