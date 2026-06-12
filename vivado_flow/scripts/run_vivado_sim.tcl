##############################################################################
# run_vivado_sim.tcl
# ==================
# Full Vivado simulation flow:
#   1. Create (or reuse) project
#   2. Add RTL sources + headers
#   3. Set top / include dirs
#   4. Launch behavioural simulation
#   5. Run until $finish
#   6. Parse sim_result.txt → exit 0 (pass) or 1 (fail)
#
# Usage (from vivado_flow/ directory):
#   vivado -mode batch -source scripts/run_vivado_sim.tcl \
#          -tclargs <repo_root> [<project_dir>]
#
# Arguments
#   argv[0]  repo_root   – absolute path to the repo root
#   argv[1]  proj_dir    – (optional) where to create the Vivado project
#                          default: <repo_root>/vivado_flow/vivado_project
#
# Outputs (in <proj_dir>)
#   sim_result.txt   – "TEST_PASS" or "TEST_FAIL ..."
#   sim_waveforms.vcd / *.wdb (if DUMP_WAVEFORMS=1)
#   vivado_sim.log
##############################################################################

# ── Resolve arguments ──────────────────────────────────────────────────────
if {[llength $argv] < 1} {
    puts "ERROR: usage: vivado -mode batch -source run_vivado_sim.tcl \
-tclargs <repo_root> \[proj_dir\]"
    exit 1
}

set repo_root [lindex $argv 0]
if {[llength $argv] >= 2} {
    set proj_dir [lindex $argv 1]
} else {
    set proj_dir [file join $repo_root vivado_flow vivado_project]
}

set flow_dir  [file join $repo_root vivado_flow]
set rtl_dir   [file join $repo_root "Archive" "files_os"]   ;# OS dataflow RTL
set tb_dir    [file join $flow_dir  rtl tb]
set cfg_dir   [file join $flow_dir  configs generated]
set tbdata_dir [file join $flow_dir tb_data]
set result_file [file join $proj_dir sim_result.txt]

puts "============================================================"
puts " Vivado Systolic Array Simulation Flow"
puts "============================================================"
puts " repo_root  : $repo_root"
puts " proj_dir   : $proj_dir"
puts " flow_dir   : $flow_dir"
puts " rtl_dir    : $rtl_dir"
puts " tb_dir     : $tb_dir"
puts " cfg_dir    : $cfg_dir"
puts "============================================================"

# ── Helper: write FAIL and exit ────────────────────────────────────────────
proc fail_exit {msg {proj_dir ""}} {
    puts "ERROR: $msg"
    if {$proj_dir ne ""} {
        set fd [open [file join $proj_dir sim_result.txt] w]
        puts $fd "TEST_FAIL errors=$msg"
        close $fd
    }
    exit 1
}

# ── Sanity checks ──────────────────────────────────────────────────────────
foreach f [list \
    [file join $cfg_dir "optimal_config.vh"] \
    [file join $cfg_dir "user_config.vh"] \
    [file join $tb_dir  "systolic_tb.sv"] \
    [file join $rtl_dir "systolic_array_top.sv"] \
    [file join $rtl_dir "systolic_array_pkg.sv"] \
] {
    if {![file exists $f]} {
        fail_exit "Required file not found: $f" $proj_dir
    }
}

# ── Create project ─────────────────────────────────────────────────────────
file mkdir $proj_dir

set proj_name "systolic_sim"
set proj_file [file join $proj_dir "${proj_name}.xpr"]

if {[file exists $proj_file]} {
    puts "Re-opening existing project: $proj_file"
    open_project $proj_file
} else {
    puts "Creating new project: $proj_file"
    create_project $proj_name $proj_dir -part xc7a100tcsg324-1 -force
}

# ── Add RTL sources ────────────────────────────────────────────────────────
# OS dataflow RTL (main design sources)
set rtl_files [list \
    [file join $rtl_dir "systolic_array_pkg.sv"] \
    [file join $rtl_dir "processing_element.sv"] \
    [file join $rtl_dir "input_fetcher.sv"] \
    [file join $rtl_dir "weight_fetcher.sv"] \
    [file join $rtl_dir "output_fetcher.sv"] \
    [file join $rtl_dir "address_generator.sv"] \
    [file join $rtl_dir "systolic_array_top.sv"] \
]

# Filter out files that do not exist (optional sub-modules)
set existing_rtl {}
foreach f $rtl_files {
    if {[file exists $f]} {
        lappend existing_rtl $f
    } else {
        puts "WARNING: RTL file not found (skipping): $f"
    }
}

add_files -norecurse $existing_rtl
set_property file_type SystemVerilog [get_files *.sv]

# Testbench
add_files -fileset sim_1 -norecurse [file join $tb_dir "systolic_tb.sv"]
set_property file_type SystemVerilog [get_files systolic_tb.sv]

# ── Add include directories (for `include "configs/…") ────────────────────
# Vivado resolves `include relative to include_dirs list; we add all three.
set_property include_dirs [list \
    $repo_root \
    $flow_dir  \
    $cfg_dir   \
] [current_fileset]

set_property include_dirs [list \
    $repo_root \
    $flow_dir  \
    $cfg_dir   \
] [get_filesets sim_1]

# ── Set simulation top ─────────────────────────────────────────────────────
set_property top systolic_tb        [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]

# ── Point simulation to tb_data/ ──────────────────────────────────────────
# xsim runs from its own scratch directory; we copy tb_data here so
# $readmemh("tb_data/…") can find the files relative to cwd.
set xsim_dir [file join $proj_dir "${proj_name}.sim" "sim_1" "behav" "xsim"]
file mkdir $xsim_dir
if {[file isdirectory $tbdata_dir]} {
    file copy -force $tbdata_dir [file join $xsim_dir tb_data]
    puts "Copied tb_data/ → $xsim_dir/tb_data"
} else {
    puts "WARNING: tb_data/ not found at $tbdata_dir — \$readmemh will fail."
}

# ── Simulation settings ────────────────────────────────────────────────────
set_property -name {xsim.simulate.runtime} -value {all} \
    -objects [get_filesets sim_1]
set_property -name {xsim.simulate.log_all_signals} -value {true} \
    -objects [get_filesets sim_1]

# ── Compile (elaborate + generate) ────────────────────────────────────────
puts "\n---- Launching simulation (compile + elaborate) ----"
if {[catch {launch_simulation} err]} {
    fail_exit "launch_simulation failed: $err" $proj_dir
}

# ── Run simulation ─────────────────────────────────────────────────────────
puts "\n---- Running simulation (run all) ----"
if {[catch {run all} err]} {
    # Non-fatal: the TB calls $finish; xsim may return a non-zero code
    puts "NOTE: run returned: $err  (may be normal \$finish)"
}

# ── Copy sim_result.txt out of xsim directory ─────────────────────────────
set src_result [file join $xsim_dir "sim_result.txt"]
if {[file exists $src_result]} {
    file copy -force $src_result $result_file
    puts "sim_result.txt → $result_file"
} else {
    # Simulation may have timed out before writing the file
    set fd [open $result_file w]
    puts $fd "TEST_FAIL errors=sim_result_not_written"
    close $fd
    puts "WARNING: sim_result.txt not found — writing FAIL."
}

# ── Parse result ───────────────────────────────────────────────────────────
set fd   [open $result_file r]
set line [gets $fd]
close $fd

puts "\n============================================================"
puts " Simulation result: $line"
puts "============================================================"

close_project

if {[string match "TEST_PASS*" $line]} {
    puts "ALL TESTS PASSED"
    exit 0
} else {
    puts "TESTS FAILED"
    exit 1
}
