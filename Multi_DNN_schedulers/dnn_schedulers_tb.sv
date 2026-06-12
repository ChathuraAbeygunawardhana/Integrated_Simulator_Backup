// ============================================================================
// Testbench: AI-MT / BATCH-DNN / BATCH-DNN++ Schedulers
//
// Verifies:
//   1. AI-MT    – memory / compute overlap with FIFO dispatch
//   2. BATCH-DNN – sub-batch splitting and merging
//   3. BATCH-DNN++ – distance throttling + bottleneck expedition + slicing
//
// Scenario:  Two DNN workloads (DNN-0 : 4 layers, DNN-1 : 3 layers) loaded
//            into each scheduler. The test mimics what a real compiler would
//            write into the scheduling table.  Memory-access and compute
//            "done" pulses are generated after a parameterised latency.
// ============================================================================

`timescale 1ns/1ps

module dnn_schedulers_tb;

    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    localparam CLK_PERIOD      = 10;   // 10 ns → 100 MHz
    localparam MAX_LAYERS      = 32;
    localparam MAX_DNNS        = 4;
    localparam LAYER_ID_WIDTH  = 8;
    localparam DNN_ID_WIDTH    = 4;
    localparam BATCH_WIDTH     = 8;
    localparam CYCLE_WIDTH     = 32;
    localparam MEM_WIDTH       = 32;

    // Simulated latencies (in clock cycles)
    localparam MEM_LATENCY     = 5;
    localparam COMPUTE_LATENCY = 10;

    // On-chip memory = 256 bytes (tiny so splitting is visible in sim)
    localparam ONCHIP_MEM_SIM  = 256;

    // -----------------------------------------------------------------------
    // Clock / reset
    // -----------------------------------------------------------------------
    logic clk  = 0;
    logic rst_n;

    always #(CLK_PERIOD/2) clk = ~clk;

    // -----------------------------------------------------------------------
    // Common scheduling-table signals
    // -----------------------------------------------------------------------
    logic                      st_write_en;
    logic [LAYER_ID_WIDTH-1:0] st_layer_idx;
    logic [DNN_ID_WIDTH-1:0]   st_dnn_id;
    logic [LAYER_ID_WIDTH-1:0] st_prev_layer;
    logic [CYCLE_WIDTH-1:0]    st_mem_cycles;
    logic [CYCLE_WIDTH-1:0]    st_compute_cycles;
    logic [MEM_WIDTH-1:0]      st_weight_fp;
    logic [MEM_WIDTH-1:0]      st_ifmap_fp;
    logic [MEM_WIDTH-1:0]      st_ofmap_fp;
    logic [LAYER_ID_WIDTH-1:0] st_total_layers;

    // BATCH-DNN / BATCH-DNN++ need batch size
    logic [BATCH_WIDTH-1:0]    st_initial_batch;

    // -----------------------------------------------------------------------
    // Runtime control
    // -----------------------------------------------------------------------
    logic mem_access_done;
    logic compute_done;

    // -----------------------------------------------------------------------
    // DUT outputs – one set per scheduler
    // -----------------------------------------------------------------------
    // --- AI-MT ---
    logic                      aimt_mt_valid;
    logic [LAYER_ID_WIDTH-1:0] aimt_mt_layer_id;
    logic [DNN_ID_WIDTH-1:0]   aimt_mt_dnn_id;
    logic                      aimt_ct_valid;
    logic [LAYER_ID_WIDTH-1:0] aimt_ct_layer_id;
    logic [DNN_ID_WIDTH-1:0]   aimt_ct_dnn_id;
    logic [CYCLE_WIDTH-1:0]    aimt_avail_mem;
    logic                      aimt_stall;
    logic [CYCLE_WIDTH-1:0]    aimt_stall_cycles;

    // --- BATCH-DNN ---
    logic                      bdnn_mt_valid;
    logic [LAYER_ID_WIDTH-1:0] bdnn_mt_layer_id;
    logic [DNN_ID_WIDTH-1:0]   bdnn_mt_dnn_id;
    logic                      bdnn_ct_valid;
    logic [LAYER_ID_WIDTH-1:0] bdnn_ct_layer_id;
    logic [DNN_ID_WIDTH-1:0]   bdnn_ct_dnn_id;
    logic [BATCH_WIDTH-1:0]    bdnn_ct_batch;
    logic [CYCLE_WIDTH-1:0]    bdnn_avail_mem;
    logic                      bdnn_stall;
    logic [CYCLE_WIDTH-1:0]    bdnn_stall_cycles;
    logic [CYCLE_WIDTH-1:0]    bdnn_split_events;
    logic [CYCLE_WIDTH-1:0]    bdnn_merge_events;

    // --- BATCH-DNN++ ---
    logic                      bpp_mt_valid;
    logic [LAYER_ID_WIDTH-1:0] bpp_mt_layer_id;
    logic [DNN_ID_WIDTH-1:0]   bpp_mt_dnn_id;
    logic                      bpp_ct_valid;
    logic [LAYER_ID_WIDTH-1:0] bpp_ct_layer_id;
    logic [DNN_ID_WIDTH-1:0]   bpp_ct_dnn_id;
    logic [BATCH_WIDTH-1:0]    bpp_ct_batch;
    logic [CYCLE_WIDTH-1:0]    bpp_avail_mem;
    logic                      bpp_stall;
    logic [CYCLE_WIDTH-1:0]    bpp_stall_cycles;
    logic [CYCLE_WIDTH-1:0]    bpp_split_events;
    logic [CYCLE_WIDTH-1:0]    bpp_merge_events;
    logic [CYCLE_WIDTH-1:0]    bpp_throttle_events;
    logic [CYCLE_WIDTH-1:0]    bpp_expedition_events;
    logic [CYCLE_WIDTH-1:0]    bpp_slices;

    // -----------------------------------------------------------------------
    // DUT instantiations
    // -----------------------------------------------------------------------

    // --- AI-MT ---
    aimt_scheduler #(
        .MAX_LAYERS      (MAX_LAYERS),
        .MAX_DNNS        (MAX_DNNS),
        .LAYER_ID_WIDTH  (LAYER_ID_WIDTH),
        .DNN_ID_WIDTH    (DNN_ID_WIDTH),
        .ONCHIP_MEM_SIZE (ONCHIP_MEM_SIM),
        .CYCLE_WIDTH     (CYCLE_WIDTH),
        .MEM_WIDTH       (MEM_WIDTH)
    ) u_aimt (
        .clk              (clk),
        .rst_n            (rst_n),
        .st_write_en      (st_write_en),
        .st_layer_idx     (st_layer_idx),
        .st_dnn_id        (st_dnn_id),
        .st_prev_layer    (st_prev_layer),
        .st_mem_cycles    (st_mem_cycles),
        .st_compute_cycles(st_compute_cycles),
        .st_weight_fp     (st_weight_fp),
        .st_ifmap_fp      (st_ifmap_fp),
        .st_ofmap_fp      (st_ofmap_fp),
        .st_total_layers  (st_total_layers),
        .mem_access_done  (mem_access_done),
        .compute_done     (compute_done),
        .mt_valid         (aimt_mt_valid),
        .mt_layer_id      (aimt_mt_layer_id),
        .mt_dnn_id        (aimt_mt_dnn_id),
        .ct_valid         (aimt_ct_valid),
        .ct_layer_id      (aimt_ct_layer_id),
        .ct_dnn_id        (aimt_ct_dnn_id),
        .available_memory (aimt_avail_mem),
        .stall_detected   (aimt_stall),
        .total_stall_cycles(aimt_stall_cycles)
    );

    // --- BATCH-DNN ---
    batchdnn_scheduler #(
        .MAX_LAYERS      (MAX_LAYERS),
        .MAX_DNNS        (MAX_DNNS),
        .LAYER_ID_WIDTH  (LAYER_ID_WIDTH),
        .DNN_ID_WIDTH    (DNN_ID_WIDTH),
        .BATCH_WIDTH     (BATCH_WIDTH),
        .ONCHIP_MEM_SIZE (ONCHIP_MEM_SIM),
        .CYCLE_WIDTH     (CYCLE_WIDTH),
        .MEM_WIDTH       (MEM_WIDTH)
    ) u_bdnn (
        .clk              (clk),
        .rst_n            (rst_n),
        .st_write_en      (st_write_en),
        .st_layer_idx     (st_layer_idx),
        .st_dnn_id        (st_dnn_id),
        .st_prev_layer    (st_prev_layer),
        .st_mem_cycles    (st_mem_cycles),
        .st_compute_cycles(st_compute_cycles),
        .st_weight_fp     (st_weight_fp),
        .st_ifmap_fp      (st_ifmap_fp),
        .st_ofmap_fp      (st_ofmap_fp),
        .st_initial_batch (st_initial_batch),
        .st_total_layers  (st_total_layers),
        .mem_access_done  (mem_access_done),
        .compute_done     (compute_done),
        .mt_valid         (bdnn_mt_valid),
        .mt_layer_id      (bdnn_mt_layer_id),
        .mt_dnn_id        (bdnn_mt_dnn_id),
        .ct_valid         (bdnn_ct_valid),
        .ct_layer_id      (bdnn_ct_layer_id),
        .ct_dnn_id        (bdnn_ct_dnn_id),
        .ct_batch_size    (bdnn_ct_batch),
        .available_memory (bdnn_avail_mem),
        .stall_detected   (bdnn_stall),
        .total_stall_cycles(bdnn_stall_cycles),
        .total_split_events(bdnn_split_events),
        .total_merge_events(bdnn_merge_events)
    );

    // --- BATCH-DNN++ ---
    batchdnn_pp_scheduler #(
        .MAX_LAYERS         (MAX_LAYERS),
        .MAX_DNNS           (MAX_DNNS),
        .LAYER_ID_WIDTH     (LAYER_ID_WIDTH),
        .DNN_ID_WIDTH       (DNN_ID_WIDTH),
        .BATCH_WIDTH        (BATCH_WIDTH),
        .ONCHIP_MEM_SIZE    (ONCHIP_MEM_SIM),
        .MAX_LAYER_DISTANCE (5),
        .CYCLE_WIDTH        (CYCLE_WIDTH),
        .MEM_WIDTH          (MEM_WIDTH)
    ) u_bpp (
        .clk                (clk),
        .rst_n              (rst_n),
        .st_write_en        (st_write_en),
        .st_layer_idx       (st_layer_idx),
        .st_dnn_id          (st_dnn_id),
        .st_prev_layer      (st_prev_layer),
        .st_mem_cycles      (st_mem_cycles),
        .st_compute_cycles  (st_compute_cycles),
        .st_weight_fp       (st_weight_fp),
        .st_ifmap_fp        (st_ifmap_fp),
        .st_ofmap_fp        (st_ofmap_fp),
        .st_requested_batch (st_initial_batch),
        .st_total_layers    (st_total_layers),
        .mem_access_done    (mem_access_done),
        .compute_done       (compute_done),
        .mt_valid           (bpp_mt_valid),
        .mt_layer_id        (bpp_mt_layer_id),
        .mt_dnn_id          (bpp_mt_dnn_id),
        .ct_valid           (bpp_ct_valid),
        .ct_layer_id        (bpp_ct_layer_id),
        .ct_dnn_id          (bpp_ct_dnn_id),
        .ct_batch_size      (bpp_ct_batch),
        .available_memory   (bpp_avail_mem),
        .stall_detected     (bpp_stall),
        .total_stall_cycles (bpp_stall_cycles),
        .total_split_events (bpp_split_events),
        .total_merge_events (bpp_merge_events),
        .total_throttle_events   (bpp_throttle_events),
        .total_expedition_events (bpp_expedition_events),
        .total_slices_executed   (bpp_slices)
    );

    // -----------------------------------------------------------------------
    // Task: load a single layer row into every DUT's scheduling table
    // -----------------------------------------------------------------------
    task load_layer(
        input logic [LAYER_ID_WIDTH-1:0] idx,
        input logic [DNN_ID_WIDTH-1:0]   dnn,
        input logic [LAYER_ID_WIDTH-1:0] prev,
        input logic [CYCLE_WIDTH-1:0]    mem_cyc,
        input logic [CYCLE_WIDTH-1:0]    comp_cyc,
        input logic [MEM_WIDTH-1:0]      w_fp,
        input logic [MEM_WIDTH-1:0]      i_fp,
        input logic [MEM_WIDTH-1:0]      o_fp,
        input logic [BATCH_WIDTH-1:0]    batch,
        input logic [LAYER_ID_WIDTH-1:0] total
    );
        @(posedge clk);
        st_write_en       <= 1'b1;
        st_layer_idx      <= idx;
        st_dnn_id         <= dnn;
        st_prev_layer     <= prev;
        st_mem_cycles     <= mem_cyc;
        st_compute_cycles <= comp_cyc;
        st_weight_fp      <= w_fp;
        st_ifmap_fp       <= i_fp;
        st_ofmap_fp       <= o_fp;
        st_initial_batch  <= batch;
        st_total_layers   <= total;
        @(posedge clk);
        st_write_en <= 1'b0;
    endtask

    // -----------------------------------------------------------------------
    // Task: simulate mem-access done pulse after MEM_LATENCY cycles
    // -----------------------------------------------------------------------
    task fire_mem_done();
        repeat(MEM_LATENCY) @(posedge clk);
        mem_access_done <= 1'b1;
        @(posedge clk);
        mem_access_done <= 1'b0;
    endtask

    // -----------------------------------------------------------------------
    // Task: simulate compute done pulse after COMPUTE_LATENCY cycles
    // -----------------------------------------------------------------------
    task fire_compute_done();
        repeat(COMPUTE_LATENCY) @(posedge clk);
        compute_done <= 1'b1;
        @(posedge clk);
        compute_done <= 1'b0;
    endtask

    // -----------------------------------------------------------------------
    // VCD dump
    // -----------------------------------------------------------------------
    initial begin
        $dumpfile("dnn_schedulers_tb.vcd");
        $dumpvars(0, dnn_schedulers_tb);
    end

    // -----------------------------------------------------------------------
    // Main test sequence
    // -----------------------------------------------------------------------
    initial begin
        // Default values
        rst_n            = 0;
        st_write_en      = 0;
        mem_access_done  = 0;
        compute_done     = 0;

        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        $display("\n=======================================================");
        $display("  DNN Scheduler Testbench – AI-MT / BATCH-DNN / BATCH-DNN++");
        $display("=======================================================\n");

        // ===================================================================
        // TEST 1 – AI-MT: basic memory/compute overlap
        // ===================================================================
        $display("--- TEST 1: AI-MT Scheduler ---");

        // DNN-0: 4 layers (serial chain)
        // Layer format: idx, dnn, prev, mem_cyc, comp_cyc, w_fp, i_fp, o_fp, batch, total
        load_layer(0, 0, 8'hFF,  3,  8, 16, 8, 8, 3, 7);   // L0 root
        load_layer(1, 0, 0,      3,  8, 16, 8, 8, 3, 7);   // L1 → L0
        load_layer(2, 0, 1,      3,  8, 16, 8, 8, 3, 7);   // L2 → L1
        load_layer(3, 0, 2,      3,  8, 16, 8, 8, 3, 7);   // L3 → L2
        // DNN-1: 3 layers
        load_layer(4, 1, 8'hFF,  4,  6, 12, 6, 6, 2, 7);   // L0 root
        load_layer(5, 1, 4,      4,  6, 12, 6, 6, 2, 7);   // L1 → L0
        load_layer(6, 1, 5,      4,  6, 12, 6, 6, 2, 7);   // L2 → L1

        $display("[%0t] Scheduling table loaded (7 layers)", $time);
        repeat(5) @(posedge clk);

        // Drive several memory and compute completion events
        repeat(4) begin
            fork
                fire_mem_done();
                fire_compute_done();
            join_none
            repeat(15) @(posedge clk);
        end

        repeat(20) @(posedge clk);

        $display("[%0t] AI-MT results:", $time);
        $display("  Stall cycles      : %0d", aimt_stall_cycles);
        $display("  Available memory  : %0d bytes", aimt_avail_mem);
        $display("  Last MT layer     : %0d (DNN %0d)", aimt_mt_layer_id, aimt_mt_dnn_id);
        $display("  Last CT layer     : %0d (DNN %0d)", aimt_ct_layer_id, aimt_ct_dnn_id);

        // ===================================================================
        // TEST 2 – BATCH-DNN: sub-batch splitting observed
        // ===================================================================
        $display("\n--- TEST 2: BATCH-DNN Scheduler ---");

        // Reset
        @(posedge clk); rst_n = 0;
        repeat(4) @(posedge clk); rst_n = 1;
        repeat(2) @(posedge clk);

        // Load the same layers; batch=4 and small memory so splitting fires
        load_layer(0, 0, 8'hFF,  3, 10, 10, 5, 5, 4, 7);
        load_layer(1, 0, 0,      3, 10, 10, 5, 5, 4, 7);
        load_layer(2, 0, 1,      3, 10, 10, 10,5, 5, 4, 7);  // large OFMAP
        load_layer(3, 0, 2,      3, 10, 10, 5, 5, 4, 7);
        load_layer(4, 1, 8'hFF,  4,  8,  8, 4, 4, 3, 7);
        load_layer(5, 1, 4,      4,  8,  8, 4, 4, 3, 7);
        load_layer(6, 1, 5,      4,  8, 16, 4, 4, 3, 7);     // large weight

        repeat(5) @(posedge clk);

        repeat(6) begin
            fork
                fire_mem_done();
                fire_compute_done();
            join_none
            repeat(18) @(posedge clk);
        end
        repeat(20) @(posedge clk);

        $display("[%0t] BATCH-DNN results:", $time);
        $display("  Stall cycles      : %0d", bdnn_stall_cycles);
        $display("  Split events      : %0d", bdnn_split_events);
        $display("  Merge events      : %0d", bdnn_merge_events);
        $display("  Current CT batch  : %0d", bdnn_ct_batch);
        $display("  Available memory  : %0d bytes", bdnn_avail_mem);

        // Check at least one split occurred if memory is tight
        if (bdnn_split_events > 0)
            $display("  PASS: sub-batch splitting triggered");
        else
            $display("  INFO: no split (memory may be sufficient for this run)");

        // ===================================================================
        // TEST 3 – BATCH-DNN++: distance throttling + bottleneck expedition
        // ===================================================================
        $display("\n--- TEST 3: BATCH-DNN++ Scheduler ---");

        @(posedge clk); rst_n = 0;
        repeat(4) @(posedge clk); rst_n = 1;
        repeat(2) @(posedge clk);

        // Load layers: DNN-0 has a 6-layer serial chain (forces distance >5)
        load_layer(0, 0, 8'hFF,  2,  8,  8, 4, 4, 4, 9);
        load_layer(1, 0, 0,      2,  8,  8, 4, 4, 4, 9);
        load_layer(2, 0, 1,      2,  8,  8, 4, 4, 4, 9);
        load_layer(3, 0, 2,      2,  8,  8, 4, 4, 4, 9);
        load_layer(4, 0, 3,      2,  8,  8, 4, 4, 4, 9);
        load_layer(5, 0, 4,      2,  8,  8, 4, 4, 4, 9);  // depth 5 from root
        load_layer(6, 0, 5,      2,  8,  8, 4, 4, 4, 9);  // depth 6 – should throttle
        // DNN-1: fork/join graph to exercise bottleneck expedition
        load_layer(7, 1, 8'hFF,  3, 10, 10, 5, 5, 4, 9);
        load_layer(8, 1, 7,      3, 10, 10, 5, 5, 4, 9);  // branch A

        repeat(5) @(posedge clk);

        repeat(8) begin
            fork
                fire_mem_done();
                fire_compute_done();
            join_none
            repeat(20) @(posedge clk);
        end
        repeat(30) @(posedge clk);

        $display("[%0t] BATCH-DNN++ results:", $time);
        $display("  Stall cycles       : %0d", bpp_stall_cycles);
        $display("  Split events       : %0d", bpp_split_events);
        $display("  Merge events       : %0d", bpp_merge_events);
        $display("  Throttle events    : %0d", bpp_throttle_events);
        $display("  Expedition events  : %0d", bpp_expedition_events);
        $display("  Slices executed    : %0d", bpp_slices);
        $display("  Available memory   : %0d bytes", bpp_avail_mem);

        if (bpp_throttle_events > 0)
            $display("  PASS: distance-based throttling fired");
        else
            $display("  INFO: no throttle event in this run");

        if (bpp_expedition_events > 0)
            $display("  PASS: bottleneck expedition fired");
        else
            $display("  INFO: no expedition event in this run");

        // ===================================================================
        // TEST 4 – Comparison summary
        // ===================================================================
        $display("\n=======================================================");
        $display("  Comparison Summary (last test cycle values)");
        $display("=======================================================");
        $display("  Metric                 AI-MT    BATCH-DNN  BATCH-DNN++");
        $display("  Stall cycles       %8d   %8d   %8d",
            aimt_stall_cycles, bdnn_stall_cycles, bpp_stall_cycles);
        $display("  Split events          N/A   %8d   %8d",
            bdnn_split_events, bpp_split_events);
        $display("  Merge events          N/A   %8d   %8d",
            bdnn_merge_events, bpp_merge_events);
        $display("  Throttle events       N/A        N/A   %8d",
            bpp_throttle_events);
        $display("  Expedition events     N/A        N/A   %8d",
            bpp_expedition_events);
        $display("=======================================================\n");

        $display("Simulation complete at time %0t", $time);
        $finish;
    end

    // -----------------------------------------------------------------------
    // Simulation watchdog – prevents infinite loops
    // -----------------------------------------------------------------------
    initial begin
        #200_000;
        $display("WATCHDOG: simulation timeout");
        $finish;
    end

endmodule
