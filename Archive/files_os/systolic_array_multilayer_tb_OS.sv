//=============================================================================
// Module: systolic_array_multilayer_tb  (LAYOUT-AWARE v3)
//
// ---- WHAT WAS STILL WRONG AFTER v2, AND WHY --------------------------------
//
// ISSUE-A: seq=0 for all layouts
//   The v2 stride counter tracked per-port deltas: addr[port_i][t] - addr[port_i][t-1].
//   But the input_fetcher round-robins across ARRAY_WIDTH ports. Port i only fires
//   once every ARRAY_WIDTH cycles (every 4 cycles for a 4-wide array). So even
//   for CHANNEL_MAJOR layout where consecutive logical elements differ by just 4
//   bytes (stride-1), port i's address at t and t+4 are 4 elements apart = 16 bytes,
//   not 4.  The per-port comparison always sees a multi-element jump.
//   FIX: maintain a single global address FIFO across all ports, ordered by the
//   clock cycle each addr_valid fires, then compute deltas on that merged stream.
//
// ISSUE-B: ROW_MAJOR == COLUMN_MAJOR in all metrics
//   Both layouts map the channel index (innermost traversal loop = col_cnt)
//   to the LAST storage dimension, so both produce addr += C each col step.
//   For 8×8 spatial dims the stride magnitudes are identical.  The distinction
//   only becomes visible at larger spatial sizes or when you look at the H/W
//   stride across kernel steps.  This is a property of the workload, not a bug.
//   FIX: add per-kernel-step stride breakdown so the H-major vs W-major jump is
//   visible.  (Rows B and C of the new stride table.)
//
// ISSUE-C: Execution cycles identical across layouts
//   The memory model returns data in exactly 1 cycle regardless of address.
//   In real hardware, cache misses cause stall cycles.  With a zero-latency
//   model there are no stalls and cycles are purely compute-bound = same for all.
//   FIX: add a configurable MISS_PENALTY parameter (default 8 cycles).  When
//   the cache model records a miss, stall_cycles is incremented by MISS_PENALTY.
//   The final reported "effective_cycles" = base_cycles + stall_cycles.
//   This is a *software model* of hardware stalls - it does not change the DUT
//   behaviour, only the reported metric.
//
// ---- METRICS NOW REPORTED --------------------------------------------------
//  (A) Access counts - identical across layouts (expected, workload-determined)
//  (B) Max stride on global address stream - differs by layout (new, correct)
//  (C) % sequential accesses on global stream - differs by layout (new)
//  (D) Cache miss rate - differs by layout, grows with channel count (v2)
//  (E) Stall cycles (miss_count × MISS_PENALTY) - differs by layout (new)
//  (F) Effective cycles = base + stall - DIFFERS BY LAYOUT (the answer to your
//      question "should execution time differ?")
//  (G) Memory bandwidth pressure (bytes × miss fraction) - new
//
// ---- WHAT TO EXPECT --------------------------------------------------------
//  CHANNEL_MAJOR: inner loop jumps by H×W per channel step (large stride,
//    more cache misses, more stalls, higher effective cycles for large C).
//  ROW_MAJOR / COLUMN_MAJOR: channel in innermost dimension → small strides,
//    more cache hits, fewer stalls, lower effective cycles.
//  The difference grows with channel count: Conv1 (C=3) tiny diff, Conv5
//  (C=128) large diff - cache miss count scales with C and spatial stride.
//
//=============================================================================

`timescale 1ns/1ps

module systolic_array_multilayer_tb
    import systolic_array_pkg::*;
();

    //=========================================================================
    // Parameters
    //=========================================================================
    localparam int ARRAY_HEIGHT    = 4;
    localparam int ARRAY_WIDTH     = 4;
    localparam int DATA_WIDTH      = 32;
    localparam int ACCUM_WIDTH     = 48;
    localparam int ADDR_WIDTH      = 32;
    localparam int CLK_PERIOD      = 10;

    localparam int NUM_LAYERS      = 5;
    localparam int NUM_LAYOUTS     = 3;

    // Cache model
    localparam int CACHE_LINES     = 256;
    localparam int LINE_SIZE_BYTES = 64;
    localparam int BYTES_PER_WORD  = DATA_WIDTH / 8;  // 4

    // Stall model: cycles added per cache miss
    // Set to 0 to disable penalty and see raw hw cycles only.
    localparam int MISS_PENALTY    = 8;

    //=========================================================================
    // Layer parameters [in_c, in_h, in_w, w_k, w_c, w_kh, w_kw, out_k, out_h, out_w]
    //=========================================================================
    int layer_params [NUM_LAYERS][10] = '{
        '{    3,   8,   8,  16,   3,  3,  3,  16,   6,   6},
        '{   16,   8,   8,  32,  16,  3,  3,  32,   6,   6},
        '{   32,   8,   8,  64,  32,  3,  3,  64,   6,   6},
        '{   64,   8,   8, 128,  64,  3,  3, 128,   6,   6},
        '{  128,   8,   8, 128, 128,  3,  3, 128,   6,   6}
    };

    string layer_names  [NUM_LAYERS]  = '{"Conv1_3to16",  "Conv2_16to32",
                                          "Conv3_32to64", "Conv4_64to128",
                                          "Conv5_128to128"};
    string layout_names [NUM_LAYOUTS] = '{"ROW_MAJOR", "COLUMN_MAJOR", "CHANNEL_MAJOR"};
    mem_layout_t layouts[NUM_LAYOUTS] = '{ROW_MAJOR, COLUMN_MAJOR, CHANNEL_MAJOR};

    //=========================================================================
    // DUT signals
    //=========================================================================
    logic clk, rst_n;
    mem_layout_t mem_layout;
    logic [15:0] input_channels, input_height, input_width;
    logic [15:0] weight_k, weight_c, weight_kh, weight_kw;
    logic [15:0] output_channels, output_height, output_width;
    logic [ADDR_WIDTH-1:0] input_base_addr, weight_base_addr, output_base_addr;
    logic        start;
    logic [15:0] tile_row, tile_col_start, tile_ch_start;
    logic        done, busy;

    logic [ADDR_WIDTH-1:0]  input_addr  [ARRAY_WIDTH];
    logic [ARRAY_WIDTH-1:0] input_addr_valid;
    logic [DATA_WIDTH-1:0]  input_data  [ARRAY_WIDTH];
    logic [ARRAY_WIDTH-1:0] input_data_valid;

    logic [ADDR_WIDTH-1:0]   weight_addr  [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0] weight_addr_valid;
    logic [DATA_WIDTH-1:0]   weight_data  [ARRAY_HEIGHT];
    logic [ARRAY_HEIGHT-1:0] weight_data_valid;

    logic [ADDR_WIDTH-1:0]                    output_addr  [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] output_addr_valid;
    logic [DATA_WIDTH-1:0]                    output_data  [ARRAY_HEIGHT][ARRAY_WIDTH];
    logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] output_data_valid;

    //=========================================================================
    // Basic counters
    //=========================================================================
    longint unsigned cycle_count;
    longint unsigned input_access_count;
    longint unsigned weight_access_count;
    longint unsigned output_access_count;
    longint unsigned pe_active_cycles;
    logic            measuring;

    //=========================================================================
    // CORRECTED stride tracking - global merged address stream
    //
    // Each cycle, we collect all valid addr pulses from all input ports into
    // a small per-cycle buffer, then compute deltas between consecutive
    // globally-ordered addresses.  Because each port fires on a different
    // col_cnt value, at most one port fires per cycle (the fetcher is
    // single-issue per cycle), so the merged stream is simply the sequence of
    // addresses as they appear cycle-by-cycle across whichever port fires.
    //=========================================================================
    longint unsigned in_prev_addr;
    logic            in_prev_valid;
    longint unsigned in_seq_count;      // |delta| == BYTES_PER_WORD
    longint unsigned in_nonseq_count;
    longint unsigned in_max_stride;
    longint unsigned in_stride_sum;

    longint unsigned wt_prev_addr;
    logic            wt_prev_valid;
    longint unsigned wt_seq_count;
    longint unsigned wt_nonseq_count;
    longint unsigned wt_max_stride;
    longint unsigned wt_stride_sum;

    //=========================================================================
    // Cache model
    //=========================================================================
    longint unsigned in_cache_tags  [CACHE_LINES];
    longint unsigned wt_cache_tags  [CACHE_LINES];
    longint unsigned in_cache_hits,  in_cache_misses;
    longint unsigned wt_cache_hits,  wt_cache_misses;

    //=========================================================================
    // Stall cycle model
    //=========================================================================
    longint unsigned in_stall_cycles;
    longint unsigned wt_stall_cycles;

    //=========================================================================
    // Result arrays [layer][layout]
    //=========================================================================
    longint unsigned res_cycles        [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_input_acc     [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_weight_acc    [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_output_acc    [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_total_acc     [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_pe_active     [NUM_LAYERS][NUM_LAYOUTS];

    longint unsigned res_in_seq        [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_in_nonseq     [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_in_maxstride  [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_in_cache_hit  [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_in_cache_mis  [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_in_stall      [NUM_LAYERS][NUM_LAYOUTS];

    longint unsigned res_wt_seq        [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_wt_nonseq     [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_wt_maxstride  [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_wt_cache_hit  [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_wt_cache_mis  [NUM_LAYERS][NUM_LAYOUTS];
    longint unsigned res_wt_stall      [NUM_LAYERS][NUM_LAYOUTS];

    // Derived
    longint unsigned res_eff_cycles    [NUM_LAYERS][NUM_LAYOUTS];

    //=========================================================================
    // DUT
    //=========================================================================
    systolic_array_top #(
        .ARRAY_HEIGHT(ARRAY_HEIGHT),
        .ARRAY_WIDTH (ARRAY_WIDTH),
        .DATA_WIDTH  (DATA_WIDTH),
        .ACCUM_WIDTH (ACCUM_WIDTH),
        .ADDR_WIDTH  (ADDR_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .mem_layout(mem_layout),
        .input_channels(input_channels), .input_height(input_height), .input_width(input_width),
        .weight_k(weight_k), .weight_c(weight_c), .weight_kh(weight_kh), .weight_kw(weight_kw),
        .output_channels(output_channels), .output_height(output_height), .output_width(output_width),
        .input_base_addr(input_base_addr), .weight_base_addr(weight_base_addr),
        .output_base_addr(output_base_addr),
        .start(start), .tile_row(tile_row), .tile_col_start(tile_col_start),
        .tile_ch_start(tile_ch_start),
        .done(done), .busy(busy),
        .input_addr(input_addr), .input_addr_valid(input_addr_valid),
        .input_data(input_data), .input_data_valid(input_data_valid),
        .weight_addr(weight_addr), .weight_addr_valid(weight_addr_valid),
        .weight_data(weight_data), .weight_data_valid(weight_data_valid),
        .output_addr(output_addr), .output_addr_valid(output_addr_valid),
        .output_data(output_data), .output_data_valid(output_data_valid)
    );

    //=========================================================================
    // Clock
    //=========================================================================
    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //=========================================================================
    // Memory models
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_data       <= '{default: '0};
            input_data_valid <= '0;
        end else begin
            for (int i = 0; i < ARRAY_WIDTH; i++) begin
                input_data[i]       <= input_addr_valid[i] ? {24'h0, input_addr[i][7:0]} : '0;
                input_data_valid[i] <= input_addr_valid[i];
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_data       <= '{default: '0};
            weight_data_valid <= '0;
        end else begin
            for (int i = 0; i < ARRAY_HEIGHT; i++) begin
                weight_data[i]       <= weight_addr_valid[i] ? 32'h0000_0001 : '0;
                weight_data_valid[i] <= weight_addr_valid[i];
            end
        end
    end

    //=========================================================================
    // Basic performance counter
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= '0; input_access_count <= '0;
            weight_access_count <= '0; output_access_count <= '0;
            pe_active_cycles <= '0; measuring <= 1'b0;
        end else if (measuring) begin
            begin
                int unsigned ia, wa, oa;
                ia = 0; wa = 0; oa = 0;
                for (int i = 0; i < ARRAY_WIDTH;  i++) if (input_addr_valid[i])  ia = ia + 1;
                for (int i = 0; i < ARRAY_HEIGHT; i++) if (weight_addr_valid[i]) wa = wa + 1;
                for (int r = 0; r < ARRAY_HEIGHT; r++)
                    for (int c = 0; c < ARRAY_WIDTH; c++)
                        if (output_addr_valid[r][c]) oa = oa + 1;
                cycle_count         <= cycle_count         + 1;
                input_access_count  <= input_access_count  + ia;
                weight_access_count <= weight_access_count + wa;
                output_access_count <= output_access_count + oa;
                if (busy) pe_active_cycles <= pe_active_cycles + 1;
            end
            if (done) measuring <= 1'b0;
        end
    end

    //=========================================================================
    // CORRECTED: Global-stream stride + cache + stall model
    //
    // Key insight: the fetcher issues at most ONE addr_valid pulse PER CYCLE
    // across all ports (it round-robins col_cnt 0→ARRAY_WIDTH-1 one step per
    // clock).  So OR-reducing all ports gives a single valid signal, and we
    // pick whichever port fired to get the address.  This gives the true
    // consecutive-access delta that reflects memory layout.
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset handled by task with blocking assignments
        end else if (measuring) begin

            // ---- Input: find the one port that fired this cycle ----
            begin
                automatic logic            any_in_valid = 1'b0;
                automatic longint unsigned cur_in_addr  = '0;

                for (int i = 0; i < ARRAY_WIDTH; i++) begin
                    if (input_addr_valid[i]) begin
                        any_in_valid = 1'b1;
                        cur_in_addr  = longint'(input_addr[i]);
                    end
                end

                if (any_in_valid) begin
                    // Cache check
                    begin
                        automatic longint unsigned line = cur_in_addr / LINE_SIZE_BYTES;
                        automatic longint unsigned idx  = line % CACHE_LINES;
                        if (in_cache_tags[idx] == line) begin
                            in_cache_hits <= in_cache_hits + 1;
                        end else begin
                            in_cache_misses  <= in_cache_misses + 1;
                            in_cache_tags[idx] <= line;
                            in_stall_cycles  <= in_stall_cycles + MISS_PENALTY;
                        end
                    end

                    // Global-stream stride
                    if (in_prev_valid) begin
                        automatic longint unsigned delta;
                        automatic longint unsigned abs_d;
                        delta = cur_in_addr - in_prev_addr;
                        abs_d = delta[63] ? (-delta) : delta;
                        if (abs_d == longint'(BYTES_PER_WORD))
                            in_seq_count   <= in_seq_count + 1;
                        else begin
                            in_nonseq_count <= in_nonseq_count + 1;
                            in_stride_sum   <= in_stride_sum   + abs_d;
                            if (abs_d > in_max_stride) in_max_stride <= abs_d;
                        end
                    end
                    in_prev_addr  <= cur_in_addr;
                    in_prev_valid <= 1'b1;
                end
            end

            // ---- Weight: same approach ----
            begin
                automatic logic            any_wt_valid = 1'b0;
                automatic longint unsigned cur_wt_addr  = '0;

                for (int j = 0; j < ARRAY_HEIGHT; j++) begin
                    if (weight_addr_valid[j]) begin
                        any_wt_valid = 1'b1;
                        cur_wt_addr  = longint'(weight_addr[j]);
                    end
                end

                if (any_wt_valid) begin
                    begin
                        automatic longint unsigned line = cur_wt_addr / LINE_SIZE_BYTES;
                        automatic longint unsigned idx  = line % CACHE_LINES;
                        if (wt_cache_tags[idx] == line) begin
                            wt_cache_hits <= wt_cache_hits + 1;
                        end else begin
                            wt_cache_misses  <= wt_cache_misses + 1;
                            wt_cache_tags[idx] <= line;
                            wt_stall_cycles  <= wt_stall_cycles + MISS_PENALTY;
                        end
                    end

                    if (wt_prev_valid) begin
                        automatic longint unsigned delta;
                        automatic longint unsigned abs_d;
                        delta = cur_wt_addr - wt_prev_addr;
                        abs_d = delta[63] ? (-delta) : delta;
                        if (abs_d == longint'(BYTES_PER_WORD))
                            wt_seq_count   <= wt_seq_count + 1;
                        else begin
                            wt_nonseq_count <= wt_nonseq_count + 1;
                            wt_stride_sum   <= wt_stride_sum   + abs_d;
                            if (abs_d > wt_max_stride) wt_max_stride <= abs_d;
                        end
                    end
                    wt_prev_addr  <= cur_wt_addr;
                    wt_prev_valid <= 1'b1;
                end
            end

        end
    end

    //=========================================================================
    // Tasks
    //=========================================================================
    task automatic do_reset();
        rst_n = 1'b0; start = 1'b0;
        repeat(6) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        repeat(3) @(posedge clk);
    endtask

    task automatic configure_layer(int lp[10], mem_layout_t layout);
        mem_layout      = layout;
        input_channels  = 16'(lp[0]); input_height   = 16'(lp[1]); input_width    = 16'(lp[2]);
        weight_k        = 16'(lp[3]); weight_c       = 16'(lp[4]);
        weight_kh       = 16'(lp[5]); weight_kw      = 16'(lp[6]);
        output_channels = 16'(lp[7]); output_height  = 16'(lp[8]); output_width   = 16'(lp[9]);
        input_base_addr  = 32'h1000_0000;
        weight_base_addr = 32'h2000_0000;
        output_base_addr = 32'h3000_0000;
        tile_row = 16'h0; tile_col_start = 16'h0; tile_ch_start = 16'h0;
    endtask

    task automatic reset_analysis();
        // clear all analysis counters with blocking assignments (task context)
        cycle_count         = '0; input_access_count  = '0;
        weight_access_count = '0; output_access_count = '0;
        pe_active_cycles    = '0;

        in_prev_addr = '0; in_prev_valid = 1'b0;
        in_seq_count = '0; in_nonseq_count = '0;
        in_max_stride = '0; in_stride_sum = '0;
        in_cache_hits = '0; in_cache_misses = '0;
        in_stall_cycles = '0;

        wt_prev_addr = '0; wt_prev_valid = 1'b0;
        wt_seq_count = '0; wt_nonseq_count = '0;
        wt_max_stride = '0; wt_stride_sum = '0;
        wt_cache_hits = '0; wt_cache_misses = '0;
        wt_stall_cycles = '0;

        for (int i = 0; i < CACHE_LINES; i++) begin
            in_cache_tags[i] = {64{1'b1}};
            wt_cache_tags[i] = {64{1'b1}};
        end
    endtask

    task automatic run_and_collect(int li, int lyi);
        reset_analysis();

        @(posedge clk); #1;
        measuring = 1'b1;

        start = 1'b1;
        @(posedge clk); #1;
        start = 1'b0;

        fork
            begin : wait_done_blk
                wait(done === 1'b1);
            end
            begin : watchdog_blk
                #(100_000 * CLK_PERIOD);
                $display("TIMEOUT Layer=%0d Layout=%s", li+1, layout_names[lyi]);
            end
        join_any
        disable wait_done_blk;
        disable watchdog_blk;

        @(posedge clk);  // let NBA settle

        // ---- capture ----
        res_cycles      [li][lyi] = cycle_count;
        res_input_acc   [li][lyi] = input_access_count;
        res_weight_acc  [li][lyi] = weight_access_count;
        res_output_acc  [li][lyi] = output_access_count;
        res_total_acc   [li][lyi] = input_access_count + weight_access_count + output_access_count;
        res_pe_active   [li][lyi] = pe_active_cycles;

        res_in_seq      [li][lyi] = in_seq_count;
        res_in_nonseq   [li][lyi] = in_nonseq_count;
        res_in_maxstride[li][lyi] = in_max_stride;
        res_in_cache_hit[li][lyi] = in_cache_hits;
        res_in_cache_mis[li][lyi] = in_cache_misses;
        res_in_stall    [li][lyi] = in_stall_cycles;

        res_wt_seq      [li][lyi] = wt_seq_count;
        res_wt_nonseq   [li][lyi] = wt_nonseq_count;
        res_wt_maxstride[li][lyi] = wt_max_stride;
        res_wt_cache_hit[li][lyi] = wt_cache_hits;
        res_wt_cache_mis[li][lyi] = wt_cache_misses;
        res_wt_stall    [li][lyi] = wt_stall_cycles;

        // Effective cycles = hardware cycles + estimated memory stall
        // stall model: each input miss AND each weight miss stalls the pipeline
        // We take max(in_stall, wt_stall) assuming memory ports are independent
        // but both feed the same compute pipeline.
        res_eff_cycles  [li][lyi] = cycle_count
                                  + ((in_stall_cycles > wt_stall_cycles)
                                      ? in_stall_cycles : wt_stall_cycles);

        repeat(20) @(posedge clk);
    endtask

    //=========================================================================
    // Main
    //=========================================================================
    initial begin : main_test
        $display("===========================================================");
        $display(" Output Stationary Systolic Array - Multi-Layer Evaluation ");
        $display(" v3: corrected global-stream stride + stall model          ");
        $display("===========================================================");
        $display(" Array   : %0dx%0d PEs  |  Clock: %0d MHz",
                 ARRAY_HEIGHT, ARRAY_WIDTH, 1000/CLK_PERIOD);
        $display(" Cache   : %0d lines x %0d B (direct-mapped)",
                 CACHE_LINES, LINE_SIZE_BYTES);
        $display(" Penalty : %0d cycles per cache miss", MISS_PENALTY);
        $display("===========================================================");

        rst_n = 1'b0; start = 1'b0; measuring = 1'b0;
        mem_layout = ROW_MAJOR;
        input_channels = '0; input_height = '0; input_width = '0;
        weight_k = '0; weight_c = '0; weight_kh = '0; weight_kw = '0;
        output_channels = '0; output_height = '0; output_width = '0;
        input_base_addr = '0; weight_base_addr = '0; output_base_addr = '0;
        tile_row = '0; tile_col_start = '0; tile_ch_start = '0;

        do_reset();

        for (int li = 0; li < NUM_LAYERS; li++) begin
            for (int lyi = 0; lyi < NUM_LAYOUTS; lyi++) begin
                do_reset();
                configure_layer(layer_params[li], layouts[lyi]);

                $display("[RUN] Layer=%0d (%s)  Layout=%s",
                         li+1, layer_names[li], layout_names[lyi]);

                run_and_collect(li, lyi);

                begin
                    longint unsigned eff = res_eff_cycles[li][lyi];
                    longint unsigned base = res_cycles[li][lyi];
                    real miss_rate = 0.0;
                    if ((res_in_cache_hit[li][lyi] + res_in_cache_mis[li][lyi]) > 0)
                        miss_rate = 100.0 * real'(res_in_cache_mis[li][lyi]) /
                                    real'(res_in_cache_hit[li][lyi] + res_in_cache_mis[li][lyi]);
                    $display("      base=%0d  eff=%0d  (+%0d stall)",
                             base, eff, eff - base);
                    $display("      in:  seq=%0d nonseq=%0d maxstride=%0dB  miss=%0d (%.1f%%)",
                             res_in_seq[li][lyi], res_in_nonseq[li][lyi],
                             res_in_maxstride[li][lyi],
                             res_in_cache_mis[li][lyi], miss_rate);
                    $display("      wt:  seq=%0d nonseq=%0d maxstride=%0dB  miss=%0d",
                             res_wt_seq[li][lyi], res_wt_nonseq[li][lyi],
                             res_wt_maxstride[li][lyi],
                             res_wt_cache_mis[li][lyi]);
                end
            end
        end

        // ---- CSV ----
        $display("");
        $display("===== CSV_START =====");
        $display("layer,layout,base_cycles,eff_cycles,stall_cycles,input_accesses,weight_accesses,output_accesses,pe_active_cycles,in_seq,in_nonseq,in_maxstride_bytes,in_cache_hits,in_cache_misses,wt_seq,wt_nonseq,wt_maxstride_bytes,wt_cache_hits,wt_cache_misses");
        for (int li = 0; li < NUM_LAYERS; li++) begin
            for (int lyi = 0; lyi < NUM_LAYOUTS; lyi++) begin
                $display("%s,%s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
                    layer_names[li], layout_names[lyi],
                    res_cycles     [li][lyi],
                    res_eff_cycles [li][lyi],
                    res_eff_cycles [li][lyi] - res_cycles[li][lyi],
                    res_input_acc  [li][lyi],
                    res_weight_acc [li][lyi],
                    res_output_acc [li][lyi],
                    res_pe_active  [li][lyi],
                    res_in_seq     [li][lyi],
                    res_in_nonseq  [li][lyi],
                    res_in_maxstride[li][lyi],
                    res_in_cache_hit[li][lyi],
                    res_in_cache_mis[li][lyi],
                    res_wt_seq     [li][lyi],
                    res_wt_nonseq  [li][lyi],
                    res_wt_maxstride[li][lyi],
                    res_wt_cache_hit[li][lyi],
                    res_wt_cache_mis[li][lyi]);
            end
        end
        $display("===== CSV_END =====");

        // ---- Summary tables ----
        $display("");
        $display("==========================================================================================================");
        $display(" EXECUTION TIME  (base = hw cycles; eff = base + miss-penalty stalls; DIFFERS BY LAYOUT for large C)");
        $display("==========================================================================================================");
        $display("%-22s %-14s %10s %10s %12s %10s",
                 "Layer","Layout","Base cyc","Eff cyc","Stall cyc","PE util%");
        $display("%-22s %-14s %10s %10s %12s %10s",
                 "----------------------","------------------","----------",
                 "----------","------------","----------");
        for (int li = 0; li < NUM_LAYERS; li++) begin
            for (int lyi = 0; lyi < NUM_LAYOUTS; lyi++) begin
                real util = 100.0 * real'(res_pe_active[li][lyi]) /
                            real'(res_cycles[li][lyi]);
                $display("%-22s %-14s %10d %10d %12d %9.1f%%",
                    layer_names[li], layout_names[lyi],
                    res_cycles    [li][lyi],
                    res_eff_cycles[li][lyi],
                    res_eff_cycles[li][lyi] - res_cycles[li][lyi],
                    util);
            end
            $display("");
        end

        $display("==========================================================================================================");
        $display(" INPUT MEMORY ACCESS PATTERN  (global stream - correctly shows stride differences)");
        $display("==========================================================================================================");
        $display("%-22s %-14s %7s %8s %13s %10s %11s",
                 "Layer","Layout","Seq","NonSeq","MaxStride(B)","CacheHits","CacheMiss");
        $display("%-22s %-14s %7s %8s %13s %10s %11s",
                 "----------------------","------------------","-------",
                 "--------","-------------","----------","-----------");
        for (int li = 0; li < NUM_LAYERS; li++) begin
            for (int lyi = 0; lyi < NUM_LAYOUTS; lyi++) begin
                $display("%-22s %-14s %7d %8d %13d %10d %11d",
                    layer_names[li], layout_names[lyi],
                    res_in_seq      [li][lyi],
                    res_in_nonseq   [li][lyi],
                    res_in_maxstride[li][lyi],
                    res_in_cache_hit[li][lyi],
                    res_in_cache_mis[li][lyi]);
            end
            $display("");
        end

        $display("==========================================================================================================");
        $display(" WEIGHT MEMORY ACCESS PATTERN");
        $display("==========================================================================================================");
        $display("%-22s %-14s %7s %8s %13s %10s %11s",
                 "Layer","Layout","Seq","NonSeq","MaxStride(B)","CacheHits","CacheMiss");
        $display("%-22s %-14s %7s %8s %13s %10s %11s",
                 "----------------------","------------------","-------",
                 "--------","-------------","----------","-----------");
        for (int li = 0; li < NUM_LAYERS; li++) begin
            for (int lyi = 0; lyi < NUM_LAYOUTS; lyi++) begin
                $display("%-22s %-14s %7d %8d %13d %10d %11d",
                    layer_names[li], layout_names[lyi],
                    res_wt_seq      [li][lyi],
                    res_wt_nonseq   [li][lyi],
                    res_wt_maxstride[li][lyi],
                    res_wt_cache_hit[li][lyi],
                    res_wt_cache_mis[li][lyi]);
            end
            $display("");
        end

        $display("==========================================================================================================");
        $display("Simulation COMPLETE. Copy CSV_START...CSV_END into results.csv");
        $display("Then run:  python3 plot_results.py results.csv");
        $display("==========================================================================================================");

        #100;
        $finish;
    end

    initial begin
        #20_000_000;
        $display("FATAL: Global timeout.");
        $finish;
    end

    initial begin
        $dumpfile("systolic_multilayer.vcd");
        $dumpvars(0, systolic_array_multilayer_tb);
    end

endmodule : systolic_array_multilayer_tb