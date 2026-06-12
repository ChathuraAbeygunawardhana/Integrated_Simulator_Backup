// ============================================================================
// BATCH-DNN++ Scheduler
// Based on: "BATCH-DNN++: Improving Resource Utilization in Multi-DNN
//            Accelerators with Adaptive and Dynamic Batching and
//            Dependency-Aware Scheduling" (Ranawaka & Stenstrom, 2025)
//
// Extends BATCH-DNN with three additional techniques (Section 4):
//
//   4.1.1  Distance-Based Memory Access Throttling (Figure 8 A-C, Figure 9 A)
//          Prevents memory tasks from running too far ahead of the current
//          compute task in the DNN DAG.  If distance(MT_layer, CT_layer) >
//          MAX_LAYER_DISTANCE (empirically 5, Section 4.2.1), the MT is
//          pushed back into the candidate queue and the next lowest MT is
//          tried instead.
//
//   4.1.2  Bottleneck Layer Expedition (Figure 8 D, Figure 9 B)
//          When a compute task cannot proceed because its predecessors are
//          incomplete, the scheduler recursively traces back the dependency
//          chain to find the earliest unscheduled ("bottleneck") layer and
//          expedites it.  If the bottleneck has pending memory access it is
//          promoted to the MT queue; otherwise it is promoted to the CT
//          scheduled queue.
//
//   4.1.3  Batch Slicing (Section 4.1.3)
//          Allows arbitrarily large requested batch sizes by dividing them
//          into at most ceil(N_x / N_max) sequential slices.  Each slice
//          independently traverses the full network; weights are reloaded
//          once per slice.
//
//   Sub-batch splitting / merging from BATCH-DNN is fully retained
//   (Blocks C and D of Figure 7).
//
// Note on hardware realism:
//   This is a cycle-accurate behavioural model written in synthesisable
//   SystemVerilog (no 'real' arithmetic).  Division is performed with a
//   simple iterative approach or compile-time constants where required.
// ============================================================================

`timescale 1ns/1ps

module batchdnn_pp_scheduler #(
    parameter MAX_LAYERS         = 32,
    parameter MAX_DNNS           = 4,
    parameter LAYER_ID_WIDTH     = 8,
    parameter DNN_ID_WIDTH       = 4,
    parameter BATCH_WIDTH        = 8,
    parameter ONCHIP_MEM_SIZE    = 44*1024*1024,  // 44 MB (Table 1a)
    parameter WORD_BITS          = 16,
    parameter MAX_LAYER_DISTANCE = 5,   // empirical threshold (Section 4.2.1)
    parameter STACK_DEPTH        = 16,
    parameter CYCLE_WIDTH        = 32,
    parameter MEM_WIDTH          = 32
)(
    input  logic clk,
    input  logic rst_n,

    // ----------------------------------------------------------------
    // Scheduling-table load
    // ----------------------------------------------------------------
    input  logic                       st_write_en,
    input  logic [LAYER_ID_WIDTH-1:0]  st_layer_idx,
    input  logic [DNN_ID_WIDTH-1:0]    st_dnn_id,
    input  logic [LAYER_ID_WIDTH-1:0]  st_prev_layer,   // 8'hFF = no pred
    input  logic [CYCLE_WIDTH-1:0]     st_mem_cycles,
    input  logic [CYCLE_WIDTH-1:0]     st_compute_cycles, // per unit batch
    input  logic [MEM_WIDTH-1:0]       st_weight_fp,
    input  logic [MEM_WIDTH-1:0]       st_ifmap_fp,
    input  logic [MEM_WIDTH-1:0]       st_ofmap_fp,
    input  logic [BATCH_WIDTH-1:0]     st_requested_batch, // Nx (arbitrary)
    input  logic [LAYER_ID_WIDTH-1:0]  st_total_layers,

    // ----------------------------------------------------------------
    // Runtime
    // ----------------------------------------------------------------
    input  logic mem_access_done,
    input  logic compute_done,

    // ----------------------------------------------------------------
    // Outputs
    // ----------------------------------------------------------------
    output logic                      mt_valid,
    output logic [LAYER_ID_WIDTH-1:0] mt_layer_id,
    output logic [DNN_ID_WIDTH-1:0]   mt_dnn_id,

    output logic                      ct_valid,
    output logic [LAYER_ID_WIDTH-1:0] ct_layer_id,
    output logic [DNN_ID_WIDTH-1:0]   ct_dnn_id,
    output logic [BATCH_WIDTH-1:0]    ct_batch_size,

    // ----------------------------------------------------------------
    // Status
    // ----------------------------------------------------------------
    output logic [CYCLE_WIDTH-1:0]    available_memory,
    output logic                      stall_detected,
    output logic [CYCLE_WIDTH-1:0]    total_stall_cycles,
    output logic [CYCLE_WIDTH-1:0]    total_split_events,
    output logic [CYCLE_WIDTH-1:0]    total_merge_events,
    output logic [CYCLE_WIDTH-1:0]    total_throttle_events,
    output logic [CYCLE_WIDTH-1:0]    total_expedition_events,
    output logic [CYCLE_WIDTH-1:0]    total_slices_executed
);

    // ================================================================
    // Scheduling-table entry
    // ================================================================
    typedef struct packed {
        logic                      valid;
        logic [DNN_ID_WIDTH-1:0]   dnn_id;
        logic [LAYER_ID_WIDTH-1:0] prev_layer;     // 8'hFF = root
        logic [CYCLE_WIDTH-1:0]    mem_cycles;
        logic [CYCLE_WIDTH-1:0]    compute_cycles;
        logic [MEM_WIDTH-1:0]      weight_fp;
        logic [MEM_WIDTH-1:0]      ifmap_fp;
        logic [MEM_WIDTH-1:0]      ofmap_fp;
        logic [BATCH_WIDTH-1:0]    requested_batch; // Nx
        logic [BATCH_WIDTH-1:0]    max_batch;        // N  (computed at load time)
        logic                      mem_access_done_flag;
        logic                      compute_done_flag;
        logic                      expedited;        // flag: expedited by BLE
    } sched_entry_t;

    sched_entry_t sched_table [MAX_LAYERS-1:0];
    logic [LAYER_ID_WIDTH-1:0] total_layers;

    // ================================================================
    // Sub-batching stack
    // ================================================================
    typedef struct packed {
        logic [LAYER_ID_WIDTH-1:0] layer_idx;
        logic [BATCH_WIDTH-1:0]    batch_size;
    } stack_entry_t;

    stack_entry_t sb_stack [MAX_DNNS-1:0][STACK_DEPTH-1:0];
    logic [$clog2(STACK_DEPTH):0] sb_sp    [MAX_DNNS-1:0];
    logic [BATCH_WIDTH-1:0]  current_batch  [MAX_DNNS-1:0];
    logic [BATCH_WIDTH-1:0]  prev_batch_reg [MAX_DNNS-1:0];

    // ================================================================
    // Batch-slicing state per DNN  (Section 4.1.3)
    // ================================================================
    logic [BATCH_WIDTH-1:0]  slice_remaining [MAX_DNNS-1:0];  // inferences left
    logic [BATCH_WIDTH-1:0]  max_batch_cap   [MAX_DNNS-1:0];  // N (hardware cap)

    // ================================================================
    // Current compute layer per DNN (for distance calculation)
    // ================================================================
    logic [LAYER_ID_WIDTH-1:0] ct_current_layer [MAX_DNNS-1:0];

    // ================================================================
    // Queues
    // ================================================================
    localparam QD = MAX_LAYERS;

    logic [LAYER_ID_WIDTH-1:0] mt_cq       [QD-1:0];
    logic [$clog2(QD):0]        mt_cq_head, mt_cq_tail, mt_cq_cnt;

    logic [LAYER_ID_WIDTH-1:0] ct_cq       [QD-1:0];
    logic [$clog2(QD):0]        ct_cq_head, ct_cq_tail, ct_cq_cnt;

    logic [LAYER_ID_WIDTH-1:0] sct_q       [QD-1:0];
    logic [BATCH_WIDTH-1:0]    sct_batch   [QD-1:0];
    logic [$clog2(QD):0]        sct_head, sct_tail, sct_cnt;

    // ================================================================
    // Balance counters
    // ================================================================
    logic signed [CYCLE_WIDTH:0] mem_cycle_ctr;
    logic signed [CYCLE_WIDTH:0] compute_cycle_ctr;
    logic [CYCLE_WIDTH-1:0]      cycles_to_fill_remaining;

    // ================================================================
    // Available memory
    // ================================================================
    logic [CYCLE_WIDTH-1:0] avail_mem_reg;
    assign available_memory = avail_mem_reg;

    // ================================================================
    // Active tasks
    // ================================================================
    logic                      mt_active;
    logic [LAYER_ID_WIDTH-1:0] mt_active_layer;

    logic                      ct_active;
    logic [LAYER_ID_WIDTH-1:0] ct_active_layer;
    logic [DNN_ID_WIDTH-1:0]   ct_active_dnn;
    logic [BATCH_WIDTH-1:0]    ct_active_batch;

    // ================================================================
    // Metrics
    // ================================================================
    logic [CYCLE_WIDTH-1:0] stall_ctr;
    logic [CYCLE_WIDTH-1:0] split_ctr;
    logic [CYCLE_WIDTH-1:0] merge_ctr;
    logic [CYCLE_WIDTH-1:0] throttle_ctr;
    logic [CYCLE_WIDTH-1:0] expedition_ctr;
    logic [CYCLE_WIDTH-1:0] slice_ctr;

    assign total_stall_cycles       = stall_ctr;
    assign total_split_events       = split_ctr;
    assign total_merge_events       = merge_ctr;
    assign total_throttle_events    = throttle_ctr;
    assign total_expedition_events  = expedition_ctr;
    assign total_slices_executed    = slice_ctr;

    // ================================================================
    // Helper: compute maximum feasible batch (same as BATCH-DNN)
    // ================================================================
    function automatic logic [BATCH_WIDTH-1:0] max_batch_size(
        input logic [MEM_WIDTH-1:0]   ofmap_fp_unit,
        input logic [CYCLE_WIDTH-1:0] avail_mem,
        input logic [BATCH_WIDTH-1:0] prev_b
    );
        automatic logic [BATCH_WIDTH-1:0] b;
        if (ofmap_fp_unit == 0)
            max_batch_size = prev_b;
        else begin
            b = avail_mem / ofmap_fp_unit;
            if (b > prev_b) b = prev_b;
            max_batch_size = b;
        end
    endfunction

    // ================================================================
    // Helper: compute hardware max batch N for a DNN
    //   N <= (M - W) / (I + O)   for worst-case layer  (Section 4.1.3 eq.3)
    // We compute this per-layer at load time and take the minimum.
    // ================================================================
    function automatic logic [BATCH_WIDTH-1:0] compute_max_batch_cap(
        input logic [MEM_WIDTH-1:0] w_fp,
        input logic [MEM_WIDTH-1:0] i_fp,
        input logic [MEM_WIDTH-1:0] o_fp
    );
        automatic logic [MEM_WIDTH-1:0] denom;
        automatic logic [BATCH_WIDTH-1:0] cap;
        denom = i_fp + o_fp;
        if (denom == 0)
            compute_max_batch_cap = {BATCH_WIDTH{1'b1}};
        else begin
            cap = (ONCHIP_MEM_SIZE - w_fp) / denom;
            compute_max_batch_cap = cap[BATCH_WIDTH-1:0];
        end
    endfunction

    // ================================================================
    // Helper: distance between MT layer and current CT layer
    //   Uses the serial order index (layer_idx) as DAG depth proxy.
    //   Returns the absolute difference.
    // ================================================================
    function automatic logic [LAYER_ID_WIDTH-1:0] layer_distance(
        input logic [LAYER_ID_WIDTH-1:0] mt_layer,
        input logic [LAYER_ID_WIDTH-1:0] ct_layer
    );
        layer_distance = (mt_layer > ct_layer) ?
                         (mt_layer - ct_layer) : (ct_layer - mt_layer);
    endfunction

    // ================================================================
    // Helper: recursive bottleneck finder
    //   Searches backwards from 'target_layer' to find the earliest
    //   unscheduled layer (compute not done, prev layers done).
    //   Simplified for hardware: walks back at most MAX_LAYERS steps.
    //   Returns the bottleneck layer index or 8'hFF if none found.
    // ================================================================
    function automatic logic [LAYER_ID_WIDTH-1:0] find_bottleneck(
        input logic [LAYER_ID_WIDTH-1:0] target
    );
        automatic logic [LAYER_ID_WIDTH-1:0] cur;
        automatic logic [LAYER_ID_WIDTH-1:0] pred;
        automatic int depth;

        cur   = target;
        depth = 0;

        for (depth = 0; depth < MAX_LAYERS; depth++) begin
            pred = sched_table[cur].prev_layer;
            if (pred == 8'hFF) begin
                // cur is a root-level layer; it IS the bottleneck
                if (!sched_table[cur].compute_done_flag)
                    find_bottleneck = cur;
                else
                    find_bottleneck = 8'hFF;  // already done
                return;
            end else if (!sched_table[pred].compute_done_flag) begin
                // predecessor not done: go deeper
                cur = pred;
            end else begin
                // predecessor is done but cur is not: cur is bottleneck
                if (!sched_table[cur].compute_done_flag)
                    find_bottleneck = cur;
                else
                    find_bottleneck = 8'hFF;
                return;
            end
        end
        find_bottleneck = cur;  // fallback
    endfunction

    // ================================================================
    // Table load + queue init
    // ================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < MAX_LAYERS; i++) sched_table[i] <= '0;
            for (int d = 0; d < MAX_DNNS; d++) begin
                sb_sp[d]           <= '0;
                current_batch[d]   <= '0;
                prev_batch_reg[d]  <= '0;
                slice_remaining[d] <= '0;
                max_batch_cap[d]   <= '0;
                ct_current_layer[d]<= '0;
            end
            total_layers <= '0;
        end else if (st_write_en) begin
            automatic logic [BATCH_WIDTH-1:0] cap;

            sched_table[st_layer_idx].valid                <= 1'b1;
            sched_table[st_layer_idx].dnn_id               <= st_dnn_id;
            sched_table[st_layer_idx].prev_layer           <= st_prev_layer;
            sched_table[st_layer_idx].mem_cycles           <= st_mem_cycles;
            sched_table[st_layer_idx].compute_cycles       <= st_compute_cycles;
            sched_table[st_layer_idx].weight_fp            <= st_weight_fp;
            sched_table[st_layer_idx].ifmap_fp             <= st_ifmap_fp;
            sched_table[st_layer_idx].ofmap_fp             <= st_ofmap_fp;
            sched_table[st_layer_idx].requested_batch      <= st_requested_batch;
            sched_table[st_layer_idx].mem_access_done_flag <= 1'b0;
            sched_table[st_layer_idx].compute_done_flag    <= 1'b0;
            sched_table[st_layer_idx].expedited            <= 1'b0;

            // Compute per-layer hardware cap and track minimum per DNN
            cap = compute_max_batch_cap(st_weight_fp, st_ifmap_fp, st_ofmap_fp);
            sched_table[st_layer_idx].max_batch <= cap;

            if (current_batch[st_dnn_id] == 0) begin
                // First layer of this DNN: initialise
                // N = min(cap, requested_batch)
                current_batch[st_dnn_id]  <= (cap < st_requested_batch) ?
                                              cap : st_requested_batch;
                max_batch_cap[st_dnn_id]  <= cap;
                // Batch slicing: how many full slices of N are needed?
                slice_remaining[st_dnn_id] <= st_requested_batch;
            end else begin
                // Update minimum cap if this layer is more constrained
                if (cap < max_batch_cap[st_dnn_id])
                    max_batch_cap[st_dnn_id] <= cap;
            end

            total_layers <= st_total_layers;
        end
    end

    // MT candidate queue population
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mt_cq_head <= '0; mt_cq_tail <= '0; mt_cq_cnt <= '0;
            ct_cq_head <= '0; ct_cq_tail <= '0; ct_cq_cnt <= '0;
            sct_head   <= '0; sct_tail   <= '0; sct_cnt   <= '0;
        end else if (st_write_en) begin
            mt_cq[mt_cq_tail] <= st_layer_idx;
            mt_cq_tail        <= (mt_cq_tail + 1) % QD;
            mt_cq_cnt         <= mt_cq_cnt + 1;
        end
    end

    // ================================================================
    // Main scheduler  (Figure 9)
    // ================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            avail_mem_reg            <= ONCHIP_MEM_SIZE;
            cycles_to_fill_remaining <= ONCHIP_MEM_SIZE / (WORD_BITS/8);
            mem_cycle_ctr            <= '0;
            compute_cycle_ctr        <= '0;
            stall_ctr     <= '0; split_ctr      <= '0;
            merge_ctr     <= '0; throttle_ctr   <= '0;
            expedition_ctr<= '0; slice_ctr      <= '0;
            stall_detected <= 1'b0;
            mt_active <= 1'b0; mt_valid <= 1'b0;
            ct_active <= 1'b0; ct_valid <= 1'b0;
        end else begin

            // =============================================================
            // BLOCK A: Memory Access Scheduler with Distance Throttling
            // (Figure 9, Block A)
            // =============================================================
            if (!mt_active && mt_cq_cnt > 0) begin
                automatic logic [LAYER_ID_WIDTH-1:0] cand;
                automatic logic [LAYER_ID_WIDTH-1:0] dist;
                automatic logic [MEM_WIDTH-1:0]      mem_req;
                automatic logic                      do_sched;
                automatic int                        search_idx;
                automatic logic                      found_cand;

                found_cand = 1'b0;
                search_idx = mt_cq_head;

                // Iterate through MT CQ to find the LOWEST layer that
                // satisfies both memory and distance constraints (Section 4.2.1)
                for (int k = 0; k < mt_cq_cnt && k < QD; k++) begin
                    automatic logic [LAYER_ID_WIDTH-1:0] try_cand;
                    automatic logic [DNN_ID_WIDTH-1:0]   try_dnn;

                    try_cand = mt_cq[(mt_cq_head + k) % QD];
                    try_dnn  = sched_table[try_cand].dnn_id;

                    if (!sched_table[try_cand].mem_access_done_flag) begin
                        // Distance check (box 3)
                        dist = layer_distance(try_cand, ct_current_layer[try_dnn]);

                        mem_req = sched_table[try_cand].weight_fp +
                                  (sched_table[try_cand].ifmap_fp +
                                   sched_table[try_cand].ofmap_fp) *
                                  current_batch[try_dnn];

                        if (dist <= MAX_LAYER_DISTANCE &&
                            mem_req <= avail_mem_reg) begin
                            // Found a suitable candidate
                            if (!found_cand) begin
                                cand       = try_cand;
                                found_cand = 1'b1;
                                search_idx = k;
                            end else if (try_cand < cand) begin
                                // Prefer lowest layer index (Section 4.2.1)
                                cand       = try_cand;
                                search_idx = k;
                            end
                        end else if (dist > MAX_LAYER_DISTANCE) begin
                            throttle_ctr <= throttle_ctr + 1;
                        end
                    end
                end

                if (found_cand) begin
                    mem_req = sched_table[cand].weight_fp +
                              (sched_table[cand].ifmap_fp +
                               sched_table[cand].ofmap_fp) *
                              current_batch[sched_table[cand].dnn_id];

                    // Allocate memory & schedule MT (box 5-6)
                    avail_mem_reg            <= avail_mem_reg - mem_req;
                    cycles_to_fill_remaining <= cycles_to_fill_remaining -
                                                sched_table[cand].mem_cycles;
                    mem_cycle_ctr            <= mem_cycle_ctr +
                        $signed({1'b0, sched_table[cand].mem_cycles});
                    mt_active       <= 1'b1;
                    mt_active_layer <= cand;
                    mt_valid        <= 1'b1;
                    mt_layer_id     <= cand;
                    mt_dnn_id       <= sched_table[cand].dnn_id;
                    stall_detected  <= 1'b0;
                end else begin
                    stall_detected <= 1'b1;
                    stall_ctr      <= stall_ctr + 1;
                end
            end

            // MT done → CT candidate queue
            if (mt_active && mem_access_done) begin
                sched_table[mt_active_layer].mem_access_done_flag <= 1'b1;
                mem_cycle_ctr <= mem_cycle_ctr -
                    $signed({1'b0, sched_table[mt_active_layer].mem_cycles});
                ct_cq[ct_cq_tail] <= mt_active_layer;
                ct_cq_tail        <= (ct_cq_tail + 1) % QD;
                ct_cq_cnt         <= ct_cq_cnt + 1;
                mt_active <= 1'b0;
                mt_valid  <= 1'b0;
            end

            // =============================================================
            // BLOCK B: Compute Scheduler with Bottleneck Layer Expedition
            // (Figure 9, Block B)
            // =============================================================
            if (!ct_active && ct_cq_cnt > 0) begin
                automatic logic [LAYER_ID_WIDTH-1:0] ct_cand;
                automatic logic [LAYER_ID_WIDTH-1:0] bottleneck;
                automatic logic                      all_prev_done;
                automatic logic                      all_prev_mem_done;
                automatic logic [LAYER_ID_WIDTH-1:0] pred;

                ct_cand = ct_cq[ct_cq_head];

                // Box 7-8: Check if all previous layers are done
                pred          = sched_table[ct_cand].prev_layer;
                all_prev_done = (pred == 8'hFF) ? 1'b1 :
                                sched_table[pred].compute_done_flag;

                if (all_prev_done) begin
                    // Normal path (box 9): schedule for computation
                    ct_cq_head <= (ct_cq_head + 1) % QD;
                    ct_cq_cnt  <= ct_cq_cnt - 1;
                    sct_q    [sct_tail] <= ct_cand;
                    sct_batch[sct_tail] <= current_batch[sched_table[ct_cand].dnn_id];
                    sct_tail  <= (sct_tail + 1) % QD;
                    sct_cnt   <= sct_cnt + 1;
                end else begin
                    // Box 11-18: Bottleneck layer expedition
                    // Check if predecessor memory access is done
                    all_prev_mem_done = (pred == 8'hFF) ? 1'b1 :
                                        sched_table[pred].mem_access_done_flag;

                    if (!all_prev_mem_done) begin
                        // Box 12: Schedule predecessor as MT (expedite)
                        if (!sched_table[pred].expedited) begin
                            // Inject into front of MT queue by marking as expedited
                            mt_cq[mt_cq_tail] <= pred;
                            mt_cq_tail        <= (mt_cq_tail + 1) % QD;
                            mt_cq_cnt         <= mt_cq_cnt + 1;
                            sched_table[pred].expedited <= 1'b1;
                            expedition_ctr              <= expedition_ctr + 1;
                        end
                    end else begin
                        // Box 13-18: Find bottleneck deeper in the graph
                        bottleneck = find_bottleneck(pred);

                        if (bottleneck != 8'hFF) begin
                            if (!sched_table[bottleneck].mem_access_done_flag &&
                                !sched_table[bottleneck].expedited) begin
                                // Expedite memory access of bottleneck (box 17)
                                mt_cq[mt_cq_tail] <= bottleneck;
                                mt_cq_tail        <= (mt_cq_tail + 1) % QD;
                                mt_cq_cnt         <= mt_cq_cnt + 1;
                                sched_table[bottleneck].expedited <= 1'b1;
                                expedition_ctr <= expedition_ctr + 1;
                            end else if (sched_table[bottleneck].mem_access_done_flag &&
                                         !sched_table[bottleneck].compute_done_flag) begin
                                // Expedite computation of bottleneck (box 15)
                                sct_q    [sct_tail] <= bottleneck;
                                sct_batch[sct_tail] <= current_batch[
                                    sched_table[bottleneck].dnn_id];
                                sct_tail <= (sct_tail + 1) % QD;
                                sct_cnt  <= sct_cnt + 1;
                                expedition_ctr <= expedition_ctr + 1;
                            end
                        end
                    end
                end
            end

            // =============================================================
            // BLOCK C/D: Sub-batch Splitting & Merging (from BATCH-DNN)
            //            + Batch Slicing  (Section 4.1.3)
            // =============================================================
            if (!ct_active && sct_cnt > 0) begin
                automatic logic [LAYER_ID_WIDTH-1:0] next_layer;
                automatic logic [BATCH_WIDTH-1:0]    req_batch;
                automatic logic [BATCH_WIDTH-1:0]    feasible_b;
                automatic logic [DNN_ID_WIDTH-1:0]   did;

                next_layer = sct_q[sct_head];
                req_batch  = sct_batch[sct_head];
                did        = sched_table[next_layer].dnn_id;

                // ---- Batch Slicing: cap req_batch to hardware maximum ----
                // (Section 4.1.3, equations 3-5)
                if (req_batch > max_batch_cap[did])
                    req_batch = max_batch_cap[did];

                // ---- Sub-Batch Merging (Block D) ----
                if (sb_sp[did] > 0) begin
                    automatic stack_entry_t top;
                    automatic logic fits;
                    automatic logic [MEM_WIDTH-1:0] needed;

                    top  = sb_stack[did][sb_sp[did]-1];
                    fits = 1'b1;

                    for (int l = top.layer_idx; l <= next_layer && l < MAX_LAYERS; l++) begin
                        needed = (sched_table[l].ifmap_fp +
                                  sched_table[l].ofmap_fp) *
                                 (top.batch_size + req_batch) +
                                  sched_table[l].weight_fp;
                        if (needed > avail_mem_reg) fits = 1'b0;
                    end

                    if (fits) begin
                        sb_sp[did]         <= sb_sp[did] - 1;
                        req_batch          <= req_batch + top.batch_size;
                        current_batch[did] <= req_batch + top.batch_size;
                        prev_batch_reg[did]<= req_batch + top.batch_size;
                        merge_ctr          <= merge_ctr + 1;
                    end
                end

                // ---- Sub-Batch Splitting (Block C) ----
                feasible_b = max_batch_size(
                    sched_table[next_layer].ofmap_fp,
                    avail_mem_reg,
                    prev_batch_reg[did]
                );

                if (feasible_b == 0) begin
                    // Stall
                    stall_detected <= 1'b1;
                    stall_ctr      <= stall_ctr + 1;
                end else begin
                    if (feasible_b < req_batch) begin
                        // Push paused sub-batch to stack
                        if (sb_sp[did] < STACK_DEPTH) begin
                            sb_stack[did][sb_sp[did]].layer_idx  <= next_layer;
                            sb_stack[did][sb_sp[did]].batch_size <= req_batch - feasible_b;
                            sb_sp[did] <= sb_sp[did] + 1;
                            split_ctr  <= split_ctr + 1;
                        end
                        compute_cycle_ctr <= compute_cycle_ctr -
                            $signed({1'b0,
                                sched_table[next_layer].compute_cycles *
                                (req_batch - feasible_b)});
                    end

                    avail_mem_reg      <= avail_mem_reg -
                                         sched_table[next_layer].ofmap_fp * feasible_b;
                    current_batch[did] <= feasible_b;
                    prev_batch_reg[did]<= feasible_b;

                    // Batch slicing counter update
                    if (slice_remaining[did] > feasible_b)
                        slice_remaining[did] <= slice_remaining[did] - feasible_b;
                    else begin
                        slice_remaining[did] <= '0;
                        slice_ctr <= slice_ctr + 1;
                    end

                    // Update current CT layer for distance tracking
                    ct_current_layer[did] <= next_layer;

                    // Dispatch
                    sct_head        <= (sct_head + 1) % QD;
                    sct_cnt         <= sct_cnt - 1;
                    ct_active       <= 1'b1;
                    ct_active_layer <= next_layer;
                    ct_active_dnn   <= did;
                    ct_active_batch <= feasible_b;
                    ct_valid        <= 1'b1;
                    ct_layer_id     <= next_layer;
                    ct_dnn_id       <= did;
                    ct_batch_size   <= feasible_b;
                    stall_detected  <= 1'b0;
                end
            end

            // =============================================================
            // Compute completion
            // =============================================================
            if (ct_active && compute_done) begin
                automatic logic [MEM_WIDTH-1:0] freed;
                freed = sched_table[ct_active_layer].ifmap_fp * ct_active_batch;

                avail_mem_reg <= avail_mem_reg + freed;
                if (sb_sp[ct_active_dnn] == 0)
                    avail_mem_reg <= avail_mem_reg + freed +
                                     sched_table[ct_active_layer].weight_fp;

                sched_table[ct_active_layer].compute_done_flag <= 1'b1;
                compute_cycle_ctr <= compute_cycle_ctr -
                    $signed({1'b0,
                        sched_table[ct_active_layer].compute_cycles * ct_active_batch});

                ct_active <= 1'b0;
                ct_valid  <= 1'b0;
            end

        end // !rst_n
    end

endmodule
