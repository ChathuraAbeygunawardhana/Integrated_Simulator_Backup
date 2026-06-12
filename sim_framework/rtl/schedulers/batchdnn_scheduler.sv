// ============================================================================
// BATCH-DNN Scheduler
// Based on: "BATCH-DNN: Adaptive and Dynamic Batching for Multi-DNN Accelerators"
//           (Ranawaka & Stenstrom, EuroPar 2025)
//
// Implements on top of AI-MT baseline:
//   C) Sub-Batch Splitting  – Figure 7 Block C (Section 3.4.2)
//   D) Sub-Batch Merging    – Figure 7 Block D (Section 3.4.2)
//
// Key design choices:
//   - Each DNN workload has a dedicated sub-batching STACK (LIFO)
//   - Batch size is reduced layer-by-layer when on-chip memory is tight
//   - Paused sub-batches are pushed to the stack; resumed and merged when
//     memory frees up
//   - Scheduling table adds a "footprint" field (weight, IFMAP, OFMAP per
//     unit batch) – used to compute maximum achievable batch size at runtime
//   - Fixed AI-MT memory/compute scheduler is reused for MT scheduling;
//     sub-batching modifies the CT dispatch path only
// ============================================================================

`timescale 1ns/1ps

module batchdnn_scheduler #(
    parameter MAX_LAYERS      = 32,
    parameter MAX_DNNS        = 4,
    parameter LAYER_ID_WIDTH  = 8,
    parameter DNN_ID_WIDTH    = 4,
    parameter BATCH_WIDTH     = 8,    // max batch size representable
    parameter ONCHIP_MEM_SIZE = 44*1024*1024,
    parameter WORD_BITS       = 16,   // each memory word is 2 bytes.
    parameter COMPUTE_BAL_THRESH = 1000,  // balance threshold for preferring compute over memory (cycles)
    parameter STACK_DEPTH     = 16,   // the maximum number of paused sub-batches that can be stored for each DNN.
    parameter CYCLE_WIDTH     = 32,
    parameter MEM_WIDTH       = 32
)(
    input  logic clk,
    input  logic rst_n,

    // ----------------------------------------------------------------
    // Scheduling-table load  (same interface as AI-MT)
    // ----------------------------------------------------------------
    input  logic                       st_write_en,
    input  logic [LAYER_ID_WIDTH-1:0]  st_layer_idx,
    input  logic [DNN_ID_WIDTH-1:0]    st_dnn_id,
    input  logic [LAYER_ID_WIDTH-1:0]  st_prev_layer,        // previous-layer dependency field for the loaded layer.
    input  logic [CYCLE_WIDTH-1:0]     st_mem_cycles,        // how many memory cycles are needed to load this layer’s data for one unit batch.
    input  logic [CYCLE_WIDTH-1:0]     st_compute_cycles,    // how many compute cycles are needed for one unit batch.
    input  logic [MEM_WIDTH-1:0]       st_weight_fp,         // how many bytes of weight data the layer needs in on-chip memory.
    input  logic [MEM_WIDTH-1:0]       st_ifmap_fp,          // how many bytes of input feature map data this layer needs in on-chip memory.
    input  logic [MEM_WIDTH-1:0]       st_ofmap_fp,          // how many bytes of output feature map data this layer needs in on-chip memory.
    input  logic [BATCH_WIDTH-1:0]     st_initial_batch,     // requested batch size for that layer when the scheduling table is loaded
    input  logic [LAYER_ID_WIDTH-1:0]  st_total_layers,

    // ----------------------------------------------------------------
    // Runtime control
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
    output logic [BATCH_WIDTH-1:0]    ct_batch_size,   // actual batch for this CT

    // ----------------------------------------------------------------
    // Status / metrics
    // ----------------------------------------------------------------
    output logic [CYCLE_WIDTH-1:0]    available_memory,   // how much on-chip memory is currently free
    output logic                      stall_detected,
    output logic [CYCLE_WIDTH-1:0]    total_stall_cycles,  // how many cycles the scheduler spent stalled
    output logic [CYCLE_WIDTH-1:0]    total_split_events,
    output logic [CYCLE_WIDTH-1:0]    total_merge_events
);

    // ================================================================
    // Scheduling-table entry  (AI-MT base + footprint field)
    // ================================================================
    typedef struct packed {
        logic                      valid;                //flag that indicates whether the scheduling-table entry contains valid layer data.
        logic [DNN_ID_WIDTH-1:0]   dnn_id;
        logic [LAYER_ID_WIDTH-1:0] prev_layer;
        logic [CYCLE_WIDTH-1:0]    mem_cycles;
        logic [CYCLE_WIDTH-1:0]    compute_cycles;       // per-unit-batch
        logic [MEM_WIDTH-1:0]      weight_fp;            // per-unit-batch
        logic [MEM_WIDTH-1:0]      ifmap_fp;
        logic [MEM_WIDTH-1:0]      ofmap_fp;
        logic [BATCH_WIDTH-1:0]    initial_batch;        //original batch size requested for that layer
        logic                      mem_access_done_flag;
        logic                      compute_done_flag;
    } sched_entry_t;

    sched_entry_t sched_table [MAX_LAYERS-1:0];
    logic [LAYER_ID_WIDTH-1:0] total_layers;

    // ================================================================
    // Sub-batching stack  (one per DNN workload, LIFO)
    // Each entry: {layer_index, sub_batch_size}
    // ================================================================
    typedef struct packed {   // one paused sub-batch entry in the per-DNN stack.
        logic [LAYER_ID_WIDTH-1:0] layer_idx;        // which layer the deferred sub-batch belongs to
        logic [BATCH_WIDTH-1:0]    batch_size;       //the size of a deferred sub-batch
    } stack_entry_t;

    stack_entry_t sb_stack [MAX_DNNS-1:0][STACK_DEPTH-1:0];   // sb_stack[d][s] , d = DNN ID, s = stack position within that DNN’s stack
    logic [$clog2(STACK_DEPTH):0] sb_sp [MAX_DNNS-1:0];  // stack pointer array for the sub-batch stacks - sb_sp[0] = A 4-bit register (e.g., 4'b0000) acting as the stack pointer for DNN 0.

    // Current batch size being used per DNN
    logic [BATCH_WIDTH-1:0] current_batch [MAX_DNNS-1:0];
    // Previous layer's batch size (constraint for splitting)
    logic [BATCH_WIDTH-1:0] prev_batch    [MAX_DNNS-1:0];

    // ================================================================
    // Queues  (same structure as AI-MT)
    // ================================================================
    localparam QD = MAX_LAYERS;

    logic [LAYER_ID_WIDTH-1:0] mt_cq       [QD-1:0];
    logic [$clog2(QD):0]        mt_cq_head, mt_cq_tail, mt_cq_cnt;

    logic [LAYER_ID_WIDTH-1:0] ct_cq       [QD-1:0];
    logic [$clog2(QD):0]        ct_cq_head, ct_cq_tail, ct_cq_cnt;

    logic [LAYER_ID_WIDTH-1:0] sct_q       [QD-1:0];
    logic [BATCH_WIDTH-1:0]    sct_batch   [QD-1:0];  // batch size per entry
    logic [$clog2(QD):0]        sct_head, sct_tail, sct_cnt;

    // ================================================================
    // Balance counters
    // ================================================================
    logic signed [CYCLE_WIDTH:0] mem_cycle_ctr;         // current pending memory-cycle load in the MT scheduler
    logic signed [CYCLE_WIDTH:0] compute_cycle_ctr;     // current pending compute-cycle load in the CT scheduler
    logic [CYCLE_WIDTH-1:0]      cycles_to_fill_remaining;  // how many cycles of memory load can still be scheduled before on-chip memory is full

    // ================================================================
    // Available memory
    // ================================================================
    logic [CYCLE_WIDTH-1:0] avail_mem_reg;           // current available on-chip memory budget
    assign available_memory = avail_mem_reg;

    // ================================================================
    // Active task tracking
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
    assign total_stall_cycles  = stall_ctr;
    assign total_split_events  = split_ctr;
    assign total_merge_events  = merge_ctr;

    // ================================================================
    // Helper function: compute maximum feasible batch size
    // Constraint 1: ofmap_fp * batch <= available_memory
    // Constraint 2: batch <= previous layer's batch
    // (Section 3.4.2, box 17)
    // ================================================================
    // calculates the maximum batch size that can be scheduled or processed at a given moment
    function automatic logic [BATCH_WIDTH-1:0] max_batch_size(   // automatic : function's internal variables are dynamically allocated each time it is called.
        input logic [MEM_WIDTH-1:0]   ofmap_fp_unit,     // How much memory (Output Feature Map) is required for one unit of the batch.
        input logic [CYCLE_WIDTH-1:0] avail_mem,         // The currently available on-chip memory.
        input logic [BATCH_WIDTH-1:0] prev_b           // The batch size of the previous layer.
    );
        automatic logic [BATCH_WIDTH-1:0] b;
        if (ofmap_fp_unit == 0) begin    // Division by Zero Protection, could happen for certain layer types that don't produce standard OFMAPs
            max_batch_size = prev_b;
        end else begin
            // integer division
            b = avail_mem / ofmap_fp_unit;   // calculates how many batch elements can actually fit into the available memory
            if (b > prev_b) b = prev_b;    // ensures the batch size doesn't exceed the requested limit
            max_batch_size = b;
        end
    endfunction

    // ================================================================
    // Reset / table load
    // ================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < MAX_LAYERS; i++) sched_table[i] <= '0;
            for (int d = 0; d < MAX_DNNS; d++) begin
                sb_sp[d]           <= '0;
                current_batch[d]   <= '0;
                prev_batch[d]      <= '0;
            end
            total_layers <= '0;
        end else if (st_write_en) begin
            sched_table[st_layer_idx].valid                <= 1'b1;
            sched_table[st_layer_idx].dnn_id               <= st_dnn_id;
            sched_table[st_layer_idx].prev_layer           <= st_prev_layer;
            sched_table[st_layer_idx].mem_cycles           <= st_mem_cycles;
            sched_table[st_layer_idx].compute_cycles       <= st_compute_cycles;
            sched_table[st_layer_idx].weight_fp            <= st_weight_fp;
            sched_table[st_layer_idx].ifmap_fp             <= st_ifmap_fp;
            sched_table[st_layer_idx].ofmap_fp             <= st_ofmap_fp;
            sched_table[st_layer_idx].initial_batch        <= st_initial_batch;
            sched_table[st_layer_idx].mem_access_done_flag <= 1'b0;
            sched_table[st_layer_idx].compute_done_flag    <= 1'b0;
            total_layers <= st_total_layers;

            // initialise current_batch for this DNN if first layer
            if (current_batch[st_dnn_id] == 0)
                current_batch[st_dnn_id] <= st_initial_batch;
        end
    end

    // ================================================================
    // MT candidate queue population on layer load
    // ================================================================
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
    // Main scheduler state machine
    // ================================================================
    // memory scheduling, compute scheduling, splitting, merging, and memory release
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            avail_mem_reg            <= ONCHIP_MEM_SIZE;
            cycles_to_fill_remaining <= ONCHIP_MEM_SIZE / (WORD_BITS/8);
            mem_cycle_ctr            <= '0;
            compute_cycle_ctr        <= '0;
            stall_ctr                <= '0;
            split_ctr                <= '0;
            merge_ctr                <= '0;
            stall_detected           <= 1'b0;
            mt_active <= 1'b0; mt_valid  <= 1'b0;
            ct_active <= 1'b0; ct_valid  <= 1'b0;
        end else begin

            // =========================================================
            // BLOCK A: Memory Access Scheduler (identical to AI-MT)
            // =========================================================
            if (!mt_active && mt_cq_cnt > 0) begin
                automatic logic [LAYER_ID_WIDTH-1:0] cand;
                automatic logic [MEM_WIDTH-1:0]      mem_req;
                automatic logic                      do_sched;   // 0 → don’t schedule, 1 → schedule this task 

                cand     = mt_cq[mt_cq_head];
                do_sched = 1'b0;

                if (sched_table[cand].valid &&
                    !sched_table[cand].mem_access_done_flag) begin

                    // scale footprints by batch size
                    mem_req = sched_table[cand].weight_fp +
                              (sched_table[cand].ifmap_fp  +
                               sched_table[cand].ofmap_fp) *
                              current_batch[sched_table[cand].dnn_id];  // Only loaded once → no multiplication, Each input needs its own memory → multiply by batch size

                    // Should I schedule this memory task now, or wait?
                    if (sched_table[cand].mem_cycles <= cycles_to_fill_remaining &&
                        mem_req <= avail_mem_reg) begin
                        if (compute_cycle_ctr > COMPUTE_BAL_THRESH ||                      // Compute side is busy → better feed it with memory
                            sched_table[cand].mem_cycles <= compute_cycle_ctr) begin       // Memory won’t overload the system
                            do_sched = 1'b1;
                        end
                    end

                    if (do_sched) begin
                        mt_cq_head               <= (mt_cq_head + 1) % QD;
                        mt_cq_cnt                <= mt_cq_cnt - 1;
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
                    end else begin
                        stall_detected <= 1'b1;
                        stall_ctr      <= stall_ctr + 1;
                    end
                end
            end

            // Memory access done → CT candidate queue
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

            // =========================================================
            // BLOCK B: Compute Scheduler → Scheduled CT Queue
            // =========================================================
            if (!ct_active && ct_cq_cnt > 0) begin
                automatic logic [LAYER_ID_WIDTH-1:0] ct_cand;
                automatic logic prev_done;

                ct_cand   = ct_cq[ct_cq_head];
                prev_done = (sched_table[ct_cand].prev_layer == 8'hFF) ?
                             1'b1 :
                             sched_table[sched_table[ct_cand].prev_layer].compute_done_flag;

                if (prev_done) begin
                    ct_cq_head <= (ct_cq_head + 1) % QD;
                    ct_cq_cnt  <= ct_cq_cnt - 1;

                    sct_q[sct_tail]     <= ct_cand;
                    sct_batch[sct_tail] <= current_batch[sched_table[ct_cand].dnn_id];
                    sct_tail            <= (sct_tail + 1) % QD;
                    sct_cnt             <= sct_cnt + 1;

                    compute_cycle_ctr <= compute_cycle_ctr +
                        $signed({1'b0,
                            sched_table[ct_cand].compute_cycles *
                            current_batch[sched_table[ct_cand].dnn_id]});
                end
            end

            // =========================================================
            // BLOCK C/D: Sub-batch Splitting & Merging
            // Dispatch from Scheduled CT Queue (box 15)
            // =========================================================
            if (!ct_active && sct_cnt > 0) begin
                automatic logic [LAYER_ID_WIDTH-1:0] next_layer;
                automatic logic [BATCH_WIDTH-1:0]    req_batch;
                automatic logic [BATCH_WIDTH-1:0]    feasible_b;
                automatic logic [DNN_ID_WIDTH-1:0]   did;
                automatic logic                      merged;

                next_layer = sct_q[sct_head];
                req_batch  = sct_batch[sct_head];
                did        = sched_table[next_layer].dnn_id;
                merged     = 1'b0;

                // ------- Try Sub-Batch Merging first (box 16) -------
                if (sb_sp[did] > 0) begin
                    automatic stack_entry_t top_entry;
                    automatic logic [LAYER_ID_WIDTH-1:0] resume_layer;
                    automatic logic                      fits;
                    automatic logic [MEM_WIDTH-1:0]      needed;

                    top_entry    = sb_stack[did][sb_sp[did]-1];
                    resume_layer = top_entry.layer_idx;
                    fits         = 1'b1;

                    // Check all layers between resumed and current fit (box 16b-16d)
                    for (int l = resume_layer; l <= next_layer; l++) begin
                        needed = (sched_table[l].ifmap_fp +
                                  sched_table[l].ofmap_fp) *
                                 (top_entry.batch_size + req_batch) +
                                  sched_table[l].weight_fp;
                        if (needed > avail_mem_reg) fits = 1'b0;
                    end

                    if (fits) begin
                        // Pop stack, merge batch sizes (box 16f)
                        sb_sp[did]         <= sb_sp[did] - 1;
                        req_batch          <= req_batch + top_entry.batch_size;
                        current_batch[did] <= req_batch + top_entry.batch_size;
                        prev_batch[did]    <= req_batch + top_entry.batch_size;
                        merged             <= 1'b1;
                        merge_ctr          <= merge_ctr + 1;
                    end
                end

                // ------- Sub-Batch Splitting (box 17-24) -------
                feasible_b = max_batch_size(
                    sched_table[next_layer].ofmap_fp,
                    avail_mem_reg,
                    (merged ? req_batch : prev_batch[did])
                );

                if (feasible_b == 0) begin
                    // Even unit batch won't fit: stall (box 19b)
                    stall_detected <= 1'b1;
                    stall_ctr      <= stall_ctr + 1;
                end else begin
                    if (feasible_b < req_batch) begin
                        // Push paused sub-batch to stack (boxes 20-22)
                        if (sb_sp[did] < STACK_DEPTH) begin
                            sb_stack[did][sb_sp[did]].layer_idx  <= next_layer;
                            sb_stack[did][sb_sp[did]].batch_size <= req_batch - feasible_b;
                            sb_sp[did]   <= sb_sp[did] + 1;
                            split_ctr    <= split_ctr + 1;
                        end
                        // Adjust compute counter for paused sub-batch (box 23)
                        compute_cycle_ctr <= compute_cycle_ctr -
                            $signed({1'b0,
                                sched_table[next_layer].compute_cycles *
                                (req_batch - feasible_b)});
                    end

                    // Update memory register (box 24)
                    avail_mem_reg     <= avail_mem_reg -
                                        sched_table[next_layer].ofmap_fp * feasible_b;
                    current_batch[did] <= feasible_b;
                    prev_batch[did]    <= feasible_b;

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

            // =========================================================
            // Compute completion: release IFMAP memory (box 25)
            // =========================================================
            if (ct_active && compute_done) begin
                automatic logic [MEM_WIDTH-1:0] freed;
                freed = sched_table[ct_active_layer].ifmap_fp * ct_active_batch;

                // Release IFMAP of completed sub-batch
                avail_mem_reg <= avail_mem_reg + freed;
                // Check if all inferences in this batch completed (stack empty)
                if (sb_sp[ct_active_dnn] == 0) begin
                    // Release weight memory too
                    avail_mem_reg <= avail_mem_reg + freed +
                                     sched_table[ct_active_layer].weight_fp;
                end

                sched_table[ct_active_layer].compute_done_flag <= 1'b1;
                compute_cycle_ctr <= compute_cycle_ctr -
                    $signed({1'b0,
                        sched_table[ct_active_layer].compute_cycles * ct_active_batch});
                ct_active <= 1'b0;
                ct_valid  <= 1'b0;
            end

        end // else (not reset)
    end

endmodule
