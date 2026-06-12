// ============================================================================
// AI-MT Scheduler
// Based on: "A Multi-Neural Network Acceleration Architecture" (Baek et al., ISCA 2020)
//
// Key concepts implemented:
//   - Layer-granularity scheduling (not sub-layer like original AI-MT)
//   - Memory Access Tasks (MT) and Compute Tasks (CT) are overlapped
//   - MT Candidate Queue → memory access → CT Candidate Queue → Scheduled CT Queue
//   - Balance condition: schedule memory only when
//       (a) MT memory_cycles <= cycles_to_fill_remaining_memory, AND
//       (b) available_compute_cycles > COMPUTE_BALANCE_THRESHOLD, OR
//           memory_cycles <= available_compute_cycles
//   - FIFO dispatch for compute tasks (prevents starvation)
//   - Scheduling table holds per-layer static metadata compiled offline
// ============================================================================

`timescale 1ns/1ps

module aimt_scheduler #(
    // ---------- queue sizing ----------
    parameter MAX_LAYERS          = 32,   // total layers across all DNN workloads
    parameter MAX_DNNS            = 4,    // max concurrent DNN workloads
    parameter LAYER_ID_WIDTH      = 8,    // bits for layer index
    parameter DNN_ID_WIDTH        = 4,    // bits for DNN workload ID
    // ---------- memory / compute model ----------
    parameter ONCHIP_MEM_SIZE     = 44*1024*1024, // 44 MB in bytes (Table 1a)
    parameter WORD_BITS           = 16,            // 16-bit words (Table 1a)
    parameter COMPUTE_BAL_THRESH  = 1000,          // It sets a minimum amount of available scheduled compute work before starting another memory task.
    // ---------- data widths ----------
    parameter CYCLE_WIDTH         = 32,   // bits for cycle counts
    parameter MEM_WIDTH           = 32    // bits for memory footprint values (bytes)
)(
    input  logic clk,
    input  logic rst_n,

    // ----------------------------------------------------------------
    // Scheduling-table load interface  (written by compiler/host once)
    // ----------------------------------------------------------------
    input  logic                       st_write_en,        // scheduling-table write enable signal
    input  logic [LAYER_ID_WIDTH-1:0]  st_layer_idx,       // row index
    input  logic [DNN_ID_WIDTH-1:0]    st_dnn_id,
    input  logic [LAYER_ID_WIDTH-1:0]  st_prev_layer,      // predecessor layer ID for each layer entry.
    input  logic [CYCLE_WIDTH-1:0]     st_mem_cycles,      // It represents how many cycles the layer’s memory access task (MT) will take to load weights/data
    input  logic [CYCLE_WIDTH-1:0]     st_compute_cycles,  // cycles to compute layer
    input  logic [MEM_WIDTH-1:0]       st_weight_fp,       // weight footprint  (bytes)  , how many bytes of weight data the layer needs in on-chip memory.
    input  logic [MEM_WIDTH-1:0]       st_ifmap_fp,        // IFMAP  footprint  (bytes)  , how many bytes of input feature map data this layer needs in on-chip memory.
    input  logic [MEM_WIDTH-1:0]       st_ofmap_fp,        // OFMAP  footprint  (bytes)  , how many bytes of output feature map data this layer needs in on-chip memory.
    input  logic [LAYER_ID_WIDTH-1:0]  st_total_layers,    // total layers loaded  , It tells the scheduler how many layers the current workload has

    // ----------------------------------------------------------------
    // Runtime control
    // ----------------------------------------------------------------
    input  logic mem_access_done,   // pulsed when the active MT finishes
    input  logic compute_done,      // pulsed when the active CT finishes

    // ----------------------------------------------------------------
    // Scheduled outputs
    // ----------------------------------------------------------------
    output logic                      mt_valid,          // memory task ready
    output logic [LAYER_ID_WIDTH-1:0] mt_layer_id,       // layer to prefetch
    output logic [DNN_ID_WIDTH-1:0]   mt_dnn_id,

    output logic                      ct_valid,          // compute task ready
    output logic [LAYER_ID_WIDTH-1:0] ct_layer_id,       // layer to compute
    output logic [DNN_ID_WIDTH-1:0]   ct_dnn_id,

    // ----------------------------------------------------------------
    // Status
    // ----------------------------------------------------------------
    output logic [CYCLE_WIDTH-1:0]    available_memory,  // free on-chip bytes
    output logic                      stall_detected,  // pulsed when scheduler wanted to schedule but was blocked by balance conditions or memory limits
    output logic [CYCLE_WIDTH-1:0]    total_stall_cycles  // it measures how much time the scheduler spent waiting instead of issuing work.
);

    // ================================================================
    // Scheduling-table entry
    // ================================================================
    typedef struct packed {   //A struct (structure) groups multiple variables into one unit. packed means the variables are stored contiguously without padding bits.
        logic                     valid;
        logic [DNN_ID_WIDTH-1:0]  dnn_id;
        logic [LAYER_ID_WIDTH-1:0] prev_layer;   // 8'hFF = no predecessor
        logic [CYCLE_WIDTH-1:0]   mem_cycles;
        logic [CYCLE_WIDTH-1:0]   compute_cycles;
        logic [MEM_WIDTH-1:0]     weight_fp;
        logic [MEM_WIDTH-1:0]     ifmap_fp;
        logic [MEM_WIDTH-1:0]     ofmap_fp;
        logic                     mem_access_done_flag;  // indicates whether that layer’s memory access task (MT) has completed.
        logic                     compute_done_flag;
    } sched_entry_t;

    sched_entry_t sched_table [MAX_LAYERS-1:0];
    logic [LAYER_ID_WIDTH-1:0] total_layers;

    // ================================================================
    // Candidate queues  (simple arrays used as circular FIFOs)
    // ================================================================
    localparam QD = MAX_LAYERS;   //can't be overridden from outside th module , QD - Queue Depth

    // MT candidate queue  (layers waiting for memory load)
    logic [LAYER_ID_WIDTH-1:0] mt_cq       [QD-1:0];
    logic [$clog2(QD):0]        mt_cq_head, mt_cq_tail, mt_cq_cnt;

    // CT candidate queue  (layers ready for compute (memory done))
    logic [LAYER_ID_WIDTH-1:0] ct_cq       [QD-1:0];
    logic [$clog2(QD):0]        ct_cq_head, ct_cq_tail, ct_cq_cnt;

    // Scheduled CT queue (final ordered compute execution (FIFO))
    logic [LAYER_ID_WIDTH-1:0] sct_q       [QD-1:0];
    logic [$clog2(QD):0]        sct_head, sct_tail, sct_cnt;

    // ================================================================
    // Shared balance counters  (Section 3.4.1 – boxes 2,3,4,7,11,13)
    // ================================================================
    logic signed [CYCLE_WIDTH:0] mem_cycle_ctr;      // Tracks total memory work currently scheduled
    logic signed [CYCLE_WIDTH:0] compute_cycle_ctr;  // Tracks total compute work currently scheduled

    // cycles_to_fill = (available_memory / word_size) estimated as mem cycles proxy
    logic [CYCLE_WIDTH-1:0] cycles_to_fill_remaining;  // how much memory access work can still be started before on-chip memory is "full"

    // ================================================================
    // Available memory register
    // ================================================================
    logic [CYCLE_WIDTH-1:0] avail_mem_reg;
    assign available_memory = avail_mem_reg;

    // ================================================================
    // Active task tracking
    // ================================================================
    logic                     mt_active;   //Is the memory unit busy?
    logic [LAYER_ID_WIDTH-1:0] mt_active_layer; //Stores which layer is currently being loaded into memory
    logic                     ct_active;   //Is computation happening right now?
    logic [LAYER_ID_WIDTH-1:0] ct_active_layer;  //Stores which layer is currently being computed

    // stall counter
    logic [CYCLE_WIDTH-1:0] stall_ctr;   //How many cycles the scheduler was stalled
    assign total_stall_cycles = stall_ctr;

    // ================================================================
    // Scheduling-table load
    // ================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < MAX_LAYERS; i++) sched_table[i] <= '0;  //All entries in sched_table are cleared
            total_layers <= '0;
        end else if (st_write_en) begin
            sched_table[st_layer_idx].valid             <= 1'b1;
            sched_table[st_layer_idx].dnn_id            <= st_dnn_id;
            sched_table[st_layer_idx].prev_layer        <= st_prev_layer;
            sched_table[st_layer_idx].mem_cycles        <= st_mem_cycles;
            sched_table[st_layer_idx].compute_cycles    <= st_compute_cycles;
            sched_table[st_layer_idx].weight_fp         <= st_weight_fp;
            sched_table[st_layer_idx].ifmap_fp          <= st_ifmap_fp;
            sched_table[st_layer_idx].ofmap_fp          <= st_ofmap_fp;
            sched_table[st_layer_idx].mem_access_done_flag <= 1'b0;  //Memory NOT done yet
            sched_table[st_layer_idx].compute_done_flag    <= 1'b0;  //Compute NOT done yet
            total_layers <= st_total_layers;
        end
    end

    // ================================================================
    // Populate MT candidate queue on reset/load
    // All valid layers initially await memory access
    // ================================================================
    // It specifically initializes the queues and pushes new layers into the MT (Memory Task) queue.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mt_cq_head <= '0; mt_cq_tail <= '0; mt_cq_cnt <= '0;
            ct_cq_head <= '0; ct_cq_tail <= '0; ct_cq_cnt <= '0;
            sct_head   <= '0; sct_tail   <= '0; sct_cnt   <= '0;
        end else if (st_write_en) begin
            // Enqueue newly loaded layer into MT candidate queue
            mt_cq[mt_cq_tail] <= st_layer_idx;   //Store the layer ID at the current tail position
            mt_cq_tail         <= (mt_cq_tail + 1) % QD; //Move tail forward
            mt_cq_cnt          <= mt_cq_cnt + 1;
        end
    end

    // ================================================================
    // Available memory & cycles_to_fill initialisation
    // ================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            avail_mem_reg          <= ONCHIP_MEM_SIZE;
            cycles_to_fill_remaining <= ONCHIP_MEM_SIZE / (WORD_BITS/8);
            mem_cycle_ctr          <= '0;
            compute_cycle_ctr      <= '0;
            stall_ctr              <= '0;
            stall_detected         <= 1'b0;
        end else begin

            // -----------------------------------------------------------
            // A) Memory Access Scheduler  (Figure 7, Block A)
            // -----------------------------------------------------------
            if (!mt_active && mt_cq_cnt > 0) begin  //No memory task is running , MT queue is not empty
                automatic logic [LAYER_ID_WIDTH-1:0] cand;
                automatic logic schedule_mt;  // schedule_mt=1 when the current MT candidate meets the balance conditions and enough memory is available.
                automatic logic [MEM_WIDTH-1:0] mem_req;  // (weight_fp + ifmap_fp + ofmap_fp) total memory footprint required by the current candidate layer 

                cand        = mt_cq[mt_cq_head];   //Take the next layer from MT queue
                schedule_mt = 1'b0;

                if (sched_table[cand].valid &&
                    !sched_table[cand].mem_access_done_flag) begin  //Layer exists and memory access not done yet

                    mem_req = sched_table[cand].weight_fp +
                              sched_table[cand].ifmap_fp  +
                              sched_table[cand].ofmap_fp;  //Total memory required for this layer

                    // Decision box (2): mem_cycles <= cycles_to_fill_remaining
                    if (sched_table[cand].mem_cycles <= cycles_to_fill_remaining) begin
                        // Decision box (3): available_compute_cycles > threshold
                        if (compute_cycle_ctr > COMPUTE_BAL_THRESH) begin
                            schedule_mt = 1'b1;
                        // Decision box (4): mem_cycles <= available compute cycles
                        end else if (sched_table[cand].mem_cycles <=
                                     compute_cycle_ctr) begin
                            schedule_mt = 1'b1;
                        end
                    end

                    if (schedule_mt && mem_req <= avail_mem_reg) begin
                        // Pop from MT CQ (box 5)
                        mt_cq_head    <= (mt_cq_head + 1) % QD;  //when it reaches the last slot, it goes back to 
                        mt_cq_cnt     <= mt_cq_cnt - 1;
                        // Allocate memory (box 7,8)
                        avail_mem_reg          <= avail_mem_reg - mem_req;
                        cycles_to_fill_remaining <= cycles_to_fill_remaining -
                                                    sched_table[cand].mem_cycles;
                        mem_cycle_ctr          <= mem_cycle_ctr +
                                                  $signed({1'b0, sched_table[cand].mem_cycles});  //$signed(...) converts that wider value into a signed number before adding.
                        // Start MT
                        mt_active       <= 1'b1;
                        mt_active_layer <= cand;
                        mt_valid        <= 1'b1;
                        mt_layer_id     <= cand;
                        mt_dnn_id       <= sched_table[cand].dnn_id;
                    end else begin
                        // Stall
                        stall_detected <= 1'b1;
                        stall_ctr      <= stall_ctr + 1;
                    end
                end
            end

            // -----------------------------------------------------------
            // A) Memory access completion → move to CT candidate queue
            //    (box 9)
            // -----------------------------------------------------------
            if (mt_active && mem_access_done) begin
                sched_table[mt_active_layer].mem_access_done_flag <= 1'b1;
                // Update counters (box 7 decrement)
                mem_cycle_ctr <= mem_cycle_ctr -
                                 $signed({1'b0, sched_table[mt_active_layer].mem_cycles});
                // Enqueue into CT candidate queue
                ct_cq[ct_cq_tail] <= mt_active_layer;
                ct_cq_tail        <= (ct_cq_tail + 1) % QD;
                ct_cq_cnt         <= ct_cq_cnt + 1;
                mt_active <= 1'b0;
                mt_valid  <= 1'b0;
            end

            // -----------------------------------------------------------
            // B) Compute Scheduler  (Figure 7, Block B)
            // -----------------------------------------------------------
            if (!ct_active && ct_cq_cnt > 0) begin
                automatic logic [LAYER_ID_WIDTH-1:0] ct_cand;
                automatic logic schedule_ct;  // schedule_ct=1 when the current CT candidate meets the scheduling conditions
                automatic logic prev_done;  // indicates whether the predecessor layer's compute is done (dependency check)

                ct_cand     = ct_cq[ct_cq_head];
                schedule_ct = 1'b0;

                // Check predecessor completed (dependency)
                if (sched_table[ct_cand].prev_layer == 8'hFF)  //8'hFF is an 8-bit hexadecimal constant with all bits set to 1.
                    prev_done = 1'b1;
                else
                    prev_done = sched_table[sched_table[ct_cand].prev_layer].compute_done_flag;

                // Decision box (11): compute_cycles <= memory_cycles (avoid over-scheduling)
                if (prev_done &&
                    sched_table[ct_cand].compute_cycles <= mem_cycle_ctr) begin
                    schedule_ct = 1'b1;
                end else if (prev_done && ct_cq_cnt > 0) begin
                    // fallback: still schedule if nothing else is pending
                    schedule_ct = 1'b1;
                end

                if (schedule_ct) begin
                    ct_cq_head      <= (ct_cq_head + 1) % QD;
                    ct_cq_cnt       <= ct_cq_cnt - 1;
                    // Move to scheduled compute queue (box 14)
                    sct_q[sct_tail] <= ct_cand;
                    sct_tail        <= (sct_tail + 1) % QD;
                    sct_cnt         <= sct_cnt + 1;
                    compute_cycle_ctr <= compute_cycle_ctr +
                                        $signed({1'b0, sched_table[ct_cand].compute_cycles});
                end
            end

            // Dispatch first entry of Scheduled CT Queue
            if (!ct_active && sct_cnt > 0) begin
                ct_active       <= 1'b1;
                ct_active_layer <= sct_q[sct_head];
                ct_valid        <= 1'b1;
                ct_layer_id     <= sct_q[sct_head];
                ct_dnn_id       <= sched_table[sct_q[sct_head]].dnn_id;
                sct_head        <= (sct_head + 1) % QD;
                sct_cnt         <= sct_cnt - 1;
            end

            // -----------------------------------------------------------
            // Compute done: free memory, update done flag
            // -----------------------------------------------------------
            if (ct_active && compute_done) begin
                automatic logic [MEM_WIDTH-1:0] freed; //It holds the amount of on-chip memory that is released by the completed layer.
                freed = sched_table[ct_active_layer].weight_fp +
                        sched_table[ct_active_layer].ifmap_fp;   // OFMAP becomes next IFMAP
                avail_mem_reg <= avail_mem_reg + freed;
                cycles_to_fill_remaining <= cycles_to_fill_remaining +
                                            (freed / (WORD_BITS/8));
                sched_table[ct_active_layer].compute_done_flag <= 1'b1;
                compute_cycle_ctr <= compute_cycle_ctr -
                                     $signed({1'b0, sched_table[ct_active_layer].compute_cycles});
                ct_active     <= 1'b0;
                ct_valid      <= 1'b0;
                stall_detected <= 1'b0;
            end
        end
    end

endmodule
