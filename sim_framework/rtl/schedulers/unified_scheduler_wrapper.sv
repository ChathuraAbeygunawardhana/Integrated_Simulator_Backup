//=============================================================================
// Module: unified_scheduler_wrapper
// Description: Top-level scheduler selector for multi_dnn_top. A 5-bit
//              scheduler_select picks among 14 schedulers:
//
//   0..6  = basic task    (FIFO/LIFO/SJF/RR/Pri/EDF/LRU) -- multi_scheduler_wrapper
//   7..10 = adv. task     (SRTF/HRRN/MLQ/MLFQ)           -- multi_scheduler_wrapper
//   11    = AI-MT         (DNN-aware)                    -- dnn_scheduler_wrapper
//   12    = BatchDNN      (DNN-aware)                    -- dnn_scheduler_wrapper
//   13    = BatchDNN++    (DNN-aware)                    -- dnn_scheduler_wrapper
//
// Output is a tagged-union sched_output_t (defined in sim_framework_pkg).
// `is_dnn_aware` lets the dispatch FSM in multi_dnn_top route memory and
// compute tasks separately for the AI-MT family.
//=============================================================================

module unified_scheduler_wrapper
    import sim_framework_pkg::*;
#(
    parameter int MAX_TASKS        = 16,
    parameter int TASK_ID_WIDTH    = 8,
    parameter int BURST_TIME_WIDTH = 16,
    parameter int PRIORITY_WIDTH   = 4,
    parameter int DEADLINE_WIDTH   = 32,
    parameter int TIME_QUANTUM     = 10,
    parameter int MAX_LAYERS       = 32,
    parameter int MAX_DNNS         = 4,
    parameter int LAYER_ID_WIDTH   = 8,
    parameter int DNN_ID_WIDTH     = 4,
    parameter int BATCH_WIDTH      = 8,
    parameter int CYCLE_WIDTH      = 32,
    parameter int MEM_WIDTH        = 32
) (
    input  logic clk,
    input  logic rst_n,

    input  logic [4:0] scheduler_select,

    //--- Basic-task interface (used when scheduler_select 0..10)
    input  logic                         task_valid,
    input  logic [TASK_ID_WIDTH-1:0]     task_id,
    input  logic [BURST_TIME_WIDTH-1:0]  burst_time,
    input  logic [PRIORITY_WIDTH-1:0]    task_priority,
    input  logic [DEADLINE_WIDTH-1:0]    deadline,
    input  logic                         task_tick,
    input  logic                         task_complete,
    output logic                         task_ready,

    //--- DNN-aware interface (used when scheduler_select 11..13)
    input  logic                       st_write_en,
    input  logic [LAYER_ID_WIDTH-1:0]  st_layer_idx,
    input  logic [DNN_ID_WIDTH-1:0]    st_dnn_id,
    input  logic [LAYER_ID_WIDTH-1:0]  st_prev_layer,
    input  logic [CYCLE_WIDTH-1:0]     st_mem_cycles,
    input  logic [CYCLE_WIDTH-1:0]     st_compute_cycles,
    input  logic [MEM_WIDTH-1:0]       st_weight_fp,
    input  logic [MEM_WIDTH-1:0]       st_ifmap_fp,
    input  logic [MEM_WIDTH-1:0]       st_ofmap_fp,
    input  logic [BATCH_WIDTH-1:0]     st_batch,
    input  logic [LAYER_ID_WIDTH-1:0]  st_total_layers,
    input  logic                       mem_access_done,
    input  logic                       compute_done,

    //--- Tagged-union output
    output sched_output_t              sched_out,
    output logic [BATCH_WIDTH-1:0]     ct_batch_size,

    //--- Stats (basic only, for now)
    output logic [31:0] basic_total_tasks_processed,
    output logic [31:0] basic_total_wait_time,
    output logic [31:0] basic_total_turnaround_time,
    output logic        dnn_stall_detected
);

    //------------------------------------------------------------------------
    // Basic-task wrapper (11 algorithms)
    //------------------------------------------------------------------------
    logic                         b_task_ready;
    logic                         b_sched_valid;
    logic [TASK_ID_WIDTH-1:0]     b_sched_id;
    logic [BURST_TIME_WIDTH-1:0]  b_sched_burst;

    multi_scheduler_wrapper #(
        .MAX_TASKS       (MAX_TASKS),
        .TASK_ID_WIDTH   (TASK_ID_WIDTH),
        .BURST_TIME_WIDTH(BURST_TIME_WIDTH),
        .PRIORITY_WIDTH  (PRIORITY_WIDTH),
        .DEADLINE_WIDTH  (DEADLINE_WIDTH),
        .TIME_QUANTUM    (TIME_QUANTUM)
    ) u_basic (
        .clk                  (clk),
        .rst_n                (rst_n),
        .scheduler_select     (scheduler_select[3:0]),
        .task_valid           (task_valid && (scheduler_select <= 5'd10)),
        .task_id              (task_id),
        .burst_time           (burst_time),
        .task_priority        (task_priority),
        .deadline             (deadline),
        .task_ready           (b_task_ready),
        .scheduled_task_valid (b_sched_valid),
        .scheduled_task_id    (b_sched_id),
        .scheduled_burst_time (b_sched_burst),
        .scheduled_priority   (),
        .scheduled_deadline   (),
        .task_tick            (task_tick),
        .task_complete        (task_complete && (scheduler_select <= 5'd10)),
        .queue_count          (),
        .queue_full           (),
        .queue_empty          (),
        .total_tasks_processed(basic_total_tasks_processed),
        .total_wait_time      (basic_total_wait_time),
        .total_turnaround_time(basic_total_turnaround_time)
    );

    //------------------------------------------------------------------------
    // DNN-aware wrapper (AI-MT / BatchDNN / BatchDNN++)
    //------------------------------------------------------------------------
    logic                      d_mt_valid, d_ct_valid;
    logic [LAYER_ID_WIDTH-1:0] d_mt_layer, d_ct_layer;
    logic [DNN_ID_WIDTH-1:0]   d_mt_dnn,   d_ct_dnn;
    logic [BATCH_WIDTH-1:0]    d_ct_batch;
    logic [CYCLE_WIDTH-1:0]    d_avail, d_total_stall;

    dnn_scheduler_wrapper #(
        .MAX_LAYERS      (MAX_LAYERS),
        .MAX_DNNS        (MAX_DNNS),
        .LAYER_ID_WIDTH  (LAYER_ID_WIDTH),
        .DNN_ID_WIDTH    (DNN_ID_WIDTH),
        .BATCH_WIDTH     (BATCH_WIDTH),
        .CYCLE_WIDTH     (CYCLE_WIDTH),
        .MEM_WIDTH       (MEM_WIDTH)
    ) u_dnn (
        .clk               (clk),
        .rst_n             (rst_n),
        .dnn_scheduler_select (scheduler_select[1:0]),
        .st_write_en       (st_write_en && (scheduler_select >= 5'd11)),
        .st_layer_idx      (st_layer_idx),
        .st_dnn_id         (st_dnn_id),
        .st_prev_layer     (st_prev_layer),
        .st_mem_cycles     (st_mem_cycles),
        .st_compute_cycles (st_compute_cycles),
        .st_weight_fp      (st_weight_fp),
        .st_ifmap_fp       (st_ifmap_fp),
        .st_ofmap_fp       (st_ofmap_fp),
        .st_batch          (st_batch),
        .st_total_layers   (st_total_layers),
        .mem_access_done   (mem_access_done && (scheduler_select >= 5'd11)),
        .compute_done      (compute_done    && (scheduler_select >= 5'd11)),
        .mt_valid          (d_mt_valid),
        .mt_layer_id       (d_mt_layer),
        .mt_dnn_id         (d_mt_dnn),
        .ct_valid          (d_ct_valid),
        .ct_layer_id       (d_ct_layer),
        .ct_dnn_id         (d_ct_dnn),
        .ct_batch_size     (d_ct_batch),
        .available_memory  (d_avail),
        .stall_detected    (dnn_stall_detected),
        .total_stall_cycles(d_total_stall)
    );

    //------------------------------------------------------------------------
    // Tagged-union output mux
    //------------------------------------------------------------------------
    always_comb begin
        sched_out                  = '0;
        ct_batch_size              = '0;
        task_ready                 = b_task_ready;

        if (scheduler_select <= 5'd10) begin
            sched_out.valid           = b_sched_valid;
            sched_out.is_dnn_aware    = 1'b0;
            sched_out.task_or_layer_id= b_sched_id;
            sched_out.dnn_id          = '0;
            sched_out.is_compute_task = 1'b1;       // basic tasks are always compute
            sched_out.burst_or_cycles = b_sched_burst;
            ct_batch_size             = 8'd1;
        end else begin
            // DNN-aware: prefer compute task over memory task (the dispatch
            // FSM decides which to action based on is_compute_task).
            if (d_ct_valid) begin
                sched_out.valid           = 1'b1;
                sched_out.is_dnn_aware    = 1'b1;
                sched_out.task_or_layer_id= d_ct_layer;
                sched_out.dnn_id          = d_ct_dnn;
                sched_out.is_compute_task = 1'b1;
                sched_out.burst_or_cycles = '0;
                ct_batch_size             = d_ct_batch;
            end else if (d_mt_valid) begin
                sched_out.valid           = 1'b1;
                sched_out.is_dnn_aware    = 1'b1;
                sched_out.task_or_layer_id= d_mt_layer;
                sched_out.dnn_id          = d_mt_dnn;
                sched_out.is_compute_task = 1'b0;
                sched_out.burst_or_cycles = '0;
                ct_batch_size             = 8'd1;
            end
        end
    end

endmodule : unified_scheduler_wrapper
