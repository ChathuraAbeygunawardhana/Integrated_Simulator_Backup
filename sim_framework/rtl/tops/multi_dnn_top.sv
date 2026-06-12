//=============================================================================
// Module: multi_dnn_top
// Description: Multi-DNN simulator top. Wraps one single_dnn_top and a
//              unified_scheduler_wrapper, plus a dispatch FSM that drives
//              start/done handshakes per task.
//
// Knobs (in addition to all single_dnn_top knobs):
//   * SCHEDULER_DEFAULT (0..13)
//   * MAX_TASKS, MAX_LAYERS, MAX_DNNS
//
// The dispatch FSM:
//   - When sched_out.valid is high, latches the task config, drives the
//     single_dnn_top start signal.
//   - When the single_dnn_top phase_mem_done fires, pulses mem_access_done
//     back to the scheduler (DNN-aware schedulers use this).
//   - When phase_compute_done fires, pulses compute_done + task_complete.
//   - For basic schedulers (is_dnn_aware = 0), only compute_done -> task_complete
//     is meaningful; memory phase is hidden internal to single_dnn_top.
//=============================================================================

module multi_dnn_top
    import sim_framework_pkg::*;
#(
    parameter dataflow_e    DATAFLOW          = DF_OS,
    parameter mem_backend_e MEMORY            = MEM_STAMP,
    parameter int           ARRAY_HEIGHT      = 4,
    parameter int           ARRAY_WIDTH       = 4,
    parameter int           DATA_WIDTH        = 32,
    parameter int           ACCUM_WIDTH       = 48,
    parameter int           ADDR_WIDTH        = 32,
    parameter int           SPAD_DEPTH        = 4096,
    parameter int           AXI_DATA_WIDTH    = 64,
    parameter int           METADATA_DEPTH    = 256,
    parameter int           NUM_PAGES         = 256,
    parameter int           PAGE_SIZE_BITS    = 12,
    parameter int           N_MEM_PORTS       = 4,

    parameter int           MAX_TASKS         = 16,
    parameter int           MAX_LAYERS        = 32,
    parameter int           MAX_DNNS          = 4,
    parameter int           TASK_ID_WIDTH     = 8,
    parameter int           LAYER_ID_WIDTH    = 8,
    parameter int           DNN_ID_WIDTH      = 4,
    parameter int           BATCH_WIDTH       = 8,
    parameter int           CYCLE_WIDTH       = 32,
    parameter int           MEM_WIDTH         = 32
) (
    input  logic clk,
    input  logic rst_n,

    //--- Scheduler selection (runtime)
    input  logic [4:0] scheduler_select,

    //--- Basic-task interface
    input  logic                         task_valid,
    input  logic [TASK_ID_WIDTH-1:0]     task_id,
    input  logic [15:0]                  burst_time,
    input  logic [3:0]                   task_priority,
    input  logic [31:0]                  deadline,
    input  logic                         task_tick,
    output logic                         task_ready,
    output logic                         task_complete_pulse,

    //--- DNN-aware scheduling-table interface
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

    //--- Per-task layer config (the dispatcher writes these into the
    //    single_dnn_top before issuing start)
    input  mem_layout_t           cfg_mem_layout,
    input  logic [15:0]           cfg_input_channels, cfg_input_height, cfg_input_width,
    input  logic [15:0]           cfg_weight_k, cfg_weight_c, cfg_weight_kh, cfg_weight_kw,
    input  logic [15:0]           cfg_output_channels, cfg_output_height, cfg_output_width,
    input  logic [ADDR_WIDTH-1:0] cfg_input_base_addr, cfg_weight_base_addr, cfg_output_base_addr,
    input  logic [15:0]           cfg_tile_row, cfg_tile_col_start, cfg_tile_ch_start,
    input  logic [15:0]           cfg_ws_input_ch, cfg_ws_kernel_row, cfg_ws_kernel_col_start,

    //--- AXI DRAM
    output logic [ADDR_WIDTH-1:0] axi_arvalid_addr,
    output logic [7:0]            axi_arlen,
    output logic                  axi_arvalid,
    input  logic                  axi_arready,
    input  logic [DATA_WIDTH-1:0] axi_rdata,
    input  logic                  axi_rvalid,
    output logic                  axi_rready,
    input  logic                  axi_rlast,

    //--- Direct memory ports passthrough (for behavioral memory model tests)
    output logic [ADDR_WIDTH-1:0]  ext_input_addr_w  [ARRAY_WIDTH],
    output logic [ARRAY_WIDTH-1:0] ext_input_addr_valid_w,
    input  logic [DATA_WIDTH-1:0]  ext_input_data_w  [ARRAY_WIDTH],
    input  logic [ARRAY_WIDTH-1:0] ext_input_data_valid_w,
    output logic [ADDR_WIDTH-1:0]   ext_input_addr_h  [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] ext_input_addr_valid_h,
    input  logic [DATA_WIDTH-1:0]   ext_input_data_h  [ARRAY_HEIGHT],
    input  logic [ARRAY_HEIGHT-1:0] ext_input_data_valid_h,
    output logic [ADDR_WIDTH-1:0]   ext_weight_addr_1d  [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] ext_weight_addr_valid_1d,
    input  logic [DATA_WIDTH-1:0]   ext_weight_data_1d  [ARRAY_HEIGHT],
    input  logic [ARRAY_HEIGHT-1:0] ext_weight_data_valid_1d,
    output logic [ADDR_WIDTH-1:0]                    ext_weight_addr_2d  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] ext_weight_addr_valid_2d,
    input  logic [DATA_WIDTH-1:0]                    ext_weight_data_2d  [ARRAY_HEIGHT][ARRAY_WIDTH],
    input  logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] ext_weight_data_valid_2d,
    output logic [ADDR_WIDTH-1:0]                    ext_output_addr_2d  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] ext_output_addr_valid_2d,
    output logic [DATA_WIDTH-1:0]                    ext_output_data_2d  [ARRAY_HEIGHT][ARRAY_WIDTH],
    output logic [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0] ext_output_data_valid_2d,
    output logic [ADDR_WIDTH-1:0]   ext_output_addr_1d  [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] ext_output_addr_valid_1d,
    output logic [DATA_WIDTH-1:0]   ext_output_data_1d  [ARRAY_HEIGHT],
    output logic [ARRAY_HEIGHT-1:0] ext_output_data_valid_1d,

    //--- Top-level stats
    output logic [31:0]                  total_tasks_processed,
    output logic [LAYER_ID_WIDTH-1:0]    current_layer_id,
    output logic [DNN_ID_WIDTH-1:0]      current_dnn_id,
    output sched_output_t                sched_out_dbg
);

    // --------------------------------------------------------------------
    // Scheduler
    // --------------------------------------------------------------------
    sched_output_t       sched_out;
    logic [BATCH_WIDTH-1:0] ct_batch_size;
    logic                dnn_stall_detected;
    logic                task_complete_to_sched;
    logic                mem_access_done_to_sched, compute_done_to_sched;

    unified_scheduler_wrapper #(
        .MAX_TASKS       (MAX_TASKS),
        .TASK_ID_WIDTH   (TASK_ID_WIDTH),
        .MAX_LAYERS      (MAX_LAYERS),
        .MAX_DNNS        (MAX_DNNS),
        .LAYER_ID_WIDTH  (LAYER_ID_WIDTH),
        .DNN_ID_WIDTH    (DNN_ID_WIDTH),
        .BATCH_WIDTH     (BATCH_WIDTH),
        .CYCLE_WIDTH     (CYCLE_WIDTH),
        .MEM_WIDTH       (MEM_WIDTH)
    ) u_sched (
        .clk             (clk),
        .rst_n           (rst_n),
        .scheduler_select(scheduler_select),
        .task_valid      (task_valid),
        .task_id         (task_id),
        .burst_time      (burst_time),
        .task_priority   (task_priority),
        .deadline        (deadline),
        .task_tick       (task_tick),
        .task_complete   (task_complete_to_sched),
        .task_ready      (task_ready),
        .st_write_en     (st_write_en),
        .st_layer_idx    (st_layer_idx),
        .st_dnn_id       (st_dnn_id),
        .st_prev_layer   (st_prev_layer),
        .st_mem_cycles   (st_mem_cycles),
        .st_compute_cycles(st_compute_cycles),
        .st_weight_fp    (st_weight_fp),
        .st_ifmap_fp     (st_ifmap_fp),
        .st_ofmap_fp     (st_ofmap_fp),
        .st_batch        (st_batch),
        .st_total_layers (st_total_layers),
        .mem_access_done (mem_access_done_to_sched),
        .compute_done    (compute_done_to_sched),
        .sched_out       (sched_out),
        .ct_batch_size   (ct_batch_size),
        .basic_total_tasks_processed(total_tasks_processed),
        .basic_total_wait_time(),
        .basic_total_turnaround_time(),
        .dnn_stall_detected(dnn_stall_detected)
    );

    assign sched_out_dbg = sched_out;

    // --------------------------------------------------------------------
    // single_dnn_top
    // --------------------------------------------------------------------
    logic        sdt_start;
    logic        sdt_done;
    logic        sdt_phase_mem_done, sdt_phase_compute_done;

    single_dnn_top #(
        .DATAFLOW      (DATAFLOW),
        .MEMORY        (MEMORY),
        .ARRAY_HEIGHT  (ARRAY_HEIGHT),
        .ARRAY_WIDTH   (ARRAY_WIDTH),
        .DATA_WIDTH    (DATA_WIDTH),
        .ACCUM_WIDTH   (ACCUM_WIDTH),
        .ADDR_WIDTH    (ADDR_WIDTH),
        .SPAD_DEPTH    (SPAD_DEPTH),
        .AXI_DATA_WIDTH(AXI_DATA_WIDTH),
        .METADATA_DEPTH(METADATA_DEPTH),
        .NUM_PAGES     (NUM_PAGES),
        .PAGE_SIZE_BITS(PAGE_SIZE_BITS),
        .N_MEM_PORTS   (N_MEM_PORTS)
    ) u_sdt (
        .clk                (clk),
        .rst_n              (rst_n),
        .mem_layout         (cfg_mem_layout),
        .input_channels     (cfg_input_channels),
        .input_height       (cfg_input_height),
        .input_width        (cfg_input_width),
        .weight_k           (cfg_weight_k),
        .weight_c           (cfg_weight_c),
        .weight_kh          (cfg_weight_kh),
        .weight_kw          (cfg_weight_kw),
        .output_channels    (cfg_output_channels),
        .output_height      (cfg_output_height),
        .output_width       (cfg_output_width),
        .input_base_addr    (cfg_input_base_addr),
        .weight_base_addr   (cfg_weight_base_addr),
        .output_base_addr   (cfg_output_base_addr),
        .start              (sdt_start),
        .tile_row           (cfg_tile_row),
        .tile_col_start     (cfg_tile_col_start),
        .tile_ch_start      (cfg_tile_ch_start),
        .ws_input_ch        (cfg_ws_input_ch),
        .ws_kernel_row      (cfg_ws_kernel_row),
        .ws_kernel_col_start(cfg_ws_kernel_col_start),
        .done               (sdt_done),
        .busy               (),
        .phase_mem_done     (sdt_phase_mem_done),
        .phase_compute_done (sdt_phase_compute_done),
        .metadata_wr_en     (1'b0),
        .metadata_wr_addr   ('0),
        .metadata_wr_data   ('0),
        .phase_start_in     (sdt_start),
        .phase_id_in        ('0),
        .num_delta_ops_in   ('0),
        .phase_base_addr_in ('0),
        .pt_write_en        (1'b0),
        .pt_write_vpn       ('0),
        .pt_write_ppn       ('0),
        .pt_write_valid     (1'b0),
        .axi_arvalid_addr   (axi_arvalid_addr),
        .axi_arlen          (axi_arlen),
        .axi_arvalid        (axi_arvalid),
        .axi_arready        (axi_arready),
        .axi_rdata          (axi_rdata),
        .axi_rvalid         (axi_rvalid),
        .axi_rready         (axi_rready),
        .axi_rlast          (axi_rlast),
        .ext_input_addr_w        (ext_input_addr_w),
        .ext_input_addr_valid_w  (ext_input_addr_valid_w),
        .ext_input_data_w        (ext_input_data_w),
        .ext_input_data_valid_w  (ext_input_data_valid_w),
        .ext_input_addr_h        (ext_input_addr_h),
        .ext_input_addr_valid_h  (ext_input_addr_valid_h),
        .ext_input_data_h        (ext_input_data_h),
        .ext_input_data_valid_h  (ext_input_data_valid_h),
        .ext_weight_addr_1d      (ext_weight_addr_1d),
        .ext_weight_addr_valid_1d(ext_weight_addr_valid_1d),
        .ext_weight_data_1d      (ext_weight_data_1d),
        .ext_weight_data_valid_1d(ext_weight_data_valid_1d),
        .ext_weight_addr_2d      (ext_weight_addr_2d),
        .ext_weight_addr_valid_2d(ext_weight_addr_valid_2d),
        .ext_weight_data_2d      (ext_weight_data_2d),
        .ext_weight_data_valid_2d(ext_weight_data_valid_2d),
        .ext_output_addr_2d      (ext_output_addr_2d),
        .ext_output_addr_valid_2d(ext_output_addr_valid_2d),
        .ext_output_data_2d      (ext_output_data_2d),
        .ext_output_data_valid_2d(ext_output_data_valid_2d),
        .ext_output_addr_1d      (ext_output_addr_1d),
        .ext_output_addr_valid_1d(ext_output_addr_valid_1d),
        .ext_output_data_1d      (ext_output_data_1d),
        .ext_output_data_valid_1d(ext_output_data_valid_1d),
        .stats_loads_or_hits     (),
        .stats_moves_or_misses   (),
        .stats_keeps             (),
        .stats_bytes_loaded      (),
        .stats_bytes_moved       (),
        .stats_compute_cycles    ()
    );

    // --------------------------------------------------------------------
    // Dispatch FSM
    // --------------------------------------------------------------------
    typedef enum logic [1:0] { D_IDLE, D_RUN, D_WAIT_DONE } dst_t;
    dst_t dstate;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dstate                   <= D_IDLE;
            sdt_start                <= 1'b0;
            mem_access_done_to_sched <= 1'b0;
            compute_done_to_sched    <= 1'b0;
            task_complete_to_sched   <= 1'b0;
            task_complete_pulse      <= 1'b0;
            current_layer_id         <= '0;
            current_dnn_id           <= '0;
        end else begin
            // Defaults: deassert pulses
            sdt_start                <= 1'b0;
            mem_access_done_to_sched <= 1'b0;
            compute_done_to_sched    <= 1'b0;
            task_complete_to_sched   <= 1'b0;
            task_complete_pulse      <= 1'b0;

            case (dstate)
                D_IDLE: if (sched_out.valid) begin
                    current_layer_id <= sched_out.task_or_layer_id;
                    current_dnn_id   <= sched_out.dnn_id;
                    sdt_start        <= 1'b1;
                    dstate           <= D_RUN;
                end
                D_RUN: begin
                    if (sdt_phase_mem_done)     mem_access_done_to_sched <= 1'b1;
                    if (sdt_phase_compute_done) compute_done_to_sched    <= 1'b1;
                    if (sdt_done) dstate <= D_WAIT_DONE;
                end
                D_WAIT_DONE: begin
                    task_complete_to_sched <= 1'b1;
                    task_complete_pulse    <= 1'b1;
                    dstate                 <= D_IDLE;
                end
            endcase
        end
    end

endmodule : multi_dnn_top
