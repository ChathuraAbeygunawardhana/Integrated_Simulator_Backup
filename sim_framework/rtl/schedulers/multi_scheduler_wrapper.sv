// ============================================================================
// Multi-Scheduler Wrapper  (Fixed for Vivado array indices)
// ============================================================================

`timescale 1ns/1ps

module multi_scheduler_wrapper #(
    parameter MAX_TASKS        = 16,
    parameter TASK_ID_WIDTH    = 8,
    parameter BURST_TIME_WIDTH = 16,
    parameter PRIORITY_WIDTH   = 4,
    parameter DEADLINE_WIDTH   = 32,
    parameter TIME_QUANTUM     = 10
)(
    input  logic clk,
    input  logic rst_n,

    input  logic [3:0] scheduler_select,

    input  logic                         task_valid,
    input  logic [TASK_ID_WIDTH-1:0]     task_id,
    input  logic [BURST_TIME_WIDTH-1:0]  burst_time,
    input  logic [PRIORITY_WIDTH-1:0]    task_priority,
    input  logic [DEADLINE_WIDTH-1:0]    deadline,
    output logic                         task_ready,

    output logic                         scheduled_task_valid,
    output logic [TASK_ID_WIDTH-1:0]     scheduled_task_id,
    output logic [BURST_TIME_WIDTH-1:0]  scheduled_burst_time,
    output logic [PRIORITY_WIDTH-1:0]    scheduled_priority,
    output logic [DEADLINE_WIDTH-1:0]    scheduled_deadline,
    input  logic                         task_tick,
    input  logic                         task_complete,

    output logic [$clog2(MAX_TASKS):0]   queue_count,
    output logic                         queue_full,
    output logic                         queue_empty,

    output logic [31:0] total_tasks_processed,
    output logic [31:0] total_wait_time,
    output logic [31:0] total_turnaround_time
);

    logic gated_task_valid;
    logic gated_task_complete;
    logic gated_task_tick;
    
    assign gated_task_valid = task_valid;
    assign gated_task_complete = task_complete;
    assign gated_task_tick = task_tick;

    // Basic arrays
    logic                          basic_task_ready      [7];
    logic                          basic_sched_valid     [7];
    logic [TASK_ID_WIDTH-1:0]      basic_sched_id        [7];
    logic [BURST_TIME_WIDTH-1:0]   basic_sched_burst     [7];
    logic [PRIORITY_WIDTH-1:0]     basic_sched_priority  [7];
    logic [DEADLINE_WIDTH-1:0]     basic_sched_deadline  [7];
    logic [$clog2(MAX_TASKS):0]    basic_queue_count     [7];
    logic                          basic_queue_full      [7];
    logic                          basic_queue_empty     [7];

    // Adv arrays
    logic                          adv_task_ready        [4];
    logic                          adv_sched_valid       [4];
    logic [TASK_ID_WIDTH-1:0]      adv_sched_id          [4];
    logic [BURST_TIME_WIDTH-1:0]   adv_sched_burst       [4];
    logic [PRIORITY_WIDTH-1:0]     adv_sched_priority    [4];
    logic [$clog2(MAX_TASKS):0]    adv_queue_count       [4];
    logic                          adv_queue_full        [4];
    logic                          adv_queue_empty       [4];

    genvar i;
    generate
        for (i = 0; i < 7; i++) begin : gen_basic_schedulers
            task_scheduler #(
                .MAX_TASKS       (MAX_TASKS),
                .TASK_ID_WIDTH   (TASK_ID_WIDTH),
                .BURST_TIME_WIDTH(BURST_TIME_WIDTH),
                .PRIORITY_WIDTH  (PRIORITY_WIDTH),
                .DEADLINE_WIDTH  (DEADLINE_WIDTH),
                .TIME_QUANTUM    (TIME_QUANTUM),
                .SCHEDULER_TYPE  (i)
            ) u_basic (
                .clk                  (clk),
                .rst_n                (rst_n),
                .task_valid           (gated_task_valid),
                .task_id              (task_id),
                .burst_time           (burst_time),
                .task_priority        (task_priority),
                .deadline             (deadline),
                .task_ready           (basic_task_ready    [i]),
                .scheduled_task_valid (basic_sched_valid   [i]),
                .scheduled_task_id    (basic_sched_id      [i]),
                .scheduled_burst_time (basic_sched_burst   [i]),
                .scheduled_priority   (basic_sched_priority[i]),
                .scheduled_deadline   (basic_sched_deadline[i]),
                .task_complete        (gated_task_complete),
                .queue_count          (basic_queue_count   [i]),
                .queue_full           (basic_queue_full    [i]),
                .queue_empty          (basic_queue_empty   [i])
            );
        end
    endgenerate

    genvar j;
    generate
        for (j = 0; j < 4; j++) begin : gen_advanced_schedulers
            advanced_task_scheduler #(
                .MAX_TASKS       (MAX_TASKS),
                .TASK_ID_WIDTH   (TASK_ID_WIDTH),
                .BURST_TIME_WIDTH(BURST_TIME_WIDTH),
                .PRIORITY_WIDTH  (PRIORITY_WIDTH),
                .DEADLINE_WIDTH  (DEADLINE_WIDTH),
                .SCHEDULER_TYPE  (j)
            ) u_adv (
                .clk                  (clk),
                .rst_n                (rst_n),
                .task_valid           (gated_task_valid),
                .task_id              (task_id),
                .burst_time           (burst_time),
                .task_priority        (task_priority),
                .deadline             (deadline),
                .task_ready           (adv_task_ready    [j]),
                .scheduled_task_valid (adv_sched_valid   [j]),
                .scheduled_task_id    (adv_sched_id      [j]),
                .scheduled_burst_time (adv_sched_burst   [j]),
                .scheduled_priority   (adv_sched_priority[j]),
                .task_tick            (gated_task_tick),
                .task_complete        (gated_task_complete),
                .queue_count          (adv_queue_count   [j]),
                .queue_full           (adv_queue_full    [j]),
                .queue_empty          (adv_queue_empty   [j])
            );
        end
    endgenerate

    always_comb begin
        task_ready           = 1'b0;
        scheduled_task_valid = 1'b0;
        scheduled_task_id    = '0;
        scheduled_burst_time = '0;
        scheduled_priority   = '0;
        scheduled_deadline   = '0;
        queue_count          = '0;
        queue_full           = 1'b1;
        queue_empty          = 1'b1;

        if (scheduler_select < 4'd7) begin
            task_ready           = basic_task_ready    [scheduler_select];
            scheduled_task_valid = basic_sched_valid   [scheduler_select];
            scheduled_task_id    = basic_sched_id      [scheduler_select];
            scheduled_burst_time = basic_sched_burst   [scheduler_select];
            scheduled_priority   = basic_sched_priority[scheduler_select];
            scheduled_deadline   = basic_sched_deadline[scheduler_select];
            queue_count          = basic_queue_count   [scheduler_select];
            queue_full           = basic_queue_full    [scheduler_select];
            queue_empty          = basic_queue_empty   [scheduler_select];
        end else if (scheduler_select >= 4'd7 && scheduler_select < 4'd11) begin
            task_ready           = adv_task_ready    [scheduler_select - 4'd7];
            scheduled_task_valid = adv_sched_valid   [scheduler_select - 4'd7];
            scheduled_task_id    = adv_sched_id      [scheduler_select - 4'd7];
            scheduled_burst_time = adv_sched_burst   [scheduler_select - 4'd7];
            scheduled_priority   = adv_sched_priority[scheduler_select - 4'd7];
            scheduled_deadline   = '0; 
            queue_count          = adv_queue_count   [scheduler_select - 4'd7];
            queue_full           = adv_queue_full    [scheduler_select - 4'd7];
            queue_empty          = adv_queue_empty   [scheduler_select - 4'd7];
        end
    end

    // Statistics collection
    typedef struct {
        logic [TASK_ID_WIDTH-1:0] id;
        logic [31:0]              arrival_time;
        logic [31:0]              completion_time;
        logic                     recorded;
    } task_stats_t;

    task_stats_t task_stats     [255:0];
    logic [7:0]  stats_write_ptr;
    logic [31:0] current_time;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_tasks_processed <= '0;
            total_wait_time       <= '0;
            total_turnaround_time <= '0;
            stats_write_ptr       <= '0;
            current_time          <= '0;
            for (int k = 0; k < 256; k++) begin
                task_stats[k].id              <= '0;
                task_stats[k].arrival_time    <= '0;
                task_stats[k].completion_time <= '0;
                task_stats[k].recorded        <= 1'b0;
            end
        end else begin
            current_time <= current_time + 1;

            if (task_valid && task_ready) begin
                task_stats[stats_write_ptr].id           <= task_id;
                task_stats[stats_write_ptr].arrival_time <= current_time;
                task_stats[stats_write_ptr].recorded     <= 1'b0;
                stats_write_ptr <= stats_write_ptr + 1;
            end

            if (task_complete && scheduled_task_valid) begin
                for (int k = 0; k < 256; k++) begin
                    if (task_stats[k].id == scheduled_task_id && !task_stats[k].recorded) begin
                        automatic logic [31:0] turnaround;
                        automatic logic [31:0] wait_t;
                        task_stats[k].completion_time <= current_time;
                        task_stats[k].recorded <= 1'b1;
                        turnaround = current_time - task_stats[k].arrival_time;
                        wait_t     = (turnaround >= scheduled_burst_time) ? (turnaround - scheduled_burst_time) : '0;
                        total_turnaround_time <= total_turnaround_time + turnaround;
                        total_wait_time       <= total_wait_time + wait_t;
                        total_tasks_processed <= total_tasks_processed + 1;
                        break;
                    end
                end
            end
        end
    end
endmodule

