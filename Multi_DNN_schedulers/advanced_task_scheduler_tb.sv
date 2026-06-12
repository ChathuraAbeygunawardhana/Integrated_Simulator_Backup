// ============================================================================
// Advanced Task Schedulers
// Includes: SRTF, HRRN, MLQ, MLFQ
// ============================================================================

`timescale 1ns/1ps

module advanced_task_scheduler #(
    parameter MAX_TASKS = 16,
    parameter TASK_ID_WIDTH = 8,
    parameter BURST_TIME_WIDTH = 16,
    parameter PRIORITY_WIDTH = 4,
    parameter DEADLINE_WIDTH = 32,
    parameter NUM_QUEUES = 4,
    parameter SCHEDULER_TYPE = 0 // 0:SRTF, 1:HRRN, 2:MLQ, 3:MLFQ
)(
    input  logic clk,
    input  logic rst_n,
    
    input  logic task_valid,
    input  logic [TASK_ID_WIDTH-1:0] task_id,
    input  logic [BURST_TIME_WIDTH-1:0] burst_time,
    input  logic [PRIORITY_WIDTH-1:0] task_priority,
    input  logic [DEADLINE_WIDTH-1:0] deadline,
    output logic task_ready,
    
    output logic scheduled_task_valid,
    output logic [TASK_ID_WIDTH-1:0] scheduled_task_id,
    output logic [BURST_TIME_WIDTH-1:0] scheduled_burst_time,
    output logic [PRIORITY_WIDTH-1:0] scheduled_priority,
    input  logic task_tick,
    input  logic task_complete,
    
    output logic [$clog2(MAX_TASKS):0] queue_count,
    output logic queue_full,
    output logic queue_empty
);

    typedef struct packed {
        logic valid;
        logic [TASK_ID_WIDTH-1:0] id;
        logic [BURST_TIME_WIDTH-1:0] burst_time;
        logic [BURST_TIME_WIDTH-1:0] remaining_time;
        logic [PRIORITY_WIDTH-1:0] task_priority;
        logic [DEADLINE_WIDTH-1:0] deadline;
        logic [DEADLINE_WIDTH-1:0] arrival_time;
        logic [DEADLINE_WIDTH-1:0] wait_time;
        logic [$clog2(NUM_QUEUES)-1:0] queue_level;
    } task_t;
    
    task_t task_queue [MAX_TASKS-1:0];
    logic [$clog2(MAX_TASKS):0] num_tasks;
    logic [DEADLINE_WIDTH-1:0] current_time;
    logic [TASK_ID_WIDTH-1:0] current_task_id;
    logic [TASK_ID_WIDTH-1:0] removing_id;
    logic task_running;
    
    assign queue_count = num_tasks;
    assign queue_full = (num_tasks == MAX_TASKS);
    assign queue_empty = (num_tasks == 0);
    assign task_ready = !queue_full;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_time <= '0;
            removing_id  <= '0;
        end else begin 
            current_time <= current_time + 1;
            removing_id  <= scheduled_task_id;
        end
    end
    
    int remove_idx;

    // =========================================================================
    // UNIFIED TASK MANAGEMENT & SCHEDULING
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            num_tasks <= '0;
            for (int i = 0; i < MAX_TASKS; i++) task_queue[i] <= '0;
            task_running <= 1'b0;
            scheduled_task_valid <= 1'b0;
            scheduled_task_id <= '0;
            scheduled_burst_time <= '0;
            scheduled_priority <= '0;
            current_task_id <= '0;
        end else begin
            
            // 1. Process Insertions
            if (task_valid && !queue_full) begin
                task_queue[num_tasks].valid <= 1'b1;
                task_queue[num_tasks].id <= task_id;
                task_queue[num_tasks].burst_time <= burst_time;
                task_queue[num_tasks].remaining_time <= burst_time;
                task_queue[num_tasks].task_priority <= task_priority;
                task_queue[num_tasks].deadline <= deadline;
                task_queue[num_tasks].arrival_time <= current_time;
                task_queue[num_tasks].wait_time <= '0;
                task_queue[num_tasks].queue_level <= '0;
                num_tasks <= num_tasks + 1;
            end
            
            // 2. Update ticks and wait times
            if (scheduled_task_valid && task_tick) begin
                for (int i = 0; i < MAX_TASKS; i++) begin
                    if (i < num_tasks && task_queue[i].id == scheduled_task_id) begin
                        task_queue[i].remaining_time <= task_queue[i].remaining_time - 1;
                    end
                end
            end
            
            for (int i = 0; i < MAX_TASKS; i++) begin
                if (i < num_tasks && (!scheduled_task_valid || task_queue[i].id != scheduled_task_id)) begin
                    task_queue[i].wait_time <= task_queue[i].wait_time + 1;
                end
            end
            
            // 3. Process Removals & Scheduling decisions
            if (task_complete) begin
                remove_idx = -1;
                for (int i = 0; i < MAX_TASKS; i++) begin
                    if (i < num_tasks && task_queue[i].id == removing_id) begin
                        remove_idx = i;
                        break;
                    end
                end
                
                if (remove_idx >= 0) begin
                    for (int i = 0; i < MAX_TASKS-1; i++) begin
                        if (i >= remove_idx && i < num_tasks - 1)
                            task_queue[i] <= task_queue[i+1];
                    end
                    task_queue[num_tasks-1] <= '0;
                    num_tasks <= num_tasks - 1;
                end
                
                task_running <= 1'b0;
                scheduled_task_valid <= 1'b0;

            end else if (!queue_empty) begin
                case (SCHEDULER_TYPE)
                    0: schedule_srtf();
                    1: schedule_hrrn();
                    2: schedule_mlq();
                    3: schedule_mlfq();
                    default: schedule_srtf();
                endcase
                task_running <= 1'b1;
            end else begin
                scheduled_task_valid <= 1'b0;
                task_running <= 1'b0;
            end
        end
    end
    
    // Declared as `task automatic` so variables are dynamically allocated cleanly by Vivado
    task automatic schedule_srtf();
        int shortest_idx = 0;
        logic [BURST_TIME_WIDTH-1:0] min_time = task_queue[0].remaining_time;
        for (int i = 1; i < MAX_TASKS; i++) begin
            if (i < num_tasks && task_queue[i].remaining_time < min_time) begin
                min_time = task_queue[i].remaining_time;
                shortest_idx = i;
            end
        end
        scheduled_task_valid <= 1'b1;
        scheduled_task_id <= task_queue[shortest_idx].id;
        scheduled_burst_time <= task_queue[shortest_idx].remaining_time;
        scheduled_priority <= task_queue[shortest_idx].task_priority;
        current_task_id <= task_queue[shortest_idx].id;
    endtask
    
    task automatic schedule_hrrn();
        int highest_idx = 0;
        real max_ratio = 0.0;
        real current_ratio;
        for (int i = 0; i < MAX_TASKS; i++) begin
            if (i < num_tasks && task_queue[i].burst_time > 0) begin
                current_ratio = real'(task_queue[i].wait_time + task_queue[i].burst_time) / real'(task_queue[i].burst_time);
                if (current_ratio > max_ratio) begin
                    max_ratio = current_ratio;
                    highest_idx = i;
                end
            end
        end
        scheduled_task_valid <= 1'b1;
        scheduled_task_id <= task_queue[highest_idx].id;
        scheduled_burst_time <= task_queue[highest_idx].burst_time;
        scheduled_priority <= task_queue[highest_idx].task_priority;
        current_task_id <= task_queue[highest_idx].id;
    endtask
    
    task automatic schedule_mlq();
        int selected_idx = -1;
        for (int q = NUM_QUEUES - 1; q >= 0; q--) begin 
            for (int i = 0; i < MAX_TASKS; i++) begin
                if (i < num_tasks && task_queue[i].queue_level == q[$clog2(NUM_QUEUES)-1:0]) begin
                    selected_idx = i;
                    break;
                end
            end
            if (selected_idx >= 0) break;
        end
        if (selected_idx >= 0) begin
            scheduled_task_valid <= 1'b1;
            scheduled_task_id <= task_queue[selected_idx].id;
            scheduled_burst_time <= task_queue[selected_idx].burst_time;
            scheduled_priority <= task_queue[selected_idx].task_priority;
        end else begin
            scheduled_task_valid <= 1'b0;
        end
    endtask
    
    task automatic schedule_mlfq();
        int selected_idx = -1;
        for (int q = 0; q < NUM_QUEUES; q++) begin
            for (int i = 0; i < MAX_TASKS; i++) begin
                if (i < num_tasks && task_queue[i].queue_level == q[$clog2(NUM_QUEUES)-1:0]) begin
                    selected_idx = i;
                    break;
                end
            end
            if (selected_idx >= 0) break;
        end
        if (selected_idx >= 0) begin
            scheduled_task_valid <= 1'b1;
            scheduled_task_id <= task_queue[selected_idx].id;
            scheduled_burst_time <= task_queue[selected_idx].remaining_time;
            scheduled_priority <= task_queue[selected_idx].task_priority;
            
            if (task_tick && !task_complete) begin
                if (task_queue[selected_idx].queue_level < NUM_QUEUES - 1) begin
                    task_queue[selected_idx].queue_level <= task_queue[selected_idx].queue_level + 1;
                end
            end
            
            for (int i = 0; i < MAX_TASKS; i++) begin
                if (i < num_tasks && task_queue[i].wait_time > 100 && task_queue[i].queue_level > 0) begin
                    task_queue[i].queue_level <= task_queue[i].queue_level - 1;
                end
            end
        end else begin
            scheduled_task_valid <= 1'b0;
        end
    endtask

endmodule
