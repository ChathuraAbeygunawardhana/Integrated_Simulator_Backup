// ============================================================================
// Multi-Algorithm Hardware Task Scheduler
// Supports: FIFO, LIFO, SJF, Round Robin, Priority, EDF, LRU
// ============================================================================

`timescale 1ns/1ps

module task_scheduler #(
    parameter MAX_TASKS        = 16,
    parameter TASK_ID_WIDTH    = 8,
    parameter BURST_TIME_WIDTH = 16,
    parameter PRIORITY_WIDTH   = 4,
    parameter DEADLINE_WIDTH   = 32,
    parameter TIME_QUANTUM     = 10,
    parameter SCHEDULER_TYPE   = 0   // 0:FIFO 1:LIFO 2:SJF 3:RR 4:Priority 5:EDF 6:LRU
)(
    input  logic clk,
    input  logic rst_n,

    // Task arrival
    input  logic                         task_valid,
    input  logic [TASK_ID_WIDTH-1:0]     task_id,
    input  logic [BURST_TIME_WIDTH-1:0]  burst_time,
    input  logic [PRIORITY_WIDTH-1:0]    task_priority,
    input  logic [DEADLINE_WIDTH-1:0]    deadline,
    output logic                         task_ready,

    // Scheduler output
    output logic                         scheduled_task_valid,
    output logic [TASK_ID_WIDTH-1:0]     scheduled_task_id,
    output logic [BURST_TIME_WIDTH-1:0]  scheduled_burst_time,
    output logic [PRIORITY_WIDTH-1:0]    scheduled_priority,
    output logic [DEADLINE_WIDTH-1:0]    scheduled_deadline,

    input  logic task_complete,

    // Status
    output logic [$clog2(MAX_TASKS):0]  queue_count,
    output logic                        queue_full,
    output logic                        queue_empty
);

    typedef struct packed {
        logic                        valid;
        logic [TASK_ID_WIDTH-1:0]    id;
        logic [BURST_TIME_WIDTH-1:0] burst_time;
        logic [BURST_TIME_WIDTH-1:0] remaining_time;
        logic [PRIORITY_WIDTH-1:0]   task_priority;
        logic [DEADLINE_WIDTH-1:0]   deadline;
        logic [DEADLINE_WIDTH-1:0]   arrival_time;
        logic [DEADLINE_WIDTH-1:0]   last_access_time;
    } task_t;

    task_t task_queue [MAX_TASKS-1:0];
    logic [$clog2(MAX_TASKS):0]  num_tasks;
    logic [DEADLINE_WIDTH-1:0]   current_time;

    // Round Robin & Removal Tracking State
    logic [$clog2(MAX_TASKS)-1:0] rr_ptr;
    logic [BURST_TIME_WIDTH-1:0]  quantum_left;
    logic [TASK_ID_WIDTH-1:0]     removing_id;

    assign queue_count = num_tasks;
    assign queue_full  = (num_tasks == MAX_TASKS);
    assign queue_empty = (num_tasks == 0);
    assign task_ready  = !queue_full;

    // Move delayed output declarations UP so the combinational lock can read them
    logic scheduled_task_valid_delayed;
    logic [TASK_ID_WIDTH-1:0] scheduled_task_id_delayed;
    logic [BURST_TIME_WIDTH-1:0] scheduled_burst_time_delayed;
    logic [PRIORITY_WIDTH-1:0] scheduled_priority_delayed;
    logic [DEADLINE_WIDTH-1:0] scheduled_deadline_delayed;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_time <= '0;
            removing_id  <= '0;
        end else begin
            current_time <= current_time + 1;
            removing_id  <= scheduled_task_id_delayed; 
        end
    end

    // =========================================================================
    // NON-PREEMPTIVE LOCK (FIXED)
    // =========================================================================
    logic is_non_preemptive;
    logic is_locked;

    assign is_non_preemptive = (SCHEDULER_TYPE == 0 || SCHEDULER_TYPE == 1 || SCHEDULER_TYPE == 2 || SCHEDULER_TYPE == 4 || SCHEDULER_TYPE == 5);
    
    // Instead of a delayed flip-flop, lock directly onto the output register
    // If we have a valid task scheduled, and it hasn't completed yet, DO NOT switch.
    assign is_locked = is_non_preemptive && scheduled_task_valid_delayed && !task_complete;

    // =========================================================================
    // COMBINATIONAL SELECTOR
    // =========================================================================
    logic [$clog2(MAX_TASKS)-1:0] sel;
    int locked_idx;
    
    logic [BURST_TIME_WIDTH-1:0] min_bt;
    logic [PRIORITY_WIDTH-1:0]   max_pri;
    logic [DEADLINE_WIDTH-1:0]   min_dl;
    logic [DEADLINE_WIDTH-1:0]   min_acc;

    always_comb begin
        sel = '0;
        locked_idx = -1;

        if (is_locked) begin
            for (int i = 0; i < MAX_TASKS; i++) begin
                if (i < num_tasks && task_queue[i].id == scheduled_task_id_delayed) begin
                    locked_idx = i;
                    break;
                end
            end
        end

        if (locked_idx != -1) begin
            sel = ($clog2(MAX_TASKS))'(locked_idx);
        end else begin
            case (SCHEDULER_TYPE)
                0: begin // FIFO
                    if (num_tasks > 0) sel = '0;
                end

                1: begin // LIFO
                    // Stack behavior: Always grab the newest task (highest index)
                    if (num_tasks > 0) sel = ($clog2(MAX_TASKS))'(num_tasks - 1);
                end

                2: begin // SJF
                    if (num_tasks > 0) begin
                        min_bt = task_queue[0].burst_time;
                        sel = '0;
                        for (int i = 1; i < MAX_TASKS; i++) begin
                            if (i < num_tasks && task_queue[i].burst_time < min_bt) begin
                                min_bt = task_queue[i].burst_time;
                                sel = ($clog2(MAX_TASKS))'(i);
                            end
                        end
                    end
                end

                3: begin // Round Robin
                    if (num_tasks > 0) begin
                        sel = (rr_ptr < num_tasks) ? rr_ptr : '0;
                    end
                end

                4: begin // Priority
                    if (num_tasks > 0) begin
                        max_pri = task_queue[0].task_priority;
                        sel = '0;
                        for (int i = 1; i < MAX_TASKS; i++) begin
                            if (i < num_tasks && task_queue[i].task_priority > max_pri) begin
                                max_pri = task_queue[i].task_priority;
                                sel = ($clog2(MAX_TASKS))'(i);
                            end
                        end
                    end
                end

                5: begin // EDF
                    if (num_tasks > 0) begin
                        min_dl = task_queue[0].deadline;
                        sel = '0;
                        for (int i = 1; i < MAX_TASKS; i++) begin
                            if (i < num_tasks && task_queue[i].deadline < min_dl) begin
                                min_dl = task_queue[i].deadline;
                                sel = ($clog2(MAX_TASKS))'(i);
                            end
                        end
                    end
                end

                6: begin // LRU
                    if (num_tasks > 0) begin
                        min_acc = task_queue[0].last_access_time;
                        sel = '0;
                        for (int i = 1; i < MAX_TASKS; i++) begin
                            if (i < num_tasks && task_queue[i].last_access_time < min_acc) begin
                                min_acc = task_queue[i].last_access_time;
                                sel = ($clog2(MAX_TASKS))'(i);
                            end
                        end
                    end
                end

                default: sel = '0;
            endcase
        end
    end

    // =========================================================================
    // UNIFIED QUEUE MANAGEMENT
    // =========================================================================
    logic inserting;
    logic removing;
    
    assign inserting = (task_valid && !queue_full);
    assign removing  = (task_complete && num_tasks > 0);

    int rem_idx;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            num_tasks <= '0;
            for (int i = 0; i < MAX_TASKS; i++) task_queue[i] <= '0;
        end else begin
            
            if (removing) begin
                rem_idx = -1;
                for (int i = 0; i < MAX_TASKS; i++) begin
                    if (i < num_tasks && task_queue[i].id == removing_id) begin
                        rem_idx = i;
                        break;
                    end
                end

                if (rem_idx != -1) begin
                    for (int j = 0; j < MAX_TASKS-1; j++) begin
                        if (j >= rem_idx && j < num_tasks - 1) begin
                            task_queue[j] <= task_queue[j+1];
                        end
                    end
                    
                    if (inserting) begin
                        task_queue[num_tasks-1].valid <= 1'b1;
                        task_queue[num_tasks-1].id <= task_id;
                        task_queue[num_tasks-1].burst_time <= burst_time;
                        task_queue[num_tasks-1].remaining_time <= burst_time;
                        task_queue[num_tasks-1].task_priority <= task_priority;
                        task_queue[num_tasks-1].deadline <= deadline;
                        task_queue[num_tasks-1].arrival_time <= current_time;
                        task_queue[num_tasks-1].last_access_time <= current_time;
                    end else begin
                        task_queue[num_tasks-1] <= '0;
                        num_tasks <= num_tasks - 1;
                    end
                end
            end else if (inserting) begin
                task_queue[num_tasks].valid <= 1'b1;
                task_queue[num_tasks].id <= task_id;
                task_queue[num_tasks].burst_time <= burst_time;
                task_queue[num_tasks].remaining_time <= burst_time;
                task_queue[num_tasks].task_priority <= task_priority;
                task_queue[num_tasks].deadline <= deadline;
                task_queue[num_tasks].arrival_time <= current_time;
                task_queue[num_tasks].last_access_time <= current_time;
                num_tasks <= num_tasks + 1;
            end

            if (scheduled_task_valid && SCHEDULER_TYPE == 6 && num_tasks > 0) begin
                for (int i = 0; i < MAX_TASKS; i++) begin
                    if (i < num_tasks && task_queue[i].id == scheduled_task_id) begin
                        task_queue[i].last_access_time <= current_time;
                        break;
                    end
                end
            end
        end
    end

    // =========================================================================
    // ROUND ROBIN POINTER
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                rr_ptr <= '0;
                quantum_left <= TIME_QUANTUM;
            end else if (SCHEDULER_TYPE == 3 && !queue_empty) begin
                
                if (task_complete) begin
                    // 1. Task finished early: Reset the quantum for the next task
                    quantum_left <= TIME_QUANTUM;
                    
                    // 2. Adjust Pointer: Since the queue array shifts left upon removal, 
                    // the next task naturally falls into the current rr_ptr index.
                    // We only need to adjust rr_ptr if we were pointing at the very end of the queue.
                    if (num_tasks > 1 && rr_ptr >= (num_tasks - 1)) begin
                        rr_ptr <= '0;
                    end
                    // Else: rr_ptr stays the same, which effectively selects the next task 
                    // that just shifted into this index!
    
                end else if (quantum_left > 1) begin
                    // Normal execution tick
                    quantum_left <= quantum_left - 1;
                    
                end else begin
                    // 3. Quantum Expired: Context Switch
                    quantum_left <= TIME_QUANTUM;
                    if (num_tasks > 0) begin
                        rr_ptr <= (rr_ptr + 1) % num_tasks;
                    end else begin
                        rr_ptr <= '0;
                    end
                end
                
            end else if (SCHEDULER_TYPE != 3) begin
                rr_ptr <= '0;
                quantum_left <= TIME_QUANTUM;
            end
        end
    
    // =========================================================================
    // OUTPUT REGISTER
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scheduled_task_valid_delayed <= 1'b0;
            scheduled_task_id_delayed <= '0;
            scheduled_burst_time_delayed <= '0;
            scheduled_priority_delayed <= '0;
            scheduled_deadline_delayed <= '0;
        end else if (task_complete) begin
            scheduled_task_valid_delayed <= 1'b0;
        end else if (!queue_empty && sel < num_tasks && task_queue[sel].valid) begin
            if (!scheduled_task_valid_delayed && inserting) begin
                scheduled_task_valid_delayed <= 1'b0;
            end else begin
                scheduled_task_valid_delayed <= 1'b1;
                scheduled_task_id_delayed <= task_queue[sel].id;
                scheduled_burst_time_delayed <= task_queue[sel].burst_time;
                scheduled_priority_delayed <= task_queue[sel].task_priority;
                scheduled_deadline_delayed <= task_queue[sel].deadline;
            end
        end else begin
            scheduled_task_valid_delayed <= 1'b0;
        end
    end

    assign scheduled_task_valid = scheduled_task_valid_delayed;
    assign scheduled_task_id = scheduled_task_id_delayed;
    assign scheduled_burst_time = scheduled_burst_time_delayed;
    assign scheduled_priority = scheduled_priority_delayed;
    assign scheduled_deadline = scheduled_deadline_delayed;

endmodule
