// ============================================================================
// Task Scheduler Testbench
// ============================================================================

`timescale 1ns/1ps

module task_scheduler_tb;

    parameter CLK_PERIOD       = 10;
    parameter MAX_TASKS        = 16;
    parameter TASK_ID_WIDTH    = 8;
    parameter BURST_TIME_WIDTH = 16;
    parameter PRIORITY_WIDTH   = 4;
    parameter DEADLINE_WIDTH   = 32;
    parameter TIME_QUANTUM     = 10;
    parameter NUM_TEST_TASKS   = 10;

    logic clk;
    logic rst_n;
    logic [3:0] scheduler_select;
    logic task_valid;
    logic [TASK_ID_WIDTH-1:0] task_id;
    logic [BURST_TIME_WIDTH-1:0] burst_time;
    logic [PRIORITY_WIDTH-1:0] task_priority;
    logic [DEADLINE_WIDTH-1:0] deadline;
    logic task_ready;
    logic scheduled_task_valid;
    logic [TASK_ID_WIDTH-1:0] scheduled_task_id;
    logic [BURST_TIME_WIDTH-1:0] scheduled_burst_time;
    logic [PRIORITY_WIDTH-1:0] scheduled_priority;
    logic [DEADLINE_WIDTH-1:0] scheduled_deadline;
    logic task_tick;
    logic task_complete;
    logic [$clog2(MAX_TASKS):0] queue_count;
    logic queue_full;
    logic queue_empty;
    logic [31:0] total_tasks_processed;
    logic [31:0] total_wait_time;
    logic [31:0] total_turnaround_time;

    multi_scheduler_wrapper #(
        .MAX_TASKS        (MAX_TASKS),
        .TASK_ID_WIDTH    (TASK_ID_WIDTH),
        .BURST_TIME_WIDTH (BURST_TIME_WIDTH),
        .PRIORITY_WIDTH   (PRIORITY_WIDTH),
        .DEADLINE_WIDTH   (DEADLINE_WIDTH),
        .TIME_QUANTUM     (TIME_QUANTUM)
    ) dut (.*);

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) task_tick <= 1'b0;
        else        task_tick <= scheduled_task_valid && !task_complete;
    end

    function automatic string get_scheduler_name(int sched_type);
        case (sched_type)
            0:  return "FIFO";
            1:  return "LIFO";
            2:  return "SJF";
            3:  return "Round Robin";
            4:  return "Priority";
            5:  return "EDF";
            6:  return "LRU";
            7:  return "SRTF";
            8:  return "HRRN";
            9:  return "MLQ";
            10: return "MLFQ";
            default: return "Unknown";
        endcase
    endfunction

    initial begin
        $display("====================================================");
        $display("Task Scheduler Testbench");
        $display("====================================================");
        
        rst_n            = 0;
        task_valid       = 0;
        task_id          = 0;
        burst_time       = 0;
        task_priority    = 0;
        deadline         = 0;
        task_complete    = 0;
        scheduler_select = 0;

        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        for (int s = 0; s <= 10; s++)
            test_scheduler(get_scheduler_name(s), s);

        $display("\n====================================================");
        $display("All scheduler tests completed successfully!");
        $display("====================================================");
        $finish;
    end

    task automatic test_scheduler(string name, int sched_type);
        automatic int local_submitted = 0;
        automatic int local_completed = 0;
        
        // Use an array to track remaining cycles for ALL tasks safely 
        // across context switches (fixes LRU, RR, and MLFQ infinite loops)
        automatic int tb_rem_cycles[256];
        automatic int last_exec_id = -1;

        for (int i = 0; i < 256; i++) tb_rem_cycles[i] = 0;

        $display("\n*** Testing %s Scheduler (select=%0d) ***", name, sched_type);

        @(posedge clk);
        rst_n            = 0;
        scheduler_select = 4'(sched_type);
        repeat(3) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        fork
            // Thread 1 - Task submission
            begin
                for (int i = 0; i < NUM_TEST_TASKS; i++) begin
                    repeat($urandom_range(1, 5)) @(posedge clk);
                    while (queue_full) @(posedge clk);

                    @(posedge clk);
                    task_valid    = 1;
                    task_id       = 8'(i);
                    burst_time    = $urandom_range(5, 20);
                    task_priority = $urandom_range(0, 15);
                    deadline      = $urandom_range(100, 500);

                    // Register initial burst time for independent tracking
                    tb_rem_cycles[int'(task_id)] = int'(burst_time);

                    $display("[%0t] %s | Submit  Task %0d  burst=%0d  pri=%0d  dl=%0d",
                             $time, name, task_id, burst_time, task_priority, deadline);

                    @(posedge clk);
                    task_valid = 0;
                    local_submitted++;
                end
            end

            // Thread 2 - Task execution / completion
            begin
                while (local_completed < NUM_TEST_TASKS) begin
                    @(posedge clk);

                    if (scheduled_task_valid && !task_complete) begin
                        automatic int id = int'(scheduled_task_id);

                        if (id != last_exec_id) begin
                            $display("[%0t] %s | Execute Task %0d  (rem=%0d)", 
                                     $time, name, id, tb_rem_cycles[id]);
                            last_exec_id = id;
                        end

                        if (tb_rem_cycles[id] > 0) begin
                            tb_rem_cycles[id]--;
                        end

                        if (tb_rem_cycles[id] == 0) begin
                            task_complete = 1;
                            $display("[%0t] %s | Complete Task %0d", $time, name, id);
                            local_completed++;
                            tb_rem_cycles[id] = -1; // Prevent double-triggering
                            last_exec_id = -1;
                        end
                    end else if (task_complete) begin
                        // Wait exactly 1 cycle for hardware to process completion
                        task_complete = 0;
                    end else if (local_submitted == NUM_TEST_TASKS && queue_empty) begin
                        // Failsafe break
                        break;
                    end
                end
                task_complete = 0;
            end
        join 

        $display("");
        $display("--- %s Summary ---", name);
        $display("  Submitted        : %0d", local_submitted);
        $display("  Completed        : %0d", local_completed);
        $display("  Queue remaining  : %0d", queue_count);
        $display("  Tasks processed  : %0d", total_tasks_processed);
        
        repeat(20) @(posedge clk);
    endtask

    initial begin
        $dumpfile("task_scheduler.vcd");
        $dumpvars(0, task_scheduler_tb);
    end

    initial begin
        #10_000_000; 
        $display("ERROR: Simulation timeout - check for deadlock in scheduler logic.");
        $finish;
    end

endmodule
