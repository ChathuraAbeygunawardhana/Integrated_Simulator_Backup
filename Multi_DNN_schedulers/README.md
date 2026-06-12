# Hardware Task Scheduler System

A comprehensive, parameterized hardware task scheduler implementation supporting multiple scheduling algorithms with random task arrivals.

## Overview

This project implements various task scheduling algorithms in hardware (SystemVerilog) with full parameterization, allowing customization of queue sizes, task properties, and scheduling policies.

## Supported Scheduling Algorithms

### Basic Schedulers (task_scheduler.sv)

1. **FIFO (First In First Out)**
   - Tasks are executed in the order they arrive
   - Simple and fair, but may cause long wait times for short tasks
   - Parameter: SCHEDULER_TYPE = 0

2. **LIFO (Last In First Out)**
   - Most recently arrived task is executed first
   - Stack-based scheduling
   - Parameter: SCHEDULER_TYPE = 1

3. **SJF (Shortest Job First)**
   - Task with shortest burst time executes first
   - Minimizes average waiting time
   - Non-preemptive version
   - Parameter: SCHEDULER_TYPE = 2

4. **Round Robin (RR)**
   - Each task gets a fixed time quantum
   - After quantum expires, task moves to end of queue
   - Fair allocation of CPU time
   - Parameter: SCHEDULER_TYPE = 3

5. **Priority Scheduling**
   - Tasks with higher priority values execute first
   - Can lead to starvation of low-priority tasks
   - Parameter: SCHEDULER_TYPE = 4

6. **EDF (Earliest Deadline First)**
   - Task with earliest deadline executes first
   - Optimal for real-time systems
   - Parameter: SCHEDULER_TYPE = 5

7. **LRU (Least Recently Used)**
   - Task that hasn't been accessed for longest time executes first
   - Tracks last access time for each task
   - Parameter: SCHEDULER_TYPE = 6

### Advanced Schedulers (advanced_task_scheduler.sv)

8. **SRTF (Shortest Remaining Time First)**
   - Preemptive version of SJF
   - Switches to task with shortest remaining time
   - Parameter: SCHEDULER_TYPE = 0

9. **HRRN (Highest Response Ratio Next)**
   - Response Ratio = (Wait Time + Burst Time) / Burst Time
   - Prevents starvation while favoring short jobs
   - Parameter: SCHEDULER_TYPE = 1

10. **MLQ (Multi-Level Queue)**
    - Static priority queues
    - Tasks remain in assigned queue
    - Higher queues have absolute priority
    - Parameter: SCHEDULER_TYPE = 2

11. **MLFQ (Multi-Level Feedback Queue)**
    - Dynamic priority adjustment
    - Tasks demoted on preemption, promoted on aging
    - Balances fairness and performance
    - Parameter: SCHEDULER_TYPE = 3

## Module Parameters

### Common Parameters

```systemverilog
parameter MAX_TASKS = 16              // Maximum tasks in queue
parameter TASK_ID_WIDTH = 8           // Bits for task ID
parameter BURST_TIME_WIDTH = 16       // Bits for burst/execution time
parameter PRIORITY_WIDTH = 4          // Bits for priority field
parameter DEADLINE_WIDTH = 32         // Bits for deadline field
parameter TIME_QUANTUM = 10           // Time quantum for Round Robin
parameter SCHEDULER_TYPE = 0          // Selects scheduling algorithm
```

## Interface Signals

### Input Ports

| Signal | Width | Description |
|--------|-------|-------------|
| clk | 1 | System clock |
| rst_n | 1 | Active-low reset |
| task_valid | 1 | New task arrival strobe |
| task_id | TASK_ID_WIDTH | Unique task identifier |
| burst_time | BURST_TIME_WIDTH | Task execution time |
| priority | PRIORITY_WIDTH | Task priority level |
| deadline | DEADLINE_WIDTH | Task deadline |
| task_complete | 1 | Current task completion signal |
| task_tick | 1 | Single time unit execution (for SRTF) |

### Output Ports

| Signal | Width | Description |
|--------|-------|-------------|
| task_ready | 1 | Scheduler ready to accept tasks |
| scheduled_task_valid | 1 | Valid task scheduled for execution |
| scheduled_task_id | TASK_ID_WIDTH | ID of scheduled task |
| scheduled_burst_time | BURST_TIME_WIDTH | Burst time of scheduled task |
| scheduled_priority | PRIORITY_WIDTH | Priority of scheduled task |
| scheduled_deadline | DEADLINE_WIDTH | Deadline of scheduled task |
| queue_count | log2(MAX_TASKS)+1 | Number of tasks in queue |
| queue_full | 1 | Queue full indicator |
| queue_empty | 1 | Queue empty indicator |

## File Structure

```
.
├── task_scheduler.sv              # Main scheduler (7 algorithms)
├── advanced_task_scheduler.sv     # Advanced schedulers (4 algorithms)
├── multi_scheduler_wrapper.sv     # Runtime-selectable wrapper
├── task_scheduler_tb.sv           # Comprehensive testbench
└── README.md                      # This file
```

## Usage Examples

### Instantiating a FIFO Scheduler

```systemverilog
task_scheduler #(
    .MAX_TASKS(32),
    .TASK_ID_WIDTH(8),
    .BURST_TIME_WIDTH(16),
    .PRIORITY_WIDTH(4),
    .DEADLINE_WIDTH(32),
    .SCHEDULER_TYPE(0)  // FIFO
) fifo_scheduler (
    .clk(clk),
    .rst_n(rst_n),
    .task_valid(new_task_valid),
    .task_id(new_task_id),
    .burst_time(new_task_burst),
    .priority(new_task_priority),
    .deadline(new_task_deadline),
    .task_ready(can_accept_task),
    .scheduled_task_valid(exec_task_valid),
    .scheduled_task_id(exec_task_id),
    .task_complete(task_done),
    // ... other connections
);
```

### Instantiating a Round Robin Scheduler

```systemverilog
task_scheduler #(
    .MAX_TASKS(16),
    .TIME_QUANTUM(20),
    .SCHEDULER_TYPE(3)  // Round Robin
) rr_scheduler (
    // ... connections
);
```

### Using the Multi-Scheduler Wrapper

```systemverilog
multi_scheduler_wrapper #(
    .MAX_TASKS(16)
) multi_sched (
    .clk(clk),
    .rst_n(rst_n),
    .scheduler_select(4'd2),  // Select SJF
    .task_valid(task_valid),
    // ... other connections
);

// Change to Priority scheduler at runtime
scheduler_select <= 4'd4;
```

## Simulation

### Running the Testbench

The testbench (`task_scheduler_tb.sv`) tests all scheduling algorithms with random task arrivals:

```bash
# Using ModelSim/QuestaSim
vlog task_scheduler.sv advanced_task_scheduler.sv task_scheduler_tb.sv
vsim -c task_scheduler_tb -do "run -all"

# Using Icarus Verilog
iverilog -g2012 -o sim task_scheduler.sv advanced_task_scheduler.sv task_scheduler_tb.sv
vvp sim

# Using Verilator
verilator --binary --trace -Wall task_scheduler_tb.sv task_scheduler.sv advanced_task_scheduler.sv
./obj_dir/Vtask_scheduler_tb
```

### Testbench Features

- Random task arrival times (1-5 cycle intervals)
- Random burst times (5-50 cycles)
- Random priorities (0-15)
- Random deadlines (100-1000)
- Tests 50 tasks per scheduler
- Displays task submission and completion
- Generates VCD waveform file

## Performance Metrics

The wrapper module automatically tracks:
- Total tasks processed
- Total wait time (sum across all tasks)
- Total turnaround time (sum across all tasks)

### Calculating Metrics

```
Average Wait Time = total_wait_time / total_tasks_processed
Average Turnaround Time = total_turnaround_time / total_tasks_processed
```

## Design Considerations

### Queue Management

- Tasks stored in array structure
- Dynamic insertion and removal
- Automatic queue shifting on task completion
- Full/empty detection

### Preemption Support

- SRTF supports task preemption via `task_tick` signal
- Tasks maintain remaining time counter
- Can switch tasks mid-execution

### Timing

- All operations synchronized to clock
- Single-cycle task acceptance (when not full)
- Single-cycle scheduling decision
- Configurable execution time per task

## Customization

### Adding a New Scheduler

1. Add new case in scheduling logic
2. Implement selection algorithm
3. Update SCHEDULER_TYPE parameter documentation
4. Add test case in testbench

Example template:

```systemverilog
task schedule_custom();
    automatic int selected_idx = 0;
    
    // Your selection algorithm here
    for (int i = 0; i < num_tasks; i++) begin
        // Compare and select
    end
    
    // Output selected task
    scheduled_task_valid <= 1'b1;
    scheduled_task_id <= task_queue[selected_idx].id;
    // ... other outputs
endtask
```

### Increasing Queue Size

Simply change the `MAX_TASKS` parameter:

```systemverilog
task_scheduler #(.MAX_TASKS(64)) large_scheduler (...);
```

## Synthesis Considerations

- Designed for FPGA/ASIC implementation
- Parameterized for area/performance tradeoffs
- Larger MAX_TASKS increases area linearly
- Complex schedulers (HRRN, MLFQ) require more logic
- Consider pipeline stages for high-frequency designs

## Known Limitations

1. HRRN uses real arithmetic (simulation only - needs fixed-point for synthesis)
2. No task priority aging in basic Priority scheduler (can starve)
3. Round Robin quantum is compile-time constant
4. MLQ queue assignment is based on initial priority only

## Future Enhancements

- [ ] Fixed-point arithmetic for HRRN
- [ ] Dynamic time quantum for Round Robin
- [ ] Task preemption for Priority scheduler
- [ ] Aging mechanism to prevent starvation
- [ ] Performance counter interface
- [ ] Multi-core scheduling support
- [ ] Power-aware scheduling

## License

This is an educational/reference implementation. Use freely for learning and projects.

## References

- Operating System Concepts (Silberschatz, Galvin, Gagne)
- Modern Operating Systems (Tanenbaum)
- Real-Time Systems (Liu)
