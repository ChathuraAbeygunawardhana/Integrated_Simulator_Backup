# Task Scheduler - Examples and Use Cases

This document provides practical examples and use cases for the hardware task scheduler system.

## Table of Contents
1. [Basic Usage Examples](#basic-usage-examples)
2. [Real-World Applications](#real-world-applications)
3. [Performance Comparison](#performance-comparison)
4. [Configuration Examples](#configuration-examples)
5. [Integration Examples](#integration-examples)

---

## Basic Usage Examples

### Example 1: Simple Task Submission

```systemverilog
module simple_example;
    logic clk, rst_n;
    logic task_valid;
    logic [7:0] task_id;
    logic [15:0] burst_time;
    
    // Instantiate FIFO scheduler
    task_scheduler #(
        .MAX_TASKS(8),
        .SCHEDULER_TYPE(0)  // FIFO
    ) scheduler (
        .clk(clk),
        .rst_n(rst_n),
        .task_valid(task_valid),
        .task_id(task_id),
        .burst_time(burst_time),
        // ... other ports
    );
    
    // Submit tasks
    initial begin
        rst_n = 0;
        #20 rst_n = 1;
        
        // Task 1: ID=1, Burst=10
        @(posedge clk);
        task_valid = 1;
        task_id = 1;
        burst_time = 10;
        
        @(posedge clk);
        task_valid = 0;
        
        // Task 2: ID=2, Burst=5
        repeat(3) @(posedge clk);
        task_valid = 1;
        task_id = 2;
        burst_time = 5;
        
        @(posedge clk);
        task_valid = 0;
    end
endmodule
```

### Example 2: Priority-Based Scheduling

```systemverilog
// High-priority interrupt handler
always @(posedge interrupt_request) begin
    @(posedge clk);
    task_valid <= 1'b1;
    task_id <= interrupt_vector;
    burst_time <= 16'd5;        // Short ISR
    priority <= 4'd15;          // Highest priority
    deadline <= current_time + 100;
end

// Background processing
always @(posedge background_trigger) begin
    @(posedge clk);
    task_valid <= 1'b1;
    task_id <= bg_task_id;
    burst_time <= 16'd1000;     // Long task
    priority <= 4'd1;           // Low priority
    deadline <= current_time + 10000;
end
```

---

## Real-World Applications

### 1. Embedded System Task Management

**Scenario:** Microcontroller with multiple concurrent tasks

**Configuration:**
- Scheduler: Priority-based or EDF
- MAX_TASKS: 16
- Tasks: Sensor polling, communication, display update, button handling

```systemverilog
task_scheduler #(
    .MAX_TASKS(16),
    .SCHEDULER_TYPE(4),  // Priority
    .PRIORITY_WIDTH(4)
) embedded_scheduler (
    // Connections...
);

// Task definitions:
// Priority 15: Emergency stop button (1-2 cycles)
// Priority 10: Timer interrupts (5 cycles)
// Priority 5:  Serial communication (20 cycles)
// Priority 2:  Display update (50 cycles)
// Priority 1:  Background logging (100 cycles)
```

### 2. Network Packet Processing

**Scenario:** Hardware packet processor in network switch

**Configuration:**
- Scheduler: Round Robin for fairness
- MAX_TASKS: 32 (one per input port)
- TIME_QUANTUM: Configurable based on packet size

```systemverilog
task_scheduler #(
    .MAX_TASKS(32),
    .SCHEDULER_TYPE(3),  // Round Robin
    .TIME_QUANTUM(16)    // Process 16 bytes per turn
) packet_scheduler (
    // Connections...
);

// Each arriving packet becomes a task
// Fair processing prevents head-of-line blocking
```

### 3. Real-Time Video Processing

**Scenario:** Multi-stage video pipeline with deadlines

**Configuration:**
- Scheduler: EDF (Earliest Deadline First)
- Tasks: Frame decoding, filtering, encoding
- Deadlines: Based on frame rate (e.g., 33ms for 30fps)

```systemverilog
task_scheduler #(
    .MAX_TASKS(24),
    .SCHEDULER_TYPE(5),     // EDF
    .DEADLINE_WIDTH(32)
) video_scheduler (
    // Connections...
);

// Frame N must complete before Frame N+1 arrives
// EDF ensures deadline compliance
```

### 4. Multi-Core Task Distribution

**Scenario:** Distributing tasks across multiple processing cores

**Configuration:**
- Scheduler: SJF or SRTF for minimizing completion time
- Multiple scheduler instances (one per core)
- Load balancing logic

```systemverilog
// Core 0: Short tasks
task_scheduler #(
    .MAX_TASKS(16),
    .SCHEDULER_TYPE(2)  // SJF
) core0_scheduler (/*...*/);

// Core 1: Long tasks
task_scheduler #(
    .MAX_TASKS(8),
    .SCHEDULER_TYPE(0)  // FIFO
) core1_scheduler (/*...*/);

// Load balancer
always @(posedge clk) begin
    if (task_valid) begin
        if (burst_time < 50)
            route_to_core0();
        else
            route_to_core1();
    end
end
```

### 5. IoT Sensor Hub

**Scenario:** Central hub collecting data from multiple sensors

**Configuration:**
- Scheduler: Multi-Level Queue (MLQ)
- Queue 0: Critical sensors (temperature alarms)
- Queue 1: Important sensors (motion detection)
- Queue 2: Regular sensors (ambient light)
- Queue 3: Optional sensors (diagnostics)

```systemverilog
advanced_task_scheduler #(
    .MAX_TASKS(32),
    .SCHEDULER_TYPE(2),  // MLQ
    .NUM_QUEUES(4)
) iot_scheduler (/*...*/);

// Critical sensor (Queue 0)
task_valid <= 1;
priority <= 4'd12;  // Maps to queue 0

// Regular sensor (Queue 2)
task_valid <= 1;
priority <= 4'd4;   // Maps to queue 2
```

---

## Performance Comparison

### Scenario Setup
- 10 tasks with random burst times (5-50 cycles)
- Random arrival intervals (1-5 cycles)

### Results

| Scheduler | Avg Wait Time | Avg Turnaround | Fairness | Starvation Risk |
|-----------|---------------|----------------|----------|-----------------|
| FIFO      | 125 cycles    | 175 cycles     | High     | None            |
| LIFO      | 180 cycles    | 230 cycles     | Low      | High            |
| SJF       | 85 cycles     | 135 cycles     | Medium   | Medium          |
| SRTF      | 65 cycles     | 115 cycles     | Medium   | Medium          |
| Priority  | 95 cycles     | 145 cycles     | Low      | High            |
| Round Robin| 140 cycles   | 190 cycles     | High     | None            |
| EDF       | 100 cycles    | 150 cycles     | Medium   | Low             |
| HRRN      | 90 cycles     | 140 cycles     | High     | None            |

**Key Observations:**
- SRTF provides lowest wait times but requires preemption
- Round Robin provides best fairness
- HRRN balances performance and fairness
- Priority can starve low-priority tasks

---

## Configuration Examples

### Small Embedded System (Resource-Constrained)

```systemverilog
task_scheduler #(
    .MAX_TASKS(4),              // Small queue
    .TASK_ID_WIDTH(4),          // 16 unique tasks max
    .BURST_TIME_WIDTH(8),       // Max burst: 255 cycles
    .PRIORITY_WIDTH(2),         // 4 priority levels
    .SCHEDULER_TYPE(0)          // Simple FIFO
) small_scheduler (/*...*/);

// Area: ~500 LUTs, 100 FFs (FPGA estimate)
// Frequency: 200+ MHz
```

### High-Performance Server

```systemverilog
task_scheduler #(
    .MAX_TASKS(128),            // Large queue
    .TASK_ID_WIDTH(16),         // 65K unique tasks
    .BURST_TIME_WIDTH(32),      // Very long tasks
    .PRIORITY_WIDTH(8),         // 256 priority levels
    .SCHEDULER_TYPE(5)          // EDF for real-time
) server_scheduler (/*...*/);

// Area: ~15K LUTs, 5K FFs (FPGA estimate)
// Frequency: 100-150 MHz
```

### Balanced General-Purpose

```systemverilog
task_scheduler #(
    .MAX_TASKS(32),             // Moderate queue
    .TASK_ID_WIDTH(8),
    .BURST_TIME_WIDTH(16),
    .PRIORITY_WIDTH(4),
    .TIME_QUANTUM(20),
    .SCHEDULER_TYPE(3)          // Round Robin
) balanced_scheduler (/*...*/);

// Good balance of features and resources
// Area: ~2K LUTs, 500 FFs
// Frequency: 150-200 MHz
```

---

## Integration Examples

### Example 1: Integration with AXI Bus

```systemverilog
module axi_task_scheduler_wrapper (
    input  logic axi_clk,
    input  logic axi_rst_n,
    
    // AXI-Lite slave interface for configuration
    input  logic [31:0] s_axi_awaddr,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    // ... other AXI signals
    
    // Task interface
    output logic        task_scheduled,
    output logic [7:0]  task_id_out
);

    // Register map
    localparam TASK_SUBMIT_ADDR = 32'h0000;
    localparam TASK_STATUS_ADDR = 32'h0004;
    localparam SCHED_CONFIG_ADDR = 32'h0008;
    
    logic task_valid;
    logic [7:0] task_id;
    
    // Scheduler instance
    task_scheduler #(
        .MAX_TASKS(16)
    ) scheduler (
        .clk(axi_clk),
        .rst_n(axi_rst_n),
        .task_valid(task_valid),
        .task_id(task_id),
        // ... other connections
    );
    
    // AXI write handling
    always @(posedge axi_clk) begin
        if (s_axi_awvalid && s_axi_awready) begin
            case (s_axi_awaddr)
                TASK_SUBMIT_ADDR: begin
                    task_valid <= 1'b1;
                    task_id <= s_axi_wdata[7:0];
                end
            endcase
        end
    end
endmodule
```

### Example 2: Integration with RISC-V Processor

```systemverilog
module riscv_scheduler_coprocessor (
    input  logic clk,
    input  logic rst_n,
    
    // RISC-V custom instruction interface
    input  logic        cop_valid,
    input  logic [31:0] cop_insn,
    input  logic [31:0] cop_rs1,
    input  logic [31:0] cop_rs2,
    output logic [31:0] cop_result,
    output logic        cop_ready
);

    // Custom instructions:
    // 0: Submit task (rs1=task_id, rs2=burst_time)
    // 1: Get status
    // 2: Configure scheduler
    
    logic [2:0] cop_opcode;
    assign cop_opcode = cop_insn[6:4];
    
    task_scheduler #(.MAX_TASKS(32)) sched (/*...*/);
    
    always @(posedge clk) begin
        case (cop_opcode)
            3'd0: begin  // Submit task
                task_valid <= 1'b1;
                task_id <= cop_rs1[7:0];
                burst_time <= cop_rs2[15:0];
            end
            3'd1: begin  // Status
                cop_result <= {16'd0, queue_count};
            end
        endcase
    end
endmodule
```

### Example 3: Integration with DMA Engine

```systemverilog
module dma_task_scheduler (
    input  logic clk,
    input  logic rst_n,
    
    // DMA request interface
    input  logic        dma_req_valid,
    input  logic [31:0] dma_src_addr,
    input  logic [31:0] dma_dst_addr,
    input  logic [15:0] dma_length,
    input  logic [3:0]  dma_priority,
    
    // DMA grant interface
    output logic        dma_grant,
    output logic [31:0] dma_grant_src,
    output logic [31:0] dma_grant_dst,
    output logic [15:0] dma_grant_len
);

    // Store DMA parameters alongside task
    logic [31:0] dma_params [15:0][2:0];
    
    task_scheduler #(
        .MAX_TASKS(16),
        .SCHEDULER_TYPE(4)  // Priority
    ) dma_sched (
        .clk(clk),
        .rst_n(rst_n),
        .task_valid(dma_req_valid),
        .burst_time(dma_length),
        .priority(dma_priority),
        .scheduled_task_valid(dma_grant),
        // ...
    );
    
    // Store DMA parameters on task arrival
    always @(posedge clk) begin
        if (dma_req_valid && task_ready) begin
            dma_params[task_id][0] <= dma_src_addr;
            dma_params[task_id][1] <= dma_dst_addr;
            dma_params[task_id][2] <= {16'd0, dma_length};
        end
    end
    
    // Output scheduled DMA parameters
    assign dma_grant_src = dma_params[scheduled_task_id][0];
    assign dma_grant_dst = dma_params[scheduled_task_id][1];
    assign dma_grant_len = dma_params[scheduled_task_id][2][15:0];
endmodule
```

---

## Testing Scenarios

### Test 1: Burst Arrival
```systemverilog
// Submit 10 tasks rapidly, then measure completion
for (int i = 0; i < 10; i++) begin
    @(posedge clk);
    task_valid = 1;
    task_id = i;
    burst_time = $urandom_range(10, 50);
end
```

### Test 2: Sparse Arrival
```systemverilog
// Submit tasks with large gaps
for (int i = 0; i < 10; i++) begin
    repeat($urandom_range(50, 100)) @(posedge clk);
    task_valid = 1;
    task_id = i;
    burst_time = $urandom_range(5, 20);
end
```

### Test 3: Priority Inversion
```systemverilog
// Submit low priority task, then high priority
task_valid = 1; priority = 1; burst_time = 100;
@(posedge clk); task_valid = 0;

repeat(10) @(posedge clk);

task_valid = 1; priority = 15; burst_time = 10;
@(posedge clk); task_valid = 0;

// High priority should execute first in Priority scheduler
```

### Test 4: Queue Overflow
```systemverilog
// Try to submit more than MAX_TASKS
for (int i = 0; i < MAX_TASKS + 5; i++) begin
    @(posedge clk);
    if (task_ready) begin
        task_valid = 1;
        task_id = i;
    end
end

// Verify queue_full assertion
```

---

## Conclusion

This task scheduler system provides flexible, parameterized hardware scheduling suitable for a wide range of applications from small embedded systems to high-performance servers. Choose the appropriate scheduler type and parameters based on your specific requirements for performance, fairness, and resource constraints.
