//=============================================================================
// Module: extended_memory_controller
// Description: Extended memory controller with prefetching and inter-layer
//              optimization support
//
// New Features:
// 1. Prefetch engine that overlaps data movement with compute
// 2. Inter-layer activation reuse tracking
// 3. Dual-port scratchpad for concurrent prefetch and compute access
// 4. Priority-based prefetch scheduling
//=============================================================================

module extended_memory_controller
    import systolic_array_pkg::*;
    import stamp_memory_pkg::*;
#(
    parameter int ADDR_WIDTH = 32,
    parameter int DATA_WIDTH = 32,
    parameter int SPAD_DEPTH = 4096,
    parameter int PREFETCH_BUFFER_DEPTH = 1024,
    parameter int AXI_DATA_WIDTH = 64,
    parameter int MAX_DELTA_OPS = 64,
    parameter int METADATA_DEPTH = 256,
    parameter int MAX_PREFETCH_OPS = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    //=========================================================================
    // Configuration Interface
    //=========================================================================
    input  logic                    metadata_wr_en,
    input  logic [$clog2(METADATA_DEPTH)-1:0] metadata_wr_addr,
    input  logic [127:0]            metadata_wr_data,
    
    // Prefetch configuration
    input  logic                    prefetch_wr_en,
    input  logic [$clog2(MAX_PREFETCH_OPS)-1:0] prefetch_wr_addr,
    input  logic [127:0]            prefetch_wr_data,  // {priority, start_phase, complete_phase, tile_id, dst_addr, size}
    
    // Phase control
    input  logic                    phase_start,
    input  logic [15:0]             phase_id,
    input  logic [15:0]             num_delta_ops,
    input  logic [15:0]             num_prefetch_ops,
    output logic                    phase_ready,
    output logic                    phase_done,
    
    //=========================================================================
    // Scratchpad Interface (Dual-Port)
    //=========================================================================
    // Port A: Delta operations
    output logic [$clog2(SPAD_DEPTH)-1:0] spad_rd_addr_a,
    input  logic [DATA_WIDTH-1:0]         spad_rd_data_a,
    output logic                          spad_rd_en_a,
    output logic [$clog2(SPAD_DEPTH)-1:0] spad_wr_addr_a,
    output logic [DATA_WIDTH-1:0]         spad_wr_data_a,
    output logic                          spad_wr_en_a,
    
    // Port B: Prefetch operations
    output logic [$clog2(SPAD_DEPTH)-1:0] spad_rd_addr_b,
    input  logic [DATA_WIDTH-1:0]         spad_rd_data_b,
    output logic                          spad_rd_en_b,
    output logic [$clog2(SPAD_DEPTH)-1:0] spad_wr_addr_b,
    output logic [DATA_WIDTH-1:0]         spad_wr_data_b,
    output logic                          spad_wr_en_b,
    
    //=========================================================================
    // Prefetch Buffer Interface
    //=========================================================================
    output logic [$clog2(PREFETCH_BUFFER_DEPTH)-1:0] prefetch_wr_addr,
    output logic [DATA_WIDTH-1:0]                     prefetch_wr_data,
    output logic                                      prefetch_wr_en,
    
    //=========================================================================
    // Off-chip Memory Interface (Dual-channel for prefetch)
    //=========================================================================
    // Channel 0: Delta operations
    output logic [ADDR_WIDTH-1:0]   mem_rd_addr_0,
    output logic [15:0]             mem_rd_len_0,
    output logic                    mem_rd_req_0,
    input  logic                    mem_rd_ready_0,
    input  logic [DATA_WIDTH-1:0]   mem_rd_data_0,
    input  logic                    mem_rd_valid_0,
    output logic                    mem_rd_ready_ack_0,
    
    // Channel 1: Prefetch operations
    output logic [ADDR_WIDTH-1:0]   mem_rd_addr_1,
    output logic [15:0]             mem_rd_len_1,
    output logic                    mem_rd_req_1,
    input  logic                    mem_rd_ready_1,
    input  logic [DATA_WIDTH-1:0]   mem_rd_data_1,
    input  logic                    mem_rd_valid_1,
    output logic                    mem_rd_ready_ack_1,
    
    //=========================================================================
    // Inter-Layer Reuse Tracking
    //=========================================================================
    input  logic [15:0]             active_layer_id,
    output logic                    inter_layer_reuse_hit,
    output logic [31:0]             inter_layer_bytes_saved,
    
    //=========================================================================
    // Status and Statistics
    //=========================================================================
    output logic [31:0]             stats_loads,
    output logic [31:0]             stats_moves,
    output logic [31:0]             stats_keeps,
    output logic [31:0]             stats_prefetches,
    output logic [31:0]             stats_prefetch_hits,
    output logic [31:0]             stats_bytes_loaded,
    output logic [31:0]             stats_bytes_moved,
    output logic [31:0]             stats_bytes_prefetched,
    output logic                    controller_busy,
    output logic                    prefetch_engine_busy
);

    //=========================================================================
    // Metadata Storage
    //=========================================================================
    typedef struct packed {
        logic [7:0]  op_type;
        logic [15:0] tile_id;
        logic [31:0] src_addr;
        logic [31:0] dst_addr;
        logic [31:0] size;
    } delta_operation_t;
    
    typedef struct packed {
        logic [7:0]  priority;
        logic [15:0] start_phase;
        logic [15:0] complete_phase;
        logic [15:0] tile_id;
        logic [31:0] dst_addr;
        logic [31:0] size;
    } prefetch_operation_t;
    
    delta_operation_t metadata_ram [METADATA_DEPTH];
    prefetch_operation_t prefetch_ram [MAX_PREFETCH_OPS];
    
    // Write to metadata RAM
    always_ff @(posedge clk) begin
        if (metadata_wr_en) begin
            metadata_ram[metadata_wr_addr] <= metadata_wr_data;
        end
        if (prefetch_wr_en) begin
            prefetch_ram[prefetch_wr_addr] <= prefetch_wr_data;
        end
    end
    
    //=========================================================================
    // Delta Controller (Original)
    //=========================================================================
    typedef enum logic [3:0] {
        DELTA_IDLE,
        DELTA_FETCH_METADATA,
        DELTA_PROCESS_KEEP,
        DELTA_PROCESS_MOVE_SETUP,
        DELTA_PROCESS_MOVE_EXEC,
        DELTA_PROCESS_LOAD_SETUP,
        DELTA_PROCESS_LOAD_EXEC,
        DELTA_WAIT_LOAD_COMPLETE,
        DELTA_DONE
    } delta_state_t;
    
    delta_state_t delta_state, delta_next_state;
    
    logic [15:0] delta_op_idx;
    logic [15:0] delta_num_ops;
    delta_operation_t delta_current_op;
    logic [31:0] delta_bytes_remaining;
    logic [31:0] delta_current_src_addr;
    logic [31:0] delta_current_dst_addr;
    logic [15:0] delta_words_remaining;
    
    //=========================================================================
    // Prefetch Engine
    //=========================================================================
    typedef enum logic [3:0] {
        PREFETCH_IDLE,
        PREFETCH_CHECK_SCHEDULE,
        PREFETCH_ISSUE_REQUEST,
        PREFETCH_RECEIVE_DATA,
        PREFETCH_COMPLETE
    } prefetch_state_t;
    
    prefetch_state_t prefetch_state, prefetch_next_state;
    
    logic [15:0] prefetch_op_idx;
    prefetch_operation_t prefetch_current_op;
    logic [31:0] prefetch_bytes_remaining;
    logic [31:0] prefetch_current_dst_addr;
    logic [15:0] prefetch_words_remaining;
    logic [15:0] current_phase_id;
    
    // Prefetch active table (track in-flight prefetches)
    logic [MAX_PREFETCH_OPS-1:0] prefetch_active;
    logic [MAX_PREFETCH_OPS-1:0] prefetch_completed;
    
    //=========================================================================
    // Inter-Layer Reuse Tracking
    //=========================================================================
    // Simple cache of recently produced activations
    typedef struct packed {
        logic        valid;
        logic [15:0] layer_id;
        logic [15:0] tile_id;
        logic [31:0] addr;
        logic [31:0] size;
    } activation_cache_entry_t;
    
    localparam int ACTIVATION_CACHE_SIZE = 16;
    activation_cache_entry_t activation_cache [ACTIVATION_CACHE_SIZE];
    
    logic reuse_hit;
    logic [31:0] reuse_addr;
    
    //=========================================================================
    // Statistics
    //=========================================================================
    logic [31:0] loads_count;
    logic [31:0] moves_count;
    logic [31:0] keeps_count;
    logic [31:0] prefetches_count;
    logic [31:0] prefetch_hits_count;
    logic [31:0] bytes_loaded_count;
    logic [31:0] bytes_moved_count;
    logic [31:0] bytes_prefetched_count;
    logic [31:0] inter_layer_bytes_saved_count;
    
    //=========================================================================
    // Delta State Machine
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            delta_state <= DELTA_IDLE;
        end else begin
            delta_state <= delta_next_state;
        end
    end
    
    always_comb begin
        delta_next_state = delta_state;
        
        case (delta_state)
            DELTA_IDLE: begin
                if (phase_start) begin
                    delta_next_state = DELTA_FETCH_METADATA;
                end
            end
            
            DELTA_FETCH_METADATA: begin
                if (delta_op_idx < delta_num_ops) begin
                    case (delta_current_op.op_type)
                        8'd0: delta_next_state = DELTA_PROCESS_KEEP;
                        8'd1: delta_next_state = DELTA_PROCESS_MOVE_SETUP;
                        8'd2: delta_next_state = DELTA_PROCESS_LOAD_SETUP;
                        default: delta_next_state = DELTA_FETCH_METADATA;
                    endcase
                end else begin
                    delta_next_state = DELTA_DONE;
                end
            end
            
            DELTA_PROCESS_KEEP: begin
                delta_next_state = DELTA_FETCH_METADATA;
            end
            
            DELTA_PROCESS_MOVE_SETUP: begin
                delta_next_state = DELTA_PROCESS_MOVE_EXEC;
            end
            
            DELTA_PROCESS_MOVE_EXEC: begin
                if (delta_words_remaining == 0) begin
                    delta_next_state = DELTA_FETCH_METADATA;
                end
            end
            
            DELTA_PROCESS_LOAD_SETUP: begin
                // Check if data was prefetched
                if (reuse_hit) begin
                    // Data available from prefetch/reuse
                    delta_next_state = DELTA_FETCH_METADATA;
                end else begin
                    delta_next_state = DELTA_PROCESS_LOAD_EXEC;
                end
            end
            
            DELTA_PROCESS_LOAD_EXEC: begin
                if (mem_rd_ready_0) begin
                    delta_next_state = DELTA_WAIT_LOAD_COMPLETE;
                end
            end
            
            DELTA_WAIT_LOAD_COMPLETE: begin
                if (delta_words_remaining == 0) begin
                    delta_next_state = DELTA_FETCH_METADATA;
                end
            end
            
            DELTA_DONE: begin
                delta_next_state = DELTA_IDLE;
            end
        endcase
    end
    
    //=========================================================================
    // Prefetch State Machine
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prefetch_state <= PREFETCH_IDLE;
        end else begin
            prefetch_state <= prefetch_next_state;
        end
    end
    
    always_comb begin
        prefetch_next_state = prefetch_state;
        
        case (prefetch_state)
            PREFETCH_IDLE: begin
                if (current_phase_id < num_prefetch_ops) begin
                    prefetch_next_state = PREFETCH_CHECK_SCHEDULE;
                end
            end
            
            PREFETCH_CHECK_SCHEDULE: begin
                // Check if any prefetch should start at current phase
                if (prefetch_current_op.start_phase == current_phase_id &&
                    !prefetch_active[prefetch_op_idx]) begin
                    prefetch_next_state = PREFETCH_ISSUE_REQUEST;
                end else if (prefetch_op_idx < num_prefetch_ops - 1) begin
                    // Check next prefetch
                    prefetch_next_state = PREFETCH_CHECK_SCHEDULE;
                end else begin
                    prefetch_next_state = PREFETCH_IDLE;
                end
            end
            
            PREFETCH_ISSUE_REQUEST: begin
                if (mem_rd_ready_1) begin
                    prefetch_next_state = PREFETCH_RECEIVE_DATA;
                end
            end
            
            PREFETCH_RECEIVE_DATA: begin
                if (prefetch_words_remaining == 0) begin
                    prefetch_next_state = PREFETCH_COMPLETE;
                end
            end
            
            PREFETCH_COMPLETE: begin
                prefetch_next_state = PREFETCH_CHECK_SCHEDULE;
            end
        endcase
    end
    
    //=========================================================================
    // Prefetch Logic
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prefetch_op_idx <= 0;
            prefetch_active <= '0;
            prefetch_completed <= '0;
            prefetches_count <= 0;
            bytes_prefetched_count <= 0;
        end else begin
            case (prefetch_state)
                PREFETCH_CHECK_SCHEDULE: begin
                    if (prefetch_op_idx < num_prefetch_ops) begin
                        prefetch_current_op <= prefetch_ram[prefetch_op_idx];
                    end
                    
                    // Move to next prefetch operation
                    if (prefetch_next_state == PREFETCH_CHECK_SCHEDULE) begin
                        prefetch_op_idx <= prefetch_op_idx + 1;
                    end
                end
                
                PREFETCH_ISSUE_REQUEST: begin
                    prefetch_active[prefetch_op_idx] <= 1'b1;
                    prefetch_words_remaining <= prefetch_current_op.size / (DATA_WIDTH/8);
                    prefetch_current_dst_addr <= prefetch_current_op.dst_addr;
                    prefetches_count <= prefetches_count + 1;
                end
                
                PREFETCH_RECEIVE_DATA: begin
                    if (mem_rd_valid_1) begin
                        prefetch_words_remaining <= prefetch_words_remaining - 1;
                        prefetch_current_dst_addr <= prefetch_current_dst_addr + (DATA_WIDTH/8);
                        bytes_prefetched_count <= bytes_prefetched_count + (DATA_WIDTH/8);
                    end
                end
                
                PREFETCH_COMPLETE: begin
                    prefetch_completed[prefetch_op_idx] <= 1'b1;
                end
            endcase
            
            // Reset prefetch state at phase boundary
            if (phase_start) begin
                prefetch_op_idx <= 0;
            end
        end
    end
    
    //=========================================================================
    // Check for Prefetch/Reuse Hits
    //=========================================================================
    always_comb begin
        reuse_hit = 1'b0;
        reuse_addr = 32'h0;
        
        // Check if current load operation can be satisfied by prefetch
        if (delta_state == DELTA_PROCESS_LOAD_SETUP) begin
            for (int i = 0; i < MAX_PREFETCH_OPS; i++) begin
                if (prefetch_completed[i] &&
                    prefetch_ram[i].tile_id == delta_current_op.tile_id) begin
                    reuse_hit = 1'b1;
                    reuse_addr = prefetch_ram[i].dst_addr;
                    break;
                end
            end
        end
        
        // Check activation cache for inter-layer reuse
        if (!reuse_hit) begin
            for (int i = 0; i < ACTIVATION_CACHE_SIZE; i++) begin
                if (activation_cache[i].valid &&
                    activation_cache[i].tile_id == delta_current_op.tile_id) begin
                    reuse_hit = 1'b1;
                    reuse_addr = activation_cache[i].addr;
                    break;
                end
            end
        end
    end
    
    //=========================================================================
    // Update Activation Cache
    //=========================================================================
    logic [3:0] cache_lru_ptr;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < ACTIVATION_CACHE_SIZE; i++) begin
                activation_cache[i].valid <= 1'b0;
            end
            cache_lru_ptr <= 0;
        end else begin
            // Add new activation to cache when output is produced
            if (delta_state == DELTA_PROCESS_LOAD_SETUP && 
                delta_current_op.op_type == 8'd2) begin  // LOAD
                activation_cache[cache_lru_ptr].valid <= 1'b1;
                activation_cache[cache_lru_ptr].layer_id <= active_layer_id;
                activation_cache[cache_lru_ptr].tile_id <= delta_current_op.tile_id;
                activation_cache[cache_lru_ptr].addr <= delta_current_op.dst_addr;
                activation_cache[cache_lru_ptr].size <= delta_current_op.size;
                
                cache_lru_ptr <= cache_lru_ptr + 1;
                if (cache_lru_ptr >= ACTIVATION_CACHE_SIZE - 1) begin
                    cache_lru_ptr <= 0;
                end
            end
        end
    end
    
    //=========================================================================
    // Memory Interface Arbitration
    //=========================================================================
    // Channel 0: Delta operations
    assign mem_rd_addr_0 = delta_current_src_addr;
    assign mem_rd_len_0 = delta_words_remaining;
    assign mem_rd_req_0 = (delta_state == DELTA_PROCESS_LOAD_EXEC);
    assign mem_rd_ready_ack_0 = (delta_state == DELTA_WAIT_LOAD_COMPLETE);
    
    // Channel 1: Prefetch operations
    assign mem_rd_addr_1 = prefetch_current_op.dst_addr;  // From off-chip
    assign mem_rd_len_1 = prefetch_words_remaining;
    assign mem_rd_req_1 = (prefetch_state == PREFETCH_ISSUE_REQUEST);
    assign mem_rd_ready_ack_1 = (prefetch_state == PREFETCH_RECEIVE_DATA);
    
    //=========================================================================
    // Scratchpad Interface
    //=========================================================================
    // Port A: Delta operations
    always_comb begin
        spad_rd_en_a = 1'b0;
        spad_rd_addr_a = '0;
        spad_wr_en_a = 1'b0;
        spad_wr_addr_a = '0;
        spad_wr_data_a = '0;
        
        case (delta_state)
            DELTA_PROCESS_MOVE_EXEC: begin
                spad_rd_en_a = 1'b1;
                spad_rd_addr_a = delta_current_src_addr[$clog2(SPAD_DEPTH)-1:0];
                spad_wr_en_a = 1'b1;
                spad_wr_addr_a = delta_current_dst_addr[$clog2(SPAD_DEPTH)-1:0];
                spad_wr_data_a = spad_rd_data_a;
            end
            
            DELTA_WAIT_LOAD_COMPLETE: begin
                if (mem_rd_valid_0) begin
                    spad_wr_en_a = 1'b1;
                    spad_wr_addr_a = delta_current_dst_addr[$clog2(SPAD_DEPTH)-1:0];
                    spad_wr_data_a = mem_rd_data_0;
                end
            end
        endcase
    end
    
    // Port B: Prefetch operations (write to prefetch buffer)
    always_comb begin
        prefetch_wr_en = 1'b0;
        prefetch_wr_addr = '0;
        prefetch_wr_data = '0;
        
        if (prefetch_state == PREFETCH_RECEIVE_DATA && mem_rd_valid_1) begin
            prefetch_wr_en = 1'b1;
            prefetch_wr_addr = prefetch_current_dst_addr[$clog2(PREFETCH_BUFFER_DEPTH)-1:0];
            prefetch_wr_data = mem_rd_data_1;
        end
    end
    
    //=========================================================================
    // Status Outputs
    //=========================================================================
    assign phase_ready = (delta_state == DELTA_IDLE);
    assign phase_done = (delta_state == DELTA_DONE);
    assign controller_busy = (delta_state != DELTA_IDLE);
    assign prefetch_engine_busy = (prefetch_state != PREFETCH_IDLE);
    
    assign stats_loads = loads_count;
    assign stats_moves = moves_count;
    assign stats_keeps = keeps_count;
    assign stats_prefetches = prefetches_count;
    assign stats_prefetch_hits = prefetch_hits_count;
    assign stats_bytes_loaded = bytes_loaded_count;
    assign stats_bytes_moved = bytes_moved_count;
    assign stats_bytes_prefetched = bytes_prefetched_count;
    
    assign inter_layer_reuse_hit = reuse_hit;
    assign inter_layer_bytes_saved = inter_layer_bytes_saved_count;
    
    //=========================================================================
    // Update phase ID for prefetch scheduling
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_phase_id <= 0;
        end else if (phase_start) begin
            current_phase_id <= phase_id;
        end
    end
    
    //=========================================================================
    // Statistics Update
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prefetch_hits_count <= 0;
            inter_layer_bytes_saved_count <= 0;
        end else begin
            if (reuse_hit && delta_state == DELTA_PROCESS_LOAD_SETUP) begin
                prefetch_hits_count <= prefetch_hits_count + 1;
                inter_layer_bytes_saved_count <= inter_layer_bytes_saved_count + delta_current_op.size;
            end
        end
    end

endmodule : extended_memory_controller
