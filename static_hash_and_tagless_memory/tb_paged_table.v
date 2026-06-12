`timescale 1ns / 1ps

module tb_page_table;

    // -----------------------------------------------------------------------
    // Parameters (must match DUT)
    // -----------------------------------------------------------------------
    localparam VADDR_WIDTH    = 32;
    localparam PADDR_WIDTH    = 32;
    localparam PAGE_SIZE_BITS = 12;
    localparam NUM_PAGES      = 256;
    localparam VPN_WIDTH      = 8;
    localparam PPN_WIDTH      = 20;
    localparam OFFSET_WIDTH   = PAGE_SIZE_BITS;

    // -----------------------------------------------------------------------
    // DUT ports
    // -----------------------------------------------------------------------
    reg                   clk;
    reg                   rst_n;
    reg  [VADDR_WIDTH-1:0] vaddr;
    wire [PADDR_WIDTH-1:0] paddr;
    wire                   hit;
    wire                   valid;
    reg                   write_en;
    reg  [VPN_WIDTH-1:0]  write_vpn;
    reg  [PPN_WIDTH-1:0]  write_ppn;
    reg                   write_valid;

    // -----------------------------------------------------------------------
    // Instantiate DUT
    // -----------------------------------------------------------------------
    page_table #(
        .VADDR_WIDTH    (VADDR_WIDTH),
        .PADDR_WIDTH    (PADDR_WIDTH),
        .PAGE_SIZE_BITS (PAGE_SIZE_BITS),
        .NUM_PAGES      (NUM_PAGES),
        .VPN_WIDTH      (VPN_WIDTH),
        .PPN_WIDTH      (PPN_WIDTH)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .vaddr      (vaddr),
        .paddr      (paddr),
        .hit        (hit),
        .valid      (valid),
        .write_en   (write_en),
        .write_vpn  (write_vpn),
        .write_ppn  (write_ppn),
        .write_valid(write_valid)
    );

    // -----------------------------------------------------------------------
    // Clock: 10 ns period
    // -----------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // -----------------------------------------------------------------------
    // Test tracking
    // -----------------------------------------------------------------------
    integer pass_count = 0;
    integer fail_count = 0;

    task check;
        input [127:0] test_name; // up to 16 ASCII chars
        input         cond;
        begin
            if (cond) begin
                $display("  PASS | %s", test_name);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL | %s  (hit=%b valid=%b paddr=%h)",
                          test_name, hit, valid, paddr);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // Helper: write one PTE and wait for it to register
    task write_pte;
        input [VPN_WIDTH-1:0] vpn;
        input [PPN_WIDTH-1:0] ppn;
        input                 v;
        begin
            @(posedge clk);
            write_en    <= 1'b1;
            write_vpn   <= vpn;
            write_ppn   <= ppn;
            write_valid <= v;
            @(posedge clk);
            write_en <= 1'b0;
            #1; // let combinational outputs settle
        end
    endtask

    // -----------------------------------------------------------------------
    // Main test sequence
    // -----------------------------------------------------------------------
    initial begin
        // ---- Initialise inputs ----
        rst_n      = 1'b0;
        write_en   = 1'b0;
        write_vpn  = 0;
        write_ppn  = 0;
        write_valid= 0;
        vaddr      = 0;

        // ===================================================================
        // TEST 1 – Reset clears all entries
        // ===================================================================
        $display("\n=== TEST 1: Reset behaviour ===");
        @(posedge clk); #1;
        rst_n = 1'b0;           // assert reset
        repeat(3) @(posedge clk);
        #1;

        // probe every 16th VPN for efficiency; all should be invalid
        begin : reset_check
            integer vpn_i;
            reg all_invalid;
            all_invalid = 1'b1;
            for (vpn_i = 0; vpn_i < NUM_PAGES; vpn_i = vpn_i + 16) begin
                vaddr = vpn_i << PAGE_SIZE_BITS; // offset = 0
                #1;
                if (hit || valid) all_invalid = 1'b0;
            end
            check("All entries invalid after rst", all_invalid);
        end

        // ===================================================================
        // TEST 2 – Basic address translation (VPN 0 → PPN 0xABCDE)
        // ===================================================================
        $display("\n=== TEST 2: Basic translation ===");
        rst_n = 1'b1;
        @(posedge clk); #1;

        write_pte(8'h00, 20'hABCDE, 1'b1);

        // Query with VPN=0, offset=0x123
        vaddr = (32'h00 << PAGE_SIZE_BITS) | 32'h123;
        #1;
        check("hit asserted",        hit   == 1'b1);
        check("valid asserted",      valid == 1'b1);
        check("PPN correct",         paddr[PADDR_WIDTH-1:PAGE_SIZE_BITS] == 20'hABCDE);
        check("Offset preserved",    paddr[PAGE_SIZE_BITS-1:0] == 12'h123);
        check("Full paddr correct",  paddr == {20'hABCDE, 12'h123});

        // ===================================================================
        // TEST 3 – Miss on unmapped VPN
        // ===================================================================
        $display("\n=== TEST 3: Miss on unmapped VPN ===");

        // VPN 1 was never written → should miss
        vaddr = (32'h01 << PAGE_SIZE_BITS) | 32'hFFF;
        #1;
        check("hit low on miss",   hit   == 1'b0);
        check("valid low on miss", valid == 1'b0);
        check("paddr 0 on miss",   paddr == {PADDR_WIDTH{1'b0}});

        // ===================================================================
        // TEST 4 – Multiple VPN → PPN mappings
        // ===================================================================
        $display("\n=== TEST 4: Multiple mappings ===");

        write_pte(8'h01, 20'h11111, 1'b1);
        write_pte(8'h55, 20'h55555, 1'b1);
        write_pte(8'hFF, 20'hFFFFF, 1'b1);

        vaddr = (32'h01 << PAGE_SIZE_BITS) | 32'hABC;
        #1;
        check("VPN=01 paddr OK", paddr == {20'h11111, 12'hABC});

        vaddr = (32'h55 << PAGE_SIZE_BITS) | 32'h000;
        #1;
        check("VPN=55 paddr OK", paddr == {20'h55555, 12'h000});

        vaddr = (32'hFF << PAGE_SIZE_BITS) | 32'hFFF;
        #1;
        check("VPN=FF paddr OK", paddr == {20'hFFFFF, 12'hFFF});

        // ===================================================================
        // TEST 5 – Offset passthrough (all 12 offset bits)
        // ===================================================================
        $display("\n=== TEST 5: Offset passthrough ===");

        write_pte(8'h02, 20'h20000, 1'b1);

        vaddr = (32'h02 << PAGE_SIZE_BITS) | 32'h000; #1;
        check("Offset 0x000", paddr[11:0] == 12'h000);

        vaddr = (32'h02 << PAGE_SIZE_BITS) | 32'hFFF; #1;
        check("Offset 0xFFF", paddr[11:0] == 12'hFFF);

        vaddr = (32'h02 << PAGE_SIZE_BITS) | 32'hA5A; #1;
        check("Offset 0xA5A", paddr[11:0] == 12'hA5A);

        // ===================================================================
        // TEST 6 – Invalidate an existing mapping (write_valid = 0)
        // ===================================================================
        $display("\n=== TEST 6: Invalidate a mapping ===");

        // VPN 0 was valid; invalidate it
        write_pte(8'h00, 20'hABCDE, 1'b0);

        vaddr = (32'h00 << PAGE_SIZE_BITS) | 32'h123;
        #1;
        check("hit low after invalidate",   hit   == 1'b0);
        check("valid low after invalidate", valid == 1'b0);

        // ===================================================================
        // TEST 7 – Re-map an existing VPN to a new PPN
        // ===================================================================
        $display("\n=== TEST 7: Remap VPN to new PPN ===");

        write_pte(8'h01, 20'h99999, 1'b1); // was 0x11111

        vaddr = (32'h01 << PAGE_SIZE_BITS) | 32'h321;
        #1;
        check("Remapped PPN correct", paddr == {20'h99999, 12'h321});

        // ===================================================================
        // TEST 8 – Reset clears all (including newly written) entries
        // ===================================================================
        $display("\n=== TEST 8: Reset after writes ===");

        rst_n = 1'b0;
        repeat(2) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk); #1;

        vaddr = (32'h01 << PAGE_SIZE_BITS) | 32'h000;
        #1;
        check("VPN=01 invalid post-reset", hit == 1'b0);

        vaddr = (32'hFF << PAGE_SIZE_BITS) | 32'h000;
        #1;
        check("VPN=FF invalid post-reset", hit == 1'b0);

        // ===================================================================
        // TEST 9 – Boundary: last valid VPN (0xFF with NUM_PAGES=256)
        // ===================================================================
        $display("\n=== TEST 9: Boundary VPN (0xFF) ===");

        write_pte(8'hFF, 20'hBEEF0, 1'b1);

        vaddr = (32'hFF << PAGE_SIZE_BITS) | 32'hBEE;
        #1;
        check("Boundary VPN hit",   hit   == 1'b1);
        check("Boundary paddr OK",  paddr == {20'hBEEF0, 12'hBEE});

        // ===================================================================
        // TEST 10 – Simultaneous write + read of same VPN
        // ===================================================================
        $display("\n=== TEST 10: Write-then-read same cycle ===");

        // First write VPN=3
        write_pte(8'h03, 20'h30000, 1'b1);

        // On the same cycle we read VPN=3, the PREVIOUS value should
        // appear (write is synchronous, read is combinational).
        vaddr     = (32'h03 << PAGE_SIZE_BITS) | 32'h456;
        write_en  = 1'b1;
        write_vpn = 8'h03;
        write_ppn = 20'hDEADB;
        write_valid = 1'b1;
        #1;  // combinational outputs reflect OLD value (0x30000)
        check("Read shows old value mid-write", paddr[PADDR_WIDTH-1:PAGE_SIZE_BITS] == 20'h30000);
        @(posedge clk) #1;
        write_en = 1'b0;
        #1;  // now the new value is registered
        check("Read shows new value after clk",  paddr[PADDR_WIDTH-1:PAGE_SIZE_BITS] == 20'hDEADB);

        // ===================================================================
        // Summary
        // ===================================================================
        $display("\n========================================");
        $display("  Results: %0d passed, %0d failed", pass_count, fail_count);
        $display("========================================\n");

        if (fail_count == 0)
            $display("ALL TESTS PASSED");

        else
            $display("SOME TESTS FAILED — review FAIL lines above");

        $finish;
    end

    // -----------------------------------------------------------------------
    // Timeout watchdog (prevents infinite loops in simulation)
    // -----------------------------------------------------------------------
    initial begin
        #50000;
        $display("TIMEOUT — simulation exceeded 50 us");
        $finish;
    end

    // -----------------------------------------------------------------------
    // Optional: waveform dump for Vivado simulator
    // -----------------------------------------------------------------------
    initial begin
        $dumpfile("tb_page_table.vcd");
        $dumpvars(0, tb_page_table);
    end

endmodule