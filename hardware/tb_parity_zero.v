`timescale 1ns/1ps
module tb;
    parameter N = 8;
    reg  [N:0]   stored;
    reg  [N:0]   base;
    wire [N-1:0] weight;
    wire         err;
    reg  [N-1:0] data;
    integer d, i, j;
    integer ok0, bad0, ok1, bad1, ok2, bad2, silent2;
    parity_zero #(.N(N)) dut (.stored(stored), .weight(weight), .err(err));
    initial begin
        ok0 = 0; bad0 = 0; ok1 = 0; bad1 = 0; ok2 = 0; bad2 = 0; silent2 = 0;
        for (d = 0; d < (1 << N); d = d + 1) begin
            data = d;
            base = {^data, data};                       // even parity bit
            stored = base; #1;                          // no fault
            if (weight === data && err === 1'b0) ok0 = ok0 + 1; else bad0 = bad0 + 1;
            for (i = 0; i <= N; i = i + 1) begin        // every single flip
                stored = base ^ ({{N{1'b0}}, 1'b1} << i); #1;
                if (weight === {N{1'b0}} && err === 1'b1) ok1 = ok1 + 1; else bad1 = bad1 + 1;
            end
            for (i = 0; i <= N; i = i + 1)              // every double flip
                for (j = i + 1; j <= N; j = j + 1) begin
                    stored = base ^ ({{N{1'b0}}, 1'b1} << i) ^ ({{N{1'b0}}, 1'b1} << j); #1;
                    if (err === 1'b0 && weight === stored[N-1:0]) ok2 = ok2 + 1; else bad2 = bad2 + 1;
                    if (weight !== data) silent2 = silent2 + 1;
                end
        end
        $display("N=%0d | no fault      : %0d ok, %0d wrong", N, ok0, bad0);
        $display("N=%0d | single flips   : %0d ok (read as 0), %0d wrong", N, ok1, bad1);
        $display("N=%0d | double flips   : %0d pass through undetected, %0d wrong", N, ok2, bad2);
        $display("N=%0d | double flips that corrupt the weight silently: %0d", N, silent2);
        if (bad0 == 0 && bad1 == 0 && bad2 == 0) $display("N=%0d RESULT: ALL TESTS PASSED", N);
        else $display("N=%0d RESULT: FAILED", N);
        $finish;
    end
endmodule
