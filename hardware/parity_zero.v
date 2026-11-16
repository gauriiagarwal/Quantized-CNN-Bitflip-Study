// Parity-zero read path. `stored` = N data bits + 1 even-parity bit (bit N).
// A fault-free word has an even number of ones, so ^stored = 0.
// If an ODD number of bits flipped, ^stored = 1 -> the weight is read as 0.
module parity_zero #(parameter N = 8) (
    input  [N:0]   stored,
    output [N-1:0] weight,
    output         err
);
    assign err    = ^stored;
    assign weight = err ? {N{1'b0}} : stored[N-1:0];
endmodule
