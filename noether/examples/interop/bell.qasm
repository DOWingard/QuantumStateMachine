OPENQASM 3.0;
include "stdgates.inc";

// A Bell pair: outcomes 00 and 11, each with probability 1/2. Every gate is Clifford, so the
// runner picks the stabilizer tableau.
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
