OPENQASM 3.0;
include "stdgates.inc";

// Teleports ry(theta)|0⟩ from q[0] to q[2] with mid-circuit measurement and feed-forward.
// P(c[2] = 1) = sin²(theta/2) whatever c[0] and c[1] read. Run with --param theta=1.1.
input float theta;
qubit[3] q;
bit[3] c;
ry(theta) q[0];
h q[1];
cx q[1], q[2];
cx q[0], q[1];
h q[0];
c[0] = measure q[0];
c[1] = measure q[1];
if (c[1]) x q[2];
if (c[0]) z q[2];
c[2] = measure q[2];
