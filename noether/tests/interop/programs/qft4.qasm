OPENQASM 3.0;
include "stdgates.inc";
// QFT on |x = 5⟩ with qubit 3 most significant: amplitude k is e^{2πi·5k/16}/4.
qubit[4] q;
x q[0];
x q[2];
h q[3];
cp(pi/2) q[2], q[3];
cp(pi/4) q[1], q[3];
cp(pi/8) q[0], q[3];
h q[2];
cp(pi/2) q[1], q[2];
cp(pi/4) q[0], q[2];
h q[1];
cp(pi/2) q[0], q[1];
h q[0];
swap q[0], q[3];
swap q[1], q[2];
