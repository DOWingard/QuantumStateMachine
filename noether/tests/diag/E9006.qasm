OPENQASM 3.0;
include "stdgates.inc";
// expect: E9006 6:1
// cx takes two qubits
qubit[2] q;
cx q[0];
