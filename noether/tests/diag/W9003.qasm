OPENQASM 3.0;
include "stdgates.inc";
// expect: W9003 5:1
// q[1] is never touched
qubit[2] q;
h q[0];
