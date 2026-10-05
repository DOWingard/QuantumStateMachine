OPENQASM 3.0;
include "stdgates.inc";
// expect: E9005 5:13
// an input needs a value from --param theta=…
input float theta;
qubit q;
rx(theta) q;
