OPENQASM 3.0;
include "stdgates.inc";
// expect: W9002 7:1
// idle time decoheres hardware, but the simulation has no noise model for it
qubit q;
h q;
delay[100ns] q;
