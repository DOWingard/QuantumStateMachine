OPENQASM 2.0;
include "qelib1.inc";
// expect: E9002 6:1
// foo is neither declared nor in qelib1.inc
qreg q[1];
foo q[0];
