OPENQASM 3.0;
include "stdgates.inc";
// Two Grover iterations marking |101⟩ (q[0] = q[2] = 1): P(101) = sin²(5·asin(1/√8)).
gate oracle a, b, c {
  x b;
  h c;
  ccx a, b, c;
  h c;
  x b;
}
gate diffuse a, b, c {
  h a; h b; h c;
  x a; x b; x c;
  h c;
  ccx a, b, c;
  h c;
  x a; x b; x c;
  h a; h b; h c;
}
qubit[3] q;
bit[3] c;
h q;
for int i in [1:2] {
  oracle q[0], q[1], q[2];
  diffuse q[0], q[1], q[2];
}
c = measure q;
