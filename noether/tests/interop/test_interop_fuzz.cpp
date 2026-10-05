// Parser robustness: mutated sources (byte flips, splices, truncations, duplicated lines) must end
// in a circuit or in diagnostics, never in a crash, hang or stray exception. Seeds are fixed, so a
// failure reproduces; the trace prints the offending text.

#include "InteropTestUtil.hpp"

using namespace InteropTest;

namespace
{

const std::string kQasm3 = R"(OPENQASM 3.0;
include "stdgates.inc";
const int n = 3;
input float theta;
gate g(a) x, y { ctrl @ rz(a / 2) x, y; inv @ s y; pow(0.5) @ x x; }
qubit[n] q;
bit[n] c;
h q;
g(theta) q[0], q[1];
for int i in [0:1] { cx q[i], q[i + 1]; }
c[0] = measure q[0];
if (c[0]) { x q[1]; } else { z q[2]; }
box { negctrl @ h q[2], q[0]; }
#pragma noise
@annotation
delay[10ns] q;
c[1:2] = measure q[1:2];
)";

const std::string kQasm2 = R"(OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
gate maj a, b, c { cx c, b; cx c, a; ccx a, b, c; }
u3(0.1, 0.2, 0.3) q[0];
maj q[0], q[1], q[2];
barrier q;
measure q[0] -> c[0];
if (c == 1) x q[1];
measure q -> c;
)";

const std::string kStim = R"(QUBIT_COORDS(0, 0) 0
R 0 1 2 3
X_ERROR(0.01) 0 1 2
TICK
CX 0 1 2 3
MR 1 3
DETECTOR(1, 0) rec[-1] rec[-2]
REPEAT 2 {
    DEPOLARIZE2(0.01) 0 2
    E(0.1) X0 Y2
    ELSE_CORRELATED_ERROR(0.2) Z1
    MPP X0*Z2 !Y1
    SPP Z0*Z1
    M 0 1
    DETECTOR rec[-1] rec[-3]
}
M(0.01) 0 2
OBSERVABLE_INCLUDE(0) rec[-1]
)";

const std::string kJson = R"({"schema": "noether.circuit/1", "num_qubits": 2, "num_clbits": 2, "qubit_labels": ["a", "b"],
"clbit_registers": [{"name": "c", "offset": 0, "size": 2}], "source": {"framework": "test"},
"ops": [{"op": "h", "targets": [0]}, {"op": "cnot", "controls": [0], "targets": [1]},
{"op": "rx", "targets": [1], "params": [0.5]}, {"op": "unitary", "targets": [0], "matrix": [[[0, 0], [1, 0]], [[1, 0], [0, 0]]]},
{"op": "pauli_channel", "targets": [0], "params": [0.1, 0, 0]}, {"op": "measure", "targets": [0], "clbit": 0},
{"op": "x", "targets": [1], "condition": {"clbits": [0], "values": [1]}}, {"op": "measure", "targets": [1], "clbit": 1}],
"notes": [{"code": "W9001", "message": "phase"}]}
)";

const std::string kAlphabet = "{}[]()<>;:,.=!&|+-*/%^~@#$\"' \n\t0123456789abcxyzqXYZ_πθ";

std::string mutate(const std::string& base, std::mt19937_64& rng)
{
    std::string s = base;
    auto pick = [&](std::size_t n) { return std::uniform_int_distribution<std::size_t>(0, n == 0 ? 0 : n - 1)(rng); };
    const int edits = std::uniform_int_distribution<int>(1, 4)(rng);
    for (int e = 0; e < edits && !s.empty(); ++e)
    {
        const std::size_t at = pick(s.size());
        switch (std::uniform_int_distribution<int>(0, 5)(rng))
        {
            case 0: s[at] = kAlphabet[pick(kAlphabet.size())]; break;
            case 1: s.insert(at, 1, kAlphabet[pick(kAlphabet.size())]); break;
            case 2: s.erase(at, 1 + pick(8)); break;
            case 3: s.resize(at); break;
            case 4:
            {
                const std::size_t from = pick(s.size()), len = 1 + pick(40);
                s.insert(at, s.substr(from, len));
                break;
            }
            default:
            {
                // Duplicate a whole line, which exercises redeclarations and repeated loops.
                const auto b = s.rfind('\n', at), e2 = s.find('\n', at);
                const std::size_t lb = b == std::string::npos ? 0 : b + 1;
                s.insert(lb, s.substr(lb, (e2 == std::string::npos ? s.size() : e2 + 1) - lb));
                break;
            }
        }
    }
    return s;
}

void fuzz(const std::string& path, const std::string& base, const std::string& format, std::uint64_t seed, int iterations)
{
    std::mt19937_64 rng(seed);
    std::size_t imported = 0;
    for (int k = 0; k < iterations; ++k)
    {
        const std::string text = mutate(base, rng);
        ImportOptions o;
        o.format = format;
        o.params = {{"theta", 0.4}};
        try
        {
            const ImportResult r = importText(path, text, o);
            if (r.circuit)
            {
                ++imported;
                EXPECT_FALSE(r.diags->hasErrors()) << text;
                for (const auto& op : r.circuit->ops)
                    ASSERT_EQ(validateOperation(op, r.circuit->numQubits, r.circuit->numClbits), "") << text;
            }
            else
            {
                ASSERT_TRUE(r.diags->hasErrors()) << "no circuit and no error for:\n" << text;
                for (const auto& d : r.diags->sorted()) ASSERT_FALSE(d.message.empty()) << text;
                (void)r.diags->render();
            }
        }
        catch (const std::exception& e)
        {
            FAIL() << "exception " << e.what() << " for:\n" << text;
        }
    }
    // Some mutations (inside comments, numbers) must still import; otherwise the base is broken.
    EXPECT_GT(imported, 0U);
}

} // namespace

TEST(InteropFuzz, BasesImport)
{
    for (const auto& [path, text, format] : {std::tuple{"f.qasm", kQasm3, "qasm"}, {"f.qasm", kQasm2, "qasm"}, {"f.stim", kStim, "stim"},
                                             {"f.json", kJson, "circuit"}})
    {
        const ImportResult r = load(path, text, format, false, {{"theta", 0.4}});
        EXPECT_TRUE(r.circuit) << path;
    }
}

TEST(InteropFuzz, Qasm3) { fuzz("f.qasm", kQasm3, "qasm", 1, 3000); }
TEST(InteropFuzz, Qasm2) { fuzz("f.qasm", kQasm2, "qasm", 2, 3000); }
TEST(InteropFuzz, Stim) { fuzz("f.stim", kStim, "stim", 3, 3000); }
TEST(InteropFuzz, CircuitJson) { fuzz("f.json", kJson, "circuit", 4, 3000); }
