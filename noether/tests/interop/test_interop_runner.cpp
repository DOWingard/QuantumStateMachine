// The runner: backend choice, sampled versus replayed shots, state readout, seeds, and the checks
// that refuse a run before anything is allocated.

#include "InteropTestUtil.hpp"

#include <chrono>

using namespace InteropTest;

namespace
{

RunResult runText(const std::string& path, const std::string& text, RunOptions o = {})
{
    const ImportResult r = load(path, text);
    if (!r.circuit) throw std::runtime_error("import failed: " + r.diags->render());
    if (!o.seed) o.seed = 7;
    return runCircuit(*r.circuit, o);
}

std::string errorCode(const std::string& path, const std::string& text, RunOptions o = {})
{
    try
    {
        runText(path, text, o);
    }
    catch (const ImportError& e)
    {
        return e.code();
    }
    return "";
}

const std::string kHeader = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n";

// Each outcome's frequency in a and b agrees within 4σ of the difference of two proportions.
void expectSameDistribution(const Qputer::Counts& a, const Qputer::Counts& b, std::size_t shots)
{
    std::set<Outcome> keys;
    for (const auto& [k, n] : a) keys.insert(k);
    for (const auto& [k, n] : b) keys.insert(k);
    const auto s = static_cast<double>(shots);
    for (const Outcome k : keys)
    {
        const double pa = a.contains(k) ? static_cast<double>(a.at(k)) / s : 0.0;
        const double pb = b.contains(k) ? static_cast<double>(b.at(k)) / s : 0.0;
        const double p = (pa + pb) / 2;
        EXPECT_LE(std::abs(pa - pb), 4 * std::sqrt(2 * p * (1 - p) / s) + 2 / s) << "outcome " << k;
    }
}

} // namespace

TEST(InteropRunner, CliffordCircuitsRunOnTheTableau)
{
    // rx(π/2) and cp(π) are Clifford up to global phase; t is not.
    const RunResult a = runText("a.qasm", kHeader + "qubit[2] q;\nbit[2] c;\nrx(pi/2) q[0];\ncp(pi) q[0], q[1];\nc = measure q;\n");
    EXPECT_EQ(a.backend, "stabilizer");
    const RunResult b = runText("b.qasm", kHeader + "qubit[2] q;\nbit[2] c;\nh q[0];\nt q[0];\nc = measure q;\n");
    EXPECT_EQ(b.backend, "statevector");
    EXPECT_NE(b.backendReason.find("t"), std::string::npos) << b.backendReason;
}

TEST(InteropRunner, RewrittenCliffordsKeepTheDistribution)
{
    const std::string text = kHeader + "qubit[3] q;\nbit[3] c;\nry(pi/2) q[0];\nrz(-pi/2) q[0];\ncx q[0], q[1];\n"
                                       "rx(3*pi/2) q[2];\ncp(-pi) q[1], q[2];\nh q[2];\nc = measure q;\n";
    RunOptions tab, sv;
    tab.shots = sv.shots = 20000;
    tab.backend = "stabilizer";
    sv.backend = "statevector";
    const RunResult a = runText("c.qasm", text, tab), b = runText("c.qasm", text, sv);
    EXPECT_EQ(a.backend, "stabilizer");
    EXPECT_EQ(b.backend, "statevector");
    expectSameDistribution(a.counts, b.counts, 20000);
}

TEST(InteropRunner, SampledAndReplayedShotsAgree)
{
    const std::string text = kHeader + "qubit[3] q;\nbit[3] c;\nry(0.7) q[0];\ncx q[0], q[1];\nrx(1.9) q[2];\nccx q[0], q[2], q[1];\n"
                                       "c = measure q;\n";
    RunOptions fast, slow;
    fast.shots = slow.shots = 20000;
    slow.sampleTerminal = false;
    const RunResult a = runText("s.qasm", text, fast), b = runText("s.qasm", text, slow);
    EXPECT_EQ(a.method, "sampled");
    EXPECT_EQ(b.method, "trajectories");
    expectSameDistribution(a.counts, b.counts, 20000);
}

TEST(InteropRunner, MidCircuitMeasurementReplaysEveryShot)
{
    const RunResult r = runText("m.qasm", kHeader + "qubit q;\nbit[2] c;\nh q;\nc[0] = measure q;\nh q;\nc[1] = measure q;\n");
    EXPECT_EQ(r.method, "trajectories");
    EXPECT_EQ(r.counts.size(), 4U);
}

TEST(InteropRunner, StateReadout)
{
    RunOptions o;
    o.emit = "statevector";
    const RunResult r = runText("v.qasm", kHeader + "qubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\ns q[1];\nc = measure q;\n", o);
    EXPECT_EQ(r.method, "state");
    ASSERT_EQ(r.amplitudes.size(), 4);
    const double h = 1 / std::sqrt(2.0);
    EXPECT_LE(std::abs(r.amplitudes[0] - cd(h, 0)), 1e-12);
    EXPECT_LE(std::abs(r.amplitudes[3] - cd(0, h)), 1e-12);
    o.emit = "probabilities";
    const RunResult p = runText("v.qasm", kHeader + "qubit[2] q;\nh q[0];\n", o);
    EXPECT_NEAR(p.probabilities.sum(), 1.0, 1e-12);
    EXPECT_NEAR(p.probabilities[1], 0.5, 1e-12);
}

TEST(InteropRunner, StateReadoutRefusesNonUnitaryCircuits)
{
    RunOptions o;
    o.emit = "statevector";
    EXPECT_EQ(errorCode("m.qasm", kHeader + "qubit q;\nbit[2] c;\nh q;\nc[0] = measure q;\nh q;\nc[1] = measure q;\n", o), "E9003");
    EXPECT_EQ(errorCode("m.qasm", kHeader + "qubit[2] q;\nbit c;\nc = measure q[0];\nif (c) x q[1];\n", o), "E9003");
    EXPECT_EQ(errorCode("n.stim", "X_ERROR(0.1) 0\nM 0\n", o), "E9003");
}

TEST(InteropRunner, CountsNeedAMeasurement)
{
    EXPECT_EQ(errorCode("e.qasm", kHeader + "qubit q;\nh q;\n"), "E5011");
}

TEST(InteropRunner, RequestedTableauRejectsNonClifford)
{
    RunOptions o;
    o.backend = "stabilizer";
    EXPECT_EQ(errorCode("t.qasm", kHeader + "qubit q;\nbit c;\nt q;\nc = measure q;\n", o), "E6001");
}

TEST(InteropRunner, LargeCircuitsFailBeforeAllocating)
{
    // 30 qubits on the state vector would take 16 GiB; the runner must refuse at once.
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(errorCode("w.qasm", kHeader + "qubit[30] q;\nbit c;\nh q;\nt q[0];\nc = measure q[29];\n"), "E9004");
    EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), 5.0);
    // The same width runs when it is Clifford.
    RunOptions o;
    o.shots = 100;
    const RunResult r = runText("g.qasm", kHeader + "qubit[30] q;\nbit[2] c;\nh q[0];\nfor int i in [0:28] { cx q[i], q[i + 1]; }\n"
                                                    "c[0] = measure q[0];\nc[1] = measure q[29];\n", o);
    EXPECT_EQ(r.backend, "stabilizer");
    for (const auto& [k, n] : r.counts) EXPECT_TRUE(k == 0 || k == 3) << k;
}

TEST(InteropRunner, SeedsReproduce)
{
    const std::string text = kHeader + "qubit[3] q;\nbit[3] c;\nry(0.3) q[0];\nry(1.3) q[1];\nh q[2];\nc = measure q;\n";
    RunOptions o;
    o.seed = 123;
    EXPECT_EQ(runText("r.qasm", text, o).counts, runText("r.qasm", text, o).counts);
    RunOptions drawn;
    const ImportResult r = load("r.qasm", text);
    ASSERT_TRUE(r.circuit);
    const RunResult a = runCircuit(*r.circuit, drawn);
    EXPECT_LT(a.seed, std::uint64_t{1} << 53); // reported in JSON, so it must be an exact double
    RunOptions again;
    again.seed = a.seed;
    EXPECT_EQ(runCircuit(*r.circuit, again).counts, a.counts);
}

TEST(InteropRunner, NoiselessDetectorsNeverFire)
{
    RunOptions o;
    o.shots = 500;
    const RunResult r = runText("q.stim", "R 0 1 2\nH 0\nCX 0 1 0 2\nM 0 1 2\nDETECTOR rec[-1] rec[-2]\nDETECTOR rec[-2] rec[-3]\n"
                                          "OBSERVABLE_INCLUDE(0) rec[-1]\n", o);
    ASSERT_EQ(r.detectorFires.size(), 2U);
    EXPECT_EQ(r.detectorFires[0], 0U);
    EXPECT_EQ(r.detectorFires[1], 0U);
    // The observable is random, but relative to the reference sample it is still random.
    ASSERT_EQ(r.observableFlips.size(), 1U);
    EXPECT_GT(r.observableFlips[0], 150U);
    EXPECT_LT(r.observableFlips[0], 350U);
}
