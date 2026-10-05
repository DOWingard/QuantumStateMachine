// Importers end to end: golden programs (CLI JSON pinned by seed, plus a physical check each), Stim
// and OpenQASM semantics, circuit JSON round trips, and resource limits.
//
// Goldens live in tests/interop/programs as <name>.<ext> with <name>.expected.json. Set
// NOETHER_UPDATE_GOLDENS=1 to rewrite them after a deliberate change, then review the diff.

#include "InteropTestUtil.hpp"

#include "interop/CircuitJson.hpp"
#include "interop/Run.hpp"

#include <cstdlib>
#include <numbers>

using namespace InteropTest;
using Noether::Json;
using NoetherTest::CliResult;
using NoetherTest::cli;

namespace
{

constexpr double kTol = 1e-10;

const std::filesystem::path kPrograms = kInteropDir / "programs";

// |observed − p| within 4σ of a binomial proportion over `shots`.
void expectRate(double observed, double p, std::size_t shots, const std::string& what)
{
    const double sigma = std::sqrt(p * (1 - p) / static_cast<double>(shots));
    EXPECT_LE(std::abs(observed - p), 4 * sigma + 1e-12) << what << ": " << observed << " vs " << p;
}

struct GoldenCase
{
    std::string file;
    std::vector<std::string> args;
};

const std::vector<GoldenCase>& goldenCases()
{
    static const std::vector<GoldenCase> cases{
        {"bell.qasm", {"--shots", "4000"}},
        {"ghz.qasm", {"--shots", "4000"}},
        {"teleport.qasm", {"--shots", "20000"}},
        {"qft4.qasm", {"--emit", "statevector"}},
        {"grover3.qasm", {"--emit", "probabilities"}},
        {"rep_code.stim", {"--shots", "20000"}},
    };
    return cases;
}

Json runJson(const std::filesystem::path& file, std::vector<std::string> args)
{
    std::vector<std::string> argv{"run", file.string(), "--json", "--no-timing", "--seed", "1"};
    argv.insert(argv.end(), args.begin(), args.end());
    const CliResult r = cli(argv);
    EXPECT_EQ(r.code, 0) << r.out << r.err;
    auto j = Json::parse(r.out);
    if (!j) ADD_FAILURE() << "not JSON: " << r.out << r.err;
    return j ? *j : Json::object();
}

// Rate of clbit `bit` being 1 in a counts object keyed with clbit 0 leftmost.
double bitRate(const Json& doc, std::size_t bit)
{
    double ones = 0, total = 0;
    for (const auto& [k, v] : doc.find("counts")->asObject())
    {
        total += v.asDouble();
        if (k[bit] == '1') ones += v.asDouble();
    }
    return ones / total;
}

class InteropGolden : public ::testing::TestWithParam<GoldenCase>
{
};

} // namespace

TEST_P(InteropGolden, MatchesExpectedJson)
{
    const auto file = kPrograms / GetParam().file;
    const auto expected = kPrograms / (file.stem().string() + ".expected.json");
    const Json got = runJson(file, GetParam().args);
    if (const char* u = std::getenv("NOETHER_UPDATE_GOLDENS"); u && std::string(u) == "1")
    {
        NoetherTest::writeText(expected, got.dump() + "\n");
        GTEST_SKIP() << "rewrote " << expected;
    }
    ASSERT_TRUE(std::filesystem::exists(expected)) << "missing " << expected << "; run with NOETHER_UPDATE_GOLDENS=1";
    const auto want = Json::parse(NoetherTest::readText(expected));
    ASSERT_TRUE(want);
    std::vector<std::string> diffs;
    NoetherTest::compareJson(*want, got, "$", diffs);
    std::string all;
    for (const auto& d : diffs) all += d + "\n";
    EXPECT_TRUE(diffs.empty()) << all;
}

INSTANTIATE_TEST_SUITE_P(Programs, InteropGolden, ::testing::ValuesIn(goldenCases()),
                         [](const auto& p)
                         {
                             std::string s = p.param.file;
                             std::ranges::replace(s, '.', '_');
                             return s;
                         });

TEST(InteropPrograms, BellAndGhzAreCorrelated)
{
    for (const auto& [file, keys] : {std::pair{"bell.qasm", std::set<std::string>{"00", "11"}}, {"ghz.qasm", {"000", "111"}}})
    {
        const Json j = runJson(kPrograms / file, {"--shots", "4000"});
        EXPECT_EQ(j.find("backend")->asString(), "stabilizer");
        std::set<std::string> seen;
        for (const auto& [k, v] : j.find("counts")->asObject()) seen.insert(k);
        EXPECT_EQ(seen, keys) << file;
        expectRate(bitRate(j, 0), 0.5, 4000, file);
    }
}

TEST(InteropPrograms, TeleportationCarriesTheState)
{
    const Json j = runJson(kPrograms / "teleport.qasm", {"--shots", "20000"});
    expectRate(bitRate(j, 2), std::pow(std::sin(0.55), 2), 20000, "P(c[2] = 1)");
    expectRate(bitRate(j, 0), 0.5, 20000, "P(c[0] = 1)");
}

TEST(InteropPrograms, QftIsTheDiscreteFourierTransform)
{
    const Json j = runJson(kPrograms / "qft4.qasm", {"--emit", "statevector"});
    const auto& amps = j.find("amplitudes")->asArray();
    ASSERT_EQ(amps.size(), 16U);
    for (std::size_t k = 0; k < 16; ++k)
    {
        const cd want = std::polar(0.25, 2 * std::numbers::pi * 5.0 * static_cast<double>(k) / 16.0);
        const cd got(amps[k].asArray()[0].asDouble(), amps[k].asArray()[1].asDouble());
        EXPECT_LE(std::abs(got - want), kTol) << k;
    }
}

TEST(InteropPrograms, GroverAmplifiesTheMarkedState)
{
    const Json j = runJson(kPrograms / "grover3.qasm", {"--emit", "probabilities"});
    const double p = j.find("probabilities")->asArray()[0b101].asDouble();
    EXPECT_NEAR(p, std::pow(std::sin(5 * std::asin(1 / std::sqrt(8.0))), 2), kTol);
}

TEST(InteropPrograms, RepetitionCodeRatesMatchTheModel)
{
    const double p = 0.1;
    const Json j = runJson(kPrograms / "rep_code.stim", {"--shots", "20000"});
    const auto& det = j.find("detectors")->asArray();
    ASSERT_EQ(det.size(), 8U);
    // A round's detector compares one ancilla with its previous value: it fires when exactly one of
    // its two data qubits took an error that round. Nothing happens between the last round and the
    // final data readout, so the final detectors never fire.
    for (std::size_t k = 0; k < 6; ++k) expectRate(det[k].find("rate")->asDouble(), 2 * p * (1 - p), 20000, std::format("detector {}", k));
    EXPECT_EQ(det[6].find("fires")->asInt(), 0);
    EXPECT_EQ(det[7].find("fires")->asInt(), 0);
    const double flips = j.find("observables")->asArray()[0].find("rate")->asDouble();
    expectRate(flips, (1 - std::pow(1 - 2 * p, 3)) / 2, 20000, "observable 0");
}

// ---- Stim ----

TEST(InteropStim, QubitIdsAreCompactedAndLabelled)
{
    const ImportResult r = load("c.stim", "H 5\nCX 5 9\nM 9 5\n");
    ASSERT_TRUE(r.circuit);
    EXPECT_EQ(r.circuit->numQubits, 2U);
    EXPECT_EQ(r.circuit->qubitLabels, (std::vector<std::string>{"q5", "q9"}));
    EXPECT_EQ(r.circuit->numClbits, 2U);
    // M 9 5: result 0 is qubit 9 (index 1), result 1 is qubit 5 (index 0).
    const auto& ops = r.circuit->ops;
    ASSERT_GE(ops.size(), 2U);
    EXPECT_EQ(ops[ops.size() - 2].targets, QubitList{1});
    EXPECT_EQ(ops[ops.size() - 2].clbit, std::optional<std::size_t>{0});
}

TEST(InteropStim, DetectorsAreAbsoluteClbitMasks)
{
    const ImportResult r = load("d.stim", "M 0 1\nREPEAT 2 {\n M 0 1\n DETECTOR rec[-1] rec[-3]\n}\nOBSERVABLE_INCLUDE(2) rec[-2]\n");
    ASSERT_TRUE(r.circuit);
    EXPECT_EQ(r.circuit->detectors, (std::vector<Outcome>{0b1010, 0b101000}));
    ASSERT_EQ(r.circuit->observables.size(), 3U);
    EXPECT_EQ(r.circuit->observables[2], Outcome{0b10000});
    EXPECT_EQ(r.circuit->observables[0], Outcome{0});
}

TEST(InteropStim, ElseChainBecomesOnePauliChannel)
{
    // q₁ = 0.1, q₂ = 0.9·0.2, q₃ = 0.9·0.8·0.3 on the strings X·I, Z·I and Y·Y over (q0, q1).
    const ImportResult r = load("e.stim", "E(0.1) X0\nELSE_CORRELATED_ERROR(0.2) Z0\nELSE_CORRELATED_ERROR(0.3) Y0 Y1\n");
    ASSERT_TRUE(r.circuit);
    ASSERT_EQ(r.circuit->ops.size(), 1U);
    const auto& op = r.circuit->ops[0];
    ASSERT_EQ(op.kind, Qputer::OpKind::PauliChannel);
    ASSERT_EQ(op.targets.size(), 2U);
    ASSERT_EQ(op.params.size(), 15U);
    // Term index 4·letter(targets[0]) + letter(targets[1]), letters I X Y Z = 0 1 2 3.
    auto term = [&](int l0, int l1)
    {
        const int a = op.targets[0] == 0 ? l0 : l1, b = op.targets[0] == 0 ? l1 : l0;
        return op.params[static_cast<std::size_t>(4 * a + b - 1)];
    };
    EXPECT_NEAR(term(1, 0), 0.1, kTol);
    EXPECT_NEAR(term(3, 0), 0.18, kTol);
    EXPECT_NEAR(term(2, 2), 0.216, kTol);
    double total = 0;
    for (const double x : op.params) total += x;
    EXPECT_NEAR(total, 0.1 + 0.18 + 0.216, kTol);
}

TEST(InteropStim, ElseChainFrequencies)
{
    const ImportResult r = load("e.stim", "E(0.1) X0\nELSE_CORRELATED_ERROR(0.2) X1\nM 0 1\n");
    ASSERT_TRUE(r.circuit);
    RunOptions o;
    o.shots = 40000;
    o.seed = 3;
    const RunResult res = runCircuit(*r.circuit, o);
    auto rate = [&](Outcome v) { return res.counts.contains(v) ? static_cast<double>(res.counts.at(v)) / 40000.0 : 0.0; };
    expectRate(rate(0b01), 0.1, 40000, "X0");
    expectRate(rate(0b10), 0.18, 40000, "X1");
    EXPECT_EQ(rate(0b11), 0.0);
}

TEST(InteropStim, WideElseChainBecomesKraus)
{
    const ImportResult r = load("k.stim", "E(0.1) X0 X1 X2\nELSE_CORRELATED_ERROR(0.2) X2\nM 0 1 2\n");
    ASSERT_TRUE(r.circuit);
    EXPECT_TRUE(std::ranges::any_of(r.circuit->ops, [](const auto& op) { return op.kind == Qputer::OpKind::Kraus; }));
    RunOptions o;
    o.shots = 40000;
    o.seed = 4;
    const RunResult res = runCircuit(*r.circuit, o);
    auto rate = [&](Outcome v) { return res.counts.contains(v) ? static_cast<double>(res.counts.at(v)) / 40000.0 : 0.0; };
    expectRate(rate(0b111), 0.1, 40000, "XXX");
    expectRate(rate(0b100), 0.18, 40000, "X2");
}

TEST(InteropStim, PauliProductMeasurementReadsTheEigenvalue)
{
    for (const auto& [text, want] : {std::pair{"H 0\nMPP X0*Z1\n", Outcome{0}}, {"H 0\nX 1\nMPP X0*Z1\n", Outcome{1}},
                                     {"H 0\nCX 0 1\nMPP !X0*X1 Z0*Z1\n", Outcome{0b01}}, {"H 0\nCX 0 1\nS 0\nS 1\nMPP Y0*Y1\n", Outcome{0}}, {"H 0\nCX 0 1\nMPP Y0*Y1\n", Outcome{1}}})
    {
        const ImportResult r = load("m.stim", text);
        ASSERT_TRUE(r.circuit);
        RunOptions o;
        o.shots = 200;
        o.seed = 5;
        const RunResult res = runCircuit(*r.circuit, o);
        EXPECT_EQ(res.counts, (Qputer::Counts{{want, 200}})) << text;
    }
}

TEST(InteropStim, NoisyMeasurementFlipsTheRecord)
{
    const ImportResult r = load("f.stim", "M(0.2) 0\n");
    ASSERT_TRUE(r.circuit);
    RunOptions o;
    o.shots = 20000;
    o.seed = 6;
    const RunResult res = runCircuit(*r.circuit, o);
    expectRate(static_cast<double>(res.counts.contains(1) ? res.counts.at(1) : 0) / 20000.0, 0.2, 20000, "flip");
}

TEST(InteropStim, RepeatUnrolls)
{
    const ImportResult r = load("r.stim", "REPEAT 3 {\n H 0\n REPEAT 2 {\n  X 1\n }\n}\n");
    ASSERT_TRUE(r.circuit);
    EXPECT_EQ(r.circuit->ops.size(), 9U);
}

TEST(InteropStim, LimitsAreReportedNotHit)
{
    std::string m65 = "M";
    for (int k = 0; k < 65; ++k) m65 += std::format(" {}", k);
    EXPECT_EQ(codes(load("m.stim", m65 + "\n", "", true)), std::vector<std::string>{"E9004"});
    // Unrolling stops at the cap whether or not the body is empty.
    EXPECT_EQ(codes(load("r.stim", "REPEAT 1000000000 {\n H 0\n}\n", "", true)), std::vector<std::string>{"E9004"});
    EXPECT_EQ(codes(load("r.stim", "REPEAT 1000000000 {\n REPEAT 1000000000 {\n }\n}\n", "", true)), std::vector<std::string>{"E9004"});
}

// ---- OpenQASM ----

TEST(InteropQasm, Qasm2RegisterConditionIsAMaskAndValue)
{
    const ImportResult r = load("c.qasm", "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\ncreg a[1];\ncreg b[2];\n"
                                          "measure q[0] -> a[0];\nif (b == 2) x q[0];\n");
    ASSERT_TRUE(r.circuit);
    const auto& op = r.circuit->ops.back();
    ASSERT_TRUE(op.condition);
    EXPECT_EQ(op.condition->mask, Outcome{0b110});
    EXPECT_EQ(op.condition->value, Outcome{0b100});
}

TEST(InteropQasm, Qasm3ConditionsAndElse)
{
    const ImportResult r = load("c.qasm", "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nbit[2] c;\nc[0] = measure q[0];\n"
                                          "if (!c[0]) { x q[1]; } else { h q[1]; }\nif (c[0] && c == 3) z q[1];\nc[1] = measure q[1];\n");
    ASSERT_TRUE(r.circuit);
    const auto& ops = r.circuit->ops;
    auto withKind = [&](Qputer::OpKind k) { return *std::ranges::find_if(ops, [&](const auto& o) { return o.kind == k; }); };
    EXPECT_EQ(withKind(Qputer::OpKind::X).condition->value, Outcome{0});
    EXPECT_EQ(withKind(Qputer::OpKind::H).condition->value, Outcome{1});
    EXPECT_EQ(withKind(Qputer::OpKind::Z).condition->mask, Outcome{0b11});
    EXPECT_EQ(withKind(Qputer::OpKind::Z).condition->value, Outcome{0b11});
}

TEST(InteropQasm, ContradictoryConditionIsSkippedWithAWarning)
{
    const ImportResult r = load("c.qasm", "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit q;\nbit[2] c;\nc[0] = measure q;\n"
                                          "if (c[0] && c == 2) x q;\n");
    ASSERT_TRUE(r.circuit);
    EXPECT_EQ(codes(r), std::vector<std::string>{"W9002"});
    EXPECT_EQ(r.circuit->ops.size(), 1U);
}

TEST(InteropQasm, LoopsInputsAndMeasureForms)
{
    const ImportResult r = load("l.qasm",
                                "OPENQASM 3.0;\ninclude \"stdgates.inc\";\ninput float theta;\nqubit[4] q;\nbit[4] c;\n"
                                "for int i in [0:2:3] { h q[i]; }\nfor uint i in {1, 3} { rx(theta * i) q[i]; }\n"
                                "c[0] = measure q[0];\nmeasure q[1] -> c[1];\nc[2:3] = measure q[2:3];\n",
                                "", false, {{"theta", 0.25}});
    ASSERT_TRUE(r.circuit);
    const auto& ops = r.circuit->ops;
    ASSERT_EQ(ops.size(), 8U);
    EXPECT_EQ(ops[0].targets, QubitList{0});
    EXPECT_EQ(ops[1].targets, QubitList{2});
    EXPECT_EQ(ops[2].kind, Qputer::OpKind::RX);
    EXPECT_NEAR(ops[2].params[0], 0.25, kTol);
    EXPECT_NEAR(ops[3].params[0], 0.75, kTol);
    for (std::size_t k = 0; k < 4; ++k) EXPECT_EQ(ops[4 + k].clbit, std::optional<std::size_t>{k});
}

TEST(InteropQasm, IgnoredStatementsWarnOncePerKind)
{
    const ImportResult r = load("w.qasm", "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit q;\nbarrier q;\ndelay[10ns] q;\n"
                                          "delay[20ns] q;\nbarrier q;\nh q;\n");
    ASSERT_TRUE(r.circuit);
    EXPECT_EQ(codes(r), std::vector<std::string>{"W9002"});
}

TEST(InteropQasm, FormatMustMatchTheHeader)
{
    EXPECT_EQ(codes(load("v.qasm", "OPENQASM 3.0;\nqubit q;\n", "qasm2", true)), std::vector<std::string>{"E9001"});
    const ImportResult r = load("v.qasm", "qubit q;\nU(0, 0, 0) q;\n", "qasm3");
    ASSERT_TRUE(r.circuit);
    EXPECT_EQ(r.circuit->format, "qasm3");
}

TEST(InteropQasm, IncludesResolveRelativeToTheIncludingFile)
{
    NoetherTest::TempDir tmp;
    NoetherTest::writeText(tmp.path / "lib" / "mine.inc", "gate flip a { U(pi, 0, pi) a; }\n");
    NoetherTest::writeText(tmp.path / "main.qasm", "OPENQASM 3.0;\ninclude \"lib/mine.inc\";\nqubit q;\nflip q;\n");
    const ImportResult r = importFile((tmp.path / "main.qasm").string(), {});
    ASSERT_TRUE(r.circuit) << r.diags->render();
    EXPECT_LE(maxDiffUpToPhase(unitaryOf(*r.circuit), (Eigen::Matrix2cd() << 0, 1, 1, 0).finished()), kTol);
}

TEST(InteropQasm, ResourceLimits)
{
    EXPECT_EQ(codes(load("l.qasm", "OPENQASM 3.0;\nqubit q;\nfor int i in [0:100000000] { U(0, 0, 0) q; }\n", "", true)),
              std::vector<std::string>{"E9004"});
    EXPECT_EQ(codes(load("l.qasm", "OPENQASM 3.0;\nqubit q;\nfor int i in [0:100000000] { }\n", "", true)),
              std::vector<std::string>{"E9004"});
    EXPECT_EQ(codes(load("l.qasm", "OPENQASM 3.0;\nqubit q;\npow(100000000) @ U(0.1, 0, 0) q;\n", "", true)), std::vector<std::string>{});
}

// ---- Circuit JSON ----

TEST(InteropJson, RoundTripPreservesEverything)
{
    const ImportResult a = importFile((kPrograms / "teleport.qasm").string(), {});
    ASSERT_TRUE(a.circuit);
    const Json doc = circuitJson(*a.circuit, *a.diags);
    const ImportResult b = load("t.json", doc.dump(-1), "circuit");
    ASSERT_TRUE(b.circuit);
    const ImportedCircuit &x = *a.circuit, &y = *b.circuit;
    EXPECT_EQ(x.numQubits, y.numQubits);
    EXPECT_EQ(x.numClbits, y.numClbits);
    EXPECT_EQ(x.qubitLabels, y.qubitLabels);
    ASSERT_EQ(x.clbitRegisters.size(), y.clbitRegisters.size());
    ASSERT_EQ(x.ops.size(), y.ops.size());
    for (std::size_t k = 0; k < x.ops.size(); ++k)
    {
        const auto &p = x.ops[k], &q = y.ops[k];
        EXPECT_EQ(p.kind, q.kind) << k;
        EXPECT_EQ(p.controls, q.controls) << k;
        EXPECT_EQ(p.targets, q.targets) << k;
        EXPECT_EQ(p.params, q.params) << k; // shortest round-trip doubles
        EXPECT_EQ(p.clbit, q.clbit) << k;
        EXPECT_EQ(p.condition.has_value(), q.condition.has_value()) << k;
        if (p.condition && q.condition)
        {
            EXPECT_EQ(std::pair(p.condition->mask, p.condition->value), std::pair(q.condition->mask, q.condition->value));
        }
    }
    EXPECT_EQ(circuitJson(y, *b.diags).dump(-1), doc.dump(-1));
}

TEST(InteropJson, CliImportThenRunGivesTheSameCounts)
{
    NoetherTest::TempDir tmp;
    for (const char* name : {"teleport.qasm", "rep_code.stim"})
    {
        const CliResult imp = cli({"import", (kPrograms / name).string()});
        ASSERT_EQ(imp.code, 0) << imp.err;
        const auto json = tmp.path / (std::string(name) + ".json");
        NoetherTest::writeText(json, imp.out);
        const Json a = runJson(kPrograms / name, {"--shots", "500"});
        const Json b = runJson(json, {"--shots", "500"});
        EXPECT_EQ(a.find("counts")->dump(-1), b.find("counts")->dump(-1)) << name;
        if (a.contains("detectors"))
        {
            EXPECT_EQ(a.find("detectors")->dump(-1), b.find("detectors")->dump(-1));
        }
    }
}

TEST(InteropJson, DenseAndChannelOperationsRoundTrip)
{
    std::mt19937_64 rng(13);
    ImportedCircuit c;
    c.format = "circuit";
    c.numQubits = 3;
    c.numClbits = 1;
    c.qubitLabels = {"a", "b", "c"};
    CircuitBuilder b(c);
    b.apply(Seq{denseGate({2, 0}, randomUnitary(2, rng))});
    b.apply(controlled(Seq{denseGate({1}, randomUnitary(1, rng))}, {0}));
    b.pauliChannel({0, 1}, std::vector<double>(15, 0.01));
    const double s = std::sqrt(0.3);
    b.kraus({2}, {(Eigen::Matrix2cd() << 1, 0, 0, std::sqrt(0.7)).finished(), (Eigen::Matrix2cd() << 0, s, 0, 0).finished()});
    b.measure(1, 0);
    Noether::SourceManager sm;
    Noether::Diagnostics d(sm);
    const Json doc = circuitJson(c, d);
    const ImportResult r = load("d.json", doc.dump(-1), "circuit");
    ASSERT_TRUE(r.circuit);
    ASSERT_EQ(r.circuit->ops.size(), c.ops.size());
    for (std::size_t k = 0; k < c.ops.size(); ++k)
    {
        EXPECT_EQ(r.circuit->ops[k].matrix, c.ops[k].matrix) << k;
        EXPECT_EQ(r.circuit->ops[k].kraus.size(), c.ops[k].kraus.size()) << k;
        for (std::size_t j = 0; j < c.ops[k].kraus.size(); ++j) EXPECT_EQ(r.circuit->ops[k].kraus[j], c.ops[k].kraus[j]);
        EXPECT_EQ(r.circuit->ops[k].params, c.ops[k].params) << k;
    }
}

TEST(InteropJson, BadDocumentsAreE9007)
{
    const std::string head = R"({"schema": "noether.circuit/1", "num_qubits": 1, "num_clbits": 0, )";
    for (const std::string& body : {std::string(R"("ops": [{"op": "frobnicate", "targets": [0]}]})"),
                                    std::string(R"("ops": [{"op": "h", "targets": [1]}]})"),
                                    std::string(R"("ops": [{"op": "rx", "targets": [0]}]})"),
                                    std::string(R"("ops": [{"op": "h", "targets": [0]}], "notes": [{"code": "W0001", "message": "x"}]})"),
                                    std::string(R"("ops": [)")})
        EXPECT_EQ(codes(load("b.json", head + body, "circuit", true)), std::vector<std::string>{"E9007"}) << body;
    EXPECT_EQ(codes(load("b.json", R"({"schema": "noether.circuit/2", "num_qubits": 1, "num_clbits": 0, "ops": []})", "circuit", true)),
              std::vector<std::string>{"E9007"});
}

TEST(InteropExamples, EveryExampleRuns)
{
    std::size_t ran = 0;
    for (const auto& e : std::filesystem::directory_iterator(NoetherTest::kExamplesDir / "interop"))
    {
        const auto ext = e.path().extension();
        if (ext != ".qasm" && ext != ".stim") continue;
        const CliResult r = cli({"run", e.path().string(), "--seed", "1", "--param", "theta=1.1", "--deny", "warnings"});
        EXPECT_EQ(r.code, 0) << e.path() << "\n" << r.out << r.err;
        ++ran;
    }
    EXPECT_GE(ran, 3U);
}
