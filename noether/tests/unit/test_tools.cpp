// Tools built on a compiled program: basis decomposition, equivalence checking, optimisation and
// gradients, resource estimates, OpenQASM export, diagrams, the skill text, the REPL and the
// command-line surface.

#include "TestUtil.hpp"

#include "Decompose.hpp"
#include "Estimate.hpp"
#include "Lowering.hpp"
#include "Tools.hpp"

#include <cstdlib>

using namespace Noether;
using namespace NoetherTest;

namespace
{

std::vector<IrOp> opsOf(const std::string& body, std::size_t n)
{
    const Executed run = runSource(std::format("noether 0.1\nbackend statevector\nqubits q[{}]\n{}\n", n, body));
    EXPECT_TRUE(run.compiled) << body << "\n" << run.diagnostics;
    std::vector<IrOp> out;
    if (run.compiled)
        for (const Event& ev : run.c->compiler->ir().events)
            if (ev.kind == EvK::Op) out.push_back(ev.op);
    return out;
}

QubitList allQubits(std::size_t n)
{
    QubitList q;
    for (std::size_t k = 0; k < n; ++k) q.push_back(static_cast<Qubit>(k));
    return q;
}

// |tr(A†B)| / dim: 1 exactly when A and B agree up to a global phase.
double phaseFreeOverlap(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b)
{
    return std::abs((a.adjoint() * b).trace()) / static_cast<double>(a.rows());
}

std::unique_ptr<Compilation> compileSource(const std::string& src)
{
    auto c = compileText("t.ntr", src, {});
    EXPECT_FALSE(c->diags->hasErrors()) << c->diags->render();
    return c;
}

} // namespace

// ---- Decomposition into gate bases ----

TEST(Decompose, EveryRuleIsExactUpToGlobalPhase)
{
    const std::vector<std::vector<std::string>> bases{
        {"H", "S", "T", "CNOT"},           {"H", "S", "S†", "T", "T†", "CNOT"}, {"Rz", "√X", "CNOT"},
        {"Rx", "Ry", "Rz", "CZ"},          {"U3", "CNOT"},                      {"H", "CP", "Rz"},
        {"P", "H", "CZ"},                  {"Rz", "Ry", "CNOT"},
    };
    const std::vector<std::string> bodies{
        "X_0", "Y_0", "Z_0", "H_0", "S_0", "S†_0", "T_0", "T†_0", "√X_0", "√X†_0", "Rx(0.37)_0", "Ry(0.37)_0", "Rz(0.37)_0",
        "P(0.37)_0", "U3(0.37, 1.1, -0.6)_0", "Rz(3π/4)_0", "Rx(π/4)_1", "Ry(-π/2)_2", "CNOT_{2→0}", "CZ_{0, 1}", "CP(0.37)_{1, 2}",
        "CP(π/2)_{0, 2}", "SWAP_{0, 2}", "Toffoli_{0, 1→2}", "Fredkin_{2→0, 1}", "C_{0}(H_1)", "C_{2}(Ry(0.37)_0)", "C_{¬1}(X_0)",
        "C_{0, 1}(Z_2)", "C_{0}(U3(0.37, 1.1, -0.6)_2)", "e^{-i 0.37 X_0 Y_2}",
    };
    const QubitList support = allQubits(3);
    for (const auto& basis : bases)
    {
        std::vector<std::string> unknown;
        const BasisLowering bl(basis, unknown);
        ASSERT_TRUE(unknown.empty());
        for (const std::string& body : bodies)
        {
            const std::vector<IrOp> ops = opsOf(body, 3);
            std::vector<IrOp> lowered;
            std::string error;
            bool ok = true;
            for (const IrOp& op : ops) ok = ok && bl.lower(op, lowered, error);
            if (!ok) continue; // not expressible in this basis (checked below for the exact ones)
            for (const IrOp& op : lowered)
            {
                const std::string name = basisName(op);
                EXPECT_TRUE(std::ranges::find(basis, name) != basis.end()) << body << " lowered to " << name << " outside the basis";
            }
            const auto want = unitaryOf(ops, support, {}), got = unitaryOf(lowered, support, {});
            EXPECT_NEAR(phaseFreeOverlap(want, got), 1.0, 1e-10) << body << " in {" << basis.front() << ", …}";
        }
    }
}

TEST(Decompose, ExactAnglesReachCliffordPlusT)
{
    std::vector<std::string> unknown;
    const BasisLowering bl({"H", "S", "T", "CNOT"}, unknown);
    for (const std::string body : {"Rz(π/4)_0", "Rx(3π/4)_0", "Ry(π/2)_0", "CP(π/2)_{0, 1}", "T†_0", "S†_0", "Toffoli_{0, 1→2}"})
    {
        std::vector<IrOp> lowered;
        std::string error;
        for (const IrOp& op : opsOf(body, 3)) EXPECT_TRUE(bl.lower(op, lowered, error)) << body << ": " << error;
    }
    std::vector<IrOp> lowered;
    std::string error;
    for (const IrOp& op : opsOf("Rz(0.37)_0", 3)) EXPECT_FALSE(bl.lower(op, lowered, error));
    EXPECT_FALSE(error.empty());
}

TEST(Decompose, ToffoliUsesSevenTGates)
{
    // The Nielsen & Chuang construction; T† must cost one T-type gate (S†·T), not seven.
    for (const std::vector<std::string>& basis : {std::vector<std::string>{"H", "S", "T", "CNOT"}, {"H", "S", "S†", "T", "T†", "CNOT"}})
    {
        std::vector<std::string> unknown;
        const BasisLowering bl(basis, unknown);
        std::vector<IrOp> lowered;
        std::string error;
        for (const IrOp& op : opsOf("Toffoli_{0, 1→2}", 3)) ASSERT_TRUE(bl.lower(op, lowered, error)) << error;
        const auto tcount = std::ranges::count_if(lowered, [](const IrOp& o) { return isTGate(o); });
        EXPECT_EQ(tcount, 7) << basis.size();
        EXPECT_EQ(std::ranges::count_if(lowered, [](const IrOp& o) { return o.kind == GK::CNOT; }), 6);
    }
}

// ---- Equivalence ----

TEST(Equivalence, UnitaryMethod)
{
    auto a = compileSource("noether 0.1\nqubits q[2]\nH_0 X_0 H_0\n");
    auto b = compileSource("noether 0.1\nqubits q[2]\nZ_0\n");
    auto c = compileSource("noether 0.1\nqubits q[2]\nX_0\n");
    const EquivResult same = equivalent(a->compiler->ir(), b->compiler->ir(), false);
    EXPECT_TRUE(same.equivalent) << same.error;
    EXPECT_EQ(same.method, "unitary");
    const EquivResult diff = equivalent(a->compiler->ir(), c->compiler->ir(), false);
    EXPECT_FALSE(diff.equivalent);
    EXPECT_FALSE(diff.counterexample.empty());
}

TEST(Equivalence, GlobalPhaseIsOptional)
{
    auto a = compileSource("noether 0.1\nqubits q[1]\nRz(0.37)_0\n");
    auto b = compileSource("noether 0.1\nqubits q[1]\nP(0.37)_0\n");
    EXPECT_FALSE(equivalent(a->compiler->ir(), b->compiler->ir(), false).equivalent);
    EXPECT_TRUE(equivalent(a->compiler->ir(), b->compiler->ir(), true).equivalent);
}

TEST(Equivalence, TableauMethodForLargeCliffordCircuits)
{
    auto a = compileSource("noether 0.1\nqubits q[40]\nfor k ∈ 0..38: SWAP_{k, k + 1}\n");
    auto b = compileSource("noether 0.1\nqubits q[40]\nfor k ∈ 0..38: CNOT_{k→k + 1} CNOT_{k + 1→k} CNOT_{k→k + 1}\n");
    auto c = compileSource("noether 0.1\nqubits q[40]\nfor k ∈ 0..38: CNOT_{k→k + 1} CNOT_{k + 1→k}\n");
    const EquivResult same = equivalent(a->compiler->ir(), b->compiler->ir(), true);
    EXPECT_TRUE(same.equivalent) << same.error;
    EXPECT_EQ(same.method, "tableau");
    EXPECT_FALSE(equivalent(a->compiler->ir(), c->compiler->ir(), true).equivalent);
}

TEST(Equivalence, RandomStatesForMidsizeCircuits)
{
    auto a = compileSource("noether 0.1\nqubits q[14]\nfor k ∈ 0..12: CNOT_{k→k + 1}\nT_13\n");
    auto b = compileSource("noether 0.1\nqubits q[14]\nfor k ∈ 0..12: CNOT_{k→k + 1}\nP(π/4)_13\n");
    auto c = compileSource("noether 0.1\nqubits q[14]\nfor k ∈ 0..12: CNOT_{k→k + 1}\nS_13\n");
    const EquivResult same = equivalent(a->compiler->ir(), b->compiler->ir(), false);
    EXPECT_TRUE(same.equivalent) << same.error;
    EXPECT_EQ(same.method, "random-states");
    EXPECT_FALSE(equivalent(a->compiler->ir(), c->compiler->ir(), false).equivalent);
}

// ---- Optimisation and gradients ----

TEST(Optimize, MinimisesALabelledReadout)
{
    auto c = compileSource("noether 0.1\nqubits q[2]\nparam θ[2] ∈ [-π, π] = 0.1\nRy(θ[0])_0 Ry(θ[1])_1\nprint ⟨Z_0 + Z_1⟩ as e\n");
    for (const std::string method : {"nelder-mead", "spsa", "adam", "lbfgs"})
    {
        OptOptions o;
        o.label = "e";
        o.method = method;
        o.restarts = 2;
        o.iterations = 600;
        o.exec.timing = false;
        const OptResult r = optimizeParams(*c->compiler, o);
        EXPECT_TRUE(r.error.empty()) << r.error;
        EXPECT_NEAR(r.best, -2.0, method == "spsa" ? 2e-2 : 1e-4) << method;
    }
}

TEST(Optimize, ParameterShiftGradientIsExact)
{
    auto c = compileSource("noether 0.1\nqubits q[2]\nparam θ ∈ [-π, π] = 0.4\nparam φ ∈ [-π, π] = -0.3\n"
                           "Ry(θ)_0 Rx(2φ)_1 Ry(θ)_1\nprint ⟨Z_0 + Z_1⟩ as e\n");
    ExecOptions eo;
    eo.timing = false;
    const GradResult g = gradientOf(*c->compiler, "e", eo);
    ASSERT_TRUE(g.error.empty()) << g.error;
    ASSERT_EQ(g.gradient.size(), 2U);
    // e = cos θ + cos(2φ) cos θ (Rx then Ry on qubit 1 from |0⟩).
    const double th = 0.4, ph = -0.3;
    EXPECT_NEAR(g.value, std::cos(th) + std::cos(2 * ph) * std::cos(th), 1e-12);
    EXPECT_NEAR(g.gradient[0], -std::sin(th) - std::cos(2 * ph) * std::sin(th), 1e-10);
    EXPECT_NEAR(g.gradient[1], -2 * std::sin(2 * ph) * std::cos(th), 1e-10);
}

TEST(Optimize, ControlledRotationsFallBackToFiniteDifferences)
{
    auto c = compileSource("noether 0.1\nqubits q[2]\nparam θ ∈ [-π, π] = 0.8\nX_0\nC_{0}(Ry(θ)_1)\nprint ⟨Z_1⟩ as e\n");
    ExecOptions eo;
    eo.timing = false;
    const GradResult g = gradientOf(*c->compiler, "e", eo);
    ASSERT_TRUE(g.error.empty()) << g.error;
    EXPECT_NEAR(g.gradient.at(0), -std::sin(0.8), 1e-5);
}

TEST(Optimize, CliWritesAParamsFileThatRunReads)
{
    TempDir dir;
    const auto prog = dir.path / "v.ntr";
    writeText(prog, "noether 0.1\nqubits q[1]\nparam θ ∈ [-π, π] = 0.3\nRy(θ)_0\nprint ⟨Z_0⟩ as e\n");
    const auto params = dir.path / "p.json";
    const CliResult opt = cli({"opt", prog.string(), "--minimize", "e", "--out", params.string(), "--json", "--no-format"});
    ASSERT_EQ(opt.code, 0) << opt.err;
    const CliResult run = cli({"run", prog.string(), "--params", params.string(), "--json", "--no-timing", "--no-format"});
    ASSERT_EQ(run.code, 0) << run.err;
    const auto j = Json::parse(run.out);
    EXPECT_NEAR(j->find("prints")->asArray()[0].find("value")->asDouble(), -1.0, 1e-6);
    // An explicit --set wins over the file.
    const CliResult set = cli({"run", prog.string(), "--params", params.string(), "--set", "theta=0", "--json", "--no-timing", "--no-format"});
    EXPECT_NEAR(Json::parse(set.out)->find("prints")->asArray()[0].find("value")->asDouble(), 1.0, 1e-12);
}

// ---- Estimates, export, views ----

TEST(Tools, ResourceEstimate)
{
    auto c = compileSource("noether 0.1\nqubits q[3]\nH_0 CNOT_{0→1} T_1 CNOT_{1→2} T†_2\n");
    const Resources r = resources(c->compiler->ir());
    EXPECT_EQ(r.ops, 5U);
    EXPECT_EQ(r.count2q, 2U);
    EXPECT_EQ(r.tcount, 2U);
    EXPECT_EQ(r.depth, 5U);
    EXPECT_FALSE(r.clifford);
    const CliResult est = cli({"estimate", (kTestDir / "programs" / "02_ghz_stabilizer.ntr").string(), "--json", "--no-format"});
    ASSERT_EQ(est.code, 0) << est.err;
    const auto j = Json::parse(est.out);
    ASSERT_TRUE(j);
    EXPECT_EQ(j->find("backend")->asString(), "stabilizer");
}

TEST(Tools, QasmExport)
{
    auto c = compileSource("noether 0.1\nqubits q[2]; bits c[2]\nH_0 CNOT_{0→1}\nRz(0.25)_1\nc ← measure_q\n");
    std::string error;
    const std::string q = qasmExport(c->compiler->ir(), error);
    EXPECT_TRUE(error.empty()) << error;
    for (const std::string s : {"OPENQASM 3", "h q[0];", "cx q[0], q[1];", "rz(0.25) q[1];", "measure q[0]"}) EXPECT_NE(q.find(s), std::string::npos) << s << "\n" << q;
}

TEST(Tools, DrawAndIr)
{
    auto c = compileSource("noether 0.1\nqubits q[2]\nH_0 CNOT_{0→1}\n");
    const std::string text = drawText(c->compiler->ir(), false);
    EXPECT_NE(text.find("H"), std::string::npos) << text;
    EXPECT_EQ(drawJson(c->compiler->ir()).size(), 2U);
    EXPECT_TRUE(irJson(*c->compiler).contains("ops"));
    TempDir dir;
    writeText(dir.path / "p.ntr", "noether 0.1\nqubits q[2]\nCNOT_{0→1} H_0\n");
    const auto ir = Json::parse(cli({"ir", (dir.path / "p.ntr").string(), "--json"}).out);
    ASSERT_TRUE(ir);
    EXPECT_EQ(ir->find("schema")->asString(), "noether.ir/1");
    const auto draw = Json::parse(cli({"draw", (dir.path / "p.ntr").string(), "--json"}).out);
    ASSERT_TRUE(draw);
    EXPECT_EQ(draw->find("layers")->size(), 2U);
    EXPECT_NE(drawText(c->compiler->ir(), true).find("H"), std::string::npos);
}

// ---- Agent skills and the REPL ----

TEST(Tools, SkillTextIsGeneratedFromTheTables)
{
    const std::string brief = skillText(false), full = skillText(true);
    EXPECT_NE(brief.find("name: qsm"), std::string::npos);
    EXPECT_GT(full.size(), brief.size());
#define NTR_GATE(name, roles, params, clifford, meaning) EXPECT_NE(full.find(std::string("`") + name + "`"), std::string::npos) << name;
#include "tokens.def"
    for (const CodeInfo& c : codeCatalog()) EXPECT_NE(full.find(std::string(c.code)), std::string::npos) << c.code;
    TempDir dir;
    std::string error;
    EXPECT_EQ(installSkills(dir.path.string(), error), 5U) << error;
    for (const char* f : {"qsm/SKILL.md", "qsm/reference.md", "qsm-research/SKILL.md", "qsm-tune/SKILL.md", "qsm-tune/reference.md"})
        EXPECT_TRUE(std::filesystem::exists(dir.path / f)) << f;
}

TEST(Tools, CheckedInSkillReferenceIsCurrent)
{
    // skills/qsm/reference.md is generated from the token table, grammar and diagnostic catalog;
    // NOETHER_UPDATE_GOLDENS=1 rewrites it after a deliberate change.
    TempDir dir;
    std::string error;
    ASSERT_EQ(installSkills(dir.path.string(), error), 5U) << error;
    const std::string generated = readText(dir.path / "qsm" / "reference.md");
    const auto checkedIn = kSkillsDir / "qsm" / "reference.md";
    if (const char* u = std::getenv("NOETHER_UPDATE_GOLDENS"); u && std::string(u) == "1")
    {
        writeText(checkedIn, generated);
        GTEST_SKIP() << "rewrote " << checkedIn;
    }
    EXPECT_EQ(readText(checkedIn), generated) << checkedIn << " is stale; run with NOETHER_UPDATE_GOLDENS=1";
}

TEST(Tools, ReplKeepsSessionState)
{
    std::istringstream in("qubits q[1]\nX_0\nprint ⟨Z_0⟩ as z\nlet = 2\n:source\n:quit\n");
    std::ostringstream out, err;
    EXPECT_EQ(runRepl(in, out, err, {}), 0);
    EXPECT_NE(out.str().find("z = -1"), std::string::npos) << out.str();
    EXPECT_NE(err.str().find("E2001"), std::string::npos) << err.str(); // the bad line is reported and dropped
    EXPECT_NE(out.str().find("X_0"), std::string::npos);
}

// ---- Command line ----

TEST(Cli, ExitCodes)
{
    TempDir dir;
    const auto ok = dir.path / "ok.ntr", bad = dir.path / "bad.ntr", fail = dir.path / "fail.ntr";
    writeText(ok, "noether 0.1\nqubits q[1]\nX_0\nassert ⟨Z_0⟩ ≈ -1\n");
    writeText(bad, "noether 0.1\nqubits q[1]\nCNOT_{0→0}\n");
    writeText(fail, "noether 0.1\nqubits q[1]\nassert ⟨Z_0⟩ ≈ -1\n");
    EXPECT_EQ(cli({"run", ok.string()}).code, 0);
    EXPECT_EQ(cli({"run", bad.string()}).code, 1);
    EXPECT_EQ(cli({"run", fail.string()}).code, 3);
    EXPECT_EQ(cli({"run", (dir.path / "missing.ntr").string()}).code, 2);
    EXPECT_EQ(cli({"frobnicate"}).code, 2);
    EXPECT_EQ(cli({"run", ok.string(), "--bogus"}).code, 2);
    // A non-Clifford program needs the state vector, whose memory --max-mem bounds.
    const auto big = dir.path / "big.ntr";
    writeText(big, "noether 0.1\nqubits q[4]\nT_0\n");
    EXPECT_EQ(cli({"run", big.string(), "--max-mem", "16"}).code, 4);
}

TEST(Cli, SelfDescription)
{
    EXPECT_NE(cli({"version"}).out.find(std::string(kVersion)), std::string::npos);
    const std::string tokens = cli({"tokens"}).out;
    EXPECT_NE(tokens.find("\\dagger"), std::string::npos);
    EXPECT_EQ(tokens.find("\\\\dagger"), std::string::npos) << "LaTeX commands are listed with one backslash";
    EXPECT_NE(cli({"grammar"}).out.find("program"), std::string::npos);
    EXPECT_EQ(cli({"explain", "E5001"}).code, 0);
    EXPECT_NE(cli({"explain", "E9999"}).code, 0);
    EXPECT_EQ(cli({"help"}).code, 0);
}
