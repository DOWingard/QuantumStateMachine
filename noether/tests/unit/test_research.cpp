// The research layer on the bundled Toffoli T-depth task: candidate checks, evaluation metrics,
// the ranking of candidates, workspace initialisation, the ledger, the spec lock and reports.

#include "TestUtil.hpp"

#include "Tools.hpp"

using namespace Noether;
using namespace NoetherTest;

namespace
{

const std::filesystem::path kSpec = kExamplesDir / "research" / "toffoli_tdepth_spec.ntr";
const std::filesystem::path kAlgo = kExamplesDir / "research" / "toffoli_tdepth_algo.ntr";

// The textbook Toffoli (T-depth 4) and a wrong candidate (a plain CNOT).
const std::string kTextbook = R"(noether 0.1
def Toffoli3_{r[3]}:
    H_{r[2]}
    CNOT_{r[1]→r[2]}; T†_{r[2]}; CNOT_{r[0]→r[2]}; T_{r[2]}; CNOT_{r[1]→r[2]}; T†_{r[2]}; CNOT_{r[0]→r[2]}
    T_{r[1]}; T_{r[2]}; H_{r[2]}; CNOT_{r[0]→r[1]}; T_{r[0]}; T†_{r[1]}; CNOT_{r[0]→r[1]}
)";
const std::string kWrong = "noether 0.1\ndef Toffoli3_{r[3]}: CNOT_{r[0]→r[2]}\n";

Json evalJson(const std::filesystem::path& spec, const std::filesystem::path& cand, int expectExit = 0)
{
    const CliResult r = cli({"eval", "--spec", spec.string(), "--candidate", cand.string(), "--json", "--no-format"});
    EXPECT_EQ(r.code, expectExit) << r.out << r.err;
    auto j = Json::parse(r.out);
    EXPECT_TRUE(j) << r.out;
    return j ? *j : Json();
}

double metric(const Json& j, const char* name) { return j.find("metrics")->find(name)->asDouble(); }

} // namespace

TEST(Research, CheckReportsStructuralMetrics)
{
    const CliResult r = cli({"check", kAlgo.string(), "--spec", kSpec.string(), "--json", "--no-format"});
    ASSERT_EQ(r.code, 0) << r.out << r.err;
    const auto j = Json::parse(r.out);
    EXPECT_TRUE(j->find("ok")->asBool());
    EXPECT_EQ(metric(*j, "tdepth"), 3);
    EXPECT_EQ(metric(*j, "tcount"), 7);
}

TEST(Research, EvalScoresTheCandidate)
{
    const Json j = evalJson(kSpec, kAlgo);
    EXPECT_EQ(j.find("status")->asString(), "ok");
    EXPECT_TRUE(j.find("feasible")->asBool());
    EXPECT_TRUE(j.find("goalMet")->asBool());
    EXPECT_NEAR(metric(j, "fidelity"), 1.0, 1e-9);
    EXPECT_EQ(metric(j, "tdepth"), 3);
    EXPECT_EQ(metric(j, "tcount"), 7);
    EXPECT_EQ(metric(j, "count2q"), 8);
}

TEST(Research, InfeasibleCandidatesNeverMeetTheGoal)
{
    TempDir dir;
    writeText(dir.path / "wrong.ntr", kWrong);
    const Json j = evalJson(kSpec, dir.path / "wrong.ntr");
    EXPECT_FALSE(j.find("feasible")->asBool());
    EXPECT_FALSE(j.find("goalMet")->asBool()) << "a T-depth goal is met trivially by a wrong circuit";
    EXPECT_LT(metric(j, "fidelity"), 0.9);
}

TEST(Research, CandidateErrorsExitWithOne)
{
    TempDir dir;
    writeText(dir.path / "broken.ntr", "noether 0.1\ndef Toffoli3_{r[3]}: CNOT_{r[0]→r[0]}\n");
    const Json j = evalJson(kSpec, dir.path / "broken.ntr", 1);
    EXPECT_EQ(j.find("status")->asString(), "error");
}

TEST(Research, WorkspaceLedgerVerdictsAndLock)
{
    TempDir dir;
    writeText(dir.path / "textbook.ntr", kTextbook);
    // init: spec.lock, a baseline row, the candidate copied in.
    CliResult r = cli({"research", "init", "toff", "--spec", kSpec.string(), "--from", (dir.path / "textbook.ntr").string(), "--dir",
                       dir.path.string(), "--json", "--no-format"});
    ASSERT_EQ(r.code, 0) << r.out << r.err;
    const auto ws = dir.path / "toff";
    for (const char* f : {"spec.ntr", "spec.lock", "algo.ntr", "results.tsv", "program.md", "notes.md", ".gitignore"})
        EXPECT_TRUE(std::filesystem::exists(ws / f)) << f;

    // The same candidate again: equal. The T-depth-3 candidate: improved. The wrong one: infeasible.
    r = cli({"eval", "--dir", ws.string(), "--record", "same again", "--json", "--no-format"});
    ASSERT_EQ(r.code, 0) << r.out;
    EXPECT_EQ(Json::parse(r.out)->find("verdict")->asString(), "equal");

    writeText(ws / "algo.ntr", readText(kAlgo));
    r = cli({"eval", "--dir", ws.string(), "--record", "tdepth 3 construction", "--json", "--no-format"});
    ASSERT_EQ(r.code, 0) << r.out;
    EXPECT_EQ(Json::parse(r.out)->find("verdict")->asString(), "improved");
    EXPECT_TRUE(std::filesystem::exists(ws / "params.json"));

    writeText(ws / "algo.ntr", kWrong);
    r = cli({"eval", "--dir", ws.string(), "--record", "wrong", "--json", "--no-format"});
    EXPECT_EQ(Json::parse(r.out)->find("verdict")->asString(), "infeasible");

    // Ledger: header + baseline + three recorded rows, one runs/<n>.json per evaluation.
    std::istringstream ledger(readText(ws / "results.tsv"));
    std::size_t lines = 0;
    for (std::string l; std::getline(ledger, l);) ++lines;
    EXPECT_EQ(lines, 5U);
    EXPECT_TRUE(std::filesystem::exists(ws / "runs" / "3.json"));

    // Status reports the best row and the goal; the report renders Markdown.
    r = cli({"research", "status", "--dir", ws.string(), "--json"});
    ASSERT_EQ(r.code, 0) << r.out << r.err;
    r = cli({"research", "report", "--dir", ws.string()});
    ASSERT_EQ(r.code, 0) << r.err;
    EXPECT_NE(r.out.find("toffoli-tdepth"), std::string::npos) << r.out;

    // Editing the frozen spec breaks the lock: E8001, exit 6, nothing recorded.
    writeText(ws / "spec.ntr", readText(ws / "spec.ntr") + "\n# a tweak\n");
    r = cli({"eval", "--dir", ws.string(), "--record", "after tampering", "--json", "--no-format"});
    EXPECT_EQ(r.code, 6);
    EXPECT_NE(r.out.find("E8001"), std::string::npos);
    std::istringstream again(readText(ws / "results.tsv"));
    lines = 0;
    for (std::string l; std::getline(again, l);) ++lines;
    EXPECT_EQ(lines, 5U);
}

TEST(Research, CouplingViolationsFailTheCheck)
{
    TempDir dir;
    std::string spec = readText(kSpec);
    spec.replace(spec.find("coupling all"), 12, "coupling line");
    writeText(dir.path / "spec.ntr", spec);
    // r[0]→r[2] is not a neighbouring pair on a line.
    const CliResult r = cli({"check", kAlgo.string(), "--spec", (dir.path / "spec.ntr").string(), "--json", "--no-format"});
    EXPECT_EQ(r.code, 1) << r.out;
    EXPECT_NE(r.out.find("E8004"), std::string::npos);
    EXPECT_NE(r.out.find("{0, 2}"), std::string::npos) << "the message names the pair";
}
