// Diagnostic fixtures: tests/diag/<CODE>.ntr triggers exactly that code at the declared position.
// Header comments drive the harness:
//   # expect: E5001 4:1     required: the code and the 1-based line:col of one of its spans
//   # command: run          optional subcommand (default check)
//   # args: --timeout 0.001 optional extra arguments; {dir} is the fixture directory
//   # exit: 3               optional exit code (default 1 for E codes, 0 for W codes)
// Fixtures run on a copy of the directory, so commands that write files cannot touch the sources.
// A fix-it, when offered, must remove the diagnostic once applied.

#include "TestUtil.hpp"

#include "Diagnostics.hpp"

#include <regex>

using namespace Noether;
using namespace NoetherTest;

namespace
{

struct Fixture
{
    std::filesystem::path path;
    std::string code;
    std::uint32_t line = 0, col = 0;
    std::string command = "check";
    std::vector<std::string> args;
    int exit = 0;
};

const std::regex kFixtureName("[EW][0-9]{4}\\.ntr");

std::vector<std::filesystem::path> fixtureFiles()
{
    std::vector<std::filesystem::path> out;
    for (const auto& e : std::filesystem::directory_iterator(kTestDir / "diag"))
        if (std::regex_match(e.path().filename().string(), kFixtureName)) out.push_back(e.path());
    std::ranges::sort(out);
    return out;
}

Fixture readFixture(const std::filesystem::path& p, const std::filesystem::path& dir)
{
    Fixture f;
    f.path = dir / p.filename();
    std::istringstream in(readText(p));
    std::string line;
    for (int n = 0; n < 12 && std::getline(in, line); ++n)
    {
        std::smatch m;
        if (!std::regex_match(line, m, std::regex(R"(#\s*(expect|command|args|exit):\s*(.*?)\s*)"))) continue;
        const std::string key = m[1], value = m[2];
        if (key == "expect")
        {
            std::smatch e;
            if (std::regex_match(value, e, std::regex(R"(([EW]\d{4})\s+(\d+):(\d+))")))
            {
                f.code = e[1];
                f.line = static_cast<std::uint32_t>(std::stoul(e[2]));
                f.col = static_cast<std::uint32_t>(std::stoul(e[3]));
            }
        }
        else if (key == "command") f.command = value;
        else if (key == "args")
        {
            std::istringstream a(value);
            for (std::string w; a >> w;)
            {
                for (std::size_t k; (k = w.find("{dir}")) != std::string::npos;) w.replace(k, 5, dir.string());
                f.args.push_back(w);
            }
        }
        else if (key == "exit") f.exit = std::stoi(value);
    }
    if (!std::regex_search(readText(p), std::regex("#\\s*exit:"))) f.exit = f.code.starts_with("E") ? 1 : 0;
    return f;
}

void copyTree(const std::filesystem::path& from, const std::filesystem::path& to)
{
    std::filesystem::copy(from, to, std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
}

struct Outcome
{
    CliResult r;
    std::optional<Json> json;
    std::set<std::string> codes;
};

Outcome runFixture(const Fixture& f)
{
    std::vector<std::string> argv{f.command, f.path.string(), "--json", "--no-format"};
    argv.insert(argv.end(), f.args.begin(), f.args.end());
    Outcome o;
    o.r = cli(argv);
    o.json = Json::parse(o.r.out);
    if (o.json)
        if (const Json* ds = o.json->find("diagnostics"); ds && ds->isArray())
            for (const Json& d : ds->asArray()) o.codes.insert(d.find("code")->asString());
    return o;
}

class Diag : public ::testing::TestWithParam<std::filesystem::path>
{
};

} // namespace

TEST_P(Diag, TriggersExactlyItsCodeAtItsSpan)
{
    TempDir tmp;
    copyTree(kTestDir / "diag", tmp.path);
    const Fixture f = readFixture(GetParam(), tmp.path);
    ASSERT_FALSE(f.code.empty()) << "no `# expect:` header in " << GetParam();
    ASSERT_EQ(f.code, GetParam().stem().string()) << "the file name must be the expected code";

    const Outcome o = runFixture(f);
    ASSERT_TRUE(o.json) << o.r.out << o.r.err;
    EXPECT_EQ(o.r.code, f.exit) << o.r.out;
    EXPECT_EQ(o.codes, std::set<std::string>{f.code}) << o.r.out;
    bool at = false;
    for (const Json& d : o.json->find("diagnostics")->asArray())
    {
        const Json* sp = d.find("span");
        if (d.find("code")->asString() == f.code && sp && sp->contains("line") && sp->find("line")->asInt() == f.line &&
            sp->find("col")->asInt() == f.col)
            at = true;
    }
    EXPECT_TRUE(at) << "no " << f.code << " at " << f.line << ":" << f.col << "\n" << o.r.out;
}

TEST_P(Diag, FixItRemovesTheDiagnostic)
{
    TempDir tmp;
    copyTree(kTestDir / "diag", tmp.path);
    const Fixture f = readFixture(GetParam(), tmp.path);
    const Outcome before = runFixture(f);
    ASSERT_TRUE(before.json);
    bool hasFix = false;
    for (const Json& d : before.json->find("diagnostics")->asArray())
        if (d.find("code")->asString() == f.code && d.find("fixes") && d.find("fixes")->size() > 0) hasFix = true;
    if (!hasFix) GTEST_SKIP() << f.code << " offers no fix-it";

    const CliResult fix = cli({"fix", f.path.string(), "--apply"});
    EXPECT_NE(fix.code, 2) << fix.err;
    const Outcome after = runFixture(f);
    EXPECT_FALSE(after.codes.contains(f.code)) << "after the fix:\n" << readText(f.path) << "\n" << after.r.out;
}

INSTANTIATE_TEST_SUITE_P(Fixtures, Diag, ::testing::ValuesIn(fixtureFiles()), [](const auto& p) { return p.param.stem().string(); });

TEST(DiagnosticCatalog, EveryCodeHasAFixtureAndAnExplanation)
{
    std::set<std::string> fixtures;
    for (const auto& p : fixtureFiles()) fixtures.insert(p.stem().string());
    for (const CodeInfo& c : codeCatalog())
    {
        const std::string code(c.code);
        EXPECT_TRUE(fixtures.contains(code)) << code << " has no tests/diag fixture";
        EXPECT_FALSE(c.title.empty()) << code;
        EXPECT_GT(c.explanation.size(), 20U) << code;
        const CliResult r = cli({"explain", code});
        EXPECT_EQ(r.code, 0) << code;
        EXPECT_NE(r.out.find(std::string(c.title)), std::string::npos) << code;
    }
    for (const auto& f : fixtures) EXPECT_NE(findCode(f), nullptr) << f << " is not in the catalog";
}

TEST(DiagnosticCatalog, CodesAreSortedAndUnique)
{
    std::vector<std::string> codes;
    for (const CodeInfo& c : codeCatalog()) codes.emplace_back(c.code);
    EXPECT_TRUE(std::ranges::is_sorted(codes));
    EXPECT_EQ(std::ranges::adjacent_find(codes), codes.end());
}

TEST(DiagnosticCatalog, WrongExamplesTriggerTheirCode)
{
    // A one-line `wrong:` example must produce its code. Multi-line examples are covered by the
    // fixtures; these need more context than a shared prelude gives (a proc, an odd N, a register
    // of the right size).
    const std::set<std::string> needsContext{"E4003", "E5006", "E5007"};
    const std::string prelude = "noether 0.1\nqubits q[4]; bits c[4]\nlet x = 1, t = 0.5\nparam θ ∈ [-π, π] = 0.5\n";
    std::size_t checked = 0;
    for (const CodeInfo& c : codeCatalog())
    {
        const std::string text(c.explanation);
        const auto w = text.find("wrong:  ");
        if (w == std::string::npos || needsContext.contains(std::string(c.code))) continue;
        const auto end = text.find('\n', w);
        if (end != std::string::npos && end + 1 < text.size() && (text[end + 1] == ' ' || text[end + 1] == '\t')) continue;
        std::string line = text.substr(w + 8, end == std::string::npos ? std::string::npos : end - w - 8);
        // Drop trailing commentary such as "     (if 1/(2π) was meant)".
        if (const auto note = line.find("  ("); note != std::string::npos) line.resize(note);
        if (line.ends_with(':')) continue; // a suite header needs a body
        const Executed run = runSource(prelude + line + "\n");
        const auto codes = run.codes();
        EXPECT_NE(std::ranges::find(codes, std::string(c.code)), codes.end()) << c.code << " example `" << line << "` gave:\n" << run.diagnostics;
        ++checked;
    }
    EXPECT_GT(checked, 20U);
}

TEST(DiagnosticCatalog, JsonDiagnosticsCarrySpansAndFixes)
{
    const CliResult r = cli({"check", (kTestDir / "diag" / "E3001.ntr").string(), "--json", "--no-format"});
    const auto j = Json::parse(r.out);
    ASSERT_TRUE(j);
    EXPECT_EQ(j->find("schema")->asString(), "noether.check/1");
    const Json& d = j->find("diagnostics")->asArray().at(0);
    for (const char* k : {"code", "severity", "message", "span", "notes", "fixes"}) EXPECT_TRUE(d.contains(k)) << k;
    for (const char* k : {"file", "line", "col", "endLine", "endCol"}) EXPECT_TRUE(d.find("span")->contains(k)) << k;
}
