// Golden programs: `noether run --json --no-timing` output compared with the checked-in
// <name>.expected.json. Numbers are compared to 1e-9 (relative) so goldens hold across build types
// and thread counts; everything else must match exactly. Set NOETHER_UPDATE_GOLDENS=1 to rewrite the
// expected files after a deliberate change, then review the diff.
//
// Examples are self-checking: each must run with exit 0 and every assert passing.

#include "TestUtil.hpp"

#include <cstdlib>
#include <regex>

using namespace Noether;
using namespace NoetherTest;

namespace
{

std::vector<std::filesystem::path> ntrFiles(const std::filesystem::path& dir)
{
    std::vector<std::filesystem::path> out;
    if (std::filesystem::is_directory(dir))
        for (const auto& e : std::filesystem::directory_iterator(dir))
            if (e.path().extension() == ".ntr") out.push_back(e.path());
    std::ranges::sort(out);
    return out;
}

std::string testName(const std::filesystem::path& p)
{
    std::string s = p.stem().string();
    for (char& c : s)
        if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
    return s;
}

// Programs without a `seed` statement draw one per run; the harness pins it so the output is reproducible.
CliResult runGolden(const std::filesystem::path& p)
{
    std::vector<std::string> argv{"run", p.string(), "--json", "--no-timing", "--no-format"};
    if (!std::regex_search(readText(p), std::regex(R"((^|[\n;])\s*seed\s)"))) argv.insert(argv.end(), {"--seed", "1"});
    return cli(argv);
}

class Golden : public ::testing::TestWithParam<std::filesystem::path>
{
};

class Example : public ::testing::TestWithParam<std::filesystem::path>
{
};

} // namespace

TEST_P(Golden, MatchesExpectedJson)
{
    const auto& p = GetParam();
    const auto expectedPath = p.parent_path() / (p.stem().string() + ".expected.json");
    const CliResult r = runGolden(p);
    EXPECT_EQ(r.code, 0) << r.err << r.out;
    const auto got = Json::parse(r.out);
    ASSERT_TRUE(got) << r.out;

    if (const char* u = std::getenv("NOETHER_UPDATE_GOLDENS"); u && std::string(u) == "1")
    {
        writeText(expectedPath, got->dump() + "\n");
        GTEST_SKIP() << "rewrote " << expectedPath;
    }
    ASSERT_TRUE(std::filesystem::exists(expectedPath)) << "missing " << expectedPath << "; run with NOETHER_UPDATE_GOLDENS=1";
    const auto want = Json::parse(readText(expectedPath));
    ASSERT_TRUE(want) << expectedPath;
    std::vector<std::string> diffs;
    compareJson(*want, *got, "$", diffs);
    std::string all;
    for (const auto& d : diffs) all += d + "\n";
    EXPECT_TRUE(diffs.empty()) << all;
}

TEST_P(Golden, IsDeterministic)
{
    const CliResult a = runGolden(GetParam());
    const CliResult b = runGolden(GetParam());
    EXPECT_EQ(a.out, b.out);
}

TEST_P(Golden, IsCanonicallyFormatted)
{
    const auto& p = GetParam();
    if (p.filename().string().ends_with("_ascii.ntr")) GTEST_SKIP() << "an ASCII spelling, checked by the formatter tests";
    EXPECT_EQ(cli({"fmt", "--check", p.string()}).code, 0) << p;
}

INSTANTIATE_TEST_SUITE_P(Programs, Golden, ::testing::ValuesIn(ntrFiles(kTestDir / "programs")),
                         [](const auto& p) { return testName(p.param); });

TEST(GoldenSuite, HasTheSpecPrograms)
{
    EXPECT_GE(ntrFiles(kTestDir / "programs").size(), 8U);
}

TEST_P(Example, RunsAndEveryAssertPasses)
{
    const auto& p = GetParam();
    const CliResult r = cli({"run", p.string(), "--json", "--no-timing", "--no-format"});
    EXPECT_EQ(r.code, 0) << r.err << r.out.substr(0, 2000);
    const auto j = Json::parse(r.out);
    ASSERT_TRUE(j) << r.out;
    const Json* asserts = j->find("asserts");
    ASSERT_TRUE(asserts);
    EXPECT_GT(asserts->size(), 0U) << "examples check their results with asserts";
    for (const Json& a : asserts->asArray()) EXPECT_TRUE(a.find("passed")->asBool()) << a.dump(-1);
}

TEST_P(Example, IsCanonicallyFormattedAndWarningFree)
{
    const auto& p = GetParam();
    EXPECT_EQ(cli({"fmt", "--check", p.string()}).code, 0) << p;
    const CliResult r = cli({"check", p.string(), "--no-format", "--deny", "warnings"});
    EXPECT_EQ(r.code, 0) << r.err;
}

INSTANTIATE_TEST_SUITE_P(Examples, Example, ::testing::ValuesIn(ntrFiles(kExamplesDir)),
                         [](const auto& p) { return testName(p.param); });
