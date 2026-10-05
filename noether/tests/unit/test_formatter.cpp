// Formatter properties over a corpus (golden programs, examples, the bundled library and a
// construct-dense ASCII program): idempotence, parse ∘ fmt = parse on the syntax tree, the
// Unicode → ASCII → Unicode round trip, and the automatic canonicalisation the CLI performs.

#include "TestUtil.hpp"

#include "StdLib.hpp"

#include <sys/stat.h>

using namespace Noether;
using namespace NoetherTest;

namespace
{

// Every construct of the language in non-canonical spellings (LaTeX, ASCII, aliases).
const std::string kKitchenSink = R"(noether 0.1
# noether: allow W0004
qubits q[4], anc[1]; bits c[2], d[1]; seed 5
backend statevector
trajectories 20
let theta = pi/3, phi = \frac{\pi}{5}
let J = 1.0, g = 0.5
let Ham = -J \sum_{j=0}^{2} Z_j Z_{j+1} - g \sum_{j=0}^{3} X_j
let U = [[1, 0], [0, exp(i*phi)]]
let Bell_{a,b} = CX_{a->b} H_a
param alpha in [-pi, pi] = 0.25
param beta[2] in [0, 1] = 0
## A doc comment on the def.
def Layer(t)_{r[n]}:
    for k in 0..n-1 by 2:
        RZ(t)_{r[k]}
    for k in [1, 3]:
        Phase(t/2)_{r[k]}
proc Fix_{r[2]}:
    d[0] <- measure Z_{r[0]} Z_{r[1]}
    if d[0] and not c[0]: X_{r[0]}
    else: I_{r[0]}
noise:
    after gate1: depolarize(0.001)
    before measure: flip(0.002)
prepare (cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>) \otimes |000> \otimes |0>
H_q Sdg_0 Tdg_1 SX_2 RX(alpha)_3 RY(beta[0])_0 U3(0.1, 0.2, 0.3)_1
CPhase(pi/4)_{0,1} SWAP_{2,3} CCX_{0,1->2} CSWAP_{0->1,2} CZ_{1,2}
C_{0, not 1}(U_{anc}) C_{q[0..1]}(Z_{q[2]}) Layer(alpha)_q Bell_{0,1}
exp(-i*theta/2 * X_0 X_1) S\dagger_2 (H_0)^2 T^\dagger_3
Fix_{q[0..1]}
c[0] <- measure Z_0
c[1] <- measure X_1 Y_2
reset_{q[3]}
print \expval{Ham} as energy, <01010|psi>, |<11110|psi>|^2, abs(-2), sqrt(4), entropy(rho_{q[0..1]})
print \langle Z_0 Z_1 \rangle, fidelity(|00000>), count(H), depth()
assert \expval{Z_0} ~= 0 +- 1
assert 1 <= 2 and 2 >= 1 and 1 != 2 and not false
counts <- run 100
print marginal(counts, c), floor(2.5), ceil(2.5), min(1, 2), max(1, 2), log(1), tan(0)
)";

std::vector<std::pair<std::string, std::string>> corpus()
{
    std::vector<std::pair<std::string, std::string>> out{{"kitchen-sink", kKitchenSink}};
    for (const auto& dir : {kTestDir / "programs", kExamplesDir, kExamplesDir / "research"})
    {
        if (!std::filesystem::is_directory(dir)) continue;
        for (const auto& e : std::filesystem::directory_iterator(dir))
            if (e.path().extension() == ".ntr") out.emplace_back(e.path().filename().string(), readText(e.path()));
    }
    for (const std::string m : {"std/qft.ntr", "std/grover.ntr", "std/arith.ntr", "std/gates.ntr"})
    {
        const auto src = stdSource(m);
        EXPECT_TRUE(src.has_value()) << m;
        if (src) out.emplace_back(m, std::string(*src));
    }
    return out;
}

std::size_t countComments(const std::string& text)
{
    std::size_t n = 0;
    for (std::size_t p = text.find('#'); p != std::string::npos; p = text.find('#', p + 1)) ++n;
    return n;
}

} // namespace

TEST(Formatter, CorpusParsesCleanly)
{
    for (const auto& [name, text] : corpus())
    {
        Parsed p = parseText(text);
        EXPECT_TRUE(p.ok()) << name << "\n" << p.d->render();
    }
}

TEST(Formatter, Idempotent)
{
    for (const auto& [name, text] : corpus())
    {
        const std::string once = format(text);
        EXPECT_EQ(format(once), once) << name;
        const std::string asciiOnce = format(text, true);
        EXPECT_EQ(format(asciiOnce, true), asciiOnce) << name;
    }
}

TEST(Formatter, PreservesTheSyntaxTree)
{
    for (const auto& [name, text] : corpus())
    {
        const std::string tree = astDump(text);
        EXPECT_EQ(astDump(format(text)), tree) << name << "\n" << format(text);
        EXPECT_EQ(astDump(format(text, true)), tree) << name << "\n" << format(text, true);
    }
}

TEST(Formatter, UnicodeAsciiUnicodeRoundTripIsIdentity)
{
    for (const auto& [name, text] : corpus())
    {
        const std::string unicode = format(text);
        EXPECT_EQ(format(format(unicode, true)), unicode) << name;
    }
}

TEST(Formatter, KeepsEveryComment)
{
    for (const auto& [name, text] : corpus())
    {
        // '#' only appears in comments in this corpus.
        EXPECT_EQ(countComments(format(text)), countComments(text)) << name;
        EXPECT_EQ(countComments(format(text, true)), countComments(text)) << name;
    }
}

TEST(Formatter, AsciiCanonicalOutputIsPureAscii)
{
    for (const auto& [name, text] : corpus())
    {
        std::string ascii = format(text, true);
        // Comments may hold any text; strip them before checking.
        std::string code;
        for (std::size_t b = 0; b < ascii.size();)
        {
            const std::size_t e = ascii.find('\n', b);
            std::string line = ascii.substr(b, e - b);
            if (const auto h = line.find('#'); h != std::string::npos) line.resize(h);
            code += line;
            if (e == std::string::npos) break;
            b = e + 1;
        }
        for (const char c : code) EXPECT_LT(static_cast<unsigned char>(c), 0x80) << name << ": " << code;
    }
}

TEST(Formatter, TeleportationAsciiFormatsToTheUnicodeGolden)
{
    // The ASCII spelling has no comments; compare code only.
    const std::string ascii = format(readText(kTestDir / "programs" / "01_teleportation_ascii.ntr"));
    std::string unicode = readText(kTestDir / "programs" / "01_teleportation.ntr");
    for (std::size_t h; (h = unicode.find("  #")) != std::string::npos;) unicode.erase(h, unicode.find('\n', h) - h);
    EXPECT_EQ(ascii, unicode);
}

TEST(Formatter, CanonicalSpellings)
{
    struct Case
    {
        std::string input, canonical;
    };
    const std::string pre = "noether 0.1\nqubits q[3]; bits c[1]\nlet x = 2\n";
    for (const Case& k : std::vector<Case>{
             {"print abs(x - 3)", "print |x - 3|"},
             {"print exp(i x)", "print e^{i x}"},
             {"print sqrt(x)", "print √(x)"},
             {"print \\sqrt{x}", "print √(x)"},
             {"print \\frac{x}{3}", "print (x)/(3)"},
             {"print \\expval{Z_0}", "print ⟨Z_0⟩"},
             {"print \\langle Z_0 \\rangle", "print ⟨Z_0⟩"},
             {"print <010|psi>", "print ⟨010|ψ⟩"},
             {"print \\braket{010|\\psi}", "print ⟨010|ψ⟩"},
             {"prepare \\ket{0} \\otimes |00>", "prepare |0⟩ ⊗ |00⟩"},
             {"prepare |0>^{\\otimes 3}", "prepare |0⟩^⊗3"},
             {"Sdg_0 Tdg_1 SX_2", "S†_0 T†_1 √X_2"},
             {"S^\\dagger_0", "S†_0"},
             {"CX_{0->1} CCX_{0,1->2} CSWAP_{0->1,2}", "CNOT_{0→1} Toffoli_{0,1→2} Fredkin_{0→1,2}"},
             {"RX(x)_0 RY(x)_1 RZ(x)_2 Phase(x)_0 CPhase(x)_{0,1}", "Rx(x)_0 Ry(x)_1 Rz(x)_2 P(x)_0 CP(x)_{0,1}"},
             {"c[0] <- measure Z_0", "c[0] ← measure Z_0"},
             {"c[0] \\gets measure Z_0", "c[0] ← measure Z_0"},
             {"assert x ~= 2 +- 0.1", "assert x ≈ 2 ± 0.1"},
             {"assert x \\approx 2 \\pm 0.1", "assert x ≈ 2 ± 0.1"},
             {"assert x <= 2 and x >= 2 and x != 3", "assert x ≤ 2 and x ≥ 2 and x ≠ 3"},
             {"assert x \\le 2 and x \\geq 2 and x \\neq 3", "assert x ≤ 2 and x ≥ 2 and x ≠ 3"},
             {"assert not false", "assert ¬false"},
             {"assert \\neg false", "assert ¬false"},
             {"print 2*x, 2 \\cdot x, 2 \\times x, 2 × x, 2 ⋅ x", "print 2·x, 2·x, 2·x, 2·x, 2·x"},
             {"print x − 1", "print x - 1"},
             {"print \\sum_{j=0}^{2} j, ∑_{j=0}^2 j, \\prod_{j=1}^{3} j", "print Σ_{j=0}^{2} j, Σ_{j=0}^{2} j, ∏_{j=1}^{3} j"},
             {"print 2^{3}, 2^{3 + 1}, 2^(3 + 1), 2^-1, 2^3^2", "print 2^3, 2^{3 + 1}, 2^(3 + 1), 2^-1, 2^3^2"},
             {"CNOT_{0 \\to 1} CNOT_{1 \\rightarrow 2} CNOT_{0 ⟶ 1}", "CNOT_{0→1} CNOT_{1→2} CNOT_{0→1}"},
             {"for k in 0..2: X_k", "for k ∈ 0..2: X_k"},
             {"print theta_0", "print θ_0"},
             {"print \\theta + \\vartheta + \\varphi + \\phi + \\varepsilon", "print θ + θ + φ + φ + ε"},
             {"print \\sin(x) + \\cos(x) + \\ln(x)", "print sin(x) + cos(x) + log(x)"},
             {"print x \\, + \\; 1 \\quad", "print x + 1"},
             {"print X_{10}", "print X_{10}"},
             {"print 2 pi + 2 x + 0.5 x", "print 2π + 2x + 0.5x"},
             {"print e^{2 pi i * 0.25}", "print e^{2πi·0.25}"},
         })
    {
        Parsed p = parseText(pre + k.input + "\n");
        ASSERT_TRUE(p.ok()) << k.input << "\n" << p.d->render();
        const std::string out = formatProgram(p.prog, p.sm->file(p.id), false);
        EXPECT_EQ(out.substr(pre.size()), k.canonical + "\n") << "input: " << k.input;
    }
}

TEST(Formatter, AutoFormatsOnRunAndHonoursNoFormat)
{
    TempDir dir;
    const auto path = dir.path / "prog.ntr";
    const std::string ascii = "noether 0.1\nqubits q[2]\nCX_{0->1} H_0\nprint \\expval{Z_0 Z_1} as zz\n";
    const std::string unicode = format(ascii);

    writeText(path, ascii);
    CliResult r = cli({"run", path.string(), "--no-format", "--json", "--no-timing"});
    EXPECT_EQ(r.code, 0) << r.err;
    EXPECT_EQ(readText(path), ascii);
    const auto j1 = Json::parse(r.out);
    ASSERT_TRUE(j1);
    EXPECT_FALSE(j1->find("formatted")->asBool());

    r = cli({"run", path.string(), "--json", "--no-timing"});
    EXPECT_EQ(r.code, 0) << r.err;
    EXPECT_EQ(readText(path), unicode);
    const auto j2 = Json::parse(r.out);
    ASSERT_TRUE(j2);
    EXPECT_TRUE(j2->find("formatted")->asBool());
    // Results are independent of whether the file was rewritten.
    ASSERT_TRUE(j1->find("prints") && j2->find("prints"));
    ASSERT_EQ(j1->find("prints")->size(), 1U);
    EXPECT_EQ(*j1->find("prints")->asArray()[0].find("value"), *j2->find("prints")->asArray()[0].find("value"));

    // Canonical files are left alone.
    r = cli({"check", path.string(), "--json"});
    EXPECT_EQ(r.code, 0);
    EXPECT_FALSE(Json::parse(r.out)->find("formatted")->asBool());
}

TEST(Formatter, CanonicalizeKeepsBrokenFilesAndPermissions)
{
    TempDir dir;
    const auto broken = dir.path / "broken.ntr";
    const std::string bad = "noether 0.1\nqubits q[2]\nCX_{0->} H_0\n";
    writeText(broken, bad);
    std::string error;
    EXPECT_FALSE(canonicalizeFile(broken.string(), error));
    EXPECT_EQ(readText(broken), bad);

    const auto exe = dir.path / "exe.ntr";
    writeText(exe, "noether 0.1\nqubits q[1]\nSdg_0\n");
    std::filesystem::permissions(exe, std::filesystem::perms::owner_all);
    EXPECT_TRUE(canonicalizeFile(exe.string(), error)) << error;
    EXPECT_EQ(readText(exe), "noether 0.1\nqubits q[1]\nS†_0\n");
    EXPECT_EQ(std::filesystem::status(exe).permissions() & std::filesystem::perms::all, std::filesystem::perms::owner_all);
}

TEST(Formatter, FmtCheckAndWrite)
{
    TempDir dir;
    const auto path = dir.path / "p.ntr";
    writeText(path, "noether 0.1\nqubits q[1]\nSdg_0\n");
    EXPECT_NE(cli({"fmt", "--check", path.string()}).code, 0);
    EXPECT_EQ(cli({"fmt", "-w", path.string()}).code, 0);
    EXPECT_EQ(cli({"fmt", "--check", path.string()}).code, 0);
    const CliResult ascii = cli({"fmt", "--ascii", path.string()});
    EXPECT_EQ(ascii.out, "noether 0.1\nqubits q[1]\nS\\dagger_0\n");
}
