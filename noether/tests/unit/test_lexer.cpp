// Lexer tests generated from tokens.def: every spelling the table lists (canonical, ASCII, Unicode
// alternative, LaTeX command) must lex to its token, so a table edit cannot leave a spelling
// untested.

#include "TestUtil.hpp"

#include "Lexer.hpp"

using namespace Noether;
using namespace NoetherTest;

namespace
{

struct Lexed
{
    std::vector<Token> tokens; // without Newline/Indent/Dedent/End
    std::vector<std::string> codes;
};

Lexed lexText(const std::string& text)
{
    SourceManager sm;
    Diagnostics d(sm);
    const auto id = sm.add("lex.ntr", text);
    LexResult r = lex(sm.file(id), id, d);
    Lexed out;
    for (Token& t : r.tokens)
        if (t.kind != Tok::Newline && t.kind != Tok::Indent && t.kind != Tok::Dedent && t.kind != Tok::End) out.tokens.push_back(std::move(t));
    for (const auto& x : d.sorted()) out.codes.push_back(x.code);
    return out;
}

std::string kinds(const Lexed& l)
{
    std::string s;
    for (const Token& t : l.tokens) s += std::format("{}[{}] ", tokName(t.kind), t.text);
    for (const auto& c : l.codes) s += c + " ";
    return s;
}

// The token sequence of `a <spelling> b` must be exactly Ident, expected, Ident.
void expectInfix(const std::string& spelling, Tok expected)
{
    const Lexed l = lexText("a " + spelling + " b");
    ASSERT_EQ(l.tokens.size(), 3U) << "`" << spelling << "`: " << kinds(l);
    EXPECT_EQ(l.tokens[1].kind, expected) << "`" << spelling << "` lexed as " << kinds(l);
    EXPECT_TRUE(l.codes.empty()) << "`" << spelling << "`: " << kinds(l);
}

// Spellings that only exist inside a construct, checked separately below.
bool contextual(Tok t) { return t == Tok::TensorPow || t == Tok::LAngle || t == Tok::RAngle; }

} // namespace

TEST(Lexer, PunctuationCanonicalAndAsciiSpellings)
{
    std::size_t checked = 0;
#define NTR_PUNCT(Name, canonical, ascii)                                                                                  \
    if (!contextual(Tok::Name))                                                                                            \
    {                                                                                                                      \
        expectInfix(canonical, Tok::Name);                                                                                 \
        expectInfix(ascii, Tok::Name);                                                                                     \
        checked += 2;                                                                                                      \
    }
#include "tokens.def"
    EXPECT_GT(checked, 60U);
}

TEST(Lexer, ContextualPunctuation)
{
    // The braces of ^{\otimes 3} stay as tokens; the parser reads them as a braced exponent.
    for (const std::string src : {"|0⟩^⊗3", "|0>^{\\otimes 3}", "|0>^\\otimes 3"})
    {
        const Lexed l = lexText(src);
        std::vector<Tok> ks;
        for (const Token& t : l.tokens)
            if (t.kind != Tok::LBrace && t.kind != Tok::RBrace) ks.push_back(t.kind);
        EXPECT_EQ(ks, (std::vector<Tok>{Tok::Ket, Tok::TensorPow, Tok::Int})) << src << ": " << kinds(l);
        EXPECT_TRUE(l.codes.empty()) << src;
    }
    for (const std::string src : {"⟨X_0⟩", "\\expval{X_0}", "\\langle X_0 \\rangle"})
    {
        const Lexed l = lexText(src);
        ASSERT_EQ(l.tokens.size(), 5U) << src << ": " << kinds(l);
        EXPECT_EQ(l.tokens.front().kind, Tok::LAngle) << src;
        EXPECT_EQ(l.tokens.back().kind, Tok::RAngle) << src;
        EXPECT_TRUE(l.codes.empty()) << src << ": " << kinds(l);
    }
}

TEST(Lexer, AsciiOperatorSpellings)
{
#define NTR_ASCII_OP(text, Name) expectInfix(text, Tok::Name);
#include "tokens.def"
}

TEST(Lexer, UnicodeAlternatives)
{
#define NTR_UNICODE_ALT(text, Name) expectInfix(text, Tok::Name);
#include "tokens.def"
}

TEST(Lexer, LatexTokenCommandsTakeOneBackslash)
{
#define NTR_LATEX_TOK(command, Name)                                                                                       \
    if (Tok::Name != Tok::LAngle && Tok::Name != Tok::RAngle) expectInfix("\\" command, Tok::Name);
#include "tokens.def"
}

TEST(Lexer, DoubledBackslashIsOneDiagnosticAndRecovers)
{
#define NTR_LATEX_TOK(command, Name)                                                                                       \
    if (Tok::Name != Tok::LAngle && Tok::Name != Tok::RAngle)                                                              \
    {                                                                                                                      \
        const Lexed l = lexText("a \\\\" command " b");                                                                    \
        ASSERT_EQ(l.codes, std::vector<std::string>{"E1004"}) << command << ": " << kinds(l);                              \
        ASSERT_EQ(l.tokens.size(), 3U) << command << ": " << kinds(l);                                                     \
        EXPECT_EQ(l.tokens[1].kind, Tok::Name) << command;                                                                 \
    }
#include "tokens.def"
}

TEST(Lexer, LatexIdentifiersAndGreekWords)
{
#define NTR_LATEX_IDENT(command, ident)                                                                                    \
    {                                                                                                                      \
        const Lexed l = lexText("\\" command);                                                                             \
        ASSERT_EQ(l.tokens.size(), 1U) << command << ": " << kinds(l);                                                     \
        EXPECT_EQ(l.tokens[0].kind, Tok::Ident);                                                                           \
        EXPECT_EQ(l.tokens[0].text, ident) << command;                                                                     \
    }
#include "tokens.def"
#define NTR_GREEK(word, letter)                                                                                            \
    for (const std::string src : {std::string(word), "\\" + std::string(word), std::string(letter)})                      \
    {                                                                                                                      \
        const Lexed l = lexText(src);                                                                                      \
        ASSERT_EQ(l.tokens.size(), 1U) << src << ": " << kinds(l);                                                         \
        EXPECT_EQ(l.tokens[0].kind, Tok::Ident) << src;                                                                    \
        EXPECT_EQ(l.tokens[0].text, letter) << src;                                                                        \
        EXPECT_TRUE(l.codes.empty()) << src << ": " << kinds(l);                                                           \
    }
#include "tokens.def"
}

TEST(Lexer, LatexLayoutCommandsAreDropped)
{
#define NTR_LATEX_LAYOUT(command)                                                                                          \
    {                                                                                                                      \
        const Lexed l = lexText("a \\" command " b");                                                                      \
        EXPECT_EQ(l.tokens.size(), 2U) << command << ": " << kinds(l);                                                     \
        EXPECT_TRUE(l.codes.empty()) << command << ": " << kinds(l);                                                       \
    }
#include "tokens.def"
}

TEST(Lexer, Keywords)
{
#define NTR_KEYWORD(Name, text)                                                                                            \
    {                                                                                                                      \
        const Lexed l = lexText(text);                                                                                     \
        ASSERT_EQ(l.tokens.size(), 1U) << text;                                                                            \
        EXPECT_EQ(l.tokens[0].kind, Tok::Name) << text;                                                                    \
        EXPECT_EQ(canonicalText(Tok::Name), text);                                                                         \
    }
#include "tokens.def"
}

TEST(Lexer, DiracNotationInEveryForm)
{
    struct Case
    {
        std::string src;
        Tok kind;
        std::string label, label2;
    };
    for (const Case& c : std::vector<Case>{
             {"|01⟩", Tok::Ket, "01", ""},
             {"|01>", Tok::Ket, "01", ""},
             {"\\ket{01}", Tok::Ket, "01", ""},
             {"|ψ⟩", Tok::Ket, "ψ", ""},
             {"|psi>", Tok::Ket, "ψ", ""},
             {"⟨01|", Tok::Bra, "01", ""},
             {"<01|", Tok::Bra, "01", ""},
             {"\\bra{01}", Tok::Bra, "01", ""},
             {"⟨01|ψ⟩", Tok::Braket, "01", "ψ"},
             {"<01|psi>", Tok::Braket, "01", "ψ"},
             {"\\braket{01|\\psi}", Tok::Braket, "01", "ψ"},
             {"|+⟩", Tok::Ket, "+", ""},
             {"|->", Tok::Ket, "-", ""},
         })
    {
        const Lexed l = lexText(c.src);
        ASSERT_EQ(l.tokens.size(), 1U) << c.src << ": " << kinds(l);
        EXPECT_EQ(l.tokens[0].kind, c.kind) << c.src;
        EXPECT_EQ(l.tokens[0].text, c.label) << c.src;
        if (c.kind == Tok::Braket)
        {
            EXPECT_EQ(l.tokens[0].text2, c.label2) << c.src;
        }
        EXPECT_TRUE(l.codes.empty()) << c.src << ": " << kinds(l);
    }
}

TEST(Lexer, LatexStructuralCommands)
{
    EXPECT_EQ(kinds(lexText("\\frac{a}{b}")), kinds(lexText("(a)/(b)")));
    EXPECT_EQ(kinds(lexText("\\sqrt{x}")), kinds(lexText("√(x)")));
    EXPECT_EQ(kinds(lexText("\\mathcal{H}")), kinds(lexText("𝓗")));
    EXPECT_EQ(kinds(lexText("\\mathbb{R}")), kinds(lexText("ℝ")));
    EXPECT_EQ(kinds(lexText("sqrt(x)")), kinds(lexText("√(x)")));
}

TEST(Lexer, UnknownLatexCommandSuggestsTheNearestOne)
{
    const Lexed l = lexText("S\\dager_0");
    ASSERT_FALSE(l.codes.empty());
    EXPECT_EQ(l.codes[0], "E1004");
}

TEST(Lexer, TabIndentationAndInvalidUtf8AreErrors)
{
    EXPECT_EQ(lexText("for k in 0..1:\n\tH_k\n").codes, std::vector<std::string>{"E1002"});
    EXPECT_EQ(lexText(std::string("a \xff b")).codes, std::vector<std::string>{"E1001"});
}

TEST(Lexer, AliasTableCoversEveryPunctuationToken)
{
    std::set<std::string> canonical;
    for (const AliasRow& r : aliasTable()) canonical.insert(r.canonical);
#define NTR_PUNCT(Name, canon, ascii)                                                                                      \
    if (std::string(canon) != std::string(ascii))                                                                          \
    {                                                                                                                      \
        EXPECT_TRUE(canonical.contains(canon)) << canon;                                                                   \
    }
#include "tokens.def"
}
