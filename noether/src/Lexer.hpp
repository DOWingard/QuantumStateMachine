#pragma once

#include "Diagnostics.hpp"
#include "Source.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>



namespace Noether
{

enum class Tok : std::uint8_t
{
    End, Newline, Indent, Dedent,
    Ident, Int, Real, String, Ket, Bra, Braket,
#define NTR_PUNCT(Name, canonical, ascii) Name,
#define NTR_KEYWORD(Name, text) Name,
#include "tokens.def"
};

std::string_view tokName(Tok t);       // for messages: "identifier", "'→'", "'let'"
std::string_view canonicalText(Tok t); // punctuation and keywords
std::string_view asciiText(Tok t);

struct Token
{
    Tok kind = Tok::End;
    Span span;
    bool spaceBefore = true; // whitespace, a comment or a line break precedes it
    std::string text;        // Ident: canonical name; Int/Real: digits; String: contents; Ket/Bra/Braket: label
    std::string text2;       // Braket: the ket label
};

struct Comment
{
    Span span;
    std::string text; // without the leading '#' or '##'
    bool doc = false;
    bool ownLine = false; // nothing but whitespace precedes it on its line
    bool blankBefore = false; // a blank line directly precedes it (set by the parser)
};

struct LexResult
{
    std::vector<Token> tokens;
    std::vector<Comment> comments;
};

LexResult lex(const SourceFile& file, std::uint32_t fileId, Diagnostics& diags);

// Exposed for `noether tokens` and the skill cheat sheet.
struct AliasRow
{
    std::string canonical;
    std::string latex;
    std::string ascii;
    std::string asciiCanonical;
    std::string meaning;
};
std::vector<AliasRow> aliasTable();

bool isGreekLetter(std::string_view name);
std::string asciiSpelling(std::string_view ident); // θ -> theta, 𝓗 -> \mathcal{H}, others unchanged

// Whether `left` and `right` written without a space lex to the tokens of each written apart,
// with no diagnostics (the formatter's test for gluing factors such as 2π).
bool joinsCleanly(std::string_view left, std::string_view right);

// Damerau-free Levenshtein distance, for did-you-mean suggestions.
std::size_t editDistance(std::string_view a, std::string_view b);

} // namespace Noether
