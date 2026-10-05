#include "Lexer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>



namespace Noether
{

namespace
{

    struct TokInfo
    {
        std::string_view name;
        std::string_view canonical;
        std::string_view ascii;
    };

    constexpr std::array kTokInfo{
        TokInfo{"end of file", "", ""},
        TokInfo{"newline", "", ""},
        TokInfo{"indent", "", ""},
        TokInfo{"dedent", "", ""},
        TokInfo{"identifier", "", ""},
        TokInfo{"integer", "", ""},
        TokInfo{"real", "", ""},
        TokInfo{"string", "", ""},
        TokInfo{"ket", "", ""},
        TokInfo{"bra", "", ""},
        TokInfo{"braket", "", ""},
#define NTR_PUNCT(Name, canonical, ascii) TokInfo{"'" canonical "'", canonical, ascii},
#define NTR_KEYWORD(Name, text) TokInfo{"'" text "'", text, text},
#include "tokens.def"
    };

    struct Alias
    {
        std::string canonical;
        bool dagger;
    };

    struct Tables
    {
        std::unordered_map<std::string, Tok> keywords;
        std::unordered_map<std::string, Tok> asciiWordOps;
        std::unordered_map<std::string, std::string> greekWords;  // theta -> θ
        std::unordered_map<std::string, std::string> greekToWord; // θ -> theta
        std::unordered_map<std::string, Tok> latexTok;
        std::unordered_map<std::string, std::string> latexIdent;
        std::unordered_set<std::string> latexLayout;
        std::unordered_map<std::string, Alias> gateAliases;
        std::vector<std::pair<std::string, Tok>> unicodeOps; // longest first
        std::vector<std::string> latexCommands;              // every known command, for suggestions

        Tables()
        {
#define NTR_KEYWORD(Name, text) keywords.emplace(text, Tok::Name);
#define NTR_GREEK(word, letter)            \
    greekWords.emplace(word, letter);      \
    greekToWord.emplace(letter, word);     \
    latexIdent.emplace(word, letter);
#define NTR_LATEX_TOK(command, Name) latexTok.emplace(command, Tok::Name);
#define NTR_LATEX_IDENT(command, ident) latexIdent.emplace(command, ident);
#define NTR_LATEX_LAYOUT(command) latexLayout.emplace(command);
#define NTR_GATE_ALIAS(alias, canonical, dagger) gateAliases.emplace(alias, Alias{canonical, dagger});
#define NTR_UNICODE_ALT(text, Name) unicodeOps.emplace_back(text, Tok::Name);
#include "tokens.def"

            for (const auto& [w, t] : std::initializer_list<std::pair<std::string, Tok>>{
                     {"not", Tok::Not}, {"in", Tok::In}, {"sqrt", Tok::Sqrt}})
                asciiWordOps.emplace(w, t);

#define NTR_PUNCT(Name, canonical, ascii)                                                        \
    if (static_cast<unsigned char>(std::string_view(canonical)[0]) >= 0x80 && Tok::Name != Tok::LAngle) \
        unicodeOps.emplace_back(canonical, Tok::Name);
#include "tokens.def"
            std::ranges::sort(unicodeOps, [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

            for (const auto& [k, v] : latexTok) latexCommands.push_back(k);
            for (const auto& [k, v] : latexIdent) latexCommands.push_back(k);
            for (const auto& k : latexLayout) latexCommands.push_back(k);
            for (const char* k : {"frac", "sqrt", "expval", "ket", "bra", "braket", "mel", "mathcal", "mathbb"})
                latexCommands.emplace_back(k);
            std::ranges::sort(latexCommands);
        }
    };

    const Tables& tables()
    {
        static const Tables t;
        return t;
    }

    bool isAsciiLetter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
    bool isDigit(char c) { return c >= '0' && c <= '9'; }
    bool isWordChar(char c) { return isAsciiLetter(c) || isDigit(c) || c == '\''; }

    // Σ (U+03A3) is the summation operator, so it is not a letter here.
    bool isGreekCp(char32_t cp)
    {
        return (cp >= 0x0391 && cp <= 0x03A9 && cp != 0x03A3 && cp != 0x03A2) || (cp >= 0x03B1 && cp <= 0x03C9) ||
               cp == 0x03D5 || cp == 0x03D1 || cp == 0x03F5;
    }

    bool isMathLetterCp(char32_t cp)
    {
        if (cp >= 0x1D400 && cp <= 0x1D7CB) return true;
        switch (cp)
        {
            case 0x2102: case 0x210B: case 0x210C: case 0x210D: case 0x210F: case 0x2110: case 0x2112:
            case 0x2115: case 0x2119: case 0x211A: case 0x211B: case 0x211D: case 0x2124: case 0x212C:
            case 0x2130: case 0x2131: case 0x2133:
                return true;
            default:
                return false;
        }
    }

    // Script and variant spellings collapse to one canonical letter per identifier.
    char32_t canonicalLetter(char32_t cp)
    {
        switch (cp)
        {
            case 0x03D5: return 0x03C6; // ϕ -> φ
            case 0x03D1: return 0x03B8; // ϑ -> θ
            case 0x03F5: return 0x03B5; // ϵ -> ε
            case 0x212C: return 0x1D4D1; // ℬ -> 𝓑
            case 0x2130: return 0x1D4D4;
            case 0x2131: return 0x1D4D5;
            case 0x210B: return 0x1D4D7; // ℋ -> 𝓗
            case 0x2110: return 0x1D4D8;
            case 0x2112: return 0x1D4DB;
            case 0x2133: return 0x1D4DC;
            case 0x211B: return 0x1D4E1;
            default: break;
        }
        if (cp >= 0x1D49C && cp <= 0x1D4B5) return cp - 0x1D49C + 0x1D4D0; // script -> bold script
        return cp;
    }

    bool isConfusableCp(char32_t cp)
    {
        switch (cp)
        {
            case 0x03BF: case 0x039F: case 0x03BD: case 0x0391: case 0x0392: case 0x0395: case 0x0396:
            case 0x0397: case 0x0399: case 0x039A: case 0x039C: case 0x039D: case 0x03A1: case 0x03A4:
            case 0x03A5: case 0x03A7: case 0x03C1: case 0x03B9: case 0x03C5:
                return true;
            default:
                return false;
        }
    }

    bool isCyrillicCp(char32_t cp) { return cp >= 0x0400 && cp <= 0x052F; }

    int subscriptDigit(char32_t cp) { return cp >= 0x2080 && cp <= 0x2089 ? static_cast<int>(cp - 0x2080) : -1; }

    int superscriptDigit(char32_t cp)
    {
        switch (cp)
        {
            case 0x2070: return 0;
            case 0x00B9: return 1;
            case 0x00B2: return 2;
            case 0x00B3: return 3;
            default: return cp >= 0x2074 && cp <= 0x2079 ? static_cast<int>(cp - 0x2070) : -1;
        }
    }


    class Lexer
    {
        public:
        Lexer(const SourceFile& file, std::uint32_t fileId, Diagnostics& diagnostics)
            : fid(fileId), d(diagnostics), s(file.text())
        {
        }

        LexResult run()
        {
            bool atLineStart = true;
            while (pos < s.size())
            {
                if (atLineStart)
                {
                    atLineStart = false;
                    lineIndent();
                    continue;
                }
                const char c = s[pos];
                if (c == '\n')
                {
                    if (nesting == 0 && lineHasTokens)
                    {
                        emit(Tok::Newline, pos, pos + 1);
                        lineHasTokens = false;
                    }
                    physLineHasTokens = false;
                    ++pos;
                    space = true;
                    atLineStart = true;
                    continue;
                }
                if (c == ' ' || c == '\r' || c == '\t')
                {
                    ++pos;
                    space = true;
                    continue;
                }
                if (c == '#')
                {
                    comment();
                    continue;
                }
                token();
            }
            if (lineHasTokens) emit(Tok::Newline, pos, pos);
            while (indents.size() > 1)
            {
                indents.pop_back();
                emit(Tok::Dedent, pos, pos);
            }
            emit(Tok::End, pos, pos);
            return std::move(out);
        }

        private:
        enum class Brace : std::uint8_t { Normal, FracNum, FracDen, Paren, Angle, MelMid };

        Span span(std::size_t b, std::size_t e) const
        {
            return {fid, static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(e)};
        }

        void emit(Tok k, std::size_t b, std::size_t e, std::string text = {}, std::string text2 = {})
        {
            Token t;
            t.kind = k;
            t.span = span(b, e);
            t.spaceBefore = space;
            t.text = std::move(text);
            t.text2 = std::move(text2);
            out.tokens.push_back(std::move(t));
            if (k != Tok::Newline && k != Tok::Indent && k != Tok::Dedent && k != Tok::End)
            {
                space = false;
                lineHasTokens = true;
                physLineHasTokens = true;
            }
            switch (k)
            {
                case Tok::LParen: case Tok::LBracket: case Tok::LBrace: case Tok::LAngle: ++nesting; break;
                case Tok::RParen: case Tok::RBracket: case Tok::RBrace: case Tok::RAngle:
                    if (nesting > 0) --nesting;
                    break;
                default: break;
            }
        }

        // Leading whitespace of a physical line: INDENT / DEDENT unless the line is blank, a comment,
        // or a continuation inside open brackets.
        void lineIndent()
        {
            std::size_t p = pos;
            std::uint32_t col = 0;
            std::optional<std::size_t> tab;
            while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\r'))
            {
                if (s[p] == '\t')
                {
                    if (!tab) tab = p;
                    col += 4;
                }
                else if (s[p] == ' ') ++col;
                ++p;
            }
            pos = p;
            space = true;
            if (nesting > 0 || p >= s.size() || s[p] == '\n' || s[p] == '#') return;
            if (tab) d.error("E1002", span(*tab, *tab + 1), "tab in indentation; indent with spaces");

            // A line at the column of an earlier bad dedent continues that block silently.
            if (badDedent && badDedent->first == col && badDedent->second == indents.size()) return;
            badDedent.reset();
            if (col > indents.back())
            {
                indents.push_back(col);
                emit(Tok::Indent, p, p);
            }
            else
            {
                while (col < indents.back())
                {
                    indents.pop_back();
                    emit(Tok::Dedent, p, p);
                }
                if (col != indents.back())
                {
                    // Recover by treating the line as part of the block it dedented to.
                    d.error("E2002", span(p, p + 1),
                            std::format("dedent to column {} matches no enclosing block (expected column {})", col + 1,
                                        indents.back() + 1));
                    badDedent = {col, indents.size()};
                }
            }
            space = true;
        }

        // Comments and strings may hold any text, but it must still be UTF-8.
        void checkUtf8(std::size_t from, std::size_t to)
        {
            for (std::size_t p = from; p < to;)
            {
                const CodePoint cp = decodeUtf8(s, p);
                if (cp.length == 0)
                {
                    d.error("E1001", span(p, p + 1), "invalid UTF-8 byte sequence");
                    return;
                }
                p += cp.length;
            }
        }

        void comment()
        {
            const std::size_t b = pos;
            std::size_t e = s.find('\n', pos);
            if (e == std::string_view::npos) e = s.size();
            checkUtf8(b, e);
            Comment c;
            c.doc = s.substr(b, 2) == "##";
            c.text = std::string(s.substr(b + (c.doc ? 2 : 1), e - b - (c.doc ? 2 : 1)));
            while (!c.text.empty() && (c.text.back() == '\r' || c.text.back() == ' ')) c.text.pop_back();
            c.ownLine = !physLineHasTokens;
            c.span = span(b, e);
            pragma(c);
            out.comments.push_back(std::move(c));
            pos = e;
            space = true;
        }

        void pragma(const Comment& c)
        {
            std::string_view t = c.text;
            while (!t.empty() && t.front() == ' ') t.remove_prefix(1);
            constexpr std::string_view kPrefix = "noether: allow";
            if (!t.starts_with(kPrefix)) return;
            t.remove_prefix(kPrefix.size());
            std::size_t i = 0;
            while (i < t.size())
            {
                while (i < t.size() && (t[i] == ' ' || t[i] == ',')) ++i;
                std::size_t j = i;
                while (j < t.size() && t[j] != ' ' && t[j] != ',') ++j;
                if (j > i) d.allow(fid, std::string(t.substr(i, j - i)));
                i = j;
            }
        }

        // ---- Dirac labels: a binary string, + - +i -i, or one identifier ----
        std::optional<std::pair<std::string, std::size_t>> label(std::size_t p) const
        {
            if (p >= s.size()) return std::nullopt;
            const char c = s[p];
            if (c == '0' || c == '1')
            {
                std::size_t e = p;
                while (e < s.size() && (s[e] == '0' || s[e] == '1')) ++e;
                return std::pair{std::string(s.substr(p, e - p)), e};
            }
            if (c == '+' || c == '-')
            {
                if (p + 1 < s.size() && s[p + 1] == 'i' && (p + 2 >= s.size() || !isWordChar(s[p + 2])))
                    return std::pair{std::string(s.substr(p, 2)), p + 2};
                return std::pair{std::string(1, c), p + 1};
            }
            if (isAsciiLetter(c))
            {
                std::size_t e = p;
                while (e < s.size() && isWordChar(s[e])) ++e;
                std::string w(s.substr(p, e - p));
                if (const auto g = tables().greekWords.find(w); g != tables().greekWords.end()) w = g->second;
                return std::pair{w, e};
            }
            if (c == '\\')
            {
                std::size_t e = p + 1;
                while (e < s.size() && isAsciiLetter(s[e])) ++e;
                const std::string cmd(s.substr(p + 1, e - p - 1));
                if (const auto g = tables().latexIdent.find(cmd); g != tables().latexIdent.end() && isGreekLetter(g->second))
                    return std::pair{g->second, e};
                return std::nullopt;
            }
            const CodePoint cp = decodeUtf8(s, p);
            if (cp.length && (isGreekCp(cp.value) || isMathLetterCp(cp.value)))
            {
                std::size_t e = p + cp.length;
                std::string name = encodeUtf8(canonicalLetter(cp.value));
                while (e < s.size() && (isDigit(s[e]) || s[e] == '\'')) name += s[e++];
                return std::pair{name, e};
            }
            return std::nullopt;
        }

        // Length of a ket closer at p: ⟩, > or \rangle.
        std::size_t ketCloser(std::size_t p) const
        {
            if (s.substr(p, 3) == "⟩") return 3;
            if (p < s.size() && s[p] == '>') return 1;
            if (s.substr(p, 7) == "\\rangle") return 7;
            return 0;
        }

        // At `⟨` or `<` (opener of length `len`): a bra or braket, if one starts here.
        bool tryBra(std::size_t len)
        {
            const std::size_t b = pos;
            const auto l1 = label(b + len);
            if (!l1 || l1->second >= s.size() || s[l1->second] != '|') return false;
            const std::size_t afterBar = l1->second + 1;
            if (const auto l2 = label(afterBar))
                if (const std::size_t close = ketCloser(l2->second))
                {
                    pos = l2->second + close;
                    emit(Tok::Braket, b, pos, l1->first, l2->first);
                    return true;
                }
            pos = afterBar;
            emit(Tok::Bra, b, pos, l1->first);
            return true;
        }

        void token()
        {
            const char c = s[pos];
            if (isDigit(c)) return number();
            if (isAsciiLetter(c)) return word();
            if (static_cast<unsigned char>(c) >= 0x80) return unicode();

            const std::size_t b = pos;
            auto one = [&](Tok k, std::size_t len = 1)
            {
                pos += len;
                emit(k, b, pos);
            };
            auto next = [&](std::size_t k) { return pos + k < s.size() ? s[pos + k] : '\0'; };

            switch (c)
            {
                case '|':
                    if (const auto l = label(pos + 1))
                        if (const std::size_t close = ketCloser(l->second))
                        {
                            pos = l->second + close;
                            emit(Tok::Ket, b, pos, l->first);
                            return;
                        }
                    return one(Tok::Abs);
                case '<':
                    if (tryBra(1)) return;
                    if (next(1) == '-') return one(Tok::LeftArrow, 2);
                    if (next(1) == '=') return one(Tok::LessEq, 2);
                    return one(Tok::Less);
                case '>': return next(1) == '=' ? one(Tok::GreaterEq, 2) : one(Tok::Greater);
                case '-': return next(1) == '>' ? one(Tok::Arrow, 2) : one(Tok::Minus);
                case '+': return next(1) == '-' ? one(Tok::PlusMinus, 2) : one(Tok::Plus);
                case '*': return one(Tok::Cdot);
                case '~':
                    if (next(1) == '=') return one(Tok::Approx, 2);
                    break;
                case '!':
                    if (next(1) == '=') return one(Tok::NotEq, 2);
                    break;
                case '=': return next(1) == '=' ? one(Tok::EqEq, 2) : one(Tok::Assign);
                case '.':
                    if (next(1) == '.') return one(Tok::DotDot, 2);
                    if (isDigit(next(1)))
                    {
                        d.error("E2001", span(b, b + 1), "a real literal needs a leading digit")
                            .fix("add the leading zero", span(b, b), "0");
                    }
                    return one(Tok::Dot);
                case '^': return caret();
                case '_': return one(Tok::Underscore);
                case '(': return one(Tok::LParen);
                case ')': return one(Tok::RParen);
                case '[': return one(Tok::LBracket);
                case ']': return one(Tok::RBracket);
                case '{':
                    braces.push_back(Brace::Normal);
                    return one(Tok::LBrace);
                case '}': return closeBrace();
                case ',': return one(Tok::Comma);
                case ';': return one(Tok::Semicolon);
                case ':': return one(Tok::Colon);
                case '/': return one(Tok::Slash);
                case '"': return string();
                case '\\': return latex();
                default: break;
            }
            d.error("E2001", span(b, b + 1), std::format("unexpected character '{}'", c));
            ++pos;
            space = true;
        }

        void number()
        {
            const std::size_t b = pos;
            while (pos < s.size() && isDigit(s[pos])) ++pos;
            bool real = false;
            if (pos + 1 < s.size() && s[pos] == '.' && isDigit(s[pos + 1]))
            {
                real = true;
                ++pos;
                while (pos < s.size() && isDigit(s[pos])) ++pos;
            }
            if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E'))
            {
                std::size_t p = pos + 1;
                if (p < s.size() && (s[p] == '+' || s[p] == '-')) ++p;
                if (p < s.size() && isDigit(s[p]))
                {
                    real = true;
                    pos = p;
                    while (pos < s.size() && isDigit(s[pos])) ++pos;
                }
            }
            emit(real ? Tok::Real : Tok::Int, b, pos, std::string(s.substr(b, pos - b)));
        }

        void word()
        {
            const std::size_t b = pos;
            while (pos < s.size() && isWordChar(s[pos])) ++pos;
            const std::string w(s.substr(b, pos - b));
            const Tables& t = tables();
            if (const auto k = t.keywords.find(w); k != t.keywords.end()) return emit(k->second, b, pos);
            if (const auto k = t.asciiWordOps.find(w); k != t.asciiWordOps.end()) return emit(k->second, b, pos);
            if (const auto g = t.greekWords.find(w); g != t.greekWords.end()) return emit(Tok::Ident, b, pos, g->second);
            if (const auto a = t.gateAliases.find(w); a != t.gateAliases.end())
            {
                if (a->second.canonical == "√X")
                {
                    emit(Tok::Sqrt, b, pos);
                    return emit(Tok::Ident, b, pos, "X");
                }
                emit(Tok::Ident, b, pos, a->second.canonical);
                if (a->second.dagger) emit(Tok::Dagger, b, pos);
                return;
            }
            emit(Tok::Ident, b, pos, w);
        }

        void unicode()
        {
            const std::size_t b = pos;
            const CodePoint cp = decodeUtf8(s, pos);
            if (cp.length == 0)
            {
                d.error("E1001", span(b, b + 1), "invalid UTF-8 byte sequence");
                ++pos;
                space = true;
                return;
            }

            if (isGreekCp(cp.value) || isMathLetterCp(cp.value))
            {
                pos += cp.length;
                std::string name = encodeUtf8(canonicalLetter(cp.value));
                while (pos < s.size() && (isDigit(s[pos]) || s[pos] == '\''))
                    name += s[pos++];
                while (s.substr(pos, 3) == "′")
                {
                    name += '\'';
                    pos += 3;
                }
                if (isConfusableCp(cp.value) &&
                    ((b > 0 && isAsciiLetter(s[b - 1])) || (pos < s.size() && isAsciiLetter(s[pos]))))
                {
                    d.error("E1003", span(b, b + cp.length),
                            std::format("'{}' is a Greek letter that looks like a Latin one and touches Latin letters",
                                        encodeUtf8(cp.value)));
                    // Recover with the whole letter run as one identifier, so that one look-alike
                    // letter gives one diagnostic.
                    pos = b + cp.length;
                    while (pos < s.size() && isWordChar(s[pos])) ++pos;
                    std::string run(s.substr(b, pos - b));
                    if (!out.tokens.empty() && out.tokens.back().kind == Tok::Ident && out.tokens.back().span.end == b)
                    {
                        out.tokens.back().text += run;
                        out.tokens.back().span.end = static_cast<std::uint32_t>(pos);
                        return;
                    }
                    return emit(Tok::Ident, b, pos, run);
                }
                return emit(Tok::Ident, b, pos, name);
            }
            if (isCyrillicCp(cp.value))
            {
                d.error("E1003", span(b, b + cp.length),
                        std::format("Cyrillic letter '{}' is not allowed in identifiers (it looks Latin)",
                                    encodeUtf8(cp.value)));
                pos += cp.length;
                return;
            }
            if (subscriptDigit(cp.value) >= 0)
            {
                emit(Tok::Underscore, b, b + cp.length);
                std::string digits;
                while (pos < s.size())
                {
                    const CodePoint q = decodeUtf8(s, pos);
                    if (q.length == 0 || subscriptDigit(q.value) < 0) break;
                    digits += static_cast<char>('0' + subscriptDigit(q.value));
                    pos += q.length;
                }
                return emit(Tok::Int, b, pos, digits);
            }
            if (superscriptDigit(cp.value) >= 0)
            {
                emit(Tok::Caret, b, b + cp.length);
                std::string digits;
                while (pos < s.size())
                {
                    const CodePoint q = decodeUtf8(s, pos);
                    if (q.length == 0 || superscriptDigit(q.value) < 0) break;
                    digits += static_cast<char>('0' + superscriptDigit(q.value));
                    pos += q.length;
                }
                return emit(Tok::Int, b, pos, digits);
            }
            if (cp.value == 0x27E8) // ⟨
            {
                if (tryBra(3)) return;
                pos += 3;
                return emit(Tok::LAngle, b, pos);
            }
            for (const auto& [text, tok] : tables().unicodeOps)
                if (s.substr(pos, text.size()) == text)
                {
                    pos += text.size();
                    return emit(tok, b, pos);
                }
            d.error("E2001", span(b, b + cp.length), std::format("unexpected character '{}'", encodeUtf8(cp.value)));
            pos += cp.length;
            space = true;
        }

        void caret()
        {
            const std::size_t b = pos;
            std::size_t p = pos + 1;
            if (s.substr(p, 7) == "\\dagger")
            {
                pos = p + 7;
                return emit(Tok::Dagger, b, pos);
            }
            if (s.substr(p, 3) == "†")
            {
                pos = p + 3;
                return emit(Tok::Dagger, b, pos);
            }
            if (s.substr(p, 3) == "⊗")
            {
                pos = p + 3;
                return emit(Tok::TensorPow, b, pos);
            }
            if (s.substr(p, 7) == "\\otimes")
            {
                pos = p + 7;
                return emit(Tok::TensorPow, b, pos);
            }
            if (p < s.size() && s[p] == '{')
            {
                std::size_t q = p + 1;
                while (q < s.size() && s[q] == ' ') ++q;
                if (s.substr(q, 7) == "\\dagger")
                {
                    std::size_t r = q + 7;
                    while (r < s.size() && s[r] == ' ') ++r;
                    if (r < s.size() && s[r] == '}')
                    {
                        pos = r + 1;
                        return emit(Tok::Dagger, b, pos);
                    }
                }
                std::size_t len = 0;
                if (s.substr(q, 7) == "\\otimes") len = 7;
                else if (s.substr(q, 3) == "⊗") len = 3;
                if (len)
                {
                    emit(Tok::TensorPow, b, p);
                    braces.push_back(Brace::Normal);
                    emit(Tok::LBrace, p, q + len);
                    pos = q + len;
                    space = true;
                    return;
                }
            }
            pos = p;
            emit(Tok::Caret, b, pos);
        }

        void string()
        {
            const std::size_t b = pos;
            std::string val;
            ++pos;
            while (pos < s.size() && s[pos] != '"' && s[pos] != '\n')
            {
                if (s[pos] == '\\' && pos + 1 < s.size() && s[pos + 1] == '"')
                {
                    val += '"';
                    pos += 2;
                }
                else val += s[pos++];
            }
            if (pos >= s.size() || s[pos] != '"')
            {
                d.error("E2001", span(b, pos), "unterminated string literal");
                return;
            }
            ++pos;
            checkUtf8(b, pos);
            emit(Tok::String, b, pos, val);
        }

        void skipSpaces()
        {
            while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) ++pos;
        }

        // Raw text of a {…} group starting at pos (which must be '{'), braces balanced.
        std::optional<std::string> group()
        {
            skipSpaces();
            if (pos >= s.size() || s[pos] != '{') return std::nullopt;
            int depth = 0;
            const std::size_t b = pos;
            for (; pos < s.size(); ++pos)
            {
                if (s[pos] == '{') ++depth;
                else if (s[pos] == '}' && --depth == 0)
                {
                    ++pos;
                    return std::string(s.substr(b + 1, pos - b - 2));
                }
                else if (s[pos] == '\n') break;
            }
            return std::nullopt;
        }

        static std::string normalizeLabel(std::string raw)
        {
            std::string t;
            for (const char c : raw)
                if (c != ' ') t += c;
            if (!t.empty() && t[0] == '\\')
            {
                if (const auto g = tables().latexIdent.find(t.substr(1)); g != tables().latexIdent.end()) return g->second;
            }
            if (const auto g = tables().greekWords.find(t); g != tables().greekWords.end()) return g->second;
            return t;
        }

        void latex()
        {
            const std::size_t b = pos;
            std::size_t e = pos + 1;
            while (e < s.size() && isAsciiLetter(s[e])) ++e;
            if (e == pos + 1 && e < s.size()) ++e; // one-character command such as \, or \;
            const std::string cmd(s.substr(pos + 1, e - pos - 1));
            pos = e;
            const Tables& t = tables();

            if (t.latexLayout.contains(cmd))
            {
                space = true;
                return;
            }
            if (const auto k = t.latexTok.find(cmd); k != t.latexTok.end()) return emit(k->second, b, pos);
            if (const auto k = t.latexIdent.find(cmd); k != t.latexIdent.end()) return emit(Tok::Ident, b, pos, k->second);

            auto expectBrace = [&]() -> bool
            {
                skipSpaces();
                if (pos < s.size() && s[pos] == '{')
                {
                    ++pos;
                    return true;
                }
                d.error("E2001", span(b, pos), std::format("\\{} needs a {{…}} argument", cmd));
                return false;
            };

            if (cmd == "frac")
            {
                if (!expectBrace()) return;
                braces.push_back(Brace::FracNum);
                return emit(Tok::LParen, b, pos);
            }
            if (cmd == "sqrt")
            {
                skipSpaces();
                if (pos < s.size() && s[pos] == '{')
                {
                    ++pos;
                    emit(Tok::Sqrt, b, pos);
                    braces.push_back(Brace::Paren);
                    return emit(Tok::LParen, b, pos);
                }
                return emit(Tok::Sqrt, b, pos);
            }
            if (cmd == "expval")
            {
                if (!expectBrace()) return;
                braces.push_back(Brace::Angle);
                return emit(Tok::LAngle, b, pos);
            }
            if (cmd == "ket" || cmd == "bra")
            {
                const auto g = group();
                if (!g) return void(d.error("E2001", span(b, pos), std::format("\\{} needs a {{label}}", cmd)));
                return emit(cmd == "ket" ? Tok::Ket : Tok::Bra, b, pos, normalizeLabel(*g));
            }
            if (cmd == "braket")
            {
                const auto g = group();
                const std::size_t bar = g ? g->find('|') : std::string::npos;
                if (bar == std::string::npos)
                    return void(d.error("E2001", span(b, pos), "\\braket needs {a|b}"));
                return emit(Tok::Braket, b, pos, normalizeLabel(g->substr(0, bar)), normalizeLabel(g->substr(bar + 1)));
            }
            if (cmd == "mel")
            {
                const auto g = group();
                if (!g) return void(d.error("E2001", span(b, pos), "\\mel needs {a}{O}{b}"));
                emit(Tok::Bra, b, pos, normalizeLabel(*g));
                if (!expectBrace()) return;
                braces.push_back(Brace::MelMid);
                return emit(Tok::LParen, b, pos);
            }
            if (cmd == "mathcal" || cmd == "mathbb")
            {
                const auto g = group();
                if (!g || g->size() != 1 || !isAsciiLetter((*g)[0]) || !std::isupper(static_cast<unsigned char>((*g)[0])))
                    return void(d.error("E2001", span(b, pos), std::format("\\{} needs one capital letter", cmd)));
                const char32_t base = cmd == "mathcal" ? 0x1D4D0 : 0x1D538;
                char32_t cp = base + static_cast<char32_t>((*g)[0] - 'A');
                if (cmd == "mathbb")
                {
                    static const std::map<char, char32_t> holes{{'C', 0x2102}, {'H', 0x210D}, {'N', 0x2115},
                                                                {'P', 0x2119}, {'Q', 0x211A}, {'R', 0x211D},
                                                                {'Z', 0x2124}};
                    if (const auto h = holes.find((*g)[0]); h != holes.end()) cp = h->second;
                }
                return emit(Tok::Ident, b, pos, encodeUtf8(cp));
            }

            // `\\cmd`: a doubled backslash, usually copied from a C or JSON string literal. Report it
            // and lex the rest as `\cmd` so one typo gives one diagnostic.
            if (cmd == "\\" && pos < s.size() && isAsciiLetter(s[pos]))
            {
                d.error("E1004", span(b, b + 2), "LaTeX commands take one backslash, not two")
                    .fix("remove one backslash", span(b, b + 2), "\\");
                pos = b + 1;
                return latex();
            }

            std::string best;
            std::size_t bestD = 3;
            for (const std::string& k : t.latexCommands)
                if (const std::size_t dist = editDistance(cmd, k); dist < bestD)
                {
                    bestD = dist;
                    best = k;
                }
            Diagnostic& diag = d.error("E1004", span(b, pos), std::format("unknown LaTeX command \\{}", cmd));
            if (!best.empty()) diag.fix(std::format("did you mean \\{}?", best), span(b, pos), "\\" + best);
            space = true;
        }

        void closeBrace()
        {
            const std::size_t b = pos;
            ++pos;
            if (braces.empty()) return emit(Tok::RBrace, b, pos);
            const Brace k = braces.back();
            braces.pop_back();
            switch (k)
            {
                case Brace::Normal: return emit(Tok::RBrace, b, pos);
                case Brace::FracDen:
                case Brace::Paren: return emit(Tok::RParen, b, pos);
                case Brace::Angle: return emit(Tok::RAngle, b, pos);
                case Brace::FracNum:
                {
                    emit(Tok::RParen, b, pos);
                    skipSpaces();
                    if (pos >= s.size() || s[pos] != '{')
                        return void(d.error("E2001", span(b, pos), "\\frac needs a second {…} argument"));
                    const std::size_t ob = pos++;
                    emit(Tok::Slash, ob, pos);
                    braces.push_back(Brace::FracDen);
                    return emit(Tok::LParen, ob, pos);
                }
                case Brace::MelMid:
                {
                    emit(Tok::RParen, b, pos);
                    const std::size_t kb = pos;
                    const auto g = group();
                    if (!g) return void(d.error("E2001", span(b, pos), "\\mel needs {a}{O}{b}"));
                    return emit(Tok::Ket, kb, pos, normalizeLabel(*g));
                }
            }
        }

        std::uint32_t fid;
        Diagnostics& d;
        std::string_view s;
        std::size_t pos = 0;
        LexResult out;
        std::vector<std::uint32_t> indents{0};
        std::optional<std::pair<std::uint32_t, std::size_t>> badDedent; // column, depth of the last E2002
        int nesting = 0;
        bool space = true;
        bool lineHasTokens = false;
        bool physLineHasTokens = false;
        std::vector<Brace> braces;
    };

} // namespace


std::string_view tokName(Tok t) { return kTokInfo[static_cast<std::size_t>(t)].name; }
std::string_view canonicalText(Tok t) { return kTokInfo[static_cast<std::size_t>(t)].canonical; }
std::string_view asciiText(Tok t) { return kTokInfo[static_cast<std::size_t>(t)].ascii; }

LexResult lex(const SourceFile& file, std::uint32_t fileId, Diagnostics& diags)
{
    return Lexer(file, fileId, diags).run();
}

bool isGreekLetter(std::string_view name)
{
    return tables().greekToWord.contains(std::string(name));
}

std::string asciiSpelling(std::string_view ident)
{
    std::string base(ident);
    std::string suffix;
    // Split trailing digits / primes off a single-letter non-ASCII identifier.
    if (!base.empty() && static_cast<unsigned char>(base[0]) >= 0x80)
    {
        const CodePoint cp = decodeUtf8(base, 0);
        suffix = base.substr(cp.length);
        base = base.substr(0, cp.length);
        if (const auto g = tables().greekToWord.find(base); g != tables().greekToWord.end())
            return suffix.empty() || !isDigit(suffix[0]) ? g->second + suffix : g->second + suffix;
        if (cp.value >= 0x1D4D0 && cp.value <= 0x1D4E9)
            return std::format("\\mathcal{{{}}}{}", static_cast<char>('A' + (cp.value - 0x1D4D0)), suffix);
        if (cp.value >= 0x1D538 && cp.value <= 0x1D551)
            return std::format("\\mathbb{{{}}}{}", static_cast<char>('A' + (cp.value - 0x1D538)), suffix);
        static const std::map<char32_t, char> bbHoles{{0x2102, 'C'}, {0x210D, 'H'}, {0x2115, 'N'}, {0x2119, 'P'},
                                                      {0x211A, 'Q'}, {0x211D, 'R'}, {0x2124, 'Z'}};
        if (const auto h = bbHoles.find(cp.value); h != bbHoles.end())
            return std::format("\\mathbb{{{}}}{}", h->second, suffix);
    }
    return std::string(ident);
}

std::vector<AliasRow> aliasTable()
{
    std::vector<AliasRow> rows;
    std::map<Tok, std::vector<std::string>> latexFor, asciiFor;
#define NTR_LATEX_TOK(command, Name) latexFor[Tok::Name].push_back("\\" command);
#define NTR_ASCII_OP(text, Name) asciiFor[Tok::Name].push_back(text);
#define NTR_UNICODE_ALT(text, Name) asciiFor[Tok::Name];
#include "tokens.def"

    auto join = [](const std::vector<std::string>& v)
    {
        std::string out;
        for (const auto& x : v) out += (out.empty() ? "" : " ") + x;
        return out;
    };
#define NTR_PUNCT(Name, canonical, ascii)                                                                 \
    if (latexFor.contains(Tok::Name) || asciiFor.contains(Tok::Name) || Tok::Name == Tok::TensorPow ||    \
        Tok::Name == Tok::LAngle)                                                                         \
        rows.push_back({canonical, join(latexFor[Tok::Name]), join(asciiFor[Tok::Name]), ascii, #Name});
#include "tokens.def"
    for (AliasRow& r : rows)
    {
        if (r.canonical == "^⊗") r.latex = "^{\\otimes n}";
        if (r.canonical == "⟨") { r.latex = "\\expval{A}, \\langle A \\rangle"; r.asciiCanonical = "\\expval{A}"; }
        if (r.canonical == "√") r.latex = "\\sqrt{x}";
        if (r.canonical == "Σ") r.latex = "\\sum_{j=a}^{b}";
        if (r.canonical == "∏") r.latex = "\\prod_{j=a}^{b}";
    }
    rows.push_back({"(a)/(b)", "\\frac{a}{b}", "", "(a)/(b)", "division"});
    rows.push_back({"|x⟩ ⟨x| ⟨a|b⟩", "\\ket{x} \\bra{x} \\braket{a|b}", "|x> <x| <a|b>", "|x> <x| <a|b>", "Dirac"});
    rows.push_back({"⟨a|O|b⟩", "\\mel{a}{O}{b}", "<a|O|b>", "<a|O|b>", "matrix element"});
    rows.push_back({"𝓗", "\\mathcal{H}", "", "\\mathcal{H}", "script identifier"});
    rows.push_back({"ℝ", "\\mathbb{R}", "", "\\mathbb{R}", "double-struck identifier"});
#define NTR_GREEK(word, letter) rows.push_back({letter, "\\" word, word, word, "Greek identifier"});
#include "tokens.def"
    rows.push_back({"φ", "\\varphi", "", "phi", "Greek identifier"});
    rows.push_back({"θ", "\\vartheta", "", "theta", "Greek identifier"});
    rows.push_back({"ε", "\\varepsilon", "", "epsilon", "Greek identifier"});
#define NTR_GATE_ALIAS(alias, canonical, dagger) \
    rows.push_back({std::string(canonical) + ((dagger) ? "†" : ""), "", alias, "", "gate alias"});
#include "tokens.def"
    rows.push_back({"(dropped)", "\\, \\; \\! \\quad \\left \\right", "", "", "layout only"});
    rows.push_back({"_k ^k", "", "", "", "Unicode subscript/superscript digits q₀ x² lex as q_0 x^2"});
    return rows;
}

bool joinsCleanly(std::string_view left, std::string_view right)
{
    auto tokens = [](std::string text, std::vector<std::pair<Tok, std::string>>& out)
    {
        SourceManager sm;
        Diagnostics d(sm);
        const std::uint32_t id = sm.add("", std::move(text));
        for (const Token& t : lex(sm.file(id), id, d).tokens)
            if (t.kind != Tok::Newline && t.kind != Tok::Indent && t.kind != Tok::Dedent && t.kind != Tok::End)
                out.emplace_back(t.kind, t.text + "|" + t.text2);
        return d.all().empty();
    };
    std::vector<std::pair<Tok, std::string>> apart, joined, second;
    if (!tokens(std::string(left), apart) || !tokens(std::string(right), second)) return false;
    if (!tokens(std::string(left) + std::string(right), joined)) return false;
    apart.insert(apart.end(), second.begin(), second.end());
    return apart == joined;
}

std::size_t editDistance(std::string_view a, std::string_view b)
{
    std::vector<std::size_t> prev(b.size() + 1), cur(b.size() + 1);
    for (std::size_t j = 0; j <= b.size(); ++j) prev[j] = j;
    for (std::size_t i = 1; i <= a.size(); ++i)
    {
        cur[0] = i;
        for (std::size_t j = 1; j <= b.size(); ++j)
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

} // namespace Noether
