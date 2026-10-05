#include "interop/Qasm.hpp"

#include "Compiler.hpp"
#include "Embedded.hpp"
#include "interop/Lowering.hpp"

#include <StabilizerState.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <deque>
#include <filesystem>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <numbers>
#include <set>
#include <unordered_map>



namespace Noether::Interop
{

namespace
{
    constexpr double kPi = std::numbers::pi;
    constexpr std::size_t kMaxNesting = 256;        // statements and expressions
    constexpr std::size_t kMaxIncludeDepth = 16;
    constexpr std::size_t kMaxOps = 10'000'000;     // after inlining and unrolling

    struct Fail
    {
        std::string code;
        Span span;
        std::string message;
    };


    // ---- Lexer: produces tokens on demand, so a file is never held as a token array ----

    enum class T : std::uint8_t { End, Ident, Number, String, Hw, Duration, Punct, Pragma, Annotation };

    struct Token
    {
        T kind = T::End;
        std::uint32_t b = 0, e = 0;
        double num = 0.0;
        bool integer = false;
        std::string_view text; // identifier, punctuation, string contents
    };

    bool identStart(unsigned char c) { return std::isalpha(c) || c == '_' || c >= 0x80; }
    bool identChar(unsigned char c) { return std::isalnum(c) || c == '_' || c >= 0x80; }

    class Lexer
    {
        public:
        Lexer(std::string_view source, std::uint32_t file) : s(source), fileId(file) {}

        Span span(std::size_t b, std::size_t e) const { return Span{fileId, static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(e)}; }

        Token next()
        {
            skip();
            Token t;
            t.b = t.e = static_cast<std::uint32_t>(pos);
            if (pos >= s.size()) return t;
            const std::size_t b = pos;
            const auto c = static_cast<unsigned char>(s[pos]);
            if (lineStart(b) && (s.substr(b).starts_with("#pragma") || word(b, "pragma"))) return rest(T::Pragma, b);
            if (c == '@' && lineStart(b) && b + 1 < s.size() && identStart(static_cast<unsigned char>(s[b + 1]))) return rest(T::Annotation, b);
            if (identStart(c))
            {
                while (pos < s.size() && identChar(static_cast<unsigned char>(s[pos]))) ++pos;
                return make(T::Ident, b);
            }
            if (c == '$' && b + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[b + 1])))
            {
                ++pos;
                while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) ++pos;
                Token h = make(T::Hw, b);
                std::uint64_t v = 0;
                const auto r = std::from_chars(s.data() + b + 1, s.data() + pos, v);
                if (r.ec != std::errc{}) throw Fail{"E9001", span(b, pos), "physical qubit index out of range"};
                h.num = static_cast<double>(v);
                h.integer = true;
                return h;
            }
            if (std::isdigit(c) || (c == '.' && b + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[b + 1])))) return number(b);
            if (c == '"' || c == '\'')
            {
                const std::size_t close = s.find(static_cast<char>(c), b + 1);
                const std::size_t eol = s.find('\n', b + 1);
                if (close == std::string_view::npos || close > eol) throw Fail{"E9001", span(b, b + 1), "unterminated string"};
                pos = close + 1;
                Token str = make(T::String, b);
                str.text = s.substr(b + 1, close - b - 1);
                return str;
            }
            static constexpr std::string_view two[] = {"->", "==", "!=", "<=", ">=", "**", "&&", "||", "++", "<<", ">>",
                                                       "+=", "-=", "*=", "/=", "%=", "^=", "&=", "|="};
            for (const std::string_view p : two)
                if (s.substr(b, 2) == p)
                {
                    pos += 2;
                    return make(T::Punct, b);
                }
            if (std::string_view("()[]{},;:=<>+-*/%^!@~&|.").find(static_cast<char>(c)) != std::string_view::npos)
            {
                ++pos;
                return make(T::Punct, b);
            }
            throw Fail{"E9001", span(b, b + 1), std::format("unexpected character `{}`", s.substr(b, 1))};
        }

        private:
        Token make(T kind, std::size_t b) const
        {
            Token t;
            t.kind = kind;
            t.b = static_cast<std::uint32_t>(b);
            t.e = static_cast<std::uint32_t>(pos);
            t.text = s.substr(b, pos - b);
            return t;
        }

        bool lineStart(std::size_t b) const
        {
            while (b > 0 && (s[b - 1] == ' ' || s[b - 1] == '\t')) --b;
            return b == 0 || s[b - 1] == '\n';
        }

        bool word(std::size_t b, std::string_view w) const
        {
            return s.substr(b, w.size()) == w && (b + w.size() >= s.size() || !identChar(static_cast<unsigned char>(s[b + w.size()])));
        }

        Token rest(T kind, std::size_t b)
        {
            pos = s.find('\n', b);
            if (pos == std::string_view::npos) pos = s.size();
            return make(kind, b);
        }

        void skip()
        {
            while (pos < s.size())
            {
                const char c = s[pos];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos;
                else if (s.substr(pos, 2) == "//")
                {
                    pos = s.find('\n', pos);
                    if (pos == std::string_view::npos) pos = s.size();
                }
                else if (s.substr(pos, 2) == "/*")
                {
                    const std::size_t end = s.find("*/", pos + 2);
                    if (end == std::string_view::npos) throw Fail{"E9001", span(pos, pos + 2), "unterminated comment"};
                    pos = end + 2;
                }
                else break;
            }
        }

        Token number(std::size_t b)
        {
            std::string digits;
            bool integer = true;
            int base = 10;
            if (s[pos] == '0' && pos + 1 < s.size() && std::string_view("xXbBoO").find(s[pos + 1]) != std::string_view::npos)
            {
                const char p = static_cast<char>(std::tolower(static_cast<unsigned char>(s[pos + 1])));
                base = p == 'x' ? 16 : p == 'b' ? 2 : 8;
                pos += 2;
                while (pos < s.size() && (std::isxdigit(static_cast<unsigned char>(s[pos])) || s[pos] == '_'))
                    if (s[pos++] != '_') digits += s[pos - 1];
            }
            else
            {
                auto run = [&]
                {
                    while (pos < s.size() && (std::isdigit(static_cast<unsigned char>(s[pos])) || s[pos] == '_'))
                        if (s[pos++] != '_') digits += s[pos - 1];
                };
                run();
                if (pos < s.size() && s[pos] == '.')
                {
                    integer = false;
                    digits += s[pos++];
                    run();
                }
                if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E'))
                {
                    std::size_t q = pos + 1;
                    if (q < s.size() && (s[q] == '+' || s[q] == '-')) ++q;
                    if (q < s.size() && std::isdigit(static_cast<unsigned char>(s[q])))
                    {
                        integer = false;
                        digits.append(s.substr(pos, q - pos));
                        pos = q;
                        run();
                    }
                }
            }
            Token t = make(T::Number, b);
            t.integer = integer;
            if (digits.empty() || digits == ".") throw Fail{"E9001", span(b, pos), "malformed number"};
            if (base == 10)
            {
                if (digits.front() == '.') digits.insert(digits.begin(), '0');
                const auto r = std::from_chars(digits.data(), digits.data() + digits.size(), t.num);
                if (r.ec != std::errc{} && r.ec != std::errc::result_out_of_range) throw Fail{"E9001", span(b, pos), "malformed number"};
            }
            else
            {
                std::uint64_t v = 0;
                const auto r = std::from_chars(digits.data(), digits.data() + digits.size(), v, base);
                if (r.ec != std::errc{}) throw Fail{"E9001", span(b, pos), "malformed integer"};
                t.num = static_cast<double>(v);
            }
            // A unit suffix makes a duration; `im` an imaginary literal.
            const std::size_t u = pos;
            if (s.substr(u).starts_with("\xC2\xB5s")) pos += 3;
            else
                while (pos < s.size() && identChar(static_cast<unsigned char>(s[pos]))) ++pos;
            const std::string_view unit = s.substr(u, pos - u);
            if (unit.empty()) return t;
            if (unit == "ns" || unit == "us" || unit == "\xC2\xB5s" || unit == "ms" || unit == "s" || unit == "dt")
            {
                t.kind = T::Duration;
                t.e = static_cast<std::uint32_t>(pos);
                return t;
            }
            if (unit == "im") throw Fail{"E9001", span(b, pos), "complex literals are not supported"};
            throw Fail{"E9001", span(b, pos), std::format("unexpected `{}` after a number", unit)};
        }

        std::string_view s;
        std::uint32_t fileId;
        std::size_t pos = 0;
    };


    // ---- Syntax ----

    struct Expr
    {
        enum class K : std::uint8_t { Num, Name, Unary, Binary, Call, Index };
        K k = K::Num;
        double v = 0.0;
        std::string_view text; // Name, Call: identifier; Unary, Binary: operator
        Span span;
        std::vector<Expr> kids;
    };

    // A register selection: all, [i], [a:b], [a:s:b] (inclusive) or [{i, j, …}].
    struct Sel
    {
        enum class K : std::uint8_t { All, One, Range, Set };
        K k = K::All;
        std::optional<Expr> start, step, end;
        std::vector<Expr> items; // One: the index; Set: the indices
    };

    struct QArg
    {
        std::string_view name;
        Span span;
        bool physical = false;
        std::size_t phys = 0;
        Sel sel;
    };

    struct Modifier
    {
        enum class K : std::uint8_t { Ctrl, NegCtrl, Inv, Pow };
        K k = K::Ctrl;
        std::optional<Expr> arg;
        Span span;
    };

    struct Stmt
    {
        enum class K : std::uint8_t { Empty, GateCall, Measure, Reset, Barrier, If, For, Block, QDecl, CDecl, Const, Input, Let, GateDef,
                                      Include, Ignored };
        K k = K::Empty;
        Span span;
        Span nameSpan;
        std::string_view name;     // gate, register, variable, loop variable, ignored construct
        bool scalar = false;       // `qubit q;` / `bit b;`
        std::vector<Modifier> mods;
        std::vector<Expr> params;  // gate call
        std::vector<QArg> qargs;   // operands; Let: the concatenated parts
        std::vector<QArg> cargs;   // measurement target
        std::optional<Expr> value; // declaration size, const value, if condition
        Sel range;                 // for
        std::vector<Stmt> body, orelse;
        bool hasElse = false;
        std::vector<std::string_view> paramNames, qubitNames; // gate definition
        std::string path;          // include
    };

    class Parser
    {
        public:
        Parser(std::string_view text, std::uint32_t file) : lex(text, file) {}

        void setVersion(int v) { version = v; }
        Span span(std::uint32_t b, std::uint32_t e) const { return lex.span(b, e); }
        bool atEnd() { return peek().kind == T::End; }

        // `OPENQASM 2.0;` / `OPENQASM 3;` when present.
        std::optional<std::pair<int, Span>> header()
        {
            if (!isWord(peek(), "OPENQASM")) return std::nullopt;
            const Token kw = take();
            const Token v = take();
            if (v.kind != T::Number) throw Fail{"E9001", span(v.b, v.e), "expected a version number after OPENQASM"};
            expect(";");
            const int major = static_cast<int>(v.num);
            if (major != 2 && major != 3) throw Fail{"E9001", span(v.b, v.e), std::format("OpenQASM version {} is not supported", v.text)};
            return std::pair{major, span(kw.b, v.e)};
        }

        Stmt statement(std::size_t depth)
        {
            if (depth > kMaxNesting) throw Fail{"E9001", here(), "statements nest too deeply"};
            const Token t = peek();
            Stmt s;
            s.span = span(t.b, t.e);
            if (t.kind == T::Pragma || t.kind == T::Annotation)
            {
                const Token p = take();
                s.k = Stmt::K::Ignored;
                s.name = p.kind == T::Pragma ? "pragma" : "annotation";
                return s;
            }
            if (isPunct(t, ";"))
            {
                take();
                return s;
            }
            if (isPunct(t, "{"))
            {
                s.k = Stmt::K::Block;
                s.body = block(depth);
                return finish(s);
            }
            if (t.kind != T::Ident) throw Fail{"E9001", span(t.b, t.e), std::format("unexpected `{}`", t.text.empty() ? "end of file" : t.text)};
            const std::string_view w = t.text;

            if (w == "include")
            {
                take();
                const Token p = take();
                if (p.kind != T::String) throw Fail{"E9001", span(p.b, p.e), "expected a quoted file name"};
                s.k = Stmt::K::Include;
                s.path = std::string(p.text);
                expect(";");
                return finish(s);
            }
            if (w == "qreg" || w == "creg")
            {
                take();
                s.k = w == "qreg" ? Stmt::K::QDecl : Stmt::K::CDecl;
                nameInto(s);
                expect("[");
                s.value = expr(depth);
                expect("]");
                expect(";");
                return finish(s);
            }
            if (w == "qubit" || w == "bit")
            {
                take();
                s.k = w == "qubit" ? Stmt::K::QDecl : Stmt::K::CDecl;
                if (isPunct(peek(), "["))
                {
                    take();
                    s.value = expr(depth);
                    expect("]");
                }
                else s.scalar = true;
                nameInto(s);
                if (s.k == Stmt::K::CDecl && isPunct(peek(), "="))
                {
                    take();
                    if (!isWord(peek(), "measure")) throw Fail{"E9001", here(), "a bit may only be initialised by `measure`"};
                    take();
                    s.qargs.push_back(qarg(depth));
                }
                expect(";");
                return finish(s);
            }
            if (w == "const" || w == "input")
            {
                take();
                s.k = w == "const" ? Stmt::K::Const : Stmt::K::Input;
                type(depth);
                nameInto(s);
                if (s.k == Stmt::K::Const)
                {
                    expect("=");
                    s.value = expr(depth);
                }
                expect(";");
                return finish(s);
            }
            if (w == "let")
            {
                take();
                s.k = Stmt::K::Let;
                nameInto(s);
                expect("=");
                s.qargs.push_back(qarg(depth));
                while (isPunct(peek(), "++"))
                {
                    take();
                    s.qargs.push_back(qarg(depth));
                }
                expect(";");
                return finish(s);
            }
            if (w == "gate") return gateDef(depth);
            if (w == "measure")
            {
                take();
                s.k = Stmt::K::Measure;
                s.qargs.push_back(qarg(depth));
                if (isPunct(peek(), "->"))
                {
                    take();
                    s.cargs.push_back(qarg(depth));
                }
                expect(";");
                return finish(s);
            }
            if (w == "reset" || w == "barrier")
            {
                take();
                s.k = w == "reset" ? Stmt::K::Reset : Stmt::K::Barrier;
                if (!isPunct(peek(), ";")) s.qargs = qargList(depth);
                expect(";");
                return finish(s);
            }
            if (w == "delay")
            {
                take();
                skipBracket();
                s.k = Stmt::K::Ignored;
                s.name = "delay";
                if (!isPunct(peek(), ";")) s.qargs = qargList(depth);
                expect(";");
                return finish(s);
            }
            if (w == "duration" || w == "stretch")
            {
                take();
                s.k = Stmt::K::Ignored;
                s.name = w;
                skipTo(";");
                return finish(s);
            }
            if (w == "if")
            {
                take();
                s.k = Stmt::K::If;
                expect("(");
                s.value = expr(depth);
                expect(")");
                s.body = bodyOf(depth);
                if (isWord(peek(), "else"))
                {
                    take();
                    s.hasElse = true;
                    s.orelse = bodyOf(depth);
                }
                return finish(s);
            }
            if (w == "for")
            {
                take();
                s.k = Stmt::K::For;
                if (isType(peek().text) && peek(1).kind == T::Ident && !isWord(peek(1), "in")) type(depth);
                else if (isType(peek().text) && isPunct(peek(1), "[")) type(depth);
                nameInto(s);
                if (!isWord(peek(), "in")) throw Fail{"E9001", here(), "expected `in`"};
                take();
                if (isPunct(peek(), "{"))
                {
                    take();
                    s.range.k = Sel::K::Set;
                    s.range.items = exprList("}", depth);
                }
                else if (isPunct(peek(), "["))
                {
                    take();
                    s.range = rangeBody(depth);
                    if (s.range.k != Sel::K::Range) throw Fail{"E9001", s.nameSpan, "a for loop needs a range [a:b] or [a:s:b], or a set {…}"};
                }
                else throw Fail{"E9001", here(), "a for loop needs a constant range [a:b] or set {…}"};
                s.body = bodyOf(depth);
                return finish(s);
            }
            if (w == "box")
            {
                take();
                skipBracket();
                s.k = Stmt::K::Block;
                if (!isPunct(peek(), "{")) throw Fail{"E9001", here(), "expected `{` after box"};
                s.body = block(depth);
                return finish(s);
            }
            static const std::set<std::string_view> unsupported{"opaque", "def", "extern", "defcal", "cal", "defcalgrammar", "while",
                                                                "switch", "break", "continue", "return", "end", "case", "default",
                                                                "output", "port", "frame", "waveform"};
            if (unsupported.contains(w))
                throw Fail{"E9001", span(t.b, t.e), std::format("`{}` is not supported by the importer", w)};
            if (isType(w))
                throw Fail{"E9001", span(t.b, t.e), std::format("runtime classical variables (`{}`) are not supported; use `const`", w)};

            // An assignment `c = measure q;` / `c[i] = measure q[j];`, or a gate application.
            std::size_t k = 1;
            if (isPunct(peek(1), "["))
            {
                int depthB = 0;
                for (;; ++k)
                {
                    const Token& x = peek(k);
                    if (x.kind == T::End) break;
                    if (isPunct(x, "[")) ++depthB;
                    else if (isPunct(x, "]") && --depthB == 0)
                    {
                        ++k;
                        break;
                    }
                }
            }
            const Token& after = peek(k);
            if (after.kind == T::Punct && (after.text == "=" || (after.text.size() == 2 && after.text[1] == '=' && after.text != "==")))
            {
                s.cargs.push_back(qarg(depth));
                const Token op = take();
                if (op.text != "=" || !isWord(peek(), "measure"))
                    throw Fail{"E9001", span(t.b, op.e), "runtime classical assignments are not supported; only `c = measure q;`"};
                take();
                s.k = Stmt::K::Measure;
                s.qargs.push_back(qarg(depth));
                expect(";");
                return finish(s);
            }
            return gateCall(depth);
        }

        private:
        static bool isPunct(const Token& t, std::string_view p) { return t.kind == T::Punct && t.text == p; }
        static bool isWord(const Token& t, std::string_view w) { return t.kind == T::Ident && t.text == w; }
        static bool isType(std::string_view w)
        {
            return w == "int" || w == "uint" || w == "float" || w == "angle" || w == "bool" || w == "complex";
        }

        const Token& peek(std::size_t k = 0)
        {
            while (buf.size() <= k) buf.push_back(lex.next());
            return buf[k];
        }

        Token take()
        {
            peek();
            Token t = buf.front();
            buf.pop_front();
            if (t.kind != T::End) lastEnd = t.e;
            return t;
        }

        Span here() { return span(peek().b, std::max(peek().e, peek().b + 1)); }

        void expect(std::string_view p)
        {
            if (!isPunct(peek(), p))
            {
                const Token& t = peek();
                throw Fail{"E9001", here(), std::format("expected `{}`, found `{}`", p, t.kind == T::End ? "end of file" : t.text)};
            }
            take();
        }

        Stmt finish(Stmt& s)
        {
            s.span.end = std::max(s.span.end, lastEnd);
            return std::move(s);
        }

        void nameInto(Stmt& s)
        {
            const Token n = take();
            if (n.kind != T::Ident) throw Fail{"E9001", span(n.b, std::max(n.e, n.b + 1)), "expected a name"};
            s.name = n.text;
            s.nameSpan = span(n.b, n.e);
        }

        void skipBracket()
        {
            if (!isPunct(peek(), "[")) return;
            int d = 0;
            do
            {
                const Token t = take();
                if (t.kind == T::End) throw Fail{"E9001", here(), "unclosed `[`"};
                if (isPunct(t, "[")) ++d;
                else if (isPunct(t, "]")) --d;
            } while (d > 0);
        }

        void skipTo(std::string_view p)
        {
            while (!isPunct(peek(), p))
                if (take().kind == T::End) throw Fail{"E9001", here(), std::format("expected `{}`", p)};
            take();
        }

        // A classical type with its optional designator (int[32], complex[float[64]]).
        void type(std::size_t)
        {
            const Token t = take();
            if (t.kind != T::Ident || !(isType(t.text) || t.text == "bit"))
                throw Fail{"E9001", span(t.b, std::max(t.e, t.b + 1)), "expected a type (int, uint, float, angle, bool)"};
            skipBracket();
        }

        std::vector<Stmt> block(std::size_t depth)
        {
            expect("{");
            std::vector<Stmt> out;
            while (!isPunct(peek(), "}"))
            {
                if (atEnd()) throw Fail{"E9001", here(), "expected `}`"};
                out.push_back(statement(depth + 1));
            }
            take();
            return out;
        }

        std::vector<Stmt> bodyOf(std::size_t depth)
        {
            if (isPunct(peek(), "{")) return block(depth);
            std::vector<Stmt> out;
            out.push_back(statement(depth + 1));
            return out;
        }

        Stmt gateDef(std::size_t depth)
        {
            const Token kw = take();
            Stmt s;
            s.k = Stmt::K::GateDef;
            s.span = span(kw.b, kw.e);
            nameInto(s);
            if (isPunct(peek(), "("))
            {
                take();
                while (!isPunct(peek(), ")"))
                {
                    const Token p = take();
                    if (p.kind != T::Ident) throw Fail{"E9001", span(p.b, std::max(p.e, p.b + 1)), "expected a parameter name"};
                    s.paramNames.push_back(p.text);
                    if (!isPunct(peek(), ")")) expect(",");
                }
                take();
            }
            while (!isPunct(peek(), "{"))
            {
                const Token q = take();
                if (q.kind != T::Ident) throw Fail{"E9001", span(q.b, std::max(q.e, q.b + 1)), "expected a qubit name or `{`"};
                s.qubitNames.push_back(q.text);
                if (!isPunct(peek(), "{")) expect(",");
            }
            s.body = block(depth);
            return finish(s);
        }

        Stmt gateCall(std::size_t depth)
        {
            Stmt s;
            s.k = Stmt::K::GateCall;
            s.span = span(peek().b, peek().e);
            for (;;)
            {
                const Token& t = peek();
                const bool mod = t.kind == T::Ident && (t.text == "ctrl" || t.text == "negctrl" || t.text == "inv" || t.text == "pow") &&
                                 (isPunct(peek(1), "@") || isPunct(peek(1), "("));
                if (!mod) break;
                const Token m = take();
                Modifier mo;
                mo.k = m.text == "ctrl" ? Modifier::K::Ctrl : m.text == "negctrl" ? Modifier::K::NegCtrl : m.text == "inv" ? Modifier::K::Inv : Modifier::K::Pow;
                mo.span = span(m.b, m.e);
                if (isPunct(peek(), "("))
                {
                    take();
                    mo.arg = expr(depth);
                    expect(")");
                }
                if (mo.k == Modifier::K::Pow && !mo.arg) throw Fail{"E9001", mo.span, "pow needs an exponent: pow(r) @"};
                if (mo.k == Modifier::K::Inv && mo.arg) throw Fail{"E9001", mo.span, "inv takes no argument"};
                expect("@");
                s.mods.push_back(std::move(mo));
            }
            nameInto(s);
            if (isPunct(peek(), "("))
            {
                take();
                s.params = exprList(")", depth);
            }
            skipBracket(); // a duration designator
            if (!isPunct(peek(), ";")) s.qargs = qargList(depth);
            expect(";");
            return finish(s);
        }

        std::vector<QArg> qargList(std::size_t depth)
        {
            std::vector<QArg> out;
            out.push_back(qarg(depth));
            while (isPunct(peek(), ","))
            {
                take();
                out.push_back(qarg(depth));
            }
            return out;
        }

        QArg qarg(std::size_t depth)
        {
            const Token t = take();
            QArg a;
            a.span = span(t.b, std::max(t.e, t.b + 1));
            if (t.kind == T::Hw)
            {
                a.physical = true;
                a.phys = static_cast<std::size_t>(t.num);
                a.name = t.text;
                return a;
            }
            if (t.kind != T::Ident) throw Fail{"E9001", a.span, std::format("expected a qubit or bit operand, found `{}`", t.kind == T::End ? "end of file" : t.text)};
            a.name = t.text;
            if (isPunct(peek(), "["))
            {
                take();
                if (isPunct(peek(), "{"))
                {
                    take();
                    a.sel.k = Sel::K::Set;
                    a.sel.items = exprList("}", depth);
                    expect("]");
                }
                else a.sel = rangeBody(depth);
                a.span.end = lastEnd;
            }
            return a;
        }

        // After `[`: i], a:b], a:s:b], :], consuming the `]`.
        Sel rangeBody(std::size_t depth)
        {
            Sel sel;
            std::vector<std::optional<Expr>> parts(1);
            for (;;)
            {
                if (!isPunct(peek(), ":") && !isPunct(peek(), "]")) parts.back() = expr(depth);
                if (isPunct(peek(), ":"))
                {
                    take();
                    parts.emplace_back();
                    if (parts.size() > 3) throw Fail{"E9001", here(), "a range has at most three parts"};
                    continue;
                }
                break;
            }
            expect("]");
            if (parts.size() == 1)
            {
                if (!parts[0]) throw Fail{"E9001", here(), "empty index"};
                sel.k = Sel::K::One;
                sel.items.push_back(std::move(*parts[0]));
                return sel;
            }
            sel.k = Sel::K::Range;
            sel.start = std::move(parts[0]);
            if (parts.size() == 3)
            {
                sel.step = std::move(parts[1]);
                sel.end = std::move(parts[2]);
            }
            else sel.end = std::move(parts[1]);
            return sel;
        }

        std::vector<Expr> exprList(std::string_view close, std::size_t depth)
        {
            std::vector<Expr> out;
            while (!isPunct(peek(), close))
            {
                out.push_back(expr(depth));
                if (!isPunct(peek(), close)) expect(",");
            }
            take();
            return out;
        }

        // ---- Expressions, by precedence (OpenQASM 3; in version 2 `^` is the power operator) ----

        int binaryPrecedence(const Token& t) const
        {
            if (t.kind != T::Punct) return -1;
            const std::string_view o = t.text;
            if (o == "||") return 1;
            if (o == "&&") return 2;
            if (o == "|") return 3;
            if (o == "^") return version == 2 ? -1 : 4;
            if (o == "&") return 5;
            if (o == "==" || o == "!=") return 6;
            if (o == "<" || o == ">" || o == "<=" || o == ">=") return 7;
            if (o == "<<" || o == ">>") return 8;
            if (o == "+" || o == "-") return 9;
            if (o == "*" || o == "/" || o == "%") return 10;
            return -1;
        }

        Expr expr(std::size_t depth, int minPrec = 1)
        {
            if (depth > kMaxNesting) throw Fail{"E9001", here(), "expression nests too deeply"};
            Expr lhs = unary(depth + 1);
            for (;;)
            {
                const int p = binaryPrecedence(peek());
                if (p < minPrec) return lhs;
                const Token op = take();
                Expr rhs = expr(depth + 1, p + 1);
                Expr b;
                b.k = Expr::K::Binary;
                b.text = op.text;
                b.span = Span::join(lhs.span, rhs.span);
                b.kids.push_back(std::move(lhs));
                b.kids.push_back(std::move(rhs));
                lhs = std::move(b);
            }
        }

        Expr unary(std::size_t depth)
        {
            if (depth > kMaxNesting) throw Fail{"E9001", here(), "expression nests too deeply"};
            const Token& t = peek();
            if (t.kind == T::Punct && (t.text == "-" || t.text == "+" || t.text == "!" || t.text == "~"))
            {
                const Token op = take();
                Expr inner = unary(depth + 1);
                if (op.text == "+") return inner;
                Expr u;
                u.k = Expr::K::Unary;
                u.text = op.text;
                u.span = span(op.b, inner.span.end);
                u.kids.push_back(std::move(inner));
                return u;
            }
            Expr base = postfix(depth + 1);
            const bool power = isPunct(peek(), "**") || (version == 2 && isPunct(peek(), "^"));
            if (!power) return base;
            const Token op = take();
            Expr exponent = unary(depth + 1); // right associative; binds tighter than a leading minus
            Expr b;
            b.k = Expr::K::Binary;
            b.text = "**";
            b.span = Span::join(base.span, exponent.span);
            b.kids.push_back(std::move(base));
            b.kids.push_back(std::move(exponent));
            (void)op;
            return b;
        }

        Expr postfix(std::size_t depth)
        {
            const Token t = take();
            Expr e;
            e.span = span(t.b, std::max(t.e, t.b + 1));
            if (t.kind == T::Number || t.kind == T::Duration)
            {
                e.k = Expr::K::Num;
                e.v = t.num;
                return e;
            }
            if (isPunct(t, "("))
            {
                Expr inner = expr(depth + 1);
                expect(")");
                return inner;
            }
            if (t.kind != T::Ident)
                throw Fail{"E9001", e.span, std::format("expected an expression, found `{}`", t.kind == T::End ? "end of file" : t.text)};
            e.text = t.text;
            e.k = Expr::K::Name;
            if (isType(t.text) && isPunct(peek(), "[")) skipBracket(); // float[64](x)
            if (isPunct(peek(), "("))
            {
                take();
                e.k = Expr::K::Call;
                e.kids = exprList(")", depth + 1);
                e.span.end = lastEnd;
            }
            else if (isPunct(peek(), "["))
            {
                take();
                e.k = Expr::K::Index;
                e.kids.push_back(expr(depth + 1));
                expect("]");
                e.span.end = lastEnd;
            }
            return e;
        }

        Lexer lex;
        std::deque<Token> buf;
        std::uint32_t lastEnd = 0;
        int version = 3;
    };


    // ---- Gate library ----

    using Params = std::vector<double>;

    Gate g1(Prim p, Qubit q, Params a = {}) { return makeGate(p, {q}, std::move(a)); }
    Gate gc(Prim p, QubitList controls, Qubit t, Params a = {}) { return makeGate(p, {t}, std::move(a), std::move(controls)); }
    Gate gphase(double a, QubitList controls = {}) { return makeGate(Prim::GPhase, {}, {a}, std::move(controls)); }

    struct NativeDef
    {
        std::string_view name;
        std::size_t params, qubits;
        Seq (*make)(const Params& p, const QubitList& q);
    };

    // stdgates.inc under the OpenQASM 3 definition U(θ, φ, λ) = e^{iθ/2}·u3(θ, φ, λ): each entry equals its
    // embedded definition exactly, global phase included, except CX (see the include test).
    const std::vector<NativeDef>& stdgatesNatives()
    {
        static const std::vector<NativeDef> v{
            {"p", 1, 1, [](const Params& p, const QubitList& q) { return Seq{g1(Prim::Phase, q[0], {p[0]})}; }},
            {"x", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::X, q[0])}; }},
            {"y", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::Y, q[0])}; }},
            {"z", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::Z, q[0])}; }},
            {"h", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::H, q[0])}; }},
            {"s", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::S, q[0])}; }},
            {"sdg", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::Sdg, q[0])}; }},
            {"t", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::T, q[0])}; }},
            {"tdg", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::Tdg, q[0])}; }},
            {"sx", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::SX, q[0])}; }},
            {"rx", 1, 1, [](const Params& p, const QubitList& q) { return Seq{g1(Prim::RX, q[0], {p[0]})}; }},
            {"ry", 1, 1, [](const Params& p, const QubitList& q) { return Seq{g1(Prim::RY, q[0], {p[0]})}; }},
            {"rz", 1, 1, [](const Params& p, const QubitList& q) { return Seq{g1(Prim::RZ, q[0], {p[0]})}; }},
            {"cx", 0, 2, [](const Params&, const QubitList& q) { return Seq{gc(Prim::X, {q[0]}, q[1])}; }},
            {"cy", 0, 2, [](const Params&, const QubitList& q) { return Seq{gc(Prim::Y, {q[0]}, q[1])}; }},
            {"cz", 0, 2, [](const Params&, const QubitList& q) { return Seq{gc(Prim::Z, {q[0]}, q[1])}; }},
            {"cp", 1, 2, [](const Params& p, const QubitList& q) { return Seq{gc(Prim::Phase, {q[0]}, q[1], {p[0]})}; }},
            {"crx", 1, 2, [](const Params& p, const QubitList& q) { return Seq{gc(Prim::RX, {q[0]}, q[1], {p[0]})}; }},
            {"cry", 1, 2, [](const Params& p, const QubitList& q) { return Seq{gc(Prim::RY, {q[0]}, q[1], {p[0]})}; }},
            {"crz", 1, 2, [](const Params& p, const QubitList& q) { return Seq{gc(Prim::RZ, {q[0]}, q[1], {p[0]})}; }},
            {"ch", 0, 2, [](const Params&, const QubitList& q) { return Seq{gc(Prim::H, {q[0]}, q[1])}; }},
            {"swap", 0, 2, [](const Params&, const QubitList& q) { return Seq{makeGate(Prim::Swap, {q[0], q[1]})}; }},
            {"ccx", 0, 3, [](const Params&, const QubitList& q) { return Seq{gc(Prim::X, {q[0], q[1]}, q[2])}; }},
            {"cswap", 0, 3, [](const Params&, const QubitList& q) { return Seq{makeGate(Prim::Swap, {q[1], q[2]}, {}, {q[0]})}; }},
            {"cu", 4, 2,
             [](const Params& p, const QubitList& q) { return Seq{gc(Prim::U3, {q[0]}, q[1], {p[0], p[1], p[2]}), gphase(p[3], {q[0]})}; }},
            {"CX", 0, 2, [](const Params&, const QubitList& q) { return Seq{gc(Prim::X, {q[0]}, q[1])}; }},
            {"phase", 1, 1, [](const Params& p, const QubitList& q) { return Seq{g1(Prim::Phase, q[0], {p[0]})}; }},
            {"cphase", 1, 2, [](const Params& p, const QubitList& q) { return Seq{gc(Prim::Phase, {q[0]}, q[1], {p[0]})}; }},
            {"id", 0, 1, [](const Params&, const QubitList&) { return Seq{}; }},
            {"u1", 1, 1, [](const Params& p, const QubitList& q) { return Seq{g1(Prim::Phase, q[0], {p[0]})}; }},
            {"u2", 2, 1,
             [](const Params& p, const QubitList& q) { return Seq{g1(Prim::U3, q[0], {kPi / 2, p[0], p[1]}), gphase(-(p[0] + p[1]) / 2)}; }},
            {"u3", 3, 1,
             [](const Params& p, const QubitList& q) { return Seq{g1(Prim::U3, q[0], {p[0], p[1], p[2]}), gphase(-(p[1] + p[2]) / 2)}; }},
        };
        return v;
    }

    // qelib1.inc as Qiskit defines each gate (u2 = u3(π/2, φ, λ), rz = Rz, …). OpenQASM 2 has no
    // modifiers, so only equality up to global phase with the embedded definitions matters there.
    const std::vector<NativeDef>& qelib1Natives()
    {
        static const std::vector<NativeDef> v{
            {"u3", 3, 1, [](const Params& p, const QubitList& q) { return Seq{g1(Prim::U3, q[0], {p[0], p[1], p[2]})}; }},
            {"u2", 2, 1, [](const Params& p, const QubitList& q) { return Seq{g1(Prim::U3, q[0], {kPi / 2, p[0], p[1]})}; }},
            {"u1", 1, 1, [](const Params& p, const QubitList& q) { return Seq{g1(Prim::Phase, q[0], {p[0]})}; }},
            {"cx", 0, 2, [](const Params&, const QubitList& q) { return Seq{gc(Prim::X, {q[0]}, q[1])}; }},
            {"id", 0, 1, [](const Params&, const QubitList&) { return Seq{}; }},
            {"u0", 1, 1, [](const Params&, const QubitList&) { return Seq{}; }},
            {"u", 3, 1, [](const Params& p, const QubitList& q) { return Seq{g1(Prim::U3, q[0], {p[0], p[1], p[2]})}; }},
            {"p", 1, 1, [](const Params& p, const QubitList& q) { return Seq{g1(Prim::Phase, q[0], {p[0]})}; }},
            {"x", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::X, q[0])}; }},
            {"y", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::Y, q[0])}; }},
            {"z", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::Z, q[0])}; }},
            {"h", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::H, q[0])}; }},
            {"s", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::S, q[0])}; }},
            {"sdg", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::Sdg, q[0])}; }},
            {"t", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::T, q[0])}; }},
            {"tdg", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::Tdg, q[0])}; }},
            {"rx", 1, 1, [](const Params& p, const QubitList& q) { return Seq{g1(Prim::RX, q[0], {p[0]})}; }},
            {"ry", 1, 1, [](const Params& p, const QubitList& q) { return Seq{g1(Prim::RY, q[0], {p[0]})}; }},
            {"rz", 1, 1, [](const Params& p, const QubitList& q) { return Seq{g1(Prim::RZ, q[0], {p[0]})}; }},
            {"sx", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::SX, q[0])}; }},
            {"sxdg", 0, 1, [](const Params&, const QubitList& q) { return Seq{g1(Prim::SX, q[0]), g1(Prim::X, q[0])}; }},
            {"cz", 0, 2, [](const Params&, const QubitList& q) { return Seq{gc(Prim::Z, {q[0]}, q[1])}; }},
            {"cy", 0, 2, [](const Params&, const QubitList& q) { return Seq{gc(Prim::Y, {q[0]}, q[1])}; }},
            {"swap", 0, 2, [](const Params&, const QubitList& q) { return Seq{makeGate(Prim::Swap, {q[0], q[1]})}; }},
            {"ch", 0, 2, [](const Params&, const QubitList& q) { return Seq{gc(Prim::H, {q[0]}, q[1])}; }},
            {"ccx", 0, 3, [](const Params&, const QubitList& q) { return Seq{gc(Prim::X, {q[0], q[1]}, q[2])}; }},
            {"cswap", 0, 3, [](const Params&, const QubitList& q) { return Seq{makeGate(Prim::Swap, {q[1], q[2]}, {}, {q[0]})}; }},
            {"crx", 1, 2, [](const Params& p, const QubitList& q) { return Seq{gc(Prim::RX, {q[0]}, q[1], {p[0]})}; }},
            {"cry", 1, 2, [](const Params& p, const QubitList& q) { return Seq{gc(Prim::RY, {q[0]}, q[1], {p[0]})}; }},
            {"crz", 1, 2, [](const Params& p, const QubitList& q) { return Seq{gc(Prim::RZ, {q[0]}, q[1], {p[0]})}; }},
            {"cu1", 1, 2, [](const Params& p, const QubitList& q) { return Seq{gc(Prim::Phase, {q[0]}, q[1], {p[0]})}; }},
            {"cp", 1, 2, [](const Params& p, const QubitList& q) { return Seq{gc(Prim::Phase, {q[0]}, q[1], {p[0]})}; }},
            {"cu3", 3, 2, [](const Params& p, const QubitList& q) { return Seq{gc(Prim::U3, {q[0]}, q[1], {p[0], p[1], p[2]})}; }},
            {"csx", 0, 2, [](const Params&, const QubitList& q) { return Seq{gc(Prim::SX, {q[0]}, q[1])}; }},
            {"cu", 4, 2,
             [](const Params& p, const QubitList& q) { return Seq{gc(Prim::U3, {q[0]}, q[1], {p[0], p[1], p[2]}), gphase(p[3], {q[0]})}; }},
            {"rxx", 1, 2,
             [](const Params& p, const QubitList& q)
             {
                 return Seq{g1(Prim::H, q[0]), g1(Prim::H, q[1]), gc(Prim::X, {q[0]}, q[1]), g1(Prim::RZ, q[1], {p[0]}),
                            gc(Prim::X, {q[0]}, q[1]), g1(Prim::H, q[0]), g1(Prim::H, q[1])};
             }},
            {"rzz", 1, 2,
             [](const Params& p, const QubitList& q)
             { return Seq{gc(Prim::X, {q[0]}, q[1]), g1(Prim::RZ, q[1], {p[0]}), gc(Prim::X, {q[0]}, q[1])}; }},
            {"c3x", 0, 4, [](const Params&, const QubitList& q) { return Seq{gc(Prim::X, {q[0], q[1], q[2]}, q[3])}; }},
            {"c3sqrtx", 0, 4, [](const Params&, const QubitList& q) { return Seq{gc(Prim::SX, {q[0], q[1], q[2]}, q[3])}; }},
            {"c4x", 0, 5, [](const Params&, const QubitList& q) { return Seq{gc(Prim::X, {q[0], q[1], q[2], q[3]}, q[4])}; }},
        };
        return v;
    }

    const NativeDef* findNative(std::string_view include, std::string_view name)
    {
        const auto& table = include == "stdgates.inc" ? stdgatesNatives() : qelib1Natives();
        for (const NativeDef& n : table)
            if (n.name == name) return &n;
        return nullptr;
    }

    std::string_view embeddedInclude(std::string_view name)
    {
        for (const EmbeddedFile& f : embeddedQasmIncludes())
            if (f.path == name) return f.text;
        return {};
    }


    // ---- Importer: runs statements as they are parsed ----

    struct GateDefn;

    struct BodyCall
    {
        std::vector<Modifier> mods;
        const GateDefn* callee = nullptr; // resolved when the enclosing gate is defined
        std::vector<Expr> params;
        std::vector<std::size_t> qubits; // indices into the enclosing gate's qubit arguments
        Span span;
    };

    struct GateDefn
    {
        enum class Builtin : std::uint8_t { None, U2, U3, CX, GPhase };
        std::string name;
        std::vector<std::string_view> params;
        std::size_t qubits = 0;
        std::vector<BodyCall> body;
        const NativeDef* native = nullptr;
        Builtin builtin = Builtin::None;
        Span span;
    };

    struct ModValue
    {
        Modifier::K k;
        double value = 1.0; // controls for ctrl/negctrl, exponent for pow
    };

    struct StringHash
    {
        using is_transparent = void;
        std::size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
    };
    template <class V>
    using NameMap = std::unordered_map<std::string, V, StringHash, std::equal_to<>>;

    struct QReg
    {
        QubitList qubits;
        Span span;
        bool alias = false;
        bool scalar = false;
        bool mainFile = false;
    };

    struct CReg
    {
        std::vector<std::size_t> bits;
        Span span;
        bool scalar = false;
    };

    // Classical feed-forward accumulated over nested ifs; `never` when it cannot hold.
    struct Ctx
    {
        std::optional<Condition> cond;
        bool never = false;
        bool inIf = false;
        std::size_t depth = 0;
    };

    class Importer
    {
        public:
        Importer(SourceManager& sm, Diagnostics& d, QasmOptions o) : sources(sm), diags(d), opts(std::move(o)), builder(circuit) {}

        ImportedCircuit run(std::uint32_t fileId)
        {
            mainFile = fileId;
            Parser p(sources.file(fileId).text(), fileId);
            int v = opts.version;
            if (const auto h = p.header())
            {
                if (opts.version != 0 && h->first != opts.version)
                    throw Fail{"E9001", h->second, std::format("the file declares OpenQASM {}, but --format asks for {}", h->first, opts.version)};
                v = h->first;
            }
            if (v == 0) v = 3;
            setVersion(v);
            p.setVersion(v);
            program(p, fileId, 0);
            return finish();
        }

        void setVersion(int v)
        {
            version = v;
            if (v == 2)
            {
                defineBuiltin("U", 3, 1, GateDefn::Builtin::U2);
                defineBuiltin("CX", 0, 2, GateDefn::Builtin::CX);
            }
            else
            {
                defineBuiltin("U", 3, 1, GateDefn::Builtin::U3);
                defineBuiltin("gphase", 1, 0, GateDefn::Builtin::GPhase);
            }
        }

        void includeEmbedded(std::string_view name, std::size_t depth)
        {
            if (!included.insert(std::string(name)).second) return;
            const std::uint32_t id = sources.add(std::string(name), std::string(embeddedInclude(name)));
            Parser p(sources.file(id).text(), id);
            p.setVersion(version);
            nativeTable = name;
            program(p, id, depth + 1);
            nativeTable.clear();
        }

        const GateDefn* gate(std::string_view name) const
        {
            const auto it = gates.find(name);
            return it == gates.end() ? nullptr : it->second;
        }

        Seq expand(const GateDefn& g, const Params& params, const QubitList& qubits, bool natives)
        {
            useNatives = natives;
            return instantiate(g, params, qubits, 0);
        }

        std::vector<std::string> definedIn(std::string_view include) const
        {
            std::vector<std::string> out;
            for (const auto& [file, name] : definitionOrder)
                if (file == include) out.push_back(name);
            return out;
        }

        private:
        // ---- Program structure ----

        void program(Parser& p, std::uint32_t fileId, std::size_t includeDepth)
        {
            while (!p.atEnd())
            {
                Stmt s = p.statement(0);
                if (s.k == Stmt::K::Include)
                {
                    include(s, fileId, includeDepth);
                    continue;
                }
                exec(s, Ctx{}, fileId);
            }
        }

        void include(const Stmt& s, std::uint32_t fileId, std::size_t depth)
        {
            if (depth >= kMaxIncludeDepth) throw Fail{"E9001", s.span, "includes nest too deeply"};
            if (!embeddedInclude(s.path).empty())
            {
                includeEmbedded(s.path, depth);
                return;
            }
            const std::filesystem::path base = std::filesystem::path(sources.file(fileId).path()).parent_path();
            const std::string path = (base / s.path).string();
            if (!included.insert(path).second) return;
            bool ok = false;
            std::string text = readFileText(path, ok);
            if (!ok) throw Fail{"E9001", s.span, std::format("cannot read included file {}", path)};
            const std::uint32_t id = sources.add(path, std::move(text));
            Parser p(sources.file(id).text(), id);
            p.setVersion(version);
            if (p.header()) throw Fail{"E9001", s.span, "an included file must not start with an OPENQASM header"};
            program(p, id, depth + 1);
        }

        void defineBuiltin(std::string name, std::size_t params, std::size_t qubits, GateDefn::Builtin b)
        {
            auto g = std::make_unique<GateDefn>();
            g->name = name;
            static const std::string_view names[] = {"θ", "φ", "λ"};
            for (std::size_t k = 0; k < params; ++k) g->params.push_back(names[k]);
            g->qubits = qubits;
            g->builtin = b;
            gates[name] = g.get();
            store.push_back(std::move(g));
        }

        // ---- Statements ----

        void exec(const Stmt& s, const Ctx& ctx, std::uint32_t fileId)
        {
            if (++steps > kMaxOps) throw Fail{"E9004", s.span, std::format("the program runs more than {} statements", kMaxOps)};
            try
            {
                execInner(s, ctx, fileId);
            }
            catch (const ImportError& e)
            {
                throw Fail{e.code(), s.span, e.what()};
            }
            if (builder.size() > kMaxOps) throw Fail{"E9004", s.span, std::format("the inlined circuit exceeds {} operations", kMaxOps)};
        }

        void execInner(const Stmt& s, const Ctx& ctx, std::uint32_t fileId)
        {
            const bool top = ctx.depth == 0;
            auto requireTop = [&](std::string_view what)
            {
                if (!top) throw Fail{"E9001", s.span, std::format("{} must appear at the top level", what)};
            };
            switch (s.k)
            {
                case Stmt::K::Empty: return;
                case Stmt::K::Include: requireTop("include"); include(s, fileId, 0); return;
                case Stmt::K::Ignored: ignored(s.span, s.name); return;
                case Stmt::K::Barrier: return; // only constrains compilers
                case Stmt::K::Block:
                {
                    Ctx inner = ctx;
                    ++inner.depth;
                    for (const Stmt& x : s.body) exec(x, inner, fileId);
                    return;
                }
                case Stmt::K::QDecl: requireTop("a qubit declaration"); declareQubits(s, fileId); return;
                case Stmt::K::CDecl:
                    requireTop("a bit declaration");
                    declareBits(s);
                    if (!s.qargs.empty()) measure(s.qargs[0], QArg{s.name, s.nameSpan, false, 0, {}}, ctx);
                    return;
                case Stmt::K::Const: requireTop("const"); define(s.name, s.nameSpan, eval(*s.value, nullptr)); return;
                case Stmt::K::Input:
                {
                    requireTop("input");
                    const auto it = opts.params.find(std::string(s.name));
                    if (it == opts.params.end())
                        throw Fail{"E9005", s.nameSpan, std::format("input `{}` has no value; pass --param {}=<value>", s.name, s.name)};
                    define(s.name, s.nameSpan, it->second);
                    return;
                }
                case Stmt::K::Let:
                {
                    requireTop("let");
                    QReg alias;
                    alias.alias = true;
                    alias.span = s.nameSpan;
                    for (const QArg& a : s.qargs)
                        for (const Qubit q : resolveQubits(a)) alias.qubits.push_back(q);
                    defineName(s.name, s.nameSpan);
                    qregs[std::string(s.name)] = std::move(alias);
                    return;
                }
                case Stmt::K::GateDef: requireTop("a gate definition"); defineGate(s); return;
                case Stmt::K::GateCall: gateCall(s, ctx); return;
                case Stmt::K::Measure:
                    measure(s.qargs[0], s.cargs.empty() ? std::optional<QArg>{} : std::optional<QArg>{s.cargs[0]}, ctx, s.span);
                    return;
                case Stmt::K::Reset:
                    if (ctx.never) return;
                    for (const QArg& a : s.qargs)
                        for (const Qubit q : resolveQubits(a)) builder.reset(q, ctx.cond);
                    return;
                case Stmt::K::If: ifStmt(s, ctx, fileId); return;
                case Stmt::K::For: forStmt(s, ctx, fileId); return;
            }
        }

        void ignored(const Span& at, std::string_view what)
        {
            if (what.empty() || !warned.insert(std::string(what)).second) return;
            diags.warning("W9002", at, std::format("`{}` has no effect on an ideal simulation and is ignored (further occurrences are not reported)", what));
        }

        void defineName(std::string_view name, const Span& at)
        {
            if (qregs.contains(name) || cregs.contains(name) || consts.contains(name))
                throw Fail{"E9001", at, std::format("`{}` is already declared", name)};
        }

        void define(std::string_view name, const Span& at, double v)
        {
            defineName(name, at);
            consts[std::string(name)] = v;
        }

        std::size_t sizeOf(const Stmt& s)
        {
            if (!s.value) return 1;
            const std::size_t n = integer(eval(*s.value, nullptr), s.value->span, "a register size");
            if (n == 0) throw Fail{"E9006", s.value->span, "a register needs at least one element"};
            return n;
        }

        void declareQubits(const Stmt& s, std::uint32_t fileId)
        {
            if (physical) throw Fail{"E9001", s.span, "declared qubits cannot be mixed with physical qubits $n"};
            const std::size_t n = sizeOf(s);
            if (circuit.numQubits + n > Qputer::kMaxStabilizerQubits)
                throw Fail{"E9004", s.span, std::format("{} qubits; the simulator holds at most {}", circuit.numQubits + n, Qputer::kMaxStabilizerQubits)};
            defineName(s.name, s.nameSpan);
            QReg r;
            r.span = s.span;
            r.scalar = s.scalar;
            r.mainFile = fileId == mainFile;
            for (std::size_t k = 0; k < n; ++k)
            {
                r.qubits.push_back(circuit.numQubits + k);
                circuit.qubitLabels.push_back(s.scalar ? std::string(s.name) : std::format("{}[{}]", s.name, k));
            }
            circuit.numQubits += n;
            qregs[std::string(s.name)] = std::move(r);
            declared.push_back(std::string(s.name));
        }

        void declareBits(const Stmt& s)
        {
            const std::size_t n = sizeOf(s);
            if (circuit.numClbits + n > Qputer::QuantumStateMachine::kMaxClbits)
                throw Fail{"E9004", s.span, std::format("{} classical bits; the classical register holds at most {}", circuit.numClbits + n,
                                                        Qputer::QuantumStateMachine::kMaxClbits)};
            defineName(s.name, s.nameSpan);
            CReg r;
            r.span = s.span;
            r.scalar = s.scalar;
            for (std::size_t k = 0; k < n; ++k) r.bits.push_back(circuit.numClbits + k);
            circuit.clbitRegisters.push_back({std::string(s.name), circuit.numClbits, n});
            circuit.numClbits += n;
            cregs[std::string(s.name)] = std::move(r);
        }

        void defineGate(const Stmt& s)
        {
            auto g = std::make_unique<GateDefn>();
            g->name = std::string(s.name);
            g->params = s.paramNames;
            g->qubits = s.qubitNames.size();
            g->span = s.nameSpan;
            for (std::size_t a = 0; a < s.qubitNames.size(); ++a)
                for (std::size_t b = a + 1; b < s.qubitNames.size(); ++b)
                    if (s.qubitNames[a] == s.qubitNames[b]) throw Fail{"E9006", s.nameSpan, std::format("gate `{}` names qubit `{}` twice", s.name, s.qubitNames[a])};
            for (const Stmt& x : s.body)
            {
                if (x.k == Stmt::K::Barrier || x.k == Stmt::K::Empty) continue;
                if (x.k != Stmt::K::GateCall) throw Fail{"E9001", x.span, "a gate body may only apply gates"};
                BodyCall bc;
                bc.span = x.span;
                bc.mods = x.mods;
                bc.params = x.params;
                bc.callee = gate(x.name);
                if (!bc.callee) throw Fail{"E9002", x.nameSpan, std::format("unknown gate `{}` in the body of `{}`", x.name, s.name)};
                for (const QArg& a : x.qargs)
                {
                    if (a.physical || a.sel.k != Sel::K::All) throw Fail{"E9001", a.span, "gate bodies address their qubit arguments by name only"};
                    const auto it = std::ranges::find(s.qubitNames, a.name);
                    if (it == s.qubitNames.end()) throw Fail{"E9006", a.span, std::format("`{}` is not a qubit argument of gate `{}`", a.name, s.name)};
                    bc.qubits.push_back(static_cast<std::size_t>(it - s.qubitNames.begin()));
                }
                g->body.push_back(std::move(bc));
            }
            if (!nativeTable.empty())
            {
                g->native = findNative(nativeTable, g->name);
                definitionOrder.emplace_back(nativeTable, g->name);
                if (g->native && (g->native->params != g->params.size() || g->native->qubits != g->qubits))
                    throw std::logic_error(std::format("native `{}` disagrees with its definition in {}", g->name, nativeTable));
            }
            gates[g->name] = g.get();
            store.push_back(std::move(g));
        }

        // ---- Gate application ----

        std::vector<ModValue> modValues(const std::vector<Modifier>& mods, const std::vector<std::pair<std::string_view, double>>* scope)
        {
            std::vector<ModValue> out;
            for (const Modifier& m : mods)
            {
                ModValue v{m.k, 1.0};
                if (m.arg)
                {
                    v.value = eval(*m.arg, scope);
                    if (m.k != Modifier::K::Pow) v.value = static_cast<double>(integer(v.value, m.arg->span, "a control count"));
                    if (m.k != Modifier::K::Pow && v.value < 1) throw Fail{"E9006", m.arg->span, "a control count must be at least 1"};
                }
                out.push_back(v);
            }
            return out;
        }

        static std::size_t controlCount(const std::vector<ModValue>& mods)
        {
            std::size_t n = 0;
            for (const ModValue& m : mods)
                if (m.k == Modifier::K::Ctrl || m.k == Modifier::K::NegCtrl) n += static_cast<std::size_t>(m.value);
            return n;
        }

        // Modifiers act right to left; ctrl(n) modifiers take the leading qubits left to right.
        Seq call(const GateDefn& g, const std::vector<ModValue>& mods, const Params& params, const QubitList& qubits, std::size_t depth)
        {
            const std::size_t nc = controlCount(mods);
            if (qubits.size() != nc + g.qubits || params.size() != g.params.size())
                throw ImportError("E9006", std::format("gate `{}` takes {} parameter{} and {} qubit{}{}; got {} and {}", g.name, g.params.size(),
                                                       g.params.size() == 1 ? "" : "s", g.qubits, g.qubits == 1 ? "" : "s",
                                                       nc ? std::format(" plus {} control{}", nc, nc == 1 ? "" : "s") : "", params.size(), qubits.size()));
            Seq seq = instantiate(g, params, QubitList(qubits.begin() + static_cast<std::ptrdiff_t>(nc), qubits.end()), depth + 1);
            std::vector<std::size_t> offset(mods.size(), 0);
            for (std::size_t k = 0, at = 0; k < mods.size(); ++k)
            {
                offset[k] = at;
                if (mods[k].k == Modifier::K::Ctrl || mods[k].k == Modifier::K::NegCtrl) at += static_cast<std::size_t>(mods[k].value);
            }
            for (std::size_t k = mods.size(); k-- > 0;)
            {
                const ModValue& m = mods[k];
                switch (m.k)
                {
                    case Modifier::K::Inv: seq = inverse(seq); break;
                    case Modifier::K::Pow: seq = power(seq, m.value); break;
                    case Modifier::K::Ctrl: case Modifier::K::NegCtrl:
                    {
                        const auto n = static_cast<std::size_t>(m.value);
                        const QubitList controls(qubits.begin() + static_cast<std::ptrdiff_t>(offset[k]),
                                                 qubits.begin() + static_cast<std::ptrdiff_t>(offset[k] + n));
                        seq = controlled(std::move(seq), controls, std::vector<bool>(n, m.k == Modifier::K::Ctrl));
                        break;
                    }
                }
                if (seq.size() > kMaxOps) throw ImportError("E9004", std::format("gate `{}` expands to more than {} operations", g.name, kMaxOps));
            }
            return seq;
        }

        Seq instantiate(const GateDefn& g, const Params& p, const QubitList& q, std::size_t depth)
        {
            if (depth > kMaxNesting) throw ImportError("E9001", std::format("gate `{}` nests too deeply", g.name));
            if (useNatives && g.native) return g.native->make(p, q);
            switch (g.builtin)
            {
                case GateDefn::Builtin::U2: return {g1(Prim::U3, q[0], {p[0], p[1], p[2]})};
                // OpenQASM 3: U(θ, φ, λ) = e^{iθ/2}·u3(θ, φ, λ), which is 2π-periodic in θ.
                case GateDefn::Builtin::U3: return {g1(Prim::U3, q[0], {p[0], p[1], p[2]}), gphase(p[0] / 2)};
                case GateDefn::Builtin::CX: return {gc(Prim::X, {q[0]}, q[1])};
                case GateDefn::Builtin::GPhase: return {gphase(p[0])};
                case GateDefn::Builtin::None: break;
            }
            std::vector<std::pair<std::string_view, double>> scope;
            for (std::size_t k = 0; k < g.params.size(); ++k) scope.emplace_back(g.params[k], p[k]);
            Seq out;
            for (const BodyCall& bc : g.body)
            {
                Params values;
                for (const Expr& e : bc.params) values.push_back(eval(e, &scope));
                QubitList qs;
                for (const std::size_t i : bc.qubits) qs.push_back(q[i]);
                Seq part = call(*bc.callee, modValues(bc.mods, &scope), values, qs, depth);
                out.insert(out.end(), std::make_move_iterator(part.begin()), std::make_move_iterator(part.end()));
                if (out.size() > kMaxOps) throw ImportError("E9004", std::format("gate `{}` expands to more than {} operations", g.name, kMaxOps));
            }
            return out;
        }

        void gateCall(const Stmt& s, const Ctx& ctx)
        {
            const GateDefn* g = gate(s.name);
            if (!g)
            {
                Diagnostic& d = diags.error("E9002", s.nameSpan, std::format("unknown gate `{}`", s.name));
                if (findNative(version == 2 ? "qelib1.inc" : "stdgates.inc", s.name) && !included.contains(version == 2 ? "qelib1.inc" : "stdgates.inc"))
                    d.note(s.nameSpan, std::format("`{}` is defined in {}; add `include \"{}\";`", s.name, version == 2 ? "qelib1.inc" : "stdgates.inc",
                                                   version == 2 ? "qelib1.inc" : "stdgates.inc"));
                throw Fail{"", s.nameSpan, ""};
            }
            if (ctx.never) return;
            const std::vector<ModValue> mods = modValues(s.mods, nullptr);
            Params params;
            for (const Expr& e : s.params) params.push_back(eval(e, nullptr));
            if (g->builtin == GateDefn::Builtin::GPhase && controlCount(mods) == 0 && !warned.contains("gphase"))
            {
                warned.insert("gphase");
                diags.warning("W9001", s.span, "an uncontrolled global phase is unobservable and is dropped (further occurrences are not reported)");
            }

            std::vector<QubitList> lists;
            for (const QArg& a : s.qargs) lists.push_back(resolveQubits(a));
            std::size_t n = 1;
            for (std::size_t k = 0; k < lists.size(); ++k)
                if (lists[k].size() != 1)
                {
                    if (n != 1 && lists[k].size() != n)
                        throw Fail{"E9006", s.qargs[k].span, std::format("register operands of {} and {} qubits in one statement", n, lists[k].size())};
                    n = lists[k].size();
                }
            for (std::size_t i = 0; i < n; ++i)
            {
                QubitList qs;
                for (const QubitList& l : lists) qs.push_back(l.size() == 1 ? l[0] : l[i]);
                builder.apply(call(*g, mods, params, qs, 0), ctx.cond);
            }
        }

        void measure(const QArg& q, const std::optional<QArg>& c, const Ctx& ctx, Span at = {})
        {
            if (ctx.inIf)
                throw Fail{"E9003", at.end > at.begin ? at : q.span,
                           "a measurement inside `if` is a conditional measurement, which the state machine does not support"};
            const QubitList qs = resolveQubits(q);
            if (!c)
            {
                for (const Qubit x : qs) builder.measure(x, std::nullopt);
                return;
            }
            const std::vector<std::size_t> cs = resolveBits(*c);
            if (cs.size() != qs.size())
                throw Fail{"E9006", c->span, std::format("measuring {} qubit{} into {} bit{}", qs.size(), qs.size() == 1 ? "" : "s", cs.size(), cs.size() == 1 ? "" : "s")};
            for (std::size_t k = 0; k < qs.size(); ++k) builder.measure(qs[k], cs[k]);
        }

        // ---- Control flow ----

        struct CondValue
        {
            Condition c;
            bool never = false;
        };

        static std::optional<Condition> merge(const std::optional<Condition>& a, const Condition& b, bool& never)
        {
            if (!a) return b;
            const Outcome shared = a->mask & b.mask;
            if ((a->value & shared) != (b.value & shared)) never = true;
            return Condition{a->mask | b.mask, a->value | b.value};
        }

        CondValue condition(const Expr& e)
        {
            auto bitOf = [&](const Expr& x) -> std::optional<std::size_t>
            {
                if (x.k == Expr::K::Index)
                {
                    const CReg* r = creg(x.text, x.span);
                    const std::int64_t i = signedIndex(eval(x.kids[0], nullptr), x.kids[0].span, r->bits.size());
                    return r->bits[static_cast<std::size_t>(i)];
                }
                if (x.k == Expr::K::Name && cregs.contains(x.text) && cregs.find(x.text)->second.bits.size() == 1)
                    return cregs.find(x.text)->second.bits[0];
                return std::nullopt;
            };
            auto single = [](std::size_t bit, bool value) { return CondValue{Condition{Outcome{1} << bit, value ? Outcome{1} << bit : 0}, false}; };
            auto constant = [&](const Expr& x) -> std::optional<double>
            {
                if (x.k == Expr::K::Name && (x.text == "true" || x.text == "false")) return x.text == "true" ? 1.0 : 0.0;
                if (x.k == Expr::K::Num || x.k == Expr::K::Unary || x.k == Expr::K::Binary || (x.k == Expr::K::Name && consts.contains(x.text)))
                    return eval(x, nullptr);
                return std::nullopt;
            };

            if (const auto b = bitOf(e)) return single(*b, true);
            if (e.k == Expr::K::Unary && e.text == "!")
                if (const auto b = bitOf(e.kids[0])) return single(*b, false);
            if (e.k == Expr::K::Name && cregs.contains(e.text))
                throw Fail{"E9003", e.span, std::format("`if ({})` on a {}-bit register means register ≠ 0, which is not a single {{mask, value}} condition",
                                                        e.text, cregs.find(e.text)->second.bits.size())};
            if (e.k == Expr::K::Binary && e.text == "&&")
            {
                CondValue a = condition(e.kids[0]), b = condition(e.kids[1]);
                bool never = a.never || b.never;
                const auto m = merge(a.c, b.c, never);
                if (never && !a.never && !b.never) warnNever(e.span);
                return CondValue{*m, never};
            }
            if (e.k == Expr::K::Binary && (e.text == "==" || e.text == "!="))
            {
                const bool flip = e.kids[1].k == Expr::K::Index || (e.kids[1].k == Expr::K::Name && cregs.contains(e.kids[1].text));
                const Expr& lhs = flip ? e.kids[1] : e.kids[0];
                const Expr& rhs = flip ? e.kids[0] : e.kids[1];
                const auto value = constant(rhs);
                if (!value) throw Fail{"E9003", rhs.span, "a condition compares a classical bit or register with a constant"};
                const bool equal = e.text == "==";
                if (const auto b = bitOf(lhs))
                {
                    if (*value != 0.0 && *value != 1.0)
                    {
                        warnNever(e.span);
                        return CondValue{Condition{Outcome{1} << *b, 0}, true};
                    }
                    return single(*b, (*value == 1.0) == equal);
                }
                if (lhs.k == Expr::K::Name && cregs.contains(lhs.text))
                {
                    if (!equal) throw Fail{"E9003", e.span, "a register ≠ value condition is not a single {mask, value} condition"};
                    const CReg& r = cregs.find(lhs.text)->second;
                    const std::size_t k = r.bits.size();
                    const auto n = static_cast<std::uint64_t>(integer(*value, rhs.span, "a register value"));
                    Condition c;
                    for (std::size_t j = 0; j < k; ++j)
                    {
                        c.mask |= Outcome{1} << r.bits[j];
                        if (j < 64 && ((n >> j) & 1U)) c.value |= Outcome{1} << r.bits[j];
                    }
                    if (k < 64 && (n >> k) != 0)
                    {
                        warnNever(e.span);
                        return CondValue{c, true};
                    }
                    return CondValue{c, false};
                }
            }
            throw Fail{"E9003", e.span, "unsupported condition; use a bit, !bit, bit == v, register == n, or && of those"};
        }

        void warnNever(const Span& at)
        {
            // Loops re-run their conditions; report each source condition once.
            if (!neverWarned.insert({at.file, at.begin}).second) return;
            diags.warning("W9002", at, "the condition can never hold, so its statement is skipped");
        }

        void ifStmt(const Stmt& s, const Ctx& ctx, std::uint32_t fileId)
        {
            const CondValue cv = condition(*s.value);
            Ctx then = ctx;
            then.inIf = true;
            ++then.depth;
            then.never = ctx.never || cv.never;
            then.cond = merge(ctx.cond, cv.c, then.never);
            if (then.never && !ctx.never && !cv.never) warnNever(s.value->span);
            for (const Stmt& x : s.body) exec(x, then, fileId);
            if (!s.hasElse) return;
            if (std::popcount(cv.c.mask) != 1)
                throw Fail{"E9003", s.value->span, "an `else` branch needs a single-bit condition (its negation is not a {mask, value} condition)"};
            Ctx orelse = ctx;
            orelse.inIf = true;
            ++orelse.depth;
            orelse.never = ctx.never;
            orelse.cond = merge(ctx.cond, Condition{cv.c.mask, cv.c.value ^ cv.c.mask}, orelse.never);
            if (orelse.never && !ctx.never && !s.orelse.empty()) warnNever(s.value->span);
            for (const Stmt& x : s.orelse) exec(x, orelse, fileId);
        }

        void forStmt(const Stmt& s, const Ctx& ctx, std::uint32_t fileId)
        {
            std::vector<double> values;
            if (s.range.k == Sel::K::Set)
                for (const Expr& e : s.range.items) values.push_back(eval(e, nullptr));
            else
            {
                if (!s.range.start || !s.range.end) throw Fail{"E9001", s.nameSpan, "a for-loop range needs a start and an end"};
                const double a = eval(*s.range.start, nullptr), b = eval(*s.range.end, nullptr);
                const double step = s.range.step ? eval(*s.range.step, nullptr) : 1.0;
                if (step == 0.0) throw Fail{"E9001", s.range.step->span, "a range step must not be zero"};
                const double count = std::floor((b - a) / step) + 1;
                if (count > static_cast<double>(kMaxOps)) throw Fail{"E9004", s.nameSpan, std::format("the loop runs {} times", count)};
                for (double v = a; step > 0 ? v <= b + 1e-9 : v >= b - 1e-9; v += step) values.push_back(v);
            }
            Ctx inner = ctx;
            ++inner.depth;
            for (const double v : values)
            {
                if (++steps > kMaxOps) throw Fail{"E9004", s.span, std::format("the program runs more than {} statements", kMaxOps)};
                locals.emplace_back(s.name, v);
                for (const Stmt& x : s.body) exec(x, inner, fileId);
                locals.pop_back();
            }
        }

        // ---- Operands ----

        std::size_t integer(double v, const Span& at, std::string_view what) const
        {
            if (!std::isfinite(v) || v != std::floor(v) || v < 0 || v > 1e15)
                throw Fail{"E9006", at, std::format("{} must be a non-negative integer, got {}", what, jsonNumber(v))};
            return static_cast<std::size_t>(v);
        }

        std::int64_t signedIndex(double v, const Span& at, std::size_t size) const
        {
            if (!std::isfinite(v) || v != std::floor(v)) throw Fail{"E9006", at, std::format("index {} is not an integer", jsonNumber(v))};
            auto i = static_cast<std::int64_t>(v);
            if (i < 0) i += static_cast<std::int64_t>(size);
            if (i < 0 || i >= static_cast<std::int64_t>(size))
                throw Fail{"E9006", at, std::format("index {} out of range for a register of {}", jsonNumber(v), size)};
            return i;
        }

        std::vector<std::size_t> select(const Sel& sel, std::size_t size, const Span& at)
        {
            std::vector<std::size_t> out;
            switch (sel.k)
            {
                case Sel::K::All:
                    for (std::size_t k = 0; k < size; ++k) out.push_back(k);
                    break;
                case Sel::K::One: out.push_back(static_cast<std::size_t>(signedIndex(eval(sel.items[0], nullptr), sel.items[0].span, size))); break;
                case Sel::K::Set:
                    for (const Expr& e : sel.items) out.push_back(static_cast<std::size_t>(signedIndex(eval(e, nullptr), e.span, size)));
                    break;
                case Sel::K::Range:
                {
                    const std::int64_t a = sel.start ? signedIndex(eval(*sel.start, nullptr), sel.start->span, size) : 0;
                    const std::int64_t b = sel.end ? signedIndex(eval(*sel.end, nullptr), sel.end->span, size) : static_cast<std::int64_t>(size) - 1;
                    std::int64_t step = 1;
                    if (sel.step)
                    {
                        const double s = eval(*sel.step, nullptr);
                        if (s == 0.0 || s != std::floor(s)) throw Fail{"E9006", sel.step->span, "a range step must be a nonzero integer"};
                        step = static_cast<std::int64_t>(s);
                    }
                    for (std::int64_t k = a; step > 0 ? k <= b : k >= b; k += step) out.push_back(static_cast<std::size_t>(k));
                    if (out.empty()) throw Fail{"E9006", at, "the range selects nothing"};
                    break;
                }
            }
            return out;
        }

        QubitList resolveQubits(const QArg& a)
        {
            if (a.physical)
            {
                if (!qregs.empty() && !physical) throw Fail{"E9001", a.span, "physical qubits $n cannot be mixed with declared qubits"};
                if (a.phys >= Qputer::kMaxStabilizerQubits) throw Fail{"E9004", a.span, std::format("physical qubit {} beyond the simulator's {}", a.phys, Qputer::kMaxStabilizerQubits)};
                physical = true;
                while (circuit.numQubits <= a.phys) circuit.qubitLabels.push_back(std::format("${}", circuit.numQubits++));
                return {a.phys};
            }
            const auto it = qregs.find(a.name);
            if (it == qregs.end())
            {
                if (cregs.contains(a.name)) throw Fail{"E9006", a.span, std::format("`{}` is a classical register, not qubits", a.name)};
                throw Fail{"E9006", a.span, std::format("unknown quantum register `{}`", a.name)};
            }
            QubitList out;
            for (const std::size_t k : select(a.sel, it->second.qubits.size(), a.span)) out.push_back(it->second.qubits[k]);
            return out;
        }

        const CReg* creg(std::string_view name, const Span& at) const
        {
            const auto it = cregs.find(name);
            if (it == cregs.end()) throw Fail{"E9006", at, std::format("unknown classical register `{}`", name)};
            return &it->second;
        }

        std::vector<std::size_t> resolveBits(const QArg& a)
        {
            if (a.physical) throw Fail{"E9006", a.span, "expected classical bits"};
            const CReg* r = creg(a.name, a.span);
            std::vector<std::size_t> out;
            for (const std::size_t k : select(a.sel, r->bits.size(), a.span)) out.push_back(r->bits[k]);
            return out;
        }

        // ---- Constant expressions ----

        double eval(const Expr& e, const std::vector<std::pair<std::string_view, double>>* scope)
        {
            switch (e.k)
            {
                case Expr::K::Num: return e.v;
                case Expr::K::Name:
                {
                    if (scope)
                        for (const auto& [n, v] : *scope)
                            if (n == e.text) return v;
                    if (!scope)
                        for (auto it = locals.rbegin(); it != locals.rend(); ++it)
                            if (it->first == e.text) return it->second;
                    if (const auto it = consts.find(e.text); it != consts.end()) return it->second;
                    if (e.text == "pi" || e.text == "π") return kPi;
                    if (e.text == "tau" || e.text == "τ") return 2 * kPi;
                    if (e.text == "euler" || e.text == "ℇ") return std::numbers::e;
                    if (e.text == "true") return 1.0;
                    if (e.text == "false") return 0.0;
                    if (cregs.contains(e.text) || qregs.contains(e.text))
                        throw Fail{"E9001", e.span, std::format("`{}` is not a compile-time constant", e.text)};
                    throw Fail{"E9001", e.span, std::format("unknown identifier `{}`", e.text)};
                }
                case Expr::K::Index: throw Fail{"E9001", e.span, std::format("`{}[…]` is not a compile-time constant", e.text)};
                case Expr::K::Unary:
                {
                    const double x = eval(e.kids[0], scope);
                    if (e.text == "-") return -x;
                    if (e.text == "!") return x == 0.0 ? 1.0 : 0.0;
                    return static_cast<double>(~static_cast<std::int64_t>(x));
                }
                case Expr::K::Binary: return binary(e, eval(e.kids[0], scope), eval(e.kids[1], scope));
                case Expr::K::Call:
                {
                    std::vector<double> a;
                    for (const Expr& k : e.kids) a.push_back(eval(k, scope));
                    return function(e, a);
                }
            }
            return 0.0;
        }

        static double finite(double v, const Expr& e)
        {
            if (!std::isfinite(v)) throw Fail{"E9001", e.span, "the expression is not finite"};
            return v;
        }

        static double binary(const Expr& e, double a, double b)
        {
            const std::string_view o = e.text;
            auto ints = [&] { return std::pair{static_cast<std::int64_t>(a), static_cast<std::int64_t>(b)}; };
            if (o == "+") return finite(a + b, e);
            if (o == "-") return finite(a - b, e);
            if (o == "*") return finite(a * b, e);
            if (o == "/") return finite(a / b, e);
            if (o == "%") return finite(std::fmod(a, b), e);
            if (o == "**") return finite(std::pow(a, b), e);
            if (o == "==") return a == b;
            if (o == "!=") return a != b;
            if (o == "<") return a < b;
            if (o == ">") return a > b;
            if (o == "<=") return a <= b;
            if (o == ">=") return a >= b;
            if (o == "&&") return a != 0 && b != 0;
            if (o == "||") return a != 0 || b != 0;
            const auto [x, y] = ints();
            if (o == "&") return static_cast<double>(x & y);
            if (o == "|") return static_cast<double>(x | y);
            if (o == "^") return static_cast<double>(x ^ y);
            if (o == "<<") return static_cast<double>(y >= 0 && y < 63 ? x << y : 0);
            if (o == ">>") return static_cast<double>(y >= 0 && y < 63 ? x >> y : 0);
            throw Fail{"E9001", e.span, std::format("unsupported operator `{}`", o)};
        }

        static double function(const Expr& e, const std::vector<double>& a)
        {
            const std::string_view f = e.text;
            auto want = [&](std::size_t n)
            {
                if (a.size() != n) throw Fail{"E9001", e.span, std::format("{} takes {} argument{}", f, n, n == 1 ? "" : "s")};
            };
            static const std::map<std::string_view, double (*)(double)> unary{
                {"sin", [](double x) { return std::sin(x); }},     {"cos", [](double x) { return std::cos(x); }},
                {"tan", [](double x) { return std::tan(x); }},     {"arcsin", [](double x) { return std::asin(x); }},
                {"arccos", [](double x) { return std::acos(x); }}, {"arctan", [](double x) { return std::atan(x); }},
                {"asin", [](double x) { return std::asin(x); }},   {"acos", [](double x) { return std::acos(x); }},
                {"atan", [](double x) { return std::atan(x); }},   {"exp", [](double x) { return std::exp(x); }},
                {"ln", [](double x) { return std::log(x); }},      {"log", [](double x) { return std::log(x); }},
                {"sqrt", [](double x) { return std::sqrt(x); }},   {"abs", [](double x) { return std::abs(x); }},
                {"floor", [](double x) { return std::floor(x); }}, {"ceiling", [](double x) { return std::ceil(x); }},
                {"ceil", [](double x) { return std::ceil(x); }},   {"float", [](double x) { return x; }},
                {"angle", [](double x) { return x; }},             {"int", [](double x) { return std::trunc(x); }},
                {"uint", [](double x) { return std::trunc(x); }},  {"bool", [](double x) { return x != 0.0 ? 1.0 : 0.0; }},
            };
            if (const auto it = unary.find(f); it != unary.end())
            {
                want(1);
                return finite(it->second(a[0]), e);
            }
            if (f == "pow")
            {
                want(2);
                return finite(std::pow(a[0], a[1]), e);
            }
            if (f == "mod")
            {
                want(2);
                return finite(std::fmod(a[0], a[1]), e);
            }
            throw Fail{"E9001", e.span, std::format("unknown function `{}`", f)};
        }

        // ---- Result ----

        ImportedCircuit finish()
        {
            if (circuit.numQubits == 0)
            {
                circuit.numQubits = 1;
                circuit.qubitLabels.push_back("(unused)");
            }
            circuit.format = version == 2 ? "qasm2" : "qasm3";
            circuit.source = Json::object();
            circuit.source["framework"] = "openqasm";
            circuit.source["version"] = version == 2 ? "2.0" : "3.0";

            std::vector<char> used(circuit.numQubits, 0);
            for (const Qputer::Operation& o : circuit.ops)
            {
                for (const Qubit q : o.controls) used[q] = 1;
                for (const Qubit q : o.targets) used[q] = 1;
            }
            for (const std::string& name : declared)
            {
                const QReg& r = qregs.find(name)->second;
                if (!r.mainFile) continue;
                std::vector<std::string> idle;
                for (std::size_t k = 0; k < r.qubits.size(); ++k)
                    if (!used[r.qubits[k]]) idle.push_back(r.scalar ? name : std::format("{}[{}]", name, k));
                if (idle.empty()) continue;
                std::string list;
                for (std::size_t k = 0; k < idle.size() && k < 8; ++k) list += (k ? ", " : "") + idle[k];
                if (idle.size() > 8) list += std::format(" and {} more", idle.size() - 8);
                diags.warning("W9003", r.span, std::format("{} declared but never used; {} stay{} in |0⟩", list, idle.size() == 1 ? "it" : "they",
                                                           idle.size() == 1 ? "s" : ""));
            }
            return std::move(circuit);
        }

        SourceManager& sources;
        Diagnostics& diags;
        QasmOptions opts;
        ImportedCircuit circuit;
        CircuitBuilder builder;
        std::uint32_t mainFile = 0;
        int version = 3;
        bool physical = false;
        bool useNatives = true;
        std::size_t steps = 0; // statements run, loop iterations included, so empty loops cannot spin
        std::set<std::pair<std::uint32_t, std::uint32_t>> neverWarned;
        std::string nativeTable; // include file whose definitions are being registered
        std::vector<std::unique_ptr<GateDefn>> store;
        NameMap<GateDefn*> gates;
        NameMap<QReg> qregs;
        NameMap<CReg> cregs;
        NameMap<double> consts;
        std::vector<std::pair<std::string_view, double>> locals; // loop variables, innermost last
        std::vector<std::string> declared;
        std::set<std::string, std::less<>> included, warned;
        std::vector<std::pair<std::string, std::string>> definitionOrder; // (include, gate)
    };

    // An importer holding only one embedded include, for the test hooks.
    struct IncludeLibrary
    {
        SourceManager sources;
        Diagnostics diags{sources};
        Importer importer{sources, diags, QasmOptions{}};

        explicit IncludeLibrary(std::string_view include)
        {
            if (embeddedInclude(include).empty()) throw std::invalid_argument(std::format("no embedded include {}", include));
            importer.setVersion(include == "qelib1.inc" ? 2 : 3);
            importer.includeEmbedded(include, 0);
        }
    };
} // namespace


std::optional<ImportedCircuit> importQasm(SourceManager& sources, std::uint32_t fileId, Diagnostics& diags, const QasmOptions& options)
{
    try
    {
        return Importer(sources, diags, options).run(fileId);
    }
    catch (const Fail& f)
    {
        if (!f.code.empty()) diags.error(f.code, f.span, f.message);
    }
    catch (const ImportError& e)
    {
        diags.error(e.code(), Span{fileId, 0, 0}, e.what());
    }
    return std::nullopt;
}

std::vector<std::string> qasmIncludeGates(std::string_view include)
{
    IncludeLibrary lib(include);
    return lib.importer.definedIn(include);
}

std::optional<QasmGateShape> qasmGateShape(std::string_view include, std::string_view gate)
{
    IncludeLibrary lib(include);
    const GateDefn* g = lib.importer.gate(gate);
    if (!g) return std::nullopt;
    return QasmGateShape{g->params.size(), g->qubits, g->native != nullptr};
}

Eigen::MatrixXcd qasmGateMatrix(std::string_view include, std::string_view gate, const std::vector<double>& params, bool native)
{
    IncludeLibrary lib(include);
    const GateDefn* g = lib.importer.gate(gate);
    if (!g) throw std::invalid_argument(std::format("{} defines no gate {}", include, gate));
    QubitList qubits, support;
    for (std::size_t k = 0; k < g->qubits; ++k) qubits.push_back(k);
    for (std::size_t k = g->qubits; k-- > 0;) support.push_back(k);
    return matrixOf(lib.importer.expand(*g, params, qubits, native), support);
}

} // namespace Noether::Interop
