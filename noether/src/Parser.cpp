#include "Parser.hpp"

#include <algorithm>
#include <charconv>
#include <format>
#include <map>
#include <set>
#include <utility>



namespace Noether
{

namespace
{

    struct ParseAbort
    {
    };

    const std::set<std::string_view> kSpecKeywords{
        "task", "candidate", "target", "metric", "gates", "coupling", "allow", "forbid", "readout", "optimize",
        "require", "minimize", "maximize", "goal", "instances", "holdout", "aggregate", "budget"};

    // Lower-case Qiskit gate methods and the Noether spelling of each, for E2009 fix-its.
    const std::map<std::string_view, std::string_view> kQiskitGates{
        {"h", "H"}, {"x", "X"}, {"y", "Y"}, {"z", "Z"}, {"s", "S"}, {"t", "T"}, {"sdg", "S†"}, {"tdg", "T†"},
        {"sx", "√X"}, {"rx", "Rx"}, {"ry", "Ry"}, {"rz", "Rz"}, {"p", "P"}, {"cx", "CNOT"}, {"cnot", "CNOT"},
        {"cz", "CZ"}, {"swap", "SWAP"}, {"ccx", "Toffoli"}, {"cswap", "Fredkin"}, {"cp", "CP"}, {"measure", "measure"},
        {"reset", "reset"}, {"u", "U3"}, {"u3", "U3"}};

    class Parser
    {
        public:
        Parser(const SourceFile& file, std::uint32_t fileId, Diagnostics& diagnostics)
            : f(file), fid(fileId), d(diagnostics)
        {
            LexResult lr = lex(file, fileId, diagnostics);
            toks = std::move(lr.tokens);
            comments = std::move(lr.comments);
        }

        Program run()
        {
            Program prog;
            prog.file = fid;
            header(prog);
            if (at(Tok::Ident) && peek().text == "task") specMode = true;
            prog.isSpec = specMode;
            prog.stmts = lines(false);
            attachComments(prog);
            return prog;
        }

        private:
        // ---- Token access ----
        const Token& peek(std::size_t k = 0) const { return toks[std::min(i + k, toks.size() - 1)]; }
        bool at(Tok k) const { return peek().kind == k; }
        const Token& advance()
        {
            const Token& t = toks[i];
            if (i + 1 < toks.size()) ++i;
            return t;
        }
        bool accept(Tok k)
        {
            if (!at(k)) return false;
            advance();
            return true;
        }
        Span prevSpan() const { return toks[i == 0 ? 0 : i - 1].span; }
        Span here() const { return peek().span; }

        [[noreturn]] void fail(std::string_view what)
        {
            const Token& t = peek();
            std::string got(tokName(t.kind));
            if (t.kind == Tok::Ident) got = std::format("identifier '{}'", t.text);
            d.error("E2001", t.span, std::format("expected {}, found {}", what, got));
            throw ParseAbort{};
        }

        const Token& expect(Tok k, std::string_view what)
        {
            if (!at(k)) fail(what);
            return advance();
        }

        // A postfix operator must touch its operand.
        bool touching(Tok k) const { return at(k) && !peek().spaceBefore; }

        std::shared_ptr<Expr> node(EK k, Span s) const
        {
            auto e = std::make_shared<Expr>();
            e->kind = k;
            e->span = s;
            return e;
        }

        // ---- Header ----
        void header(Program& prog)
        {
            while (at(Tok::Newline)) advance();
            if (!at(Tok::KwNoether))
            {
                d.error("E2005", here(), std::format("missing version header; the first line must be `noether {}`",
                                                     kLanguageVersion))
                    .fix("insert the header", Span{fid, 0, 0}, std::format("noether {}\n", kLanguageVersion));
                prog.version = std::string(kLanguageVersion);
                return;
            }
            const Token& kw = advance();
            prog.headerSpan = kw.span;
            if (at(Tok::Real) || at(Tok::Int))
            {
                const Token& v = advance();
                prog.version = v.text;
                prog.headerSpan = Span::join(kw.span, v.span);
                if (versionNewer(v.text))
                    d.error("E2006", v.span, std::format("language version {} is newer than this noether ({})", v.text,
                                                         kLanguageVersion));
            }
            else d.error("E2001", here(), std::format("expected a version number after `noether`, e.g. `noether {}`",
                                                      kLanguageVersion));
            if (!accept(Tok::Newline) && !at(Tok::End))
            {
                d.error("E2001", here(), "expected a newline after the version header");
                sync();
            }
        }

        static bool versionNewer(std::string_view v)
        {
            auto parts = [](std::string_view s)
            {
                int major = 0, minor = 0;
                const std::size_t dot = s.find('.');
                std::from_chars(s.data(), s.data() + (dot == std::string_view::npos ? s.size() : dot), major);
                if (dot != std::string_view::npos) std::from_chars(s.data() + dot + 1, s.data() + s.size(), minor);
                return std::pair{major, minor};
            };
            return parts(v) > parts(kLanguageVersion);
        }

        // ---- Recovery ----
        void sync()
        {
            int depth = 0;
            while (!at(Tok::End))
            {
                if (at(Tok::Indent)) ++depth;
                if (at(Tok::Dedent))
                {
                    if (depth == 0) return;
                    --depth;
                }
                if (at(Tok::Newline) && depth == 0)
                {
                    advance();
                    if (at(Tok::Indent)) continue; // also skip the block of a broken compound header
                    return;
                }
                advance();
            }
        }

        void skipBlock()
        {
            int depth = 0;
            do
            {
                if (at(Tok::Indent)) ++depth;
                else if (at(Tok::Dedent)) --depth;
                advance();
            } while (depth > 0 && !at(Tok::End));
        }

        // ---- Statements ----
        Block lines(bool untilDedent)
        {
            Block out;
            while (!at(Tok::End) && !(untilDedent && at(Tok::Dedent)))
            {
                if (accept(Tok::Newline)) continue;
                if (at(Tok::Indent))
                {
                    d.error("E2001", here(), "unexpected indentation");
                    skipBlock();
                    continue;
                }
                const std::size_t start = i;
                try
                {
                    line(out);
                }
                catch (const ParseAbort&)
                {
                    sync();
                }
                if (i == start) advance(); // guarantee progress
            }
            if (untilDedent) accept(Tok::Dedent);
            return out;
        }

        void line(Block& out)
        {
            switch (peek().kind)
            {
                case Tok::KwFor: out.push_back(forStmt()); return;
                case Tok::KwIf: out.push_back(ifStmt()); return;
                case Tok::KwDef:
                case Tok::KwProc: out.push_back(defStmt()); return;
                case Tok::KwNoise: out.push_back(noiseStmt()); return;
                default: break;
            }
            simpleLine(out);
            if (!accept(Tok::Newline) && !at(Tok::End) && !at(Tok::Dedent)) fail("end of statement");
        }

        void simpleLine(Block& out)
        {
            do
            {
                if (at(Tok::Newline) || at(Tok::End)) break;
                if (StmtPtr s = simple()) out.push_back(std::move(s));
            } while (accept(Tok::Semicolon));
        }

        // suite := simple_line | NEWLINE INDENT line+ DEDENT
        Block suite()
        {
            if (accept(Tok::Newline))
            {
                if (!at(Tok::Indent)) fail("an indented block");
                advance();
                return lines(true);
            }
            Block b;
            simpleLine(b);
            if (!accept(Tok::Newline) && !at(Tok::End) && !at(Tok::Dedent)) fail("end of statement");
            return b;
        }

        std::shared_ptr<Stmt> stmt(SK k, Span s)
        {
            auto st = std::make_shared<Stmt>();
            st->kind = k;
            st->span = s;
            st->head = s;
            return st;
        }

        void finish(Stmt& st) const
        {
            st.span = Span::join(st.span, prevSpan());
            st.head = st.span;
        }

        StmtPtr simple()
        {
            const Token& t = peek();
            switch (t.kind)
            {
                case Tok::KwQubits:
                case Tok::KwBits:
                {
                    auto st = stmt(t.kind == Tok::KwQubits ? SK::Qubits : SK::Bits, advance().span);
                    do
                    {
                        RegDecl r;
                        const Token& name = expect(Tok::Ident, "a register name");
                        r.name = name.text;
                        r.span = name.span;
                        if (!touching(Tok::LBracket)) fail("'[' with the register size");
                        advance();
                        r.size = expr();
                        expect(Tok::RBracket, "']'");
                        r.span = Span::join(r.span, prevSpan());
                        st->regs.push_back(std::move(r));
                    } while (accept(Tok::Comma));
                    finish(*st);
                    return st;
                }
                case Tok::KwSeed:
                case Tok::KwTrajectories:
                {
                    auto st = stmt(t.kind == Tok::KwSeed ? SK::Seed : SK::Trajectories, advance().span);
                    st->expr = expr();
                    finish(*st);
                    return st;
                }
                case Tok::KwBackend:
                {
                    auto st = stmt(SK::Backend, advance().span);
                    const Token& w = expect(Tok::Ident, "auto, statevector or stabilizer");
                    if (w.text != "auto" && w.text != "statevector" && w.text != "stabilizer")
                    {
                        d.error("E2001", w.span, std::format("unknown backend '{}'; expected auto, statevector or "
                                                             "stabilizer", w.text));
                    }
                    st->text = w.text;
                    finish(*st);
                    return st;
                }
                case Tok::KwLet: return letStmt();
                case Tok::KwParam: return paramStmt();
                case Tok::KwPrepare:
                {
                    auto st = stmt(SK::Prepare, advance().span);
                    st->expr = expr();
                    finish(*st);
                    return st;
                }
                case Tok::KwReset:
                {
                    auto st = stmt(SK::Reset, advance().span);
                    if (!touching(Tok::Underscore)) fail("'_' and the qubits to reset (reset_q)");
                    st->qlist = subscript();
                    finish(*st);
                    return st;
                }
                case Tok::KwPrint:
                {
                    auto st = stmt(SK::Print, advance().span);
                    do
                    {
                        PrintItem item;
                        item.expr = expr();
                        item.span = item.expr->span;
                        if (accept(Tok::KwAs))
                        {
                            item.label = expect(Tok::Ident, "a label").text;
                            item.span = Span::join(item.span, prevSpan());
                        }
                        st->items.push_back(std::move(item));
                    } while (accept(Tok::Comma));
                    finish(*st);
                    return st;
                }
                case Tok::KwAssert:
                {
                    auto st = stmt(SK::Assert, advance().span);
                    st->expr = expr();
                    finish(*st);
                    return st;
                }
                case Tok::KwImport:
                {
                    auto st = stmt(SK::Import, advance().span);
                    st->text = expect(Tok::String, "a quoted path").text;
                    finish(*st);
                    return st;
                }
                case Tok::KwElse:
                    d.error("E2001", t.span, "`else` must start its own line at the indentation of its `if`");
                    throw ParseAbort{};
                case Tok::Ident:
                    if (specMode && kSpecKeywords.contains(t.text)) return specStmt();
                    if (peek(1).kind == Tok::Dot && peek(2).kind == Tok::Ident) return qiskitMethod();
                    if (isAssignment()) return assignment();
                    if (kQiskitGates.contains(t.text) && peek(1).kind == Tok::LParen && !peek(1).spaceBefore &&
                        qiskitCall())
                        return nullptr;
                    break;
                default: break;
            }
            auto st = stmt(SK::Apply, here());
            st->expr = expr();
            finish(*st);
            return st;
        }

        bool isAssignment() const
        {
            if (!at(Tok::Ident)) return false;
            if (peek(1).kind == Tok::LeftArrow) return true;
            if (peek(1).kind != Tok::LBracket || peek(1).spaceBefore) return false;
            int depth = 0;
            for (std::size_t k = 1; i + k < toks.size(); ++k)
            {
                const Tok tk = peek(k).kind;
                if (tk == Tok::LBracket) ++depth;
                else if (tk == Tok::RBracket && --depth == 0) return peek(k + 1).kind == Tok::LeftArrow;
                else if (tk == Tok::Newline || tk == Tok::End) return false;
            }
            return false;
        }

        StmtPtr assignment()
        {
            const Token& name = advance();
            auto lv = node(EK::Name, name.span);
            lv->name = name.text;
            ExprPtr target = lv;
            if (touching(Tok::LBracket))
            {
                advance();
                auto idx = node(EK::Index, name.span);
                idx->kids = {lv, indexOrSlice()};
                expect(Tok::RBracket, "']'");
                idx->span = Span::join(name.span, prevSpan());
                target = idx;
            }
            expect(Tok::LeftArrow, "'←'");
            if (accept(Tok::KwRun))
            {
                auto st = stmt(SK::Run, name.span);
                st->lvalue = target;
                st->expr = expr();
                finish(*st);
                return st;
            }
            if (!accept(Tok::KwMeasure)) fail("`measure` or `run` after '←'");
            auto st = stmt(SK::Measure, name.span);
            st->lvalue = target;
            if (touching(Tok::Underscore))
            {
                st->measureSub = true;
                st->qlist = subscript();
            }
            else st->expr = expr();
            finish(*st);
            return st;
        }

        StmtPtr letStmt()
        {
            auto st = stmt(SK::Let, advance().span);
            do
            {
                Binding b;
                const Token& name = expect(Tok::Ident, "a name to bind");
                b.name = name.text;
                b.nameSpan = name.span;
                if (touching(Tok::Underscore))
                {
                    b.hasSub = true;
                    b.sub = subParams();
                }
                expect(Tok::Assign, "'='");
                b.value = expr();
                st->bindings.push_back(std::move(b));
            } while (accept(Tok::Comma));
            finish(*st);
            return st;
        }

        StmtPtr paramStmt()
        {
            auto st = stmt(SK::Param, advance().span);
            const Token& name = expect(Tok::Ident, "a parameter name");
            st->name = name.text;
            st->nameSpan = name.span;
            if (touching(Tok::LBracket))
            {
                advance();
                st->paramSize = expr();
                expect(Tok::RBracket, "']'");
            }
            expect(Tok::In, "'∈' and an interval [lo, hi]");
            expect(Tok::LBracket, "'[' starting the interval");
            st->lo = expr();
            expect(Tok::Comma, "','");
            st->hi = expr();
            expect(Tok::RBracket, "']'");
            if (accept(Tok::Assign)) st->init = expr();
            finish(*st);
            return st;
        }

        StmtPtr forStmt()
        {
            auto st = stmt(SK::For, advance().span);
            const Token& var = expect(Tok::Ident, "a loop variable");
            st->name = var.text;
            st->nameSpan = var.span;
            expect(Tok::In, "`in`");
            ExprPtr first = expr();
            if (accept(Tok::DotDot))
            {
                st->lo = first;
                st->hi = expr();
                if (accept(Tok::KwBy)) st->step = expr();
            }
            else if (first->kind == EK::List) st->expr = first;
            else
            {
                d.error("E2001", first->span, "expected a range a..b or a list [a, b, …]");
                throw ParseAbort{};
            }
            expect(Tok::Colon, "':'");
            st->head = Span::join(st->span, prevSpan());
            st->body = suite();
            st->span = Span::join(st->span, prevSpan());
            return st;
        }

        StmtPtr ifStmt()
        {
            auto st = stmt(SK::If, advance().span);
            st->expr = expr();
            expect(Tok::Colon, "':'");
            st->head = Span::join(st->span, prevSpan());
            st->body = suite();
            if (at(Tok::KwElse))
            {
                advance();
                expect(Tok::Colon, "':' after else");
                st->orelse = suite();
            }
            st->span = Span::join(st->span, prevSpan());
            return st;
        }

        StmtPtr defStmt()
        {
            const bool isProc = at(Tok::KwProc);
            auto st = stmt(isProc ? SK::Proc : SK::Def, advance().span);
            const Token& name = expect(Tok::Ident, "a name");
            st->name = name.text;
            st->nameSpan = name.span;
            if (touching(Tok::LParen))
            {
                advance();
                if (!at(Tok::RParen))
                    do st->cparams.push_back(expect(Tok::Ident, "a parameter name").text);
                    while (accept(Tok::Comma));
                expect(Tok::RParen, "')'");
            }
            if (!touching(Tok::Underscore)) fail("'_' and the qubit parameters, e.g. _{a,b} or _{r[n]}");
            st->sub = subParams();
            expect(Tok::Colon, "':'");
            st->head = Span::join(st->span, prevSpan());
            st->body = suite();
            st->span = Span::join(st->span, prevSpan());
            return st;
        }

        StmtPtr noiseStmt()
        {
            auto st = stmt(SK::Noise, advance().span);
            expect(Tok::Colon, "':'");
            st->head = Span::join(st->span, prevSpan());
            expect(Tok::Newline, "a newline and an indented list of rules");
            expect(Tok::Indent, "an indented list of rules");
            while (!at(Tok::Dedent) && !at(Tok::End))
            {
                if (accept(Tok::Newline)) continue;
                try
                {
                    NoiseRule r;
                    r.span = here();
                    if (accept(Tok::KwAfter)) r.after = true;
                    else if (accept(Tok::KwBefore)) r.after = false;
                    else fail("`after` or `before`");
                    if (at(Tok::KwMeasure) || at(Tok::KwReset) || at(Tok::KwPrepare))
                        r.event = std::string(canonicalText(advance().kind));
                    else r.event = expect(Tok::Ident, "gate1, gate2, measure, reset or prepare").text;
                    if (r.event != "gate1" && r.event != "gate2" && r.event != "measure" && r.event != "reset" &&
                        r.event != "prepare")
                        d.error("E2001", prevSpan(), std::format("unknown noise event '{}'; expected gate1, gate2, "
                                                                 "measure, reset or prepare", r.event));
                    expect(Tok::Colon, "':'");
                    r.channel = expr();
                    r.span = Span::join(r.span, prevSpan());
                    st->rules.push_back(std::move(r));
                    if (!accept(Tok::Newline) && !at(Tok::Dedent)) fail("end of rule");
                }
                catch (const ParseAbort&)
                {
                    sync();
                }
            }
            accept(Tok::Dedent);
            st->span = Span::join(st->span, prevSpan());
            return st;
        }

        // subparams := '_' ( IDENT | '{' qparam { (',' | '→') qparam } '}' )
        SubParams subParams()
        {
            expect(Tok::Underscore, "'_'");
            SubParams sp;
            auto qparam = [&]
            {
                QParam q;
                const Token& n = expect(Tok::Ident, "a qubit parameter name");
                q.name = n.text;
                q.span = n.span;
                if (touching(Tok::LBracket))
                {
                    advance();
                    if (at(Tok::Int)) q.sizeName = advance().text;
                    else q.sizeName = expect(Tok::Ident, "a size name or number, as in r[n] or r[3]").text;
                    expect(Tok::RBracket, "']'");
                    q.span = Span::join(q.span, prevSpan());
                }
                sp.params.push_back(std::move(q));
            };
            if (accept(Tok::LBrace))
            {
                sp.braced = true;
                qparam();
                while (at(Tok::Comma) || at(Tok::Arrow))
                {
                    sp.seps.push_back(advance().kind);
                    qparam();
                }
                expect(Tok::RBrace, "'}'");
            }
            else qparam();
            return sp;
        }

        // ---- Spec statements ----
        StmtPtr specStmt()
        {
            const Token& kw = advance();
            auto st = stmt(SK::Spec, kw.span);
            SpecStmt& sp = st->spec;
            sp.keyword = kw.text;
            const std::string& k = sp.keyword;

            auto gateName = [&]() -> std::pair<std::string, bool>
            {
                if (accept(Tok::Sqrt)) return {"√" + expect(Tok::Ident, "a gate name").text, false};
                std::string n = expect(Tok::Ident, "a gate name").text;
                return {n, accept(Tok::Dagger)};
            };
            auto options = [&]
            {
                do
                {
                    const std::string key = expect(Tok::Ident, "an option name").text;
                    expect(Tok::Assign, "'='");
                    ExprPtr v;
                    if ((at(Tok::Int) || at(Tok::Real)) && peek(1).kind == Tok::Ident && !peek(1).spaceBefore)
                    {
                        const Token& num = advance();
                        const Token& unit = advance();
                        auto e = node(EK::String, Span::join(num.span, unit.span));
                        e->name = num.text + unit.text;
                        v = e;
                    }
                    else v = expr();
                    sp.options.emplace_back(key, v);
                } while (accept(Tok::Comma));
            };

            if (k == "task") sp.name = expect(Tok::String, "a quoted task name").text;
            else if (k == "candidate")
            {
                if (accept(Tok::KwDef)) sp.word = "def";
                else if (accept(Tok::KwProc)) sp.word = "proc";
                else fail("`def` or `proc`");
                sp.name = expect(Tok::Ident, "the candidate name").text;
                if (!touching(Tok::Underscore)) fail("'_' and the candidate's qubit parameters");
                sp.sub = subParams();
            }
            else if (k == "target")
            {
                sp.word = expect(Tok::Ident, "state, unitary or ground").text;
                if (sp.word != "state" && sp.word != "unitary" && sp.word != "ground")
                    d.error("E2001", prevSpan(), "expected `state`, `unitary` or `ground`");
                sp.expr = expr();
            }
            else if (k == "metric")
            {
                sp.name = expect(Tok::Ident, "a metric name").text;
                expect(Tok::Assign, "'='");
                sp.expr = expr();
            }
            else if (k == "gates")
            {
                expect(Tok::LBrace, "'{' and a gate set");
                do sp.gates.push_back(gateName());
                while (accept(Tok::Comma));
                expect(Tok::RBrace, "'}'");
            }
            else if (k == "forbid")
            {
                do sp.gates.push_back(gateName());
                while (accept(Tok::Comma));
            }
            else if (k == "allow") sp.word = expect(Tok::Ident, "`matrix`").text;
            else if (k == "coupling")
            {
                if (at(Tok::LBrace))
                {
                    auto set = node(EK::Set, advance().span);
                    do
                    {
                        auto pair = node(EK::List, here());
                        expect(Tok::LParen, "'(' starting a qubit pair");
                        pair->kids.push_back(expr());
                        expect(Tok::Comma, "','");
                        pair->kids.push_back(expr());
                        expect(Tok::RParen, "')'");
                        set->kids.push_back(pair);
                    } while (accept(Tok::Comma));
                    expect(Tok::RBrace, "'}'");
                    sp.word = "set";
                    sp.expr = set;
                }
                else
                {
                    sp.word = expect(Tok::Ident, "all, line, ring, grid(r, c) or a set of pairs").text;
                    if (sp.word == "grid")
                    {
                        expect(Tok::LParen, "'('");
                        auto lst = node(EK::List, here());
                        lst->kids.push_back(expr());
                        expect(Tok::Comma, "','");
                        lst->kids.push_back(expr());
                        expect(Tok::RParen, "')'");
                        sp.expr = lst;
                    }
                    else if (sp.word != "all" && sp.word != "line" && sp.word != "ring")
                        d.error("E2001", prevSpan(), "expected all, line, ring, grid(r, c) or {(a, b), …}");
                }
            }
            else if (k == "readout")
            {
                if (accept(Tok::KwRun)) fail("exact or shots(n)");
                sp.word = expect(Tok::Ident, "exact or shots(n)").text;
                if (sp.word == "shots")
                {
                    expect(Tok::LParen, "'('");
                    sp.expr = expr();
                    expect(Tok::RParen, "')'");
                }
                else if (sp.word != "exact") d.error("E2001", prevSpan(), "expected exact or shots(n)");
            }
            else if (k == "optimize" || k == "budget") options();
            else if (k == "require" || k == "goal") sp.expr = expr();
            else if (k == "minimize" || k == "maximize") sp.expr = expr();
            else if (k == "instances" || k == "holdout")
            {
                sp.name = expect(Tok::Ident, "the name of a spec let").text;
                expect(Tok::In, "'∈'");
                auto set = node(EK::Set, here());
                expect(Tok::LBrace, "'{'");
                do set->kids.push_back(expr());
                while (accept(Tok::Comma));
                expect(Tok::RBrace, "'}'");
                set->span = Span::join(set->span, prevSpan());
                sp.expr = set;
            }
            else if (k == "aggregate")
            {
                sp.word = expect(Tok::Ident, "max or mean").text;
                if (sp.word != "max" && sp.word != "mean") d.error("E2001", prevSpan(), "expected max or mean");
            }
            finish(*st);
            return st;
        }

        // ---- Qiskit-isms (E2009): reported with the Noether spelling as a fix-it ----
        static std::string literalArg(const std::vector<Token>& args, std::size_t k)
        {
            return k < args.size() ? args[k].text : "?";
        }

        std::string qiskitFix(std::string_view method, const std::vector<Token>& args) const
        {
            const auto it = kQiskitGates.find(method);
            if (it == kQiskitGates.end()) return {};
            const std::string_view g = it->second;
            auto a = [&](std::size_t k) { return literalArg(args, k); };
            if (method == "measure") return std::format("c[{}] ← measure Z_{}", a(1), a(0));
            if (method == "reset") return std::format("reset_{}", a(0));
            if (g == "CNOT") return std::format("CNOT_{{{}→{}}}", a(0), a(1));
            if (g == "CZ" || g == "SWAP") return std::format("{}_{{{},{}}}", g, a(0), a(1));
            if (g == "CP") return std::format("CP({})_{{{},{}}}", a(0), a(1), a(2));
            if (g == "Toffoli") return std::format("Toffoli_{{{},{}→{}}}", a(0), a(1), a(2));
            if (g == "Fredkin") return std::format("Fredkin_{{{}→{},{}}}", a(0), a(1), a(2));
            if (g == "Rx" || g == "Ry" || g == "Rz" || g == "P") return std::format("{}({})_{}", g, a(0), a(1));
            if (g == "U3") return std::format("U3({}, {}, {})_{}", a(0), a(1), a(2), a(3));
            return std::format("{}_{}", g, a(0));
        }

        // Parses `( args )` keeping the raw token text of each argument; false if malformed.
        bool rawArgs(std::vector<Token>& args)
        {
            if (!accept(Tok::LParen)) return false;
            while (!at(Tok::RParen) && !at(Tok::Newline) && !at(Tok::End))
            {
                Token a = advance();
                // Join simple signed literals and pi expressions into one text chunk.
                while (!at(Tok::Comma) && !at(Tok::RParen) && !at(Tok::Newline) && !at(Tok::End))
                {
                    const Token& more = advance();
                    a.text += more.kind == Tok::Ident || more.kind == Tok::Int || more.kind == Tok::Real
                                  ? more.text
                                  : std::string(canonicalText(more.kind));
                    a.span = Span::join(a.span, more.span);
                }
                if (a.kind != Tok::Ident && a.kind != Tok::Int && a.kind != Tok::Real && a.text.empty())
                    a.text = std::string(canonicalText(a.kind));
                args.push_back(std::move(a));
                if (!accept(Tok::Comma)) break;
            }
            return accept(Tok::RParen);
        }

        StmtPtr qiskitMethod()
        {
            const Token& obj = advance();
            advance(); // '.'
            const Token& method = advance();
            std::vector<Token> args;
            const bool ok = rawArgs(args);
            const Span all = Span::join(obj.span, prevSpan());
            Diagnostic& diag = d.error("E2009", all, std::format("Qiskit-style call `{}.{}(…)` is not Noether syntax",
                                                                 obj.text, method.text));
            if (const std::string fix = ok ? qiskitFix(method.text, args) : std::string(); !fix.empty())
                diag.fix("use the Noether form", all, fix);
            return nullptr;
        }

        bool qiskitCall()
        {
            const std::size_t save = i;
            const Token& name = advance();
            std::vector<Token> args;
            const bool ok = rawArgs(args);
            if (!ok || !(at(Tok::Newline) || at(Tok::End) || at(Tok::Semicolon)) || args.empty() ||
                (args[0].kind != Tok::Int && name.text != "rx" && name.text != "ry" && name.text != "rz" &&
                 name.text != "p" && name.text != "cp" && name.text != "u" && name.text != "u3"))
            {
                i = save;
                return false;
            }
            const Span all = Span::join(name.span, prevSpan());
            Diagnostic& diag = d.error("E2009", all, std::format("Qiskit-style call `{}(…)` is not Noether syntax",
                                                                 name.text));
            if (const std::string fix = qiskitFix(name.text, args); !fix.empty())
                diag.fix("use the Noether form", all, fix);
            return true;
        }

        // ---- Expressions ----
        ExprPtr expr() { return disj(); }

        ExprPtr binary(Tok op, ExprPtr l, ExprPtr r)
        {
            auto e = node(EK::Binary, Span::join(l->span, r->span));
            e->op = op;
            e->kids = {std::move(l), std::move(r)};
            return e;
        }

        ExprPtr disj()
        {
            ExprPtr l = conj();
            while (accept(Tok::KwOr)) l = binary(Tok::KwOr, l, conj());
            return l;
        }

        ExprPtr conj()
        {
            ExprPtr l = cmp();
            while (accept(Tok::KwAnd)) l = binary(Tok::KwAnd, l, cmp());
            return l;
        }

        static bool isCmpOp(Tok k)
        {
            return k == Tok::EqEq || k == Tok::NotEq || k == Tok::Less || k == Tok::LessEq || k == Tok::Greater ||
                   k == Tok::GreaterEq || k == Tok::Approx;
        }

        ExprPtr cmp()
        {
            ExprPtr l = sum();
            if (isCmpOp(peek().kind))
            {
                const Token& opTok = advance();
                ExprPtr r = sum();
                auto e = node(EK::Compare, Span::join(l->span, r->span));
                e->op = opTok.kind;
                e->kids = {l, r};
                if (at(Tok::PlusMinus))
                {
                    const Span pm = advance().span;
                    if (opTok.kind != Tok::Approx)
                        d.error("E2004", pm, "a tolerance '±' needs an approximate comparison '≈'")
                            .fix("compare with ≈", opTok.span, "≈");
                    e->kids.push_back(sum());
                    e->span = Span::join(e->span, prevSpan());
                }
                if (isCmpOp(peek().kind))
                {
                    d.error("E2001", here(), "comparisons do not chain; combine them with `and`");
                    throw ParseAbort{};
                }
                return e;
            }
            if (at(Tok::PlusMinus))
            {
                d.error("E2004", here(), "a tolerance '±' needs an approximate comparison '≈'");
                throw ParseAbort{};
            }
            return l;
        }

        ExprPtr sum()
        {
            ExprPtr l = tensor();
            while (at(Tok::Plus) || at(Tok::Minus))
            {
                const Tok op = advance().kind;
                l = binary(op, l, tensor());
            }
            return l;
        }

        ExprPtr tensor()
        {
            ExprPtr l = product();
            while (accept(Tok::Otimes)) l = binary(Tok::Otimes, l, product());
            return l;
        }

        bool canStartFactor() const
        {
            switch (peek().kind)
            {
                case Tok::Ident: case Tok::Int: case Tok::Real: case Tok::Ket: case Tok::Bra: case Tok::Braket:
                case Tok::LParen: case Tok::LAngle: case Tok::LBracket: case Tok::Sqrt: case Tok::Sum:
                case Tok::Prod: case Tok::String:
                    return true;
                case Tok::Abs: return absDepth == 0;
                default: return false;
            }
        }

        ExprPtr product()
        {
            ExprPtr first = muldiv();
            if (!canStartFactor()) return first;
            auto p = node(EK::Product, first->span);
            p->kids.push_back(first);
            while (canStartFactor())
            {
                const bool tight = !peek().spaceBefore;
                ExprPtr factor = muldiv();
                if (tight)
                {
                    auto copy = std::make_shared<Expr>(*factor);
                    copy->touching = true;
                    factor = copy;
                }
                p->kids.push_back(factor);
            }
            p->span = Span::join(p->span, p->kids.back()->span);
            return p;
        }

        ExprPtr muldiv()
        {
            ExprPtr l = prefix();
            while (at(Tok::Cdot) || at(Tok::Slash))
            {
                const Tok op = advance().kind;
                l = binary(op, l, prefix());
            }
            return l;
        }

        // √ touching X names the √X gate, so postfix operators apply to the whole gate: √X† is (√X)†.
        bool atSqrtX() const { return at(Tok::Sqrt) && peek(1).kind == Tok::Ident && peek(1).text == "X" && !peek(1).spaceBefore; }

        ExprPtr prefix()
        {
            if (atSqrtX()) return postfix();
            if (at(Tok::Minus) || at(Tok::Plus) || at(Tok::Not) || at(Tok::Sqrt))
            {
                const Token& op = advance();
                ExprPtr x = prefix();
                auto e = node(EK::Unary, Span::join(op.span, x->span));
                e->op = op.kind;
                e->kids = {x};
                return e;
            }
            if (at(Tok::Sum) || at(Tok::Prod)) return bigop();
            return postfix();
        }

        ExprPtr bigop()
        {
            const Token& op = advance();
            auto e = node(EK::BigOp, op.span);
            e->op = op.kind;
            if (!touching(Tok::Underscore)) fail("'_{j=a}' after Σ or ∏");
            advance();
            expect(Tok::LBrace, "'{'");
            e->name = expect(Tok::Ident, "the summation variable").text;
            expect(Tok::Assign, "'='");
            ExprPtr lo = expr();
            expect(Tok::RBrace, "'}'");
            if (!at(Tok::Caret)) fail("'^' and the upper bound");
            advance();
            ExprPtr hi;
            if (accept(Tok::LBrace))
            {
                hi = expr();
                expect(Tok::RBrace, "'}'");
                e->braced = true;
            }
            else hi = atom();
            ExprPtr body = product();
            e->kids = {lo, hi, body};
            e->span = Span::join(op.span, body->span);
            return e;
        }

        static bool callableHead(const Expr& e)
        {
            const Expr* p = &e;
            while (p->kind == EK::Sub || p->kind == EK::Dagger || p->kind == EK::Power || p->kind == EK::Call)
                p = p->kids[0].get();
            return p->kind == EK::Name;
        }

        // `abs(x)` and `exp(x)` are spellings of |x| and e^{x} (`e` is reserved), so they parse to
        // the same tree and the formatter writes the symbol form, as it does for sqrt → √.
        ExprPtr canonicalCall(const std::shared_ptr<Expr>& call) const
        {
            const Expr& callee = *call->kids[0];
            if (callee.kind != EK::Name || call->kids.size() != 2 || !call->argNames[0].empty()) return call;
            if (callee.name == "abs")
            {
                auto e = node(EK::Abs, call->span);
                e->kids = {call->kids[1]};
                return e;
            }
            if (callee.name == "exp")
            {
                auto base = node(EK::Name, callee.span);
                base->name = "e";
                auto e = node(EK::Power, call->span);
                e->kids = {base, call->kids[1]};
                e->braced = true;
                return e;
            }
            return call;
        }

        ExprPtr postfix()
        {
            ExprPtr base = primary();
            while (true)
            {
                const Token& t = peek();
                if (t.spaceBefore) break;
                if (t.kind == Tok::Underscore)
                {
                    auto e = node(EK::Sub, base->span);
                    QList q = subscript();
                    e->kids = {base};
                    e->qlist = std::move(q);
                    e->span = Span::join(base->span, prevSpan());
                    base = e;
                }
                else if (t.kind == Tok::LParen && callableHead(*base))
                {
                    advance();
                    auto e = node(EK::Call, base->span);
                    e->kids = {base};
                    if (!at(Tok::RParen))
                        do
                        {
                            if (at(Tok::Ident) && peek(1).kind == Tok::Assign)
                            {
                                e->argNames.push_back(advance().text);
                                advance();
                            }
                            else e->argNames.emplace_back();
                            e->kids.push_back(expr());
                        } while (accept(Tok::Comma));
                    expect(Tok::RParen, "')'");
                    e->span = Span::join(base->span, prevSpan());
                    base = canonicalCall(e);
                }
                else if (t.kind == Tok::LBracket && base->kind != EK::Int && base->kind != EK::Real)
                {
                    advance();
                    auto e = node(EK::Index, base->span);
                    e->kids = {base, indexOrSlice()};
                    expect(Tok::RBracket, "']'");
                    e->span = Span::join(base->span, prevSpan());
                    base = e;
                }
                else if (t.kind == Tok::Dagger)
                {
                    advance();
                    auto e = node(EK::Dagger, Span::join(base->span, prevSpan()));
                    e->kids = {base};
                    base = e;
                }
                else if (t.kind == Tok::Caret)
                {
                    advance();
                    auto e = node(EK::Power, base->span);
                    bool braced = false;
                    ExprPtr ex = powerExponent(braced);
                    e->kids = {base, ex};
                    e->braced = braced;
                    e->span = Span::join(base->span, prevSpan());
                    base = e;
                }
                else if (t.kind == Tok::TensorPow)
                {
                    advance();
                    auto e = node(EK::TensorPow, base->span);
                    ExprPtr ex;
                    if (accept(Tok::LBrace))
                    {
                        ex = expr();
                        expect(Tok::RBrace, "'}'");
                        e->braced = true;
                    }
                    else if (at(Tok::LParen)) ex = primary();
                    else ex = atom();
                    e->kids = {base, ex};
                    e->span = Span::join(base->span, prevSpan());
                    base = e;
                }
                else break;
            }
            return base;
        }

        // power := '^' ( atom | '-' atom | '{' expr '}' | '(' expr ')' ); right-associative.
        ExprPtr powerExponent(bool& braced)
        {
            if (accept(Tok::LBrace))
            {
                ExprPtr e = expr();
                expect(Tok::RBrace, "'}'");
                braced = true;
                return e;
            }
            if (at(Tok::LParen)) return primary();
            if (at(Tok::Minus))
            {
                const Token& m = advance();
                bool inner = false;
                ExprPtr x = powerExponent(inner);
                auto e = node(EK::Unary, Span::join(m.span, x->span));
                e->op = Tok::Minus;
                e->kids = {x};
                return e;
            }
            ExprPtr a = atom();
            if (touching(Tok::Caret))
            {
                advance();
                bool inner = false;
                ExprPtr ex = powerExponent(inner);
                auto e = node(EK::Power, Span::join(a->span, prevSpan()));
                e->kids = {a, ex};
                e->braced = inner;
                return e;
            }
            return a;
        }

        ExprPtr atom()
        {
            const Token& t = peek();
            if (t.kind == Tok::Ident || t.kind == Tok::Int || t.kind == Tok::Real) return primary();
            fail("a name or number");
        }

        // qlist after a touching '_': INT | IDENT | '{' qitem { (',' | '→') qitem } '}'
        QList subscript()
        {
            expect(Tok::Underscore, "'_'");
            QList q;
            if (accept(Tok::LBrace))
            {
                q.braced = true;
                q.items.push_back(qitem());
                while (at(Tok::Comma) || at(Tok::Arrow))
                {
                    q.seps.push_back(advance().kind);
                    q.items.push_back(qitem());
                }
                expect(Tok::RBrace, "'}'");
                return q;
            }
            if (at(Tok::Int) || at(Tok::Ident))
            {
                const Token& t = advance();
                auto e = node(t.kind == Tok::Int ? EK::Int : EK::Name, t.span);
                e->name = t.text;
                q.items.push_back({false, e, t.span});
                return q;
            }
            fail("a qubit index, a register name or {…} after '_'");
        }

        QItem qitem()
        {
            QItem it;
            const Span b = here();
            if (accept(Tok::Not)) it.negated = true;
            ExprPtr e = expr();
            if (accept(Tok::DotDot))
            {
                ExprPtr hi = expr();
                auto s = node(EK::Slice, Span::join(e->span, hi->span));
                s->kids = {e, hi};
                e = s;
            }
            it.expr = e;
            it.span = Span::join(b, prevSpan());
            return it;
        }

        ExprPtr indexOrSlice()
        {
            ExprPtr e = expr();
            if (accept(Tok::DotDot))
            {
                ExprPtr hi = expr();
                auto s = node(EK::Slice, Span::join(e->span, hi->span));
                s->kids = {e, hi};
                return s;
            }
            return e;
        }

        ExprPtr primary()
        {
            if (atSqrtX())
            {
                const Token& op = advance();
                ExprPtr x = primary();
                auto e = node(EK::Unary, Span::join(op.span, x->span));
                e->op = Tok::Sqrt;
                e->kids = {x};
                return e;
            }
            const Token& t = peek();
            switch (t.kind)
            {
                case Tok::Ident: case Tok::Int: case Tok::Real: case Tok::String: case Tok::Ket: case Tok::Bra:
                {
                    advance();
                    EK k = EK::Name;
                    if (t.kind == Tok::Int) k = EK::Int;
                    else if (t.kind == Tok::Real) k = EK::Real;
                    else if (t.kind == Tok::String) k = EK::String;
                    else if (t.kind == Tok::Ket) k = EK::Ket;
                    else if (t.kind == Tok::Bra) k = EK::Bra;
                    auto e = node(k, t.span);
                    e->name = t.text;
                    return e;
                }
                case Tok::Braket:
                {
                    advance();
                    auto e = node(EK::Braket, t.span);
                    e->name = t.text;
                    e->label2 = t.text2;
                    return e;
                }
                case Tok::KwTrue:
                case Tok::KwFalse:
                {
                    advance();
                    auto e = node(EK::Bool, t.span);
                    e->name = t.kind == Tok::KwTrue ? "true" : "false";
                    return e;
                }
                case Tok::LParen:
                {
                    const Span b = advance().span;
                    const int savedAbs = absDepth;
                    absDepth = 0;
                    ExprPtr x = expr();
                    absDepth = savedAbs;
                    expect(Tok::RParen, "')'");
                    auto e = node(EK::Paren, Span::join(b, prevSpan()));
                    e->kids = {x};
                    return e;
                }
                case Tok::LAngle:
                {
                    const Span b = advance().span;
                    const int savedAbs = absDepth;
                    absDepth = 0;
                    ExprPtr x = expr();
                    absDepth = savedAbs;
                    expect(Tok::RAngle, "'⟩' closing the expectation value");
                    auto e = node(EK::Expval, Span::join(b, prevSpan()));
                    e->kids = {x};
                    return e;
                }
                case Tok::Abs:
                {
                    const Span b = advance().span;
                    if (absDepth > 0)
                    {
                        d.error("E2007", b, "absolute-value bars cannot be nested; use abs(…)");
                        throw ParseAbort{};
                    }
                    ++absDepth;
                    ExprPtr x = expr();
                    --absDepth;
                    expect(Tok::Abs, "'|' closing the absolute value");
                    auto e = node(EK::Abs, Span::join(b, prevSpan()));
                    e->kids = {x};
                    return e;
                }
                case Tok::LBracket:
                {
                    const Span b = advance().span;
                    auto e = node(EK::List, b);
                    if (!at(Tok::RBracket))
                        do e->kids.push_back(expr());
                        while (accept(Tok::Comma));
                    expect(Tok::RBracket, "']'");
                    e->span = Span::join(b, prevSpan());
                    return e;
                }
                case Tok::Dagger:
                    d.error("E2001", t.span, "'†' must touch the operator it applies to, e.g. S†_0");
                    throw ParseAbort{};
                default: break;
            }
            fail("an expression");
        }

        // ---- Comments ----
        void attachComments(Program& prog)
        {
            // Statements in source order, with their nesting, so comments find their owners.
            std::vector<Stmt*> order;
            auto walk = [&](auto&& self, const Block& b) -> void
            {
                for (const StmtPtr& s : b)
                {
                    auto* m = const_cast<Stmt*>(s.get());
                    order.push_back(m);
                    self(self, m->body);
                    self(self, m->orelse);
                }
            };
            walk(walk, prog.stmts);
            std::ranges::stable_sort(order, [](const Stmt* a, const Stmt* b) { return a->span.begin < b->span.begin; });

            auto lineOf = [&](std::uint32_t off) { return f.lineCol(off).line; };
            auto blankAbove = [&](std::uint32_t line)
            {
                const std::uint32_t header = lineOf(prog.headerSpan.end);
                return line > header + 1 && f.lineText(line - 1).find_first_not_of(" \t\r") == std::string_view::npos;
            };

            std::size_t next = 0;
            for (Comment c : comments)
            {
                const std::uint32_t cl = lineOf(c.span.begin);
                c.blankBefore = c.ownLine && blankAbove(cl);
                if (!c.ownLine)
                {
                    // Trailing: the innermost statement whose own line(s) contain the comment.
                    Stmt* owner = nullptr;
                    for (Stmt* s : order)
                        if (lineOf(s->head.begin) <= cl && lineOf(s->head.end) >= cl && s->head.begin <= c.span.begin)
                            owner = s;
                    if (owner)
                    {
                        owner->comments.trailing.push_back(c);
                        continue;
                    }
                }
                while (next < order.size() && order[next]->span.begin < c.span.begin) ++next;
                if (next < order.size())
                {
                    Stmt* s = order[next];
                    s->comments.leading.push_back(c);
                    if (c.doc) s->doc += (s->doc.empty() ? "" : "\n") + trim(c.text);
                }
                else prog.trailingComments.push_back(c);
            }
            for (Stmt* s : order) s->comments.blankBefore = blankAbove(lineOf(s->span.begin));
        }

        static std::string trim(std::string s)
        {
            while (!s.empty() && s.front() == ' ') s.erase(s.begin());
            return s;
        }

        const SourceFile& f;
        std::uint32_t fid;
        Diagnostics& d;
        std::vector<Token> toks;
        std::vector<Comment> comments;
        std::size_t i = 0;
        int absDepth = 0;
        bool specMode = false;
    };

} // namespace


Program parse(const SourceFile& file, std::uint32_t fileId, Diagnostics& diags)
{
    return Parser(file, fileId, diags).run();
}

std::string_view grammarText()
{
    return R"EBNF((* Noether 0.1 grammar. Terminals in quotes or capitals; lexer terminals:
   NEWLINE INDENT DEDENT IDENT INT REAL STRING KET BRA BRAKET ABS. Unicode, LaTeX and ASCII
   spellings are normalised by the lexer (see `noether tokens`). *)

program      = header , { line } , EOF ;
header       = "noether" , REAL , NEWLINE ;
line         = simple_line | compound ;
simple_line  = simple , { ";" , simple } , NEWLINE ;
suite        = simple_line | NEWLINE , INDENT , line , { line } , DEDENT ;

simple       = decl | config | let | param | prepare | assign | reset
             | print | assert | import | apply | spec_stmt ;
compound     = for | if | def | proc | noise ;

decl         = ( "qubits" | "bits" ) , reg , { "," , reg } ;
reg          = IDENT , "[" , expr , "]" ;
config       = "seed" , INT
             | "backend" , ( "auto" | "statevector" | "stabilizer" )
             | "trajectories" , INT ;
let          = "let" , binding , { "," , binding } ;
binding      = IDENT , [ subparams ] , "=" , expr ;
param        = "param" , IDENT , [ "[" , expr , "]" ] , "∈" , interval , [ "=" , expr ] ;
interval     = "[" , expr , "," , expr , "]" ;
prepare      = "prepare" , expr ;
assign       = lvalue , "←" , ( measure | run ) ;
lvalue       = IDENT , [ "[" , slice , "]" ] ;
measure      = "measure" , ( sub | expr ) ;          (* sub: Z per qubit; expr: one Pauli string *)
run          = "run" , expr ;
reset        = "reset" , sub ;
print        = "print" , item , { "," , item } ;
item         = expr , [ "as" , IDENT ] ;
assert       = "assert" , expr ;
import       = "import" , STRING ;
apply        = expr ;                                (* type Op or channel, explicit support *)

for          = "for" , IDENT , "in" , range , ":" , suite ;
range        = expr , ".." , expr , [ "by" , expr ] | list ;
if           = "if" , expr , ":" , suite , [ "else" , ":" , suite ] ;
def          = "def" , IDENT , [ cparams ] , subparams , ":" , suite ;
proc         = "proc" , IDENT , [ cparams ] , subparams , ":" , suite ;
cparams      = "(" , [ IDENT , { "," , IDENT } ] , ")" ;
subparams    = "_" , ( IDENT | "{" , qparam , { ( "," | "→" ) , qparam } , "}" ) ;
qparam       = IDENT , [ "[" , ( IDENT | INT ) , "]" ] ;  (* r[n]: whole register, binds n; r[3]: fixed size *)
noise        = "noise" , ":" , NEWLINE , INDENT , rule , { rule } , DEDENT ;
rule         = ( "after" | "before" ) , event , ":" , expr , NEWLINE ;
event        = "gate1" | "gate2" | "measure" | "reset" | "prepare" ;

expr         = disj ;
disj         = conj , { "or" , conj } ;
conj         = cmp , { "and" , cmp } ;
cmp          = sum , [ cmpop , sum , [ "±" , sum ] ] ;
cmpop        = "==" | "≠" | "<" | "≤" | ">" | "≥" | "≈" ;
sum          = tensor , { ( "+" | "-" ) , tensor } ;
tensor       = product , { "⊗" , product } ;
product      = muldiv , { muldiv } ;                 (* juxtaposition = operator product *)
muldiv       = prefix , { ( "·" | "/" ) , prefix } ;
prefix       = ( "-" | "+" | "¬" | "√" ) , prefix | bigop | postfix ;
bigop        = ( "Σ" | "∏" ) , "_" , "{" , IDENT , "=" , expr , "}" ,
               "^" , ( atom | "{" , expr , "}" ) , product ;   (* body = rest of the chain *)
postfix      = primary , { sub | call | index | "†" | power | tpower } ;   (* must touch *)
sub          = "_" , ( INT | IDENT | "{" , qlist , "}" ) ;
qlist        = qitem , { ( "," | "→" ) , qitem } ;
qitem        = [ "¬" ] , ( expr | slice ) ;
call         = "(" , [ arg , { "," , arg } ] , ")" ;
arg          = expr | IDENT , "=" , expr ;
index        = "[" , ( slice | expr ) , "]" ;
slice        = expr , ".." , expr ;
power        = "^" , ( atom | "-" , atom | "{" , expr , "}" | "(" , expr , ")" ) ;
tpower       = "^⊗" , ( atom | "{" , expr , "}" | "(" , expr , ")" ) ;
primary      = atom | KET | BRA | BRAKET | STRING | "true" | "false"
             | "(" , expr , ")" | "⟨" , expr , "⟩" | ABS , expr , ABS | list ;
atom         = IDENT | INT | REAL ;
list         = "[" , [ expr , { "," , expr } ] , "]" ;

(* Spec files (first statement `task`) add: *)
spec_stmt    = "task" , STRING
             | "candidate" , ( "def" | "proc" ) , IDENT , subparams
             | "target" , ( "state" | "unitary" | "ground" ) , expr
             | "metric" , IDENT , "=" , expr
             | "gates" , "{" , gate , { "," , gate } , "}"
             | "forbid" , gate , { "," , gate }
             | "allow" , "matrix"
             | "coupling" , ( "all" | "line" | "ring" | "grid" , "(" , expr , "," , expr , ")"
                            | "{" , pair , { "," , pair } , "}" )
             | "readout" , ( "exact" | "shots" , "(" , expr , ")" )
             | "optimize" , option , { "," , option }
             | ( "require" | "goal" | "minimize" | "maximize" ) , expr
             | ( "instances" | "holdout" ) , IDENT , "∈" , "{" , expr , { "," , expr } , "}"
             | "aggregate" , ( "max" | "mean" )
             | "budget" , option , { "," , option } ;
gate         = IDENT , [ "†" ] | "√" , IDENT ;
pair         = "(" , expr , "," , expr , ")" ;
option       = IDENT , "=" , ( expr | INT , IDENT ) ;      (* 30s, 2h *)
)EBNF";
}

} // namespace Noether
