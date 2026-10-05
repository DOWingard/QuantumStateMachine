#include "Formatter.hpp"

#include "Lexer.hpp"

#include <algorithm>
#include <format>



namespace Noether
{

namespace
{

    class Printer
    {
        public:
        explicit Printer(bool asciiMode) : ascii(asciiMode) {}

        std::string tok(Tok t) const { return std::string(ascii ? asciiText(t) : canonicalText(t)); }

        std::string name(const std::string& n) const { return ascii ? asciiSpelling(n) : n; }

        std::string label(const std::string& l) const { return ascii ? asciiSpelling(l) : l; }

        static std::string quote(const std::string& s)
        {
            std::string out = "\"";
            for (const char c : s)
            {
                if (c == '"') out += '\\';
                out += c;
            }
            return out + "\"";
        }

        std::string qlist(const QList& q) const
        {
            if (!q.braced && q.items.size() == 1 && !q.items[0].negated &&
                (q.items[0].expr->kind == EK::Int || q.items[0].expr->kind == EK::Name))
                return "_" + expr(*q.items[0].expr);
            bool simple = true;
            for (const QItem& it : q.items)
                if (it.expr->kind != EK::Int && it.expr->kind != EK::Name) simple = false;
            std::string out = "_{";
            for (std::size_t k = 0; k < q.items.size(); ++k)
            {
                if (k) out += q.seps[k - 1] == Tok::Arrow ? tok(Tok::Arrow) : (simple ? "," : ", ");
                if (q.items[k].negated) out += ascii ? "not " : "¬";
                out += expr(*q.items[k].expr);
            }
            return out + "}";
        }

        std::string subParams(const SubParams& sp) const
        {
            auto one = [&](const QParam& p) { return name(p.name) + (p.sizeName.empty() ? "" : "[" + name(p.sizeName) + "]"); };
            if (!sp.braced && sp.params.size() == 1) return "_" + one(sp.params[0]);
            std::string out = "_{";
            for (std::size_t k = 0; k < sp.params.size(); ++k)
            {
                if (k) out += sp.seps[k - 1] == Tok::Arrow ? tok(Tok::Arrow) : ",";
                out += one(sp.params[k]);
            }
            return out + "}";
        }

        // An exponent the parser reads without braces: an atom, a parenthesised group, -atom, or a
        // right-nested power of atoms (2^3^2). Braces depend on this shape alone, never on how the
        // input was written, so every spelling formats alike.
        static bool bareExponent(const Expr& x)
        {
            switch (x.kind)
            {
                case EK::Int: case EK::Real: case EK::Name: case EK::Paren: return true;
                case EK::Unary:
                    return x.op == Tok::Minus && (x.kids[0]->kind == EK::Int || x.kids[0]->kind == EK::Real || x.kids[0]->kind == EK::Name);
                case EK::Power:
                    return (x.kids[0]->kind == EK::Int || x.kids[0]->kind == EK::Real || x.kids[0]->kind == EK::Name) && bareExponent(*x.kids[1]);
                default: return false;
            }
        }
        std::string exponent(const Expr& x) const { return bareExponent(x) ? expr(x) : "{" + expr(x) + "}"; }
        // (a^b)^c written as a^{b}^c: braces stop a^b^c from re-reading as a^(b^c).
        std::string powerBase(const Expr& x) const
        {
            if (x.kind != EK::Power) return expr(x);
            return powerBase(*x.kids[0]) + "^{" + expr(*x.kids[1]) + "}";
        }

        std::string expr(const Expr& e) const
        {
            switch (e.kind)
            {
                case EK::Name: return name(e.name);
                case EK::Int:
                case EK::Real: return e.name;
                case EK::String: return quote(e.name);
                case EK::Bool: return e.name;
                case EK::Ket: return "|" + label(e.name) + (ascii ? ">" : "⟩");
                case EK::Bra: return (ascii ? "<" : "⟨") + label(e.name) + "|";
                case EK::Braket: return (ascii ? "<" : "⟨") + label(e.name) + "|" + label(e.label2) + (ascii ? ">" : "⟩");
                case EK::List:
                case EK::Set:
                {
                    std::string out = e.kind == EK::List ? "[" : "{";
                    for (std::size_t k = 0; k < e.kids.size(); ++k) out += (k ? ", " : "") + expr(*e.kids[k]);
                    return out + (e.kind == EK::List ? "]" : "}");
                }
                case EK::Paren: return "(" + expr(*e.kids[0]) + ")";
                case EK::Unary:
                {
                    const Expr& x = *e.kids[0];
                    switch (e.op)
                    {
                        case Tok::Minus: return "-" + expr(x);
                        case Tok::Plus: return "+" + expr(x);
                        case Tok::Not: return ascii ? "not " + expr(x) : "¬" + expr(x);
                        case Tok::Sqrt:
                        {
                            if (!ascii) return "√" + expr(x);
                            const Expr* base = &x;
                            while (base->kind == EK::Sub || base->kind == EK::Dagger) base = base->kids[0].get();
                            if (base->kind == EK::Name && base->name == "X") return "S" + expr(x); // the SX alias
                            return x.kind == EK::Paren ? "sqrt" + expr(x) : "sqrt(" + expr(x) + ")";
                        }
                        default: break;
                    }
                    return expr(x);
                }
                case EK::Binary:
                {
                    const std::string a = expr(*e.kids[0]), b = expr(*e.kids[1]);
                    switch (e.op)
                    {
                        case Tok::Plus: return a + " + " + b;
                        case Tok::Minus: return a + " - " + b;
                        case Tok::Cdot: return a + tok(Tok::Cdot) + b;
                        case Tok::Slash: return a + "/" + b;
                        case Tok::Otimes: return a + " " + tok(Tok::Otimes) + " " + b;
                        case Tok::KwAnd: return a + " and " + b;
                        case Tok::KwOr: return a + " or " + b;
                        default: break;
                    }
                    return a + " " + tok(e.op) + " " + b;
                }
                case EK::Product:
                {
                    // Spacing depends only on the tree, so every surface form formats alike: a ket
                    // hugs the factor before it (cos(θ/2)|0⟩), and atoms stay tight when they
                    // re-lex unchanged (2π, 2x, iφ, but x y, i ρ, 2 e).
                    std::string out, prevText;
                    for (std::size_t k = 0; k < e.kids.size(); ++k)
                    {
                        const Expr& cur = *e.kids[k];
                        const std::string f = expr(cur);
                        if (k)
                        {
                            const EK prev = e.kids[k - 1]->kind;
                            const bool dirac = prev == EK::Ket || prev == EK::Bra || prev == EK::Braket;
                            // The factor's leftmost leaf: x in x, x^2, x·y, x/2, x[k].
                            const Expr* lead = &cur;
                            while ((lead->kind == EK::Power || lead->kind == EK::Index ||
                                    (lead->kind == EK::Binary && (lead->op == Tok::Cdot || lead->op == Tok::Slash))) &&
                                   !lead->kids.empty())
                                lead = lead->kids[0].get();
                            const bool styled = (cur.kind == EK::Ket && !dirac) ||
                                                ((prev == EK::Int || prev == EK::Real || prev == EK::Name) && lead->kind == EK::Name);
                            if (!styled || !joinsCleanly(prevText, f)) out += ' ';
                        }
                        out += f;
                        prevText = f;
                    }
                    return out;
                }
                case EK::Compare:
                {
                    std::string out = expr(*e.kids[0]) + " " + tok(e.op) + " " + expr(*e.kids[1]);
                    if (e.kids.size() > 2) out += " " + tok(Tok::PlusMinus) + " " + expr(*e.kids[2]);
                    return out;
                }
                case EK::Sub: return expr(*e.kids[0]) + qlist(e.qlist);
                case EK::Call:
                {
                    std::string out = expr(*e.kids[0]) + "(";
                    for (std::size_t k = 1; k < e.kids.size(); ++k)
                    {
                        if (k > 1) out += ", ";
                        if (!e.argNames[k - 1].empty()) out += e.argNames[k - 1] + "=";
                        out += expr(*e.kids[k]);
                    }
                    return out + ")";
                }
                case EK::Index: return expr(*e.kids[0]) + "[" + expr(*e.kids[1]) + "]";
                case EK::Slice: return expr(*e.kids[0]) + ".." + expr(*e.kids[1]);
                case EK::Dagger: return expr(*e.kids[0]) + tok(Tok::Dagger);
                case EK::Power: return powerBase(*e.kids[0]) + "^" + exponent(*e.kids[1]);
                case EK::TensorPow:
                    if (ascii)
                    {
                        const Expr& x = *e.kids[1];
                        return expr(*e.kids[0]) + "^{\\otimes " + (x.kind == EK::Paren ? expr(*x.kids[0]) : expr(x)) + "}";
                    }
                    return expr(*e.kids[0]) + "^⊗" + exponent(*e.kids[1]);
                case EK::BigOp:
                    // Both bounds always braced: Σ_{j=0}^{n - 1}.
                    return tok(e.op) + "_{" + name(e.name) + "=" + expr(*e.kids[0]) + "}^{" + expr(*e.kids[1]) + "} " + expr(*e.kids[2]);
                case EK::Expval: return tok(Tok::LAngle) + expr(*e.kids[0]) + tok(Tok::RAngle);
                case EK::Abs: return "|" + expr(*e.kids[0]) + "|";
            }
            return {};
        }

        bool ascii;
    };


    class ProgramPrinter
    {
        public:
        ProgramPrinter(const Program& p, const SourceFile& f, bool asciiMode) : prog(p), file(f), pr(asciiMode) {}

        std::string run()
        {
            out = (pr.ascii ? "noether " : "noether ") + prog.version + "\n";
            block(prog.stmts, 0, true);
            for (const Comment& c : prog.trailingComments)
            {
                if (c.blankBefore) blank();
                out += comment(c) + "\n";
            }
            return out;
        }

        private:
        std::uint32_t line(std::uint32_t off) const { return file.lineCol(off).line; }

        void blank()
        {
            if (out.size() >= 2 && out.ends_with("\n\n")) return;
            out += "\n";
        }

        static std::string comment(const Comment& c) { return (c.doc ? "##" : "#") + c.text; }

        static std::string trailing(const Stmt& s)
        {
            std::string t;
            for (const Comment& c : s.comments.trailing) t += "  " + comment(c);
            return t;
        }

        static bool compound(const Stmt& s)
        {
            return s.kind == SK::For || s.kind == SK::If || s.kind == SK::Def || s.kind == SK::Proc || s.kind == SK::Noise;
        }

        void leading(const Stmt& s, const std::string& ind, bool first)
        {
            for (const Comment& c : s.comments.leading)
            {
                if (c.blankBefore && !first) blank();
                out += ind + comment(c) + "\n";
                first = false;
            }
            if (s.comments.blankBefore && !first) blank();
        }

        void block(const Block& b, int depth, bool top)
        {
            const std::string ind(static_cast<std::size_t>(depth) * 4, ' ');
            for (std::size_t k = 0; k < b.size(); ++k)
            {
                const Stmt& s = *b[k];
                leading(s, ind, k == 0 && !top);
                if (compound(s))
                {
                    compoundStmt(s, depth);
                    continue;
                }
                // Simple statements that shared a source line stay on one line.
                std::string text = ind + simple(s);
                std::string trail = trailing(s);
                while (k + 1 < b.size() && !compound(*b[k + 1]) && b[k + 1]->comments.leading.empty() &&
                       line(b[k + 1]->span.begin) == line(s.span.end))
                {
                    const Stmt& nx = *b[++k];
                    text += "; " + simple(nx);
                    trail += trailing(nx);
                    if (line(nx.span.end) != line(nx.span.begin)) break;
                }
                out += text + trail + "\n";
            }
        }

        // A suite written on the head's line stays there: `def Bell_{a,b}: CNOT_{a→b} H_a`.
        void suite(const Block& body, std::uint32_t headLine, int depth, const std::string& head, const std::string& trail)
        {
            if (!body.empty() && line(body.front()->span.begin) == headLine && body.front()->comments.leading.empty() &&
                std::ranges::none_of(body, [](const StmtPtr& s) { return compound(*s); }))
            {
                std::string text = head + " ";
                std::string tr = trail;
                for (std::size_t k = 0; k < body.size(); ++k)
                {
                    text += (k ? "; " : "") + simple(*body[k]);
                    tr += trailing(*body[k]);
                }
                out += text + tr + "\n";
                return;
            }
            out += head + trail + "\n";
            block(body, depth + 1, false);
        }

        void compoundStmt(const Stmt& s, int depth)
        {
            const std::string ind(static_cast<std::size_t>(depth) * 4, ' ');
            const std::uint32_t headLine = line(s.head.end);
            switch (s.kind)
            {
                case SK::For:
                {
                    std::string head = ind + "for " + pr.name(s.name) + " " + pr.tok(Tok::In) + " ";
                    if (s.expr) head += pr.expr(*s.expr);
                    else
                    {
                        head += pr.expr(*s.lo) + ".." + pr.expr(*s.hi);
                        if (s.step) head += " by " + pr.expr(*s.step);
                    }
                    suite(s.body, headLine, depth, head + ":", trailing(s));
                    return;
                }
                case SK::If:
                {
                    suite(s.body, headLine, depth, ind + "if " + pr.expr(*s.expr) + ":", trailing(s));
                    if (!s.orelse.empty())
                    {
                        const std::uint32_t elseLine = line(s.orelse.front()->span.begin);
                        const std::string_view text = file.lineText(elseLine);
                        const std::size_t col = file.lineCol(s.orelse.front()->span.begin).col;
                        const bool sameLine = text.substr(0, std::min(text.size(), static_cast<std::size_t>(col))).find("else") !=
                                              std::string_view::npos;
                        suite(s.orelse, sameLine ? elseLine : elseLine + 1, depth, ind + "else:", "");
                    }
                    return;
                }
                case SK::Def:
                case SK::Proc:
                {
                    std::string head = ind + (s.kind == SK::Def ? "def " : "proc ") + pr.name(s.name);
                    if (!s.cparams.empty())
                    {
                        head += "(";
                        for (std::size_t k = 0; k < s.cparams.size(); ++k) head += (k ? ", " : "") + pr.name(s.cparams[k]);
                        head += ")";
                    }
                    head += pr.subParams(s.sub) + ":";
                    suite(s.body, headLine, depth, head, trailing(s));
                    return;
                }
                case SK::Noise:
                {
                    out += ind + "noise:" + trailing(s) + "\n";
                    for (const NoiseRule& r : s.rules)
                        out += ind + "    " + (r.after ? "after " : "before ") + r.event + ": " + pr.expr(*r.channel) + "\n";
                    return;
                }
                default: break;
            }
        }

        std::string simple(const Stmt& s) const
        {
            switch (s.kind)
            {
                case SK::Qubits:
                case SK::Bits:
                {
                    std::string t = s.kind == SK::Qubits ? "qubits " : "bits ";
                    for (std::size_t k = 0; k < s.regs.size(); ++k)
                        t += (k ? ", " : "") + pr.name(s.regs[k].name) + "[" + pr.expr(*s.regs[k].size) + "]";
                    return t;
                }
                case SK::Seed: return "seed " + pr.expr(*s.expr);
                case SK::Trajectories: return "trajectories " + pr.expr(*s.expr);
                case SK::Backend: return "backend " + s.text;
                case SK::Let:
                {
                    std::string t = "let ";
                    for (std::size_t k = 0; k < s.bindings.size(); ++k)
                    {
                        const Binding& b = s.bindings[k];
                        t += (k ? ", " : "") + pr.name(b.name) + (b.hasSub ? pr.subParams(b.sub) : "") + " = " + pr.expr(*b.value);
                    }
                    return t;
                }
                case SK::Param:
                {
                    std::string t = "param " + pr.name(s.name);
                    if (s.paramSize) t += "[" + pr.expr(*s.paramSize) + "]";
                    t += " " + pr.tok(Tok::In) + " [" + pr.expr(*s.lo) + ", " + pr.expr(*s.hi) + "]";
                    if (s.init) t += " = " + pr.expr(*s.init);
                    return t;
                }
                case SK::Prepare: return "prepare " + pr.expr(*s.expr);
                case SK::Measure:
                    return pr.expr(*s.lvalue) + " " + pr.tok(Tok::LeftArrow) + " measure" +
                           (s.measureSub ? pr.qlist(s.qlist) : " " + pr.expr(*s.expr));
                case SK::Run: return pr.expr(*s.lvalue) + " " + pr.tok(Tok::LeftArrow) + " run " + pr.expr(*s.expr);
                case SK::Reset: return "reset" + pr.qlist(s.qlist);
                case SK::Print:
                {
                    std::string t = "print ";
                    for (std::size_t k = 0; k < s.items.size(); ++k)
                    {
                        t += (k ? ", " : "") + pr.expr(*s.items[k].expr);
                        if (!s.items[k].label.empty()) t += " as " + pr.name(s.items[k].label);
                    }
                    return t;
                }
                case SK::Assert: return "assert " + pr.expr(*s.expr);
                case SK::Import: return "import " + Printer::quote(s.text);
                case SK::Apply: return pr.expr(*s.expr);
                case SK::Spec: return spec(s.spec);
                default: break;
            }
            return {};
        }

        std::string gate(const std::pair<std::string, bool>& g) const
        {
            if (g.first.starts_with("√")) return pr.ascii ? "SX" : g.first;
            return g.first + (g.second ? pr.tok(Tok::Dagger) : "");
        }

        std::string spec(const SpecStmt& sp) const
        {
            const std::string& k = sp.keyword;
            auto options = [&]
            {
                std::string t;
                for (std::size_t j = 0; j < sp.options.size(); ++j)
                    t += (j ? ", " : "") + sp.options[j].first + "=" +
                         (sp.options[j].second->kind == EK::String ? sp.options[j].second->name : pr.expr(*sp.options[j].second));
                return t;
            };
            if (k == "task") return "task " + Printer::quote(sp.name);
            if (k == "candidate") return "candidate " + sp.word + " " + pr.name(sp.name) + pr.subParams(sp.sub);
            if (k == "target") return "target " + sp.word + " " + pr.expr(*sp.expr);
            if (k == "metric") return "metric " + sp.name + " = " + pr.expr(*sp.expr);
            if (k == "gates" || k == "forbid")
            {
                std::string t = k + (k == "gates" ? " {" : " ");
                for (std::size_t j = 0; j < sp.gates.size(); ++j) t += (j ? ", " : "") + gate(sp.gates[j]);
                return t + (k == "gates" ? "}" : "");
            }
            if (k == "allow") return "allow " + sp.word;
            if (k == "coupling")
            {
                if (sp.word == "set")
                {
                    std::string t = "coupling {";
                    for (std::size_t j = 0; j < sp.expr->kids.size(); ++j)
                    {
                        const Expr& pair = *sp.expr->kids[j];
                        t += (j ? ", " : "") + std::string("(") + pr.expr(*pair.kids[0]) + ", " + pr.expr(*pair.kids[1]) + ")";
                    }
                    return t + "}";
                }
                if (sp.word == "grid") return "coupling grid(" + pr.expr(*sp.expr->kids[0]) + ", " + pr.expr(*sp.expr->kids[1]) + ")";
                return "coupling " + sp.word;
            }
            if (k == "readout") return sp.word == "shots" ? "readout shots(" + pr.expr(*sp.expr) + ")" : "readout " + sp.word;
            if (k == "optimize" || k == "budget") return k + " " + options();
            if (k == "require" || k == "goal" || k == "minimize" || k == "maximize") return k + " " + pr.expr(*sp.expr);
            if (k == "instances" || k == "holdout") return k + " " + pr.name(sp.name) + " " + pr.tok(Tok::In) + " " + pr.expr(*sp.expr);
            if (k == "aggregate") return "aggregate " + sp.word;
            return k;
        }

        const Program& prog;
        const SourceFile& file;
        Printer pr;
        std::string out;
    };
} // namespace

std::string formatExpr(const Expr& e, bool ascii) { return Printer(ascii).expr(e); }

std::string formatProgram(const Program& prog, const SourceFile& file, bool ascii)
{
    return ProgramPrinter(prog, file, ascii).run();
}

} // namespace Noether
