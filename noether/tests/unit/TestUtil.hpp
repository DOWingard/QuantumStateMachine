#pragma once

#include "Ast.hpp"
#include "Cli.hpp"
#include "Compiler.hpp"
#include "Executor.hpp"
#include "Formatter.hpp"
#include "Json.hpp"
#include "Parser.hpp"

#include <gtest/gtest.h>

#include <complex>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>



namespace NoetherTest
{

using cd = std::complex<double>;

inline const std::filesystem::path kTestDir = NOETHER_TEST_DIR;
inline const std::filesystem::path kExamplesDir = NOETHER_EXAMPLES_DIR;
inline const std::filesystem::path kSkillsDir = QSM_SKILLS_DIR;

struct CliResult
{
    int code = 0;
    std::string out;
    std::string err;
};

inline CliResult cli(const std::vector<std::string>& args)
{
    std::ostringstream out, err;
    const int code = Noether::runCli(args, out, err);
    return {code, out.str(), err.str()};
}

inline std::string readText(const std::filesystem::path& p)
{
    std::ifstream in(p, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

inline void writeText(const std::filesystem::path& p, std::string_view text)
{
    std::filesystem::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary | std::ios::trunc) << text;
}

// A fresh directory under the system temp dir, removed when the object dies.
class TempDir
{
    public:
    TempDir()
    {
        std::random_device rd;
        path = std::filesystem::temp_directory_path() / std::format("noether-test-{:x}{:x}", rd(), rd());
        std::filesystem::create_directories(path);
    }
    ~TempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    std::filesystem::path path;
};

// A compiled and executed program.
struct Executed
{
    std::unique_ptr<Noether::Compilation> c;
    Noether::ExecResult r;
    bool compiled = false;
    std::string diagnostics; // rendered, for failure messages

    const Noether::PrintRecord* print(std::string_view labelOrText) const
    {
        for (const auto& p : r.prints)
            if (p.label == labelOrText || p.text == labelOrText) return &p;
        return nullptr;
    }
    double number(std::string_view labelOrText) const
    {
        const auto* p = print(labelOrText);
        if (!p) ADD_FAILURE() << "no print " << labelOrText << "\n" << diagnostics;
        return p && p->number ? *p->number : std::nan("");
    }
    bool allAssertsPassed() const
    {
        for (const auto& a : r.asserts)
            if (!a.passed) return false;
        return true;
    }
    std::vector<std::string> codes() const
    {
        std::vector<std::string> out;
        for (const auto& d : c->diags->sorted()) out.push_back(d.code);
        return out;
    }
};

inline Executed runSource(std::string text, const Noether::CompileOptions& opt = {}, Noether::ExecOptions eo = {})
{
    Executed run;
    run.c = Noether::compileText("test.ntr", std::move(text), opt);
    run.compiled = !run.c->diags->hasErrors();
    if (run.compiled)
    {
        eo.timing = false;
        run.r = Noether::execute(*run.c->compiler, eo);
    }
    run.diagnostics = run.c->diags->render();
    return run;
}

// Compiles and runs; fails the test unless it compiles, exits 0 and every assert passes.
inline Executed expectRuns(const std::string& text, const Noether::CompileOptions& opt = {})
{
    Executed run = runSource(text, opt);
    EXPECT_TRUE(run.compiled) << text << "\n" << run.diagnostics;
    EXPECT_EQ(run.r.exitCode, 0) << text << "\n" << run.diagnostics;
    for (const auto& a : run.r.asserts) EXPECT_TRUE(a.passed) << "assert failed: " << a.text << "\nlhs " << a.lhs.dump(-1) << " rhs " << a.rhs.dump(-1) << "\n" << text;
    return run;
}

// Amplitudes of a printed ket, keyed by basis label (q0 leftmost).
inline std::map<std::string, cd> amplitudes(const Noether::PrintRecord& p)
{
    std::map<std::string, cd> out;
    const Noether::Json* a = p.value.find("amplitudes");
    if (!a) return out;
    for (const auto& [k, v] : a->asObject())
        out[k] = v.isArray() ? cd(v.asArray()[0].asDouble(), v.asArray()[1].asDouble()) : cd(v.asDouble(), 0.0);
    return out;
}

// ---- Parsing and formatting ----

struct Parsed
{
    std::unique_ptr<Noether::SourceManager> sm = std::make_unique<Noether::SourceManager>();
    std::unique_ptr<Noether::Diagnostics> d = std::make_unique<Noether::Diagnostics>(*sm);
    Noether::Program prog;
    std::uint32_t id = 0;
    bool ok() const { return !d->hasErrors(); }
};

inline Parsed parseText(const std::string& text)
{
    Parsed p;
    p.id = p.sm->add("test.ntr", text);
    p.prog = Noether::parse(p.sm->file(p.id), p.id, *p.d);
    return p;
}

inline std::string format(const std::string& text, bool ascii = false)
{
    Parsed p = parseText(text);
    EXPECT_TRUE(p.ok()) << text << "\n" << p.d->render();
    return Noether::formatProgram(p.prog, p.sm->file(p.id), ascii);
}

// Structural dump of the syntax tree without spans or comments, for AST equality.
inline void dumpExpr(const Noether::Expr* e, std::string& out);

inline void dumpQList(const Noether::QList& q, std::string& out)
{
    out += "{";
    for (std::size_t k = 0; k < q.items.size(); ++k)
    {
        if (q.items[k].negated) out += "¬";
        dumpExpr(q.items[k].expr.get(), out);
        if (k < q.seps.size()) out += std::format(" s{} ", static_cast<int>(q.seps[k]));
    }
    out += "}";
}

inline void dumpExpr(const Noether::Expr* e, std::string& out)
{
    if (!e)
    {
        out += "∅";
        return;
    }
    out += std::format("({} {} {} op{}", static_cast<int>(e->kind), e->name, e->label2, static_cast<int>(e->op));
    for (const auto& n : e->argNames) out += " arg:" + n;
    if (!e->qlist.items.empty()) dumpQList(e->qlist, out);
    for (const auto& k : e->kids)
    {
        out += " ";
        dumpExpr(k.get(), out);
    }
    out += ")";
}

inline void dumpBlock(const Noether::Block& b, std::string& out);

inline void dumpStmt(const Noether::Stmt& s, std::string& out)
{
    out += std::format("[{} {} {}", static_cast<int>(s.kind), s.text, s.name);
    dumpExpr(s.expr.get(), out);
    for (const auto& r : s.regs)
    {
        out += " reg " + r.name;
        dumpExpr(r.size.get(), out);
    }
    for (const auto& b : s.bindings)
    {
        out += " let " + b.name;
        for (const auto& p : b.sub.params) out += " " + p.name + ":" + p.sizeName;
        dumpExpr(b.value.get(), out);
    }
    for (const auto* x : {s.paramSize.get(), s.lo.get(), s.hi.get(), s.init.get(), s.lvalue.get(), s.step.get()}) dumpExpr(x, out);
    out += s.measureSub ? " msub" : "";
    dumpQList(s.qlist, out);
    for (const auto& i : s.items)
    {
        out += " item " + i.label;
        dumpExpr(i.expr.get(), out);
    }
    for (const auto& c : s.cparams) out += " c:" + c;
    for (const auto& p : s.sub.params) out += " q:" + p.name + ":" + p.sizeName;
    for (const auto& r : s.rules)
    {
        out += std::format(" rule {} {}", r.after, r.event);
        dumpExpr(r.channel.get(), out);
    }
    out += " spec " + s.spec.keyword + " " + s.spec.word + " " + s.spec.name;
    dumpExpr(s.spec.expr.get(), out);
    for (const auto& [k, v] : s.spec.options)
    {
        out += " " + k + "=";
        dumpExpr(v.get(), out);
    }
    for (const auto& [g, dag] : s.spec.gates) out += std::format(" g:{}{}", g, dag ? "†" : "");
    out += " body";
    dumpBlock(s.body, out);
    out += " else";
    dumpBlock(s.orelse, out);
    out += "]";
}

inline void dumpBlock(const Noether::Block& b, std::string& out)
{
    out += "{";
    for (const auto& s : b) dumpStmt(*s, out);
    out += "}";
}

inline std::string astDump(const std::string& text)
{
    Parsed p = parseText(text);
    EXPECT_TRUE(p.ok()) << text << "\n" << p.d->render();
    std::string out = p.prog.version;
    dumpBlock(p.prog.stmts, out);
    return out;
}

} // namespace NoetherTest
