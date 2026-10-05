#pragma once

// Internal state shared by Compiler.cpp (statements, emission, imports) and Eval.cpp
// (expressions, operator algebra, gates, readouts). Not part of the public interface.

#include "Compiler.hpp"

#include <set>



namespace Noether
{

enum class Ctx : std::uint8_t { Top, Def, Proc };

// Where emitted operations go: top-level events, or the op list of a def/proc body being expanded.
struct Sink
{
    Ctx ctx = Ctx::Top;
    std::vector<IrOp> ops;
    std::shared_ptr<const CondNode> guard; // conjunction of enclosing run-time if conditions
};

struct NoiseRuleIr
{
    bool after = true;
    std::string event;
    GateV channel;
    Span span;
};

// One evaluated subscript item: the qubits it names.
struct QArg
{
    QubitList q;
    bool single = false;
    bool negated = false;
    Span span;
};

struct QArgs
{
    std::vector<QArg> items;
    std::vector<Tok> seps;
    Span span;
};

struct Compiler::Impl
{
    Impl(Compiler& self) : owner(self), sm(self.sm), d(self.d), ir(self.irv) {}

    Compiler& owner;
    SourceManager& sm;
    Diagnostics& d;
    Ir& ir;

    RuntimeHooks* rt = nullptr;      // set while evaluating at run time
    bool readoutContext = false;     // compiling a print / assert / run-stage let
    std::vector<Ctx> ctx{Ctx::Top};
    std::vector<const Stmt*> expanding; // defs being expanded (recursion check)
    std::map<std::string, Span, std::less<>> laterNames; // every top-level name of the main file
    std::vector<NoiseRuleIr> noise;
    bool measuredSincePrepare = false;
    bool noiseEmitted = false;
    std::vector<Span> noisyReadouts; // readouts after noise, for W0005 at finish()
    std::vector<std::unique_ptr<Program>> imported;
    std::set<std::string> importedPaths;
    std::vector<std::string> fileDirs; // directory of each file id, for relative imports
    bool seedSet = false, backendSet = false, trajSet = false;

    // ---- errors ----
    [[noreturn]] void fail(std::string code, Span s, std::string msg);
    void warn(std::string code, Span s, std::string msg);

    // ---- statements (Compiler.cpp) ----
    void block(const Block& b, const ScopePtr& sc, Sink& sink);
    void stmt(const Stmt& s, const ScopePtr& sc, Sink& sink);
    void letStmt(const Stmt& s, const ScopePtr& sc, Sink& sink);
    void paramStmt(const Stmt& s, const ScopePtr& sc);
    void regStmt(const Stmt& s, const ScopePtr& sc);
    void prepareStmt(const Stmt& s, const ScopePtr& sc);
    void measureStmt(const Stmt& s, const ScopePtr& sc, Sink& sink);
    void runStmt(const Stmt& s, const ScopePtr& sc);
    void printStmt(const Stmt& s, const ScopePtr& sc);
    void assertStmt(const Stmt& s, const ScopePtr& sc);
    void forStmt(const Stmt& s, const ScopePtr& sc, Sink& sink);
    void ifStmt(const Stmt& s, const ScopePtr& sc, Sink& sink);
    void defStmt(const Stmt& s, const ScopePtr& sc);
    void noiseStmt(const Stmt& s, const ScopePtr& sc);
    void importStmt(const Stmt& s, const ScopePtr& sc);
    void applyStmt(const Stmt& s, const ScopePtr& sc, Sink& sink);
    void specStmt(const Stmt& s, const ScopePtr& sc);

    void emit(Sink& sink, IrOp op);
    void emitOps(Sink& sink, const OpV& op, Span at);
    void emitTop(IrOp op);
    void insertNoise(const IrOp& op, bool after);
    Symbol& define(const ScopePtr& sc, const std::string& name, Symbol s, Span at);
    void checkBindable(const std::string& name, Span at);
    int newSlot();

    // ---- expressions (Eval.cpp) ----
    Value eval(const Expr& e, const ConstScopePtr& sc);
    Value lookup(const Expr& e, const ConstScopePtr& sc);
    [[noreturn]] void unknownName(const std::string& name, Span at, const ConstScopePtr& sc);
    Value literal(const Expr& e);
    Value applySub(const Expr& e, const ConstScopePtr& sc);
    Value call(const Expr& e, const ConstScopePtr& sc);
    Value callFunction(const std::string& name, const Expr& e, const ConstScopePtr& sc);
    Value index(const Expr& e, const ConstScopePtr& sc);
    Value power(const Expr& e, const ConstScopePtr& sc);
    Value bigop(const Expr& e, const ConstScopePtr& sc);
    Value expval(const Expr& e, const ConstScopePtr& sc);
    Value absValue(const Expr& e, const ConstScopePtr& sc);
    Value compare(const Expr& e, const ConstScopePtr& sc);
    Value logic(const Expr& e, const ConstScopePtr& sc);
    Value product(const Expr& e, const ConstScopePtr& sc);
    Value ketLabel(const std::string& label, Span at, const ConstScopePtr& sc);

    // algebra
    Value add(const Value& a, const Value& b, Span at);
    Value neg(const Value& a, Span at);
    Value mul(const Value& a, const Value& b, Span at);
    Value divide(const Value& a, const Value& b, Span at);
    Value tensor(const Value& a, const Value& b, Span at);
    Value dagger(const Value& a, Span at);
    Value raise(const Value& base, const Value& exp, Span at);
    Value sqrtValue(const Value& a, Span at);
    Value expValue(const Value& a, Span at);
    Value notValue(const Value& a, Span at);

    // conversions
    std::int64_t needInt(const Value& v, Span at, std::string_view what);
    Affine needAngle(const Value& v, Span at, std::string_view what);
    double needReal(const Value& v, Span at, std::string_view what);
    Num needNum(const Value& v, Span at, std::string_view what);
    bool isOperator(const Value& v) const;
    LinOpV toLin(const Value& v, Span at);
    OpV toOp(const Value& v, Span at);
    MatV toMatrix(const Value& v, Span at);
    std::optional<PTerm> asPauli(const OpV& op) const;
    std::shared_ptr<const CondNode> toCond(const Value& v, Span at);
    Eigen::MatrixXcd denseOf(const LinOpV& op, QubitList& support, Span at);
    Eigen::MatrixXcd denseOf(const OpV& op, const QubitList& support, Span at);

    // gates
    QArgs qargs(const QList& ql, const ConstScopePtr& sc, Span at);
    bool oneQubitGate(const GateV& g) const;
    OpV applyGate(const GateV& g, const QArgs& qa, Span at);
    OpV applyBuiltin(const GateV& g, const QArgs& qa, Span at);
    OpV expandDef(const GateV& g, const QArgs& qa, Span at);
    OpV applyChannel(const GateV& g, const QArgs& qa, Span at);
    OpV daggerOp(const OpV& op, Span at);
    OpV powerOp(const OpV& op, const Num& n, Span at);
    OpV controlled(const ControlV& cv, const OpV& op, Span at);
    OpV expOperator(const LinOpV& a, Span at);
    OpV trotter(const LinOpV& ham, const Num& t, std::int64_t steps, std::int64_t order, Span at);
    IrOp pauliRotation(const PTerm& term, const Affine& angle, Span at);
    void checkQubit(Qubit q, Span at);
    void checkDistinct(const QubitList& q, Span at);

    // readouts
    void readout(std::string kind, bool svOnly, Span at, std::size_t terms = 1, std::size_t width = 0);
    Value liveKet(Span at);
    Value runtimeValue(const Symbol& sym, const Scope& holder);
    void noteStateReadout(const Value& v, Span at);
    Value probabilityOf(const std::string& bits, Span at);
};

std::vector<Cube> toCubes(const std::shared_ptr<const CondNode>& f, std::size_t& count);
std::string formatExpr(const Expr& e, bool ascii);

} // namespace Noether
