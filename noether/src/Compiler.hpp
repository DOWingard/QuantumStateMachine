#pragma once

#include "Ast.hpp"
#include "Diagnostics.hpp"
#include "Ir.hpp"
#include "Source.hpp"
#include "Values.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>



namespace Noether
{

struct CompileOptions
{
    std::vector<std::pair<std::string, std::string>> sets; // --set name=value (params)
    std::optional<std::string> backend;                    // --backend overrides the file
    std::optional<std::uint64_t> seed;                     // --seed overrides the file
    bool definitionsOnly = false; // candidate files: only header, import, let, param, def, proc (E8002)
    std::map<std::string, std::string> lets; // values replacing top-level lets of the main file (research instances)
};

// Run-time services for readouts; implemented by the executor over the live QuantumStateMachine.
class RuntimeHooks
{
    public:
    virtual ~RuntimeHooks() = default;
    virtual double param(std::uint32_t index) const = 0;
    virtual const Value& slot(int index) const = 0;
    virtual std::uint64_t creg() const = 0;
    virtual cd expectation(const LinOpV& op, Span at) = 0; // ⟨ψ|A|ψ⟩
    virtual cd overlap(const KetV& bra, Span at) = 0;      // ⟨φ|ψ⟩
    virtual double probability(const std::string& bits, Span at) = 0;
    virtual double entropy(const QubitList& q, Span at) = 0;
    virtual Eigen::MatrixXcd density(const QubitList& q, Span at) = 0;
    virtual KetV liveKet(Span at) = 0;
    virtual double metric(std::string_view name, std::optional<GK> gate) = 0; // count(G), depth()
};

// Raised by run-time evaluation (never by compilation): message and span become a diagnostic.
struct RuntimeError
{
    std::string code;
    std::string message;
    Span span;
};

// Spec statements collected from a task file, with their scope, for the research layer.
struct SpecInfo
{
    std::vector<const SpecStmt*> stmts;
    std::vector<Span> spans;
    ConstScopePtr scope;
};

class Compiler
{
    public:
    Compiler(SourceManager& sources, Diagnostics& diags, CompileOptions options = {});
    ~Compiler();
    Compiler(const Compiler&) = delete;
    Compiler& operator=(const Compiler&) = delete;

    // Compiles a whole program (statements in time order), then finish() selects the backend.
    void compile(const Program& prog);
    void finish();

    // Definitions only (imports, candidates): E8002 for anything else.
    void compileDefinitions(const Program& prog);

    Ir& ir() { return irv; }
    const Ir& ir() const { return irv; }
    const ScopePtr& globals() const { return global; }
    Diagnostics& diagnostics() { return d; }
    SourceManager& sources() { return sm; }
    const SpecInfo& spec() const { return specInfo; }

    // Run-time evaluation of a print / assert / let expression against the live state.
    Value evalRuntime(const Expr& e, const ConstScopePtr& scope, RuntimeHooks& hooks);

    // Applies a def/proc by name to qubits, appending top-level events (research driver).
    void applyByName(const std::string& name, const QubitList& qubits, Span at);
    // Compiles one extra statement at top level (research driver: readouts, prepare).
    void compileStatement(const Stmt& s);

    // Canonical source text of an expression, for print/assert labels.
    std::string text(const Expr& e) const;

    private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    SourceManager& sm;
    Diagnostics& d;
    CompileOptions opt;
    Ir irv;
    ScopePtr global;
    SpecInfo specInfo;
};

// Reads, parses and compiles a file (imports resolved relative to it, std/ to the bundled library).
struct Compilation
{
    std::unique_ptr<SourceManager> sources;
    std::unique_ptr<Diagnostics> diags;
    std::vector<std::unique_ptr<Program>> programs; // [0] = main file
    std::unique_ptr<Compiler> compiler;
    std::uint32_t mainFile = 0;
    bool readFailed = false;
    std::string readError;
};

std::unique_ptr<Compilation> compileFile(const std::string& path, const CompileOptions& options);
std::unique_ptr<Compilation> compileText(const std::string& path, std::string text, const CompileOptions& options);

// Parse only (fmt, draw of syntax): no semantic analysis.
std::unique_ptr<Compilation> parseFile(const std::string& path);

std::string readFileText(const std::string& path, bool& ok);

} // namespace Noether
