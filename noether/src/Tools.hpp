#pragma once

// Commands built on a compiled program: IR and circuit views, equivalence, parameter
// optimisation and gradients, OpenQASM export, autoresearch, agent skills and the REPL.

#include "Compiler.hpp"
#include "Executor.hpp"
#include "Json.hpp"

#include <functional>
#include <iosfwd>
#include <string>



namespace Noether
{

// ---- IR and circuit diagrams ----
Json irJson(Compiler& compiler);           // ops and events of `noether.ir/1`
std::string irText(Compiler& compiler);
Json drawJson(const Ir& ir);               // layers [[{op, qubits}]]
std::string drawText(const Ir& ir, bool ascii);

// ---- Equivalence of the unitary parts of two programs ----
struct EquivResult
{
    bool equivalent = false;
    std::string method;          // unitary | tableau | random-states
    double maxDeviation = 0.0;
    std::string counterexample;  // input state, q0 leftmost, when not equivalent
    std::string error;
};
EquivResult equivalent(const Ir& a, const Ir& b, bool ignorePhase);

// ---- Optimisation and gradients over params ----
struct OptOptions
{
    std::string label;
    bool maximize = false;
    std::string method = "nelder-mead"; // nelder-mead | spsa | adam
    std::size_t restarts = 1;
    std::size_t iterations = 400;
    ExecOptions exec;
};
struct OptResult
{
    double best = 0.0;
    std::vector<double> params;
    Json paramsJson;
    std::size_t evaluations = 0;
    std::string error;
};
OptResult optimizeParams(Compiler& compiler, const OptOptions& options);

struct GradResult
{
    double value = 0.0;
    std::vector<double> gradient;
    Json gradientJson;
    std::string method;
    std::string error;
};
GradResult gradientOf(Compiler& compiler, const std::string& label, const ExecOptions& exec);

// Bounded minimisation of a black-box function: nelder-mead, spsa, adam or lbfgs (the gradient
// methods use central differences). Restart 0 starts from x0, later restarts from seeded uniform
// points in the box. `stop` is polled between evaluations (time budgets).
struct MinimizeResult
{
    std::vector<double> x;
    double value = 0.0;
    std::size_t evaluations = 0;
};
MinimizeResult minimizeBlackBox(const std::function<double(const std::vector<double>&)>& f, const std::vector<double>& x0,
                                const std::vector<double>& lo, const std::vector<double>& hi, const std::string& method,
                                std::size_t iterations, std::size_t restarts, std::uint64_t seed,
                                const std::function<bool()>& stop = {});

// Value of the print labelled `label` for the given params; nullopt when absent or not real.
std::optional<double> labelledValue(Compiler& compiler, const std::string& label, const ExecOptions& exec, std::string& error);

// ---- OpenQASM 3 ----
std::string qasmExport(const Ir& ir, std::string& error);

// ---- Autoresearch ----
struct ResearchOptions
{
    ExecOptions exec;
    std::string dir;       // workspace (research/<tag>); eval reads spec.ntr and algo.ntr from it
    std::string spec;      // task spec file (eval without a workspace, init)
    std::string candidate; // candidate file (eval without a workspace)
    std::string from;      // init: initial candidate to copy
    std::string holdout;   // report: spec file whose `holdout` replaces the workspace's
    std::optional<std::string> record; // eval: append a ledger row with this description
    bool noFormat = false;
};
struct ResearchOutcome
{
    int exitCode = 0;
    Json json;
    std::string text;
};
ResearchOutcome evaluateCommand(const ResearchOptions& options);
ResearchOutcome checkCandidate(const ResearchOptions& options);
ResearchOutcome researchInit(const std::string& tag, const ResearchOptions& options);
ResearchOutcome researchStatus(const ResearchOptions& options);
ResearchOutcome researchReport(const ResearchOptions& options);

// ---- Agent skills ----
std::string skillText(bool full);
std::size_t installSkills(const std::string& dir, std::string& error);

// ---- REPL ----
int runRepl(std::istream& in, std::ostream& out, std::ostream& err, const CompileOptions& options);

} // namespace Noether
