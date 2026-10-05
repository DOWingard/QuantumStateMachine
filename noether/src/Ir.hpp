#pragma once

#include "Ast.hpp"
#include "Values.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>



namespace Noether
{

// A name in scope.
struct Symbol
{
    enum class Kind : std::uint8_t { Let, LetGate, Param, Def, Proc, Loop, CParam, QParam, QReg, BReg, Runtime };
    Kind kind = Kind::Let;
    Value value;
    const Stmt* stmt = nullptr;       // Def / Proc / LetGate: the defining statement
    const Binding* binding = nullptr; // LetGate
    Span span;
    int slot = -1;                    // Runtime: run-time slot holding the value
    bool global = false;
    mutable bool used = false;
};

// Lexical scope. Def and proc bodies open a hygienic boundary: past it only lets, params, defs and
// procs are visible (plus bit registers for procs), and only those defined before the def.
class Scope : public std::enable_shared_from_this<Scope>
{
    public:
    enum class Boundary : std::uint8_t { None, Def, Proc };

    // `def`: span of the def/proc statement opening a boundary; globals defined after it in the same
    // file are invisible inside.
    explicit Scope(std::shared_ptr<const Scope> parentScope = nullptr, Boundary b = Boundary::None, Span def = {})
        : parent(std::move(parentScope)), boundary(b), defSpan(def)
    {
    }

    // The symbol and the scope that holds it, honouring def/proc hygiene; {nullptr, nullptr} if unbound.
    std::pair<const Symbol*, const Scope*> find(std::string_view name) const;
    const Symbol* lookup(std::string_view name) const { return find(name).first; }
    const Symbol* localLookup(std::string_view name) const;
    Symbol& define(const std::string& name, Symbol s);
    bool isGlobal() const { return parent == nullptr; }
    const std::map<std::string, Symbol, std::less<>>& symbols() const { return names; }
    std::map<std::string, Symbol, std::less<>>& symbols() { return names; }
    const Scope* parentScope() const { return parent.get(); }

    private:
    std::shared_ptr<const Scope> parent;
    Boundary boundary;
    Span defSpan;
    std::map<std::string, Symbol, std::less<>> names;
};

using ScopePtr = std::shared_ptr<Scope>;
using ConstScopePtr = std::shared_ptr<const Scope>;


struct RegInfo
{
    std::string name;
    std::size_t offset = 0;
    std::size_t size = 0;
    Span span;
};

struct ParamInfo
{
    std::string name;
    std::size_t size = 1;
    bool isVector = false;
    std::uint32_t base = 0; // index of element 0 in the flat param vector
    double lo = 0.0, hi = 0.0;
    std::vector<double> init;
    Span span;
    std::string doc;
};

// A readout seen at compile time: its kind decides backend support and cost.
struct ReadoutUse
{
    Span span;
    std::string kind; // pauli, probability, amplitudes, overlap, matrix-element, density, entropy, dense-observable, fidelity, bits, counts
    bool statevectorOnly = false;
    std::size_t terms = 1;  // Pauli terms
    std::size_t width = 0;  // qubits involved (entropy / density / dense observable)
};

enum class EvK : std::uint8_t { Op, Prepare, Print, Assert, Run, Let };

struct PrintItemIr
{
    ExprPtr expr;
    std::string label;
    std::string text;
    Span span;
};

struct Event
{
    EvK kind = EvK::Op;
    Span span;
    IrOp op;                          // Op
    std::shared_ptr<const KetV> ket;  // Prepare
    std::vector<PrintItemIr> items;   // Print
    ExprPtr expr;                     // Assert, Let
    std::string text;                 // Assert source text
    ConstScopePtr scope;              // Print, Assert, Let: names visible to the expression
    std::uint64_t shots = 0;          // Run
    int slot = -1;                    // Run, Let
    std::string name;                 // Run, Let
};

struct Ir
{
    std::size_t nQubits = 0;
    std::size_t nClbits = 0;
    std::vector<RegInfo> qregs, bregs;
    std::vector<ParamInfo> params;
    std::vector<std::string> paramNames; // flat: θ, φ[0], φ[1], …
    std::vector<double> paramValues;     // defaults, then --set overrides
    std::vector<Event> events;
    std::optional<std::uint64_t> seed;
    std::string backendRequest = "auto";
    std::size_t trajectories = 0;
    bool hasNoise = false;
    std::vector<ReadoutUse> readouts;
    int slots = 0;

    // Filled by selectBackend().
    std::string backend;       // statevector | stabilizer
    std::string backendReason;
    std::string sourceSha256;
};

} // namespace Noether
