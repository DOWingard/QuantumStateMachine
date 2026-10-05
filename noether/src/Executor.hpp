#pragma once

#include "Compiler.hpp"
#include "Json.hpp"

#include <QuantumStateMachine.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>



namespace Noether
{

inline constexpr std::string_view kVersion = "0.1.0";
inline constexpr std::string_view kLangVersion = "0.1";

struct ExecOptions
{
    bool timing = true;
    std::optional<double> timeoutSeconds;
    std::uint64_t maxMemBytes = std::uint64_t{8} << 30;
    std::size_t top = 0;                       // 0 keeps every count and amplitude
    std::optional<std::vector<double>> params; // rebinding (opt, grad, eval); defaults otherwise
    // Parameter-shift support: adds `delta` to angle `angle` of the op at event index `event`.
    struct Shift
    {
        std::size_t event = 0;
        std::size_t angle = 0;
        double delta = 0.0;
    };
    std::optional<Shift> shift;
    bool stopOnAssert = true;
    // Estimate ⟨A⟩ and basis probabilities from this many measurement shots instead of reading the
    // state exactly (research `readout shots(n)`); other readouts are then E8006.
    std::optional<std::uint64_t> readoutShots;
};

struct PrintRecord
{
    Span span;
    std::string label;
    std::string text;
    Json value;
    std::optional<double> stderrValue;
    std::optional<double> number; // real value, when the readout is a real number
};

struct AssertRecord
{
    Span span;
    std::string text;
    bool passed = false;
    Json lhs, rhs;
    std::optional<double> tol;
    std::optional<double> stderrValue;
};

struct RunRecord
{
    Span span;
    std::string name;
    std::uint64_t shots = 0;
    std::map<std::string, std::uint64_t> counts;
};

struct ExecResult
{
    int exitCode = 0;         // 0 ok, 3 assertion failed, 4 resource limit, 5 internal error
    std::uint64_t seed = 0;
    std::vector<PrintRecord> prints;
    std::vector<AssertRecord> asserts;
    std::vector<RunRecord> runs;
    double executeMs = 0.0;
    std::string internalError;
};

// Runs a compiled program on the QuantumStateMachine: the live pass applies every op, evaluates
// prints, asserts and lets against the live state, and replays the record for each `run`.
// Runtime failures are reported to the compiler's diagnostics.
ExecResult execute(Compiler& compiler, const ExecOptions& options);

// Common header of every JSON document: schema, version, lang, ok, diagnostics.
Json documentHeader(std::string_view kind, bool ok, const Diagnostics& diags);

// The `noether.run/1` document.
Json runJson(Compiler& compiler, const ExecResult& r, std::optional<double> compileMs, const ExecOptions& options);

// Human-readable results: one line per print, assert and run.
std::string runText(Compiler& compiler, const ExecResult& r, const ExecOptions& options);

// Value of a run-time readout as JSON: numbers, [re, im], bit strings, counts, kets, matrices.
Json valueJson(const Value& v, std::uint64_t creg, std::size_t top);

// Counts keys: every clbit in declaration order, c0 leftmost, '|' between registers.
std::string countsKey(const Ir& ir, Qputer::Outcome value);

} // namespace Noether
