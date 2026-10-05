#pragma once

#include "Values.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>



namespace Noether
{

// Version of the decomposition rules; recorded in spec.lock because metrics depend on it.
inline constexpr int kDecompositionRulesVersion = 2;

// Lowers IR ops into a target gate set by fixed rules. Every rule is exact up to a global phase on
// uncontrolled gates; controls are removed first with exact (phase-correct) constructions, so the
// lowered circuit implements the same unitary up to one global phase. Measurements, resets and
// channels pass through unchanged.
class BasisLowering
{
    public:
    // `gates`: canonical gate names (I X Y Z H S S† T T† √X √X† Rx Ry Rz P U3 CNOT CZ CP SWAP
    // Toffoli Fredkin). `unknown` receives names that are not gates.
    BasisLowering(const std::vector<std::string>& gates, std::vector<std::string>& unknown);

    // Appends the lowering of `op`; false with `error` set when the op has no exact form in the set.
    bool lower(const IrOp& op, std::vector<IrOp>& out, std::string& error) const;

    // Whether a gate kind (with arbitrary angles) is expressible in the set.
    bool expressible(std::string_view type) const;

    // Index of the rule that expresses a type outside the set, SIZE_MAX for types in the set.
    std::size_t chosenRule(std::string_view type) const;

    private:
    std::set<std::string, std::less<>> basis;
    std::map<std::string, std::size_t, std::less<>> chosen; // type -> index of the rule used
};

// Name of an IR op's gate type for the basis check: "Rz", "CNOT", "T†", …
std::string basisName(const IrOp& op);

} // namespace Noether
