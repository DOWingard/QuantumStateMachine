#pragma once

#include "Json.hpp"

#include <QuantumStateMachine.hpp>

#include <cstddef>
#include <string>
#include <vector>



namespace Noether::Interop
{

using Qputer::Outcome;
using Qputer::Qubit;
using Qputer::QubitList;

struct ClbitRegister
{
    std::string name;
    std::size_t offset = 0;
    std::size_t size = 0;
};

// The hub every importer produces: a flat list of core operations plus the labels needed to
// report results in the source framework's terms.
//
// Conventions are the core's: qubit q is bit q of an amplitude index, targets[0] is the most
// significant bit of a dense matrix's index, and clbit c is bit c of the classical register.
struct ImportedCircuit
{
    std::string format; // qasm2 | qasm3 | stim | circuit
    std::size_t numQubits = 0;
    std::size_t numClbits = 0;
    std::vector<std::string> qubitLabels; // one per qubit
    std::vector<ClbitRegister> clbitRegisters;
    std::vector<Qputer::Operation> ops;
    Json source = Json::object(); // {"framework", "version"} when the exporter recorded them

    // Stim annotations as clbit masks: a detector or observable value is the parity of its clbits.
    std::vector<Outcome> detectors;
    std::vector<Outcome> observables; // index k = OBSERVABLE_INCLUDE(k)
};

} // namespace Noether::Interop
