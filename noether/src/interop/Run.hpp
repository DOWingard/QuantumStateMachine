#pragma once

#include "interop/Circuit.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>



namespace Noether::Interop
{

struct RunOptions
{
    std::size_t shots = 1024;
    std::optional<std::uint64_t> seed;
    std::string backend = "auto"; // auto | statevector | stabilizer
    std::string emit = "counts";  // counts | probabilities | statevector
    bool sampleTerminal = true;   // false forces per-shot replay even when every measurement is terminal
};

struct RunResult
{
    std::string backend; // statevector | stabilizer
    std::string backendReason;
    std::string method;  // sampled | trajectories | state
    std::uint64_t seed = 0;
    Qputer::Counts counts;
    Eigen::VectorXcd amplitudes;    // emit statevector; index bit q = qubit q
    Eigen::VectorXd probabilities;  // emit probabilities
    // Shots in which each detector's (observable's) parity differs from the noiseless reference,
    // as Stim reports detection events and observable flips.
    std::vector<std::size_t> detectorFires;
    std::vector<std::size_t> observableFlips;
};

// Runs an imported circuit on the QuantumStateMachine.
//
// Backend under `auto`: the stabilizer tableau at any size when every operation is Clifford (an
// uncontrolled rotation by a multiple of π/2 counts, rewritten exactly up to global phase),
// otherwise the state vector. When every measurement is terminal the unitary part is simulated
// once and the shots are sampled from it; otherwise every shot replays the circuit.
//
// Throws ImportError before allocating when the circuit cannot run as asked: too many qubits for
// the state vector (E9004), a non-Clifford operation on a requested tableau (E6001), or a state
// readout of a circuit with mid-circuit measurement, conditions or channels (E9003).
RunResult runCircuit(const ImportedCircuit& circuit, const RunOptions& options);

} // namespace Noether::Interop
