#pragma once

#include <iostream>
#include <complex>
#include <cmath>
#include <cstddef>
#include <Eigen/Dense>


namespace Qputer
{

// 2^25 amplitudes * 16 B = 512 MiB; Raise as needed. QuantumStateMachine's Backend::Auto
// uses the stabilizer tableau (Clifford circuits only) above this size.
inline constexpr std::size_t kMaxQubits = 25;

class QuantumStateVector
{

    private:
    std::size_t n_qubits;
    Eigen::VectorXcd ket;


    public:
    const Eigen::VectorXcd& vector() const { return ket; } // read only reference
    std::complex<double>& operator[](size_t i) { return ket[static_cast<Eigen::Index>(i)]; }
    const std::complex<double>& operator[](size_t i) const { return ket[static_cast<Eigen::Index>(i)]; }

    // Contiguous amplitude storage; amplitude index bit q is the state of qubit q.
    std::complex<double>* data() { return ket.data(); }
    const std::complex<double>* data() const { return ket.data(); }


    explicit QuantumStateVector(size_t n); // |0...0> on n qubits, 1 <= n <= kMaxQubits

    // Copies amplitudes as given (not renormalized); size must be 2^n with 1 <= n <= kMaxQubits.
    explicit QuantumStateVector(const Eigen::VectorXcd& amplitudes);

    // Construction and copies write the amplitudes in parallel with the same static partition as
    // the readout kernels, so on multi-socket hosts each page is first touched, and therefore
    // placed, on the NUMA node of the thread that later streams it.
    QuantumStateVector(const QuantumStateVector& other);
    QuantumStateVector& operator=(const QuantumStateVector& other);
    QuantumStateVector(QuantumStateVector&&) noexcept = default;
    QuantumStateVector& operator=(QuantumStateVector&&) noexcept = default;
    ~QuantumStateVector() = default;


    size_t num_qubits() const { return n_qubits; }

    size_t size() const; // return size of vector: 2^n_qubits

    double norm() const; // return norm of vector

    void normalize(); // normalize vector in place

};


} // namespace qstate
