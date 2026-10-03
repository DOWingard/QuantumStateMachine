#pragma once

#include <iostream>
#include <complex>
#include <cmath>
#include <cstddef>
#include <Eigen/Dense>


namespace Qputer
{

// 2^25 amplitudes * 16 B = 512 MiB; anything larger is refused to protect host memory.
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


    size_t num_qubits() const { return n_qubits; }

    size_t size() const; // return size of vector: 2^n_qubits

    double norm() const; // return norm of vector

    void normalize(); // normalize vector in place

};


} // namespace qstate
