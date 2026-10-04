#include <QuantumState.hpp>

#include "Parallel.hpp"

#include <bit>
#include <format>
#include <stdexcept>
#include <Eigen/Dense>



namespace Qputer
{

namespace
{

    using cd = std::complex<double>;

    // Eigen allocates without initializing; these loops are the first touch of every page.
    void fillParallel(cd* dst, std::size_t len, cd value)
    {
        QPUTER_OMP(parallel for schedule(static) if(len >= detail::kParallelThreshold))
        for (std::size_t i = 0; i < len; ++i) dst[i] = value;
    }

    void copyParallel(const cd* src, cd* dst, std::size_t len)
    {
        QPUTER_OMP(parallel for schedule(static) if(len >= detail::kParallelThreshold))
        for (std::size_t i = 0; i < len; ++i) dst[i] = src[i];
    }

} // namespace


    QuantumStateVector::QuantumStateVector(size_t n) : n_qubits(n)
    {
        if (n == 0 || n > kMaxQubits)
        {
            throw std::length_error(std::format("num_qubits={} outside [1, {}]", n, kMaxQubits));
        }
        ket.resize(Eigen::Index{1} << n);
        fillParallel(ket.data(), size(), cd{0.0, 0.0});
        ket[0] = 1.0;
    }

    QuantumStateVector::QuantumStateVector(const Eigen::VectorXcd& amplitudes) : n_qubits(0)
    {
        const auto len = static_cast<std::size_t>(amplitudes.size());
        if (!std::has_single_bit(len) || len < 2 || len > (std::size_t{1} << kMaxQubits))
        {
            throw std::length_error(std::format("amplitude count {} is not 2^n for n in [1, {}]", len, kMaxQubits));
        }
        n_qubits = static_cast<std::size_t>(std::countr_zero(len));
        ket.resize(amplitudes.size());
        copyParallel(amplitudes.data(), ket.data(), len);
    }

    QuantumStateVector::QuantumStateVector(const QuantumStateVector& other) : n_qubits(other.n_qubits)
    {
        ket.resize(other.ket.size());
        copyParallel(other.data(), ket.data(), size());
    }

    QuantumStateVector& QuantumStateVector::operator=(const QuantumStateVector& other)
    {
        if (this == &other) return *this;
        if (ket.size() != other.ket.size())
        {
            // Release first: reallocating in place would briefly hold both buffers.
            ket.resize(0);
            ket.resize(other.ket.size());
        }
        n_qubits = other.n_qubits;
        copyParallel(other.data(), ket.data(), size());
        return *this;
    }

    size_t QuantumStateVector::size() const
    {
        return static_cast<size_t>(ket.size());
    }

    double QuantumStateVector::norm() const
    {
        return ket.norm();
    }

    void QuantumStateVector::normalize()
    {
        ket.normalize();
    }


} // namespace qstate
