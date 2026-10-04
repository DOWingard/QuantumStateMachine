#include <QuantumState.hpp>

#include <bit>
#include <format>
#include <stdexcept>
#include <Eigen/Dense>



namespace Qputer
{


    QuantumStateVector::QuantumStateVector(size_t n) : n_qubits(n)
    {
        if (n == 0 || n > kMaxQubits)
        {
            throw std::length_error(std::format("num_qubits={} outside [1, {}]", n, kMaxQubits));
        }
        ket = Eigen::VectorXcd::Zero(Eigen::Index{1} << n);
        ket[0] = 1.0;
    }

    QuantumStateVector::QuantumStateVector(const Eigen::VectorXcd& amplitudes) : n_qubits(0), ket(amplitudes)
    {
        const auto len = static_cast<std::size_t>(amplitudes.size());
        if (!std::has_single_bit(len) || len < 2 || len > (std::size_t{1} << kMaxQubits))
        {
            throw std::length_error(std::format("amplitude count {} is not 2^n for n in [1, {}]", len, kMaxQubits));
        }
        n_qubits = static_cast<std::size_t>(std::countr_zero(len));
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
