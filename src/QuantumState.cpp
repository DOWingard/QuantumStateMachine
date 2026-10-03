#include <QuantumState.hpp>

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
