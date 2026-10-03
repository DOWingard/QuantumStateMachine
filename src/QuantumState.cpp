#include <QuantumState.hpp>

#include <Eigen/Dense>



namespace Qputer
{


    size_t QuantumStateVector::size()
    {
        return ket.size();
    }

    float QuantumStateVector::norm()
    {
        float sum = 0;
        for (int i = 0; i < n_qubits; i++)
        {
            sum += std::norm(ket[i]);
        };
        return std::sqrt(sum);
    }

    void QuantumStateVector::normalize()
    {
        ket.normalize();
    }


} // namespace qstate