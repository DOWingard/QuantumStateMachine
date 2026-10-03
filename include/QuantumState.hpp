#pragma once

#include <iostream>
#include <complex>
#include <cmath>
#include <Eigen/Dense>


namespace Qputer
{

class QuantumStateVector
{
     
    private:
    Eigen::VectorXcd ket;
    int n_qubits;
    

    public:
    std::complex<double>& operator[](size_t i) { return ket[i]; }
    const std::complex<double>& operator[](size_t i) const { return ket[i]; }


    explicit QuantumStateVector(size_t n) : n_qubits(pow(2,n)), ket(Eigen::VectorXcd::Zero(pow(2,n))) {};


    size_t size()
    {
        return ket.size();
    }

    float norm()
    {
        float sum = 0;
        for (int i = 0; i < n_qubits; i++)
        {
            sum += std::norm(ket[i]);
        };
        return std::sqrt(sum);
    }

    void normalize()
    {
        ket.normalize();
    }
    

};


} // namespace qstate