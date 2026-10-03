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
    const Eigen::VectorXcd& vector() const { return ket; }
    std::complex<double>& operator[](size_t i) { return ket[i]; }
    const std::complex<double>& operator[](size_t i) const { return ket[i]; }


    explicit QuantumStateVector(size_t n) : n_qubits(pow(2,n)), ket(Eigen::VectorXcd::Zero(pow(2,n))) {};


    size_t size();

    float norm();

    void normalize();

};


} // namespace qstate