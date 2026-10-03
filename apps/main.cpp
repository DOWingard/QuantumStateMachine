#include <QuantumState.hpp>


int main()
{

    Qputer::QuantumStateVector ket{5};

    ket[0] = {1,0};
    ket[1] = {0,1};


   std::cout << ket[0].real() << " + " << ket[1].imag() << "i\n";

   std::cout << ket.norm() << "\n";

   ket.normalize();

   std::cout << ket.norm() << "\n";

   
}