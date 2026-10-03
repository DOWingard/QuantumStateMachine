#include "TestSupport.hpp"

#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace QputerTest;

// The kernels only fork threads once a gate visits >= 2^14 subspaces, which no <10-qubit
// test reaches. 17 qubits puts every 1-, 2- and 3-qubit gate over that threshold, and the
// single-threaded run of the same circuit (verified against the reference elsewhere) is
// the oracle for the parallel partitioning.
TEST(ParallelConsistency, MultiThreadedCircuitMatchesSingleThreaded)
{
#ifndef _OPENMP
    GTEST_SKIP() << "built without OpenMP; kernels are single-threaded";
#else
    constexpr std::size_t n = 17;
    std::mt19937_64 rng{31};

    std::vector<GateOp> circuit;
    for (int k = 0; k < 60; ++k) circuit.push_back(randomGate(n, rng));
    const Eigen::VectorXcd psi = randomAmplitudes(n, rng);

    auto runWith = [&](int threads)
    {
        const int saved = omp_get_max_threads();
        omp_set_num_threads(threads);
        QuantumStateVector s = fromAmplitudes(n, psi);
        for (const auto& g : circuit) g(s);
        omp_set_num_threads(saved);
        return s;
    };

    const QuantumStateVector serial = runWith(1);
    for (const int threads : {2, 3, std::max(4, omp_get_num_procs())})
    {
        SCOPED_TRACE(std::format("threads = {}", threads));
        const QuantumStateVector parallel = runWith(threads);
        EXPECT_TRUE(statesNear(parallel, serial));
    }
    EXPECT_NEAR(serial.norm(), 1.0, 1e-11);
#endif
}
