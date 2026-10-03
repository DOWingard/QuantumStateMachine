#include "TestSupport.hpp"

#include <stdexcept>

using namespace QputerTest;


TEST(QuantumStateVector, ConstructsGroundStateOfSizeTwoToTheN)
{
    for (std::size_t n = 1; n <= 9; ++n)
    {
        SCOPED_TRACE(std::format("n = {}", n));
        const QuantumStateVector s{n};
        EXPECT_EQ(s.num_qubits(), n);
        EXPECT_EQ(s.size(), bit(n));
        EXPECT_TRUE(statesNear(s, amplitudes(n, {{0, 1.0}}), 0.0));
        EXPECT_DOUBLE_EQ(s.norm(), 1.0);
    }
}

TEST(QuantumStateVector, RejectsQubitCountsOutsideSupportedRange)
{
    EXPECT_THROW(QuantumStateVector{0}, std::length_error);
    EXPECT_THROW(QuantumStateVector{Qputer::kMaxQubits + 1}, std::length_error);
}

TEST(QuantumStateVector, AmplitudeAccessorsAliasTheSameStorage)
{
    QuantumStateVector s{3};
    s[5] = cd{0.25, -0.5};

    EXPECT_EQ(s.vector()[5], (cd{0.25, -0.5}));
    EXPECT_EQ(s.data()[5], (cd{0.25, -0.5}));

    const QuantumStateVector& cs = s;
    EXPECT_EQ(cs[5], (cd{0.25, -0.5}));
}

TEST(QuantumStateVector, NormalizeRescalesToUnitNorm)
{
    // (|0> + i|1>) has norm sqrt(2) before normalization.
    QuantumStateVector s{5};
    s[0] = 1.0;
    s[1] = kI;

    EXPECT_NEAR(s.norm(), std::numbers::sqrt2, kTol);
    s.normalize();
    EXPECT_NEAR(s.norm(), 1.0, kTol);
    EXPECT_TRUE(statesNear(s, amplitudes(5, {{0, kInvSqrt2}, {1, kInvSqrt2 * kI}})));
}
