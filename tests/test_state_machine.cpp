#include "TestSupport.hpp"

#include <QuantumStateMachine.hpp>

#include <limits>
#include <stdexcept>

using namespace QputerTest;
using Qputer::OpKind;
using Qputer::Operation;
using Qputer::QuantumStateMachine;

namespace
{

Operation op(OpKind kind, QubitList controls, QubitList targets, std::vector<double> params = {})
{
    Operation o;
    o.kind = kind;
    o.controls = std::move(controls);
    o.targets = std::move(targets);
    o.params = std::move(params);
    return o;
}

} // namespace


TEST(StateMachine, StartsInGroundStateWithEmptyCircuit)
{
    const QuantumStateMachine m{4, 3, 7};
    EXPECT_EQ(m.num_qubits(), 4u);
    EXPECT_EQ(m.num_clbits(), 3u);
    EXPECT_EQ(m.seed(), 7u);
    EXPECT_EQ(m.classical_register(), 0u);
    EXPECT_TRUE(m.circuit().empty());
    EXPECT_TRUE(statesNear(m.state(), amplitudes(4, {{0, 1.0}}), 0.0));
}

TEST(StateMachine, RejectsInvalidRegisterSizes)
{
    EXPECT_THROW((QuantumStateMachine{0}), std::length_error);
    EXPECT_THROW((QuantumStateMachine{Qputer::kMaxQubits + 1}), std::length_error);
    EXPECT_THROW((QuantumStateMachine{2, QuantumStateMachine::kMaxClbits + 1}), std::length_error);
}

TEST(StateMachine, NamedGatesMatchDirectKernelCalls)
{
    std::mt19937_64 rng{101};
    const Eigen::VectorXcd psi = randomAmplitudes(5, rng);
    const Eigen::MatrixXcd U2 = randomUnitary(4, rng);
    const Eigen::MatrixXcd U1 = randomUnitary(2, rng);

    QuantumStateMachine m{5, 0, 1};
    m.prepare_state(psi);
    m.x(0).y(1).z(2).h(3).s(4).sdg(0).t(1).tdg(2).sx(3)
     .rx(4, 0.3).ry(0, -1.1).rz(1, 2.2).phase(2, 0.7).u3(3, 0.4, -0.9, 1.6)
     .cnot(0, 4).cz(1, 3).cphase(2, 0, -0.5).swap(1, 4)
     .toffoli(0, 1, 2).fredkin(3, 0, 4)
     .mcx({0, 1, 2}, 3).mcz({1, 2, 4}).mcphase({0, 3}, 1.3)
     .unitary({4, 1}, U2).controlled_unitary({0, 2}, {3}, U1);

    QuantumStateVector ref = fromAmplitudes(5, psi);
    QuantumGate::x(ref, 0); QuantumGate::y(ref, 1); QuantumGate::z(ref, 2); QuantumGate::h(ref, 3);
    QuantumGate::s(ref, 4); QuantumGate::sdg(ref, 0); QuantumGate::t(ref, 1); QuantumGate::tdg(ref, 2);
    QuantumGate::sx(ref, 3);
    QuantumGate::rx(ref, 4, 0.3); QuantumGate::ry(ref, 0, -1.1); QuantumGate::rz(ref, 1, 2.2);
    QuantumGate::phase(ref, 2, 0.7); QuantumGate::u3(ref, 3, 0.4, -0.9, 1.6);
    QuantumGate::cnot(ref, 0, 4); QuantumGate::cz(ref, 1, 3); QuantumGate::cphase(ref, 2, 0, -0.5);
    QuantumGate::swap(ref, 1, 4);
    QuantumGate::toffoli(ref, 0, 1, 2); QuantumGate::fredkin(ref, 3, 0, 4);
    QuantumGate::mcx(ref, {0, 1, 2}, 3); QuantumGate::mcz(ref, {1, 2, 4}); QuantumGate::mcphase(ref, {0, 3}, 1.3);
    QuantumGate::apply(ref, {4, 1}, U2); QuantumGate::controlled(ref, {0, 2}, {3}, U1);

    EXPECT_TRUE(statesNear(m.state(), ref));
    EXPECT_EQ(m.circuit().size(), 25u);
}

TEST(StateMachine, CircuitRecordsEveryOperationInOrder)
{
    QuantumStateMachine m{3, 2, 1};
    m.h(0).cnot(0, 1).rz(2, 0.25).mcphase({0, 1, 2}, 0.5);
    m.measure(1, 0);
    m.reset(2);

    const auto& c = m.circuit();
    ASSERT_EQ(c.size(), 6u);
    EXPECT_EQ(c[0].kind, OpKind::H);
    EXPECT_EQ(c[0].targets, QubitList{0});
    EXPECT_EQ(c[1].kind, OpKind::CNOT);
    EXPECT_EQ(c[1].controls, QubitList{0});
    EXPECT_EQ(c[1].targets, QubitList{1});
    EXPECT_EQ(c[2].kind, OpKind::RZ);
    EXPECT_EQ(c[2].params, std::vector<double>{0.25});
    EXPECT_EQ(c[3].kind, OpKind::MCPhase);
    EXPECT_EQ(c[3].targets, (QubitList{0, 1, 2}));
    EXPECT_EQ(c[4].kind, OpKind::Measure);
    EXPECT_EQ(c[4].clbit, std::optional<std::size_t>{0});
    EXPECT_EQ(c[5].kind, OpKind::Reset);
}

TEST(StateMachine, GenericAppendMatchesNamedMethods)
{
    QuantumStateMachine a{3, 0, 1}, b{3, 0, 1};
    a.h(0).cphase(0, 2, 0.8).toffoli(0, 2, 1).u3(1, 0.1, 0.2, 0.3);
    b.append(op(OpKind::H, {}, {0}))
     .append(op(OpKind::CPhase, {0}, {2}, {0.8}))
     .append(op(OpKind::Toffoli, {0, 2}, {1}))
     .append(op(OpKind::U3, {}, {1}, {0.1, 0.2, 0.3}));
    EXPECT_TRUE(statesNear(a.state(), b.state(), 0.0));
}

TEST(StateMachine, OperationNamesRoundTrip)
{
    for (std::size_t k = 0; k <= static_cast<std::size_t>(OpKind::Reset); ++k)
    {
        const auto kind = static_cast<OpKind>(k);
        SCOPED_TRACE(std::string{Qputer::opName(kind)});
        EXPECT_EQ(Qputer::opKindFromName(Qputer::opName(kind)), kind);
    }
    EXPECT_EQ(Qputer::opKindFromName("hadamard"), std::nullopt);
    EXPECT_EQ(Qputer::opName(OpKind::CNOT), "cnot");
}

TEST(StateMachine, InvalidOperationsLeaveSystemUntouched)
{
    std::mt19937_64 rng{102};
    QuantumStateMachine m{3, 1, 1};
    m.h(0).cnot(0, 1);
    const Eigen::VectorXcd before = m.state().vector();
    const std::size_t opsBefore = m.circuit().size();

    Eigen::MatrixXcd notUnitary = Eigen::MatrixXcd::Identity(2, 2);
    notUnitary(0, 0) = 2.0;
    Operation badMeasure = op(OpKind::Measure, {}, {0});
    badMeasure.clbit = 1;
    Operation clbitOnGate = op(OpKind::X, {}, {0});
    clbitOnGate.clbit = 0;
    Operation conditionedMeasure = op(OpKind::Measure, {}, {0});
    conditionedMeasure.condition = Qputer::Condition{0, true};

    EXPECT_THROW(m.x(3), std::out_of_range);
    EXPECT_THROW(m.cnot(1, 1), std::invalid_argument);
    EXPECT_THROW(m.rx(0, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(m.unitary({0}, notUnitary), std::invalid_argument);
    EXPECT_THROW(m.unitary({0, 1}, randomUnitary(2, rng)), std::invalid_argument);
    EXPECT_THROW(m.mcz({}), std::invalid_argument);
    EXPECT_THROW(m.append(op(OpKind::RX, {}, {0})), std::invalid_argument);          // missing angle
    EXPECT_THROW(m.append(op(OpKind::CNOT, {}, {0, 1})), std::invalid_argument);     // wrong roles
    EXPECT_THROW(m.append(op(static_cast<OpKind>(200), {}, {0})), std::invalid_argument);
    EXPECT_THROW(m.append(badMeasure), std::out_of_range);
    EXPECT_THROW(m.append(clbitOnGate), std::invalid_argument);
    EXPECT_THROW(m.append(conditionedMeasure), std::invalid_argument);
    EXPECT_THROW(m.measure(0, 5), std::out_of_range);
    EXPECT_THROW(m.when(4), std::out_of_range);

    EXPECT_TRUE(statesNear(m.state(), before, 0.0));
    EXPECT_EQ(m.circuit().size(), opsBefore);
    EXPECT_EQ(m.classical_register(), 0u);
}

TEST(StateMachine, PreparationRestartsCircuitAndRegister)
{
    QuantumStateMachine m{3, 2, 1};
    m.x(0);
    m.measure(0, 1);
    ASSERT_EQ(m.classical_register(), 2u);

    m.prepare_basis(5);
    EXPECT_TRUE(statesNear(m.state(), amplitudes(3, {{5, 1.0}}), 0.0));
    EXPECT_TRUE(m.circuit().empty());
    EXPECT_EQ(m.classical_register(), 0u);

    m.h(1);
    m.prepare();
    EXPECT_TRUE(statesNear(m.state(), amplitudes(3, {{0, 1.0}}), 0.0));
    EXPECT_TRUE(m.circuit().empty());

    EXPECT_THROW(m.prepare_basis(8), std::out_of_range);
}

TEST(StateMachine, PrepareStateValidatesAmplitudes)
{
    std::mt19937_64 rng{103};
    QuantumStateMachine m{3, 0, 1};
    const Eigen::VectorXcd psi = randomAmplitudes(3, rng);
    m.prepare_state(psi);
    EXPECT_TRUE(statesNear(m.state(), psi, 0.0));

    EXPECT_THROW(m.prepare_state(randomAmplitudes(2, rng)), std::invalid_argument);
    EXPECT_THROW(m.prepare_state(2.0 * psi), std::invalid_argument);
    Eigen::VectorXcd nan = psi;
    nan[1] = cd{std::numeric_limits<double>::quiet_NaN(), 0.0};
    EXPECT_THROW(m.prepare_state(nan), std::invalid_argument);
    EXPECT_TRUE(statesNear(m.state(), psi, 0.0));
}

TEST(StateMachine, WhenConditionsOnlyTheNextOperation)
{
    QuantumStateMachine m{3, 1, 1};
    m.x(0);
    ASSERT_EQ(m.measure(0, 0), 1);

    m.when(0).x(1);           // clbit 0 == 1: runs
    m.when(0, false).x(2);    // clbit 0 != 0: skipped
    m.x(2);                   // unconditioned again
    EXPECT_TRUE(statesNear(m.state(), amplitudes(3, {{0b111, 1.0}}), 0.0));

    ASSERT_TRUE(m.circuit()[2].condition.has_value());
    EXPECT_EQ(m.circuit()[2].condition->clbit, 0u);
    EXPECT_FALSE(m.circuit()[4].condition.has_value());

    m.when(0);
    EXPECT_THROW(m.measure(1, 0), std::invalid_argument);
}

TEST(StateMachine, SeedReproducesMeasurementSequence)
{
    auto outcomes = [](std::uint64_t seed)
    {
        QuantumStateMachine m{1, 0, seed};
        std::vector<int> out;
        for (int k = 0; k < 64; ++k)
        {
            m.prepare();
            m.h(0);
            out.push_back(m.measure(0));
        }
        return out;
    };
    EXPECT_EQ(outcomes(42), outcomes(42));
    EXPECT_NE(outcomes(42), outcomes(43));

    QuantumStateMachine unseeded{1};
    QuantumStateMachine replica{1, 0, unseeded.seed()};
    EXPECT_EQ(unseeded.h(0).sample({0}, 32), replica.h(0).sample({0}, 32));
}
