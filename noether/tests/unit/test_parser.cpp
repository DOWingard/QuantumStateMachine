// Parser tests: precedence and associativity (checked by evaluation), time order of operator
// products, the postfix adjacency rule, and error recovery.

#include "TestUtil.hpp"

#include "Ir.hpp"

using namespace Noether;
using namespace NoetherTest;

namespace
{

double value(const std::string& expr, const std::string& prelude = "")
{
    const Executed run = expectRuns("noether 0.1\n" + prelude + "print " + expr + " as v\n");
    return run.number("v");
}

// Gate names of the op events, in time order.
std::vector<std::string> opOrder(const std::string& body)
{
    const Executed run = runSource("noether 0.1\nqubits q[3]\n" + body + "\n");
    EXPECT_TRUE(run.compiled) << run.diagnostics;
    std::vector<std::string> out;
    for (const Event& ev : run.c->compiler->ir().events)
        if (ev.kind == EvK::Op) out.push_back(std::format("{}{}", gkName(ev.op.kind), ev.op.qubits().empty() ? "" : std::to_string(ev.op.qubits().back())));
    return out;
}

} // namespace

TEST(Parser, ArithmeticPrecedenceAndAssociativity)
{
    EXPECT_DOUBLE_EQ(value("2 + 3·4"), 14);
    EXPECT_DOUBLE_EQ(value("2·3^2"), 18);
    EXPECT_DOUBLE_EQ(value("2^3^2"), 512); // right-associative
    EXPECT_DOUBLE_EQ(value("-2^2"), -4);   // power binds tighter than unary minus
    EXPECT_DOUBLE_EQ(value("6/2/3"), 1);   // left-associative
    EXPECT_DOUBLE_EQ(value("2^-1"), 0.5);
    EXPECT_DOUBLE_EQ(value("2^{1 + 1}"), 4);
    EXPECT_DOUBLE_EQ(value("(2 + 3)·4"), 20);
    EXPECT_DOUBLE_EQ(value("10 - 3 - 2"), 5);
}

TEST(Parser, JuxtapositionIsMultiplicationAfterDivision)
{
    // π/4 √(2^n) is (π/4)·√(2^n): the Grover iteration count.
    EXPECT_NEAR(value("π/4 √(32)"), std::numbers::pi / 4 * std::sqrt(32.0), 1e-15);
    EXPECT_DOUBLE_EQ(value("1/2 x", "let x = 3\n"), 1.5);
    EXPECT_DOUBLE_EQ(value("2x^2", "let x = 3\n"), 18);
    EXPECT_DOUBLE_EQ(value("-x^2", "let x = 3\n"), -9);
    EXPECT_NEAR(value("sin(π/2)^2"), 1.0, 1e-15);
}

TEST(Parser, ComparisonAndLogic)
{
    const Executed run = expectRuns("noether 0.1\n"
                               "assert 1 < 2 and 2 ≤ 2 and 3 > 2 and 3 ≥ 3 and 1 ≠ 2 and 1 == 1\n"
                               "assert ¬(1 > 2) or false\n"
                               "assert true and not false\n");
    EXPECT_EQ(run.r.asserts.size(), 3U);
}

TEST(Parser, OperatorProductsApplyRightToLeft)
{
    // The rightmost factor acts first, as in A B |ψ⟩.
    EXPECT_EQ(opOrder("X_0 H_1 Z_2"), (std::vector<std::string>{"Z2", "H1", "X0"}));
    // Statements on separate lines run top to bottom.
    EXPECT_EQ(opOrder("X_0\nH_1"), (std::vector<std::string>{"X0", "H1"}));
    // A def body is itself a product.
    EXPECT_EQ(opOrder("def G_{a}: H_a X_a\nG_0"), (std::vector<std::string>{"X0", "H0"}));
}

TEST(Parser, PostfixOperatorsMustTouchTheirOperand)
{
    // `S †_0` is not `S†_0`: the dagger needs adjacency.
    EXPECT_FALSE(runSource("noether 0.1\nqubits q[1]\nS †_0\n").compiled);
    EXPECT_TRUE(runSource("noether 0.1\nqubits q[1]\nS†_0\n").compiled);
    // A space before a subscript separates it.
    EXPECT_FALSE(runSource("noether 0.1\nqubits q[1]\nX _0\n").compiled);
}

TEST(Parser, CallNeedsAdjacentParenthesis)
{
    const Executed run = runSource("noether 0.1\nlet x = 1\nprint sin (x)\n");
    EXPECT_EQ(run.codes(), std::vector<std::string>{"W0009"}) << run.diagnostics;
    EXPECT_NEAR(run.number("sin (x)"), std::sin(1.0), 1e-15); // still evaluated as the call
}

TEST(Parser, RecoveryReportsEveryIndependentError)
{
    const Executed run = runSource("noether 0.1\nqubits q[2]\nCNOT_{0→}\nlet = 3\nH_0\nprint (1 + \n");
    EXPECT_GE(run.c->diags->errorCount(), 3U) << run.diagnostics;
}

TEST(Parser, HeaderIsRequired)
{
    EXPECT_FALSE(runSource("qubits q[1]\nH_0\n").compiled);
    EXPECT_FALSE(runSource("noether 9.9\nqubits q[1]\nH_0\n").compiled);
}

TEST(Parser, SubscriptForms)
{
    // Single index, braced expression, register, slice, explicit list, arrows.
    expectRuns("noether 0.1\nqubits q[4], r[2]\nlet k = 1\n"
               "X_0 X_{k + 1} H_r H_{q[0..1]} Z_{q[2], q[3]} CNOT_{0→1} Toffoli_{0, 1→2} Fredkin_{0→1, 2}\n");
}

TEST(Parser, OneLineSuitesAndSemicolons)
{
    const Executed run = expectRuns("noether 0.1\nqubits q[2]; bits c[1]\nfor k ∈ 0..1: X_k\nif true: X_0\nelse: X_1\nprint ⟨Z_0⟩ as z\n");
    EXPECT_DOUBLE_EQ(run.number("z"), 1.0); // the loop flips q0 and the if flips it back
}
