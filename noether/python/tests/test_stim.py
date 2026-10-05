from __future__ import annotations

import random

import numpy as np
import pytest

stim = pytest.importorskip("stim")

import noether_interop as ni  # noqa: E402
from conftest import assert_rates_close, fidelity  # noqa: E402

UNITARY = sorted(g.name for g in stim.gate_data().values() if g.is_unitary and g.name not in ("SPP", "SPP_DAG"))
TWO = {g.name for g in stim.gate_data().values() if g.is_two_qubit_gate}


def random_clifford(n, length, rng):
    lines = []
    for _ in range(length):
        g = rng.choice(UNITARY)
        k = 2 if g in TWO else 1
        lines.append(f"{g} " + " ".join(map(str, rng.sample(range(n), k))))
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("seed", range(20))
def test_random_cliffords_match_the_tableau(seed):
    rng = random.Random(seed)
    text = random_clifford(4, 40, rng) + "".join(f"I {q}\n" for q in range(4))
    sim = stim.TableauSimulator()
    sim.do(stim.Circuit(text))
    want = sim.state_vector(endian="little")
    assert fidelity(ni.statevector(stim.Circuit(text)), want) >= 1 - 1e-10, text


@pytest.mark.parametrize("k", range(15))
def test_pauli_channel_2_term_order(k):
    # Bell pairs (0, 2) and (1, 3) read back the Pauli applied to 0 and 1 deterministically.
    probs = ["0"] * 15
    probs[k] = "1"
    text = f"H 0 1\nCX 0 2 1 3\nPAULI_CHANNEL_2({', '.join(probs)}) 0 1\nCX 0 2 1 3\nH 0 1\nM 0 2 1 3\n"
    want = stim.Circuit(text).compile_sampler().sample(1)[0]
    got = ni.run(stim.Circuit(text), shots=5, seed=1)["records"]
    assert (got == want).all(), (k, got[0], want)


@pytest.mark.parametrize("k", range(3))
def test_pauli_channel_1_term_order(k):
    probs = ["0"] * 3
    probs[k] = "1"
    text = f"H 0\nCX 0 1\nPAULI_CHANNEL_1({', '.join(probs)}) 0\nCX 0 1\nH 0\nM 0 1\n"
    want = stim.Circuit(text).compile_sampler().sample(1)[0]
    assert (ni.run(stim.Circuit(text), shots=5, seed=1)["records"] == want).all()


def compare_marginals(circuit, shots, seed):
    ours = ni.run(circuit, shots=shots, seed=seed)
    theirs = circuit.compile_sampler(seed=seed).sample(shots)
    for j in range(circuit.num_measurements):
        assert_rates_close(ours["records"][:, j].mean(), shots, theirs[:, j].mean(), shots, f"record {j}")
    if circuit.num_detectors:
        det, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
        for j in range(circuit.num_detectors):
            assert_rates_close(ours["detectors"][j], shots, det[:, j].mean(), shots, f"detector {j}")
        for j in range(circuit.num_observables):
            assert_rates_close(ours["observables"][j], shots, obs[:, j].mean(), shots, f"observable {j}")


@pytest.mark.parametrize("task", ["repetition_code:memory", "surface_code:rotated_memory_z", "color_code:memory_xyz"])
def test_generated_codes_match_stim_samplers(task):
    circuit = stim.Circuit.generated(task, distance=3, rounds=3, after_clifford_depolarization=0.02,
                                     before_round_data_depolarization=0.01, after_reset_flip_probability=0.01)
    compare_marginals(circuit, 100_000, seed=4)


def test_correlated_errors_and_pauli_products_match_stim():
    circuit = stim.Circuit("""
        H 0 1 2
        E(0.1) X0 Z1
        ELSE_CORRELATED_ERROR(0.2) Y2
        ELSE_CORRELATED_ERROR(0.3) X0 X1 X2
        MPP X0*X1 Z1*Z2 !Y0*Y2
        SPP X0*Z2
        DEPOLARIZE2(0.05) 0 2
        H 1
        M 0 1 2
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-3]
    """)
    compare_marginals(circuit, 100_000, seed=5)


def test_records_are_per_shot_rows():
    out = ni.run(stim.Circuit("X 0\nM 0 1\nMR 0\nM 0\n"), shots=7, seed=1)
    assert out["records"].shape == (7, 4)
    assert (out["records"] == np.array([True, False, True, False])).all()
