"""Utilities for sampling random pyzx circuits."""

from fractions import Fraction

import numpy as np
import pyzx as zx

# Gate-type indices for the categorical sampling in random_circuit().
_CNOT = 0
_HADAMARD = 1
_Z_PHASE = 2
_X_PHASE = 3

_PHASE_GATE_NAMES = ("XPhase", "ZPhase")


def random_circuit(
    n_qubit: int = 5,
    num_gates: int = 40,
    p_two_qubit: float = 0.1,
    p_H: float = 0.7,
    p_z: float = 0.1,
    p_x: float = 0.1,
    many_pi_gates: bool = False,
    clifford_plus_T: bool = False,
    add_cz: bool = False,
    p_cz: float = 0.5,
    rng: np.random.Generator | None = None,
) -> zx.Circuit:
    """Sample a random circuit.

    Args:
        n_qubit: Number of qubits.
        num_gates: Total number of gates in the circuit.
        p_two_qubit: Probability for a two-qubit gate (CNOT, or CNOT/CZ if add_cz).
        p_H: Probability for a Hadamard gate.
        p_z: Probability for a Z-phase gate.
        p_x: Probability for an X-phase gate.
        many_pi_gates: If True, replace the phases of ~50% of the phase gates by pi.
            Used to check the correctness of the pi_copy rule.
        clifford_plus_T: If True, only sample Clifford+T phases (multiples of pi/4).
        add_cz: If True, two-qubit gates are split between CNOT and CZ according to p_cz.
        p_cz: Probability that a two-qubit gate is a CZ rather than a CNOT, when
            add_cz is True. Ignored otherwise.
        rng: Optional numpy Generator for reproducible sampling. If None, the legacy
            global ``np.random`` state is used (honours ``np.random.seed``).

    Note:
        The probability vector is ordered ``[p_two_qubit, p_H, p_z, p_x]`` matching
        the gate branches. Versions before 2026-10 used the scrambled order
        ``[p_two_qubit, p_z, p_x, p_H]`` (p_H controlled X-phases etc.), so
        freshly sampled circuits are not identically distributed to the training
        data of checkpoints created before the fix.
    """
    if n_qubit < 1:
        raise ValueError(f"n_qubit must be positive, got {n_qubit}")
    if num_gates < 0:
        raise ValueError(f"num_gates must be non-negative, got {num_gates}")
    probs = np.array([p_two_qubit, p_H, p_z, p_x], dtype=float)
    if np.any(probs < 0):
        raise ValueError(f"Gate probabilities must be non-negative, got {probs.tolist()}")
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError(f"Gate probabilities must sum to 1, got {probs.sum()}")
    if p_two_qubit > 0 and n_qubit < 2:
        raise ValueError("n_qubit must be at least 2 when two-qubit gates are allowed")
    if not 0 <= p_cz <= 1:
        raise ValueError(f"p_cz must be between 0 and 1, got {p_cz}")

    # The np.random module and np.random.Generator both provide choice()/random(),
    # so the legacy global-seed workflow keeps working when rng is None.
    rand = rng if rng is not None else np.random

    c = zx.Circuit(qubit_amount=n_qubit)
    for _ in range(num_gates):
        gate = rand.choice(len(probs), p=probs)
        if gate == _CNOT:
            target = rand.choice(n_qubit)
            control = rand.choice([q for q in range(n_qubit) if q != target])
            if add_cz and rand.random() < p_cz:
                c.add_gate("CZ", control, target)
            else:
                c.add_gate("CNOT", control, target)
        elif gate == _HADAMARD:
            c.add_gate("H", rand.choice(n_qubit))
        elif gate == _Z_PHASE:
            c.add_gate("ZPhase", rand.choice(n_qubit), phase=_sample_phase(rand, clifford_plus_T))
        else:  # _X_PHASE
            c.add_gate("XPhase", rand.choice(n_qubit), phase=_sample_phase(rand, clifford_plus_T))
    if many_pi_gates:
        replace_by_pi_gates(c, rng=rng)
    return c


def _sample_phase(rand, clifford_plus_T: bool) -> Fraction:
    """Draw a random phase as a fraction of pi, normalized to [0, 2).

    Clifford+T phases are multiples of pi/4, so a is drawn from range(8) to keep
    all 8 distinct phases equally likely (range(10) would alias a=8,9 onto the
    same phases as a=0,1, double-sampling 0 and pi/4).
    """
    if clifford_plus_T:
        a, b = rand.choice(8), 4
    else:
        a, b = rand.choice(10), 1 + rand.choice(9)
    return Fraction(a, b) % 2


def replace_by_pi_gates(
    circuit: zx.Circuit, p: float = 0.5, rng: np.random.Generator | None = None
) -> zx.Circuit:
    """Replace the phase of each X/Z-phase gate by pi with probability p (in place)."""
    if not 0 <= p <= 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")
    rand = rng if rng is not None else np.random
    for gate in circuit.gates:
        if gate.name in _PHASE_GATE_NAMES and rand.random() <= p:
            gate.phase = Fraction(1, 1)
    return circuit

