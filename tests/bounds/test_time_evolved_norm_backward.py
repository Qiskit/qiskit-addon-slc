# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit tests for the backward-evolved commutator bound of a single Pauli error term."""

from __future__ import annotations

import numpy as np
import pytest
from pauli_prop.propagation import RotationGates
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_addon_slc.bounds.backward import _time_evolved_norm_backward

# A circuit with no non-Clifford content: the error term is not evolved at all.
NO_GATES = RotationGates([], [], [])

# Most tests keep the term limit well below the default, to make truncation behavior observable.
MAX_TERMS = 1000


def _rotation_gates(circuit: QuantumCircuit) -> RotationGates:
    """Converts a circuit into the reversed rotation gates the backward bound expects.

    Args:
        circuit: the circuit whose rotations to collect.

    Returns:
        The rotation gates, in reverse circuit order.
    """
    gates = RotationGates([], [], [])
    for instruction in circuit.data:
        gates.append_circuit_instruction(
            instruction,
            [circuit.find_bit(qubit).index for qubit in instruction.qubits],
            circuit.num_qubits,
        )
    return RotationGates(gates.gates[::-1], gates.qargs[::-1], gates.thetas[::-1])


def test_the_default_term_limit_is_usable():
    """Test that the function can be called on an evolved term without passing a term limit.

    The default used to be the largest unsigned integer, which the propagation routine pre-allocates
    as a capacity rather than treating as "unlimited" -- so calling this function with its own
    defaults asked the allocator for terabytes and aborted the process.
    """
    circuit = QuantumCircuit(3)
    for qubit in range(3):
        circuit.rx(0.3, qubit)
    for qubit in range(2):
        circuit.rzz(0.2, qubit, qubit + 1)

    bounds = _time_evolved_norm_backward(Pauli("XII"), _rotation_gates(circuit))

    assert 0.0 <= bounds.commutator_bound <= 2.0 + 1e-8


@pytest.mark.parametrize(
    ("pauli", "expected"),
    [
        # A Pauli acting only in the Z basis leaves the all-zero state untouched, so it cannot bias
        # any expectation value.
        ("ZI", 0.0),
        ("IZ", 0.0),
        ("ZZ", 0.0),
        # A Pauli with an X component flips the state, giving the maximal bound of 2.
        ("XI", 2.0),
        ("IX", 2.0),
        ("YI", 2.0),
    ],
    ids=["ZI", "IZ", "ZZ", "XI", "IX", "YI"],
)
def test_unevolved_bound_depends_on_the_x_component(pauli, expected):
    """Test the bound of an unevolved error term against the all-zero state.

    The nuclear norm of the commutator with the state is set by whether the error term acts on the
    computational basis at all, i.e. whether it has an X or Y component.

    Args:
        pauli: the error Pauli term.
        expected: the bound it should produce.
    """
    bounds = _time_evolved_norm_backward(Pauli(pauli), NO_GATES, evolution_max_terms=MAX_TERMS)

    assert bounds.commutator_bound == pytest.approx(expected)
    assert bounds.truncation_bias == 0.0
    # The backward bound never falls back on a triangle inequality; it is computed in closed form.
    assert not bounds.fallback_to_tri_ineq


@pytest.mark.parametrize("num_qubits", [2, 4, 6])
def test_bound_never_exceeds_the_theoretical_maximum(num_qubits):
    """Test that the computed bound stays within ``[0, 2]`` for an evolved error term.

    Args:
        num_qubits: the width of the circuit to evolve through.
    """
    circuit = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.rx(0.3, qubit)
    for qubit in range(num_qubits - 1):
        circuit.rzz(0.2, qubit, qubit + 1)

    bounds = _time_evolved_norm_backward(
        Pauli("X" + "I" * (num_qubits - 1)),
        _rotation_gates(circuit),
        evolution_max_terms=MAX_TERMS,
    )

    assert 0.0 <= bounds.commutator_bound <= 2.0 + 1e-8
    assert bounds.truncation_bias >= 0.0


def test_a_tight_term_limit_reports_truncation_bias():
    """Test that truncating the evolution is accounted for in the truncation bias."""
    num_qubits = 6
    circuit = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.rx(0.5, qubit)
    for qubit in range(num_qubits - 1):
        circuit.rzz(0.5, qubit, qubit + 1)

    generous = _time_evolved_norm_backward(
        Pauli("X" + "I" * (num_qubits - 1)),
        _rotation_gates(circuit),
        evolution_max_terms=MAX_TERMS,
    )
    truncated = _time_evolved_norm_backward(
        Pauli("X" + "I" * (num_qubits - 1)),
        _rotation_gates(circuit),
        evolution_max_terms=2,
    )

    assert generous.truncation_bias == pytest.approx(0.0, abs=1e-9)
    assert truncated.truncation_bias > generous.truncation_bias


def test_an_overwhelming_truncation_bias_abandons_the_bound():
    """Test that a truncation bias at or above 2 abandons the computation.

    Once truncation alone accounts for the whole theoretical bound of 2, there is nothing left for a
    norm computation to tighten, so the bound is reported as ``NaN``.
    """
    num_qubits = 10
    # A rotation angle far from any Clifford angle spreads the operator over many terms, so keeping
    # only one of them discards most of its 1-norm.
    circuit = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.rx(np.pi / 3, qubit)
    for qubit in range(num_qubits - 1):
        circuit.rzz(np.pi / 3, qubit, qubit + 1)

    bounds = _time_evolved_norm_backward(
        Pauli("X" + "I" * (num_qubits - 1)),
        _rotation_gates(circuit),
        evolution_max_terms=1,
    )

    assert np.isnan(bounds.commutator_bound)
    assert bounds.truncation_bias >= 2.0
