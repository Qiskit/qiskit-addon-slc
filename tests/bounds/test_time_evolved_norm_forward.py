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

"""Unit tests for the forward-evolved commutator bound of a single Pauli error term.

The end-to-end test exercises this function through the full workflow, which leaves most of its
branches uncovered. These tests drive it directly, one branch at a time.
"""

from __future__ import annotations

import numpy as np
import pytest
from pauli_prop.propagation import RotationGates
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_addon_slc.bounds.forward import time_evolved_norm_forward

# A circuit with no non-Clifford content: the error term reaches the observable unevolved.
NO_GATES = RotationGates([], [], [])

# Most tests keep the term limit well below the default, to make truncation behavior observable.
MAX_TERMS = 1000


def _rotation_gates(circuit: QuantumCircuit) -> RotationGates:
    """Converts a circuit into the reversed rotation gates the forward bound expects.

    This mirrors what :func:`~qiskit_addon_slc.bounds.compute_bounds` does when it hands a light cone
    to the norm function: the gates are collected in circuit order and then reversed, because the
    error term is evolved forwards from where it occurs towards the end of the circuit.

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

    bounds = time_evolved_norm_forward(Pauli("XII"), _rotation_gates(circuit), Pauli("ZZI"))

    assert 0.0 <= bounds.commutator_bound <= 2.0 + 1e-8


def test_single_pauli_anticommuting_with_the_observable():
    """Test the bound of an unevolved error term that anticommutes with the observable.

    A single Pauli that anticommutes gives the maximal commutator bound of 2, with no truncation and
    no fallback.
    """
    bounds = time_evolved_norm_forward(
        Pauli("XI"), NO_GATES, Pauli("ZI"), evolution_max_terms=MAX_TERMS
    )

    assert bounds.commutator_bound == pytest.approx(2.0)
    assert bounds.truncation_bias == 0.0
    assert not bounds.fallback_to_tri_ineq


def test_single_pauli_commuting_with_the_observable():
    """Test that an unevolved error term commuting with the observable contributes no bias."""
    bounds = time_evolved_norm_forward(
        Pauli("XI"), NO_GATES, Pauli("XI"), evolution_max_terms=MAX_TERMS
    )

    assert bounds.commutator_bound == 0.0
    assert not bounds.fallback_to_tri_ineq


@pytest.mark.parametrize("num_qubits", [2, 3, 5])
def test_bound_never_exceeds_the_theoretical_maximum(num_qubits):
    """Test that the computed bound stays within ``[0, 2]`` for an evolved error term.

    The nuclear norm of a commutator of two Paulis cannot exceed 2. ``num_qubits`` is chosen to cover
    both the exact eigensolver used at 4 qubits or fewer and the iterative one used above that.

    Args:
        num_qubits: the width of the circuit to evolve through.
    """
    circuit = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.rz(0.3, qubit)
    for qubit in range(num_qubits - 1):
        circuit.rzz(0.2, qubit, qubit + 1)

    observable = Pauli("Z" * num_qubits)
    error = Pauli("X" + "I" * (num_qubits - 1))

    bounds = time_evolved_norm_forward(
        error, _rotation_gates(circuit), observable, evolution_max_terms=MAX_TERMS
    )

    assert 0.0 <= bounds.commutator_bound <= 2.0 + 1e-8
    assert bounds.truncation_bias >= 0.0


def test_exceeding_the_qubit_limit_falls_back_to_the_triangle_inequality():
    """Test that a commutator wider than ``eigval_max_qubits`` uses the looser bound.

    Rather than attempt an expensive eigenvalue computation, the bound is approximated by a triangle
    inequality, which the result flags.
    """
    circuit = QuantumCircuit(3)
    circuit.rz(0.4, 0)
    circuit.rzz(0.3, 0, 1)
    circuit.rzz(0.2, 1, 2)

    bounds = time_evolved_norm_forward(
        Pauli("XII"),
        _rotation_gates(circuit),
        Pauli("ZZI"),
        evolution_max_terms=MAX_TERMS,
        eigval_max_qubits=1,
    )

    assert bounds.fallback_to_tri_ineq
    # The triangle inequality is an upper bound, so it can only be looser than the true norm.
    assert bounds.commutator_bound >= 0.0


def test_a_non_default_norm_order_skips_the_eigensolver():
    """Test that requesting a norm order other than 2 computes that norm directly.

    Only the spectral norm needs an eigensolver; any other order is evaluated straight away and so is
    never flagged as a fallback.
    """
    circuit = QuantumCircuit(2)
    circuit.rz(0.4, 0)
    circuit.rzz(0.3, 0, 1)

    bounds = time_evolved_norm_forward(
        Pauli("XI"),
        _rotation_gates(circuit),
        Pauli("ZI"),
        evolution_max_terms=MAX_TERMS,
        comm_norm_order=1,
    )

    assert not bounds.fallback_to_tri_ineq
    assert bounds.commutator_bound >= 0.0


def test_a_tight_term_limit_reports_truncation_bias():
    """Test that truncating the evolution is accounted for in the truncation bias.

    Keeping fewer operator terms during the evolution loses part of the operator's 1-norm, and that
    loss has to be reported so the caller can account for it.
    """
    num_qubits = 6
    circuit = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.rx(0.5, qubit)
    for qubit in range(num_qubits - 1):
        circuit.rzz(0.5, qubit, qubit + 1)

    generous = time_evolved_norm_forward(
        Pauli("X" + "I" * (num_qubits - 1)),
        _rotation_gates(circuit),
        Pauli("Z" * num_qubits),
        evolution_max_terms=MAX_TERMS,
    )
    truncated = time_evolved_norm_forward(
        Pauli("X" + "I" * (num_qubits - 1)),
        _rotation_gates(circuit),
        Pauli("Z" * num_qubits),
        evolution_max_terms=2,
    )

    assert generous.truncation_bias == pytest.approx(0.0, abs=1e-9)
    assert truncated.truncation_bias > generous.truncation_bias


def test_an_overwhelming_truncation_bias_abandons_the_bound():
    """Test that a truncation bias at or above 2 abandons the computation.

    Once the truncation alone accounts for the whole theoretical bound of 2, computing a commutator
    norm cannot tighten anything, so the bound is reported as ``NaN``.
    """
    num_qubits = 10
    # A rotation angle far from any Clifford angle spreads the operator over many terms, so keeping
    # only one of them discards most of its 1-norm.
    circuit = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.rx(np.pi / 3, qubit)
    for qubit in range(num_qubits - 1):
        circuit.rzz(np.pi / 3, qubit, qubit + 1)

    bounds = time_evolved_norm_forward(
        Pauli("X" + "I" * (num_qubits - 1)),
        _rotation_gates(circuit),
        Pauli("Z" * num_qubits),
        # Keeping a single term throws away almost the entire operator.
        evolution_max_terms=1,
    )

    assert np.isnan(bounds.commutator_bound)
    assert bounds.truncation_bias >= 2.0
