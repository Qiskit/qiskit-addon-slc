# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Warning: this module is not documented and it does not have an RST file.
# If we ever publicly expose interfaces users can import from this module,
# we should set up its RST file.

"""Tests for the circuit iteration utility."""

from __future__ import annotations

from itertools import count

from qiskit.circuit import CircuitInstruction, QuantumCircuit
from qiskit.circuit.library import CXGate, HGate
from qiskit_addon_slc.utils.circuit_iter import iter_circuit
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.transpiler.passes import AddInjectNoise


def test_circuit_iter() -> None:
    """Test the circuit iteration utility."""
    num_qubits = 6
    circ = QuantumCircuit(num_qubits)
    # repeat layers some number times
    for _ in range(2):
        for first_qubit in range(2):
            if first_qubit == 0:
                for idx in range(num_qubits):
                    circ.rx(0.1, idx)
            for idx in range(first_qubit + 1, num_qubits, 2):
                circ.rzz(-0.1, idx - 1, idx)

    # NOTE: we hack around the fact that we check against hard-coded annotation IDs by forcing the
    # annotation counter to reset here. This should NOT be relied upon in the future.
    AddInjectNoise._REF_COUNTER = count()
    AddInjectNoise._MODIFIER_REF_COUNTER = count()

    boxes_pm = generate_boxing_pass_manager(
        inject_noise_targets="all", inject_noise_strategy="individual_modification"
    )
    boxed_circ = boxes_pm.run(circ)

    # insert some additional circuit in front of the boxed circuit, simply to test the behavior of:
    #  - gates outside boxes
    #  - BoxOp instructions without annotations

    pre_circ = QuantumCircuit(num_qubits)
    for idx in range(num_qubits):
        pre_circ.h(idx)
    with pre_circ.box():
        for idx in range(num_qubits - 1):
            pre_circ.cx(idx, idx + 1)

    full_circ = pre_circ.compose(boxed_circ)

    iterator_data = list(iter_circuit(full_circ))

    expected_data: tuple[CircuitInstruction, list[int], str | None, str | None] = []
    for idx in range(num_qubits):
        expected_data.append(
            (
                CircuitInstruction(HGate(), qubits=[full_circ.qubits[idx]]),
                [idx],
                None,
                None,
            )
        )
    for idx in range(num_qubits - 1):
        expected_data.append(
            (
                CircuitInstruction(CXGate(), qubits=full_circ.qubits[idx : idx + 2]),
                [idx, idx + 1],
                None,
                None,
            )
        )
    expected_data.append((boxed_circ.data[0], [0, 1, 2, 3, 4, 5], "m0", "r0"))
    expected_data.append((boxed_circ.data[1], [1, 2, 3, 4], "m1", "r1"))
    expected_data.append((boxed_circ.data[2], [0, 1, 2, 3, 4, 5], "m2", "r0"))
    expected_data.append((boxed_circ.data[3], [1, 2, 3, 4], "m3", "r1"))

    for actual, expected in zip(iterator_data, expected_data, strict=True):
        assert actual == expected
