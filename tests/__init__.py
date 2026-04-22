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

"""Tests for SLC."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def construct_trotter_circuit(
    num_qubits: int,
    num_trotter_steps: int,
    rx_angle: float,
    rzz_angle: float,
    use_clifford: bool = False,
) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)

    for _ in range(num_trotter_steps):
        circuit.rx(rx_angle, range(num_qubits))
        circuit.barrier()
        for first_qubit in (0, 1):
            for idx in range(first_qubit, num_qubits - 1, 2):
                if use_clifford:
                    assert np.isclose(rzz_angle, -np.pi / 2)
                    circuit.sdg([idx, idx + 1])
                    circuit.cz(idx, idx + 1)
                else:
                    circuit.rzz(rzz_angle, idx, idx + 1)
        circuit.barrier()

    return circuit
