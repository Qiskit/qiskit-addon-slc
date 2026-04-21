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

"""Tests for the circuit annotations utilities."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit_addon_slc.utils.annotations import map_modifier_ref_to_ref
from samplomatic.transpiler import generate_boxing_pass_manager


def test_noise_model_paulis() -> None:
    """Test parsing of the InjectNoise annotations."""
    num_qubits = 8
    circ = QuantumCircuit(num_qubits)
    # repeat layers some number times
    for _ in range(4):
        for first_qubit in range(2):
            for idx in range(first_qubit + 1, num_qubits, 2):
                circ.cx(idx - 1, idx)

    boxes_pm = generate_boxing_pass_manager(
        inject_noise_targets="all",
        inject_noise_strategy="individual_modification",
        twirling_strategy="active",
    )
    boxed_circ = boxes_pm.run(circ)

    actual_map = map_modifier_ref_to_ref(boxed_circ)

    expected_map = {}
    for mod_ref in ("m0", "m2", "m4", "m6"):
        expected_map[mod_ref] = actual_map["m0"]
    for mod_ref in ("m1", "m3", "m5", "m7"):
        expected_map[mod_ref] = actual_map["m1"]

    assert actual_map == expected_map
