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

"""Tests for the circuit annotations utilities."""

from __future__ import annotations

from itertools import count

from qiskit.circuit import QuantumCircuit
from qiskit_addon_slc.utils.annotations import map_modifier_ref_to_ref
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.transpiler.passes import AddInjectNoise


def test_noise_model_paulis() -> None:
    """Test parsing of the InjectNoise annotations."""
    num_qubits = 8
    expected_map = {}
    circ = QuantumCircuit(num_qubits)
    # repeat layers some number times
    for rep in range(4):
        for first_qubit in range(2):
            expected_map[f"m{rep * 2 + first_qubit}"] = f"r{first_qubit}"
            for idx in range(first_qubit + 1, num_qubits, 2):
                circ.cx(idx - 1, idx)

    # NOTE: we hack around the fact that we check against hard-coded annotation IDs by forcing the
    # annotation counter to reset here. This should NOT be relied upon in the future.
    AddInjectNoise._REF_COUNTER = count()
    AddInjectNoise._MODIFIER_REF_COUNTER = count()

    boxes_pm = generate_boxing_pass_manager(
        inject_noise_targets="all",
        inject_noise_strategy="individual_modification",
        twirling_strategy="active",
    )
    boxed_circ = boxes_pm.run(circ)

    assert map_modifier_ref_to_ref(boxed_circ) == expected_map
