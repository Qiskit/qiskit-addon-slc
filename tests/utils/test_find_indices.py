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

"""Tests for the ``find_indices`` utility."""

from __future__ import annotations

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit_addon_slc.utils.find_indices import find_indices


def test_find_indices(subtests) -> None:
    """Test finding qubit indices from different inputs.

    Args:
        subtests: the pytest-subtests fixture.
    """
    circ = QuantumCircuit(4)
    circ.s(0)
    circ.rx(0.1, 1)
    circ.rzz(0.1, 2, 3)

    with subtests.test("single qubit"):
        assert find_indices(circ, circ.qubits[1]) == 1

    with subtests.test("multiple qubits"):
        assert find_indices(circ, circ.qubits[1:2]) == [1]

    with subtests.test("circuit instruction"):
        assert find_indices(circ, circ.data[1]) == [1]

    with subtests.test("unsupported input") and pytest.raises(TypeError):
        assert find_indices(circ, None)  # type: ignore[arg-type]

    with subtests.test("unsupported input inside sequence") and pytest.raises(TypeError):
        assert find_indices(circ, circ.data[1:2])
