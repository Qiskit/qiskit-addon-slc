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

from qiskit.circuit import QuantumCircuit
from qiskit_addon_slc.utils.remove_measure import remove_measure


def test_remove_measure() -> None:
    """Test removal of measure operations."""
    circ = QuantumCircuit(4)
    for i in range(2):
        circ.h(i)

    circ_no_meas = remove_measure(circ)

    assert circ_no_meas == circ
    assert circ.count_ops().get("measure", None) is None

    circ.measure_active()

    circ_no_meas = remove_measure(circ)

    assert circ_no_meas.count_ops().get("measure", None) is None
    assert circ.count_ops().get("measure", None) == 2
