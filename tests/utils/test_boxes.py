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

"""Tests for BoxOp instruction utilities."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import BoxOp, QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_addon_slc.utils.boxes import box_op_inverse


def test_box_op_inverse(subtests) -> None:
    """Test inverting a BoxOp inside a QuantumCircuit.

    Note: the box_op_inverse monkeypatch of BoxOp.inverse is done as part of qiskit_addon_slc's
    package-level initialization, so we must ensure that package gets imported!

    Args:
        subtests: the pytest-subtests fixture.
    """
    circ = QuantumCircuit(4)
    with circ.box():
        circ.s(0)
        circ.rx(0.1, 1)
        circ.rzz(0.1, 2, 3)

    box_op = circ.data[0].operation

    expected_inverted_box = QuantumCircuit(4)
    expected_inverted_box.sdg(0)
    expected_inverted_box.rx(-0.1, 1)
    expected_inverted_box.rzz(-0.1, 2, 3)

    with subtests.test("direct"):
        box_op_inv = box_op_inverse(box_op, annotated=False)
        assert np.allclose(Operator(box_op_inv.body).data, Operator(expected_inverted_box).data)

    with subtests.test("annotated=True") and pytest.raises(NotImplementedError):
        box_op_inverse(box_op, annotated=True)

    circ_inv = circ.inverse()

    with subtests.test("in-situ"):
        actual_inverted_box = circ_inv.data[0].operation
        assert isinstance(actual_inverted_box, BoxOp)

        assert np.allclose(
            Operator(actual_inverted_box.body).data, Operator(expected_inverted_box).data
        )

        assert circ_inv.inverse() == circ
