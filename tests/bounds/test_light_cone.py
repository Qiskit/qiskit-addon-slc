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

"""Tests for the LightCone commutation checker."""

from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.circuit.library import CXGate, CZGate, PauliGate
from qiskit_addon_slc.bounds.light_cone import LightCone


def test_zxz(subtests):
    """Test commutation of different instructions with ZXZ Pauli.

    Args:
        subtests: the pytest-subtests fixture.
    """
    qubits = list([Qubit() for _ in range(3)])
    operations = [(PauliGate("ZXZ"), qubits)]
    lc = LightCone(set(qubits), operations)

    with subtests.test("cz(0,2)"):
        assert lc.commutes(CircuitInstruction(CZGate(), (qubits[0], qubits[2])))
        assert len(lc.operations) == 1

    with subtests.test("cx(0,1)"):
        assert lc.commutes(CircuitInstruction(CXGate(), (qubits[0], qubits[1])))
        assert len(lc.operations) == 1


def test_zzz(subtests):
    """Test commutation of different instructions with ZZZ Pauli.

    Args:
        subtests: the pytest-subtests fixture.
    """
    qubits = list([Qubit() for _ in range(3)])
    operations = [(PauliGate("ZZZ"), qubits)]
    lc = LightCone(set(qubits), operations)

    with subtests.test("cz(0,2)"):
        assert lc.commutes(CircuitInstruction(CZGate(), (qubits[0], qubits[2])))
        assert len(lc.operations) == 1

    with subtests.test("cx(0,1)"):
        assert not lc.commutes(CircuitInstruction(CXGate(), (qubits[0], qubits[1])))
        assert len(lc.operations) == 2
