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

"""Tests for the Davidson solver."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_slc.utils.davidson import get_extremal_eigenvalue


def test_davidson() -> None:
    """Test finding the extremal eigenvalue of an operator using the Davidson algorithm."""
    pytest.importorskip("pyscf")

    spo = SparsePauliOp.from_sparse_list(
        [("ZX", [0, 3], 0.2), ("Y", [2], 0.3), ("XYZ", [3, 5, 2], 1.34)], num_qubits=6
    )
    converged, eigval = get_extremal_eigenvalue(spo, tol=1e-5)
    assert converged
    assert np.isclose(eigval, -1.57317)


def test_davidson_accepts_legacy_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test legacy PySCF-style solver kwargs are still accepted."""
    seen_kwargs = {}

    def davidson1(_, _x0, _precond, **kwargs):
        seen_kwargs.update(kwargs)
        return np.array([True]), np.array([-1.0]), None

    monkeypatch.setitem(
        sys.modules,
        "pyscf",
        SimpleNamespace(lib=SimpleNamespace(davidson1=davidson1)),
    )

    spo = SparsePauliOp.from_list([("ZI", 1.0)])
    converged, eigval = get_extremal_eigenvalue(
        spo, max_cycle=10, max_space=4, lindep=1e-11, max_memory=100
    )
    assert converged[0]
    assert np.isclose(eigval, -1.0)
    assert seen_kwargs["max_cycle"] == 10
    assert seen_kwargs["max_space"] == 4
    assert seen_kwargs["lindep"] == 1e-11
    assert seen_kwargs["max_memory"] == 100
