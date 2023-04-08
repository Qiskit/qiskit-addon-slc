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

"""Tests for the Davidson solver."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_slc.utils.davidson import get_extremal_eigenvalue


def test_davidson() -> None:
    """Test finding the extremal eigenvalue of an operator using the Davidson algorithm."""
    spo = SparsePauliOp.from_sparse_list(
        [("ZX", [0, 3], 0.2), ("Y", [2], 0.3), ("XYZ", [3, 5, 2], 1.34)], num_qubits=6
    )
    converged, eigval = get_extremal_eigenvalue(spo, tol=1e-5)
    assert converged
    assert np.isclose(eigval, -1.57317)
