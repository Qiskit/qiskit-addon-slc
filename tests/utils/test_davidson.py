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

import numpy as np
import pytest
import qiskit_addon_slc.utils.davidson as davidson
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit_addon_slc.utils.davidson import get_extremal_eigenvalue


def _dense_min(spo: SparsePauliOp) -> float:
    """Most-negative eigenvalue computed by brute-force dense diagonalization."""
    return float(np.linalg.eigvalsh(spo.to_matrix())[0])


def test_davidson() -> None:
    """Test finding the extremal eigenvalue of an operator using the Davidson algorithm."""
    spo = SparsePauliOp.from_sparse_list(
        [("ZX", [0, 3], 0.2), ("Y", [2], 0.3), ("XYZ", [3, 5, 2], 1.34)], num_qubits=6
    )
    converged, eigval = get_extremal_eigenvalue(spo, tol=1e-5)
    assert converged
    assert np.isclose(eigval, -1.57317)


DENSE_OPERATORS = {
    "single_pauli": SparsePauliOp(["XYZ"], [1.3]),
    "diagonal": SparsePauliOp(["ZZI", "IZZ", "ZIZ", "IIZ"], [0.5, -1.2, 0.3, 0.9]),
    "one_pair_plus_central": SparsePauliOp(["XI", "ZI", "IZ"], [0.7, -0.4, 1.1]),
    "mixed": SparsePauliOp(["XII", "ZII", "IXX", "IZZ", "IYI"], [1.0, 0.5, -0.3, 0.8, 0.6]),
    # Positive-definite: a large identity term shifts the whole spectrum above zero, so the
    # per-sector minimum must not be clamped at 0.0.
    "positive_definite": SparsePauliOp(["XI", "ZI", "II"], [0.5, 0.5, 3.0]),
    # Positive-definite and fully commuting -> takes the p == 0 diagonal path.
    "positive_definite_diagonal": SparsePauliOp(["ZI", "IZ", "II"], [0.5, 0.25, 2.0]),
}


@pytest.mark.parametrize("name", list(DENSE_OPERATORS))
def test_exact_paths_match_dense(name: str) -> None:
    """The exact paths (``p == 0`` diagonal and dense per-sector) are exact and always converge."""
    spo = DENSE_OPERATORS[name]
    converged, eigval = get_extremal_eigenvalue(spo)
    assert converged
    assert np.isclose(eigval, _dense_min(spo), atol=1e-10)


def test_iterative_path_matches_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the iterative fallback (via a low cutoff) and check it matches dense diagonalization."""
    monkeypatch.setattr(davidson, "_MAX_REDUCED_LOG2_DIM", 1)
    np.random.seed(0)  # the Davidson initial guess is random
    # p = 1 (X0, Z0 anticommute), c = 2 (IXX, IZZ central) -> p + c = 3 > cutoff -> iterative.
    spo = SparsePauliOp(["XII", "ZII", "IXX", "IZZ"], [1.0, 0.5, -0.3, 0.8])
    converged, eigval = get_extremal_eigenvalue(spo, tol=1e-10)
    assert converged
    assert np.isclose(eigval, _dense_min(spo), atol=1e-6)


def test_large_diagonal_matches_dense() -> None:
    """A many-qubit fully-commuting (p == 0) operator takes the diagonal path and stays exact."""
    rng = np.random.default_rng(0)
    z = rng.integers(0, 2, (40, 10), dtype=bool)  # Z-only Paulis -> diagonal, 10 > cutoff
    spo = SparsePauliOp(PauliList.from_symplectic(z, np.zeros_like(z)), rng.standard_normal(40))
    converged, eigval = get_extremal_eigenvalue(spo)
    assert converged
    assert np.isclose(eigval, _dense_min(spo), atol=1e-10)
