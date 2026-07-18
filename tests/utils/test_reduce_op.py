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

"""Tests for the symplectic operator reduction."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit_addon_slc.utils.reduce_op import _reduce_operator


def _random_hermitian(num_qubits: int, num_terms: int, seed: int) -> SparsePauliOp:
    """A random Hermitian Pauli sum (real coefficients on random Pauli strings)."""
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=(num_terms, num_qubits), dtype=bool)
    x = rng.integers(0, 2, size=(num_terms, num_qubits), dtype=bool)
    paulis = PauliList.from_symplectic(z, x)
    return SparsePauliOp(paulis, rng.standard_normal(num_terms)).simplify(atol=0)


def _reduced_spectrum(spo: SparsePauliOp) -> np.ndarray:
    """Full spectrum of the reduced operator gathered over all central sign-sectors."""
    logicals, amps, exps = _reduce_operator(spo)
    p = logicals.num_qubits
    c = exps.shape[1]
    sector_bits = (np.arange(1 << c)[:, None] >> np.arange(c)) & 1
    sector_sign = 1 - 2 * ((sector_bits @ exps.T) & 1)
    if p == 0:
        # Each sector is a single number: the signed sum of coefficients.
        return (sector_sign * amps.real).sum(axis=1)
    base = SparsePauliOp(logicals, amps)
    base_coeffs = base.coeffs.copy()
    eigs = []
    for signs in sector_sign:
        base.coeffs = base_coeffs * signs
        eigs.append(np.linalg.eigvalsh(base.to_matrix()))
    return np.concatenate(eigs)


OPERATORS = {
    "single_pauli": SparsePauliOp(["XYZ"], [1.3]),
    "diagonal": SparsePauliOp(["ZZI", "IZZ", "ZIZ", "IIZ"], [0.5, -1.2, 0.3, 0.9]),
    "one_pair_plus_central": SparsePauliOp(["XI", "ZI", "IZ"], [0.7, -0.4, 1.1]),
    "structured": SparsePauliOp.from_sparse_list(
        [("ZX", [0, 3], 0.2), ("Y", [2], 0.3), ("XYZ", [3, 5, 2], 1.34)], num_qubits=6
    ),
    "random5": _random_hermitian(5, 25, 0),
    "random6": _random_hermitian(6, 40, 1),
}


@pytest.mark.parametrize("name", list(OPERATORS))
def test_reduced_spectrum_matches_original(name: str) -> None:
    """The reduced operator's spectrum equals the original's, up to uniform degeneracy.

    The operator is unitarily equivalent to the reduced block tensored with an identity of dimension
    ``2^(n - p - c)``, so every reduced eigenvalue appears in the full spectrum with that fixed
    multiplicity. Tiling the reduced spectrum by it must reproduce the full spectrum exactly.
    """
    spo = OPERATORS[name]
    reduced = _reduced_spectrum(spo)
    original = np.linalg.eigvalsh(spo.to_matrix())
    assert original.size % reduced.size == 0
    multiplicity = original.size // reduced.size
    np.testing.assert_allclose(
        np.sort(original), np.sort(np.tile(reduced, multiplicity)), atol=1e-9
    )


@pytest.mark.parametrize("name", list(OPERATORS))
def test_reduce_operator_shapes(name: str) -> None:
    """Reduced size ``2^(p + c)`` never exceeds the full ``2^n``, and outputs are consistent."""
    spo = OPERATORS[name]
    logicals, amps, exps = _reduce_operator(spo)
    p, c = logicals.num_qubits, exps.shape[1]
    assert p + c <= spo.num_qubits
    assert amps.shape == (len(spo),)
    assert exps.shape == (len(spo), c)


def test_fully_commuting_has_no_pairs() -> None:
    """A fully-commuting (diagonalizable-in-place) operator reduces to ``p == 0``."""
    logicals, _, exps = _reduce_operator(OPERATORS["diagonal"])
    assert logicals.num_qubits == 0
    assert exps.shape[1] >= 1
