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
import scipy.sparse
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit_addon_slc.utils.davidson import get_extremal_eigenvalue
from qiskit_addon_slc.utils.reduce_op import _reduce_operator


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
    # Only anticommuting pairs (c == 0): a single block covering the whole reduced operator.
    "pairs_only": SparsePauliOp(["XI", "ZI", "IX", "IZ"], [0.7, -0.4, 1.1, 0.2]),
    "mixed": SparsePauliOp(["XII", "ZII", "IXX", "IZZ", "IYI"], [1.0, 0.5, -0.3, 0.8, 0.6]),
    # Positive-definite: a large identity term shifts the whole spectrum above zero, so the
    # per-sector minimum must not be clamped at 0.0.
    "positive_definite": SparsePauliOp(["XI", "ZI", "II"], [0.5, 0.5, 3.0]),
    # Positive-definite and fully commuting -> takes the p == 0 diagonal path.
    "positive_definite_diagonal": SparsePauliOp(["ZI", "IZ", "II"], [0.5, 0.25, 2.0]),
    # Multiples of the identity reduce to a 0-qubit operator; its single eigenvalue is the sum of
    # the coefficients, which must survive the reduction.
    "identity_only": SparsePauliOp(["II"], [-0.73]),
    "identity_only_unsimplified": SparsePauliOp(["II", "II"], [-0.5, -0.23]),
    "identity_only_positive": SparsePauliOp(["III"], [2.5]),
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
    monkeypatch.setattr(davidson, "_MAX_SECTOR_LOG2_DIM", 0)
    # p = 1 (X0, Z0 anticommute), c = 2 (IXX, IZZ central) -> p = 1 > cutoff -> iterative.
    spo = SparsePauliOp(["XII", "ZII", "IXX", "IZZ"], [1.0, 0.5, -0.3, 0.8])
    converged, eigval = get_extremal_eigenvalue(spo, tol=1e-10)
    assert converged
    assert np.isclose(eigval, _dense_min(spo), atol=1e-6)


def _near_degenerate_sectors(seed: int, p: int, c: int) -> SparsePauliOp:
    """A reduced-form operator whose ``2^c`` sector minima differ by only ``1e-4..1e-1``.

    Half the terms act on the ``p`` leading qubits alone; the other half carry trailing ``Z``'s
    and split the sectors apart with small coefficients.
    """
    rng = np.random.default_rng(seed)
    num_terms = 24
    z = np.zeros((num_terms, p + c), dtype=bool)
    x = np.zeros((num_terms, p + c), dtype=bool)
    z[:, :p] = rng.random((num_terms, p)) < 0.5
    x[:, :p] = rng.random((num_terms, p)) < 0.5
    z[num_terms // 2 :, p:] = rng.random((num_terms - num_terms // 2, c)) < 0.5
    coeffs = rng.standard_normal(num_terms)
    coeffs[num_terms // 2 :] *= 10.0 ** rng.uniform(-4, -1, num_terms - num_terms // 2)
    return SparsePauliOp(PauliList.from_symplectic(z, x), coeffs).simplify()


@pytest.mark.parametrize(("seed", "p"), [(seed, p) for seed in range(4) for p in (1, 2, 3)])
def test_near_degenerate_sectors_are_exact(seed: int, p: int) -> None:
    """Block-diagonal operators with near-degenerate sector minima are solved exactly.

    These have more than ``_MAX_SECTOR_LOG2_DIM`` qubits in total, so they used to go to the
    iterative solver, which mostly gave up on them (``converged=False``).
    """
    spo = _near_degenerate_sectors(seed, p, 9 - p)
    converged, eigval = get_extremal_eigenvalue(spo)
    assert converged
    assert np.isclose(eigval, _dense_min(spo), atol=1e-10)


def test_sector_chunks_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Processing the sectors in chunks, down to one at a time, does not change the result."""
    spo = _near_degenerate_sectors(0, 2, 5)
    expected = get_extremal_eigenvalue(spo)
    monkeypatch.setattr(davidson, "_MAX_SECTOR_CHUNK_ENTRIES", 1)
    assert get_extremal_eigenvalue(spo) == expected


def _davidson(spo: SparsePauliOp, tol: float = 1e-10) -> tuple[bool, float]:
    """Run the iterative solver directly on ``spo``, without the symplectic reduction."""
    return davidson._davidson_extremal_eigenvalue(
        spo, tol=tol, max_cycle=500, max_space=12, lindep=1e-11
    )


def test_iterative_path_rejects_interior_eigenvalue() -> None:
    """A residual-converged interior eigenpair must not be reported as the smallest eigenvalue.

    The Jacobi preconditioner pulls the iteration towards the eigenvalue nearest the current Ritz
    value. On this operator it used to settle on the exact eigenpair at -4.685 and report it as
    converged, understating the norm, while the smallest eigenvalue is -6.602.
    """
    spo = SparsePauliOp.from_list(
        [
            ("IIZIZII", -0.9584991779408014),
            ("IZIIZII", 1.6669776941321173),
            ("IZZZIZI", 1.2237204510824284),
            ("ZZIIIZX", -0.3653449625439495),
            ("ZZZZIII", -2.161505965703816),
            ("ZZZZZZZ", -0.46508930674903803),
        ]
    )
    converged, eigval = _davidson(spo, tol=1e-6)
    assert converged
    assert np.isclose(eigval, _dense_min(spo), atol=1e-6)


def test_iterative_path_converges_on_hard_operator() -> None:
    """The iterative solver converges on a reduced operator on which it used to stall.

    Restarting from a single Ritz vector and judging linear dependence on the unnormalized correction
    made the iteration give up here, reporting ``converged=False``. (The public API now solves this
    operator exactly, block by block, so it is passed to the solver directly.)
    """
    spo = SparsePauliOp.from_list(
        [
            ("ZIIZZIIXY", 1.34),
            ("IZIZZIIIX", 0.3),
            ("IZZZIZIIX", 1.73),
            ("ZIZIIIIYI", -0.05),
            ("IZIIZZIYY", 0.37),
            ("ZZIZZIIZI", -1.17),
            ("IZZZIZZXX", -1.91),
            ("ZIIZZZIZI", 0.21),
            ("ZZIIIIIXX", -1.05),
            ("IZIZIIZXZ", 1.37),
            ("ZZZIZZZIY", -0.3),
        ]
    )
    converged, eigval = _davidson(_reduce_operator(spo)[0])
    assert converged
    assert np.isclose(eigval, _dense_min(spo), atol=1e-9)


def _solve(matrix, seed=None, *, indptr=None, indices=None, data=None, **options):
    """Call the Rust solver directly on a dense ``matrix``, optionally replacing its CSR arrays."""
    csr = scipy.sparse.csr_matrix(np.asarray(matrix, dtype=np.complex128))
    options = {"tol": 1e-10, "max_cycle": 500, "max_space": 12, "lindep": 1e-11} | options
    return davidson._accelerate.davidson_smallest(
        csr.indptr.astype(np.int64) if indptr is None else np.asarray(indptr, dtype=np.int64),
        csr.indices.astype(np.int64) if indices is None else np.asarray(indices, dtype=np.int64),
        csr.data if data is None else np.asarray(data, dtype=np.complex128),
        davidson._initial_guess(csr.shape[0]) if seed is None else np.asarray(seed, np.complex128),
        **options,
    )


def test_solver_converges_on_diagonal_matrix() -> None:
    """The Rust solver converges on a diagonal matrix, where plain Jacobi corrections vanish."""
    converged, eigval = _solve(np.diag(np.arange(-4, 4, dtype=float)))
    assert converged
    assert np.isclose(eigval, -4.0, atol=1e-10)


def test_solver_normalizes_seed() -> None:
    """The result does not depend on the norm of the start vector."""
    matrix = np.diag(np.arange(-4, 4, dtype=float)) + 0.1 * (np.eye(8, k=1) + np.eye(8, k=-1))
    seed = davidson._initial_guess(8)
    expected = _solve(matrix, seed)
    assert expected[0]
    assert np.isclose(expected[1], np.linalg.eigvalsh(matrix)[0], atol=1e-10)
    for scale in (2.0, 1e-8, 1e8):
        converged, eigval = _solve(matrix, scale * seed)
        assert converged
        assert np.isclose(eigval, expected[1], atol=1e-10)


def test_solver_converges_at_tiny_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``tol`` far below the rounding noise of the operator still converges."""
    monkeypatch.setattr(davidson, "_MAX_SECTOR_LOG2_DIM", 0)
    spo = SparsePauliOp(["XII", "ZII", "IXX", "IZZ"], [1.0, 0.5, -0.3, 0.8])
    converged, eigval = get_extremal_eigenvalue(spo, tol=1e-18)
    assert converged
    assert np.isclose(eigval, _dense_min(spo), atol=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"seed": np.zeros(4)}, "`seed` must be non-zero and finite"),
        ({"seed": [1, np.nan, 0, 0]}, "`seed` must be non-zero and finite"),
        ({"seed": np.ones(3)}, "`seed` must have length"),
        ({"indptr": [1, 2, 3, 4, 4]}, r"`indptr\[0\]` must be 0"),
        ({"indptr": [0, 2, 1, 3, 4]}, "`indptr` must be non-decreasing"),
        ({"indptr": [0]}, "`indptr` must have at least two entries"),
        ({"indptr": [0, 1, 2, 3, 3]}, "must both have length `indptr"),
        ({"indices": [0, 1, 2, 4]}, "`indices` entries must be in"),
        ({"indices": [0, 1, 2, -1]}, "`indices` entries must be in"),
        ({"data": [1, 2, 3, np.inf]}, "`data` entries must be finite"),
        ({"data": [1, 2, 3]}, "must both have length `indptr"),
        ({"tol": 0.0}, "`tol` must be positive and finite"),
        ({"max_space": 1}, "`max_space` must be at least 2"),
        ({"lindep": 1.0}, r"`lindep` must be in `\[0, 1\)`"),
    ],
)
def test_solver_rejects_invalid_input(kwargs: dict, match: str) -> None:
    """Inconsistent arrays or out-of-range options raise instead of being silently misread."""
    with pytest.raises(ValueError, match=match):
        _solve(np.diag([1.0, 2.0, 3.0, 4.0]), **kwargs)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"tol": 0.0}, "tol must be positive and finite"),
        ({"tol": np.nan}, "tol must be positive and finite"),
        ({"max_cycle": -1}, "max_cycle must be positive"),
        ({"max_space": 1}, "max_space must be at least 2"),
        ({"lindep": -0.1}, r"lindep must be in \[0, 1\)"),
    ],
)
@pytest.mark.parametrize("name", ["one_pair_plus_central", "diagonal"])
def test_invalid_options_raise_on_every_path(options: dict, match: str, name: str) -> None:
    """Invalid solver options raise even for operators that never reach the iterative solver."""
    with pytest.raises(ValueError, match=match):
        get_extremal_eigenvalue(DENSE_OPERATORS[name], **options)


def test_solver_reports_overflow_as_non_convergence() -> None:
    """Entries that overflow during the iteration end it unconverged instead of hanging or panicking."""
    big = 1e308
    matrix = np.array([[0.0, big, 0.0], [big, 0.0, big], [0.0, big, 0.0]])
    converged, _ = _solve(matrix)
    assert not converged


def test_solver_handles_one_dimensional_matrix() -> None:
    """A 1x1 matrix converges to its entry, as there is no complement to search for lower ones."""
    converged, eigval = _solve([[-2.5]])
    assert converged
    assert np.isclose(eigval, -2.5, atol=1e-12)


def test_extra_kwargs_are_deprecated() -> None:
    """Keyword arguments once forwarded to PySCF are ignored, with a deprecation warning."""
    spo = SparsePauliOp(["XI", "ZI", "IZ"], [0.7, -0.4, 1.1])
    with pytest.warns(DeprecationWarning, match="Ignoring: follow_state, nroots"):
        converged, eigval = get_extremal_eigenvalue(spo, nroots=1, follow_state=True)
    assert (converged, eigval) == get_extremal_eigenvalue(spo)


def _random_reduced(seed: int) -> SparsePauliOp:
    """A random operator of the shape the reduction produces.

    That is ``p`` anticommuting pairs on the leading qubits and commuting ``Z`` generators on the
    trailing ones.
    """
    rng = np.random.default_rng(seed)
    p, c = int(rng.integers(1, 6)), int(rng.integers(2, 6))
    num_terms = int(rng.integers(6, 25))
    z = np.zeros((num_terms, p + c), dtype=bool)
    x = np.zeros((num_terms, p + c), dtype=bool)
    z[:, :p] = rng.random((num_terms, p)) < 0.4
    x[:, :p] = rng.random((num_terms, p)) < 0.4
    z[:, p:] = rng.random((num_terms, c)) < 0.5
    return SparsePauliOp(PauliList.from_symplectic(z, x), rng.standard_normal(num_terms)).simplify()


@pytest.mark.parametrize("seed", range(40))
def test_iterative_path_is_sound(seed: int) -> None:
    """Whenever the iterative solver reports convergence, its eigenvalue is the smallest one.

    ``converged=False`` is allowed, since the caller then falls back to the triangle inequality, but
    a wrong converged value would make the bound unsound.
    """
    spo = _random_reduced(seed)
    for tol in (1e-6, 1e-10):
        converged, eigval = _davidson(spo, tol=tol)
        if converged:
            assert np.isclose(eigval, _dense_min(spo), atol=max(1e-6, 1e3 * tol))


@pytest.mark.parametrize("tol", [1e-6, 1e-10])
def test_iterative_path_mostly_converges(tol: float) -> None:
    """The iterative solver converges on most operators, rather than falling back by default.

    It converges on 38 of these 40; the margin allows for platform-dependent rounding.
    """
    converged = sum(_davidson(_random_reduced(seed), tol=tol)[0] for seed in range(40))
    assert converged >= 36


def test_large_diagonal_matches_dense() -> None:
    """A many-qubit fully-commuting (p == 0) operator takes the diagonal path and stays exact."""
    rng = np.random.default_rng(0)
    z = rng.integers(0, 2, (40, 10), dtype=bool)  # Z-only Paulis -> diagonal, 10 > cutoff
    spo = SparsePauliOp(PauliList.from_symplectic(z, np.zeros_like(z)), rng.standard_normal(40))
    converged, eigval = get_extremal_eigenvalue(spo)
    assert converged
    assert np.isclose(eigval, _dense_min(spo), atol=1e-10)
