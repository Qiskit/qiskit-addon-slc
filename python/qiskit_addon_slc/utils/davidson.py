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

"""Extremal-eigenvalue solver with a symplectic-reduction fast path."""

from __future__ import annotations

import warnings

import numpy as np
from qiskit.quantum_info import PauliList, SparsePauliOp

from .. import _accelerate
from .reduce_op import _reduce_operator

# Largest ``p`` (log2 of the size of each of the ``2^c`` diagonal blocks of the reduced operator) for
# which the blocks are diagonalized exactly; above this the reduced operator is solved with
# iterative Davidson instead. The exact path costs ``O(2^c * (2^p)^3)``. For random 40-term
# operators of up to 14 qubits it beat Davidson for every ``p <= 8`` (e.g. ``p = 8, c = 6`` took
# ~0.3s against ~1.5s), while Davidson took over from ``p = 9`` or ``10``.
_MAX_SECTOR_LOG2_DIM = 8

# Upper bound on the number of matrix entries (16 bytes each) of the diagonal blocks that the exact
# path holds in memory at once; the sectors are processed in chunks below it.
_MAX_SECTOR_CHUNK_ENTRIES = 1 << 20


def get_extremal_eigenvalue(
    spo: SparsePauliOp,
    *,
    tol: float = 1e-10,
    max_cycle: int = 500,
    max_space: int = 12,
    lindep: float = 1e-11,
    **kwargs,
) -> tuple[bool, float]:
    """Compute the spectral norm of a Hermitian Pauli operator (as a signed extremal eigenvalue).

    Given a Hermitian operator written as a weighted sum of Paulis, ``C = sum_k a_k P_k``, this
    returns its most-negative eigenvalue. Because such an operator has a spectrum that is symmetric
    about zero, the magnitude of that eigenvalue is the operator's spectral norm ``||C||`` (its
    largest eigenvalue in absolute value) -- take ``abs()`` of the result if that is what you need.
    In this addon it is used to evaluate the commutator-norm error bound ``||[E, O]||`` for an error
    Pauli ``E`` and observable ``O``.

    This exploits how a sum of Paulis lives in a space of dimension ``2^(p + c)``, where ``p`` is the
    number of independent anticommuting pairs the Paulis generate and ``c`` counts the remaining
    commuting directions. The commuting directions are conserved, so this reduced operator is
    block-diagonal, with ``2^c`` blocks of size ``2^p``. When these blocks are small enough each is
    diagonalized exactly, giving a result exact to machine precision. Otherwise, the reduced
    operator is instead solved with the compiled Rust Davidson solver (a diagonally-preconditioned
    Davidson iteration, which checks a converged eigenvalue with a short Lanczos iteration for a
    lower one before reporting it). This path is accurate but not exact, and the arguments below
    affect only it.

    Note the sign convention: the returned eigenvalue is the most-negative one, so the spectral norm
    is its absolute value.

    .. plot::
       :include-source:
       :nofigs:
       :context: reset

       >>> from qiskit.quantum_info import SparsePauliOp
       >>> from qiskit_addon_slc.utils import get_extremal_eigenvalue

       >>> operator = SparsePauliOp(["ZI", "IZ"], [1.0, 0.5])
       >>> converged, eigenvalue = get_extremal_eigenvalue(operator)
       >>> converged
       True
       >>> float(eigenvalue)
       -1.5
       >>> abs(eigenvalue)  # the spectral norm
       1.5

    Args:
        spo: the Hermitian operator whose most-negative eigenvalue (and hence spectral norm) to
            compute.
        tol: the convergence threshold of the iterative solver. It bounds the change of the
            eigenvalue between iterations and, unless rounding errors at the scale of the operator
            norm dominate, the residual norm of the eigenvector.
        max_cycle: the maximum number of iterations of the iterative solver.
        max_space: the maximum size of the subspace built up by the iterative solver. Once reached,
            the subspace is restarted from its ``max_space // 2`` lowest Ritz vectors.
        lindep: the fraction of its squared norm below which a new subspace vector, after being
            orthogonalized against the existing subspace, is considered linearly dependent on it
            (and, thus, discarded).
        kwargs: **deprecated and ignored.** Additional keyword arguments used to be passed on to the
            ``pyscf.lib.davidson1`` solver that this function no longer uses. They are still accepted
            for backwards compatibility, but have no effect and raise a ``DeprecationWarning``.

    Returns:
        A ``(converged, eigenvalue)`` pair. ``converged`` reports whether the computation succeeded
        (may be ``False`` if Davidson fails to converge). ``eigenvalue`` is the most-negative
        eigenvalue of ``spo``.

    Raises:
        ValueError: if ``tol`` is not positive and finite, ``max_cycle`` is not positive,
            ``max_space`` is less than 2, or ``lindep`` is not in ``[0, 1)``. These are checked for
            every ``spo``, even one that is solved exactly.
    """
    if not (np.isfinite(tol) and tol > 0):
        raise ValueError(f"tol must be positive and finite, got {tol}.")
    if max_cycle < 1:
        raise ValueError(f"max_cycle must be positive, got {max_cycle}.")
    if max_space < 2:
        raise ValueError(f"max_space must be at least 2, got {max_space}.")
    if not 0 <= lindep < 1:
        raise ValueError(f"lindep must be in [0, 1), got {lindep}.")

    if kwargs:
        warnings.warn(
            "Passing additional keyword arguments to get_extremal_eigenvalue() is deprecated as of "
            "qiskit-addon-slc 0.2.0 and will be removed no sooner than 3 months after the release "
            "date. They used to be passed on to pyscf.lib.davidson1, which is no longer used, and "
            f"have no effect. Ignoring: {', '.join(sorted(kwargs))}.",
            category=DeprecationWarning,
            stacklevel=2,
        )

    reduced_op, c = _reduce_operator(spo)
    p = reduced_op.num_qubits - c

    if reduced_op.num_qubits == 0:
        # Edge case: ``spo`` was multiple of identity. Handled because ``to_matrix()`` on
        # 0-qubit operator discards coefficients.
        result = True, float(np.sum(reduced_op.coeffs).real)
    elif p == 0:
        # Fully diagonal (only commuting generators): the answer is the smallest diagonal entry. A
        # sparse build stays O(2^c) in memory (only c <= eigval_max_qubits reaches here).
        diagonal = reduced_op.to_matrix(sparse=True, force_serial=True).diagonal()
        result = True, float(diagonal.real.min())
    elif p > _MAX_SECTOR_LOG2_DIM:
        # Large blocks: hand the whole operator to Davidson.
        result = _davidson_extremal_eigenvalue(
            reduced_op,
            tol=tol,
            max_cycle=max_cycle,
            max_space=max_space,
            lindep=lindep,
        )
    else:
        result = True, _sector_minimum(reduced_op, p, c)

    return result


def _sector_minimum(reduced_op: SparsePauliOp, p: int, c: int) -> float:
    """Exact smallest eigenvalue of a reduced operator, block by block.

    ``reduced_op`` is block-diagonal over the ``2^c`` sign sectors of its trailing commuting ``Z``
    generators. In each sector, a term acts on the ``p`` leading qubits with its coefficient
    multiplied by ``-1`` to the parity of the trailing ``Z``'s it carries. This builds the blocks of
    many sectors at once and diagonalizes them as a batch, keeping the overall minimum.
    """
    paulis = reduced_op.paulis
    dim = 1 << p
    # A p-qubit Pauli is a phased permutation matrix, with exactly one entry per row; record each
    # term's column and value per row. Serial: this runs inside a process pool.
    block_paulis = PauliList.from_symplectic(paulis.z[:, :p], paulis.x[:, :p])
    rows = np.arange(dim)
    cols = np.empty((len(block_paulis), dim), dtype=np.intp)
    vals = np.empty((len(block_paulis), dim), dtype=np.complex128)
    for k, pauli in enumerate(block_paulis):
        mat = pauli.to_matrix(sparse=True)
        cols[k] = mat.indices
        vals[k] = mat.data

    generators = paulis.z[:, p:].astype(np.int64)  # (K, c)
    sectors_per_chunk = max(1, _MAX_SECTOR_CHUNK_ENTRIES // (dim * dim))
    lowest = np.inf
    for start in range(0, 1 << c, sectors_per_chunk):
        sectors = np.arange(start, min(start + sectors_per_chunk, 1 << c))
        bits = (sectors[:, None] >> np.arange(c)) & 1  # (S, c)
        signs = 1 - 2 * ((bits @ generators.T) & 1)  # (S, K)
        coeffs = signs * reduced_op.coeffs  # (S, K)
        blocks = np.zeros((len(sectors), dim, dim), dtype=np.complex128)
        for k in range(len(block_paulis)):
            blocks[:, rows, cols[k]] += coeffs[:, k, None] * vals[k]
        lowest = min(lowest, float(np.linalg.eigvalsh(blocks)[:, 0].min()))
    return lowest


def _davidson_extremal_eigenvalue(
    spo: SparsePauliOp,
    *,
    tol: float,
    max_cycle: int,
    max_space: int,
    lindep: float,
) -> tuple[bool, float]:
    """Iterative Davidson fallback, dispatched to the compiled Rust solver.

    The options are those of :func:`get_extremal_eigenvalue`, whose default ``tol`` is tight because
    the eigenvalue error runs well above ``tol`` (roughly ``tol`` divided by the spectral gap, which
    is small for these near-degenerate commutators).
    """
    spmat = spo.to_matrix(sparse=True, force_serial=True)  # already CSR
    # The solver borrows these arrays, so only convert (and copy) those not yet in its format.
    return _accelerate.davidson_smallest(
        np.ascontiguousarray(spmat.indptr, dtype=np.int64),
        np.ascontiguousarray(spmat.indices, dtype=np.int64),
        np.ascontiguousarray(spmat.data, dtype=np.complex128),
        _initial_guess(spmat.shape[0]),
        tol=float(tol),
        max_cycle=int(max_cycle),
        max_space=int(max_space),
        lindep=float(lindep),
    )


def _initial_guess(dim: int) -> np.ndarray:
    """Produces a deterministic normalized starting vector of length ``dim``.

    A fixed-seed local generator is used so that the Davidson iteration is reproducible: the same
    operator always yields the same result, independent of any surrounding random state. A
    pseudo-random (rather than constant) vector is used to avoid initial guesses that are
    accidentally orthogonal to the target eigenvector.

    Args:
        dim: the length of the vector.

    Returns:
        A unit-norm complex vector, whose real and imaginary parts are drawn uniformly from ``[0, 1)``
        before normalization.
    """
    rng = np.random.default_rng(0)
    x = rng.random(dim) + 1.0j * rng.random(dim)
    return x / float(np.linalg.norm(x))
