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
from typing import cast

import numpy as np
from qiskit.quantum_info import PauliList, SparsePauliOp

from .. import _accelerate
from .reduce_op import _reduce_operator

# Largest ``p + c`` (log2 of the reduced-operator size) handled by dense diagonalization; above this
# the reduced operator is solved with iterative Davidson instead. Dense ``eigvalsh`` computes the
# whole spectrum in ``O((2^(p+c))^3)`` when only the smallest eigenvalue is needed, so it is only
# worth it while that is cheap: ``p + c = 8`` is a single 256x256 solve (~10ms). Beyond this,
# iterative (which targets just the extremal eigenvalue via sparse matvecs) is far faster -- e.g. a
# generic ``p + c = 12`` operator takes ~50s dense vs a fraction of a second iterative.
_MAX_REDUCED_LOG2_DIM = 8


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
    commuting directions. When this reduced operator is small enough it is diagonalized densely,
    giving a result exact to machine precision. Otherwise, the reduced operator is instead solved
    with the compiled Rust Davidson solver (a diagonally-preconditioned Davidson iteration). This
    path is accurate but not exact, and the arguments below affect only it.

    Args:
        spo: the Hermitian operator whose most-negative eigenvalue (and hence spectral norm) to
            compute.
        tol: the convergence threshold on the residual norm of the iterative solver.
        max_cycle: the maximum number of iterations of the iterative solver.
        max_space: the maximum size of the subspace built up by the iterative solver.
        lindep: the threshold below which a new subspace vector is considered linearly dependent on
            the existing subspace (and, thus, discarded).
        kwargs: **ignored!** Any additional keyword arguments are parsed for backwards compatibility
            but do not have any effect at runtime and, thus, are being ignored!

    Returns:
        A ``(converged, eigenvalue)`` pair. ``converged`` reports whether the computation succeeded
        (may be ``False`` if Davidson fails to converge). ``eigenvalue`` is the most-negative
        eigenvalue of ``spo``.
    """
    if len(kwargs) > 0:
        warnings.warn(
            f"These keyword arguments do not have any effect and are ignored: {kwargs}",
            category=UserWarning,
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
    elif reduced_op.num_qubits > _MAX_REDUCED_LOG2_DIM:
        # Large, with anticommuting pairs -> hand the whole operator to Davidson.
        result = _davidson_extremal_eigenvalue(
            reduced_op,
            tol=tol,
            max_cycle=max_cycle,
            max_space=max_space,
            lindep=lindep,
        )
    else:
        # Small: reduced_op is block-diagonal over the 2^c commuting-Z sectors. Diagonalize the
        # p-qubit block in each sector and take the overall minimum. A term's sign in a sector is -1
        # to the parity of the trailing Z's (commuting generators) it carries.
        paulis = reduced_op.paulis
        commuting_gen_mask = paulis.z[:, p:]  # (K, c)
        sector_bits = ((np.arange(1 << c)[:, None] >> np.arange(c)) & 1).astype(
            np.uint8
        )  # (2^c, c)
        parity = (sector_bits @ commuting_gen_mask.T.astype(np.uint8)) & 1  # (2^c, K)
        sector_sign = np.where(parity, np.int8(-1), np.int8(1))  # (2^c, K)
        block = SparsePauliOp(
            PauliList.from_symplectic(paulis.z[:, :p], paulis.x[:, :p]), reduced_op.coeffs
        )
        base_coeffs = block.coeffs.copy()
        lowest = np.inf
        for signs in sector_sign:
            block.coeffs = base_coeffs * signs
            mat = block.to_matrix(force_serial=True)  # serial: this runs inside a process pool
            lowest = min(lowest, float(np.linalg.eigvalsh(mat)[0]))
        result = True, lowest

    return result


def _davidson_extremal_eigenvalue(
    spo: SparsePauliOp,
    *,
    tol: float,
    max_cycle: int,
    max_space: int,
    lindep: float,
) -> tuple[bool, float]:
    """Iterative Davidson fallback, dispatched to the compiled Rust solver.

    The default ``tol`` is tight because the eigenvalue error runs well above ``tol`` (roughly
    ``tol`` divided by the spectral gap, which is small for these near-degenerate commutators).
    """
    spmat = spo.to_matrix(sparse=True, force_serial=True).tocsr()
    dim = spmat.shape[0]
    data = spmat.data.astype(np.complex128)
    diag = spmat.diagonal().astype(np.complex128)
    seed = _initial_guess((dim,)).astype(np.complex128)

    return _accelerate.davidson_smallest(
        spmat.indptr.astype(np.int64),
        spmat.indices.astype(np.int64),
        np.ascontiguousarray(data.real),
        np.ascontiguousarray(data.imag),
        np.ascontiguousarray(diag.real),
        np.ascontiguousarray(diag.imag),
        np.ascontiguousarray(seed.real),
        np.ascontiguousarray(seed.imag),
        dim,
        float(tol),
        int(max_cycle),
        int(max_space),
        float(lindep),
    )


def _initial_guess(shape: tuple[int, ...]) -> np.ndarray:
    """Produces a deterministic normalized starting vector of the requested shape.

    A fixed-seed local generator is used so that the
    Davidson iteration is reproducible: the same operator always yields the same result, independent
    of any surrounding random state. A pseudo-random (rather than constant) vector is used to avoid
    initial guesses that are accidentally orthogonal to the target eigenvector.

    Args:
        shape: the requested shape.

    Returns:
        A unit-norm array of complex values with their real and imaginary parts lying in the interval
        ``[0, 1)``.
    """
    rng = np.random.default_rng(0)

    norm = 0.0
    while norm == 0:
        x = rng.random(shape[0]) + 1.0j * rng.random(shape[0])
        norm = cast(float, np.linalg.norm(x))

    return x / norm
