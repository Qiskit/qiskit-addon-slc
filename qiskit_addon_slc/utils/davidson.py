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

from typing import cast

import numpy as np
import pyscf
from qiskit.quantum_info import PauliList, SparsePauliOp

from .reduce_op import _reduce_operator

# Largest ``p + c`` (log2 of the reduced-operator size) handled by dense diagonalization; above this
# the reduced operator is solved with iterative Davidson instead. Dense ``eigvalsh`` computes the
# whole spectrum in ``O((2^(p+c))^3)`` when only the smallest eigenvalue is needed, so it is only
# worth it while that is cheap: ``p + c = 8`` is a single 256x256 solve (~10ms). Beyond this,
# iterative (which targets just the extremal eigenvalue via sparse matvecs) is far faster -- e.g. a
# generic ``p + c = 12`` operator takes ~50s dense vs a fraction of a second iterative.
_MAX_REDUCED_LOG2_DIM = 8


def get_extremal_eigenvalue(spo: SparsePauliOp, **kwargs) -> tuple[bool, float]:
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
    with an iterative eigensolver (Davidson). This path is accurate but not exact, and the ``kwargs``
    below affect only it.

    Args:
        spo: the Hermitian operator whose most-negative eigenvalue (and hence spectral norm) to
            compute.
        kwargs: keyword arguments for the iterative fallback,
            :func:`~pyscf.lib.linalg_helper.davidson1` (defaults: ``tol=1e-10``, ``max_cycle=500``,
            ``max_space=12``, ``lindep=1e-11``, ``max_memory=2000``; anything else falls back to
            PySCF's own defaults). Ignored by the exact fast path.

    Returns:
        A ``(converged, eigenvalue)`` pair. ``converged`` reports whether the computation succeeded
        (may be ``False`` if Davidson fails to converge). ``eigenvalue`` is the most-negative
        eigenvalue of ``spo``.
    """
    reduced_op, c = _reduce_operator(spo)
    p = reduced_op.num_qubits - c

    if p == 0:
        # Fully diagonal (only commuting generators): the answer is the smallest diagonal entry. A
        # sparse build stays O(2^c) in memory (only c <= eigval_max_qubits reaches here).
        diagonal = reduced_op.to_matrix(sparse=True, force_serial=True).diagonal()
        result = True, float(diagonal.real.min())
    elif reduced_op.num_qubits > _MAX_REDUCED_LOG2_DIM:
        # Large, with anticommuting pairs -> hand the whole operator to Davidson.
        result = _davidson_extremal_eigenvalue(reduced_op, **kwargs)
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
        # Seed with +inf, not 0.0: a positive-definite operator has no sector minimum below zero, so
        # seeding at 0.0 would clamp the result and report 0.0 instead of the true minimum.
        lowest = float("inf")
        for signs in sector_sign:
            block.coeffs = base_coeffs * signs
            mat = block.to_matrix(force_serial=True)  # serial: this runs inside a process pool
            lowest = min(lowest, float(np.linalg.eigvalsh(mat)[0]))
        result = True, lowest

    return result


def _davidson_extremal_eigenvalue(spo: SparsePauliOp, **kwargs) -> tuple[bool, float]:
    """Iterative Davidson fallback.

    The default ``tol`` is tight because the eigenvalue error runs well above ``tol`` (roughly
    ``tol`` divided by the spectral gap, which is small for these near-degenerate commutators).
    """
    default_kwargs = {
        "tol": 1e-10,
        "max_cycle": 500,
        "max_space": 12,
        "lindep": 1e-11,
        "max_memory": 2000,
    }
    default_kwargs.update(kwargs)

    spmat = spo.to_matrix(sparse=True, force_serial=True)
    diag = spmat.diagonal()

    def precond(dx, e, _):
        x = diag - e
        x[np.abs(x) < default_kwargs["tol"]] = default_kwargs["tol"]
        return dx / x

    converged, e, _ = pyscf.lib.davidson1(
        lambda vecs: [spmat.dot(v) for v in vecs],
        [_random_initial_guess(spmat.shape)],
        precond,
        **default_kwargs,
    )
    return bool(np.atleast_1d(converged)[0]), float(e[0])


def _random_initial_guess(shape: tuple[int, ...]) -> np.ndarray:
    """A random unit-norm complex vector of length ``shape[0]``."""
    norm = 0.0
    while norm == 0:
        x = np.random.rand(shape[0]) + 1.0j * np.random.rand(shape[0])
        norm = cast(float, np.linalg.norm(x))
    return x / norm
