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

"""Reduce a Hermitian Pauli sum to its logical form via symplectic reduction.

The terms are split into independent anticommuting pairs plus a commuting center by a symplectic
Gram-Schmidt procedure over GF(2); see M. M. Wilde, "Logical operators of quantum codes",
Phys. Rev. A 79, 062322 (2009), arXiv:0903.5256.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import PauliList, SparsePauliOp


def _reduce_operator(spo: SparsePauliOp) -> tuple[PauliList, np.ndarray, np.ndarray]:
    """Reduce a Hermitian Pauli sum to its logical form on ``p + c`` generators.

    Returns ``(logicals, amps, exps)``: for each of the ``K`` input terms, its logical Pauli on the
    ``p`` anticommuting-pair qubits, its complex amplitude, and its ``c`` central-generator exponents
    (a ``(K, c)`` matrix). The original operator is unitarily equivalent to the direct sum, over the
    ``2^c`` central sign-sectors, of ``sum_k amps[k] * (+-1) * logicals[k]``.
    """
    paulis = spo.paulis
    coeffs = spo.coeffs
    n = paulis.num_qubits

    vectors = np.hstack([paulis.z, paulis.x])

    # Gram-Schmidt -> ``p`` anticommuting pairs + ``c`` centrals
    _, span_basis, _ = _xor_row_reduce(vectors)
    span = PauliList.from_symplectic(span_basis[:, :n], span_basis[:, n:])
    pair_gens, center = _symplectic_gram_schmidt(span)
    p = len(pair_gens) // 2

    # Ordered generators A_0,B_0,...,A_{p-1},B_{p-1}, then centrals. On the ``p`` logical qubits
    # A_i->X_i, B_i->Z_i (centrals act only through their per-sector sign).
    gen_phys = PauliList.from_symplectic(
        np.vstack([pair_gens.z, center.z]), np.vstack([pair_gens.x, center.x])
    )
    gen_vecs = np.hstack([gen_phys.z, gen_phys.x])
    gen_log = _logical_generators(p, len(gen_phys))

    # Express every term over the generators at once: ``coords[k]`` is the coordinate vector of
    # ``P_k``, so ``P_k`` equals the ordered product of the chosen generators up to a scalar.
    pivot_cols, basis, provenance = _xor_row_reduce(gen_vecs, track_provenance=True)
    coords = _xor_coordinates(vectors, pivot_cols, basis, provenance)  # (K, G)

    # Build each term's physical product ``prod`` and matching logical Pauli by composing the
    # generators into the terms that use them.
    prods = _identities(len(coeffs), n)
    logicals = _identities(len(coeffs), p)
    for i in range(len(gen_vecs)):
        mask = coords[:, i]
        if mask.any():
            prods[mask] = prods[mask].compose(gen_phys[i])
            logicals[mask] = logicals[mask].compose(gen_log[i])

    # Extract every ``omega`` at once: ``P_k = omega * prod``
    # so ``P_k . prod^dagger = omega * I``.
    omega_phase = paulis.compose(prods.adjoint()).phase
    amps = coeffs * (-1j) ** omega_phase
    exps = coords[:, 2 * p :].astype(int)  # (K, c) central-generator exponents
    return logicals, amps, exps


def _symplectic_gram_schmidt(paulis: PauliList) -> tuple[PauliList, PauliList]:
    """Split a set of Paulis into ordered anticommuting pair generators A_0,B_0,A_1,B_1,... plus a
    mutually commuting center.

    This is the symplectic Gram-Schmidt procedure of M. M. Wilde, "Logical operators of quantum
    codes", Phys. Rev. A 79, 062322 (2009), arXiv:0903.5256. Uses ``PauliList.anticommutes`` for the
    commutation tests and combines terms by XOR-ing their ``z``/``x`` arrays (rebuilding via
    ``from_symplectic``), so no Pauli phases are ever computed -- only the symplectic vectors matter.
    """
    work = paulis
    pair_gens, center = paulis[:0], paulis[:0]  # empty PauliLists that keep ``num_qubits``
    while len(work):
        v, rest = work[0], work[1:]
        anti = rest.anticommutes(v)
        if not anti.any():
            center = center.insert(len(center), v)
            work = rest
            continue
        j = int(np.argmax(anti))
        w = rest[j]
        pair_gens = pair_gens.insert(len(pair_gens), v)
        pair_gens = pair_gens.insert(len(pair_gens), w)
        # Project the rest to commute with both ``v`` and ``w`` (both flags read from the original
        # terms): u -> u . w^<u,w> . v^<u,v>, where <.,.> is 1 iff the pair anticommutes.
        work = rest[np.arange(len(rest)) != j]
        add_v, add_w = work.anticommutes(w), work.anticommutes(v)
        z, x = work.z.copy(), work.x.copy()
        z[add_v] ^= v.z
        x[add_v] ^= v.x
        z[add_w] ^= w.z
        x[add_w] ^= w.x
        work = PauliList.from_symplectic(z, x)
    # Deferred centrals may still pair among themselves; resolve recursively.
    if len(center) and any(center.anticommutes(g).any() for g in center):
        more, center = _symplectic_gram_schmidt(center)
        if len(more):
            pair_gens = pair_gens.insert(len(pair_gens), more)
    return pair_gens, center


def _xor_row_reduce(
    mat: np.ndarray, track_provenance: bool = False
) -> tuple[list[int], np.ndarray, np.ndarray | None]:
    """XOR row reduction of a boolean matrix.

    Returns ``(pivot_cols, basis, provenance)``. ``basis`` holds the nonzero reduced rows -- a basis
    of the row space -- with ``basis[i]`` having its leading 1 in column ``pivot_cols[i]``; every
    pivot column is cleared from all other rows, so ``basis`` is fully reduced. When
    ``track_provenance`` is set, ``provenance[i]`` is the boolean mask over the original rows whose
    XOR yields ``basis[i]`` (otherwise ``None``).
    """
    work = mat.copy()
    provenance = np.eye(len(work), dtype=bool) if track_provenance else None
    pivot_cols: list[int] = []
    row = 0
    for col in range(work.shape[1]):
        below = np.nonzero(work[row:, col])[0]
        if below.size == 0:
            continue
        pivot = row + below[0]
        if pivot != row:
            work[[row, pivot]] = work[[pivot, row]]
            if provenance is not None:
                provenance[[row, pivot]] = provenance[[pivot, row]]
        others = work[:, col].copy()
        others[row] = False  # eliminate this column from every other row
        work[others] ^= work[row]
        if provenance is not None:
            provenance[others] ^= provenance[row]
        pivot_cols.append(col)
        row += 1
        if row == len(work):
            break
    return pivot_cols, work[:row], (provenance[:row] if provenance is not None else None)


def _xor_coordinates(
    targets: np.ndarray, pivot_cols: list[int], basis: np.ndarray, provenance: np.ndarray
) -> np.ndarray:
    """Express each row of ``targets`` in the reduced basis, vectorized over all targets.

    ``pivot_cols``, ``basis``, ``provenance`` come from :func:`_xor_row_reduce` (with provenance) of
    the generator vectors. Returns a ``(len(targets), n_generators)`` boolean matrix whose row ``k``
    selects the generators whose XOR equals ``targets[k]``. Every target must lie in the row space.
    """
    work = targets.copy()
    coords = np.zeros((len(targets), provenance.shape[1]), dtype=bool)
    for col, basis_row, prov_row in zip(pivot_cols, basis, provenance):
        hit = work[:, col]
        coords[hit] ^= prov_row
        work[hit] ^= basis_row
    return coords


def _identities(count: int, num_qubits: int) -> PauliList:
    shape = (count, num_qubits)
    return PauliList.from_symplectic(np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool))


def _logical_generators(p: int, num_gens: int) -> PauliList:
    """Logical generators on ``p`` qubits, in ``gen_vecs`` order: pair generator ``2j`` -> X on qubit
    ``j``, ``2j+1`` -> Z on qubit ``j``; any trailing central generators map to identity."""
    z = np.zeros((num_gens, p), dtype=bool)
    x = np.zeros((num_gens, p), dtype=bool)
    j = np.arange(p)
    x[2 * j, j] = True  # A_j -> X_j
    z[2 * j + 1, j] = True  # B_j -> Z_j
    return PauliList.from_symplectic(z, x)
