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

    # Gram-Schmidt on a basis of the term span -> p anticommuting pairs + c commuting centrals.
    _, span_basis = _xor_row_reduce(vectors)
    span = PauliList.from_symplectic(span_basis[:, :n], span_basis[:, n:])
    pair_gens, center = _symplectic_gram_schmidt(span)
    p = len(pair_gens) // 2

    # Row-reduce the centrals so a term's coefficient on central generator j is just its bit in that
    # generator's pivot column (used below).
    center_pivots, center_rref = _xor_row_reduce(np.hstack([center.z, center.x]))

    # Generators A_0,B_0,...,A_{p-1},B_{p-1}, then centrals, as Hermitian Paulis (Y, not iXZ). On the
    # p logical qubits A_i->X_i, B_i->Z_i; centrals act only through their per-sector sign.
    pair_vecs = np.hstack([pair_gens.z, pair_gens.x])
    gen_vecs = np.vstack([pair_vecs, center_rref])
    gen_phys = PauliList.from_symplectic(gen_vecs[:, :n], gen_vecs[:, n:])
    gen_log = _logical_generators(p, len(gen_phys))

    # Decompose each term over the generators: P_k = scalar * product of the chosen generators.
    # Pair coefficients follow from commutation -- A_i is present iff P_k anticommutes with B_i, and
    # B_i iff P_k anticommutes with A_i. Central coefficients (commutation is blind to them) are the
    # residual's bits in the central pivot columns, after removing the pair part.
    coords = np.zeros((len(coeffs), len(gen_phys)), dtype=bool)
    for i in range(p):
        coords[:, 2 * i] = paulis.anticommutes(gen_phys[2 * i + 1])
        coords[:, 2 * i + 1] = paulis.anticommutes(gen_phys[2 * i])
    residual = vectors ^ ((coords[:, : 2 * p].astype(int) @ pair_vecs.astype(int)) & 1).astype(bool)
    coords[:, 2 * p :] = residual[:, center_pivots]

    # Rebuild each term as the ordered product of its generators, physical and logical in lockstep;
    # the phase relating P_k to that product is read from Qiskit (P_k . prod^dagger = omega * I).
    prods = _identities(len(coeffs), n)
    logicals = _identities(len(coeffs), p)
    for i in range(len(gen_phys)):
        mask = coords[:, i]
        if mask.any():
            prods[mask] = prods[mask].compose(gen_phys[i])
            logicals[mask] = logicals[mask].compose(gen_log[i])

    omega_phase = paulis.compose(prods.adjoint()).phase
    amps = coeffs * (-1j) ** omega_phase
    exps = coords[:, 2 * p :].astype(int)  # (K, c) central-generator exponents
    return logicals, amps, exps


def _symplectic_gram_schmidt(paulis: PauliList) -> tuple[PauliList, PauliList, PauliList]:
    """Split Paulis into anticommuting generator pairs plus a mutually commuting center.

    This is the symplectic Gram-Schmidt procedure of M. M. Wilde, "Logical operators of quantum
    codes", Phys. Rev. A 79, 062322 (2009), arXiv:0903.5256. Returns ``(a_paulis, b_paulis, center)``:
    the two halves of the ``p`` anticommuting pairs (``a_paulis[i]`` anticommutes with ``b_paulis[i]``)
    and the commuting center.
    """
    work = paulis
    a_paulis, b_paulis, center = (
        paulis[:0],
        paulis[:0],
        paulis[:0],
    )  # empty PauliLists that keep ``num_qubits``
    while len(work):
        v, rest = work[0], work[1:]
        anti = rest.anticommutes(v)
        if not anti.any():
            center = center.insert(len(center), v)
            work = rest
            continue
        j = int(np.argmax(anti))
        w = rest[j]
        a_paulis += v
        b_paulis += w
        # Make the rest commute with both v and w: multiply each by v where it anticommutes with w,
        # and by w where it anticommutes with v (flags read before either multiply).
        work = rest[np.arange(len(rest)) != j]
        add_v, add_w = work.anticommutes(w), work.anticommutes(v)
        z, x = work.z.copy(), work.x.copy()
        z[add_v] ^= v.z
        x[add_v] ^= v.x
        z[add_w] ^= w.z
        x[add_w] ^= w.x
        work = PauliList.from_symplectic(z, x)
    # Deferred commuting terms may still pair among themselves; resolve recursively.
    if len(center) and any(center.anticommutes(g).any() for g in center):
        a_more, b_more, center = _symplectic_gram_schmidt(center)
        if len(a_more):
            a_paulis += a_more
            b_paulis += b_more
    return a_paulis, b_paulis, center


def _get_basis(paulis: PauliList) -> tuple[list[int], PauliList]:
    """Reduce a ``PauliList`` to an independent generating subset, by row reduction mod 2 (XOR).

    Returns ``(pivot_cols, basis)``: ``basis`` is a ``PauliList`` whose symplectic vectors are the
    reduced rows (an independent basis of the span); row ``i`` has its leading 1 at ``pivot_cols[i]``.
    """
    work = np.hstack([paulis.z, paulis.x])
    pivot_cols: list[int] = []
    row = 0
    for col in range(work.shape[1]):
        below = np.nonzero(work[row:, col])[0]
        if below.size == 0:
            continue
        pivot = row + below[0]
        if pivot != row:
            work[[row, pivot]] = work[[pivot, row]]
        others = work[:, col].copy()
        others[row] = False  # eliminate this column from every other row
        work[others] ^= work[row]
        pivot_cols.append(col)
        row += 1
        if row == len(work):
            break
    zx = work[:row]
    n = paulis.num_qubits
    return pivot_cols, PauliList.from_symplectic(zx[:, :n], zx[:, n:])


def _identities(count: int, num_qubits: int) -> PauliList:
    shape = (count, num_qubits)
    return PauliList.from_symplectic(np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool))


def _logical_generators(p: int, num_gens: int) -> PauliList:
    z = np.zeros((num_gens, p), dtype=bool)
    x = np.zeros((num_gens, p), dtype=bool)
    j = np.arange(p)
    x[2 * j, j] = True
    z[2 * j + 1, j] = True
    return PauliList.from_symplectic(z, x)
def _commutation_matrix(pl1: PauliList, pl2: PauliList, negate=False):
    a_dot_b = (np.asarray(pl1._x, dtype=np.uint8) @ np.asarray(pl2._z.T, dtype=np.uint8)) & 1
    b_dot_a = (np.asarray(pl1._z, dtype=np.uint8) @ np.asarray(pl2._x.T, dtype=np.uint8)) & 1
    if negate:
        return a_dot_b != b_dot_a
    else:
        return a_dot_b == b_dot_a
