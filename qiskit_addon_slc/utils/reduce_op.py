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

"""Reduce a Hermitian Pauli sum to a smaller operator with the same eigenvalues.

Its terms are split into ``p`` independent anticommuting pairs plus ``c`` mutually commuting terms,
so the sum acts on only ``p + c`` effective qubits instead of ``n``. The split uses the symplectic
Gram-Schmidt of M. M. Wilde, "Logical operators of quantum codes", Phys. Rev. A 79, 062322 (2009),
arXiv:0903.5256.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import PauliList, SparsePauliOp


def _reduce_operator(spo: SparsePauliOp) -> tuple[SparsePauliOp, int]:
    """Reduce ``spo`` to an operator with the same eigenvalues on fewer qubits.

    Symplectic Gram-Schmidt splits the terms into ``p`` anticommuting generator pairs plus ``c``
    mutually commuting generators. Returns ``(reduced_op, num_trailing_Zs)``:

    - ``reduced_op``: a ``SparsePauliOp`` on ``p + c`` qubits with the same distinct eigenvalues as
      ``spo``. Qubits ``0 .. p-1`` carry the pairs; the last ``c`` qubits carry the commuting
      generators, each as a single-qubit ``Z``.
    - ``num_trailing_Zs``: ``c``. Those commuting generators are conserved, so ``reduced_op`` is
      block-diagonal over their ``2^c`` sign sectors.
    """
    paulis = spo.paulis
    coeffs = spo.coeffs
    n = paulis.num_qubits
    zx = np.hstack([paulis.z, paulis.x])

    # Reduce the terms to a basis (products of which give every original term), then split it into
    # p anticommuting pairs + c commuting generators.
    _, span_basis = _get_basis(paulis)
    a_gens, b_gens, center = _symplectic_gram_schmidt(span_basis)
    p = len(a_gens)
    c = len(center)

    # Put the commuting generators in echelon form so a term's coefficient on generator j is its bit
    # in that generator's pivot column (used for the residual below).
    center_pivots, center_gens = _get_basis(center)
    pair_gens = a_gens + b_gens
    generators = pair_gens + center_gens

    # Which generators make up each term? For a pair, A_i (B_i) is present iff the term anticommutes
    # with B_i (A_i). Commuting generators are read from the leftover bits at their pivot columns,
    # after removing the pair part.
    anticommutes = _commutation_matrix(paulis, generators, negate=True)  # (K, len(generators))
    gen_mask = np.zeros_like(anticommutes)
    gen_mask[:, 0:p] = anticommutes[:, p : 2 * p]
    gen_mask[:, p : 2 * p] = anticommutes[:, 0:p]
    pair_zx = np.hstack([pair_gens.z, pair_gens.x])
    residual = zx ^ (np.matmul(gen_mask[:, : 2 * p], pair_zx, dtype=np.uint8) & 1).astype(bool)
    gen_mask[:, 2 * p :] = residual[:, center_pivots]

    # Reduced generators on p + c qubits: pair (A_q, B_q) -> (Z_q, X_q) on qubit q; commuting
    # generator j -> Z on qubit p + j.
    gens_reduced = _identities(len(generators), p + c)
    q = np.arange(p)
    gens_reduced.z[q, q] = True  # A_q -> Z_q
    gens_reduced.x[p + q, q] = True  # B_q -> X_q
    j = np.arange(c)
    gens_reduced.z[2 * p + j, p + j] = True  # commuting generator j -> Z on qubit p + j

    # Product of each term's generators on n qubits (all generators, needed for the phase below) and
    # on the p + c reduced qubits, in lockstep.
    prods = _identities(len(coeffs), n)
    prods_reduced = _identities(len(coeffs), p + c)
    for i in range(len(generators)):
        mask = gen_mask[:, i]
        if mask.any():
            prods[mask] @= generators[i]
            prods_reduced[mask] @= gens_reduced[i]

    # P_k = omega * prod, so P_k . prod^dagger = omega * I gives the phase.
    omega_phase = (paulis @ prods.adjoint()).phase
    amps = coeffs * (-1j) ** omega_phase
    return SparsePauliOp(prods_reduced, amps), c


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
