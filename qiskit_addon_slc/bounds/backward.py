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

"""Backward evolved unequal time commutator bounds."""

from __future__ import annotations

import logging
from functools import partial

import numpy as np
from pauli_prop.propagation import (
    RotationGates,
    propagate_through_rotation_gates,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Pauli,
    QubitSparsePauliList,
    SparsePauliOp,
)

from ..utils import remove_measure
from .commutator_bounds import Bounds, CommutatorBounds, compute_bounds
from .light_cone import LightCone

LOGGER = logging.getLogger(__name__)


def _time_evolved_norm_backward(
    pauli: Pauli,
    gates: RotationGates,
    *,
    evolution_max_terms: int = np.iinfo(np.uint).max,
) -> CommutatorBounds:
    r"""Bound the effect of an error Pauli term on the quantum state by evolving the error backward.

    Evolving backwards means moving the error through the circuit to its start.

    Given an error Pauli term (``pauli``) and the non-Clifford component of a circuit (``gates``),
    this function computes the unequal-time commutator between the Pauli error and the quantum state.
    The error Pauli term (which may reside anywhere in the middle of the
    target circuit) is evolved backwards to the *start* of the circuit. In doing so, it must
    be evolved through the non-Clifford component of the circuit within the light-cone (here, given
    by ``gates``).

    Note, that this function is designed for the context of :func:`.compute_backward_bounds` which
    inverts the target circuit first. Hence, a backward evolution actually amounts to the
    Schrödinger frame evolution of ``pauli`` through ``gates``.

    Args:
        pauli: the error Pauli term to be evolved.
        gates: the non-Clifford circuit component encoded as rotation gates (as produced by
            :func:`~pauli_prop.propagation.propagate_through_rotation_gates`).
        evolution_max_terms: the maximum number of operator terms to keep track of during the
            evolution.

    Returns:
        The unequal-time commutator bound :math:`\| \left[E, \rho\right] \|_1` for Pauli error
        :math:`E` and state :math:`\rho`, where the norm is the Schatten 1 norm (nuclear norm).
    """
    # Convert the single Pauli to a SparsePauliOp which we can then evolve
    pauli = SparsePauliOp(pauli)
    pauli, trunc_onenorm = propagate_through_rotation_gates(
        operator=pauli,
        rot_gates=gates,
        max_terms=evolution_max_terms,
        atol=0,
        frame="s",
    )
    trunc_bias = 2 * trunc_onenorm

    acts_on_zero = np.any(pauli.paulis.x, axis=1)
    x = pauli.paulis.x[acts_on_zero]
    z = pauli.paulis.z[acts_on_zero]
    c = pauli.coeffs[acts_on_zero]

    uniques, which_unique_x = np.unique(x, return_inverse=True, axis=0)
    s = np.zeros(len(uniques), dtype=complex)
    np.add.at(s, which_unique_x, c * ((1j) ** np.sum(z * x, axis=1)))
    sqrt_s = np.linalg.norm(s)
    comm_norm = 2 * sqrt_s

    return CommutatorBounds(float(comm_norm), trunc_bias, False)


def compute_backward_bounds(
    circuit: QuantumCircuit,
    noise_model_paulis: dict[str, QubitSparsePauliList],
    /,
    *,
    evolution_max_terms: int = 1_000_000,
    num_processes: int = 1,
    timeout: float | None = None,
) -> Bounds:
    r"""Compute the backward-evolved unequal-time commutator bounds.

    Starting at the beginning of the circuit, compute the backward-evolved unequal-time commutator
    bounds for all Pauli error terms of each noisy layer in the target circuit.

    That is, compute :math:`\| \left[ E_I, \rho_I \right] \|_1` (using the Schatten 1-norm aka
    nuclear norm) for all error terms, :math:`E_I`, where :math:`\rho_I` is assumed to be the
    all-zero state, :math:`\ket{0 \ldots 0}`, on all active qubits in ``circuit``.

    The error terms, :math:`E_I`, are dictated by ``noise_model_paulis``. This dictionary maps noise
    model identifiers (:attr:`samplomatic.InjectNoise.ref`) to a list of Pauli error terms. The
    corresponding terms will be used whenever a :class:`~qiskit.circuit.BoxOp` with a matching
    :class:`~samplomatic.InjectNoise` annotation is encountered during the iteration over
    ``circuit``.

    .. caution::
      Before computing the bounds, this function removes **all** :class:`.Measure` operations from
      ``circuit``. This is required because the circuit is being inverted before being
      processed in reverse order, which allows the backward evolution to be treated like a forward
      evolution (in the inverted circuit).

    Args:
        circuit: the target circuit.
        noise_model_paulis: the Pauli error terms to consider for each noise model.
        evolution_max_terms: the maximum number of operator terms to keep track of during the
            evolution. (If the operator exceeds this size, the smallest terms are truncated).
        num_processes: the number of parallel processes to use.
        timeout: an optional timeout (in seconds) after which all remaining layers are filled with
            trivial numerical bounds of ``2.0``. Note, that this is not a strict timeout and the
            layer being processed at the time of reaching this timeout will complete normally.

    Returns:
        The backward-evolved unequal-time commutator bounds.
    """
    LOGGER.info("Evolving Pauli error terms backwards through the circuit.")
    LOGGER.info("Modelling errors as though they happen *after* each noise layer.")

    circuit = remove_measure(circuit).inverse()

    norm_fn = partial(
        _time_evolved_norm_backward,
        evolution_max_terms=evolution_max_terms,
    )

    lc = LightCone.initialize_from_measurements(circuit, measure_active=True)

    comm_norms = compute_bounds(
        circuit,
        noise_model_paulis,
        lc,
        norm_fn,
        backwards=True,
        num_processes=num_processes,
        timeout=timeout,
    )

    return comm_norms
