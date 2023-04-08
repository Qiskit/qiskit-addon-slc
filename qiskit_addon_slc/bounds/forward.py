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

"""Forward evolved unequal time commutator bounds."""

from __future__ import annotations

import logging
from functools import partial

import numpy as np
import scipy
from pauli_prop.propagation import (
    RotationGates,
    propagate_through_rotation_gates,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Pauli,
    PauliList,
    QubitSparsePauliList,
    SparseObservable,
    SparsePauliOp,
)

from ..utils import get_extremal_eigenvalue, remove_measure
from .commutator_bounds import Bounds, CommutatorBounds, compute_bounds
from .light_cone import LightCone

LOGGER = logging.getLogger(__name__)
WARNING_TOL: float = 1e-8


def time_evolved_norm_forward(
    pauli: Pauli,
    gates: RotationGates,
    observable: Pauli,
    *,
    evolution_max_terms: int = np.iinfo(np.uint).max,
    eigval_max_qubits: int = np.iinfo(np.uint).max,
    comm_norm_order: int = 2,
    atol: float = 1e-8,
) -> CommutatorBounds:
    """Compute the bound of an error Pauli term evolved forward to the target observable.

    Given an error Pauli term (``pauli``), the non-Clifford component of a circuit (``gates``), and
    a target observable (``observable``) that is to be measured at the end of the circuit, this
    function computes the unequal-time commutator between the noise term and observable. That means,
    the error Pauli term (which may reside anywhere in the middle of the target circuit) is being
    evolved forwards to the *end* of the circuit where the target observable is being measured. In
    doing so, it must be evolved through the non-Clifford component of the circuit within the
    light-cone (here, given by ``gates``).

    Args:
        pauli: the error Pauli term to be evolved.
        gates: the non-Clifford circuit component encoded as rotation gates (as produced by
            :func:`~pauli_prop.propagation.propagate_through_rotation_gates`).
        observable: the target observable to be measured at the end of the circuit.
        evolution_max_terms: the maximum number of operator terms to keep track of during the
            evolution.
        eigval_max_qubits: the maximum number of qubits of a commutator for which the eigenvalue
            computation will still be attempted. When this value is exceeded, the bound is
            approximated via a simpler and more loose triangle inequality.
        comm_norm_order: the order of the commutator norm to compute.
        atol: the absolute tolerance used for trimming terms from the commutator and for detecting
            convergence of the commutator's eigenvalue.

    Returns:
        The unequal-time commutator bound.
    """
    # Convert the single Pauli to a SparsePauliOp which we can then evolve
    orig = pauli
    pauli = SparsePauliOp(pauli)
    pauli, trunc_onenorm = propagate_through_rotation_gates(
        operator=pauli,
        rot_gates=gates,
        max_terms=evolution_max_terms,
        atol=0,
        frame="s",
    )
    trunc_bias = 2 * trunc_onenorm

    # Handle case of a single-Pauli:
    # Ignore limit on num qubits since don't need to go to computational basis.
    # Efficiently handles Clifford case.
    if len(pauli.paulis) == 1:
        comm_norm = 2 * np.abs(pauli.coeffs[0]) * (pauli.paulis[0].anticommutes(observable))
        return CommutatorBounds(comm_norm, trunc_bias, False)

    # NOTE: we must use .dot for the second operation because we need the implementation of
    # `SparsePauliOp.dot(Pauli)` since `Pauli.compose(SparsePauliOp)` is not implemented
    commutator = pauli.compose(observable) - pauli.dot(observable)
    # NOTE: since pauli and observable are both hermitian, we know that their commutator is
    # anti-hermitian. Therefore, multiplying it by 1j below we can make it hermitian again (without
    # changing its norm). This guarantees the imaginary phase of all coefficients to be zero (but
    # asserting this will work only after a call to simplify(atol=0) to de-duplicate terms and
    # ensure coefficients cancel correctly).
    commutator *= -1j
    # NOTE: even though we do not call simplify (yet) we force all imaginary phases to be exactly 0
    # to avoid numerical noise.
    commutator.coeffs.imag = 0

    # compute 1-norm before simplifying to atol
    commutator = commutator.simplify(atol=0)
    one_norm_before = np.linalg.norm(commutator.coeffs, ord=1)
    # compute 1-norm after simplifying to atol
    commutator = commutator.simplify(atol=atol)
    one_norm_after = np.linalg.norm(commutator.coeffs, ord=1)
    # compute loss in 1-norm due to simplifying to atol
    one_norm_loss = one_norm_before - one_norm_after
    one_norm_loss = max(one_norm_loss, np.float64(0.0))
    trunc_bias += one_norm_loss

    # Handle case where commutator is 0:
    if np.logical_not(np.any((commutator.paulis.x, commutator.paulis.z))) and np.isclose(
        np.sum(commutator.coeffs), 0
    ):
        return CommutatorBounds(0.0, trunc_bias, False)

    # If any qubits have only identity Paulis, remove those qubits.
    # Not that important for operator evolution but possibly important for evaluating spectral norm:
    identity_qb_mask = np.logical_not(
        np.any((commutator.paulis.z, commutator.paulis.x), axis=(0, 1))
    )
    if np.any(identity_qb_mask):
        identity_qbs = np.where(identity_qb_mask)[0]
        commutator = SparsePauliOp(
            commutator.paulis.delete(identity_qbs, qubit=True),
            commutator.coeffs.copy(),
            copy=True,
        ).simplify(atol=0)

    # Handle case where a comm_norm_order other than 2 was requested:
    if comm_norm_order != 2:
        comm_norm = np.linalg.norm(commutator, ord=comm_norm_order)
        return CommutatorBounds(float(comm_norm), trunc_bias, False)

    def fallback_to_tri_ineq(coeffs, trunc_bias_) -> CommutatorBounds:
        comm_norm_ = 2 * np.abs(coeffs).sum()
        return CommutatorBounds(float(comm_norm_), trunc_bias_, True)

    # When the number of qubits is too large, fall back
    if commutator.num_qubits > eigval_max_qubits:
        return fallback_to_tri_ineq(commutator.coeffs, trunc_bias)

    # When the number of qubits is sufficiently small, compute the smallest eigenvalue directly
    if commutator.num_qubits <= 4:
        commutator = commutator.to_matrix()
        comm_norm = np.abs(
            scipy.linalg.eigvalsh(
                commutator,
                subset_by_index=(
                    commutator.shape[0] - 1,
                    commutator.shape[0] - 1,
                ),
            )[0]
        )
        success = True

    else:
        success, comm_norm = get_extremal_eigenvalue(commutator, tol=atol)

    if success:
        comm_norm = np.abs(comm_norm)
        if comm_norm - 2.0 > WARNING_TOL:
            LOGGER.info(
                f"Solver found comm norm {comm_norm:.6f} > {2.0 + WARNING_TOL} for Pauli error "
                f"{orig!s}."
            )
    else:
        # If this failure is common, could sort SPO by |coeffs|, break into chunks, and call
        # Davidson on each chunk.
        LOGGER.debug("Eigensolver failed, reverting to triangle inequality...")
        return fallback_to_tri_ineq(commutator.coeffs, trunc_bias)

    return CommutatorBounds(comm_norm, trunc_bias, False)


def compute_forward_bounds(
    circuit: QuantumCircuit,
    noise_model_paulis: dict[str, QubitSparsePauliList],
    /,
    observable: Pauli | PauliList | SparseObservable | SparsePauliOp,
    *,
    evolution_max_terms: int = 1_000_000,
    eigval_max_qubits: int = 14,
    atol: float = 1e-8,
    num_processes: int = 1,
    timeout: float | None = None,
) -> Bounds:
    r"""Compute the forward-evolved unequal-time commutator bounds.

    Starting at the end of the circuit, compute the forward-evolved unequal-time commutator bounds
    for all Pauli error terms of each noisy layer in the target circuit.

    That is, compute :math:`\| \left[ E_F, A_F \right] \|_2` for all error terms, :math:`E_F`, where
    :math:`A_F` is the target ``observable`` to be measured on ``circuit``.

    The error terms, :math:`E_I`, are dictated by ``noise_model_paulis``. This dictionary maps noise
    model identifiers (:attr:`samplomatic.InjectNoise.ref`) to a list of Pauli error terms. The
    corresponding terms will be used whenever a :class:`~qiskit.circuit.BoxOp` with a matching
    :class:`~samplomatic.InjectNoise` annotation is encountered during the iteration over
    ``circuit``.

    Args:
        circuit: the target circuit.
        noise_model_paulis: the Pauli error terms to consider for each noise model.
        observable: the target observable to be measured at the end of the circuit.
        evolution_max_terms: the maximum number of operator terms to keep track of during the
            evolution.
        eigval_max_qubits: the maximum number of qubits of a commutator for which the eigenvalue
            will still be attempted to be computed. When this value is exceeded, the bound is
            approximated via a simpler and more loose triangle inequality.
        atol: the absolute tolerance used for trimming terms from the commutator and for detecting
            convergence of the commutator's eigenvalue.
        num_processes: the number of parallel processes to use.
        timeout: an optional timeout (in seconds) after which all remaining layers are filled with
            trivial numerical bounds of ``2.0``. Note, that this is not a strict timeout and the
            layer being processed at the time of reaching this timeout will complete normally.

    Returns:
        The unequal-time commutator bound.

    Raises:
        NotImplementedError: when the ``observable`` contains more than a single Pauli term. If you
            run into this, you will need to call this function for each target Pauli separately.
    """
    if not isinstance(observable, Pauli):
        if len(observable) != 1:
            raise NotImplementedError(
                "Cannot compute the bounds for an observable with more than 1 Pauli term! "
                "Please iterate over the Paulis one at a time to compute their bounds."
            )
        if isinstance(observable, PauliList):
            pauli = observable[0]
        elif isinstance(observable, SparsePauliOp):
            pauli = observable.paulis[0]
        elif isinstance(observable, SparseObservable):
            pauli = SparsePauliOp.from_sparse_observable(observable).paulis[0]
    else:
        pauli = observable

    # NOTE: Forward evolution means evolve errors to later times, e.g. towards measurements.
    LOGGER.info("Evolving Pauli error terms forwards through the circuit.")
    LOGGER.info("Modelling errors as though they happen *after* each noise layer.")

    circuit = remove_measure(circuit)

    norm_fn = partial(
        time_evolved_norm_forward,
        observable=pauli,
        evolution_max_terms=evolution_max_terms,
        eigval_max_qubits=eigval_max_qubits,
        comm_norm_order=2,
        atol=atol,
    )

    lc = LightCone.initialize_from_pauli(circuit, pauli)

    comm_norms = compute_bounds(
        circuit,
        noise_model_paulis,
        lc,
        norm_fn,
        backwards=False,
        num_processes=num_processes,
        timeout=timeout,
    )

    return comm_norms
