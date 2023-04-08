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

"""Bounds tightening leveraging speed limits of information propagation."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import cast

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, Operation
from qiskit.quantum_info import (
    PTM,
    Pauli,
    PauliLindbladMap,
    PauliList,
    QubitSparsePauliList,
    SparseObservable,
    SparsePauliOp,
)

from ..utils import find_indices, iter_circuit, remove_measure
from .commutator_bounds import Bounds

LOGGER = logging.getLogger(__name__)


def _evolve_by_gate(
    speed_limit_bounds: np.ndarray,
    gate: Operation,
    qubits: list[int],
    *,
    it_lightcone: bool = False,
) -> None:
    # update the speed_limit_bounds based on an operator u supported on 1 or 2 qubits in qubits
    # if it_lightcone = True, we will compute the information theoretic light cone instead
    nqubits = len(qubits)

    ptm = np.abs(PTM(gate)).reshape((4,) * 2 * nqubits)
    # reshape docs: last index changes fastest (reading + writing).
    # So last index is Q0 input Pauli; first is Q1 output Pauli.
    # Denote by subscripts 'BAba'. (CAPS indicate output pauli).
    if it_lightcone:
        ptm = (ptm > 0) * 1
    # axes: [qb1 paulis out, qb0 paulis out, qb1 paulis in, qb0 paulis in]

    # get the speed_limit_bounds on the supported qubits:
    w = speed_limit_bounds[qubits]
    # prepend 1 for identity Pauli component:
    w = np.concatenate((np.ones((nqubits, 1)), w), axis=1)
    # axes: [qubits, 1Q-paulis]

    if nqubits == 2:  # if u is a 2-qubit gate
        M = np.minimum.outer(w[0], w[1])
        # axes: [1Q-paulis, 1Q-paulis]

        # compute new speed_limit_bounds
        # lowercase (UPPERCASE) = gate input (OUTPUT)
        w0 = np.einsum("BAba,ab->A", ptm, M)
        w1 = np.einsum("BAba,ab->B", ptm, M)
        new_w = np.array((w0, w1))

    elif nqubits == 1:  # if u is a 1-qubit gate
        w0 = np.einsum("Aa,a->A", ptm, w[0])
        new_w = np.array((w0,))

    else:
        raise ValueError("Gates on more than 2 qubits not supported.")

    new_w = new_w[:, 1:]  # drop dummy identity elements
    new_w[new_w > 1] = 1  # speed_limit_bounds cannot be more than 1
    speed_limit_bounds[qubits] = new_w


def _sync_weight1_speed_limit_bounds(
    speed_limit_bounds: np.ndarray,
    bounds: PauliLindbladMap,
    wt1_mask: np.ndarray,
    wt1_qargs: np.ndarray,
    wt1_xyz_idx: np.ndarray,
) -> PauliLindbladMap:
    # TODO: update docstring
    """Update both self and previously-computed bounds by whichever is smaller.

    Previously-computed forward-commutator bounds on X,Y,Z may let us tighten bounds on observable.
    """
    bounds_ = np.asarray(bounds.rates.copy(), dtype=float)
    # Get available speed_limit_bounds of the backwards-evolved observable from commutator
    # computation, and convert from sparse (lists) to dense (array) format:

    # Initialize dense array at trivial bound of 2:
    # Tighten bounds with whatever sparse info we have from previously-computed bounds:
    bounds_wt1 = bounds_[wt1_mask]

    # Let info flow from comm-bounds to bounds on backpropped observable:
    # (if observable almost commutes with a Z error, then it can't have much X or Y component there)
    for offset in (1, 2):
        xyz = (wt1_xyz_idx + offset) % 3

        speed_limit_bounds[wt1_qargs, xyz] = np.min(
            [
                speed_limit_bounds[wt1_qargs, xyz],
                bounds_wt1 / 2,
            ],
            axis=0,
        )

    # Let info flow from bounds on backpropped observable to comm-bounds:
    # (if observable has low X and Y components on a qubit then it must almost commute with Z there)
    bounds_[wt1_mask] = np.min(
        [
            bounds_wt1,
            2
            * (
                speed_limit_bounds[wt1_qargs, (wt1_xyz_idx + 1) % 3]
                + speed_limit_bounds[wt1_qargs, (wt1_xyz_idx + 2) % 3]
            ),
        ],
        axis=0,
    )
    return PauliLindbladMap.from_components(bounds_, bounds.get_qubit_sparse_pauli_list_copy())


def _update_high_weight_bounds(
    bounds: PauliLindbladMap,
    wt1_mask: np.ndarray,
    wtm_qubit_idx: list[list[int]],
    wtm_xyz_idx: list[list[int]],
    wt1_lookup: dict[tuple[list[int], list[int]], int],
) -> PauliLindbladMap:
    bounds_ = np.asarray(bounds.rates.copy(), dtype=float)
    # Apply LR calculation to higher-weight bounds:
    # (if obs mostly commutes with X on qubit 0 and Y on qubit 1 then mostly commutes with XY on 01)

    bounds_wt1 = bounds_[wt1_mask]
    bounds_wtm = bounds_[~wt1_mask]

    for i, (qb_idx, xyz_idx) in enumerate(
        zip(wtm_qubit_idx, wtm_xyz_idx, strict=False)  # type: ignore[call-overload]
    ):
        bounds_wtm[i] = min(
            bounds_wtm[i],
            sum(
                bounds_wt1[wt1_lookup[qb, xyz]]  # type: ignore[index]
                for qb, xyz in zip(qb_idx, xyz_idx, strict=False)  # type: ignore[call-overload]
            ),
        )

    bounds_[~wt1_mask] = bounds_wtm
    return PauliLindbladMap.from_components(bounds_, bounds.get_qubit_sparse_pauli_list_copy())


def _pre_process_noise_model(
    noise_model_paulis: dict[str, QubitSparsePauliList],
) -> tuple[dict[str, tuple], dict[str, tuple]]:
    # pre-process the noise models into single-qubit and higher weight terms in a sparse format
    noise_model_wt1 = {}
    noise_model_high = {}

    for noise_id, noise_terms in noise_model_paulis.items():
        pauli_list = noise_terms.to_pauli_list()
        x = pauli_list.x
        z = pauli_list.z

        # convert Pauli type to index: X == 0, Y == 1, Z == 2 (I == 0 but will be ignored)
        xyz_idx = 1 * (z & x) + 2 * (z & ~x)

        # find positions of non-identity Paulis
        non_id = z | x

        # masking array to filter weight-1 Pauli strings
        wt1_mask = np.sum(non_id, axis=1) == 1

        # qubit indices of weight-1 Paulis
        wt1_qubit_idx = np.argmax(non_id[wt1_mask], axis=1)
        # Pauli types of weight-1 Paulis
        wt1_xyz_type = np.asarray(
            [xyz_idx[wt1_mask][i, q] for i, q in enumerate(wt1_qubit_idx)], dtype=int
        )

        # for the single-qubit noise model we later need:
        #  - the weight-1 mask
        #  - the single-qubit indices
        #  - the single-qubit Pauli types
        noise_model_wt1[noise_id] = (wt1_mask, wt1_qubit_idx, wt1_xyz_type)

        # qubit-indices of higher weight Pauli strings
        wtm_qubit_idx = [[i for i, q in enumerate(pauli) if q] for pauli in non_id[~wt1_mask]]
        # Pauli types of higher weight Paulis
        wtm_xyz_type = [xyz_idx[~wt1_mask][i, q].tolist() for i, q in enumerate(wtm_qubit_idx)]

        # lookup table to map from sparse single-qubit Pauli (qubit idx, Pauli type) to index along
        # the length of wt1_mask's `True` elements
        wt1_lookup = {
            (qb, xyz): idx
            for idx, (qb, xyz) in enumerate(
                zip(wt1_qubit_idx, wt1_xyz_type, strict=False)  # type: ignore[call-overload]
            )
        }

        # for the higher weight noise model we later need:
        #  - the multi-qubit indices
        #  - the multi-qubit Pauli types
        #  - the lookup table of sparse single-qubit Paulis in the masked positions
        noise_model_high[noise_id] = (wtm_qubit_idx, wtm_xyz_type, wt1_lookup)

    return noise_model_wt1, noise_model_high


def tighten_with_speed_limit(
    bounds: Bounds,
    circuit: QuantumCircuit,
    noise_model_paulis: dict[str, QubitSparsePauliList],
    /,
    observable: Pauli | PauliList | SparseObservable | SparsePauliOp,
) -> Bounds:
    """Tighten the provided bounds using limits on the speed of information propagation.

    Inspired by the ideas behind the Lieb-Robinson bounds, this function leverages limits on the
    speed of information propagation to tighten previously computed forward-evolved unequal-time
    commutator bounds (see also :func:`.compute_forward_bounds`).

    Args:
        bounds: the previously computed forward-evolved unequal-time commutator bounds.
        circuit: the target circuit.
        noise_model_paulis: the Pauli error terms to consider for each noise model.
        observable: the target observable to be measured at the end of the circuit.

    Returns:
        A tightened copy of the unequal-time commutator bounds.

    Raises:
        NotImplementedError: when the ``observable`` contains more than a single Pauli term. If you
            run into this, you will need to call this function for each target Pauli separately.
        ValueError: when encountering a gate that acts on more than 2 qubits.
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

    LOGGER.info("Tighting bounds using information propagation speed limits")
    LOGGER.info("Modelling errors as though they happen *after* each noise layer.")

    # convert pauli into array of speed_limit_bounds
    speed_limit_bounds = np.zeros((len(pauli), 3))
    speed_limit_bounds[:, 0] = pauli.x & ~pauli.z  # X
    speed_limit_bounds[:, 1] = pauli.x & pauli.z  # Y
    speed_limit_bounds[:, 2] = ~pauli.x & pauli.z  # Z

    noise_model_wt1, noise_model_high = _pre_process_noise_model(noise_model_paulis)

    # ensure we do not accidentally overwrite any of the input bounds
    out_bounds = deepcopy(bounds)

    # (aka evolve forward through inverted circuit)
    circuit_dg = remove_measure(circuit).inverse()
    for circ_inst, qargs, box_id, noise_id in iter_circuit(circuit_dg):
        if box_id is None:
            gate_qargs = cast(list[int], find_indices(circuit, circ_inst))
            _evolve_by_gate(speed_limit_bounds, circ_inst.operation, gate_qargs)
            continue

        assert noise_id is not None

        wt1_mask, wt1_qubit_idx, wt1_xyz_type = noise_model_wt1[noise_id]
        wt1_qargs = np.array(qargs, dtype=int)[wt1_qubit_idx]

        out_bounds[box_id] = _sync_weight1_speed_limit_bounds(
            speed_limit_bounds,
            out_bounds[box_id],
            wt1_mask,
            wt1_qargs,
            wt1_xyz_type,
        )

        out_bounds[box_id] = _update_high_weight_bounds(
            out_bounds[box_id],
            wt1_mask,
            *noise_model_high[noise_id],
        )

        # Now evolve bounds through the gates in the box:
        for inst in circ_inst.operation.body:
            if isinstance(inst.operation, Barrier):
                LOGGER.debug(f"Ignoring instruction of type '{type(inst.operation)}'")
                continue

            qargs = cast(list[int], find_indices(circuit_dg, inst))
            _evolve_by_gate(speed_limit_bounds, inst.operation, qargs)

    return out_bounds
