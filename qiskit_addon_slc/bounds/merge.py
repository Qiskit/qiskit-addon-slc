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

"""A function for merging the forward and backward bounds taking into account the learned rates."""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import PauliLindbladMap, QubitSparsePauliList

from ..utils import map_modifier_ref_to_ref
from .commutator_bounds import Bounds

NoiseRates = dict[str, PauliLindbladMap | None]

LOGGER = logging.getLogger(__name__)


def merge_bounds(
    circuit: QuantumCircuit,
    forward_bounds: Bounds | None,
    backward_bounds: Bounds | None,
    /,
    noise_rates: NoiseRates | None = None,
    *,
    is_clifford_circuit: bool = False,
) -> Bounds | None:
    """Merge forward and backward bounds.

    The layer at which the switch from using backward bounds to using forward bounds takes place
    will be the same for all qubits. It is determined by taking into account the provided learned
    ``noise_rates``. If these are not provided, uniform noise rates are assumed. While this is an
    unrealistic assumption, previewing the merged bounds may still be useful.

    Args:
        circuit: the target circuit.
        forward_bounds: the forward bounds (see also :func:`.compute_forward_bounds`).
        backward_bounds: the backward bounds (see also :func:`.compute_backward_bounds`).
        noise_rates: the noise rates learned on the target backend.
        is_clifford_circuit: whether the target circuit is fully Clifford.

    Returns:
        The merged bounds.

    Raises:
        ValueError: when both provided bounds are ``None``.
        KeyError: when the ``bounds`` contain an :attr:`.InjectNoise.modifier_ref` key which does
            not occur in the target ``circuit`` or whose :attr:`.InjectNoise.ref` is not found.
        ValueError: if the noise model Pauli terms whose bounds are computed for a given
            :attr:`.InjectNoise.modifier_ref` do not match between the ``forward_bounds`` and
            ``backward_bounds``.
        NotImplementedError: when ``is_clifford_circuit`` is ``True``.
    """
    if forward_bounds is None and backward_bounds is None:
        raise ValueError("No bounds to merge.")

    if forward_bounds is None:
        return deepcopy(backward_bounds)

    if backward_bounds is None:
        return deepcopy(forward_bounds)

    if is_clifford_circuit:
        raise NotImplementedError("Update the code to support Bounds containing PauliLindbladMap")
        merged_bounds = {}
        for box_id in forward_bounds:
            merged_bounds[box_id] = forward_bounds[box_id] * backward_bounds[box_id] / 2.0
        return merged_bounds

    id_map = map_modifier_ref_to_ref(circuit)

    missing_noise_rates = False
    to_partition_fwd = {}
    to_partition_bwd = {}
    if noise_rates is None:
        missing_noise_rates = True
    else:
        for box_id in forward_bounds:
            noise_id = id_map.get(box_id, None)
            if noise_id is None:
                raise KeyError(f"The {box_id = } could not be found in the target circuit!")
            if (learned_plm := noise_rates[noise_id]) is None:
                missing_noise_rates = True
                break

            fwd_plm = forward_bounds[box_id]
            fwd_paulis = fwd_plm.get_qubit_sparse_pauli_list_copy()
            fwd_bounds = fwd_plm.rates.copy()
            bwd_plm = backward_bounds[box_id]
            bwd_paulis = bwd_plm.get_qubit_sparse_pauli_list_copy()
            bwd_bounds = bwd_plm.rates.copy()

            if fwd_paulis != bwd_paulis:
                raise ValueError(
                    "The Pauli error terms of the forward and backward bounds do not match for "
                    f"{box_id = }"
                )

            # find order to match learned noise rates with computed bounds
            learned_paulis = learned_plm.get_qubit_sparse_pauli_list_copy()
            sparse_list = learned_paulis.to_sparse_list()
            order = [sparse_list.index(p) for p in fwd_paulis.to_sparse_list()]
            reordered = QubitSparsePauliList.from_sparse_list(
                [sparse_list[i] for i in order], learned_paulis.num_qubits
            )
            assert reordered == fwd_paulis

            # convert learned noise rates to error probabilities
            # FIXME: can we use learned_plm.probabilities for this?
            err_probs = (1 - np.exp(-2 * learned_plm.rates)) / 2
            err_probs = err_probs[order]

            to_partition_fwd[box_id] = err_probs * fwd_bounds
            to_partition_bwd[box_id] = err_probs * bwd_bounds

    if missing_noise_rates:
        LOGGER.warning(
            "Missing noise rates. Partitioning backward/forward commutator bounds by assuming "
            "uniform error rates."
        )
        to_partition_fwd = {box_id: bound.rates.copy() for box_id, bound in forward_bounds.items()}
        to_partition_bwd = {box_id: bound.rates.copy() for box_id, bound in backward_bounds.items()}

    LOGGER.warning(
        "Optimal spacetime partitioning not implemented!Just partitioning list of noisy boxes."
    )
    total_bias_vs_partition = [sum(bias_fwd.sum() for bias_fwd in to_partition_fwd.values())]
    # NOTE: we exploit the fact that `id_map` will contain the `InjectNoise.modifier_ref` as keys in
    # their order of occurrence in the circuit
    for box_id in id_map:
        bias_fwd = to_partition_fwd[box_id]
        bias_bwd = to_partition_bwd[box_id]
        total_bias_vs_partition.append(
            total_bias_vs_partition[-1] + (bias_bwd.sum() - bias_fwd.sum())
        )

    partition_box_idx = np.argmin(total_bias_vs_partition)
    LOGGER.info(f"Determined Box idx for partitioning to be {partition_box_idx}.")

    merged_bounds = {
        box_id: (backward_bounds[box_id] if idx < partition_box_idx else forward_bounds[box_id])
        for idx, box_id in enumerate(id_map.keys())
    }

    return merged_bounds
