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

"""Prioritization scheme for computing the ``local_scales`` inputs to a :class:`~samplomatic.samplex.Samplex`."""

from __future__ import annotations

import logging

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import PauliLindbladMap, QubitSparsePauliList

from ..utils import map_modifier_ref_to_ref
from .commutator_bounds import Bounds

NoiseRates = dict[str, PauliLindbladMap | None]

LOGGER = logging.getLogger(__name__)


def compute_local_scales(
    circuit: QuantumCircuit,
    bounds: Bounds,
    /,
    noise_rates: NoiseRates,
    *,
    sampling_cost_budget: float = np.inf,
    bias_tolerance: float = 0.0,
) -> tuple[Bounds, float, float]:
    r"""Computes the ``local_scales`` argument to a :class:`~samplomatic.samplex.Samplex`.

    This ``local_scales`` argument is used to specify which individual error terms to mitigate.

    Either the ``sampling_cost_budget`` or ``bias_tolerance`` must be specified. The former puts an
    upper bound on the sampling cost while the latter puts an upper bound on the remaining bias to
    tolerate.

    .. note::
      If the order of Pauli terms in ``bounds`` and ``noise_rates`` do not match, the output of this
      function will assume the order set forth by ``noise_rates`` in order to ensure that the scales
      are compatible with the rates that will also be provided to the :class:`.QuantumProgram`.

    Args:
        circuit: the target circuit.
        bounds: the shaded lightcone bounds.
        noise_rates: the learned noise model rates.
        sampling_cost_budget: the maximum sampling cost to allow.
        bias_tolerance: the maximum bias to tolerate.

    Returns:
        A tuple of length 3, the items of which are:
        - the ``local_scales`` dictionary to be provided as the direct input to the
          :meth:`samplomatic.samplex.Samplex.inputs`.
        - the sampling cost overhead (:math:`\gamma^2`) required to perform the sampling of
          ``local_scales``.
        - the remaining bias on expectation values computed with these bounds.

    Raises:
        ValueError: if non-default values are provided for both, the ``sampling_cost_budget`` and
            ``bias_tolerance``.
        KeyError: if ``noise_rates`` is missing an entry for any noise model identifier
            (:attr:`.InjectNoise.ref`) encountered in ``circuit``.
    """
    ## Validation:
    if (not np.isposinf(sampling_cost_budget)) and (bias_tolerance != 0):
        raise ValueError(
            "Only one of either sampling_cost_budget or bias_tolerance may be specified."
        )

    id_map = map_modifier_ref_to_ref(circuit)

    # Collect the bounds and noise model rates
    num_terms_each_box = {}
    comm_bounds_flat: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
    rates_flat: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
    for box_id, bound in bounds.items():
        num_terms_each_box[box_id] = len(bound)
        learned_plm = noise_rates[id_map[box_id]]
        if learned_plm is None:
            raise KeyError(f"Missing noise rate for {box_id = }")

        computed_paulis = bound.get_qubit_sparse_pauli_list_copy()

        # find order to match learned noise rates with computed bounds
        learned_paulis = learned_plm.get_qubit_sparse_pauli_list_copy()
        sparse_list = learned_paulis.to_sparse_list()
        order = [sparse_list.index(p) for p in computed_paulis.to_sparse_list()]
        reordered = QubitSparsePauliList.from_sparse_list(
            [sparse_list[i] for i in order], learned_paulis.num_qubits
        )
        assert reordered == computed_paulis

        comm_bounds_flat.append(np.asarray(bound.rates[np.argsort(order)], dtype=float))
        rates_flat.append(learned_plm.rates)

    # Flatten them into numpy arrays
    comm_bounds_flat_np = np.concatenate(comm_bounds_flat)
    rates_flat_np = np.concatenate(rates_flat)

    # Compute the priority
    exp_rates_flat = np.exp(-2 * rates_flat_np)
    bias_bounds_flat = comm_bounds_flat_np * (1 - exp_rates_flat) / 2
    priority_flat = comm_bounds_flat_np * exp_rates_flat

    # Find sorting according to priority (in decreasing order, hence [::-1])
    by_decr_priority = np.argsort(priority_flat)[::-1]
    undo_sort_decr = np.argsort(by_decr_priority)

    # Compute cumulative noise rates sorted by decreasing priority
    rates_cumulative = np.cumsum(rates_flat_np[by_decr_priority])

    # Compute accumulated sampling cost
    samp_cost_accum = np.exp(4 * rates_cumulative)

    # Compute cumulatively decreasing bias
    cum_decr_bias = np.cumsum(bias_bounds_flat[by_decr_priority])
    bias_total = cum_decr_bias[-1]

    # Compute remaining bias before mitigation is applied
    bias_before_mitigating = np.concatenate(([bias_total], bias_total - cum_decr_bias[:-1]))

    # Mitigate term if:
    mask = (
        (samp_cost_accum <= sampling_cost_budget)  # can afford to mitigate it
        & (bias_before_mitigating > bias_tolerance)  # not yet in accuracy tol
        & (comm_bounds_flat_np[by_decr_priority] > 0)  # bias bound is nonzero
    )

    # The remaining bias is 0 if all terms can be mitigated, otherwise it is found at the
    # respective index of `bias_before_mitigating`
    bias_remaining = 0 if np.all(mask) else bias_before_mitigating[np.sum(mask)]

    # The sampling cost is found at the index of last term being masked
    sampling_cost = samp_cost_accum[mask][-1]

    # Undo the by-priority-sorting of mask
    mask = mask[undo_sort_decr]

    # Build mask in the original dictionary form (with box_id as keys).
    # NOTE: this mask_ will become `local_scales` for samplex
    mask_ = {}
    # At the same time, build the truncated noise models.

    for box_id, num_terms in num_terms_each_box.items():
        # split off the correct num_terms into the dict and keep the rest in the array
        this_mask, mask = np.split(mask, [num_terms])
        mask_[box_id] = np.asarray(this_mask, dtype=float)

    return mask_, sampling_cost, bias_remaining
