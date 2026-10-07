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

"""The bounds accumulation and filtering method."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, QubitSparsePauliList

from ..bounds.commutator_bounds import Bounds
from ..utils import map_modifier_ref_to_ref


def accumulate_filtered_bounds(
    circuit: QuantumCircuit,
    bounds: Bounds,
    noise_model_paulis: dict[str, QubitSparsePauliList],
    pauli_filter: Pauli | str | int | None = None,
) -> dict[str, dict[tuple[int, ...], float]]:
    """Accumulates the bound values filtered by a specified Pauli type.

    Bound values are summed per non-identity support, so without a filter the bounds of *all* noise
    terms sharing a support are accumulated into a single value.

    A ``str`` or :class:`~qiskit.quantum_info.Pauli` filter is matched against each noise term
    **reduced to its own non-identity support**, not against its full-width label. Thus ``"X"``
    selects every single-qubit :math:`X` term regardless of which qubit it acts on, ``"XX"`` selects
    the weight-2 :math:`XX` terms, and a full-width label such as ``"IXI"`` matches nothing.

    .. seealso::
       :func:`.draw_shaded_lightcone` for a runnable example of the full pipeline, as well as
       :func:`.overlay_bounds_onto_circuit` and :func:`.render_bounds`.

    Args:
        circuit: the circuit whose bounds to accumulate.
        bounds: the bounds whose values to accumulate.
        noise_model_paulis: the Pauli noise terms for each noise model.
        pauli_filter: the optional Pauli type to filter by. It behaves as follows:
            - ``None``: accumulates all noise term bounds of equal support.
            - ``int``: only accumulates noise term bounds of the specified Pauli weight.
            - ``str``: selects noise terms whose non-identity support equals this Pauli.
            - ``Pauli``: selects noise terms whose non-identity support equals this Pauli.

    Returns:
        A nested dictionary. The outer most key is the :attr:`.InjectNoise.modifier_ref` (same as
        the original ``bounds``). The next key is the support of the accumulated and filtered bound
        value (i.e. a tuple of integers). This maps to the actual bound value as a float.
    """
    if isinstance(pauli_filter, str):
        pauli_filter = Pauli(pauli_filter)

    id_map = map_modifier_ref_to_ref(circuit)

    # First, we preprocess the noise model paulis by:
    #  - reducing each noise term to its non-identity support
    #  - checking if it matches the `pauli_filter` criterion
    #  - tracking the non-identity support for later use
    noise_model_supports: dict[str, list[tuple[int, ...] | None]] = {}
    for noise_id, noise_terms in noise_model_paulis.items():
        noise_supports: list[tuple[int, ...] | None] = []

        for noise_term in noise_terms:
            pauli = noise_term.to_pauli()
            support = np.where(pauli.x | pauli.z)[0].tolist()
            local_noise_term = pauli[support]

            if (
                (pauli_filter is None)
                or len(support) == pauli_filter
                or local_noise_term == pauli_filter
            ):
                noise_supports.append(tuple(support))
            else:
                noise_supports.append(None)

        noise_model_supports[noise_id] = noise_supports

    # Second, we sum the bounds of the filtered noise terms on their non-identity supports
    pauli_bounds: dict[str, dict[tuple[int, ...], float]] = {}
    for box_id, box_bounds in bounds.items():
        pauli_bounds_this_box: dict[tuple[int, ...], float] = defaultdict(float)

        for noise_support, noise_bound in zip(  # type: ignore[call-overload]
            noise_model_supports[id_map[box_id]], box_bounds.rates, strict=True
        ):
            if noise_support is None:
                continue

            pauli_bounds_this_box[noise_support] += noise_bound

        pauli_bounds[box_id] = pauli_bounds_this_box

    return pauli_bounds
