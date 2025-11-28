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

"""Visualization methods for shaded lightcones.

.. currentmodule:: qiskit_addon_slc.visualization

This module provides visualization methods for shaded lightcones.

.. autofunction:: animate_shaded_lightcone

.. autofunction:: draw_shaded_lightcone

.. autofunction:: accumulate_filtered_bounds

.. autofunction:: overlay_bounds_onto_circuit

.. autofunction:: render_bounds
"""

from __future__ import annotations

import matplotlib as mpl
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, QubitSparsePauliList

from ..bounds.commutator_bounds import Bounds
from .accumulate_filtered_bounds import accumulate_filtered_bounds
from .animate_shaded_lightcone import animate_shaded_lightcone
from .overlay_bounds import overlay_bounds_onto_circuit
from .render_bounds import render_bounds


def draw_shaded_lightcone(
    circuit: QuantumCircuit,
    bounds: Bounds,
    noise_model_paulis: dict[str, QubitSparsePauliList],
    *,
    pauli_filter: Pauli | str | int | None = None,
    include_empty_boxes: bool = True,
    **rendering_kwargs,
) -> mpl.figure.Figure:
    """Draws a shaded lightcone.

    First, the provided ``bounds`` are accumulated and filtered according to ``pauli_filter``. See
    also :func:`.accumulate_filtered_bounds`.
    Then, the resulting bounds are overlaid onto the provided ``circuit`` (see
    :func:`.overlay_bounds_onto_circuit`) and subsequently rendered (see :func:`.render_bounds`).

    Args:
        circuit: the circuit whose shaded lightcone to draw.
        bounds: the bounds to use for the shaded lightcone.
        noise_model_paulis: the Pauli error terms of the circuit's noise models.
        pauli_filter: the optional Pauli type by which the bounds were filtered.
        include_empty_boxes: whether to include empty boxes or not.
        rendering_kwargs: any additional keyword arguments are forwarded to
            :meth:`.QuantumCircuit.draw`.

    Returns:
        The ``mpl`` figure.
    """
    pauli_bounds = accumulate_filtered_bounds(circuit, bounds, noise_model_paulis, pauli_filter)
    bounds_circuit = overlay_bounds_onto_circuit(
        pauli_bounds, circuit, include_empty_boxes=include_empty_boxes
    )
    return render_bounds(bounds_circuit, pauli_filter=pauli_filter, **rendering_kwargs)


__all__ = [
    "accumulate_filtered_bounds",
    "animate_shaded_lightcone",
    "draw_shaded_lightcone",
    "overlay_bounds_onto_circuit",
    "render_bounds",
]
