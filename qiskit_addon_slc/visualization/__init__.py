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

from .accumulate_filtered_bounds import accumulate_filtered_bounds
from .animate_shaded_lightcone import animate_shaded_lightcone
from .draw_shaded_lightcone import draw_shaded_lightcone
from .overlay_bounds import overlay_bounds_onto_circuit
from .render_bounds import render_bounds

__all__ = [
    "accumulate_filtered_bounds",
    "animate_shaded_lightcone",
    "draw_shaded_lightcone",
    "overlay_bounds_onto_circuit",
    "render_bounds",
]
