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

"""Bound computation functions.

.. currentmodule:: qiskit_addon_slc.bounds

This module provides various functions for computing the error bounds that make up a shaded
lightcone.

.. autofunction:: compute_forward_bounds

.. autofunction:: tighten_with_speed_limit

.. autofunction:: compute_backward_bounds

.. autofunction:: merge_bounds

.. autofunction:: compute_local_scales
"""

from .backward import compute_backward_bounds
from .forward import compute_forward_bounds
from .local_scales import compute_local_scales
from .merge import merge_bounds
from .speed_limit import tighten_with_speed_limit

__all__ = [
    "compute_backward_bounds",
    "compute_forward_bounds",
    "compute_local_scales",
    "merge_bounds",
    "tighten_with_speed_limit",
]
