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

"""Various utilities.

.. currentmodule:: qiskit_addon_slc.utils

This module provides a number of utility functions.
Some of these exist only temporarily to work around open issues of the Qiskit SDK.
When this is the case, they are marked as such and may be removed without deprecation or further
notice.

.. autofunction:: find_indices

.. autofunction:: generate_noise_model_paulis

.. autofunction:: get_extremal_eigenvalue

.. autofunction:: iter_circuit

.. autofunction:: map_modifier_ref_to_ref

.. autofunction:: remove_measure
"""

from qiskit.circuit import BoxOp

from .annotations import map_modifier_ref_to_ref
from .boxes import box_op_inverse
from .circuit_iter import iter_circuit
from .davidson import get_extremal_eigenvalue
from .find_indices import find_indices
from .noise_model_paulis import generate_noise_model_paulis
from .remove_measure import remove_measure

# NOTE: working around lack of BoxOp.inverse - see https://github.com/Qiskit/qiskit/issues/14473
BoxOp.inverse = box_op_inverse

__all__ = [
    "find_indices",
    "generate_noise_model_paulis",
    "get_extremal_eigenvalue",
    "iter_circuit",
    "map_modifier_ref_to_ref",
    "remove_measure",
]
