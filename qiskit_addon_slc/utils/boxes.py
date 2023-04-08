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

"""Utilities to simplify working with BoxOp instructions.

In the long run these features are likely to be merged into the Qiskit SDK.
"""

from __future__ import annotations

from qiskit.circuit import BoxOp


def box_op_inverse(box_op: BoxOp, annotated: bool = False) -> BoxOp:
    """Inverts a BoxOp.

    .. caution::
       This function is not considered part of the stable API! It will get removed without warning
       or deprecation once the :class:`.BoxOp` implements the :meth:`.Instruction.inverse` method
       natively. See `this issue <https://github.com/Qiskit/qiskit/issues/14473>`_ for more details.

    Args:
        box_op: the instance to invert.
        annotated: whether this instruction is annotated. ``True`` is not supported.

    Returns:
        The inverted box.

    Raises:
        NotImplementedError: if ``annotated=True``.
    """
    if annotated:
        raise NotImplementedError("Cannot invert a BoxOp with annotated=True.")

    inv_box_op = box_op.replace_blocks([box_op.blocks[0].inverse(annotated=False)])

    if box_op.name.endswith("_dg"):
        inv_box_op.name = box_op.name[:-3]
    else:
        inv_box_op.name = box_op.name + "_dg"

    return inv_box_op
