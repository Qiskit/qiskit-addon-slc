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

"""Utilities to simplify working with circuit annotations."""

from __future__ import annotations

from qiskit.circuit import BoxOp, QuantumCircuit
from samplomatic.annotations import InjectNoise
from samplomatic.utils import get_annotation


def map_modifier_ref_to_ref(circuit: QuantumCircuit) -> dict[str, str]:
    """Iterate a circuit and map :class:`~samplomatic.InjectNoise` annotation references.

    Args:
        circuit: the circuit to iterate over.

    Returns:
        A dictionary mapping each ``InjectNoise.modifier_ref`` to its ``InjectNoise.ref``.
    """
    id_map = {}
    for inst in circuit:
        op = inst.operation
        if not isinstance(op, BoxOp):
            continue

        inject_noise = get_annotation(op, InjectNoise)
        if inject_noise is None:
            continue

        id_map[inject_noise.modifier_ref] = inject_noise.ref

    return id_map
