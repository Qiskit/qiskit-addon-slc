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

"""A utility function to identify qubit indices inside a ``QuantumCircuit``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from qiskit.circuit import Bit, CircuitError, CircuitInstruction, QuantumCircuit


def find_indices(
    circuit: QuantumCircuit,
    bits_or_instruction: Bit | CircuitInstruction | Sequence[Bit],
    /,
) -> int | list[int]:
    """Find the qubit-indices of the provided bit indices or circuit instructions.

    .. caution::
       This function is not considered part of the stable API! It will get removed without warning
       or deprecation when the same functionality is supported natively by the Qiskit SDK. See
       `this issue <https://github.com/Qiskit/qiskit/issues/14558>`_ for more details.

    Args:
        circuit: the quantum circuit whose qubit indices to find.
        bits_or_instruction: the bits whose indices to find. If this is a
            :class:`~qiskit.circuit.CircuitInstruction`, the qubits which this instruction acts upon
            are used.

    Returns:
        The indices of the queried bits in the circuit's registers. If a single bit object was
        provided, a single ``int`` is returned for its index. Otherwise the return type will be a
        ``list[int]`` whose length equals the number of provided bits.

    Raises:
        TypeError: when an unexpected type of ``bits_or_instruction`` gets provided.
    """
    if isinstance(bits_or_instruction, Bit):
        return cast(int, circuit.find_bit(bits_or_instruction).index)

    if isinstance(bits_or_instruction, CircuitInstruction):
        return [circuit.find_bit(qb).index for qb in bits_or_instruction.qubits]

    if isinstance(bits_or_instruction, (list, tuple, np.ndarray)):
        try:
            return [circuit.find_bit(qb).index for qb in bits_or_instruction]
        except CircuitError:
            pass

    raise TypeError(f"Unexpected {type(bits_or_instruction) = }")
