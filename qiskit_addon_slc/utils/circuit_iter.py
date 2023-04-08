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

"""A circuit iteration utility."""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import cast

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, BoxOp, CircuitInstruction
from samplomatic.annotations import InjectNoise
from samplomatic.utils import get_annotation

from .find_indices import find_indices

LOGGER = logging.getLogger(__name__)


def iter_circuit(
    circuit: QuantumCircuit, *, reverse: bool = False, log_process: bool = True
) -> Generator[tuple[CircuitInstruction, list[int], str | None, str | None]]:
    """Iterates over the instructions in a circuit.

    .. note::
       This function recurses into :class:`~qiskit.circuit.BoxOp` instructions.

    .. note::
       Barriers in the circuit are being ignored.

    Args:
        circuit: the circuit to iterate over.
        reverse: whether to iterate the circuit in reverse order.
        log_process: whether to log process.

    Yields:
        Tuples of length four, consisting of the encountered circuit instruction, the `canonical
        qubit indices
        <https://qiskit.github.io/samplomatic/guides/samplex_io.html#qubit-ordering-convention>`_
        (i.e. the integer indices of the acted-upon qubits in the context of the input ``circuit``),
        the :class:`~samplomatic.InjectNoise` attributes: ``modifier_ref`` and ``ref``. The latter
        two items may be ``None`` indicating a circuit instruction that was **not** part of an
        unrolled :class:`~qiskit.circuit.BoxOp`.
    """
    # NOTE: we ensure a minimum length of 1 in case we iterate an empty BoxOp
    circ_len = max(1, len(circuit))
    digits = int(np.ceil(np.log10(circ_len)))

    for circ_inst_idx, circ_inst in enumerate(circuit[:: -1 if reverse else 1]):
        if log_process:
            LOGGER.debug(
                f"Handling circuit instruction #{circ_inst_idx + 1:0{digits}}/{circ_len:0{digits}}"
            )

        op = circ_inst.operation

        # Ignore barriers
        if isinstance(op, Barrier):
            if log_process:
                LOGGER.debug(f"Ignoring instruction of type '{type(op)}'")
            continue

        box_id: str | None = None
        noise_id: str | None = None

        if not isinstance(op, BoxOp):
            canonical_qubits = [qubit for qubit in circuit.qubits if qubit in circ_inst.qubits]
            # NOTE: we know that the following will be of type list[int] because canonical_qubits is
            # of type list[int]
            qargs = cast(list[int], find_indices(circuit, canonical_qubits))
            yield circ_inst, qargs, box_id, noise_id
            continue

        inject_noise = get_annotation(op, InjectNoise)

        if inject_noise is None:
            # NOTE: we unroll the BoxOp immediately to allow gates contained within the box be
            # pruned by the LightCone pass
            box_circ = op.body
            if log_process:
                LOGGER.debug("Noiseless box encountered.")
            yield from iter_circuit(box_circ, reverse=reverse)
            continue

        noise_id = inject_noise.ref
        box_id = inject_noise.modifier_ref

        if log_process:
            LOGGER.info(f"Noisy box '{box_id}'")

        canonical_qubits = [qubit for qubit in circuit.qubits if qubit in circ_inst.qubits]
        # NOTE: we know that the following will be of type list[int] because canonical_qubits is of
        # type list[int]
        qargs = cast(list[int], find_indices(circuit, canonical_qubits))
        yield circ_inst, qargs, box_id, noise_id
