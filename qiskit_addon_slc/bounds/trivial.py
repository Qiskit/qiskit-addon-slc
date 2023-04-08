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

"""A utility for constructing a Bounds data structure with trivial limits."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import BoxOp, QuantumCircuit
from qiskit.quantum_info import PauliLindbladMap, QubitSparsePauliList
from samplomatic.annotations import InjectNoise
from samplomatic.utils import get_annotation

from .commutator_bounds import Bounds


def trivial_bounds(
    circuit: QuantumCircuit,
    noise_model_paulis: dict[str, QubitSparsePauliList],
) -> Bounds:
    """Constructs a Bounds mapping with trivial bound values.

    This is a utility function for testing and debugging purposes and is unlikely to be useful
    during an actual PEC+SLC computation.

    Args:
        circuit: the circuit with :class:`.BoxOp` instructions that contain :class:`.InjectNoise`
            annotations.
        noise_model_paulis: the mapping of :attr:`.InjectNoise.ref` identifiers to the noise model's
            list of sparse Pauli terms.

    Returns:
        A :class:`.Bounds` mapping with all bound values set to ``2.0``. The value of ``2.0`` is
        used because a Pauli observable bounded on the range ``[-1, +1]`` cannot be biased by more
        than ``2.0``.
    """
    bounds: Bounds = {}
    for circ_inst in circuit:
        op = circ_inst.operation
        if not isinstance(op, BoxOp):
            continue

        inject_noise = get_annotation(op, InjectNoise)
        if inject_noise is None:
            # non-noisy box
            continue

        noise_terms = noise_model_paulis[inject_noise.ref]
        num_terms = len(noise_terms)
        bounds[inject_noise.modifier_ref] = PauliLindbladMap.from_components(
            np.full(num_terms, 2.0),
            noise_terms,
        )

    return bounds
