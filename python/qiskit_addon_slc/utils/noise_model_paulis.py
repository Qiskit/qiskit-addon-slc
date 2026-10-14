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

"""Utilities for working with sparse noise models."""

from __future__ import annotations

from typing import cast

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.quantum_info import QubitSparsePauliList, pauli_basis
from qiskit.transpiler import CouplingMap
from samplomatic.annotations import InjectNoise
from samplomatic.utils import get_annotation

from qiskit_addon_slc.utils.find_indices import find_indices


def generate_noise_model_paulis(
    instructions: list[CircuitInstruction],
    coupling_map: CouplingMap | None = None,
    circuit: QuantumCircuit | None = None,
) -> dict[str, QubitSparsePauliList]:
    """Generate the 1- and 2-weight Pauli terms for each of the unique 2q layer boxes provided.

    Args:
        instructions: the output of :func:`~samplomatic.utils.find_unique_box_instructions`. Any of
            the provided instructions are assumed to either consists of only measurement gates or
            correspond to a layer of 2-qubit gate instructions. For the former, the generated noise
            model will contain only single-qubit ``X`` errors, for the latter all 1- and 2-weight
            Pauli errors on the reduced coupling map will be included.
        coupling_map: the coupling map of the backend on which the instructions have been laid out.
            If this is ``None``, a 1d line of qubits is assumed.
        circuit: the transpiled circuit which has been laid out on the provided coupling map. This
            may only be ``None`` when the ``coupling_map`` is also ``None``.

    Returns:
        A dictionary mapping the ``ref`` attributes of the :class:`~samplomatic.InjectNoise`
        annotation of each unique Box to the 1- and 2-weight Pauli terms whose errors are learned
        for this box.
    """
    noise_model_paulis = {}
    for box in instructions:
        op = box.operation
        inject_noise = get_annotation(op, InjectNoise)
        if inject_noise is None:
            # non-noisy box
            continue

        reduced_coupling_map, layout = _find_reduced_coupling_map(box.qubits, circuit, coupling_map)

        num_box_qubits = op.num_qubits
        noise_terms: list[tuple[str, list[int]]] = []

        # differentiate instruction type into the two categories:
        #  1. containing only measurements
        #  2. a layer of 2-qubit gates
        is_measurement_layer = set(op.body.count_ops().keys()) == {"measure"}

        if is_measurement_layer:
            for qb in reduced_coupling_map.physical_qubits:
                noise_terms.append(("X", [qb]))
        else:
            # weight-1 errors
            weight_one = pauli_basis(num_qubits=1, weight=True)[-3:]  # ignore I
            for qb in reduced_coupling_map.physical_qubits:
                noise_terms.extend([(pauli.to_label(), [qb]) for pauli in weight_one])

            # weight-2 errors
            weight_two = pauli_basis(num_qubits=2, weight=True)[-9:]  # ignore II, I* and *I
            completed: set[tuple[int, int]] = set()
            for edge in reduced_coupling_map.get_edges():
                if edge in completed or edge[::-1] in completed:
                    continue
                noise_terms.extend([(pauli.to_label(), edge) for pauli in weight_two])
                completed.add(edge)

        pauli_list = QubitSparsePauliList.from_sparse_list(noise_terms, num_box_qubits)
        noise_model_paulis[inject_noise.ref] = pauli_list.apply_layout(layout)

    return noise_model_paulis


def _find_reduced_coupling_map(
    qubits: list[Qubit], circuit: QuantumCircuit | None, coupling_map: CouplingMap | None
) -> tuple[CouplingMap, list[int]]:
    if coupling_map is None:
        return CouplingMap.from_line(len(qubits)), list(range(len(qubits)))

    indices: list[int]
    layout: list[int]
    if circuit is None:
        indices = [qb._index for qb in qubits]
        layout = list(range(len(indices)))
    else:
        canonical_qubits = [qubit for qubit in circuit.qubits if qubit in qubits]
        indices = cast(list[int], find_indices(circuit, canonical_qubits))
        layout = np.argsort(indices).tolist()

    return coupling_map.reduce(indices, check_if_connected=False), layout
