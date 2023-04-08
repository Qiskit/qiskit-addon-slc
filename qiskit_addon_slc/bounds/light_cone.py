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

"""A stateful light-cone data structure."""

from __future__ import annotations

import logging
import sys
from typing import NamedTuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Gate, Qubit
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.circuit.library import PauliGate, ZGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Pauli

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

LOGGER = logging.getLogger(__name__)


class LightCone(NamedTuple):
    # NOTE: we are not using Qiskit's LightCone transpiler pass because we want to leverage the
    # iterative nature of the circuit processing, allowing us to add the circuit instructions to the
    # light cone one at a time. However, transpiler passes are (by design) stateless and, thus,
    # cannot support such an iterative processing. Furthermore, we want to avoid back-and-forth
    # conversions of our QuantumCircuit and a DAGCircuit.
    """A simple light cone data structure."""

    qubits: set[Qubit]
    """The qubits spanned by the light cone."""

    operations: list[tuple[Gate, list[Qubit]]]
    """The operations contained in the light cone."""

    def commutes(self, inst: CircuitInstruction) -> bool:
        """Checks whether the provided instruction commutes with this light cone.

        If it does not, the instruction gets added to this light cone.

        Args:
            inst: the circuit instruction to check.

        Returns:
            Whether the instruction commutes with this light cone.
        """
        # Check if the node belongs to the light-cone
        if not self.qubits.intersection(inst.qubits):
            return True

        commutes_bool = True
        # Check commutation with all previous operations
        for op in self.operations:
            max_num_qubits = max(len(op[1]), len(inst.qubits))
            if max_num_qubits > 10:
                LOGGER.warning(
                    f"Checking commutation of operators of size {max_num_qubits}."
                    "This operation can be slow."
                )
            commute_bool = scc.commute(
                op[0],
                op[1],
                [],
                inst.operation,
                inst.qubits,
                [],
                max_num_qubits=max_num_qubits,
            )
            if not commute_bool:
                # If the current node does not commute, update the light-cone
                self.qubits.update(inst.qubits)
                self.operations.append((inst.operation, inst.qubits))
                commutes_bool = False
                break

        return commutes_bool

    @classmethod
    def initialize_from_pauli(cls, circuit: QuantumCircuit, pauli: Pauli) -> Self:
        """Initialize a LightCone assuming the given pauli is measured on the given circuit.

        Args:
            circuit: the QuantumCircuit for which to initialize the LightCone.
            pauli: the Pauli to be measured.

        Returns:
            An empty LightCone initialized with the provided pauli measured on the circuit.
        """
        mask = pauli.z | pauli.x
        indices = list(np.where(mask)[0])
        bit_terms = str(pauli[mask])
        qargs = [circuit.qubits[i] for i in indices]

        return cls(set(qargs), [(PauliGate(bit_terms), qargs)])

    @classmethod
    def initialize_from_measurements(
        cls,
        circuit: QuantumCircuit,
        *,
        measure_active: bool = False,
        measure_all: bool = False,
    ) -> Self:
        """Initialize a LightCone using the measurement gates inside the circuit.

        Args:
            circuit: the QuantumCircuit for which to initialize the LightCone.
            measure_active: ignore any measurement information from the circuit and simply assume
                all non-idle qubits to be measured.
            measure_all: ignore any measurement information from the circuit and simply assume all
                qubits to be measured. This takes precedence over ``measure_active``.

        Returns:
            An empty LightCone initialized with the to-be-measured qubits.
        """
        if measure_all:
            qargs = circuit.qubits
        elif measure_active:
            dag = circuit_to_dag(circuit)
            qargs = [qubit for qubit in circuit.qubits if qubit not in dag.idle_wires()]
        else:
            raise NotImplementedError

        return cls(set(qargs), [(ZGate(), [idx]) for idx in qargs])
