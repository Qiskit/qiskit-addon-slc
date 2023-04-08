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

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import FilterOpNodes

LOGGER = logging.getLogger(__name__)


def remove_measure(circuit: QuantumCircuit) -> QuantumCircuit:
    """Remove any :class:`~qiskit.circuit.Measure` operations from the provided circuit.

    .. note::
       This function recurses into :class:`~qiskit.circuit.BoxOp` instructions.

    Args:
        circuit: the circuit whose measurements to remove.

    Returns:
        The circuit without any Measure operations.
    """
    LOGGER.info("Removing ANY Measure operations from the provided circuit!")
    remove_meas = FilterOpNodes(predicate=lambda op: op.name != "measure")
    return dag_to_circuit(remove_meas.run(circuit_to_dag(circuit)))
