# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# The development of this module was assisted by Bob 1.0.1 and Claude Opus 5

"""The span attributes of the bound computation tasks.

These record what the tasks of a bound computation work on, such that their spans can be told
apart. The generic tracing utilities in :mod:`qiskit_addon_slc.utils._tracing` make no assumptions
about what gets traced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp

from ..utils._tracing import get_task_index

if TYPE_CHECKING:
    from opentelemetry.trace import Span
    from pauli_prop.propagation import RotationGates

    from ..utils._tracing import NoOpSpan
    from .commutator_bounds import CommutatorBounds


def pauli_attribute(pauli: Pauli) -> str:
    """Format a Pauli as a span attribute value.

    The string representation of a :class:`~qiskit.quantum_info.Pauli` gets truncated for wide
    circuits, which hides its non-identity terms. Instead, this lists only the non-identity terms
    together with the qubits they act on, ordered by qubit index.

    Args:
        pauli: the :class:`~qiskit.quantum_info.Pauli` to format.

    Returns:
        A sparse label such as ``"X_12 Y_24 Z_36"``, preceded by the phase of the Pauli if it has
        one (e.g. ``"-iX_3"``), or ``"I"`` if the Pauli is the identity.
    """
    label: str = pauli.to_label()
    phase, dense = label[: len(label) - pauli.num_qubits], label[len(label) - pauli.num_qubits :]
    # NOTE: Qiskit labels are little-endian, i.e. qubit 0 is the rightmost character
    terms = [f"{char}_{qubit}" for qubit, char in enumerate(reversed(dense)) if char != "I"]
    return phase + (" ".join(terms) or "I")


def num_active_qubits(paulis: Pauli | PauliList) -> int:
    """Count the qubits on which any of ``paulis`` acts non-trivially.

    Args:
        paulis: the Paulis to inspect.

    Returns:
        The number of qubits on which any of ``paulis`` acts non-trivially.
    """
    return int(np.atleast_2d(paulis.x | paulis.z).any(axis=0).sum())


def record_task_inputs(span: Span | NoOpSpan, pauli: Pauli, gates: RotationGates) -> None:
    """Record the inputs of a bound computation task on its span.

    The worker which runs the task is not recorded, since the span of the task is a child of the
    span of that worker.

    Args:
        span: the span of the task.
        pauli: the Pauli error term to bound.
        gates: the gates that ``pauli`` gets evolved through.
    """
    if not span.is_recording():
        return
    span.set_attribute("pauli.label", pauli_attribute(pauli))
    span.set_attribute("pauli.num_active_qubits", num_active_qubits(pauli))
    span.set_attribute("gates.count", len(gates.gates))
    span.set_attribute("task.index", get_task_index())


def record_observable(span: Span | NoOpSpan, observable: Pauli) -> None:
    """Record the target observable of a forward bound computation on its span.

    Args:
        span: the span to record the observable on.
        observable: the Pauli observable to be measured at the end of the circuit.
    """
    if not span.is_recording():
        return
    span.set_attribute("observable.label", pauli_attribute(observable))
    span.set_attribute("observable.num_active_qubits", num_active_qubits(observable))


def record_propagation(
    span: Span | NoOpSpan, operator: SparsePauliOp, trunc_onenorm: float
) -> None:
    """Record the result of propagating an error term through the circuit on its span.

    Args:
        span: the span of the propagation.
        operator: the propagated operator.
        trunc_onenorm: the one-norm of the terms which got truncated during the propagation.
    """
    if not span.is_recording():
        return
    span.set_attribute("pauli_prop.truncation.one_norm", float(trunc_onenorm))
    span.set_attribute("pauli_prop.num_terms", len(operator))
    span.set_attribute("pauli_prop.num_active_qubits", num_active_qubits(operator.paulis))


def record_bound(span: Span | NoOpSpan, result: CommutatorBounds) -> CommutatorBounds:
    """Record the result of a bound computation task on its span.

    Args:
        span: the span of the task.
        result: the computed bound.

    Returns:
        The unchanged ``result``, such that this can wrap the returned value.
    """
    if not span.is_recording():
        return result
    # NOTE: a NaN bound is not recorded, since not every tracing backend can handle a float NaN,
    # and recording it as a string instead would give the attribute different types across spans
    aborted = bool(np.isnan(result.commutator_bound))
    if not aborted:
        span.set_attribute("result.commutator_bound", result.commutator_bound)
    span.set_attribute("result.aborted", aborted)
    span.set_attribute("result.truncation_bias", result.truncation_bias)
    span.set_attribute("result.fallback_to_tri_ineq", result.fallback_to_tri_ineq)
    return result
