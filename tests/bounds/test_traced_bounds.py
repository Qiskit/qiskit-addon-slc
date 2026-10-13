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

"""Tests for the spans which the bound computations record.

Only the spans of the main process get exported to the ``finished_spans`` fixture. Hence, the spans
of the tasks are tested by calling the norm functions directly, while the spans of the workers and
their tasks are tested in ``tests/utils/test_tracing_workers.py``.
"""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_addon_slc import globals as slc_globals
from qiskit_addon_slc.bounds import compute_backward_bounds, compute_forward_bounds
from qiskit_addon_slc.bounds.backward import time_evolved_norm_backward
from qiskit_addon_slc.bounds.forward import time_evolved_norm_forward
from qiskit_addon_slc.optionals import HAS_OPENTELEMETRY

from .test_compute_bounds import (
    _boxed_trotter_circuit,
    _compute_bounds_with,
    _interrupted_after,
    _slow_norm_fn,
)
from .test_time_evolved_norm_forward import NO_GATES, _rotation_gates

pytestmark = pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")


def _span(spans, name):
    """Returns the only span called ``name``."""
    (span,) = (span for span in spans if span.name == name)
    return span


def _children(spans, parent):
    """Returns the names of the children of ``parent``, in the order in which they started."""
    children = [
        span for span in spans if span.parent and span.parent.span_id == parent.context.span_id
    ]
    return [span.name for span in sorted(children, key=lambda span: span.start_time)]


def _entangling_circuit() -> QuantumCircuit:
    """A circuit which spreads an X error on its last qubit over several Pauli terms."""
    circuit = QuantumCircuit(3)
    for qubit in range(3):
        circuit.rx(0.3, qubit)
    for qubit in range(2):
        circuit.rzz(0.2, qubit, qubit + 1)
    return circuit


def test_forward_norm_computation_records_its_steps(finished_spans):
    """Test that the span of a forward task records its inputs, its steps and its result."""
    gates = _rotation_gates(_entangling_circuit())
    result = time_evolved_norm_forward(Pauli("XII"), gates, Pauli("ZZI"))

    spans = finished_spans.get_finished_spans()
    task = _span(spans, "forward_norm_computation")
    assert _children(spans, task) == ["pauli_prop", "commutator", "direct_eigenvalue_computation"]
    assert task.attributes["pauli.label"] == "X_2"
    assert task.attributes["observable.label"] == "Z_1 Z_2"
    assert task.attributes["gates.count"] == len(gates.gates)
    # NOTE: the task index is only known within a worker process
    assert task.attributes["task.index"] == -1
    assert task.attributes["result.commutator_bound"] == result.commutator_bound
    assert task.attributes["result.aborted"] is False

    propagation = _span(spans, "pauli_prop").attributes
    assert propagation["pauli_prop.num_terms"] > 1
    assert propagation["pauli_prop.num_active_qubits"] > 1
    assert propagation["pauli_prop.truncation.one_norm"] >= 0.0

    commutator = _span(spans, "commutator").attributes
    assert commutator["commutator.num_terms"] > 1
    assert 0 < commutator["commutator.num_active_qubits"] <= 3
    assert (
        commutator["commutator.min_abs_coeff"]
        <= commutator["commutator.mean_abs_coeff"]
        <= commutator["commutator.max_abs_coeff"]
    )


def test_forward_norm_computation_records_its_shortcuts(finished_spans):
    """Test that a shortcut of a forward task is recorded as an event on its span."""
    time_evolved_norm_forward(Pauli("X"), NO_GATES, Pauli("Z"))

    spans = finished_spans.get_finished_spans()
    task = _span(spans, "forward_norm_computation")
    assert [event.name for event in task.events] == ["single_pauli_optimization"]
    assert _children(spans, task) == ["pauli_prop"]


def test_backward_norm_computation_records_its_steps(finished_spans):
    """Test that the span of a backward task records its inputs, its steps and its result."""
    gates = _rotation_gates(_entangling_circuit())
    result = time_evolved_norm_backward(Pauli("XII"), gates)

    spans = finished_spans.get_finished_spans()
    task = _span(spans, "backward_norm_computation")
    assert _children(spans, task) == ["pauli_prop"]
    assert task.attributes["pauli.label"] == "X_2"
    assert task.attributes["result.commutator_bound"] == result.commutator_bound
    assert _span(spans, "pauli_prop").attributes["pauli_prop.num_terms"] > 1


@pytest.mark.parametrize("backwards", [False, True])
def test_bound_computation_records_its_preparation_and_inputs(finished_spans, backwards):
    """Test that the span of a bound computation records what is measured, and how it is split up."""
    num_qubits = 6
    circuit, noise_model_paulis = _boxed_trotter_circuit(num_qubits)
    if backwards:
        compute_backward_bounds(circuit, noise_model_paulis, num_processes=1)
    else:
        observable = Pauli("I" * num_qubits).compose("Z", [3])
        compute_forward_bounds(circuit, noise_model_paulis, observable, num_processes=1)

    spans = finished_spans.get_finished_spans()
    root = _span(spans, "compute_backward_bounds" if backwards else "compute_forward_bounds")
    assert root.parent is None
    assert _children(spans, root) == ["preparation", "compute_bounds"]
    assert {"worker_0", "task_spawning", "progress_tracking"} <= set(
        _children(spans, _span(spans, "compute_bounds"))
    )
    assert root.attributes["stop_reason"] == "completed"
    if backwards:
        assert root.attributes["measurements.num_qubits"] == num_qubits
    else:
        assert root.attributes["observable.label"] == "Z_3"


def test_interrupt_while_tracking_progress_is_traced(finished_spans, monkeypatch):
    """Test that an interrupt while waiting for the tasks is marked in the trace."""
    # NOTE: see test_interrupt_while_tracking_progress_keeps_the_partial_bounds
    monkeypatch.setattr(slc_globals, "PROGRESS_POLLING_PERIOD", 0.1)
    with _interrupted_after(1.0):
        _compute_bounds_with(_slow_norm_fn, num_processes=2)

    spans = finished_spans.get_finished_spans()
    bounds_span = _span(spans, "compute_bounds")
    assert bounds_span.attributes["stop_reason"] == "keyboard_interrupt"
    assert bounds_span.attributes["tasks.completed"] < bounds_span.attributes["tasks.total"]
    marker = _span(spans, "keyboard_interrupt_received")
    assert marker.parent.span_id == bounds_span.context.span_id
    workers = [span for span in spans if span.name.startswith("worker_")]
    assert len(workers) == 2
    assert all(span.attributes["worker.terminated"] for span in workers)
