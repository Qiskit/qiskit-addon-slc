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

"""Tests for the span attributes of the bound computation tasks."""

import pytest
from qiskit.quantum_info import Pauli, PauliList
from qiskit_addon_slc.bounds import CommutatorBounds
from qiskit_addon_slc.bounds._span_attributes import (
    num_active_qubits,
    pauli_attribute,
    record_bound,
    record_observable,
)
from qiskit_addon_slc.optionals import HAS_OPENTELEMETRY
from qiskit_addon_slc.utils._tracing import NoOpSpan


@pytest.mark.parametrize(
    ("pauli", "expected"),
    [
        (Pauli("I" * 156).compose("XYZ", [12, 24, 36]), "Z_12 Y_24 X_36"),
        (Pauli("-iIXIZ"), "-iZ_0 X_2"),
        (Pauli("Y"), "Y_0"),
        (Pauli("III"), "I"),
    ],
)
def test_pauli_attribute(pauli, expected):
    """Test that Paulis are formatted as complete sparse labels, regardless of their width."""
    assert pauli_attribute(pauli) == expected


@pytest.mark.parametrize(
    ("paulis", "expected"),
    [
        (Pauli("I" * 156).compose("XYZ", [12, 24, 36]), 3),
        (PauliList(["IXI", "ZII", "III"]), 2),
        (PauliList(["XI", "IZ"]), 2),
        (Pauli("III"), 0),
    ],
)
def test_num_active_qubits(paulis, expected):
    """Test that a qubit counts as active if any of the Paulis acts on it non-trivially."""
    assert num_active_qubits(paulis) == expected


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.parametrize(("bound", "aborted"), [(0.5, False), (float("NaN"), True)])
def test_record_bound_keeps_the_type_of_the_bound(recording_span, bound, aborted):
    """Test that an aborted bound is marked as such, rather than being recorded as a string."""
    result = CommutatorBounds(bound, 0.1, False)
    assert record_bound(recording_span, result) is result

    assert recording_span.attributes["result.aborted"] is aborted
    if aborted:
        assert "result.commutator_bound" not in recording_span.attributes
    else:
        assert recording_span.attributes["result.commutator_bound"] == bound


def test_record_bound_returns_the_bound_when_not_recording():
    """Test that the bound is passed on when the span records nothing."""
    result = CommutatorBounds(0.5, 0.1, False)
    assert record_bound(NoOpSpan(), result) is result


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_record_observable(recording_span):
    """Test that the observable is recorded by its complete sparse label and its active qubits."""
    record_observable(recording_span, Pauli("I" * 156).compose("XZ", [7, 99]))

    assert recording_span.attributes["observable.label"] == "Z_7 X_99"
    assert recording_span.attributes["observable.num_active_qubits"] == 2


def test_record_observable_when_not_recording():
    """Test that nothing gets computed for a span which records nothing."""
    record_observable(NoOpSpan(), Pauli("XZ"))
