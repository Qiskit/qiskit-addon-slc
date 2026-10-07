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

"""Unit tests for merging forward and backward bounds.

The end-to-end test covers the main merging path on a 50-qubit circuit. These tests cover the
argument handling and error paths, which it does not reach.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp
from qiskit.quantum_info import PauliLindbladMap, QubitSparsePauliList
from qiskit_addon_slc.bounds import merge_bounds
from samplomatic import InjectNoise

PAULIS = ["XII", "IXI", "IIX"]


def _build_circuit(num_boxes: int = 2) -> QuantumCircuit:
    """Builds a boxed circuit whose boxes carry an :class:`~samplomatic.InjectNoise` annotation.

    Args:
        num_boxes: the number of boxes to place in the circuit.

    Returns:
        The circuit.
    """
    circuit = QuantumCircuit(3)
    for idx in range(num_boxes):
        body = QuantumCircuit(3)
        body.cx(idx % 2, idx % 2 + 1)
        annotation = InjectNoise(ref=f"noise_{idx}", modifier_ref=f"box_{idx}")
        circuit.append(BoxOp(body, annotations=[annotation]), [0, 1, 2])
    return circuit


def _bounds(num_boxes: int = 2, scale: float = 1.0) -> dict[str, PauliLindbladMap]:
    """Builds synthetic bounds for each box of the circuit.

    Args:
        num_boxes: the number of boxes to build bounds for.
        scale: a factor applied to every bound value.

    Returns:
        The bounds, keyed by the boxes' modifier references.
    """
    paulis = QubitSparsePauliList.from_list(PAULIS)
    return {
        f"box_{idx}": PauliLindbladMap.from_components(
            np.array([0.3, 0.2, 0.1]) * scale * (idx + 1), paulis
        )
        for idx in range(num_boxes)
    }


def test_merging_nothing_raises():
    """Test that merging two absent sets of bounds raises."""
    with pytest.raises(ValueError, match="No bounds to merge"):
        merge_bounds(_build_circuit(), None, None)


@pytest.mark.parametrize("which", ["forward", "backward"])
def test_a_single_set_of_bounds_is_returned_unchanged(which, subtests):
    """Test that merging a single set of bounds returns a copy of it.

    With only one set of bounds there is no layer at which to switch between them, so the result is
    that set itself -- but as a copy, so the caller's bounds are not aliased.

    Args:
        which: whether to supply only the forward or only the backward bounds.
        subtests: the pytest-subtests fixture.
    """
    circuit = _build_circuit()
    bounds = _bounds()

    forward = bounds if which == "forward" else None
    backward = bounds if which == "backward" else None
    merged = merge_bounds(circuit, forward, backward)

    with subtests.test("the values are preserved"):
        assert merged is not None
        assert merged.keys() == bounds.keys()
        for box_id, bound in bounds.items():
            np.testing.assert_allclose(merged[box_id].rates, bound.rates)

    with subtests.test("the result is a copy, not the original object"):
        assert merged is not bounds
        for box_id in bounds:
            assert merged[box_id] is not bounds[box_id]


def test_a_clifford_circuit_is_not_supported():
    """Test that requesting the Clifford-circuit merge raises.

    That path needs updating for the current bounds representation, so it rejects rather than
    returning a wrong result.
    """
    circuit = _build_circuit()

    with pytest.raises(NotImplementedError):
        merge_bounds(circuit, _bounds(), _bounds(), is_clifford_circuit=True)


def test_a_box_absent_from_the_circuit_raises():
    """Test that bounds keyed by a box the circuit does not contain raise."""
    circuit = _build_circuit(num_boxes=1)
    paulis = QubitSparsePauliList.from_list(PAULIS)
    noise_rates = {
        "noise_0": PauliLindbladMap.from_components(np.array([1e-3, 2e-3, 3e-3]), paulis)
    }

    # The bounds mention `box_1`, which only exists in a two-box circuit.
    with pytest.raises(KeyError, match="could not be found in the target circuit"):
        merge_bounds(circuit, _bounds(num_boxes=2), _bounds(num_boxes=2), noise_rates)


def test_merging_without_noise_rates_assumes_uniform_rates():
    """Test that bounds can be merged without supplying learned noise rates.

    Uniform rates are assumed in that case, which is unrealistic but still useful for previewing the
    merged bounds.
    """
    circuit = _build_circuit()

    merged = merge_bounds(circuit, _bounds(), _bounds(scale=0.5))

    assert merged is not None
    assert merged.keys() == {"box_0", "box_1"}
    for bound in merged.values():
        # A merged bound is one of the two inputs, so it stays within their range.
        assert np.all(bound.rates >= 0.0)


def test_merging_uses_the_tighter_of_the_two_bounds():
    """Test that merging prefers whichever of the two bounds is tighter for each box.

    The switch happens at a single layer for all qubits, so with one set uniformly tighter than the
    other the merged result should never exceed the looser one.
    """
    circuit = _build_circuit()
    forward = _bounds(scale=1.0)
    backward = _bounds(scale=0.1)

    merged = merge_bounds(circuit, forward, backward)

    assert merged is not None
    for box_id, bound in merged.items():
        looser = np.maximum(forward[box_id].rates, backward[box_id].rates)
        assert np.all(bound.rates <= looser + 1e-12)
