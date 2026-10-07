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

"""Unit tests for tightening bounds with a speed limit.

The end-to-end test runs this on a 50-qubit circuit and checks the result against a fixture. These
tests use a circuit small enough to reason about, so that the tightening itself can be asserted.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp
from qiskit.quantum_info import (
    Pauli,
    PauliLindbladMap,
    PauliList,
    SparseObservable,
    SparsePauliOp,
)
from qiskit_addon_slc.bounds import tighten_with_speed_limit
from qiskit_addon_slc.utils import generate_noise_model_paulis
from samplomatic import InjectNoise
from samplomatic.utils import find_unique_box_instructions

# The theoretical maximum of a commutator bound, i.e. the loosest possible starting point.
TRIVIAL_BOUND = 2.0


def _build_inputs(
    num_boxes: int = 2,
) -> tuple[QuantumCircuit, dict[str, PauliLindbladMap], dict]:
    """Builds a boxed circuit with its noise models and trivial starting bounds.

    Starting from the trivial bound of 2 everywhere means any tightening is visible as a decrease.

    Args:
        num_boxes: the number of boxes to place in the circuit.

    Returns:
        A tuple of the circuit, the trivial bounds and the noise model Paulis.
    """
    circuit = QuantumCircuit(3)
    for idx in range(num_boxes):
        body = QuantumCircuit(3)
        body.cx(idx % 2, idx % 2 + 1)
        annotation = InjectNoise(ref=f"noise_{idx}", modifier_ref=f"box_{idx}")
        circuit.append(BoxOp(body, annotations=[annotation]), [0, 1, 2])

    noise_model_paulis = generate_noise_model_paulis(find_unique_box_instructions(circuit))
    bounds = {
        f"box_{idx}": PauliLindbladMap.from_components(
            np.full(len(noise_model_paulis[f"noise_{idx}"]), TRIVIAL_BOUND),
            noise_model_paulis[f"noise_{idx}"],
        )
        for idx in range(num_boxes)
    }
    return circuit, bounds, noise_model_paulis


def test_tightening_never_loosens_a_bound(subtests):
    """Test that the tightened bounds are no looser than the ones supplied.

    Args:
        subtests: the pytest-subtests fixture.
    """
    circuit, bounds, noise_model_paulis = _build_inputs()

    tightened = tighten_with_speed_limit(bounds, circuit, noise_model_paulis, Pauli("ZZI"))

    with subtests.test("the same boxes are reported"):
        assert tightened.keys() == bounds.keys()

    with subtests.test("no bound increases"):
        for box_id, bound in tightened.items():
            assert np.all(bound.rates <= bounds[box_id].rates + 1e-12)

    with subtests.test("at least one bound is actually tightened"):
        assert any(
            np.any(bound.rates < bounds[box_id].rates - 1e-12)
            for box_id, bound in tightened.items()
        )

    with subtests.test("the input bounds are left untouched"):
        for bound in bounds.values():
            np.testing.assert_allclose(bound.rates, TRIVIAL_BOUND)


def test_errors_outside_the_lightcone_are_tightened_to_zero():
    """Test that an error which cannot reach the observable gets a bound of zero.

    Information propagates at a limited speed, so an error term too far from the observable's support
    to influence it within the remaining circuit depth contributes no bias at all.
    """
    circuit, bounds, noise_model_paulis = _build_inputs()

    # An observable supported only on qubit 0 leaves the far end of the circuit outside its cone.
    tightened = tighten_with_speed_limit(bounds, circuit, noise_model_paulis, Pauli("IIZ"))

    assert any(np.any(bound.rates == 0.0) for bound in tightened.values())


@pytest.mark.parametrize(
    "observable",
    [
        Pauli("ZZI"),
        PauliList(["ZZI"]),
        SparsePauliOp(["ZZI"]),
        SparseObservable.from_sparse_list([("ZZ", [2, 1], 1.0)], num_qubits=3),
    ],
    ids=["Pauli", "PauliList", "SparsePauliOp", "SparseObservable"],
)
def test_accepts_every_single_term_observable_type(observable):
    """Test that each accepted observable type yields the same tightening.

    Args:
        observable: the single-term observable to tighten against.
    """
    circuit, bounds, noise_model_paulis = _build_inputs()

    tightened = tighten_with_speed_limit(bounds, circuit, noise_model_paulis, observable)

    reference = tighten_with_speed_limit(bounds, circuit, noise_model_paulis, Pauli("ZZI"))
    for box_id, bound in tightened.items():
        np.testing.assert_allclose(bound.rates, reference[box_id].rates)


@pytest.mark.parametrize(
    "observable",
    [PauliList(["ZZI", "XXI"]), SparsePauliOp(["ZZI", "XXI"])],
    ids=["PauliList", "SparsePauliOp"],
)
def test_a_multi_term_observable_is_rejected(observable):
    """Test that an observable of more than one Pauli term raises.

    Such an observable has to be handled one Pauli at a time, which the caller must do explicitly.

    Args:
        observable: the multi-term observable to reject.
    """
    circuit, bounds, noise_model_paulis = _build_inputs()

    with pytest.raises(NotImplementedError, match="more than 1 Pauli term"):
        tighten_with_speed_limit(bounds, circuit, noise_model_paulis, observable)


def test_a_three_qubit_gate_is_rejected():
    """Test that a circuit containing a gate on more than two qubits raises.

    The speed limit is derived from the circuit's two-qubit connectivity, so a wider gate has no
    defined propagation speed.
    """
    circuit = QuantumCircuit(3)
    body = QuantumCircuit(3)
    body.ccx(0, 1, 2)
    annotation = InjectNoise(ref="noise_0", modifier_ref="box_0")
    circuit.append(BoxOp(body, annotations=[annotation]), [0, 1, 2])

    noise_model_paulis = {
        "noise_0": generate_noise_model_paulis(find_unique_box_instructions(circuit))["noise_0"]
    }
    bounds = {
        "box_0": PauliLindbladMap.from_components(
            np.full(len(noise_model_paulis["noise_0"]), TRIVIAL_BOUND),
            noise_model_paulis["noise_0"],
        )
    }

    with pytest.raises(ValueError, match="more than 2 qubits"):
        tighten_with_speed_limit(bounds, circuit, noise_model_paulis, Pauli("ZZI"))
