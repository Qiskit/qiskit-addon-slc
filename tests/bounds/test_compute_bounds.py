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

"""Tests for the generic bound computation."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from pauli_prop.propagation import RotationGates
from qiskit.quantum_info import Pauli
from qiskit_addon_slc.bounds import CommutatorBounds, LightCone, compute_bounds
from qiskit_addon_slc.bounds.commutator_bounds import Bounds
from qiskit_addon_slc.utils import generate_noise_model_paulis, remove_measure
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import find_unique_box_instructions

from .. import construct_trotter_circuit

CUSTOM_BOUND = 0.5


def _custom_norm_fn(_pauli: Pauli, _rotation_gates: RotationGates) -> CommutatorBounds:
    """A custom norm function of the documented signature, with arbitrarily named parameters."""
    return CommutatorBounds(CUSTOM_BOUND, 0.0, False)


def _failing_norm_fn(pauli: Pauli, _gates: RotationGates) -> CommutatorBounds:
    """A custom norm function which fails for every error term containing a Pauli Y."""
    if np.any(pauli.x & pauli.z):
        raise ValueError("cannot bound error terms containing a Pauli Y")
    return CommutatorBounds(CUSTOM_BOUND, 0.0, False)


def _compute_bounds_with(norm_fn: Callable[[Pauli, RotationGates], CommutatorBounds]) -> Bounds:
    """Computes the bounds of a small Trotter circuit with the provided norm function."""
    num_qubits = 10
    circuit = construct_trotter_circuit(
        num_qubits=num_qubits,
        num_trotter_steps=2,
        rx_angle=np.pi / 16,
        rzz_angle=-np.pi / 2,
        use_clifford=False,
    )
    boxed_circuit = generate_boxing_pass_manager(
        enable_gates=True,
        enable_measures=True,
        twirling_strategy="active",
        inject_noise_targets="all",
        inject_noise_strategy="individual_modification",
        measure_annotations="all",
        remove_barriers=False,
    ).run(circuit)
    noise_model_paulis = generate_noise_model_paulis(find_unique_box_instructions(boxed_circuit))

    circuit = remove_measure(boxed_circuit)
    observable = Pauli("I" * num_qubits).compose("Z", [num_qubits // 2])
    light_cone = LightCone.initialize_from_pauli(circuit, observable)

    return compute_bounds(circuit, noise_model_paulis, light_cone, norm_fn, backwards=False)


def test_custom_norm_fn():
    """Test that compute_bounds passes its arguments to a custom norm function positionally."""
    bounds = _compute_bounds_with(_custom_norm_fn)

    rates = np.concatenate([bound.rates for bound in bounds.values()])
    # If calling the norm function failed, every error term would keep the trivial bound of 2.0.
    assert np.all(rates == CUSTOM_BOUND)


def test_failing_norm_fn_is_logged(caplog):
    """Test that failures of the norm function are logged, once in detail and then summarized."""
    with caplog.at_level(logging.WARNING, logger="qiskit_addon_slc.bounds.commutator_bounds"):
        bounds = _compute_bounds_with(_failing_norm_fn)

    # The error terms whose bound failed keep the trivial bound, all others got bounded
    num_failed = 0
    for bound in bounds.values():
        generators = bound.get_qubit_sparse_pauli_list_copy().to_pauli_list()
        has_y = np.any(generators.x & generators.z, axis=1)
        assert np.all(bound.rates[has_y] == 2.0)
        assert np.all(bound.rates[~has_y] == CUSTOM_BOUND)
        num_failed += int(np.sum(has_y))
    assert num_failed > 0

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 2
    first, summary = warnings

    # The first failure gets logged with the traceback from within the worker process
    assert "failed" in first.getMessage()
    assert first.exc_info is not None
    assert first.exc_info[0] is ValueError
    assert "cannot bound error terms containing a Pauli Y" in caplog.text
    assert "_failing_norm_fn" in caplog.text

    # All failures get summarized once the computation has finished
    assert f"[{num_failed}/" in summary.getMessage()
    assert f"{num_failed} x ValueError" in summary.getMessage()
