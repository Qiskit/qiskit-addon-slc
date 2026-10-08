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

import numpy as np
from pauli_prop.propagation import RotationGates
from qiskit.quantum_info import Pauli
from qiskit_addon_slc.bounds import CommutatorBounds, LightCone, compute_bounds
from qiskit_addon_slc.utils import generate_noise_model_paulis, remove_measure
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import find_unique_box_instructions

from .. import construct_trotter_circuit

CUSTOM_BOUND = 0.5


def _custom_norm_fn(_pauli: Pauli, _rotation_gates: RotationGates) -> CommutatorBounds:
    """A custom norm function of the documented signature, with arbitrarily named parameters."""
    return CommutatorBounds(CUSTOM_BOUND, 0.0, False)


def test_custom_norm_fn():
    """Test that compute_bounds passes its arguments to a custom norm function positionally."""
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

    bounds = compute_bounds(
        circuit, noise_model_paulis, light_cone, _custom_norm_fn, backwards=False
    )

    rates = np.concatenate([bound.rates for bound in bounds.values()])
    # If calling the norm function failed, every error term would keep the trivial bound of 2.0.
    assert np.all(rates == CUSTOM_BOUND)
