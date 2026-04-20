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

"""Tests for the forward bounds."""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit_addon_slc.bounds import compute_forward_bounds, merge_bounds
from qiskit_addon_slc.bounds.trivial import trivial_bounds
from qiskit_addon_slc.utils import generate_noise_model_paulis
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import find_unique_box_instructions

from .. import construct_trotter_circuit

RANDOM_SEED = 42


def test_max_num_boxes():
    """Test limiting the maximum number of BoxOps for which to compute forward bounds."""
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    num_qubits = 50
    circuit = construct_trotter_circuit(
        num_qubits=num_qubits,
        num_trotter_steps=4,
        rx_angle=np.pi / 16,
        rzz_angle=-np.pi / 2,
        use_clifford=False,
    )

    boxes_pm = generate_boxing_pass_manager(
        enable_gates=True,
        enable_measures=True,
        twirling_strategy="active",
        inject_noise_targets="all",
        inject_noise_strategy="individual_modification",
        measure_annotations="all",
        remove_barriers=False,
    )
    boxed_circuit = boxes_pm.run(circuit)

    instructions = find_unique_box_instructions(boxed_circuit)

    noise_model_paulis = generate_noise_model_paulis(instructions)

    obs_pauli = Pauli("I" * num_qubits).compose("XYZ", [12, 24, 36])

    max_num_boxes = 2
    forward_bounds = compute_forward_bounds(
        boxed_circuit,
        noise_model_paulis,
        obs_pauli,
        eigval_max_qubits=20,
        evolution_max_terms=1000,
        atol=1e-18,
        max_num_boxes=max_num_boxes,
    )

    trivial = trivial_bounds(boxed_circuit, noise_model_paulis)

    # NOTE: here one would get the the noise model rates from the NoiseLearner
    noise_model_rates = {noise_id: None for noise_id in noise_model_paulis}

    merged_bounds = merge_bounds(
        boxed_circuit,
        forward_bounds,
        trivial,
        noise_model_rates,
        is_clifford_circuit=False,
    )

    with open(Path(__file__).parent.parent / "expected_fwd_bounds.pickle", "rb") as file:
        actual_fwd_bounds = {box: bounds for box, bounds in merged_bounds.items()}
        expected_fwd_bounds = pickle.load(file)
        for idx, key in enumerate(expected_fwd_bounds):
            assert key in actual_fwd_bounds
            expected = (
                np.full(len(expected_fwd_bounds[key]), 2.0)
                if idx >= max_num_boxes
                else expected_fwd_bounds[key]
            )
            assert np.allclose(actual_fwd_bounds[key].rates, expected)


if __name__ == "__main__":
    test_max_num_boxes()
