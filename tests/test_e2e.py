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

"""End-to-end tests for the entire SLC bounds computation."""

from __future__ import annotations

import logging
import multiprocessing
import pickle
import random
from itertools import count
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, PauliLindbladMap, QubitSparsePauliList
from qiskit_addon_slc.bounds import (
    compute_backward_bounds,
    compute_forward_bounds,
    compute_local_scales,
    merge_bounds,
    tighten_with_speed_limit,
)
from qiskit_addon_slc.utils import generate_noise_model_paulis
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.transpiler.passes import AddInjectNoise
from samplomatic.utils import find_unique_box_instructions

RANDOM_SEED = 42
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(module)s %(message)s")


def _construct_trotter_circuit(
    num_qubits: int,
    num_trotter_steps: int,
    rx_angle: float,
    rzz_angle: float,
    use_clifford: bool = False,
) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)

    for _ in range(num_trotter_steps):
        circuit.rx(rx_angle, range(num_qubits))
        circuit.barrier()
        for first_qubit in (0, 1):
            for idx in range(first_qubit, num_qubits - 1, 2):
                if use_clifford:
                    assert np.isclose(rzz_angle, -np.pi / 2)
                    circuit.sdg([idx, idx + 1])
                    circuit.cz(idx, idx + 1)
                else:
                    circuit.rzz(rzz_angle, idx, idx + 1)
        circuit.barrier()

    return circuit


@pytest.mark.parametrize("use_clifford", [False, True])
def test_e2e(use_clifford: bool):
    """Test the entire SLC bounds computation in an end-to-end workflow.

    Args:
        use_clifford: whether to use Clifford gates for the Trotter circuit.
    """
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # NOTE: we hack around the fact that we check against hard-coded annotation IDs by forcing the
    # annotation counter to reset here. This should NOT be relied upon in the future.
    AddInjectNoise._REF_COUNTER = count()
    AddInjectNoise._MODIFIER_REF_COUNTER = count()

    num_qubits = 50
    circuit = _construct_trotter_circuit(
        num_qubits=num_qubits,
        num_trotter_steps=4,
        rx_angle=np.pi / 16,
        rzz_angle=-np.pi / 2,
        use_clifford=use_clifford,
    )
    if use_clifford:
        # test transparent handling of measurement gates only in one arbitrary case
        circuit.measure_active()

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

    forward_bounds = compute_forward_bounds(
        boxed_circuit,
        noise_model_paulis,
        obs_pauli,
        eigval_max_qubits=20,
        evolution_max_terms=1000,
        atol=1e-18,
        num_processes=8,
    )

    forward_tightened_bounds = tighten_with_speed_limit(
        forward_bounds, boxed_circuit, noise_model_paulis, obs_pauli
    )

    backward_bounds = compute_backward_bounds(
        boxed_circuit,
        noise_model_paulis,
        evolution_max_terms=2000,
        num_processes=8,
    )

    # NOTE: here one would get the the noise model rates from the NoiseLearner
    noise_model_rates = {noise_id: None for noise_id in noise_model_paulis}

    merged_bounds = merge_bounds(
        boxed_circuit,
        forward_bounds,
        backward_bounds,
        noise_model_rates,
        is_clifford_circuit=False,
    )

    with open(Path(__file__).parent / "expected_fwd_bounds.pickle", "rb") as file:
        actual_fwd_bounds = {box: bounds for box, bounds in forward_bounds.items()}
        expected_fwd_bounds = pickle.load(file)
        for key in expected_fwd_bounds:
            assert key in actual_fwd_bounds
            assert np.allclose(actual_fwd_bounds[key].rates, expected_fwd_bounds[key])

    with open(Path(__file__).parent / "expected_fwd_tightened_bounds.pickle", "rb") as file:
        actual_fwd_tightened_bounds = {
            box: bounds for box, bounds in forward_tightened_bounds.items()
        }
        expected_fwd_tightened_bounds = pickle.load(file)
        for key in expected_fwd_tightened_bounds:
            assert key in actual_fwd_tightened_bounds
            assert np.allclose(
                actual_fwd_tightened_bounds[key].rates, expected_fwd_tightened_bounds[key]
            )

    with open(Path(__file__).parent / "expected_bwd_bounds.pickle", "rb") as file:
        actual_bwd_bounds = {box: bounds for box, bounds in backward_bounds.items()}
        expected_bwd_bounds = pickle.load(file)
        for key in expected_bwd_bounds:
            assert key in actual_bwd_bounds
            assert np.allclose(actual_bwd_bounds[key].rates, expected_bwd_bounds[key])

    with open(Path(__file__).parent / "expected_merged_bounds.pickle", "rb") as file:
        actual_merged_bounds = {box: bounds for box, bounds in merged_bounds.items()}
        expected_merged_bounds = pickle.load(file)
        for key in expected_merged_bounds:
            assert key in actual_merged_bounds
            assert np.allclose(actual_merged_bounds[key].rates, expected_merged_bounds[key])

    ## Get results from noise learning now (not shown)
    for noise_model_id, paulis in noise_model_paulis.items():
        shuffled_paulis = [p for p in paulis.to_sparse_list()]
        random.shuffle(shuffled_paulis)
        noise_model_rates[noise_model_id] = PauliLindbladMap.from_components(
            np.random.random(len(paulis)) * 5e-3,
            QubitSparsePauliList.from_sparse_list(shuffled_paulis, paulis.num_qubits),
        )

    _, sampling_cost_without_layout, residual_bias_bound_without_layout = compute_local_scales(
        boxed_circuit,
        merged_bounds,
        noise_model_rates,
        sampling_cost_budget=np.inf,
        bias_tolerance=0.1,
    )

    # Apply some desired layout:
    layout = list(range(circuit.num_qubits))
    random.shuffle(layout)
    circuit = transpile(circuit, initial_layout=layout, optimization_level=0)

    merged_bounds = merge_bounds(
        boxed_circuit,
        forward_bounds,
        backward_bounds,
        noise_model_rates,
        is_clifford_circuit=False,
    )

    _, sampling_cost_with_layout, residual_bias_bound_with_layout = compute_local_scales(
        boxed_circuit,
        merged_bounds,
        noise_model_rates,
        sampling_cost_budget=np.inf,
        bias_tolerance=0.1,
    )

    assert np.isclose(sampling_cost_with_layout, sampling_cost_without_layout)
    assert np.isclose(residual_bias_bound_with_layout, residual_bias_bound_without_layout)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    test_e2e(use_clifford=True)
