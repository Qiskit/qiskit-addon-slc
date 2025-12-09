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

"""Tests for the noise model utilities."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import PauliList
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_addon_slc.utils.noise_model_paulis import generate_noise_model_paulis
from qiskit_ibm_runtime.fake_provider import FakeFez
from samplomatic.annotations import InjectNoise
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import find_unique_box_instructions, get_annotation


def _compare_pauli_lists(left: PauliList, right: PauliList) -> bool:
    return list(sorted(left.to_labels())) == list(sorted(right.to_labels()))


@pytest.mark.parametrize("num_qubits", range(4, 10))
def test_noise_model_paulis(subtests, num_qubits: int) -> None:
    """Test generating the noise model paulis of a given circuit layer.

    Args:
        subtests: the pytest-subtests fixture.
        num_qubits: the number of qubits of the 2-layered circuit.
    """
    circ = QuantumCircuit(num_qubits)
    # even layer
    for idx in range(1, num_qubits, 2):
        circ.cx(idx - 1, idx)
    # odd layer
    for idx in range(2, num_qubits, 2):
        circ.cx(idx - 1, idx)

    boxes_pm = generate_boxing_pass_manager(
        inject_noise_targets="all",
        inject_noise_strategy="individual_modification",
        twirling_strategy="active",
    )
    boxed_circ = boxes_pm.run(circ)

    instructions = find_unique_box_instructions(boxed_circ)
    refs = [get_annotation(inst.operation, InjectNoise).ref for inst in instructions[:2]]

    noise_model_paulis = generate_noise_model_paulis(instructions)

    xyz_1q_x = np.asarray([1, 1, 0], dtype=bool)
    xyz_1q_z = xyz_1q_x[::-1]

    xyz_2q_x = np.asarray(
        [[1, 1], [1, 1], [0, 1], [1, 1], [1, 1], [0, 1], [1, 0], [1, 0], [0, 0]], dtype=bool
    )
    xyz_2q_z = xyz_2q_x[::-1]

    def tile(n_q: int):
        """Tile the sparse noise model's PauliList for the given number of qubits."""
        rows = 3 * n_q + 9 * (n_q - 1)
        x = np.zeros((rows, n_q), dtype=bool)
        z = np.zeros((rows, n_q), dtype=bool)
        for idx in range(n_q):
            start = 3 * idx
            end = 3 * (idx + 1)
            x[start:end, idx] = xyz_1q_x
            z[start:end, idx] = xyz_1q_z
            if idx > 0:
                offset = 3 * n_q
                start = 9 * (idx - 1)
                end = 9 * idx
                x[offset + start : offset + end, idx - 1 : idx + 1] = xyz_2q_x
                z[offset + start : offset + end, idx - 1 : idx + 1] = xyz_2q_z

        return PauliList.from_symplectic(z, x)

    r0 = tile(instructions[0].operation.body.num_qubits)
    with subtests.test("r0 noise model"):
        assert _compare_pauli_lists(noise_model_paulis[refs[0]].to_pauli_list(), r0)

    r1 = tile(instructions[1].operation.body.num_qubits)
    with subtests.test("r1 noise model"):
        assert _compare_pauli_lists(noise_model_paulis[refs[1]].to_pauli_list(), r1)


def test_noise_model_paulis_with_backend(subtests) -> None:
    """Test generating the noise model paulis while taking into account a backend coupling map.

    Args:
        subtests: the pytest-subtests fixture.
    """
    num_qubits = 12
    circ = QuantumCircuit(num_qubits)
    # even layer
    for idx in range(1, num_qubits, 2):
        circ.cx(idx - 1, idx)
    # odd layer
    for idx in range(2, num_qubits, 2):
        circ.cx(idx - 1, idx)

    backend = FakeFez()
    layout = [3, 4, 5, 6, 7, 17, 27, 26, 25, 24, 23, 16]

    preset_pm = generate_preset_pass_manager(
        backend=backend, initial_layout=layout, optimization_level=0
    )
    isa_circ = preset_pm.run(circ)

    boxes_pm = generate_boxing_pass_manager(
        inject_noise_targets="all",
        inject_noise_strategy="individual_modification",
        twirling_strategy="active",
    )
    boxed_circ = boxes_pm.run(isa_circ)

    instructions = find_unique_box_instructions(boxed_circ)
    refs = [get_annotation(inst.operation, InjectNoise).ref for inst in instructions[:2]]

    noise_model_paulis = generate_noise_model_paulis(
        instructions, coupling_map=backend.coupling_map, circuit=boxed_circ
    )

    # fmt: off
    even_paulilist = PauliList(
        [
            "IIIIIIIIIIIX", "IIIIIIIIIIIY", "IIIIIIIIIIIZ", "IIIIIIIIIIXI", "IIIIIIIIIIYI",
            "IIIIIIIIIIZI", "IIIIIIIIIXII", "IIIIIIIIIYII", "IIIIIIIIIZII", "IIIIIIIIXIII",
            "IIIIIIIIYIII", "IIIIIIIIZIII", "IIIIIIIXIIII", "IIIIIIIYIIII", "IIIIIIIZIIII",
            "IIIIIIXIIIII", "IIIIIIYIIIII", "IIIIIIZIIIII", "IIIIIXIIIIII", "IIIIIYIIIIII",
            "IIIIIZIIIIII", "IIIIXIIIIIII", "IIIIYIIIIIII", "IIIIZIIIIIII", "IIIXIIIIIIII",
            "IIIYIIIIIIII", "IIIZIIIIIIII", "IIXIIIIIIIII", "IIYIIIIIIIII", "IIZIIIIIIIII",
            "IXIIIIIIIIII", "IYIIIIIIIIII", "IZIIIIIIIIII", "XIIIIIIIIIII", "YIIIIIIIIIII",
            "ZIIIIIIIIIII", "IIIIIIIIIIXX", "IIIIIIIIIIXY", "IIIIIIIIIIXZ", "IIIIIIIIIIYX",
            "IIIIIIIIIIYY", "IIIIIIIIIIYZ", "IIIIIIIIIIZX", "IIIIIIIIIIZY", "IIIIIIIIIIZZ",
            "IIIIIIXIIIIX", "IIIIIIXIIIIY", "IIIIIIXIIIIZ", "IIIIIIYIIIIX", "IIIIIIYIIIIY",
            "IIIIIIYIIIIZ", "IIIIIIZIIIIX", "IIIIIIZIIIIY", "IIIIIIZIIIIZ", "IIIIIIIIIXXI",
            "IIIIIIIIIXYI", "IIIIIIIIIXZI", "IIIIIIIIIYXI", "IIIIIIIIIYYI", "IIIIIIIIIYZI",
            "IIIIIIIIIZXI", "IIIIIIIIIZYI", "IIIIIIIIIZZI", "IIIIIIIIXXII", "IIIIIIIIXYII",
            "IIIIIIIIXZII", "IIIIIIIIYXII", "IIIIIIIIYYII", "IIIIIIIIYZII", "IIIIIIIIZXII",
            "IIIIIIIIZYII", "IIIIIIIIZZII", "IIIIIIIXXIII", "IIIIIIIXYIII", "IIIIIIIXZIII",
            "IIIIIIIYXIII", "IIIIIIIYYIII", "IIIIIIIYZIII", "IIIIIIIZXIII", "IIIIIIIZYIII",
            "IIIIIIIZZIII", "IIIIIXIXIIII", "IIIIIXIYIIII", "IIIIIXIZIIII", "IIIIIYIXIIII",
            "IIIIIYIYIIII", "IIIIIYIZIIII", "IIIIIZIXIIII", "IIIIIZIYIIII", "IIIIIZIZIIII",
            "IIIIXIXIIIII", "IIIIXIYIIIII", "IIIIXIZIIIII", "IIIIYIXIIIII", "IIIIYIYIIIII",
            "IIIIYIZIIIII", "IIIIZIXIIIII", "IIIIZIYIIIII", "IIIIZIZIIIII", "XIIIIXIIIIII",
            "XIIIIYIIIIII", "XIIIIZIIIIII", "YIIIIXIIIIII", "YIIIIYIIIIII", "YIIIIZIIIIII",
            "ZIIIIXIIIIII", "ZIIIIYIIIIII", "ZIIIIZIIIIII", "IIIXXIIIIIII", "IIIXYIIIIIII",
            "IIIXZIIIIIII", "IIIYXIIIIIII", "IIIYYIIIIIII", "IIIYZIIIIIII", "IIIZXIIIIIII",
            "IIIZYIIIIIII", "IIIZZIIIIIII", "IIXXIIIIIIII", "IIXYIIIIIIII", "IIXZIIIIIIII",
            "IIYXIIIIIIII", "IIYYIIIIIIII", "IIYZIIIIIIII", "IIZXIIIIIIII", "IIZYIIIIIIII",
            "IIZZIIIIIIII", "IXXIIIIIIIII", "IXYIIIIIIIII", "IXZIIIIIIIII", "IYXIIIIIIIII",
            "IYYIIIIIIIII", "IYZIIIIIIIII", "IZXIIIIIIIII", "IZYIIIIIIIII", "IZZIIIIIIIII",
            "XXIIIIIIIIII", "XYIIIIIIIIII", "XZIIIIIIIIII", "YXIIIIIIIIII", "YYIIIIIIIIII",
            "YZIIIIIIIIII", "ZXIIIIIIIIII", "ZYIIIIIIIIII", "ZZIIIIIIIIII"
        ]
    )
    # fmt: on
    with subtests.test("even layer noide model"):
        assert _compare_pauli_lists(noise_model_paulis[refs[0]].to_pauli_list(), even_paulilist)

    # fmt: off
    odd_paulilist = PauliList(
        [
            "IIIIIIIIIX", "IIIIIIIIIY", "IIIIIIIIIZ", "IIIIIIIIXI", "IIIIIIIIYI", "IIIIIIIIZI",
            "IIIIIIIXII", "IIIIIIIYII", "IIIIIIIZII", "IIIIIIXIII", "IIIIIIYIII", "IIIIIIZIII",
            "IIIIIXIIII", "IIIIIYIIII", "IIIIIZIIII", "IIIIXIIIII", "IIIIYIIIII", "IIIIZIIIII",
            "IIIXIIIIII", "IIIYIIIIII", "IIIZIIIIII", "IIXIIIIIII", "IIYIIIIIII", "IIZIIIIIII",
            "IXIIIIIIII", "IYIIIIIIII", "IZIIIIIIII", "XIIIIIIIII", "YIIIIIIIII", "ZIIIIIIIII",
            "IIIIIIIIXX", "IIIIIIIIXY", "IIIIIIIIXZ", "IIIIIIIIYX", "IIIIIIIIYY", "IIIIIIIIYZ",
            "IIIIIIIIZX", "IIIIIIIIZY", "IIIIIIIIZZ", "IIIIIIIXXI", "IIIIIIIXYI", "IIIIIIIXZI",
            "IIIIIIIYXI", "IIIIIIIYYI", "IIIIIIIYZI", "IIIIIIIZXI", "IIIIIIIZYI", "IIIIIIIZZI",
            "IIIIIIXXII", "IIIIIIXYII", "IIIIIIXZII", "IIIIIIYXII", "IIIIIIYYII", "IIIIIIYZII",
            "IIIIIIZXII", "IIIIIIZYII", "IIIIIIZZII", "IIIIIXXIII", "IIIIIXYIII", "IIIIIXZIII",
            "IIIIIYXIII", "IIIIIYYIII", "IIIIIYZIII", "IIIIIZXIII", "IIIIIZYIII", "IIIIIZZIII",
            "XIIIIXIIII", "XIIIIYIIII", "XIIIIZIIII", "YIIIIXIIII", "YIIIIYIIII", "YIIIIZIIII",
            "ZIIIIXIIII", "ZIIIIYIIII", "ZIIIIZIIII", "IIIXXIIIII", "IIIXYIIIII", "IIIXZIIIII",
            "IIIYXIIIII", "IIIYYIIIII", "IIIYZIIIII", "IIIZXIIIII", "IIIZYIIIII", "IIIZZIIIII",
            "IIXXIIIIII", "IIXYIIIIII", "IIXZIIIIII", "IIYXIIIIII", "IIYYIIIIII", "IIYZIIIIII",
            "IIZXIIIIII", "IIZYIIIIII", "IIZZIIIIII", "IXXIIIIIII", "IXYIIIIIII", "IXZIIIIIII",
            "IYXIIIIIII", "IYYIIIIIII", "IYZIIIIIII", "IZXIIIIIII", "IZYIIIIIII", "IZZIIIIIII",
            "XXIIIIIIII", "XYIIIIIIII", "XZIIIIIIII", "YXIIIIIIII", "YYIIIIIIII", "YZIIIIIIII",
            "ZXIIIIIIII", "ZYIIIIIIII", "ZZIIIIIIII"
        ]
    )
    # fmt: on
    with subtests.test("odd layer noide model"):
        assert _compare_pauli_lists(noise_model_paulis[refs[1]].to_pauli_list(), odd_paulilist)
