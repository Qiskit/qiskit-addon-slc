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

"""Unequal time commutator bounds."""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import time
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import numpy as np
from pauli_prop.propagation import (
    KNOWN_CLIFFS,
    RotationGates,
)
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, CircuitInstruction
from qiskit.quantum_info import (
    Clifford,
    Pauli,
    PauliLindbladMap,
    QubitSparsePauliList,
    SparsePauliOp,
)

from ..utils import find_indices, iter_circuit
from .light_cone import LightCone

Bounds = dict[str, PauliLindbladMap]

LOGGER = logging.getLogger(__name__)


class CommutatorBounds(NamedTuple):
    """A dataclass to store metadata about the computed commutator bounds."""

    commutator_bound: float
    """The bound on the commutator.

    This bound will be computed in different means depending on the application. For example,
    backward bounds will compute the nuclear norm (Schatten 1-norm) while forward bounds are
    typically computed using the spectral norm (Schatten infinity-norm).

    If the norm computation exceeds specified difficulty limits, it will be abandoned in favor of a
    simpler bound based on the triangle inequality, which is indicated by
    :attr:`fallback_to_tri_ineq` being set to ``True``.
    """

    truncation_bias: float
    """The bias on the bound due to truncation of the commutator."""

    fallback_to_tri_ineq: bool
    """Whether :attr:`commutator_bound` was computed "loosely" using a simple triangle inequality.
    """
    # TODO: document the triangle inequality that is being used here

    def min(self) -> float:
        """Returns the minimum bound encoded by this metadata.

        The minimal bound is the smaller of the sum of :attr:`commutator_bound` and
        :attr:`truncation_bias` or the theoretical bound of ``2.0``.

        The value of ``2.0`` is used because a Pauli observable bounded on the range
        ``[-1, +1]`` cannot be biased by more than ``2.0``.
        """
        return min(self.commutator_bound + self.truncation_bias, 2.0)


def compute_bounds(
    circuit: QuantumCircuit,
    noise_model_paulis: dict[str, QubitSparsePauliList],
    light_cone: LightCone,
    norm_fn: Callable[[Pauli, RotationGates], CommutatorBounds],
    *,
    backwards: bool,
    num_processes: int = 1,
    timeout: float | None = None,
) -> Bounds:
    """Computes the unequal time commutator bounds.

    Given a circuit with :class:`.BoxOp` instructions with :class:`.InjectNoise` annotations and a
    mapping of noise model identifiers (:attr:`.InjectNoise.ref`) to list of Pauli error terms
    (``noise_model_paulis``), this function computes the unequal time commutator bounds (the details
    of which are implemented by ``norm_fn``). In doing so, it only considers gates that lie within
    the light-cone of the observable (initialized by ``light_cone``). These computed bounds form the
    basis of the shaded light-cone.

    Args:
        circuit: the target circuit.
        noise_model_paulis: the Pauli error terms to consider for each noise model.
        light_cone: the initialized and stateful :class:`.LightCone` tracker.
        norm_fn: the function implementing the specific unequal time commutator.
        backwards: whether to iterate over the ``circuit`` in reverse.
        num_processes: the number of parallel processes to use.
        timeout: an optional timeout (in seconds) after which all remaining layers are filled with
            trivial numerical bounds of ``2.0``. Note, that this is not a strict timeout and the
            layer being processed at the time of reaching this timeout will complete normally.

    Returns:
        The computed unequal time commutator bounds.

    Raises:
        ValueError: if ``num_processes`` is lower than 1.
    """
    if num_processes < 1:
        raise ValueError("Cannot use fewer than 1 processes!")

    LOGGER.debug(f"Using {num_processes} processes")

    net_clifford = Clifford.from_label("I" * circuit.num_qubits)
    rot_gates = RotationGates([], [], [])

    def _handle_circuit_instruction(instruction: CircuitInstruction) -> None:
        """Handles a circuit instruction.

        1. If the instruction commutes with the current light-cone, do nothing. Note, that calling
           ``light_cone.commutes`` will append the provided instruction to the stateful
           ``light_cone`` object!
        2. Find ``qargs`` which are the indices of the instruction's qubits in the context of the
           global circuit from which this instruction stems.
        3. If the gate is Clifford, update our global ``net_clifford``, effectively accumulating all
           Clifford gates at the beginning of the circuit.
        4. If the gate is *not* Clifford, append it to our global ``rot_gates`` under which our
           Pauli terms are later going to be evolved. When doing so, we provide ``net_clifford`` to
           ensure the rotation gate gets moved through it, ensuring the ``net_clifford`` remains at
           the beginning of the circuit.
        """
        nonlocal light_cone
        nonlocal circuit
        nonlocal net_clifford
        nonlocal rot_gates

        if light_cone.commutes(instruction):
            return

        qargs = find_indices(circuit, instruction)

        if isinstance(instruction.operation, Barrier):
            LOGGER.debug(f"Ignoring instruction of type '{type(instruction.operation)}'")
        elif instruction.name in KNOWN_CLIFFS:
            net_clifford = net_clifford.dot(instruction.operation, qargs)
        else:
            rot_gates.append_circuit_instruction(
                instruction, qargs, circuit.num_qubits, clifford=net_clifford
            )

    awaitables = []
    gathered_rates = {}
    gathered_noise_terms = {}

    def _insert_rate(bound: CommutatorBounds, box_id: str, rate_idx: int) -> None:
        nonlocal gathered_rates

        gathered_rates[box_id][rate_idx] = bound.min()

    pool = mp.Pool(num_processes)

    timed_out = False
    start = time.time()
    for circ_inst, qargs, box_id, noise_id in iter_circuit(circuit, reverse=True):
        if not timed_out and timeout is not None:
            timed_out = time.time() - start > timeout
            if timed_out:
                LOGGER.warning("Bounds computation timed out.")

        if noise_id is not None:
            noise_terms: QubitSparsePauliList = noise_model_paulis[noise_id]

        if timed_out:
            if box_id is not None:
                # once timeout is reached, fill with trivial bound of 2
                comm_norms[box_id] = PauliLindbladMap.from_components(
                    np.full(len(noise_terms), 2.0),
                    noise_terms,
                )
            continue

        if box_id is None:
            _handle_circuit_instruction(circ_inst)
            continue

        gathered_rates[box_id] = np.full(len(noise_terms), 2.0)

        if backwards:
            # NOTE: we unroll the BoxOp immediately to allow gates contained within the box be
            # pruned by the LightCone pass
            for inst in circ_inst.operation.body[::-1]:
                _handle_circuit_instruction(inst)

        # Ensure that the noise model Pauli terms are defined on the entire width of the circuit.
        local_noise_terms = noise_terms.apply_layout(qargs, num_qubits=circuit.num_qubits)
        # NOTE: both FIXMEs below can be resolved by simply implementing QubitSparsePauliList.evolve
        local_noise_terms = QubitSparsePauliList.from_sparse_list(
            [
                tuple(parts)
                # FIXME: we convert temporarily to SparsePauliOp to leverage its to_sparse_list
                for *parts, _ in SparsePauliOp(
                    # FIXME: we convert temporarily to PauliList to leverage its evolve
                    local_noise_terms.to_pauli_list().evolve(net_clifford, frame="s")
                ).to_sparse_list()
            ],
            circuit.num_qubits,
        )

        norm_fn = partial(  # type: ignore[call-arg]
            norm_fn,
            gates=RotationGates(
                rot_gates.gates[::-1], rot_gates.qargs[::-1], rot_gates.thetas[::-1]
            ),
        )

        gathered_noise_terms[box_id] = noise_terms
        # PERF: using map_async would be more performant than apply_async for each term
        # individually. But that would imply we are still forced to process all layers as a whole
        # and cannot process individual terms.
        for pauli_idx, pauli in enumerate(local_noise_terms):
            async_res = pool.apply_async(
                norm_fn,
                [pauli.to_pauli()],
                callback=partial(_insert_rate, box_id=box_id, rate_idx=pauli_idx),
            )
            awaitables.append(async_res)

        if not backwards:
            # NOTE: we unroll the BoxOp immediately to allow gates contained within the box be
            # pruned by the LightCone pass
            for inst in circ_inst.operation.body[::-1]:
                _handle_circuit_instruction(inst)

    pool.close()
    pool.join()

    comm_norms: Bounds = {
        box_id: PauliLindbladMap.from_components(gathered_rates[box_id], noise_terms)
        for box_id, noise_terms in gathered_noise_terms.items()
    }

    return comm_norms
