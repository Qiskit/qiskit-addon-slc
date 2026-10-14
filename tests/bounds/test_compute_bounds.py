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

import _thread
import logging
import multiprocessing as mp
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager

import numpy as np
import pytest
from pauli_prop.propagation import RotationGates
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, QubitSparsePauliList
from qiskit_addon_slc import globals as slc_globals
from qiskit_addon_slc.bounds import CommutatorBounds, LightCone, compute_bounds
from qiskit_addon_slc.bounds.commutator_bounds import (
    Bounds,
    _worker_pool,
)
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


def _slow_norm_fn(_pauli: Pauli, _gates: RotationGates) -> CommutatorBounds:
    """A custom norm function which takes long enough for its tasks to queue up."""
    time.sleep(0.5)
    return CommutatorBounds(CUSTOM_BOUND, 0.0, False)


class _InterruptingNoiseModel(dict):
    """A noise model which raises a ``KeyboardInterrupt`` when its second box gets looked up."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_lookups = 0

    def __getitem__(self, key):
        self.num_lookups += 1
        if self.num_lookups > 1:
            raise KeyboardInterrupt
        return super().__getitem__(key)


@contextmanager
def _interrupted_after(seconds: float) -> Generator[None, None, None]:
    """Raises a ``KeyboardInterrupt`` in the main thread after ``seconds``, unless left before."""
    timer = threading.Timer(seconds, _thread.interrupt_main)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def _boxed_trotter_circuit(
    num_qubits: int,
) -> tuple[QuantumCircuit, dict[str, QubitSparsePauliList]]:
    """Builds a small boxed Trotter circuit, together with the Pauli error terms of its boxes."""
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
    return boxed_circuit, generate_noise_model_paulis(find_unique_box_instructions(boxed_circuit))


def _compute_bounds_with(
    norm_fn: Callable[[Pauli, RotationGates], CommutatorBounds],
    *,
    noise_model_type: type[dict] = dict,
    num_processes: int = 1,
) -> Bounds:
    """Computes the bounds of a small Trotter circuit with the provided norm function."""
    num_qubits = 10
    boxed_circuit, noise_model_paulis = _boxed_trotter_circuit(num_qubits)
    noise_model_paulis = noise_model_type(noise_model_paulis)

    circuit = remove_measure(boxed_circuit)
    observable = Pauli("I" * num_qubits).compose("Z", [num_qubits // 2])
    light_cone = LightCone.initialize_from_pauli(circuit, observable)

    return compute_bounds(
        circuit,
        noise_model_paulis,
        light_cone,
        norm_fn,
        backwards=False,
        num_processes=num_processes,
    )


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


def test_interrupt_while_spawning_tasks_terminates_pool():
    """Test that the worker pool gets terminated when task spawning gets interrupted."""
    workers_before = set(mp.active_children())

    with pytest.raises(KeyboardInterrupt):
        _compute_bounds_with(
            _slow_norm_fn, noise_model_type=_InterruptingNoiseModel, num_processes=2
        )

    # Without terminating the pool, its workers would still be running the queued tasks
    assert set(mp.active_children()) <= workers_before


def test_interrupt_while_tracking_progress_keeps_the_partial_bounds(monkeypatch):
    """Test that an interrupt while waiting for the tasks stops them, keeping what got bounded."""
    # NOTE: the interrupt takes effect once the wait for a task returns, which this makes quick
    monkeypatch.setattr(slc_globals, "PROGRESS_POLLING_PERIOD", 0.1)
    workers_before = set(mp.active_children())

    # NOTE: the tasks get spawned within a fraction of a second, but all of them take minutes
    with _interrupted_after(1.0):
        bounds = _compute_bounds_with(_slow_norm_fn, num_processes=2)

    # The interrupt is not raised, and the error terms of the unfinished tasks keep the trivial bound
    rates = np.concatenate([bound.rates for bound in bounds.values()])
    assert np.any(rates == 2.0)
    assert set(mp.active_children()) <= workers_before


def test_worker_pool_terminates_pool_and_ends_worker_spans_on_error():
    """Test that leaving the worker pool through an exception stops it and ends the worker spans."""
    workers_before = set(mp.active_children())
    with (
        pytest.raises(RuntimeError),
        _worker_pool(1) as (_pool, worker_spans),
    ):
        raise RuntimeError

    assert set(mp.active_children()) <= workers_before
    assert worker_spans._ended
