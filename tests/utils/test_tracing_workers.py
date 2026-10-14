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

"""Tests for the tracing of the worker processes of a bound computation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest
from qiskit_addon_slc.optionals import HAS_OPENTELEMETRY

NUM_PROCESSES = 2

# A bound computation in which all but the first tasks run for much longer than the timeout, such
# that the worker pool gets terminated while they are still running, after each worker completed one
# task. It runs in a separate process, because tracing is configured once per process, and it uses
# the spawn start method, because its workers re-import this script and must still be traced,
# although tracing is only enabled at runtime.
#
# The long tasks either sleep, or run a single long call into native code, during which Python
# cannot handle any signal, as is the case for the Pauli propagation or an eigensolver. The script
# takes the kind of long task as its first argument, and optionally the start method of the workers
# as its second one. With "fail" as its first argument instead, every task with an odd index raises
# an exception, and the computation runs without a timeout.
SCRIPT = f"""
import multiprocessing as mp
import sys
import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import find_unique_box_instructions

from qiskit_addon_slc import globals as slc_globals
from qiskit_addon_slc.bounds import CommutatorBounds, LightCone, compute_bounds
from qiskit_addon_slc.utils import generate_noise_model_paulis, remove_measure
from qiskit_addon_slc.utils._tracing import get_task_index, traced_span


def norm_fn(pauli, gates):
    with traced_span("fake_norm_computation"):
        if sys.argv[1] == "fail":
            if get_task_index() % 2:
                raise ValueError("odd task")
        # Only the first task of each worker completes, the others are still running at the timeout
        elif get_task_index() < {NUM_PROCESSES}:
            time.sleep(0.01)
        elif sys.argv[1] == "sleep":
            time.sleep(60)
        else:
            # NOTE: summing a range runs entirely in C, for well over a minute
            sum(range(10**10))
    return CommutatorBounds(0.5, 0.0, False)


if __name__ == "__main__":
    mp.set_start_method(sys.argv[2] if len(sys.argv) > 2 else "spawn")
    slc_globals.TRACING_ENABLED = True
    slc_globals.PROGRESS_POLLING_PERIOD = 0.2

    circuit = QuantumCircuit(4)
    for _ in range(2):
        circuit.rx(0.1, range(4))
        circuit.barrier()
        for i in range(3):
            circuit.rzz(0.2, i, i + 1)
        circuit.barrier()
    boxed_circuit = generate_boxing_pass_manager(
        enable_gates=True, inject_noise_targets="all", remove_barriers=False
    ).run(circuit)
    noise_model_paulis = generate_noise_model_paulis(find_unique_box_instructions(boxed_circuit))
    circuit = remove_measure(boxed_circuit)
    light_cone = LightCone.initialize_from_pauli(circuit, Pauli("IZII"))
    with traced_span("root"):
        compute_bounds(
            circuit,
            noise_model_paulis,
            light_cone,
            norm_fn,
            backwards=False,
            num_processes={NUM_PROCESSES},
            timeout=None if sys.argv[1] == "fail" else 1,
        )
"""

# A bound computation run by an application which set up its own tracer provider and calls
# compute_bounds within a span of its own. The script takes the start method of the workers as its
# only argument. Forked workers inherit that provider, while spawned ones do not, since the
# application sets it up only when run as a script. The exporter settings of qiskit_addon_slc
# select no exporter at all, such that no span gets printed unless it went through the provider of
# the application.
APP_SCRIPT = f"""
import multiprocessing as mp
import sys

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import find_unique_box_instructions

from qiskit_addon_slc import globals as slc_globals
from qiskit_addon_slc.bounds import CommutatorBounds, LightCone, compute_bounds
from qiskit_addon_slc.utils import generate_noise_model_paulis, remove_measure
from qiskit_addon_slc.utils._tracing import traced_span


def norm_fn(pauli, gates):
    with traced_span("fake_norm_computation"):
        return CommutatorBounds(0.5, 0.0, False)


if __name__ == "__main__":
    mp.set_start_method(sys.argv[1])
    provider = TracerProvider(resource=Resource(attributes={{SERVICE_NAME: "my-app"}}))
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    slc_globals.TRACING_ENABLED = True
    slc_globals.OTEL_TRACES_EXPORTER = "none"

    circuit = QuantumCircuit(4)
    circuit.rx(0.1, range(4))
    circuit.barrier()
    for i in range(3):
        circuit.rzz(0.2, i, i + 1)
    boxed_circuit = generate_boxing_pass_manager(
        enable_gates=True, inject_noise_targets="all", remove_barriers=False
    ).run(circuit)
    noise_model_paulis = generate_noise_model_paulis(find_unique_box_instructions(boxed_circuit))
    circuit = remove_measure(boxed_circuit)
    light_cone = LightCone.initialize_from_pauli(circuit, Pauli("IZII"))
    with trace.get_tracer("my_app").start_as_current_span("app") as span:
        span.set_attribute("timeout.seconds", "of the application")
        compute_bounds(
            circuit,
            noise_model_paulis,
            light_cone,
            norm_fn,
            backwards=False,
            num_processes={NUM_PROCESSES},
        )
    provider.shutdown()
"""


def _parse_console_spans(output: str) -> list[dict]:
    """Parses the spans which the console exporter printed as concatenated JSON objects."""
    decoder = json.JSONDecoder()
    spans, idx = [], 0
    while (start := output.find("{\n", idx)) >= 0:
        try:
            obj, idx = decoder.raw_decode(output, start)
        except ValueError:
            idx = start + 1
            continue
        if isinstance(obj, dict) and "context" in obj:
            spans.append(obj)
    return spans


def _run_script(
    tmp_path, *args: str, script_text: str = SCRIPT
) -> tuple[dict[str, list[dict]], list[dict], float, str]:
    """Runs a script with the given arguments, returning its spans, run time and standard error."""
    script = tmp_path / "traced_timeout.py"
    script.write_text(script_text)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OTEL_TRACES_EXPORTER": "console"}
    env.pop("QISKIT_SLC_TRACING_ENABLED", None)

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    elapsed = time.time() - start
    assert result.returncode == 0, result.stderr
    assert "Exception ignored" not in result.stderr

    spans = _parse_console_spans(result.stdout)
    by_name: dict[str, list[dict]] = {}
    for span in spans:
        by_name.setdefault(span["name"], []).append(span)
    return by_name, spans, elapsed, result.stderr


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_spans_are_exported_when_workers_get_terminated(tmp_path):
    """Test that a computation stopped by its timeout exports complete traces of its workers."""
    by_name, spans, elapsed, _ = _run_script(tmp_path, "sleep")

    # The workers got terminated rather than finishing their tasks
    assert elapsed < 30

    # The computation recorded that and how it was stopped
    (bounds_span,) = by_name["compute_bounds"]
    assert bounds_span["attributes"]["stop_reason"] == "timeout"
    assert bounds_span["attributes"]["tasks.completed"] == NUM_PROCESSES
    assert by_name["root"][0]["attributes"]["stop_reason"] == "timeout"
    (marker,) = by_name["timeout_reached"]
    assert marker["parent_id"] == bounds_span["context"]["span_id"]

    # The steps run per instruction are summarized on the task_spawning span instead of traced
    (spawning,) = by_name["task_spawning"]
    assert spawning["attributes"]["light_cone.commutes.calls"] > 0
    assert spawning["attributes"]["light_cone.commutes.seconds"] > 0
    assert "light_cone.commutes" not in by_name

    # Every worker has a span, numbered from zero and marked as terminated, which counts its tasks
    workers = sorted(name for name in by_name if name.startswith("worker_"))
    assert workers == [f"worker_{idx}" for idx in range(NUM_PROCESSES)]
    worker_spans = [by_name[name][0] for name in workers]
    assert all(span["attributes"]["worker.terminated"] for span in worker_spans)
    assert all(span["parent_id"] == bounds_span["context"]["span_id"] for span in worker_spans)
    assert sum(span["attributes"]["tasks.completed"] for span in worker_spans) == NUM_PROCESSES

    # The completed tasks were exported, each nested under the span of the worker which ran it,
    # while the tasks which were still running are lost
    tasks = by_name["fake_norm_computation"]
    assert len(tasks) == NUM_PROCESSES
    worker_ids = {span["context"]["span_id"] for span in worker_spans}
    assert all(s["parent_id"] in worker_ids for s in tasks)

    # No span lost its parent
    span_ids = {span["context"]["span_id"] for span in spans}
    orphans = [s["name"] for s in spans if s.get("parent_id") and s["parent_id"] not in span_ids]
    assert not orphans


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_failed_tasks_are_recorded(tmp_path):
    """Test that the failed tasks are counted by their workers and by the computation."""
    by_name, _, _, _ = _run_script(tmp_path, "fail")

    (bounds_span,) = by_name["compute_bounds"]
    attributes = bounds_span["attributes"]
    assert attributes["stop_reason"] == "completed"
    assert attributes["tasks.completed"] == attributes["tasks.total"]
    assert attributes["tasks.failed"] == attributes["tasks.total"] // 2

    workers = [by_name[f"worker_{idx}"][0] for idx in range(NUM_PROCESSES)]
    assert not any(span["attributes"]["worker.terminated"] for span in workers)
    assert (
        sum(span["attributes"]["tasks.completed"] for span in workers) == attributes["tasks.total"]
    )
    assert sum(span["attributes"]["tasks.failed"] for span in workers) == attributes["tasks.failed"]

    # The spans of the failed tasks carry their exception
    failed = [s for s in by_name["fake_norm_computation"] if s["status"]["status_code"] == "ERROR"]
    assert len(failed) == attributes["tasks.failed"]


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.skipif(sys.platform == "win32", reason="the forkserver start method is not available")
@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_workers_stuck_in_native_code_get_terminated(tmp_path, start_method):
    """Test that the timeout still bounds the run time when workers are stuck in native code."""
    by_name, _, elapsed, _ = _run_script(tmp_path, "native", start_method)

    # The workers got terminated instead of finishing their tasks, which run for well over a minute
    assert elapsed < 30
    (bounds_span,) = by_name["compute_bounds"]
    assert bounds_span["attributes"]["stop_reason"] == "timeout"


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.skipif(sys.platform == "win32", reason="the fork start method is not available")
def test_tracer_provider_of_application_is_used(tmp_path):
    """Test that the spans go to the application's tracer provider, without touching its span."""
    by_name, spans, _, stderr = _run_script(tmp_path, "fork", script_text=APP_SCRIPT)

    # Every span went through the provider of the application, including those of the tasks, which
    # the workers exported themselves
    assert {span["resource"]["attributes"]["service.name"] for span in spans} == {"my-app"}
    workers = sorted(name for name in by_name if name.startswith("worker_"))
    assert workers == [f"worker_{idx}" for idx in range(NUM_PROCESSES)]
    assert by_name["fake_norm_computation"]

    # How the computation ended is recorded on the span of compute_bounds, but not on the span of
    # the application
    (bounds_span,) = by_name["compute_bounds"]
    assert bounds_span["attributes"]["stop_reason"] == "completed"
    (app_span,) = by_name["app"]
    assert app_span["attributes"] == {"timeout.seconds": "of the application"}
    assert "cannot use the tracer provider of the application" not in stderr


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_spawned_workers_warn_about_tracer_provider_of_application(tmp_path):
    """Test that spawned workers, which cannot use the application's provider, say so."""
    by_name, spans, _, stderr = _run_script(tmp_path, "spawn", script_text=APP_SCRIPT)

    # The spans of the main process, including those of the workers, went through the provider of
    # the application, while the workers fell back to the exporter settings of qiskit_addon_slc,
    # which select none
    assert {span["resource"]["attributes"]["service.name"] for span in spans} == {"my-app"}
    assert "worker_0" in by_name
    assert "fake_norm_computation" not in by_name
    assert stderr.count("cannot use the tracer provider of the application") == 1
