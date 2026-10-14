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

"""Tests for OpenTelemetry tracing utilities."""

import dataclasses
import logging
import subprocess
import sys
import time

import pytest
from qiskit_addon_slc import globals as slc_globals
from qiskit_addon_slc.optionals import HAS_OPENTELEMETRY
from qiskit_addon_slc.utils import _tracing as tracing
from qiskit_addon_slc.utils._tracing import (
    _WORKER_INFO_UNSET,
    NoOpSpan,
    StepTimer,
    WorkerSpans,
    _create_span_exporters,
    _get_tracer,
    _get_tracing_settings,
    _initialize_tracing,
    _is_own_span,
    _is_tracing_enabled,
    _TaskRecord,
    get_task_index,
    mark_moment,
    record_outcome,
    run_worker_task,
    traced_span,
)


@pytest.mark.parametrize(
    ("value", "enabled"),
    [(None, False), ("true", True), ("On", True), ("1", True), ("yes", True), ("false", False)],
)
def test_tracing_enabled_setting_is_read_from_the_environment(monkeypatch, value, enabled):
    """Test which values of QISKIT_SLC_TRACING_ENABLED enable tracing.

    The setting is read when qiskit_addon_slc.globals gets imported, which happens only once per
    process, so each value is read by a process of its own.
    """
    if value is None:
        monkeypatch.delenv("QISKIT_SLC_TRACING_ENABLED", raising=False)
    else:
        monkeypatch.setenv("QISKIT_SLC_TRACING_ENABLED", value)
    script = "from qiskit_addon_slc import globals; print(globals.TRACING_ENABLED)"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=60
    )
    assert result.stdout.strip() == str(enabled)


@pytest.mark.parametrize(("value", "limit"), [(None, 1024), ("16", 16), ("many", 1024)])
def test_span_event_count_limit_is_read_from_the_environment(monkeypatch, value, limit):
    """Test that OTEL_SPAN_EVENT_COUNT_LIMIT sets the limit, unless it is not an integer."""
    if value is None:
        monkeypatch.delenv("OTEL_SPAN_EVENT_COUNT_LIMIT", raising=False)
    else:
        monkeypatch.setenv("OTEL_SPAN_EVENT_COUNT_LIMIT", value)
    script = "from qiskit_addon_slc import globals; print(globals.OTEL_SPAN_EVENT_COUNT_LIMIT)"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=60
    )
    assert result.stdout.strip() == str(limit)
    assert ("not an integer" in result.stderr) == (value == "many")


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_is_tracing_enabled_follows_the_setting(monkeypatch):
    """Test that tracing can be enabled and disabled at runtime."""
    assert not _is_tracing_enabled()
    monkeypatch.setattr(slc_globals, "TRACING_ENABLED", True)
    assert _is_tracing_enabled()


def test_is_tracing_enabled_requires_opentelemetry(monkeypatch):
    """Test that tracing stays disabled without OpenTelemetry, even when the setting enables it."""
    monkeypatch.setattr(slc_globals, "TRACING_ENABLED", True)
    monkeypatch.setattr(tracing, "HAS_OPENTELEMETRY", False)
    assert not _is_tracing_enabled()


def test_noop_span_operations():
    """Test that NoOpSpan accepts all operations without errors."""
    span = NoOpSpan()

    # Test all methods - should not raise any exceptions
    span.add_event("test_event")
    span.add_event("test_event_with_attrs", {"key": "value"})
    span.set_attribute("test_key", "test_value")
    span.end()
    assert not span.is_recording()


def test_noop_span_context_manager():
    """Test that NoOpSpan is its own context manager, which lets exceptions propagate."""
    span = NoOpSpan()

    with span as s:
        assert s is span

    with pytest.raises(ValueError), span:
        raise ValueError("test")


def test_traced_span_when_disabled():
    """Test that traced_span hands out the shared NoOpSpan when tracing is disabled."""
    with (
        traced_span("first") as first,
        traced_span("second", attributes={"key": "value"}) as second,
    ):
        assert isinstance(first, NoOpSpan)
        assert second is first


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.usefixtures("tracing_enabled")
def test_traced_span_nests_spans():
    """Test that a span is the parent of the spans created within it, unless another is given."""
    with (
        traced_span("outer", attributes={"key": "value", "number": 42}) as outer,
        traced_span("inner") as inner,
        traced_span("sibling", parent_span=outer) as sibling,
    ):
        pass

    assert outer.parent is None
    assert inner.parent.span_id == outer.get_span_context().span_id
    assert sibling.parent.span_id == outer.get_span_context().span_id
    assert dict(outer.attributes) == {"key": "value", "number": 42}
    assert all(span.end_time is not None for span in (outer, inner, sibling))


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.usefixtures("tracing_enabled")
def test_noop_span_parent_nests_spans_under_the_current_span():
    """Test that a NoOpSpan, obtained before tracing got enabled, acts like no span at all."""
    with traced_span("outer") as outer:
        with traced_span("child", parent_span=tracing._NOOP_SPAN) as child:
            pass
        carrier = tracing._inject_trace_context(tracing._NOOP_SPAN)

    outer_context = outer.get_span_context()
    assert child.parent.span_id == outer_context.span_id
    assert f"{outer_context.span_id:016x}" in carrier["traceparent"]


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.usefixtures("tracing_enabled")
def test_traced_span_records_exceptions():
    """Test that an exception propagates and marks its span as failed, but not as interrupted."""
    from opentelemetry.trace import StatusCode

    with pytest.raises(ValueError), traced_span("failing") as span:
        raise ValueError("test")

    assert span.status.status_code == StatusCode.ERROR
    assert "interrupted" not in span.attributes


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.usefixtures("tracing_enabled")
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_traced_span_marks_interrupted_spans(exception):
    """Test that a span cut short by the termination of its process is marked as interrupted."""
    from opentelemetry.trace import StatusCode

    with pytest.raises(exception), traced_span("interrupted") as span:
        raise exception

    assert span.attributes["interrupted"] is True
    assert [event.name for event in span.events] == ["exception"]
    assert span.events[0].attributes["exception.type"] == exception.__name__
    assert span.status.status_code == StatusCode.UNSET


def test_run_worker_task_records_the_task():
    """Test that a task reports its index while it runs, and which worker completed it when."""
    tracing._WORKER_STORAGE.worker_index = 1

    result, record = run_worker_task(4, lambda value: (value, get_task_index()), 7)

    assert result == (7, 4)
    assert record.worker_index == 1
    assert record.end_time > 0


def test_run_worker_task_ends_the_task_before_exporting_its_spans(monkeypatch):
    """Test that the time it takes to export the spans of a task does not count towards the task."""
    monkeypatch.setattr(tracing, "_export_finished_spans", lambda: time.sleep(0.2))

    _, record = run_worker_task(0, lambda: None)

    assert time.time_ns() - record.end_time >= 0.2e9


def _fail():
    raise ValueError("failed task")


def test_run_worker_task_records_a_failed_task():
    """Test that the exception of a failed task carries the record of that task."""
    tracing._WORKER_STORAGE.worker_index = 1

    with pytest.raises(ValueError, match="failed task") as info:
        run_worker_task(4, _fail)

    record = getattr(info.value, tracing._TASK_RECORD_ATTRIBUTE)
    assert record.worker_index == 1
    assert record.end_time > 0


@dataclasses.dataclass(frozen=True)
class _FrozenError(Exception):
    reason: str


def _fail_frozen():
    raise _FrozenError("frozen")


def test_run_worker_task_raises_an_exception_which_cannot_carry_the_record():
    """Test that an exception which does not allow setting the record still gets raised as is."""
    with pytest.raises(_FrozenError) as info:
        run_worker_task(0, _fail_frozen)

    assert not hasattr(info.value, tracing._TASK_RECORD_ATTRIBUTE)


def _exporter_settings(exporter: str, endpoint: str | None) -> dict[str, str | None]:
    """The provider settings which select the span exporters."""
    return {"OTEL_TRACES_EXPORTER": exporter, "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint}


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_create_span_exporter_otlp():
    """Test that the OTLP exporter is used when it is selected and configured."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    (exporter,) = _create_span_exporters(_exporter_settings("otlp", "http://localhost:4318"))
    assert isinstance(exporter, OTLPSpanExporter)
    assert exporter._endpoint == "http://localhost:4318/v1/traces"


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_create_span_exporter_otlp_endpoint_with_trailing_slash():
    """Test that a trailing slash of the OTLP endpoint does not end up doubled in its path."""
    (exporter,) = _create_span_exporters(_exporter_settings("otlp", "http://localhost:4318/"))
    assert exporter._endpoint == "http://localhost:4318/v1/traces"


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.parametrize("endpoint", [None, "http://localhost:4318"])
def test_create_span_exporter_otlp_traces_endpoint(monkeypatch, endpoint):
    """Test that the traces endpoint of the OpenTelemetry specification takes precedence."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://collector:4318/custom/traces")
    (exporter,) = _create_span_exporters(_exporter_settings("otlp", endpoint))
    assert type(exporter).__name__ == "OTLPSpanExporter"
    assert exporter._endpoint == "http://collector:4318/custom/traces"


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ("console", ["ConsoleSpanExporter"]),
        ("", ["ConsoleSpanExporter"]),
        ("none", []),
        ("otlp, Console", ["OTLPSpanExporter", "ConsoleSpanExporter"]),
        ("otlp,otlp", ["OTLPSpanExporter"]),
        ("zipkin", ["ConsoleSpanExporter"]),
        ("zipkin,console", ["ConsoleSpanExporter"]),
    ],
)
def test_create_span_exporters_from_list(setting, expected):
    """Test that the exporter setting is read as a comma-separated list of exporters."""
    exporters = _create_span_exporters(_exporter_settings(setting, "http://localhost:4318"))
    assert [type(exporter).__name__ for exporter in exporters] == expected


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.parametrize(
    ("endpoint", "otlp_installed"),
    [(None, True), ("http://localhost:4318", False)],
)
def test_create_span_exporter_falls_back_to_console(monkeypatch, endpoint, otlp_installed):
    """Test that the console exporter is used when the OTLP one is not configured or installed."""
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter

    if not otlp_installed:
        # NOTE: this makes importing the module raise an ImportError
        monkeypatch.setitem(
            sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None
        )
    (exporter,) = _create_span_exporters(_exporter_settings("otlp", endpoint))
    assert isinstance(exporter, ConsoleSpanExporter)


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_initialize_tracing_creates_its_own_tracer_provider():
    """Test that a tracer provider gets created and set up globally if the application has none."""
    from opentelemetry import trace

    _initialize_tracing()

    assert tracing._owns_tracer_provider()
    assert trace.get_tracer_provider() is tracing._tracer_provider
    assert tracing._provider_settings["OTEL_TRACES_EXPORTER"] == "none"


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_initialize_tracing_uses_tracer_provider_of_application():
    """Test that a tracer provider which the application set up already gets used as is."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    app_provider = TracerProvider()
    trace.set_tracer_provider(app_provider)

    _initialize_tracing()

    assert tracing._tracer_provider is app_provider
    assert not tracing._owns_tracer_provider()
    assert trace.get_tracer_provider() is app_provider


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.usefixtures("tracing_enabled")
def test_is_own_span():
    """Test that only the spans of this package's tracer count as its own."""
    from opentelemetry import trace
    from opentelemetry.trace import INVALID_SPAN

    with traced_span("own") as own:
        pass
    (worker,) = WorkerSpans(1).spans
    app = trace.get_tracer("my_application").start_span("app")

    assert _is_own_span(own)
    assert _is_own_span(worker)
    assert not _is_own_span(app)
    assert not _is_own_span(INVALID_SPAN)
    assert not _is_own_span(NoOpSpan())


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_initialize_tracing_failure_does_not_raise(monkeypatch, caplog):
    """Test that a failure to set up tracing gets logged, and is not retried."""
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace.export", None)

    with caplog.at_level(logging.ERROR, logger="qiskit_addon_slc.utils._tracing"):
        _initialize_tracing()
        _initialize_tracing()

    (error,) = caplog.records
    assert "Failed to initialize tracing" in error.getMessage()
    assert tracing._setup_attempted
    assert tracing._tracer_provider is None


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.usefixtures("tracing_enabled")
def test_get_tracer_reuses_its_tracer():
    """Test that the tracer gets created only once, since every traced span needs it."""
    assert _get_tracer() is _get_tracer()


def test_get_tracing_settings_of_own_tracer_provider(monkeypatch, caplog):
    """Test that the workers get the exporter settings which the main process actually uses."""
    applied = {name: getattr(slc_globals, name) for name in tracing._PROVIDER_SETTINGS}
    monkeypatch.setattr(tracing, "_provider_settings", {**applied, "OTEL_TRACES_EXPORTER": "none"})
    monkeypatch.setattr(slc_globals, "OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setattr(slc_globals, "TRACING_ENABLED", True)

    with caplog.at_level(logging.WARNING, logger="qiskit_addon_slc.utils._tracing"):
        settings = _get_tracing_settings()

    # The exporter was set up already, so changing it has no effect, unlike enabling tracing
    assert settings["OTEL_TRACES_EXPORTER"] == "none"
    assert settings["TRACING_ENABLED"] is True
    (warning,) = caplog.records
    assert "OTEL_TRACES_EXPORTER" in warning.getMessage()


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.usefixtures("tracing_enabled")
def test_worker_spans_end_with_the_last_task_of_their_worker():
    """Test that the span of a worker records its tasks and ends when its last task did."""
    worker_spans = WorkerSpans(3)
    results = []
    callback = worker_spans.recording(results.append)

    callback(("a", _TaskRecord(0, 20)))
    callback(("b", _TaskRecord(0, 10)))
    callback(("c", _TaskRecord(1, 30)))
    # A worker without a span, e.g. one which replaced a worker that died, is not recorded
    callback(("d", _TaskRecord(3, 40)))
    callback(("e", _TaskRecord(_WORKER_INFO_UNSET, 40)))
    worker_spans.end(terminated=False)

    # Every result is passed on, whether or not its task got recorded
    assert results == ["a", "b", "c", "d", "e"]

    spans = worker_spans.spans
    assert [span.attributes["tasks.completed"] for span in spans] == [2, 1, 0]
    assert not any(span.attributes["worker.terminated"] for span in spans)
    # A worker which never got a task has a span of zero length
    assert [span.end_time for span in spans[:2]] == [20, 30]
    assert spans[2].end_time == spans[2].start_time

    # The spans are ended only once
    worker_spans.end(terminated=True)
    assert spans[0].end_time == 20
    assert not spans[0].attributes["worker.terminated"]


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.usefixtures("tracing_enabled")
def test_worker_spans_of_a_terminated_pool_end_with_the_pool():
    """Test that the workers of a terminated pool, which may have still been busy, end with it."""
    worker_spans = WorkerSpans(2)

    worker_spans.recording(lambda _: None)((None, _TaskRecord(0, 20)))
    worker_spans.end(terminated=True)

    assert all(span.attributes["worker.terminated"] for span in worker_spans.spans)
    # This includes the worker which never finished a task, since it may have been stuck on one
    assert all(span.end_time > 20 for span in worker_spans.spans)


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.usefixtures("tracing_enabled")
def test_worker_spans_record_failed_tasks():
    """Test that a failed task counts towards its worker, and that its exception is passed on."""
    worker_spans = WorkerSpans(2)
    errors = []
    on_error = worker_spans.recording_failure(errors.append)

    tracing._WORKER_STORAGE.worker_index = 0
    with pytest.raises(ValueError) as failed:
        run_worker_task(0, _fail)
    on_error(failed.value)
    worker_spans.recording(lambda _: None)(run_worker_task(1, lambda: None))
    # A task which could not even be sent to a worker has no record
    unsent = ValueError("not sent")
    on_error(unsent)
    worker_spans.end(terminated=False)

    assert errors == [failed.value, unsent]
    first, second = worker_spans.spans
    assert first.attributes["tasks.completed"] == 2
    assert first.attributes["tasks.failed"] == 1
    assert second.attributes["tasks.completed"] == 0
    assert second.attributes["tasks.failed"] == 0


def test_worker_spans_when_disabled():
    """Test that the workers get spans which record nothing when tracing is disabled."""
    worker_spans = WorkerSpans(2)
    results = []

    worker_spans.recording(results.append)(("a", _TaskRecord(0, 20)))
    worker_spans.end(terminated=False)

    assert results == ["a"]
    assert len(worker_spans.spans) == 2
    assert not any(span.is_recording() for span in worker_spans.spans)


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_step_timer_records_the_steps_on_its_span(recording_span):
    """Test that the steps are counted and timed on a span which gets recorded."""
    steps = StepTimer(recording_span, ("step", "other"))

    assert steps.run("step", max, 1, 2) == 2
    assert steps.run("step", min, 1, 2) == 1
    # A step which was not named upfront gets recorded once it ran
    assert steps.run("unnamed", abs, -3) == 3
    steps.record()

    assert recording_span.attributes["step.calls"] == 2
    assert recording_span.attributes["step.seconds"] > 0.0
    assert recording_span.attributes["other.calls"] == 0
    assert recording_span.attributes["other.seconds"] == 0.0
    assert recording_span.attributes["unnamed.calls"] == 1


def test_step_timer_runs_the_steps_when_not_recording():
    """Test that the steps still run when their span records nothing."""
    steps = StepTimer(NoOpSpan(), ("step",))

    assert steps.run("step", max, 1, 2) == 2
    assert steps.run("unnamed", abs, -3) == 3
    steps.record()


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_mark_moment_records_a_child_of_its_parent(finished_spans):
    """Test that a moment is a span of the given parent rather than of the current span."""
    with traced_span("parent") as parent, traced_span("current"):
        mark_moment("moment", parent, {"key": "value"})

    (moment,) = (span for span in finished_spans.get_finished_spans() if span.name == "moment")
    assert moment.parent.span_id == parent.get_span_context().span_id
    assert dict(moment.attributes) == {"key": "value"}


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@pytest.mark.usefixtures("tracing_enabled")
def test_record_outcome_records_on_its_own_parent_only():
    """Test that the outcome goes onto the parent only if this package created that parent."""
    from opentelemetry import trace

    outcome = {"stop_reason": "timeout", "tasks.completed": 3}
    with traced_span("own") as own, traced_span("computation") as span:
        record_outcome(span, own, outcome)
    app = trace.get_tracer("my_application").start_span("app")
    with traced_span("computation") as other_span:
        record_outcome(other_span, app, outcome)
        record_outcome(other_span, NoOpSpan(), outcome)

    assert dict(span.attributes) == outcome
    assert dict(own.attributes) == outcome
    assert dict(other_span.attributes) == outcome
    assert "stop_reason" not in app.attributes
