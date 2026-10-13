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

# The development of this module was assisted by Bob 1.0.1 and Claude Opus 5

"""OpenTelemetry tracing utilities, including for computations run by a pool of worker processes.

This module provides:

- the setup of the tracer provider and its span exporters, as configured by the tracing settings of
  :mod:`qiskit_addon_slc.globals`,
- :func:`traced_span`, a span context manager which costs next to nothing when tracing is disabled,
- the tracing of a pool of worker processes: the spans which the main process records for its
  workers, the setup of each worker, and the trace context passed between the processes, and
- helpers for tracing a computation: counting and timing frequent steps, marking moments, and
  recording how the computation ended.

This module implements the tracing of ``qiskit-addon-slc``, which users control through the tracing
settings of :mod:`qiskit_addon_slc.globals`. It is not public API itself, which is why it has no API
documentation: its interface may change or go away without deprecation. Its names have no leading
underscore where other modules of this package use them.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
import time
import types
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
from contextlib import AbstractContextManager, contextmanager, suppress
from multiprocessing import util as mp_util
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from .. import globals as slc_globals
from ..optionals import HAS_OPENTELEMETRY

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

    from opentelemetry.sdk.trace.export import SpanExporter
    from opentelemetry.trace import Span, Tracer, TracerProvider

LOGGER = logging.getLogger(__name__)

# The tracer provider which records the spans of this process, once tracing is set up
_tracer_provider: TracerProvider | None = None
# Whether setting up tracing was attempted. If that failed, _tracer_provider stays None.
_setup_attempted: bool = False
_INIT_LOCK = threading.Lock()
# The name of the tracer of this package, which marks the spans it creates as its own
_TRACER_NAME = "qiskit_addon_slc"
# The tracer handed out by _get_tracer, once it was created
_tracer: Tracer | None = None
# The settings with which this module created _tracer_provider, or None if it did not create it
_provider_settings: dict[str, Any] | None = None

# The tracing settings which configure the tracer provider, and so take effect only once, when this
# module creates that provider.
_PROVIDER_SETTINGS = (
    "OTEL_SERVICE_NAME",
    "OTEL_TRACES_EXPORTER",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_SPAN_EVENT_COUNT_LIMIT",
)

# The settings of qiskit_addon_slc.globals that configure tracing, which must list all of them. A
# worker process started with the spawn or forkserver start method re-imports that module, which
# resets these settings to the values given by the environment variables. Hence, any value that was
# assigned at runtime in the main process must be forwarded to the workers explicitly.
_TRACING_SETTINGS = ("TRACING_ENABLED", *_PROVIDER_SETTINGS)

# The metadata of this worker process: its worker_index, and the task_idx of its current task
_WORKER_STORAGE = types.SimpleNamespace()
# The value of the worker metadata which is not set, e.g. outside of a worker
_WORKER_INFO_UNSET = -1
# The attribute of the exception of a failed task which holds the record of that task
_TASK_RECORD_ATTRIBUTE = "_qiskit_addon_slc_task_record"

_T = TypeVar("_T")


class NoOpSpan:
    """A span which records nothing, for when tracing is disabled.

    This provides the parts of the interface of OpenTelemetry spans which this package uses, so that
    the instrumented code works the same whether or not tracing is enabled, and without
    OpenTelemetry installed. It holds no state, so a single instance (``_NOOP_SPAN``) gets shared.
    """

    def is_recording(self) -> bool:
        """Returns ``False``, such that the computation of attribute values can be skipped."""
        return False

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Does nothing."""

    def end(self, end_time: int | None = None) -> None:
        """Does nothing."""

    def __enter__(self) -> NoOpSpan:
        """Returns this span, such that it is its own context manager (see :func:`traced_span`)."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Does nothing, such that exceptions propagate."""


_NOOP_SPAN = NoOpSpan()


def _is_tracing_enabled() -> bool:
    """Check if tracing is enabled.

    Returns:
        Whether :const:`~qiskit_addon_slc.globals.TRACING_ENABLED` is set and OpenTelemetry is
        installed.

    Note:
        The setting can be changed at runtime. Its default is read from the
        ``QISKIT_SLC_TRACING_ENABLED`` environment variable when :mod:`qiskit_addon_slc.globals`
        gets imported.
    """
    # NOTE: the setting is checked first, because checking HAS_OPENTELEMETRY imports the SDK
    return bool(slc_globals.TRACING_ENABLED) and bool(HAS_OPENTELEMETRY)


def _owns_tracer_provider() -> bool:
    """Check whether this module created the tracer provider, rather than the application.

    Returns:
        Whether :func:`_initialize_tracing` created the tracer provider of this process.
    """
    return _provider_settings is not None


def _get_tracing_settings() -> dict[str, Any]:
    """Get the tracing settings of the current process.

    :class:`WorkerSpans` puts these into the :class:`WorkerSetup` of its workers, such that the
    workers are traced with the same settings as this process.

    The settings which configure the tracer provider take effect only when that provider gets
    created. If this module created it, the values it was created with are returned for these
    settings, such that the spans of the workers go to the same exporters as those of this
    process. A warning is logged if any of them has been changed since.

    Returns:
        The values of the tracing settings in :mod:`qiskit_addon_slc.globals` which apply to this
        process.
    """
    settings = {name: getattr(slc_globals, name) for name in _TRACING_SETTINGS}
    if _provider_settings is not None:
        changed = [name for name, value in _provider_settings.items() if settings[name] != value]
        if changed:
            LOGGER.warning(
                "The tracing settings %s were changed after tracing was set up, which has no "
                "effect until the Python process is restarted.",
                ", ".join(changed),
            )
        settings.update(_provider_settings)
    return settings


def _uses_application_tracer_provider() -> bool:
    """Check whether the spans of this process go to a tracer provider of the application.

    :class:`WorkerSpans` puts the result into the :class:`WorkerSetup` of its workers. It must be
    called once tracing is set up, since there is no tracer provider to check before.

    Returns:
        Whether tracing is enabled and uses a tracer provider which the application set up, rather
        than one which this module created.
    """
    return _is_tracing_enabled() and _tracer_provider is not None and not _owns_tracer_provider()


def _initialize_tracing() -> None:
    """Initialize the OpenTelemetry tracer provider.

    This sets up the tracer provider, with the exporters selected by the tracing settings of
    :mod:`qiskit_addon_slc.globals`. It does nothing when called again. A failure gets logged rather
    than raised. Tracing then stays set up without a tracer provider, so its spans are not recorded.

    It must only be called once tracing is enabled (see :func:`_is_tracing_enabled`), which ensures
    that OpenTelemetry is installed. It does not check this itself: its only caller,
    :func:`_get_tracer`, has the same precondition, which every caller of that checks.

    OpenTelemetry allows the global tracer provider to be set only once. If the application has set
    one up already, that one is used together with its own exporters and limits, and the tracing
    settings of :mod:`qiskit_addon_slc.globals` that configure the exporter have no effect.
    """
    global _tracer_provider, _setup_attempted, _provider_settings

    with _INIT_LOCK:
        if _setup_attempted:
            return

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import SERVICE_NAME, Resource
            from opentelemetry.sdk.trace import SpanLimits, TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            current_provider = trace.get_tracer_provider()
            if not isinstance(current_provider, trace.ProxyTracerProvider):
                # NOTE: the spans get recorded by this provider, so it is the one which the workers
                # must flush before they exit
                _tracer_provider = current_provider
                LOGGER.info("OpenTelemetry tracing uses the tracer provider of the application")
            else:
                # NOTE: the settings are read only once, such that the provider is built with
                # exactly the values recorded in _provider_settings
                settings = {name: getattr(slc_globals, name) for name in _PROVIDER_SETTINGS}
                resource = Resource(attributes={SERVICE_NAME: settings["OTEL_SERVICE_NAME"]})
                provider = TracerProvider(
                    resource=resource,
                    span_limits=SpanLimits(max_events=settings["OTEL_SPAN_EVENT_COUNT_LIMIT"]),
                )
                for exporter in _create_span_exporters(settings):
                    provider.add_span_processor(BatchSpanProcessor(exporter))
                trace.set_tracer_provider(provider)
                _tracer_provider = provider
                _provider_settings = settings
                LOGGER.info("OpenTelemetry tracing initialized successfully")

        except Exception as e:
            # NOTE: without a tracer provider, the OpenTelemetry API hands out spans which are not
            # recorded, so the instrumented code keeps working
            LOGGER.error("Failed to initialize tracing: %s", e, exc_info=e)

        finally:
            _setup_attempted = True


def _create_otlp_span_exporter(base_endpoint: str | None) -> SpanExporter | None:
    """Create the OTLP span exporter.

    As in the OpenTelemetry specification, the ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` environment
    variable takes precedence over ``base_endpoint``. Unlike the latter, it is the full URL to which
    the spans get sent.

    Args:
        base_endpoint: the value of :const:`~qiskit_addon_slc.globals.OTEL_EXPORTER_OTLP_ENDPOINT`.

    Returns:
        An OTLP span exporter, or ``None`` if no endpoint is configured or the exporter is not
        installed.
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if not endpoint:
        if not base_endpoint:
            LOGGER.warning(
                "OTLP exporter requested but neither OTEL_EXPORTER_OTLP_ENDPOINT nor "
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is set."
            )
            return None
        # NOTE: the base endpoint may or may not end in a slash
        endpoint = f"{base_endpoint.rstrip('/')}/v1/traces"
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError as e:
        LOGGER.warning(
            "OTLP exporter requested but it could not be imported: %s. "
            "Install 'opentelemetry-exporter-otlp-proto-http' for OTLP support.",
            e,
        )
        return None
    LOGGER.info("Initialized OTLP span exporter to %s", endpoint)
    return OTLPSpanExporter(endpoint=endpoint)


def _create_span_exporters(settings: dict[str, Any]) -> list[SpanExporter]:
    """Create the span exporters selected by the tracing settings.

    :const:`~qiskit_addon_slc.globals.OTEL_TRACES_EXPORTER` is a comma-separated list of exporters,
    as in the OpenTelemetry specification. Each of ``"otlp"`` and ``"console"`` adds that exporter,
    while ``"none"`` adds none.

    Args:
        settings: the provider settings, as read by :func:`_initialize_tracing`.

    Returns:
        The selected exporters. An exporter which is unknown, or one which is selected but not
        configured or installed, is replaced by a console span exporter.
    """
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter

    names = {name.strip().lower() for name in settings["OTEL_TRACES_EXPORTER"].split(",")} - {""}
    # NOTE: an empty setting selects the default, which is the console exporter
    use_console = not names or "console" in names
    for name in sorted(names - {"console", "otlp", "none"}):
        LOGGER.warning("Unknown span exporter '%s'. Falling back to console exporter.", name)
        use_console = True
    exporters: list[SpanExporter] = []
    if "otlp" in names:
        exporter = _create_otlp_span_exporter(settings["OTEL_EXPORTER_OTLP_ENDPOINT"])
        if exporter is not None:
            exporters.append(exporter)
        else:
            LOGGER.warning("Falling back to console exporter.")
            use_console = True
    if use_console:
        LOGGER.info("Initialized console span exporter")
        exporters.append(ConsoleSpanExporter())
    return exporters


def _get_tracer() -> Tracer:
    """Get the tracer of this package, which creates all of its spans.

    This must only be called when tracing is enabled (see :func:`_is_tracing_enabled`), since it
    sets up tracing on its first call.

    Returns:
        The OpenTelemetry tracer of this package.
    """
    global _tracer

    # NOTE: this runs for every traced span, so the tracer only gets created once
    if _tracer is None:
        _initialize_tracing()

        from opentelemetry import trace

        _tracer = trace.get_tracer(_TRACER_NAME)
    return _tracer


def traced_span(
    name: str,
    *,
    parent_span: Span | NoOpSpan | None = None,
    attributes: dict[str, Any] | None = None,
) -> AbstractContextManager[Span | NoOpSpan]:
    """Context manager for creating a traced span.

    The span is the current span while the context is active, such that the spans created within it
    become its children. A span which gets cut short by a ``KeyboardInterrupt`` or ``SystemExit`` is
    marked as ``interrupted``.

    Args:
        name: the name of the span.
        parent_span: the parent of the span. Defaults to the current span.
        attributes: the attributes of the span.

    Returns:
        A context manager yielding the created span, or a :class:`NoOpSpan` if tracing is
        disabled.

    Example:
        Simple usage::

            with traced_span("my_operation") as span:
                span.add_event("started")
                do_work()

        With parent span::

            with traced_span("child_op", parent_span=parent) as span:
                do_work()
    """
    # NOTE: this runs several times per task, so a disabled span is the shared NoOpSpan rather than
    # a generator-based context manager, which costs about three times as much to enter and exit
    if not _is_tracing_enabled():
        return _NOOP_SPAN
    return _traced_span(name, parent_span, attributes)


@contextmanager
def _traced_span(
    name: str, parent_span: Span | NoOpSpan | None, attributes: dict[str, Any] | None
) -> Generator[Span, None, None]:
    """Implements :func:`traced_span` once tracing is enabled."""
    tracer = _get_tracer()

    ctx = None
    # NOTE: a NoOpSpan parent was obtained before tracing got enabled. Like no parent at all, it
    # leaves the span to nest under the current one, rather than to start a trace of its own.
    if parent_span is not None and not isinstance(parent_span, NoOpSpan):
        from opentelemetry.trace import set_span_in_context

        ctx = set_span_in_context(parent_span)

    with tracer.start_as_current_span(name, context=ctx, attributes=attributes) as span:
        try:
            yield span
        except (SystemExit, KeyboardInterrupt) as exc:
            # OpenTelemetry only records exceptions deriving from Exception, so a span cut short by
            # a Ctrl+C or sys.exit would otherwise look like one that completed normally. Unlike
            # such a failure, this does not set the ERROR status of the span.
            span.set_attribute("interrupted", True)
            span.record_exception(exc)
            raise


def get_current_span() -> Span | NoOpSpan:
    """Get the span which is current in this execution context.

    Returns:
        The current span, which does not record anything if there is none, or a :class:`NoOpSpan`
        if tracing is disabled.
    """
    if not _is_tracing_enabled():
        return _NOOP_SPAN

    from opentelemetry import trace

    return trace.get_current_span()


def _is_own_span(span: Span | NoOpSpan) -> bool:
    """Check whether a span was created by this package.

    Args:
        span: the span to check.

    Returns:
        Whether ``span`` was started by the tracer of this package. This is ``False`` for a span of
        the application, and for a span which records nothing.
    """
    scope = getattr(span, "instrumentation_scope", None)
    return scope is not None and scope.name == _TRACER_NAME


def _inject_trace_context(span: Span | NoOpSpan | None = None) -> dict[str, str] | None:
    """Serialize a trace context, such that another process can nest its spans under it.

    Args:
        span: the span whose context to serialize. If ``None``, or a :class:`NoOpSpan` obtained
            before tracing got enabled, the current context is serialized.

    Returns:
        The serialized trace context, or ``None`` if tracing is disabled.
    """
    if not _is_tracing_enabled():
        return None

    from opentelemetry.propagate import inject
    from opentelemetry.trace import set_span_in_context

    carrier: dict[str, str] = {}
    if span is None or isinstance(span, NoOpSpan):
        inject(carrier)
    else:
        inject(carrier, context=set_span_in_context(span))
    return carrier


class WorkerSetup(NamedTuple):
    """What a worker process of a pool needs to know to be traced like the main process.

    This gets created in the main process by :class:`WorkerSpans` and passed to
    :func:`initialize_worker` in every worker process.
    """

    worker_contexts: list[dict[str, str] | None]
    """The serialized trace context of the span of each worker, by its index."""
    trace_context: dict[str, str] | None
    """The serialized trace context under which a worker without a span nests its spans.

    Only a worker which replaces one that died has no span of its own.
    """
    tracing_settings: dict[str, Any]
    """The tracing settings of the main process, as returned by :func:`_get_tracing_settings`."""
    application_tracer_provider: bool
    """Whether the main process traces through a tracer provider of the application, as returned by
    :func:`_uses_application_tracer_provider`."""
    worker_counter: Synchronized[int]
    """A shared integer counting the workers of the pool, from which each worker draws its index.

    This numbers the workers of every pool from zero.
    """


def initialize_worker(setup: WorkerSetup) -> None:
    """Initialize a worker process, such that the spans of its tasks are nested under its span.

    This must be called once in every worker process when it starts, typically as the
    ``initializer`` of a :class:`multiprocessing.pool.Pool`. The worker does not record a span of
    its own, because a worker which gets terminated is killed before it could end that span.
    Instead, the main process records a span for each worker (see :class:`WorkerSpans`), whose
    context it passes to that worker through ``setup``.

    The tracing settings of the main process are applied to this worker process before anything
    else, so that the worker is traced exactly when the main process is.

    A worker started with the ``spawn`` or ``forkserver`` start method is a fresh Python process. It
    has the tracer provider of the application only if the application sets that provider up when
    its main module gets imported. Otherwise, the worker creates its own provider from the tracing
    settings, so its spans go to the exporter configured in :mod:`qiskit_addon_slc.globals`. If the
    main process traces through a provider of the application, the first worker logs a warning
    about this.

    Args:
        setup: the setup of the workers, as created by the main process.

    Example:
        In main process::

            worker_spans = WorkerSpans(num_processes)
            pool = mp.Pool(
                num_processes,
                initializer=initialize_worker,
                initargs=(worker_spans.worker_setup,)
            )
    """
    for name, value in setup.tracing_settings.items():
        setattr(slc_globals, name, value)

    if not _is_tracing_enabled():
        return

    with setup.worker_counter.get_lock():
        worker_index = setup.worker_counter.value
        setup.worker_counter.value += 1
    if worker_index < len(setup.worker_contexts):
        trace_context = setup.worker_contexts[worker_index]
    else:
        trace_context = setup.trace_context

    pid = os.getpid()
    _WORKER_STORAGE.worker_index = worker_index

    LOGGER.debug("Initializing worker %s (PID: %s)", worker_index, pid)

    from opentelemetry import context
    from opentelemetry.propagate import extract

    # NOTE: the context stays attached for as long as the worker runs, such that the span of every
    # task becomes a child of the span given by trace_context
    if trace_context is not None:
        context.attach(extract(trace_context))

    # NOTE: this creates the tracer provider of a worker which did not inherit one
    _get_tracer()

    if setup.application_tracer_provider and _owns_tracer_provider() and worker_index == 0:
        LOGGER.warning(
            "The worker processes cannot use the tracer provider of the application, because they "
            "were not forked from the main process and the application did not set up its provider "
            "when its main module got imported. Their spans go to the exporter configured in "
            "qiskit_addon_slc.globals instead."
        )

    # A worker exports its spans after every task (see _export_finished_spans). A Finalize with an
    # exitpriority also exports them when the worker exits, even on Python versions which do not
    # run the atexit handlers of forked workers.
    mp_util.Finalize(None, _export_finished_spans, exitpriority=100)

    LOGGER.debug("Worker %s (PID: %s) initialized successfully", worker_index, pid)


def _export_finished_spans() -> None:
    """Export the spans which this process has finished, rather than waiting for the next batch.

    A worker process calls this after each of its tasks. A terminated worker is killed before it can
    export any span, so this limits the spans which get lost to those of the tasks still running.
    This blocks until the spans are exported, for at most the timeout of the exporter, such as the
    ``OTEL_EXPORTER_OTLP_TIMEOUT`` of the OTLP exporter.
    """
    if not _is_tracing_enabled():
        return
    # NOTE: only the provider of the SDK can be flushed, which a provider of the application need
    # not be
    force_flush = getattr(_tracer_provider, "force_flush", None)
    if force_flush is None:
        return
    try:
        force_flush()
    except Exception as e:
        LOGGER.debug("Failed to flush tracer provider: %s", e)


def get_task_index() -> int:
    """Get the index of the task which this worker process is running, or ran last.

    Returns:
        The index of the task, or ``_WORKER_INFO_UNSET`` (-1) if no task was started with
        :func:`run_worker_task`, e.g. outside of a worker process.
    """
    return getattr(_WORKER_STORAGE, "task_idx", _WORKER_INFO_UNSET)


class _TaskRecord(NamedTuple):
    """The record of a task which a worker finished, as created by :func:`run_worker_task`."""

    worker_index: int
    """The index of the worker that ran the task, or ``_WORKER_INFO_UNSET`` if it has none."""
    end_time: int
    """The time at which the task finished, in nanoseconds since the epoch."""


def run_worker_task(task_idx: int, fn: Callable[..., _T], *args: Any) -> tuple[_T, _TaskRecord]:
    """Run a task in a worker process, recording it for tracing.

    This stores ``task_idx`` such that :func:`get_task_index` reports it while ``fn(*args)`` runs,
    and exports the spans of the task once it finished. Keeping this out of ``fn`` means that
    ``fn`` does not need to know about tracing.

    If ``fn`` raises an exception, the record of the task gets attached to that exception, which the
    pool passes on to the error callback of the task. The error callback created by
    :meth:`WorkerSpans.recording_failure` finds it there.

    Args:
        task_idx: the index of the task.
        fn: the function computing the result of the task.
        args: the arguments of ``fn``.

    Returns:
        The result of ``fn(*args)``, together with a record of the task, which the callback created
        by :meth:`WorkerSpans.recording` unpacks.
    """
    _WORKER_STORAGE.task_idx = task_idx
    worker_index = getattr(_WORKER_STORAGE, "worker_index", _WORKER_INFO_UNSET)
    # NOTE: the task ends before its spans get exported, which may take a while
    try:
        result = fn(*args)
        end_time = time.time_ns()
    except Exception as exc:
        # NOTE: the attributes of an exception are pickled along with it, so this record reaches
        # the main process. An exception which does not allow setting it stays unrecorded.
        with suppress(AttributeError):
            setattr(exc, _TASK_RECORD_ATTRIBUTE, _TaskRecord(worker_index, time.time_ns()))
        raise
    finally:
        _export_finished_spans()
    return result, _TaskRecord(worker_index, end_time)


class WorkerSpans:
    """The spans of the worker processes of a pool, which the main process records for them.

    A worker which gets terminated, e.g. because the computation reached its timeout, is killed
    before it can end a span of its own. Hence, the main process records a ``worker_<index>`` span
    for each worker instead, as a child of the current span. These spans start before the pool
    does, so they include the startup of the worker processes. Each worker nests the spans of its
    tasks under its own span, which it finds in :attr:`worker_setup`.

    The tasks of the workers must be run by :func:`run_worker_task`, their callbacks wrapped by
    :meth:`recording`, and their error callbacks by :meth:`recording_failure`. Once the pool has
    stopped, :meth:`end` records how many tasks each worker finished and how many of them failed,
    and ends the span of each worker when its last task finished, or right where it started if the
    worker never got a task. Hence, the durations of these spans show how evenly the work was spread
    across the workers.
    """

    def __init__(self, num_workers: int) -> None:
        """Starts the spans of the workers.

        Args:
            num_workers: the number of worker processes of the pool.
        """
        self._start_time = time.time_ns()
        self.spans: list[Span | NoOpSpan]
        """The span of each worker, by its index."""
        if _is_tracing_enabled():
            tracer = _get_tracer()
            self.spans = [
                tracer.start_span(
                    f"worker_{idx}", attributes={"worker.index": idx}, start_time=self._start_time
                )
                for idx in range(num_workers)
            ]
        else:
            self.spans = [_NOOP_SPAN] * num_workers
        self.worker_setup = WorkerSetup(
            [_inject_trace_context(span) for span in self.spans],
            _inject_trace_context(),
            _get_tracing_settings(),
            _uses_application_tracer_provider(),
            mp.Value("i", 0),
        )
        """The setup to pass to :func:`initialize_worker` in every worker process."""
        self._tasks_completed = [0] * num_workers
        self._tasks_failed = [0] * num_workers
        self._last_task_end: list[int | None] = [None] * num_workers
        self._ended = False

    def recording(self, callback: Callable[[_T], Any]) -> Callable[[tuple[_T, _TaskRecord]], None]:
        """Wraps the callback of a task run by :func:`run_worker_task`, such that it gets recorded.

        Args:
            callback: the callback receiving the result of the task.

        Returns:
            A callback which records the task and passes its result on to ``callback``.
        """

        def _record_and_call(result: tuple[_T, _TaskRecord]) -> None:
            value, record = result
            self._record_task(record, failed=False)
            callback(value)

        return _record_and_call

    def recording_failure(
        self, error_callback: Callable[[BaseException], Any]
    ) -> Callable[[BaseException], None]:
        """Wraps the error callback of a task, such that the task gets recorded when it fails.

        Only a task whose exception carries the record which :func:`run_worker_task` attached to it
        gets recorded. A task which could not even be sent to a worker has no such record.

        Args:
            error_callback: the callback receiving the exception which the task raised.

        Returns:
            A callback which records the failed task and passes its exception on to
            ``error_callback``.
        """

        def _record_and_call(exc: BaseException) -> None:
            record = getattr(exc, _TASK_RECORD_ATTRIBUTE, None)
            if record is not None:
                self._record_task(record, failed=True)
            error_callback(exc)

        return _record_and_call

    def _record_task(self, record: _TaskRecord, failed: bool) -> None:
        """Records a task which a worker finished.

        The tasks of a worker without a span, such as one which replaced a worker that died, or one
        which is not traced, are not recorded.

        Args:
            record: the record of the task.
            failed: whether the task raised an exception.
        """
        # NOTE: unlike the error callbacks of compute_bounds, this needs no lock. The pool runs the
        # callbacks of the tasks which reached a worker in a single thread, while the error callback
        # of a task which it could not send to a worker gets no record, and so does not get here.
        worker_index, end_time = record
        if not 0 <= worker_index < len(self.spans):
            return
        self._tasks_completed[worker_index] += 1
        if failed:
            self._tasks_failed[worker_index] += 1
        last = self._last_task_end[worker_index]
        self._last_task_end[worker_index] = end_time if last is None else max(last, end_time)

    def end(self, terminated: bool) -> None:
        """Ends the spans of the workers. Calling this again does nothing.

        Args:
            terminated: whether the pool was terminated. Its workers may have still been running a
                task then, so their spans end now rather than with their last finished task.
        """
        if self._ended:
            return
        self._ended = True
        now = time.time_ns()
        for span, completed, failed, last in zip(
            self.spans, self._tasks_completed, self._tasks_failed, self._last_task_end, strict=True
        ):
            span.set_attribute("tasks.completed", completed)
            span.set_attribute("tasks.failed", failed)
            span.set_attribute("worker.terminated", terminated)
            if terminated:
                end_time = now
            elif last is None:
                # NOTE: a worker of a completed pool which never got a task was idle all along
                end_time = self._start_time
            else:
                end_time = last
            span.end(end_time=end_time)


class StepTimer:
    """Counts and times steps which run too often to be traced as spans of their own.

    Tracing every run of such a step as its own span floods the queue of spans to export. Instead,
    the number of runs of each step and their total duration get recorded on a single span. The
    steps are only timed if that span gets recorded at all.
    """

    def __init__(self, span: Span | NoOpSpan, names: Iterable[str]) -> None:
        """Creates a timer for steps whose runs get recorded on ``span``.

        Args:
            span: the span on which to record the steps.
            names: the names of the steps, which get recorded even if they never run. Other steps
                get recorded once they ran.
        """
        self._span = span
        self._enabled = span.is_recording()
        self._calls: defaultdict[str, int] = defaultdict(int, dict.fromkeys(names, 0))
        self._seconds: defaultdict[str, float] = defaultdict(float, dict.fromkeys(names, 0.0))

    def run(self, name: str, step: Callable[..., _T], *args: Any) -> _T:
        """Runs a step, counting and timing it.

        Args:
            name: the name of the step.
            step: the function implementing the step.
            args: the arguments of ``step``.

        Returns:
            The result of ``step(*args)``.
        """
        if not self._enabled:
            return step(*args)
        start = time.perf_counter()
        result = step(*args)
        self._calls[name] += 1
        self._seconds[name] += time.perf_counter() - start
        return result

    def record(self) -> None:
        """Records the ``<name>.calls`` and ``<name>.seconds`` attributes of every step."""
        for name, calls in self._calls.items():
            self._span.set_attribute(f"{name}.calls", calls)
            self._span.set_attribute(f"{name}.seconds", self._seconds[name])


def mark_moment(name: str, parent_span: Span | NoOpSpan, attributes: dict[str, Any]) -> None:
    """Record a span of next to no length, marking a moment in time.

    Unlike an event, this span appears as its own row in the timeline of a trace.

    Args:
        name: the name of the span.
        parent_span: the parent of the span.
        attributes: the attributes of the span.
    """
    with traced_span(name, parent_span=parent_span, attributes=attributes):
        pass


def record_outcome(
    span: Span | NoOpSpan, parent_span: Span | NoOpSpan, attributes: dict[str, Any]
) -> None:
    """Record how a computation ended on its span and on the span of its caller.

    This makes an incomplete computation visible at a glance and searchable by these attributes. The
    span of the caller gets them as well if this package created it, such as that of
    :func:`.compute_forward_bounds`, which is the root of the trace unless the application traces
    too. A span of the application is left untouched, since its attributes may well have the same
    names.

    Args:
        span: the span of the computation.
        parent_span: the span that was current when the computation started.
        attributes: the attributes describing how the computation ended.
    """
    for outcome_span in (span, parent_span) if _is_own_span(parent_span) else (span,):
        for key, value in attributes.items():
            outcome_span.set_attribute(key, value)
