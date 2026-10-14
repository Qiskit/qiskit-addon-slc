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

"""Fixtures shared by the tests of the tracing."""

import types

import pytest
from qiskit_addon_slc import globals as slc_globals
from qiskit_addon_slc.optionals import HAS_OPENTELEMETRY
from qiskit_addon_slc.utils import _tracing as tracing


@pytest.fixture(autouse=True)
def isolated_tracing(monkeypatch):
    """Gives every test the tracing state of a fresh process, in which tracing is disabled.

    Both this package and OpenTelemetry set up tracing only once per process. Without resetting
    both, the first test which sets up tracing would decide how every later test gets traced. Being
    an autouse fixture, this runs before any fixture which a test requests, such as the ones below.
    """
    monkeypatch.setattr(slc_globals, "TRACING_ENABLED", False)
    # NOTE: without any exporter, setting up tracing neither prints spans nor starts threads
    monkeypatch.setattr(slc_globals, "OTEL_TRACES_EXPORTER", "none")
    # NOTE: this is the only environment variable which the tracing module reads itself
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.setattr(tracing, "_tracer_provider", None)
    monkeypatch.setattr(tracing, "_setup_attempted", False)
    monkeypatch.setattr(tracing, "_tracer", None)
    monkeypatch.setattr(tracing, "_provider_settings", None)
    monkeypatch.setattr(tracing, "_WORKER_STORAGE", types.SimpleNamespace())
    if HAS_OPENTELEMETRY:
        from opentelemetry import trace
        from opentelemetry.util._once import Once

        # NOTE: OpenTelemetry allows setting its global tracer provider only once per process. This
        # resets it like reset_trace_globals of the opentelemetry-test-utils package does.
        monkeypatch.setattr(trace, "_TRACER_PROVIDER_SET_ONCE", Once())
        monkeypatch.setattr(trace, "_TRACER_PROVIDER", None)
        monkeypatch.setattr(trace, "_PROXY_TRACER_PROVIDER", trace.ProxyTracerProvider())


@pytest.fixture
def tracing_enabled(monkeypatch):
    """Enables tracing, which then gets set up without any exporter when it is first used."""
    monkeypatch.setattr(slc_globals, "TRACING_ENABLED", True)


@pytest.fixture
def recording_span():
    """A span which gets recorded, by a tracer provider without any exporter."""
    from opentelemetry.sdk.trace import TracerProvider

    return TracerProvider().get_tracer("qiskit_addon_slc").start_span("test")


@pytest.fixture
def finished_spans(monkeypatch):
    """Enables tracing, with an exporter which keeps the finished spans in memory.

    Only the spans of the current process end up there, not those of any worker process.
    """
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setattr(slc_globals, "TRACING_ENABLED", True)
    tracing._get_tracer()
    exporter = InMemorySpanExporter()
    tracing._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter
