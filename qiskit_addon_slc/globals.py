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

"""Global settings.

.. currentmodule:: qiskit_addon_slc.globals

This module provides a number of globally configurable settings.

.. autoclass:: PROGRESS_POLLING_PERIOD

.. autoclass:: ZERO_ATOL

.. autoclass:: TRACING_ENABLED

.. autoclass:: OTEL_SERVICE_NAME

.. autoclass:: OTEL_TRACES_EXPORTER

.. autoclass:: OTEL_EXPORTER_OTLP_ENDPOINT

.. autoclass:: OTEL_SPAN_EVENT_COUNT_LIMIT
"""

import logging
import os
import sys

PROGRESS_POLLING_PERIOD = 5
"""The polling period for the progress indicator of the commutator bound task computation.

This number corresponds to the number of seconds to wait between progress indicator updates.
It defaults to ``5``.
"""

ZERO_ATOL = 10 * sys.float_info.epsilon
"""The absolute tolerance value below which terms are considered truly zero and are truncated.

This defaults to the value of ``10 * sys.float_info.epsilon``.
"""

# The tracing settings below configure the tracing implemented in qiskit_addon_slc.utils._tracing.
# That module forwards them to the worker processes, so each one must also be listed there: in
# _PROVIDER_SETTINGS if it configures the tracer provider, else in _TRACING_SETTINGS.
TRACING_ENABLED = os.getenv("QISKIT_SLC_TRACING_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
    "on",
)
"""Whether the bound computations are traced with OpenTelemetry.

This can be controlled via the ``QISKIT_SLC_TRACING_ENABLED`` environment variable.
Set to ``"true"``, ``"1"``, ``"yes"``, or ``"on"`` to enable tracing. Defaults to ``False``.

Unlike the other tracing settings, this one can be changed at any time.
"""

OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "qiskit-addon-slc")
"""The service name to use for OpenTelemetry traces.

This can be controlled via the ``OTEL_SERVICE_NAME`` environment variable.
Defaults to ``"qiskit-addon-slc"``.

This setting takes effect only once per process, when the first span gets created.
"""

OTEL_TRACES_EXPORTER = os.getenv("OTEL_TRACES_EXPORTER", "console")
"""The OpenTelemetry traces exporter type to use.

This can be controlled via the ``OTEL_TRACES_EXPORTER`` environment variable.
Supported values are ``"console"``, ``"otlp"`` and ``"none"``, or a comma-separated list of these,
such as ``"otlp,console"``, to use several exporters at once.
Defaults to ``"console"``.

The OpenTelemetry specification defaults to ``"otlp"`` instead. This package deviates from it,
because the worker processes export their spans after every task: without a collector to receive
them, every such export of the OTLP exporter blocks for seconds, which stalls the computation.

This setting takes effect only once per process, when the first span gets created.
"""

OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
"""The base URL of the OTLP exporter endpoint.

This can be controlled via the ``OTEL_EXPORTER_OTLP_ENDPOINT`` environment variable.
The spans get sent to ``<endpoint>/v1/traces``, unless the ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``
environment variable gives the full URL to send the spans to, which then takes precedence.
Defaults to ``"http://localhost:4318"``, as in the OpenTelemetry specification.

This setting takes effect only once per process, when the first span gets created.
"""


def _getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Ignoring %s=%r, which is not an integer. Using %d instead.", name, value, default
        )
        return default


OTEL_SPAN_EVENT_COUNT_LIMIT = _getenv_int("OTEL_SPAN_EVENT_COUNT_LIMIT", 1024)
"""The maximum number of events to keep for any span.

This can be controlled via the ``OTEL_SPAN_EVENT_COUNT_LIMIT`` environment variable.
Once a span reaches this limit, each further event drops the oldest one.
Defaults to ``1024``.

The OpenTelemetry specification defaults to ``128`` instead. This package raises it, because the
progress tracking of a bound computation adds an event every :const:`PROGRESS_POLLING_PERIOD`
seconds. With the defaults of both settings, a computation can run for more than an hour (about 85
minutes) before its earliest events get dropped.

This setting takes effect only once per process, when the first span gets created.
"""
