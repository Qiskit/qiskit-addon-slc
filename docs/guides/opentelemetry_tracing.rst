###############################
Collecting OpenTelemetry traces
###############################

Computing bounds for a large circuit can take a long time, and most of that time is spent in
parallel worker processes. To see where the time goes, the bound computations can emit
`OpenTelemetry <https://opentelemetry.io/>`_ traces. Each trace records the main computation, every
worker process, and every individual error term that a worker bounded, together with timings and
metadata such as the number of operator terms after Pauli propagation.

Tracing is turned off by default. When it is turned off, or when the OpenTelemetry packages are not
installed, the instrumentation does nothing.


Install the optional dependencies
---------------------------------

The OpenTelemetry API, SDK and OTLP/HTTP exporter are an optional dependency group:

.. code-block:: bash

   pip install 'qiskit-addon-slc[opentelemetry]'


Configure tracing
-----------------

Tracing is configured with the following environment variables:

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Meaning
   * - ``QISKIT_SLC_TRACING_ENABLED``
     - ``false``
     - Set to ``true``, ``1``, ``yes`` or ``on`` to turn tracing on.
   * - ``OTEL_TRACES_EXPORTER``
     - ``console``
     - Where to send spans: ``console`` prints them to standard output as JSON, ``otlp`` sends them
       to an OpenTelemetry collector, and ``none`` discards them. A comma-separated list such as
       ``otlp,console`` uses several exporters at once.
   * - ``OTEL_EXPORTER_OTLP_ENDPOINT``
     - ``http://localhost:4318``
     - The base URL of the collector. Spans are sent over OTLP/HTTP to ``<endpoint>/v1/traces``.
   * - ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``
     - unset
     - The full URL to send the spans to, such as ``http://localhost:4318/v1/traces``. When set,
       it takes precedence over ``OTEL_EXPORTER_OTLP_ENDPOINT``.
   * - ``OTEL_EXPORTER_OTLP_TIMEOUT``
     - ``10``
     - How many seconds the OTLP exporter waits for the collector to accept a batch of spans.
   * - ``OTEL_SERVICE_NAME``
     - ``qiskit-addon-slc``
     - The service name under which the traces are reported.
   * - ``OTEL_SPAN_EVENT_COUNT_LIMIT``
     - ``1024``
     - The maximum number of events kept for any span. Beyond it, each new event drops the oldest.

Most of these variables are read when :mod:`qiskit_addon_slc` is imported, and the others when the
first span is created, so set them before that import. Setting them in the shell that starts Python
is the simplest way:

.. code-block:: bash

   export QISKIT_SLC_TRACING_ENABLED=true
   export OTEL_TRACES_EXPORTER=otlp
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
   python my_slc_script.py

In a Jupyter notebook, where the kernel is already running, set the corresponding attributes of
:mod:`qiskit_addon_slc.globals` instead. Do so before the first bound computation, because the
exporter is set up the first time a span is created. Changing the exporter afterwards has no effect
until the kernel is restarted, and a warning says so:

.. code-block:: python

   from qiskit_addon_slc import globals as slc_globals

   slc_globals.TRACING_ENABLED = True
   slc_globals.OTEL_TRACES_EXPORTER = "otlp"
   slc_globals.OTEL_EXPORTER_OTLP_ENDPOINT = "http://localhost:4318"

If your application sets up its own OpenTelemetry tracer provider before the first bound
computation, the spans go to that provider and its exporters instead, and the variables above that
select and configure the exporter have no effect. This way, the spans of the bound computations
become part of your application's traces. Worker processes started with the ``spawn`` or
``forkserver`` start method of :mod:`multiprocessing` only use that provider if your application
sets it up when its main module is imported. Otherwise, they fall back to the exporter configured
by the variables above, and a warning is logged.

To check that tracing works without running a collector, enable it and leave the exporter at its
``console`` default. Every finished span is then printed as a JSON object. This output is verbose,
and every worker process writes its own, so use a collector for anything beyond a quick check.


Run a collector
---------------

Any backend that accepts OTLP over HTTP can receive the traces.
`Jaeger <https://www.jaegertracing.io/>`_ is a convenient choice for local use, because a single
container both collects the traces and serves a web interface for browsing them:

.. code-block:: bash

   docker run --rm --name jaeger \
     -p 4318:4318 \
     -p 16686:16686 \
     jaegertracing/jaeger:latest

Port ``4318`` receives OTLP/HTTP and port ``16686`` serves the web interface. With this container
running, set ``OTEL_TRACES_EXPORTER=otlp``. The default ``OTEL_EXPORTER_OTLP_ENDPOINT`` of
``http://localhost:4318`` already points to it.


Trace a bound computation
-------------------------

With tracing configured, run the bound computations as usual. Nothing about the calls themselves
changes. Every worker process gets its own span in the trace, so using several processes shows how
the work is spread across them:

.. code-block:: python

   from qiskit_addon_slc.bounds import compute_backward_bounds, compute_forward_bounds

   forward_bounds = compute_forward_bounds(
       boxed_circuit, noise_model_paulis, observable, num_processes=8
   )
   backward_bounds = compute_backward_bounds(boxed_circuit, noise_model_paulis, num_processes=8)

The main process exports spans in batches every few seconds while the computation runs, and any
remaining spans are flushed when it exits. The worker processes export the spans of each task as
soon as the task completes, and wait for that export before they start their next task. When the
collector cannot be reached, the OTLP exporter keeps retrying for several seconds after every task,
which slows the computation down accordingly. Start the collector first, or lower
``OTEL_EXPORTER_OTLP_TIMEOUT`` to bound this delay.

Each process buffers at most 2048 finished spans until they are exported. When many short tasks
produce spans faster than the collector accepts them, OpenTelemetry logs the warning
``Queue full, dropping Span.`` and the trace is incomplete. In that case, increase the buffer with
the ``OTEL_BSP_MAX_QUEUE_SIZE`` environment variable, set before the first bound computation.

Open the Jaeger interface at ``http://localhost:16686``, choose the ``qiskit-addon-slc`` service and
click **Find Traces**. Each call to :func:`.compute_forward_bounds` or
:func:`.compute_backward_bounds` appears as its own trace.


Read a trace
------------

A trace consists of nested spans. The main process records the following:

- ``compute_forward_bounds`` or ``compute_backward_bounds``: the entire call. Its attributes record
  what is measured: the observable (``observable.label`` and ``observable.num_active_qubits``) for
  forward bounds, and the number of measured qubits (``measurements.num_qubits``) for backward
  bounds. They also record how the computation ended (see below). It has two children:
  ``preparation``, which covers preparing the circuit and initializing the light cone, and
  ``compute_bounds``.
- ``compute_bounds``: the generic bound computation, with the circuit width and number of processes
  as attributes, as well as the following attributes on how the computation ended:

  - ``stop_reason``: ``completed``, ``timeout`` or ``keyboard_interrupt``.
  - ``timeout.seconds``: the ``timeout`` that was set, or ``-1`` if none was set.
  - ``tasks.total`` and ``tasks.completed``: the number of spawned and of completed tasks. When the
    computation was stopped early, the error terms of the tasks that did not complete are given the
    trivial bound of ``2.0``.
  - ``tasks.failed``: the number of completed tasks whose norm computation raised an exception.
    Their error terms keep the trivial bound of ``2.0`` as well.

  When the computation was stopped early, a ``timeout_reached`` or ``keyboard_interrupt_received``
  span of zero length marks the point in time at which it was stopped. Its ``elapsed.seconds`` and
  ``tasks.completed`` attributes give the time and progress at that moment.

  Its children are the following, besides the worker spans described below:

  - ``task_spawning``: the iteration over the circuit, during which the light cone is grown and one
    task is spawned per error term. Its ``tasks.total`` and ``boxes.encountered`` attributes give
    the size of the computation. Two steps run once for every instruction of the circuit: checking
    whether it commutes with the light cone, and absorbing a Clifford gate into the accumulated
    Clifford. Their ``light_cone.commutes.calls`` and ``net_clifford.dot.calls`` attributes count
    how often they ran, and their ``light_cone.commutes.seconds`` and ``net_clifford.dot.seconds``
    attributes give the total time spent in them. A ``noise_terms_evolution`` child span per box
    shows how long it takes to move that box's error terms through the accumulated Clifford.
  - ``progress_tracking``: waiting for the workers to finish. Its ``progress_update`` events track
    the number of completed tasks over time, in their ``tasks.completed``, ``tasks.total`` and
    ``tasks.completed_percentage`` attributes.

Each worker process has a ``worker_<index>`` span, which the main process records on its behalf. It
starts with the worker pool and ends when the worker finished its last task, or has no length at all
if the worker never got a task. Its ``tasks.completed`` attribute counts the tasks the worker
finished, and its ``tasks.failed`` attribute those of them which raised an exception. Under it,
every task gets a ``forward_norm_computation`` or ``backward_norm_computation`` span with the Pauli
error term (``pauli.label``), the task index (``task.index``) and the computed ``result.*`` values
as attributes. When the computation of a bound was aborted, its ``result.aborted`` attribute is
``true`` and it has no ``result.commutator_bound`` attribute. When it raised an exception, its
status is ``ERROR`` and an ``exception`` event records the exception. Its children break the task
down further:

- ``pauli_prop``: the propagation of the error term through the circuit. The
  ``pauli_prop.num_terms`` attribute and the ``pauli_prop.num_active_qubits`` attribute, which
  counts the qubits that the operator acts on non-trivially, give the size of the propagated
  operator. This is usually what makes a task expensive.
- ``commutator`` (forward bounds only): building and simplifying the commutator with the observable,
  with its size and coefficient statistics as attributes.
- ``direct_eigenvalue_computation`` or ``davidson_eigenvalue_computation`` (forward bounds only):
  computing the commutator's spectral norm.

Events on the norm computation spans record shortcuts and fallbacks, such as
``single_pauli_optimization``, ``zero_commutator``, ``qubit_limit_exceeded`` or
``fallback_to_triangle_inequality``. A task that was abandoned because its truncation bias already
exceeds the trivial bound of ``2.0`` carries a ``computation_aborted`` event.

Each worker exports the spans of a task as soon as the task completes. When the computation is
stopped early, the worker processes get terminated right away, even when they are busy in a long
native call such as the Pauli propagation or an eigensolver. Hence, the spans of the tasks that were
still running are missing from the trace, while the ``worker_<index>`` spans end at that moment and
carry the ``worker.terminated`` attribute. To find computations that did not complete, search for
the tag ``stop_reason=timeout`` or ``stop_reason=keyboard_interrupt`` in Jaeger.

Only an interrupt while waiting for the workers stops the computation early and returns its partial
bounds. A ``KeyboardInterrupt`` or ``SystemExit`` at any other point, such as while the tasks get
spawned, propagates as usual. Every span which it cuts short carries an ``interrupted`` attribute
and an ``exception`` event, but no ``ERROR`` status, since an interruption is not a failure.

In practice, the per-task spans answer the most common questions: comparing the durations of the
``worker_<index>`` spans shows whether the work was spread evenly across the processes, and sorting
the norm computation spans by duration reveals which error terms dominate the run time, and whether
that time goes into the Pauli propagation or the eigensolver. These are the costs that the
``evolution_max_terms`` argument of :func:`.compute_forward_bounds` and
:func:`.compute_backward_bounds`, and the ``eigval_max_qubits`` argument of
:func:`.compute_forward_bounds`, let you trade against the tightness of the bounds. The guide on
:doc:`tuning the configuration options <configuration_options>` shows how.
