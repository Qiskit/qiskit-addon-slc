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

"""Optional dependencies.

.. currentmodule:: qiskit_addon_slc.optionals

Availability
------------

Indicators for the presence of optional runtime dependencies.

.. autoclass:: HAS_OPENTELEMETRY
"""

from qiskit.utils import LazyImportTester as _LazyImportTester

# NOTE: this tests for the SDK rather than the API, which other packages often depend on, because
# spans can only be recorded with the SDK
HAS_OPENTELEMETRY = _LazyImportTester(
    "opentelemetry.sdk", install="pip install qiskit-addon-slc[opentelemetry]"
)
"""Indicates whether the optional ``opentelemetry`` dependency group is installed."""
