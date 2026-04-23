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

# Warning: this module is not documented and it does not have an RST file.
# If we ever publicly expose interfaces users can import from this module,
# we should set up its RST file.

"""Global settings.

.. currentmodule:: qiskit_addon_slc.globals

This module provides a number of globally configurable settings.

.. autoclass:: PROGRESS_POLLING_PERIOD

.. autoclass:: ZERO_ATOL
"""

import sys

PROGRESS_POLLING_PERIOD = 1
"""The polling period for the progress indicator of the commutator bound task computation.

This number corresponds to the number of seconds to wait between progress indicator updates.
It defaults to ``1``.
"""

ZERO_ATOL = 10 * sys.float_info.epsilon
"""The absolute tolerance value below which terms are considered truly zero and are truncated.

This defaults to the value of ``10 * sys.float_info.epsilon``.
"""
