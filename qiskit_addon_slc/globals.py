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

.. autoclass:: ZERO_ATOL
"""

import sys

ZERO_ATOL = 10 * sys.float_info.epsilon
"""The absolute tolerance value below which terms are considered truly zero and are truncated.

This defaults to the value of ``10 * sys.float_info.epsilon``.
"""
