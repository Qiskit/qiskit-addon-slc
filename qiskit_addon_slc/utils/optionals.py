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

# ruff: noqa: D205,D212,D415
"""
=====================
Optional Dependencies
=====================

.. currentmodule:: qiskit_addon_slc.utils.optionals

This module defines lazy availability testers for optional third-party dependencies. The checkers
are instances of :external:class:`~qiskit.utils.LazyImportTester` and can be used as booleans
(evaluated lazily), or to raise a :external:class:`~qiskit.exceptions.MissingOptionalLibraryError`
when a dependency is required.

-----------------
Available Testers
-----------------

.. autodata:: HAS_PYSCF
"""

from qiskit.utils import LazyImportTester

HAS_PYSCF = LazyImportTester(
    "pyscf",
    name="PySCF",
    install="pip install qiskit-addon-slc[pyscf]",
)
"""`PySCF <https://pyscf.org/>`__ is a quantum chemistry package used by the
Davidson eigensolver helper.

.. seealso::
   :external:class:`~qiskit.utils.LazyDependencyManager` for usage examples and the available
   methods of this object.
"""
