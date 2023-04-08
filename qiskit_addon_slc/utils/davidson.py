# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
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

"""A basic Davidson solver."""

from typing import cast

import numpy as np
import pyscf
from qiskit.quantum_info import SparsePauliOp


def get_extremal_eigenvalue(spo: SparsePauliOp, **kwargs) -> tuple[bool, float]:
    """Finds the extremal eigenvalue of the provided operator.

    This converts the provided operator to a sparse matrix whose minimal eigenvalue is required.

    .. note::
        The current implementation is definitely not optimized in terms of performance.

    Args:
        spo: the operator whose minimal eigenvalue to find.
        kwargs: additional keyword arguments for :func:`~pyscf.lib.linalg_helper.davidson1`. When
            not specified otherwise, the following defaults will be used:

            * `tol`: 1e-6
            * `max_cycle`: 500
            * `max_space`: 12
            * `lindep`: 1e-11
            * `max_memory`: 2000

            Other values will default to PySCF's default values.

    Returns:
        A pair indicating whether the Davidson algorithm has converged and the obtained minimal
        eigenvalue.
    """
    default_kwargs = {
        "tol": 1e-6,
        "max_cycle": 500,
        "max_space": 12,
        "lindep": 1e-11,
        "max_memory": 2000,
    }
    default_kwargs.update(kwargs)

    spmat = spo.to_matrix(sparse=True, force_serial=True)

    x0 = [_random_initial_guess(spmat.shape)]

    diag = spmat.diagonal()

    def precond(dx, e, _):
        x = diag - e
        x[np.abs(x) < default_kwargs["tol"]] = default_kwargs["tol"]
        return dx / x

    converged, e, _ = pyscf.lib.davidson1(
        lambda vecs: [spmat.dot(v) for v in vecs],
        x0,
        precond,
        **default_kwargs,
    )

    return converged, e[0]


def _random_initial_guess(shape: tuple[int, ...]) -> np.ndarray:
    """Produces a random array of the requested shape.

    Args:
        shape: the requested shape.

    Returns:
        An array of random complex values with their real and imaginary parts lying in the interval
        ``[0, 1)``.
    """
    norm = 0.0

    while norm == 0:
        x = np.random.rand(shape[0]) + 1.0j * np.random.rand(shape[0])
        norm = cast(float, np.linalg.norm(x))

    return x / norm
