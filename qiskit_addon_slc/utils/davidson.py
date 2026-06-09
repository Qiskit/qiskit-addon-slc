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
import scipy.linalg
import scipy.sparse.linalg
from qiskit.quantum_info import SparsePauliOp


def get_extremal_eigenvalue(spo: SparsePauliOp, **kwargs) -> tuple[bool, float]:
    """Finds the extremal eigenvalue of the provided operator.

    This converts the provided operator to a sparse matrix whose minimal eigenvalue is required.

    .. note::
        The current implementation is definitely not optimized in terms of performance.

    Args:
        spo: the operator whose minimal eigenvalue to find.
        kwargs: additional keyword arguments for :func:`~scipy.sparse.linalg.eigsh`. When
            not specified otherwise, the following defaults will be used:

            * `tol`: 1e-6
            * `maxiter`: 500

            Other values will default to SciPy's default values.

    Returns:
        A pair indicating whether the Davidson algorithm has converged and the obtained minimal
        eigenvalue.
    """
    default_kwargs = {
        "tol": 1e-6,
        "maxiter": 500,
    }
    if "max_cycle" in kwargs and "maxiter" not in kwargs:
        kwargs["maxiter"] = kwargs.pop("max_cycle")
    for pyscf_only_arg in ("max_space", "lindep", "max_memory"):
        kwargs.pop(pyscf_only_arg, None)
    default_kwargs.update(kwargs)

    spmat = spo.to_matrix(sparse=True, force_serial=True)

    if spmat.shape[0] <= 2:
        eigenvalues = scipy.linalg.eigvalsh(spmat.toarray(), subset_by_index=(0, 0))
        return True, float(eigenvalues[0])

    try:
        eigenvalues = scipy.sparse.linalg.eigsh(
            spmat,
            k=1,
            which="SA",
            v0=_random_initial_guess(spmat.shape),
            return_eigenvectors=False,
            **default_kwargs,
        )
    except scipy.sparse.linalg.ArpackNoConvergence as exc:
        if exc.eigenvalues is None or len(exc.eigenvalues) == 0:
            return False, np.nan
        return False, float(exc.eigenvalues[0])

    return True, float(eigenvalues[0])


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
