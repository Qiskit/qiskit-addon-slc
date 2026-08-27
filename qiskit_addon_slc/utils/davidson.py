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
from qiskit.quantum_info import SparsePauliOp


def get_extremal_eigenvalue(spo: SparsePauliOp, **kwargs) -> tuple[bool, float]:
    """Finds the extremal eigenvalue of the provided operator.

    This converts the provided operator to a sparse matrix whose minimal eigenvalue is required.

    The minimal eigenvalue is found using the Davidson algorithm with a diagonal preconditioner.
    The implementation relies only on ``numpy`` and ``scipy`` and, thus, has no platform-specific
    dependencies.

    .. note::
        The current implementation is definitely not optimized in terms of performance.

    Args:
        spo: the operator whose minimal eigenvalue to find.
        kwargs: additional keyword arguments for the Davidson algorithm. When not specified
            otherwise, the following defaults will be used:

            * `tol`: 1e-6
            * `max_cycle`: 500
            * `max_space`: 12
            * `lindep`: 1e-11

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

    tol = float(default_kwargs["tol"])
    max_cycle = int(default_kwargs["max_cycle"])
    max_space = int(default_kwargs["max_space"])
    lindep = float(default_kwargs["lindep"])

    dim = spmat.shape[0]
    diag = spmat.diagonal()

    # Diagonal preconditioner, clamping near-zero shifts to ``tol``.
    def precond(residual: np.ndarray, eigval: float) -> np.ndarray:
        shifted = diag - eigval
        shifted[np.abs(shifted) < tol] = tol
        return cast(np.ndarray, residual / shifted)

    subspace: np.ndarray = _random_initial_guess(spmat.shape).reshape(dim, 1)
    images: np.ndarray = spmat.dot(subspace)

    converged = False
    eigval = 0.0
    prev_eigval = np.inf
    for _ in range(max_cycle):
        # Project onto the subspace and diagonalize (Rayleigh-Ritz).
        projected = subspace.conj().T @ images
        projected = (projected + projected.conj().T) / 2.0
        sub_eigvals, sub_eigvecs = np.linalg.eigh(projected)

        eigval = float(sub_eigvals[0].real)
        ritz_vec = subspace @ sub_eigvecs[:, 0]
        ritz_image = images @ sub_eigvecs[:, 0]

        residual = ritz_image - eigval * ritz_vec
        if abs(eigval - prev_eigval) < tol or cast(float, np.linalg.norm(residual)) < tol:
            converged = True
            break
        prev_eigval = eigval

        correction = precond(residual, eigval)

        # Collapse the subspace once it reaches ``max_space``.
        if subspace.shape[1] >= max_space:
            subspace = ritz_vec.reshape(dim, 1)
            images = ritz_image.reshape(dim, 1)

        # Orthonormalize the correction against the subspace.
        for _ in range(2):
            correction = correction - subspace @ (subspace.conj().T @ correction)
        norm = cast(float, np.linalg.norm(correction))
        if norm < lindep:
            # Correction is linearly dependent; the subspace already spans the eigenvector.
            converged = True
            break
        correction = correction / norm

        subspace = np.hstack([subspace, correction.reshape(dim, 1)])
        images = np.hstack([images, spmat.dot(correction).reshape(dim, 1)])

    return converged, eigval


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
