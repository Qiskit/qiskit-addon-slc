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

"""Type stubs for the compiled Rust extension ``qiskit_addon_slc._accelerate``."""

import numpy as np
import numpy.typing as npt

def davidson_smallest(
    indptr: npt.NDArray[np.int64],
    indices: npt.NDArray[np.int64],
    data_re: npt.NDArray[np.float64],
    data_im: npt.NDArray[np.float64],
    diag_re: npt.NDArray[np.float64],
    diag_im: npt.NDArray[np.float64],
    seed_re: npt.NDArray[np.float64],
    seed_im: npt.NDArray[np.float64],
    dim: int,
    tol: float,
    max_cycle: int,
    max_space: int,
    lindep: float,
) -> tuple[bool, float]: ...
