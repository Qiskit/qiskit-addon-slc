// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Davidson eigensolver for the algebraically smallest eigenvalue of a Hermitian sparse matrix.
//!
//! The dense linear algebra uses `nalgebra`, while the sparse operator is held as a plain CSR triple
//! whose matrix-vector product is applied directly, because `nalgebra-sparse` does not implement
//! sparse-times-dense multiplication for complex scalars.
//!
//! The stopping rule mirrors `pyscf.lib.davidson1`. A cycle converges when the Ritz value has settled
//! (`|Δθ| < tol`) and the residual `A·x - θ·x` of the current Ritz pair `(θ, x)` has norm below the
//! gate `max(tol, 64·ε·max(‖A‖, 1))`. The `ε‖A‖` term keeps the gate reachable at a tiny `tol`, since
//! a converged eigenvector still leaves a residual on that order, while the `tol` term keeps accuracy
//! tracking the request when `tol` dominates. `‖A‖` is estimated by the maximum absolute row sum. If
//! the correction vanishes after orthogonalization the subspace is exhausted, and that same residual
//! test then decides convergence, as in `davidson1` (`conv = dx_norm < toloose`).
//!
//! The Jacobi preconditioner divides each correction entry by the shift `diag[i] - θ`. A shift whose
//! magnitude falls below `floor = 1e-12 * max(diag_scale, 1)`, where `diag_scale` is the largest
//! `|diag[i]|`, is clamped up to `floor`, keeping its sign so it is not pushed the wrong way. This
//! floor is relative to the operator scale rather than to `tol`, because `tol` is a residual
//! threshold and clamping to it would corrupt operators with a near-zero diagonal. When the diagonal
//! is entirely zero the preconditioner has no useful shift to apply, so it is skipped altogether.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64 as C64;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A Hermitian operator held in compressed-sparse-row form.
struct CsrOp {
    indptr: Vec<i64>,
    indices: Vec<i64>,
    data: Vec<C64>,
    dim: usize,
}

impl CsrOp {
    /// Returns `self @ x`.
    fn apply(&self, x: &DVector<C64>) -> DVector<C64> {
        let mut y = DVector::zeros(self.dim);
        for row in 0..self.dim {
            let mut acc = C64::default();
            for k in self.indptr[row] as usize..self.indptr[row + 1] as usize {
                acc += self.data[k] * x[self.indices[k] as usize];
            }
            y[row] = acc;
        }
        y
    }

    /// Maximum absolute row sum, the induced infinity-norm, which for a Hermitian operator upper
    /// bounds the spectral radius. Used as a cheap operator-scale estimate.
    fn norm_bound(&self) -> f64 {
        (0..self.dim)
            .map(|row| {
                (self.indptr[row] as usize..self.indptr[row + 1] as usize)
                    .map(|k| self.data[k].norm())
                    .sum::<f64>()
            })
            .fold(0.0_f64, f64::max)
    }
}

/// Diagonalizes the small `k x k` Hermitian Rayleigh-Ritz matrix, returning the smallest eigenvalue
/// and its eigenvector.
fn smallest_eigenpair(projected: &DMatrix<C64>) -> (f64, DVector<C64>) {
    let eig = projected.clone().symmetric_eigen();
    let mut best = 0;
    for i in 1..eig.eigenvalues.len() {
        if eig.eigenvalues[i] < eig.eigenvalues[best] {
            best = i;
        }
    }
    (eig.eigenvalues[best], eig.eigenvectors.column(best).into())
}

/// Stacks a set of column vectors into a dense matrix.
fn columns_to_matrix(cols: &[DVector<C64>]) -> DMatrix<C64> {
    DMatrix::from_columns(&cols.iter().map(|c| c.column(0)).collect::<Vec<_>>())
}

/// Iterates for the algebraically smallest eigenvalue of `op`, returning `(converged, eigenvalue)`.
fn davidson(
    op: &CsrOp,
    diag: &DVector<C64>,
    seed: DVector<C64>,
    tol: f64,
    max_cycle: usize,
    max_space: usize,
    lindep: f64,
) -> (bool, f64) {
    let dim = op.dim;

    let diag_scale = diag.iter().map(|z| z.norm()).fold(0.0_f64, f64::max);
    let precondition = diag_scale > 0.0;
    let floor = 1e-12 * diag_scale.max(1.0);

    // Residual gate relative to the operator scale (see the module documentation).
    let anorm = op.norm_bound();
    let residual_tol = tol.max(64.0 * f64::EPSILON * anorm.max(1.0));

    // Subspace basis vectors `s` and their images `A @ s`, grown one vector per cycle.
    let mut images: Vec<DVector<C64>> = vec![op.apply(&seed)];
    let mut s: Vec<DVector<C64>> = vec![seed];

    let mut converged = false;
    let mut eigval = 0.0f64;
    let mut prev = f64::INFINITY;

    for _ in 0..max_cycle {
        let s_mat = columns_to_matrix(&s);
        let images_mat = columns_to_matrix(&images);

        // Rayleigh-Ritz: project the operator onto the subspace and Hermitize.
        let projected = s_mat.adjoint() * &images_mat;
        let projected = (&projected + projected.adjoint()).scale(0.5);
        let (theta, y) = smallest_eigenpair(&projected);
        eigval = theta;

        let ritz = &s_mat * &y;
        let ritz_image = &images_mat * &y;
        let residual = &ritz_image - ritz.scale(theta);

        let residual_norm = residual.norm();
        let de = (theta - prev).abs();
        prev = theta;
        if residual_norm < residual_tol && de < tol {
            converged = true;
            break;
        }

        // Apply the preconditioner, flooring the shift while keeping its sign.
        let mut correction = residual;
        if precondition {
            for i in 0..dim {
                let mut d = diag[i] - C64::new(theta, 0.0);
                if d.norm() < floor {
                    let sign = if d.re < 0.0 { -floor } else { floor };
                    d = C64::new(sign, 0.0);
                }
                correction[i] /= d;
            }
        }

        // Collapse the subspace to the current best estimate before it exceeds `max_space`.
        if s.len() >= max_space {
            s = vec![ritz.clone()];
            images = vec![ritz_image];
        }

        // Classical Gram-Schmidt with one re-orthogonalization pass (numerically comparable to MGS).
        let s_mat = columns_to_matrix(&s);
        correction -= &s_mat * (s_mat.adjoint() * &correction);
        correction -= &s_mat * (s_mat.adjoint() * &correction);
        let cnorm = correction.norm();
        if cnorm < lindep {
            converged = residual_norm < residual_tol;
            break;
        }
        correction.unscale_mut(cnorm);

        images.push(op.apply(&correction));
        s.push(correction);
    }

    (converged, eigval)
}

/// Python entry point: builds a [`CsrOp`] from the split real/imaginary arrays and runs [`davidson`].
///
/// Raises `ValueError` if the CSR arrays, `dim`, `diag`, or `seed` are inconsistent.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn davidson_smallest(
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data_re: PyReadonlyArray1<f64>,
    data_im: PyReadonlyArray1<f64>,
    diag_re: PyReadonlyArray1<f64>,
    diag_im: PyReadonlyArray1<f64>,
    seed_re: PyReadonlyArray1<f64>,
    seed_im: PyReadonlyArray1<f64>,
    dim: usize,
    tol: f64,
    max_cycle: usize,
    max_space: usize,
    lindep: f64,
) -> PyResult<(bool, f64)> {
    fn to_complex(re: &[f64], im: &[f64]) -> Vec<C64> {
        re.iter().zip(im).map(|(a, b)| C64::new(*a, *b)).collect()
    }

    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let data = to_complex(data_re.as_slice()?, data_im.as_slice()?);
    let diag = to_complex(diag_re.as_slice()?, diag_im.as_slice()?);
    let seed = to_complex(seed_re.as_slice()?, seed_im.as_slice()?);

    if dim == 0 {
        return Err(PyValueError::new_err("`dim` must be positive"));
    }
    if max_space < 2 {
        return Err(PyValueError::new_err("`max_space` must be at least 2"));
    }
    if indptr.len() != dim + 1 {
        return Err(PyValueError::new_err("`indptr` must have length `dim + 1`"));
    }
    if !indptr.windows(2).all(|w| w[0] <= w[1]) {
        return Err(PyValueError::new_err("`indptr` must be non-decreasing"));
    }
    if indptr[dim] as usize != indices.len() || indices.len() != data.len() {
        return Err(PyValueError::new_err(
            "`indptr[dim]` must equal `indices.len()` and `data.len()`",
        ));
    }
    if indices.iter().any(|&j| j < 0 || j as usize >= dim) {
        return Err(PyValueError::new_err(
            "`indices` entries must be in `[0, dim)`",
        ));
    }
    if diag.len() != dim || seed.len() != dim {
        return Err(PyValueError::new_err(
            "`diag` and `seed` must have length `dim`",
        ));
    }

    let op = CsrOp {
        indptr: indptr.to_vec(),
        indices: indices.to_vec(),
        data,
        dim,
    };
    let diag = DVector::from_vec(diag);
    let seed = DVector::from_vec(seed);

    let (conv, ev) = davidson(&op, &diag, seed, tol, max_cycle, max_space, lindep);
    Ok((conv, ev))
}

#[pymodule]
fn _accelerate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(davidson_smallest))?;
    Ok(())
}
