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
//! Each cycle takes the smallest Ritz pair `(θ, x)` of the subspace. A Ritz pair is a convergence
//! candidate when both the change of its value since the previous cycle and the norm of its residual
//! `A·x - θ·x` lie below the gate `max(tol, 64·ε·max(‖A‖, 1))`. The `ε‖A‖` term keeps the gate
//! reachable at a tiny `tol`, since a converged eigenvector still leaves a residual, and its Ritz value
//! still fluctuates, on that order, while the `tol` term keeps accuracy tracking the request when `tol`
//! dominates. `‖A‖` is estimated by the maximum absolute row sum. If the next subspace vector turns
//! out linearly dependent on the subspace (no more than a `lindep` fraction of its squared norm
//! survives orthogonalization, as in `pyscf.lib.davidson1`), the subspace is exhausted, and the
//! residual gate alone decides whether `(θ, x)` is a candidate.
//!
//! A candidate shows only that `(θ, x)` is *an* eigenpair, not the smallest one, and the
//! preconditioner pulls towards the eigenvalue nearest `θ`. Before accepting it, a short Lanczos
//! iteration orthogonal to `x` looks for a lower eigenvalue. If it finds one, its Ritz vector joins
//! the subspace and the iteration continues. This matters because the caller turns the eigenvalue
//! into an upper bound on a norm, which an interior eigenvalue would understate.
//!
//! The correction is Olsen's variant of the Jacobi-preconditioned residual, `M⁻¹r - ε·M⁻¹x` with `ε`
//! chosen to make it orthogonal to `x`, where `M` is the diagonal of `A - θ`. Each entry of `M` whose
//! magnitude falls below `floor = 1e-12 * max(diag_scale, 1)`, where `diag_scale` is the largest
//! `|diag[i]|`, is clamped up to `floor`, keeping its sign so it is not pushed the wrong way. This
//! floor is relative to the operator scale rather than to `tol`, because `tol` is a residual
//! threshold and clamping to it would corrupt operators with a near-zero diagonal. When the diagonal
//! is entirely zero the preconditioner has no useful shift to apply, so the plain residual is used.
//!
//! Before the subspace exceeds `max_space` vectors it is restarted from its lowest `max_space / 2`
//! Ritz vectors.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use num_complex::Complex64 as C64;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A Hermitian operator held in compressed-sparse-row form, borrowing the caller's arrays.
///
/// Only [`CsrOp::new`] constructs it, after checking that the arrays form a valid CSR matrix, so the
/// index casts below cannot go out of bounds.
struct CsrOp<'a> {
    indptr: &'a [i64],
    indices: &'a [i64],
    data: &'a [C64],
    dim: usize,
}

impl<'a> CsrOp<'a> {
    /// Validates the CSR arrays of a square `dim x dim` matrix, where `dim = indptr.len() - 1`.
    fn new(indptr: &'a [i64], indices: &'a [i64], data: &'a [C64]) -> PyResult<Self> {
        let dim = indptr.len().saturating_sub(1);
        if dim == 0 {
            return Err(PyValueError::new_err(
                "`indptr` must have at least two entries",
            ));
        }
        if indptr[0] != 0 {
            return Err(PyValueError::new_err("`indptr[0]` must be 0"));
        }
        // Together with `indptr[0] == 0` this also makes every entry non-negative.
        if !indptr.windows(2).all(|w| w[0] <= w[1]) {
            return Err(PyValueError::new_err("`indptr` must be non-decreasing"));
        }
        if indices.len() != data.len() || indptr[dim] as u64 != indices.len() as u64 {
            return Err(PyValueError::new_err(
                "`indices` and `data` must both have length `indptr[-1]`",
            ));
        }
        if indices.iter().any(|&j| j < 0 || j as u64 >= dim as u64) {
            return Err(PyValueError::new_err(
                "`indices` entries must be in `[0, len(indptr) - 1)`",
            ));
        }
        if !data.iter().all(|z| z.is_finite()) {
            return Err(PyValueError::new_err("`data` entries must be finite"));
        }
        Ok(Self {
            indptr,
            indices,
            data,
            dim,
        })
    }

    /// The index range into `indices` and `data` of the entries of `row`.
    fn row(&self, row: usize) -> std::ops::Range<usize> {
        self.indptr[row] as usize..self.indptr[row + 1] as usize
    }

    /// Returns `self @ x`.
    fn apply(&self, x: &DVector<C64>) -> DVector<C64> {
        DVector::from_fn(self.dim, |row, _| {
            self.row(row)
                .map(|k| self.data[k] * x[self.indices[k] as usize])
                .sum()
        })
    }

    /// Returns the diagonal, summing duplicate entries as `scipy.sparse` does.
    fn diagonal(&self) -> DVector<C64> {
        DVector::from_fn(self.dim, |row, _| {
            self.row(row)
                .filter(|&k| self.indices[k] as usize == row)
                .map(|k| self.data[k])
                .sum()
        })
    }

    /// Maximum absolute row sum, the induced infinity-norm, which for a Hermitian operator upper
    /// bounds the spectral radius. Used as a cheap operator-scale estimate.
    fn norm_bound(&self) -> f64 {
        (0..self.dim)
            .map(|row| self.row(row).map(|k| self.data[k].norm()).sum::<f64>())
            .fold(0.0_f64, f64::max)
    }
}

/// Diagonalizes the small `k x k` Hermitian Rayleigh-Ritz matrix, returning its eigenvalues in
/// ascending order together with the matching eigenvectors as columns.
///
/// Returns `None` if `projected` holds a non-finite entry (an operator entry near the overflow
/// threshold can produce one) or if the eigensolver does not converge, instead of handing NaN or
/// infinity to an iteration that may not terminate.
fn ritz_pairs(projected: &DMatrix<C64>) -> Option<(Vec<f64>, DMatrix<C64>)> {
    if !projected.iter().all(|z| z.is_finite()) {
        return None;
    }
    let max_niter = 1000 * projected.nrows().max(1);
    let eig = SymmetricEigen::try_new(projected.clone(), f64::EPSILON, max_niter)?;
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    let values = order.iter().map(|&i| eig.eigenvalues[i]).collect();
    let vectors = DMatrix::from_columns(
        &order
            .iter()
            .map(|&i| eig.eigenvectors.column(i))
            .collect::<Vec<_>>(),
    );
    Some((values, vectors))
}

/// Stacks a set of column vectors into a dense matrix.
fn columns_to_matrix<'a>(cols: impl IntoIterator<Item = &'a DVector<C64>>) -> DMatrix<C64> {
    DMatrix::from_columns(&cols.into_iter().map(|c| c.column(0)).collect::<Vec<_>>())
}

/// Iterates for the algebraically smallest eigenvalue of `op` from the unit vector `seed`, returning
/// `(converged, eigenvalue)`.
fn davidson(
    op: &CsrOp,
    seed: DVector<C64>,
    tol: f64,
    max_cycle: usize,
    max_space: usize,
    lindep: f64,
) -> (bool, f64) {
    let dim = op.dim;

    let diag = op.diagonal();
    let diag_scale = diag.iter().map(|z| z.norm()).fold(0.0_f64, f64::max);
    let precondition = diag_scale > 0.0;
    let floor = 1e-12 * diag_scale.max(1.0);

    // Residual and eigenvalue-change gates relative to the operator scale (see the module
    // documentation). Both share the floor, since the Ritz value of a converged vector still
    // fluctuates by rounding errors on the order of `ε‖A‖` between cycles.
    let anorm = op.norm_bound();
    let gate = tol.max(64.0 * f64::EPSILON * anorm.max(1.0));

    // Subspace basis vectors `s` and their images `A @ s`, grown one vector per cycle, together with
    // the Rayleigh-Ritz projection `sᴴ A s`, which is extended by one row and column per new vector
    // rather than recomputed, so a cycle costs `O(dim·k)` instead of `O(dim·k²)`.
    let mut images: Vec<DVector<C64>> = vec![op.apply(&seed)];
    let mut projected = DMatrix::from_element(1, 1, C64::new(seed.dotc(&images[0]).re, 0.0));
    let mut s: Vec<DVector<C64>> = vec![seed];

    let mut converged = false;
    let mut eigval = 0.0f64;
    let mut prev = f64::INFINITY;

    // A Lanczos Ritz value has to undercut the converged one by more than rounding noise before it
    // counts as evidence of a lower eigenvalue (see `lower_ritz_vector`).
    let verify_margin = gate.max(1e3 * f64::EPSILON * anorm.max(1.0));
    let mut rng = SplitMix64(0x5eed);

    for _ in 0..max_cycle {
        let s_mat = columns_to_matrix(&s);
        let images_mat = columns_to_matrix(&images);

        let Some((thetas, ys)) = ritz_pairs(&projected) else {
            break; // non-finite projection: report non-convergence
        };
        let theta = thetas[0];
        eigval = theta;

        let ritz = &s_mat * ys.column(0);
        let ritz_image = &images_mat * ys.column(0);
        let residual = &ritz_image - ritz.scale(theta);

        let residual_norm = residual.norm();
        let de = (theta - prev).abs();
        prev = theta;

        let correction = if residual_norm < gate && de < gate {
            // A small residual only shows that `(θ, x)` is *an* eigenpair. The preconditioner pulls
            // towards the eigenvalue nearest `θ`, so the iteration can settle on an interior one,
            // which would understate the norm. Accept only if no lower eigenvalue turns up.
            match lower_ritz_vector(op, &ritz, theta - verify_margin, anorm, &mut rng) {
                Verification::NoLower => {
                    converged = true;
                    break;
                }
                Verification::Lower(lower) => {
                    prev = f64::INFINITY;
                    lower
                }
                Verification::Inconclusive => break,
            }
        } else if precondition {
            // Apply the preconditioner, flooring the shift while keeping its sign.
            let shift = DVector::from_fn(dim, |i, _| {
                let d = diag[i] - C64::new(theta, 0.0);
                if d.norm() < floor {
                    C64::new(if d.re < 0.0 { -floor } else { floor }, 0.0)
                } else {
                    d
                }
            });
            // Olsen's correction: subtract the multiple of `M⁻¹x` that makes the correction
            // orthogonal to `x`. Plain `M⁻¹r` reproduces `x` itself wherever `M` matches `A - θ`
            // (e.g. on diagonal blocks), so the new direction vanishes and the iteration stalls.
            let m_r = residual.component_div(&shift);
            let m_x = ritz.component_div(&shift);
            let denom = ritz.dotc(&m_x);
            if denom.norm() > f64::EPSILON * m_x.norm() {
                &m_r - m_x * (ritz.dotc(&m_r) / denom)
            } else {
                m_r
            }
        } else {
            residual
        };

        // Thick restart: before the subspace exceeds `max_space`, shrink it to its lowest Ritz
        // vectors. Keeping several of them, rather than only the current estimate, retains the
        // components along the neighbouring eigenvectors; dropping those made the iteration stall,
        // or settle on an interior eigenvalue when the ground state had been projected out. Their
        // images follow from the same combination, so restarting costs no matrix-vector products,
        // and in the basis of Ritz vectors the projection is just the diagonal of their values.
        let restarted = s.len() >= max_space;
        if restarted {
            let keep = (max_space / 2).max(1);
            let kept = ys.columns(0, keep);
            let s_new = &s_mat * kept;
            let images_new = &images_mat * kept;
            s = s_new.column_iter().map(|c| c.into_owned()).collect();
            images = images_new.column_iter().map(|c| c.into_owned()).collect();
            projected = DMatrix::from_diagonal(&DVector::from_iterator(
                keep,
                thetas[..keep].iter().map(|&t| C64::new(t, 0.0)),
            ));
        }

        // Only a restart changes the basis since `s_mat` was stacked.
        let s_mat = if restarted {
            columns_to_matrix(&s)
        } else {
            s_mat
        };
        let next = match orthonormalize(&s_mat, correction, lindep) {
            Some(next) => next,
            // The subspace is exhausted, so no further cycle can move `θ`, and the change since
            // the previous cycle says nothing: the residual gate alone applies. A converged
            // estimate still has to pass the same check for a lower eigenvalue, whose
            // approximation then extends the subspace instead.
            None if residual_norm < gate => {
                match lower_ritz_vector(op, &ritz, theta - verify_margin, anorm, &mut rng) {
                    Verification::NoLower => {
                        converged = true;
                        break;
                    }
                    Verification::Lower(lower) => match orthonormalize(&s_mat, lower, lindep) {
                        Some(next) => {
                            prev = f64::INFINITY;
                            next
                        }
                        // A lower eigenvalue exists but cannot be pursued: report non-convergence.
                        None => break,
                    },
                    Verification::Inconclusive => break,
                }
            }
            None => break,
        };

        let image = op.apply(&next);
        let k = s.len();
        projected = projected.resize(k + 1, k + 1, C64::default());
        for (j, sj) in s.iter().enumerate() {
            let entry = sj.dotc(&image);
            projected[(j, k)] = entry;
            projected[(k, j)] = entry.conj();
        }
        projected[(k, k)] = C64::new(next.dotc(&image).re, 0.0);
        images.push(image);
        s.push(next);
    }

    (converged, eigval)
}

/// Orthogonalizes `v` against the orthonormal columns of `basis` and normalizes it. Returns `None` when
/// `v` is linearly dependent on `basis`, i.e. when no more than a `lindep` fraction of its squared
/// norm survives the orthogonalization (the criterion of `pyscf.lib.davidson1`).
fn orthonormalize(basis: &DMatrix<C64>, v: DVector<C64>, lindep: f64) -> Option<DVector<C64>> {
    let before = v.norm();
    let mut v = orthogonalized(basis, v);
    let after = v.norm();
    if after * after <= lindep * before * before {
        return None;
    }
    v.unscale_mut(after);
    Some(v)
}

/// Number of Lanczos steps [`lower_ritz_vector`] takes.
const VERIFY_STEPS: usize = 20;

/// Number of pseudo-random start vectors [`lower_ritz_vector`] tries before giving up.
const VERIFY_START_ATTEMPTS: usize = 3;

/// Outcome of [`lower_ritz_vector`].
enum Verification {
    /// No eigenvalue below the threshold turned up.
    NoLower,
    /// An eigenvalue below the threshold exists; this is the matching Lanczos Ritz vector.
    Lower(DVector<C64>),
    /// The check could not run, as every start vector was numerically dependent on `x`.
    Inconclusive,
}

/// Looks for an eigenvalue of `op` below `threshold` with a short Lanczos iteration from a
/// pseudo-random start vector, kept orthogonal to the converged Ritz vector `x`. Unlike the
/// preconditioned Davidson correction, Lanczos is not biased towards any part of the spectrum, so its
/// lowest Ritz value quickly approaches the smallest eigenvalue in the complement of `x`.
fn lower_ritz_vector(
    op: &CsrOp,
    x: &DVector<C64>,
    threshold: f64,
    anorm: f64,
    rng: &mut SplitMix64,
) -> Verification {
    if op.dim == 1 {
        return Verification::NoLower; // `x` spans the whole space
    }
    let x_mat = DMatrix::from_columns(&[x.column(0)]);
    let Some(mut v) = (0..VERIFY_START_ATTEMPTS).find_map(|_| {
        let start = DVector::from_fn(op.dim, |_, _| C64::new(rng.next_f64(), rng.next_f64()));
        orthonormalize(&x_mat, start, f64::EPSILON)
    }) else {
        return Verification::Inconclusive;
    };

    let mut basis: Vec<DVector<C64>> = Vec::new();
    let mut alphas: Vec<f64> = Vec::new();
    let mut betas: Vec<f64> = Vec::new();
    let steps = VERIFY_STEPS.min(op.dim.saturating_sub(1)).max(1);
    for step in 0..steps {
        let w = op.apply(&v);
        alphas.push(v.dotc(&w).re);
        basis.push(v);
        if step + 1 == steps {
            break;
        }
        // Full re-orthogonalization against the Lanczos basis and `x`.
        let w = orthogonalized(&columns_to_matrix(basis.iter().chain([x])), w);
        let beta = w.norm();
        if beta <= 1e-12 * anorm.max(1.0) {
            break; // invariant subspace: the Ritz values below are exact
        }
        betas.push(beta);
        v = w.unscale(beta);
    }

    let k = alphas.len();
    let tridiagonal = DMatrix::from_fn(k, k, |i, j| {
        let value = if i == j {
            alphas[i]
        } else if i + 1 == j || j + 1 == i {
            betas[i.min(j)]
        } else {
            0.0
        };
        C64::new(value, 0.0)
    });
    let Some((values, vectors)) = ritz_pairs(&tridiagonal) else {
        return Verification::Inconclusive;
    };
    if values[0] < threshold {
        Verification::Lower(columns_to_matrix(&basis) * vectors.column(0))
    } else {
        Verification::NoLower
    }
}

/// Orthogonalizes `v` against the orthonormal columns of `basis` (classical Gram-Schmidt with one
/// re-orthogonalization pass, numerically comparable to MGS).
fn orthogonalized(basis: &DMatrix<C64>, mut v: DVector<C64>) -> DVector<C64> {
    v -= basis * (basis.adjoint() * &v);
    v -= basis * (basis.adjoint() * &v);
    v
}

/// SplitMix64, a small deterministic generator for the start vectors of [`lower_ritz_vector`], so
/// that results stay reproducible without pulling in a random-number crate.
struct SplitMix64(u64);

impl SplitMix64 {
    /// Returns the next value, uniform in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Python entry point: runs [`davidson`] on the Hermitian CSR matrix `(data, indices, indptr)`
/// (`scipy.sparse.csr_matrix` order) from the start vector `seed`, which is normalized here.
///
/// The arrays are borrowed, not copied, so they must be C-contiguous. Raises `ValueError` if they do
/// not form a valid CSR matrix, if `seed` does not match it or is zero or non-finite, or if a solver
/// option is out of range. The solve itself runs with the GIL released.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, seed, *, tol, max_cycle, max_space, lindep))]
#[allow(clippy::too_many_arguments)]
fn davidson_smallest(
    py: Python<'_>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: PyReadonlyArray1<C64>,
    seed: PyReadonlyArray1<C64>,
    tol: f64,
    max_cycle: usize,
    max_space: usize,
    lindep: f64,
) -> PyResult<(bool, f64)> {
    let op = CsrOp::new(indptr.as_slice()?, indices.as_slice()?, data.as_slice()?)?;

    let seed = seed.as_slice()?;
    if seed.len() != op.dim {
        return Err(PyValueError::new_err(
            "`seed` must have length `len(indptr) - 1`",
        ));
    }
    let mut seed = DVector::from_column_slice(seed);
    let seed_norm = seed.norm();
    if !(seed_norm.is_finite() && seed_norm > 0.0) {
        return Err(PyValueError::new_err("`seed` must be non-zero and finite"));
    }
    // Rayleigh-Ritz and the orthogonalizations assume an orthonormal basis, starting with `seed`.
    seed.unscale_mut(seed_norm);

    if !(tol.is_finite() && tol > 0.0) {
        return Err(PyValueError::new_err("`tol` must be positive and finite"));
    }
    if max_space < 2 {
        return Err(PyValueError::new_err("`max_space` must be at least 2"));
    }
    if !(0.0..1.0).contains(&lindep) {
        return Err(PyValueError::new_err("`lindep` must be in `[0, 1)`"));
    }

    // The read-only borrows of the arrays keep them from being resized or freed meanwhile, and
    // `op` only holds slices of them.
    Ok(py.detach(|| davidson(&op, seed, tol, max_cycle, max_space, lindep)))
}

#[pymodule]
fn _accelerate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(davidson_smallest))?;
    Ok(())
}
