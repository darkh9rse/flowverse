//! A small Levenberg–Marquardt solver for nonlinear least squares.
//!
//! Problem: find parameters p that minimise `Σ rᵢ(p)²` for some residual
//! functions rᵢ (model minus measurement). Models like "a vortex at
//! (x0, y0)" are nonlinear in their parameters, so there is no one-shot
//! formula. Instead we iterate:
//!
//! - **Gauss–Newton** approximates r as linear near p, using the Jacobian
//!   `J = ∂r/∂p`, and jumps to the minimum of that approximation by solving
//!   `(JᵀJ) δ = −Jᵀr`. It is fast near the answer but can overshoot wildly
//!   far from it.
//! - **Gradient descent** takes small safe steps downhill.
//!
//! Levenberg–Marquardt blends the two by solving
//! `(JᵀJ + λ·diag(JᵀJ)) δ = −Jᵀr`. A large λ gives small, cautious steps; a
//! small λ gives Gauss–Newton. λ shrinks after a successful step and grows
//! after a failed one. Reference: Press et al., *Numerical Recipes*, §15.5.
//!
//! The problems here have at most 5 parameters, so plain dense matrices and
//! Gaussian elimination are fine.

/// Minimise `Σ r²` starting from `p0`. `residuals(p, r)` must fill `r`
/// (length `m`) with the residuals for parameters `p`. Returns the best
/// parameters found.
pub(crate) fn levenberg_marquardt(
    residuals: &dyn Fn(&[f64], &mut [f64]),
    m: usize,
    p0: &[f64],
) -> Vec<f64> {
    const MAX_ITERATIONS: usize = 200;
    let k = p0.len();
    let mut p = p0.to_vec();
    let mut r = vec![0.0; m];
    residuals(&p, &mut r);
    let mut cost = sum_of_squares(&r);
    if !cost.is_finite() {
        return p;
    }

    let mut lambda = 1e-3;
    let mut jac = vec![0.0; m * k]; // row-major m×k
    let mut r_trial = vec![0.0; m];
    let mut p_trial = vec![0.0; k];

    for _ in 0..MAX_ITERATIONS {
        // Jacobian by forward differences: nudge one parameter at a time.
        // The step scales with the parameter, so a centre at x = 500 px and a
        // core radius of 0.5 px are both nudged sensibly.
        for j in 0..k {
            let h = 1e-6 * p[j].abs().max(1.0);
            p_trial.copy_from_slice(&p);
            p_trial[j] += h;
            residuals(&p_trial, &mut r_trial);
            for i in 0..m {
                jac[i * k + j] = (r_trial[i] - r[i]) / h;
            }
        }
        // Normal equations: A = JᵀJ (k×k), g = Jᵀr.
        let mut a = vec![0.0; k * k];
        let mut g = vec![0.0; k];
        for i in 0..m {
            let row = &jac[i * k..(i + 1) * k];
            for j in 0..k {
                g[j] += row[j] * r[i];
                for l in 0..k {
                    a[j * k + l] += row[j] * row[l];
                }
            }
        }

        // Try steps, raising λ until one lowers the cost.
        let old_cost = cost;
        loop {
            let mut damped = a.clone();
            for j in 0..k {
                // The small constant keeps the system solvable even when a
                // parameter has no effect at all (a zero column of J).
                damped[j * k + j] += lambda * (a[j * k + j] + 1e-12);
            }
            let neg_g: Vec<f64> = g.iter().map(|x| -x).collect();
            if let Some(step) = solve(&mut damped, &neg_g, k) {
                for j in 0..k {
                    p_trial[j] = p[j] + step[j];
                }
                residuals(&p_trial, &mut r_trial);
                let trial_cost = sum_of_squares(&r_trial);
                if trial_cost.is_finite() && trial_cost < cost {
                    p.copy_from_slice(&p_trial);
                    r.copy_from_slice(&r_trial);
                    cost = trial_cost;
                    lambda = (lambda * 0.1).max(1e-12);
                    break;
                }
            }
            lambda *= 10.0;
            if lambda > 1e12 {
                return p; // no downhill step exists: we are at a minimum
            }
        }
        // Converged when an iteration barely improves the cost.
        if old_cost - cost <= 1e-12 * old_cost {
            break;
        }
    }
    p
}

fn sum_of_squares(r: &[f64]) -> f64 {
    r.iter().map(|x| x * x).sum()
}

/// Solve the k×k system `A x = b` by Gaussian elimination with partial
/// pivoting (swapping in the row with the largest pivot keeps rounding
/// errors small). `A` is destroyed. Returns `None` if `A` is singular.
fn solve(a: &mut [f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut x = b.to_vec();
    for col in 0..k {
        let pivot =
            (col..k).max_by(|&i, &j| a[i * k + col].abs().total_cmp(&a[j * k + col].abs()))?;
        if a[pivot * k + col].abs() < 1e-300 {
            return None;
        }
        if pivot != col {
            for c in 0..k {
                a.swap(col * k + c, pivot * k + c);
            }
            x.swap(col, pivot);
        }
        for row in col + 1..k {
            let f = a[row * k + col] / a[col * k + col];
            for c in col..k {
                a[row * k + c] -= f * a[col * k + c];
            }
            x[row] -= f * x[col];
        }
    }
    // Back substitution, from the last row up.
    for row in (0..k).rev() {
        let mut s = x[row];
        for c in row + 1..k {
            s -= a[row * k + c] * x[c];
        }
        x[row] = s / a[row * k + row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_small_linear_system() {
        // 2x + y = 5, x + 3y = 10 → x = 1, y = 3. The zero pivot in the
        // first column of the second system forces a row swap.
        let x = solve(&mut [2.0, 1.0, 1.0, 3.0], &[5.0, 10.0], 2).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 3.0).abs() < 1e-12);
        let x = solve(&mut [0.0, 1.0, 1.0, 0.0], &[2.0, 7.0], 2).unwrap();
        assert!((x[0] - 7.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12);
        assert!(solve(&mut [1.0, 2.0, 2.0, 4.0], &[1.0, 1.0], 2).is_none());
    }

    #[test]
    fn fits_an_exponential_from_a_poor_start() {
        // Data y = 3·exp(-0.7 x); start from a = 1, b = 0.
        let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.25).collect();
        let res = |p: &[f64], r: &mut [f64]| {
            for (ri, &x) in r.iter_mut().zip(&xs) {
                *ri = p[0] * (p[1] * x).exp() - 3.0 * (-0.7 * x).exp();
            }
        };
        let p = levenberg_marquardt(&res, xs.len(), &[1.0, 0.0]);
        assert!(
            (p[0] - 3.0).abs() < 1e-6 && (p[1] + 0.7).abs() < 1e-6,
            "{p:?}"
        );
    }
}
