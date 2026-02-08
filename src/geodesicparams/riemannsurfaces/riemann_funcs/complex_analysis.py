from jax import config, jit, lax, vmap, grad, asarray, broadcast_arrays, isnan, isinf, logical_and
from jax.numpy import (arange, complex128, cos, exp, inf, log, maximum,
                       nan, pi, sin, sqrt, sum, where, int32)

config.update("jax_enable_x64", True)

@jit
def jacobi_theta(n, z, q, derivative=0, n_terms=50):
    """
    Jacobi theta function θ_n(z, q) with optional derivative.
    Fully GPU-optimized with JAX. Supports scalar or batched inputs, real or complex.
    
    Parameters
    ----------
    n : int (1, 2, 3, or 4)
        Which Jacobi theta function to compute.
    z : array-like, complex or real
        Argument of the theta function.
    q : complex or real
        Nome parameter (|q| < 1).
    derivative : int, optional
        Order of derivative (default 0).
    n_terms : int, optional
        Number of terms in the series expansion (default 50).
    
    Returns
    -------
    complex or array
        Value(s) of the theta function.
    """
    z = asarray(z, dtype=complex128)
    q = asarray(q, dtype=complex128)
    
    if n not in [1, 2, 3, 4]:
        raise ValueError(f"n must be 1, 2, 3, or 4, got {n}")
    
    q_abs = abs(q)
    if any(q_abs >= 1.0):
        print(f"Warning: |q| = {q_abs} may not converge (requires |q| < 1)")
    
    z_is_scalar = z.ndim == 0
    if z_is_scalar:
        z = z.reshape(1)
    
    k_values = arange(n_terms, dtype=int32)
    
    if n in [1, 2]:
        # For θ₁ and θ₂: k + 0.5
        exponents = (k_values + 0.5) ** 2
    else:
        # For θ₃ and θ₄: k (starting from 1 for sum)
        k_values_nonzero = arange(1, n_terms + 1, dtype=int32)
        exponents = k_values_nonzero ** 2
    
    log_q = log(q)
    q_powers = exp(exponents * log_q)
    
    @jit
    def compute_theta(z_val):
        """Compute theta function for a single z value."""
        if n == 1:
            # θ₁(z, q) = 2∑_{k=0}^∞ (-1)^k q^{(k+0.5)^2} sin((2k+1)z)
            signs = (-1) ** k_values
            angles = (2 * k_values + 1) * z_val
            terms = 2 * signs * q_powers * sin(angles)
            return sum(terms)
        
        elif n == 2:
            # θ₂(z, q) = 2∑_{k=0}^∞ q^{(k+0.5)^2} cos((2k+1)z)
            angles = (2 * k_values + 1) * z_val
            terms = 2 * q_powers * cos(angles)
            return sum(terms)
        
        elif n == 3:
            # θ₃(z, q) = 1 + 2∑_{k=1}^∞ q^{k^2} cos(2kz)
            angles = 2 * k_values_nonzero * z_val
            terms = 2 * q_powers * cos(angles)
            return 1.0 + sum(terms)
        
        else:  # n == 4
            # θ₄(z, q) = 1 + 2∑_{k=1}^∞ (-1)^k q^{k^2} cos(2kz)
            signs = (-1) ** k_values_nonzero
            angles = 2 * k_values_nonzero * z_val
            terms = 2 * signs * q_powers * cos(angles)
            return 1.0 + sum(terms)
    
    if derivative > 0:
        def theta_with_grad(z_val, deriv_order):
            if deriv_order == 0:
                return compute_theta(z_val)
            else:
                f = compute_theta
                for _ in range(deriv_order):
                    f = grad(f, holomorphic=True)
                return f(z_val)
        
        result = vmap(lambda z_val: theta_with_grad(z_val, derivative))(z)
    else:
        result = vmap(compute_theta)(z)
    
    if z_is_scalar:
        return result[0]
    else:
        return result

def qfrom(tau):
    """
    Compute the nome q from the half-period ratio tau.

    Parameters
    ----------
    tau : complex or array-like
        The half-period ratio omega3 / omega1.

    Returns
    -------
    complex or ndarray
        The nome q = exp(i * pi * tau)
    """
    tau = asarray(tau, dtype=complex128)
    return exp(1j * pi * tau)


def chop(x, tol=1e-15):
    """
    Set values very close to zero to exactly zero (like mpmath.chop).
    """
    return where(abs(x) < tol, 0.0 + 0.0j, x)

@jit
def carlson_first(x, y, z, tol=1e-12, max_iter=60):
    """
    JAX implementation of Carlson symmetric elliptic integral R_F(x,y,z).
    Supports scalar or array inputs (broadcastable), real or complex.
    
    Returns scalar if inputs scalar, otherwise an array of same broadcast shape.
    """
    x_in = asarray(x, dtype=complex128)
    y_in = asarray(y, dtype=complex128)
    z_in = asarray(z, dtype=complex128)
    
    xa, ya, za = broadcast_arrays(x_in, y_in, z_in)
    shape = xa.shape
    
    flat_x = xa.ravel()
    flat_y = ya.ravel()
    flat_z = za.ravel()
    
    is_nan = isnan(flat_x) | isnan(flat_y) | isnan(flat_z)
    is_inf = isinf(flat_x) | isinf(flat_y) | isinf(flat_z)
    
    zero_x = abs(flat_x) < tol
    zero_y = abs(flat_y) < tol
    zero_z = abs(flat_z) < tol
    
    # Two or more zeros -> infinite result
    two_zero_mask = (zero_x & zero_y) | (zero_y & zero_z) | (zero_z & zero_x)
    
    # All three zeros -> special case
    three_zero_mask = zero_x & zero_y & zero_z
    
    X = flat_x
    Y = flat_y
    Z = flat_z
    
    safe_X = where(zero_x, tol, X)
    safe_Y = where(zero_y, tol, Y)
    safe_Z = where(zero_z, tol, Z)
    
    A = (safe_X + safe_Y + safe_Z) / 3.0
    
    def cond_fn(state):
        it, xc, yc, zc, ac = state
        dx = abs(ac - xc)
        dy = abs(ac - yc)
        dz = abs(ac - zc)
        max_diff = maximum(maximum(dx, dy), dz)
        scale = maximum(abs(ac), 1.0)
        not_converged = max_diff > (tol * scale)
        any_not_converged = any(not_converged)
        return (it < max_iter) & any_not_converged
    
    def body_fn(state):
        it, xc, yc, zc, ac = state
        sqrt_X = sqrt(xc)
        sqrt_Y = sqrt(yc)
        sqrt_Z = sqrt(zc)
        lam = sqrt_X * sqrt_Y + sqrt_Y * sqrt_Z + sqrt_Z * sqrt_X
        
        xn = (xc + lam) / 4.0
        yn = (yc + lam) / 4.0
        zn = (zc + lam) / 4.0
        an = (xn + yn + zn) / 3.0
        
        return (it + 1, xn, yn, zn, an)
    
    init_state = (0, safe_X, safe_Y, safe_Z, A)
    
    final_state = lax.while_loop(cond_fn, body_fn, init_state)
    _, xf, yf, zf, af = final_state
    
    # Compute series expansion for R_F
    # Use normalized deviations for better numerical stability
    rx = (af - xf) / af
    ry = (af - yf) / af
    rz = (af - zf) / af
    
    E2 = rx * ry - rz**2
    E3 = rx * ry * rz
    
    # Carlson's series expansion for R_F
    # R_F ≈ 1/sqrt(A) * [1 - E2/10 + E3/14 + E2²/24 - 3*E2*E3/44 - E2³/208 + E3²/104 + ...]
    series = (1.0 - E2/10.0 + E3/14.0 + 
              E2**2/24.0 - 3.0*E2*E3/44.0 - 
              E2**3/208.0 + E3**2/104.0 + 
              3.0*E2**2*E3/136.0)
    
    rf_flat = series / sqrt(af)
    
    # 1. Two or more zeros: R_F → ∞
    rf_flat = where(two_zero_mask, inf + 0.0j, rf_flat)
    
    
    # 3. any infinite input (but not two zeros): R_F → 0
    rf_flat = where(is_inf & ~two_zero_mask, 0.0 + 0.0j, rf_flat)
    
    # 4. any NaN input: propagate NaN
    rf_flat = where(is_nan, nan + 1j*nan, rf_flat)
    
    rf = rf_flat.reshape(shape)
    
    if rf.shape == ():
        return rf.item()
    return rf

@jit
def agm(a, b=1.0, tol=1e-15, max_iter=200):
    """
    Compute the arithmetic-geometric mean of a and b using JAX.
    
    Parameters
    ----------
    a : float or complex or array-like
        First number.
    b : float or complex or array-like, optional
        Second number (default = 1).
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.
    
    Returns
    -------
    float or complex or array
        Arithmetic-geometric mean of (a, b).
    """
    a_in = asarray(a, dtype=complex128)
    b_in = asarray(b, dtype=complex128)
    
    a_arr, b_arr = broadcast_arrays(a_in, b_in)
    
    a_flat = a_arr.ravel()
    b_flat = b_arr.ravel()
    
    def agm_single(a0, b0):
        """AGM for a single pair (a0, b0)."""
        
        def cond_fn(state):
            a, b, i, converged = state
            rel_diff = abs(a - b) / (0.5 * (abs(a) + abs(b)) + 1e-30)
            still_not_converged = rel_diff > tol
            continue_iterating = i < max_iter
            return logical_and(still_not_converged, continue_iterating)
        
        def body_fn(state):
            a, b, i, converged = state
            an = 0.5 * (a + b)
            
            prod = a * b
            bn = sqrt(prod)
            
            return an, bn, i + 1, converged
        
        init_state = (a0, b0, 0, False)
        
        final_state = lax.while_loop(cond_fn, body_fn, init_state)
        a_final, b_final, iters, _ = final_state
        
        return 0.5 * (a_final + b_final), iters
    
    results, iterations = vmap(agm_single)(a_flat, b_flat)
    
    results = results.reshape(a_arr.shape)
    
    if results.shape == ():
        return results.item()
    return results
