#!/usr/bin/env python3
"""
A collection of procedures for computing elliptic functions.

Functions include the Weierstrass P, sigma, and zeta, as well as the inverse
of the weierstrass P function. There are also helper functions for conversions
between the elliptic invariants g2 and g3, the half periods omega1 and 3, and the
roots of the Weierstrass cubic (4z**3 - g2z - g3) e1, e2, and e3.

References
----------
[] T. mpmath development team, mpmath: a Python library for arbitrary-precision floating-point arithmetic
   (version 1.3.0) (2023), http://mpmath.org/.
[] M. Abramowitz and I. Stegun, Handbook of Mathematical Functions: With Formulas, Graphs,
   and Mathematical Tables, Applied mathematics series (Dover Publications, 1965).
[] D. Zwillinger and A. Jeffrey, Table of Integrals, Series, and Products
   (Elsevier Science, 2007).

"""


from jax import config, vmap
from jax.numpy import (
    array,
    asarray,
    broadcast_arrays,
    complex128,
    cos,
    cosh,
    exp,
    isinf,
    pi,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    where,
)

from .complex_analysis import carlson_first, chop, jacobi_theta, qfrom

config.update("jax_enable_x64", True)

def _to_c128(x):
    """Convert any numeric type (SymPy, mpmath, Python, JAX) to a JAX complex128 array."""
    try:
        return asarray(x, dtype=complex128)
    except TypeError:
        return asarray(complex(x), dtype=complex128)

def _eta_from_periods(omega1, omega3):
    """JAX-compatible computation of eta (first period of the second kind)."""
    tau = omega3 / omega1
    nome = qfrom(tau)
    eta = -(pi**2) * jacobi_theta(1, 0.0, nome, derivative=3) / (12 * omega1 * jacobi_theta(1, 0.0, nome, derivative=1))
    return eta

def weierstrass_roots(omega1, omega3):
    """
    Computes the roots of the Weierstrass cubic 4z**3 - g2z - g3 using the half
    periods <omega1> and <omega3>.

    Parameters
    ----------
    omega1 : complex
        The first half period within the period lattice.
    omega3 : complex
        The second half period within the period lattice.

    Returns
    -------
    e1 : complex
        The first root of the Weierstrass cubic.
    e2 : complex
        The second root of the Weierstrass cubic.
    e3 : complex
        The third root of the Weierstrass cubic.
    """
    omega1_arr = _to_c128(omega1)
    omega3_arr = _to_c128(omega3)

    # Broadcast arrays to same shape if needed
    omega1_bc, omega3_bc = broadcast_arrays(omega1_arr, omega3_arr)
    shape = omega1_bc.shape

    # Flatten for vectorized processing
    omega1_flat = omega1_bc.ravel()
    omega3_flat = omega3_bc.ravel()

    def roots_single(o1, o3):
        """Compute roots for a single pair of periods."""
        inf_omega3 = isinf(o3)
        inf_omega1 = isinf(o1)

        # Compute all three cases with safe values to avoid NaN
        safe_o1 = where(isinf(o1), array(1.0 + 0.0j, dtype=complex128), o1)
        safe_o3 = where(isinf(o3), array(1.0 + 0.0j, dtype=complex128), o3)

        # Case: omega3 infinite
        c3 = pi**2 / (12 * safe_o1**2)
        e1_inf3, e2_inf3, e3_inf3 = 2 * c3, -c3, -c3

        # Case: omega1 infinite
        c1 = 1j * pi**2 / (12 * safe_o3**2)
        e1_inf1, e2_inf1, e3_inf1 = c1, c1, -2 * c1

        # General case
        e1_gen = chop(weierstrass_P(safe_o1, safe_o1, safe_o3))
        e2_gen = chop(weierstrass_P(safe_o3, safe_o1, safe_o3))
        e3_gen = chop(weierstrass_P(-safe_o1 - safe_o3, safe_o1, safe_o3))

        e1 = where(inf_omega3, e1_inf3, where(inf_omega1, e1_inf1, e1_gen))
        e2 = where(inf_omega3, e2_inf3, where(inf_omega1, e2_inf1, e2_gen))
        e3 = where(inf_omega3, e3_inf3, where(inf_omega1, e3_inf1, e3_gen))

        return e1, e2, e3

    # Vectorize over all input pairs
    e1_flat, e2_flat, e3_flat = vmap(roots_single)(omega1_flat, omega3_flat)

    # Reshape to original broadcast shape
    e1 = e1_flat.reshape(shape)
    e2 = e2_flat.reshape(shape)
    e3 = e3_flat.reshape(shape)

    if shape == ():
        return e1, e2, e3
    return e1, e2, e3

def invariants_from_periods(omega1, omega3):
    """
    Computes the elliptic invariants g2 and g3 in the Weierstrass cubic 4z**3 - g2z - g3
    using the half periods <omega1> and <omega3>.

    Parameters
    ----------
    omega1 : complex
        The first half period within the period lattice.
    omega3 : complex
        The second half period within the period lattice.

    Returns
    -------
    g2 : complex
        A potentially complex number, the g2 in the polynomial above.
    g3 : complex
        A potentially complex number, the g3 in the polynomial above.
    """
    omega1_arr = _to_c128(omega1)
    omega3_arr = _to_c128(omega3)

    # Broadcast arrays to same shape if needed
    omega1_bc, omega3_bc = broadcast_arrays(omega1_arr, omega3_arr)
    shape = omega1_bc.shape

    # Flatten for vectorized processing
    omega1_flat = omega1_bc.ravel()
    omega3_flat = omega3_bc.ravel()

    def compute_single(o1, o3):
        """Compute invariants for a single pair of periods."""
        e1, e2, e3 = weierstrass_roots(o1, o3)
        g2 = 2 * (e1**2 + e2**2 + e3**2)
        g3 = 4 * e1 * e2 * e3
        return g2, g3

    # Vectorize over all input pairs
    g2_flat, g3_flat = vmap(compute_single)(omega1_flat, omega3_flat)

    # Reshape to original broadcast shape
    g2 = g2_flat.reshape(shape)
    g3 = g3_flat.reshape(shape)

    if shape == ():
        return g2, g3
    return g2, g3

def weierstrass_P(z, omega1, omega3, derivative=0):
    """
    Evaluates the Weierstrass P function at the value <z>, with half periods <omega1> and
    <omega3>. Optional derivatives can also be computed (a derivative of 0 means no derivative
    is computed).

    Parameters
    ----------
    z : complex
        The complex number to evaluate the Weierstrass P at.
    omega1 : complex
        The first half period within the period lattice.
    omega3 : complex
        The second half period within the period lattice.
    derivative : int, optional
        The order of derivative to compute (a derivative of 0 means no derivative is computed).

    Returns
    -------
    complex
        The value of the Weierstrass P function at <z> with half periods <omega1> and <omega3>.
    """
    # Convert inputs to JAX arrays
    z_arr = _to_c128(z)
    omega1_arr = _to_c128(omega1)
    omega3_arr = _to_c128(omega3)

    # Broadcast arrays if needed
    z_bc, omega1_bc, omega3_bc = broadcast_arrays(z_arr, omega1_arr, omega3_arr)
    shape = z_bc.shape

    # Flatten for vectorized processing
    z_flat = z_bc.ravel()
    omega1_flat = omega1_bc.ravel()
    omega3_flat = omega3_bc.ravel()

    def compute_single(z_val, o1_val, o3_val):
        """Compute Weierstrass P for a single set of values."""

        inf_omega3 = isinf(o3_val)
        inf_omega1 = isinf(o1_val)
        any_inf = inf_omega3 | inf_omega1

        # Safe values to avoid NaN in general case computation
        safe_o1 = where(isinf(o1_val), array(1.0 + 0.0j, dtype=complex128), o1_val)
        safe_o3 = where(isinf(o3_val), array(1.0 + 0.0j, dtype=complex128), o3_val)

        if derivative == 0:
            c3 = pi**2 / (12 * safe_o1**2)
            c1 = 1j * pi**2 / (12 * safe_o3**2)

            # Value for omega3 infinite
            sin_term = sin(sqrt(3 * c3) * z_val)
            val_inf3 = -c3 + 3 * c3 * sin_term ** (-2)

            # Value for omega1 infinite
            sinh_term = sinh(sqrt(3 * c1) * z_val)
            val_inf1 = c1 + 3 * c1 * sinh_term ** (-2)

            val_inf = where(inf_omega3, val_inf3, val_inf1)

            # General case
            tau = safe_o3 / safe_o1
            nome = qfrom(tau)
            theta2 = jacobi_theta(2, 0.0, nome)
            theta3 = jacobi_theta(3, 0.0, nome)
            modified_z = z_val / (2 * safe_o1)
            modified_in = pi * modified_z
            theta4_val = jacobi_theta(4, modified_in, nome)
            theta1_val = jacobi_theta(1, modified_in, nome)
            val_gen = (
                (pi * theta2 * theta3 * theta4_val / theta1_val) ** 2
                - pi**2 * (theta2**4 + theta3**4) / 3
            ) / (4 * safe_o1**2)

            return where(any_inf, val_inf, chop(val_gen))

        elif derivative == 1:
            c3 = pi**2 / (12 * safe_o1**2)
            c1 = 1j * pi**2 / (12 * safe_o3**2)

            sqrt_3c3 = sqrt(3 * c3)
            sin_term = sin(sqrt_3c3 * z_val)
            cos_term = cos(sqrt_3c3 * z_val)
            val_inf3 = -2 * sqrt_3c3**3 * cos_term / sin_term**3

            sqrt_3c1 = sqrt(3 * c1)
            sinh_term = sinh(sqrt_3c1 * z_val)
            cosh_term = cosh(sqrt_3c1 * z_val)
            val_inf1 = -2 * (3 * c1)**(3/2) * cosh_term / sinh_term**3

            val_inf = where(inf_omega3, val_inf3, val_inf1)

            # General case
            tau = safe_o3 / safe_o1
            nome = qfrom(tau)
            modified_in = pi * z_val / (2 * safe_o1)

            theta2_val = jacobi_theta(2, modified_in, nome)
            theta3_val = jacobi_theta(3, modified_in, nome)
            theta4_val = jacobi_theta(4, modified_in, nome)
            theta1_prime_0 = jacobi_theta(1, 0.0, nome, derivative=1)
            numerator = theta2_val * theta3_val * theta4_val * theta1_prime_0 ** 3

            theta2_0 = jacobi_theta(2, 0.0, nome)
            theta3_0 = jacobi_theta(3, 0.0, nome)
            theta4_0 = jacobi_theta(4, 0.0, nome)
            theta1_val = jacobi_theta(1, modified_in, nome)
            denominator = theta2_0 * theta3_0 * theta4_0 * theta1_val ** 3

            val_gen = -(pi**3 / (4 * safe_o1**3)) * (numerator / denominator)

            return where(any_inf, val_inf, chop(val_gen))

        elif derivative == 2:
            g2 = invariants_from_periods(o1_val, o3_val)[0]
            P_val = weierstrass_P(z_val, o1_val, o3_val, derivative=0)
            return 6 * P_val ** 2 - g2 / 2

        elif derivative == 3:
            P_val = weierstrass_P(z_val, o1_val, o3_val, derivative=0)
            P_prime = weierstrass_P(z_val, o1_val, o3_val, derivative=1)
            return 12 * P_val * P_prime

        elif derivative == 4:
            P_val = weierstrass_P(z_val, o1_val, o3_val, derivative=0)
            P_prime = weierstrass_P(z_val, o1_val, o3_val, derivative=1)
            P_double_prime = weierstrass_P(z_val, o1_val, o3_val, derivative=2)
            return 12 * (P_prime ** 2 + P_val * P_double_prime)

        else:
            raise ValueError(f'"{derivative}" is not a valid derivative.')

    # Vectorize over all inputs
    results_flat = vmap(compute_single)(z_flat, omega1_flat, omega3_flat)

    # Reshape to original broadcast shape
    results = results_flat.reshape(shape)

    if results.shape == ():
        return results
    return results

def inverse_weierstrass_P(z, omega1, omega3):
    """
    Evaluates the inverse Weierstrass P function at the value <z>, with half periods <omega1>
    and <omega3>.

    Note: the Weierstrass P function is not injective in the complex plane. Hence its inverse is
    not well defined.

    Parameters
    ----------
    z : complex
        The complex number to evaluate the inverse Weierstrass P at.
    omega1 : complex
        The first half period within the period lattice.
    omega3 : complex
        The second half period within the period lattice.

    Returns
    -------
    result : complex
        The value of the inverse Weierstrass P function at <z> with half periods <omega1>
        and <omega3>.
    """
    # Convert inputs to JAX arrays
    z_arr = _to_c128(z)
    omega1_arr = _to_c128(omega1)
    omega3_arr = _to_c128(omega3)

    # Broadcast arrays if needed
    z_bc, omega1_bc, omega3_bc = broadcast_arrays(z_arr, omega1_arr, omega3_arr)
    shape = z_bc.shape

    # Flatten for vectorized processing
    z_flat = z_bc.ravel()
    omega1_flat = omega1_bc.ravel()
    omega3_flat = omega3_bc.ravel()

    def compute_single(z_val, o1_val, o3_val):
        """Compute inverse Weierstrass P for a single set of values."""
        e1, e2, e3 = weierstrass_roots(o1_val, o3_val)
        result = carlson_first(z_val - e1, z_val - e2, z_val - e3)
        return chop(result)

    # Vectorize over all inputs
    results_flat = vmap(compute_single)(z_flat, omega1_flat, omega3_flat)

    # Reshape to original broadcast shape
    results = results_flat.reshape(shape)

    if results.shape == ():
        return results
    return results

def weierstrass_zeta(z, omega1, omega3):
    """
    Evaluates the Weierstrass zeta function at the value <z>, with half periods <omega1>
    and <omega3>.

    Parameters
    ----------
    z : complex
        The complex number to evaluate the Weierstrass zeta function at.
    omega1 : complex
        The first half period within the period lattice.
    omega3 : complex
        The second half period within the period lattice.

    Returns
    -------
    complex
        The value of the Weierstrass zeta function at <z> with half periods <omega1>
        and <omega3>.
    """
    z_arr = _to_c128(z)
    omega1_arr = _to_c128(omega1)
    omega3_arr = _to_c128(omega3)

    # Broadcast arrays to same shape if needed
    z_bc, omega1_bc, omega3_bc = broadcast_arrays(z_arr, omega1_arr, omega3_arr)
    shape = z_bc.shape

    # Flatten for vectorized processing
    z_flat = z_bc.ravel()
    omega1_flat = omega1_bc.ravel()
    omega3_flat = omega3_bc.ravel()

    def compute_single(z_val, o1_val, o3_val):
        """Compute Weierstrass zeta for a single set of values."""

        inf_omega3 = isinf(o3_val)
        inf_omega1 = isinf(o1_val)
        any_inf = inf_omega3 | inf_omega1

        safe_o1 = where(isinf(o1_val), array(1.0 + 0.0j, dtype=complex128), o1_val)
        safe_o3 = where(isinf(o3_val), array(1.0 + 0.0j, dtype=complex128), o3_val)

        # Case: omega3 infinite
        c3 = pi**2 / (12 * safe_o1**2)
        sqrt_3c3 = sqrt(3 * c3)
        val_inf3 = c3 * z_val + sqrt_3c3 / tan(sqrt_3c3 * z_val)

        # Case: omega1 infinite
        c1 = 1j * pi**2 / (12 * safe_o3**2)
        sqrt_3c1 = sqrt(3 * c1)
        val_inf1 = -c1 * z_val + sqrt_3c1 / tanh(sqrt_3c1 * z_val)

        val_inf = where(inf_omega3, val_inf3, val_inf1)

        # General case
        tau = safe_o3 / safe_o1
        nome = qfrom(tau)
        eta = _eta_from_periods(safe_o1, safe_o3)
        v = pi * z_val / (2 * safe_o1)
        theta1_val = jacobi_theta(1, v, nome)
        theta1_prime_val = jacobi_theta(1, v, nome, derivative=1)
        val_gen = eta * z_val / safe_o1 + pi * theta1_prime_val / (2 * safe_o1 * theta1_val)

        return where(any_inf, val_inf, val_gen)

    # Vectorize over all inputs
    results_flat = vmap(compute_single)(z_flat, omega1_flat, omega3_flat)

    # Reshape to original broadcast shape
    results = results_flat.reshape(shape)

    if results.shape == ():
        return results
    return results


def weierstrass_sigma(z, omega1, omega3):
    """
    Evaluates the Weierstrass sigma at the value <z>, with half periods <omega1>
    and <omega3>.

    Parameters
    ----------
    z : complex
        The complex number to evaluate the Weierstrass sigma function at.
    omega1 : complex
        The first half period within the period lattice.
    omega3 : complex
        The second half period within the period lattice.

    Returns
    -------
    complex
        The value of the Weierstrass sigma function at <z> with half periods <omega1>
        and <omega3>.
    """
    z_arr = _to_c128(z)
    omega1_arr = _to_c128(omega1)
    omega3_arr = _to_c128(omega3)

    # Broadcast arrays to same shape if needed
    z_bc, omega1_bc, omega3_bc = broadcast_arrays(z_arr, omega1_arr, omega3_arr)
    shape = z_bc.shape

    # Flatten for vectorized processing
    z_flat = z_bc.ravel()
    omega1_flat = omega1_bc.ravel()
    omega3_flat = omega3_bc.ravel()

    def compute_single(z_val, o1_val, o3_val):
        """Compute Weierstrass sigma for a single set of values."""

        inf_omega3 = isinf(o3_val)
        inf_omega1 = isinf(o1_val)
        any_inf = inf_omega3 | inf_omega1

        safe_o1 = where(isinf(o1_val), array(1.0 + 0.0j, dtype=complex128), o1_val)
        safe_o3 = where(isinf(o3_val), array(1.0 + 0.0j, dtype=complex128), o3_val)

        # Case: omega3 infinite
        c3 = pi**2 / (12 * safe_o1**2)
        sqrt_3c3 = sqrt(3 * c3)
        val_inf3 = (1.0 / sqrt_3c3) * sin(sqrt_3c3 * z_val) * exp(c3 * z_val**2 / 2)

        # Case: omega1 infinite
        c1 = 1j * pi**2 / (12 * safe_o3**2)
        sqrt_3c1 = sqrt(3 * c1)
        val_inf1 = (1.0 / sqrt_3c1) * sinh(sqrt_3c1 * z_val) * exp(-c1 * z_val**2 / 2)

        val_inf = where(inf_omega3, val_inf3, val_inf1)

        # General case
        tau = safe_o3 / safe_o1
        nome = qfrom(tau)
        eta = _eta_from_periods(safe_o1, safe_o3)
        v = pi * z_val / (2 * safe_o1)
        theta1_v = jacobi_theta(1, v, nome)
        theta1_prime_0 = jacobi_theta(1, 0, nome, derivative=1)
        val_gen = (
            2 * safe_o1 / pi *
            exp(eta * z_val**2 / (2 * safe_o1)) *
            theta1_v / theta1_prime_0
        )

        return where(any_inf, val_inf, val_gen)

    # Vectorize over all inputs
    results_flat = vmap(compute_single)(z_flat, omega1_flat, omega3_flat)

    # Reshape to original broadcast shape
    results = results_flat.reshape(shape)

    if results.shape == ():
        return results
    return results
