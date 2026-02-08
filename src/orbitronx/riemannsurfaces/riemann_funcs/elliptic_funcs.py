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

from jax import config, jit, vmap, asarray, broadcast_arrays, isinf, array, where
from jax.numpy import complex128, pi, sin, sinh, cos, cosh, sqrt, exp, tan, tanh

from ..period_matrices.periods_genus1_second import periods_secondkind
from .complex_analysis import carlson_first, chop, jacobi_theta, qfrom

config.update("jax_enable_x64", True)

@jit
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
    omega1_arr = asarray(omega1, dtype=complex128)
    omega3_arr = asarray(omega3, dtype=complex128)
    
    # Broadcast arrays to same shape if needed
    omega1_bc, omega3_bc = broadcast_arrays(omega1_arr, omega3_arr)
    shape = omega1_bc.shape
    
    # Flatten for vectorized processing
    omega1_flat = omega1_bc.ravel()
    omega3_flat = omega3_bc.ravel()
    
    def roots_single(o1, o3):
        """Compute roots for a single pair of periods."""
        # Check for infinite periods
        inf_omega3 = isinf(o3)
        inf_omega1 = isinf(o1)
        
        # Handle special cases for infinite periods
        if inf_omega3:
            c = pi**2 / (12 * o1**2)
            return 2 * c, -c, -c
        elif inf_omega1:
            c = 1j * pi**2 / (12 * o3**2)
            return c, c, -2 * c
        else:
            # General case: evaluate Weierstrass P at half-periods
            e1 = chop(weierstrass_P(o1, o1, o3))
            e2 = chop(weierstrass_P(o3, o1, o3))
            e3 = chop(weierstrass_P(-o1 - o3, o1, o3))
            return e1, e2, e3
    
    # Vectorize over all input pairs
    e1_flat, e2_flat, e3_flat = vmap(roots_single)(omega1_flat, omega3_flat)
    
    # Reshape to original broadcast shape
    e1 = e1_flat.reshape(shape)
    e2 = e2_flat.reshape(shape)
    e3 = e3_flat.reshape(shape)
    
    # Return scalars if inputs were scalars
    if shape == ():
        return e1.item(), e2.item(), e3.item()
    return e1, e2, e3

@jit
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
    omega1_arr = asarray(omega1, dtype=complex128)
    omega3_arr = asarray(omega3, dtype=complex128)
    
    # Broadcast arrays to same shape if needed
    omega1_bc, omega3_bc = broadcast_arrays(omega1_arr, omega3_arr)
    shape = omega1_bc.shape
    
    # Flatten for vectorized processing
    omega1_flat = omega1_bc.ravel()
    omega3_flat = omega3_bc.ravel()
    
    def compute_single(o1, o3):
        """Compute invariants for a single pair of periods."""
        # Compute Weierstrass roots
        e1, e2, e3 = weierstrass_roots(o1, o3)
        
        # Compute invariants from roots
        g2 = 2 * (e1**2 + e2**2 + e3**2)
        g3 = 4 * e1 * e2 * e3
        
        return g2, g3
    
    # Vectorize over all input pairs
    g2_flat, g3_flat = vmap(compute_single)(omega1_flat, omega3_flat)
    
    # Reshape to original broadcast shape
    g2 = g2_flat.reshape(shape)
    g3 = g3_flat.reshape(shape)
    
    # Return scalars if inputs were scalars
    if shape == ():
        return g2.item(), g3.item()
    return g2, g3

@jit
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
    z_arr = asarray(z, dtype=complex128)
    omega1_arr = asarray(omega1, dtype=complex128)
    omega3_arr = asarray(omega3, dtype=complex128)
    
    # Broadcast arrays if needed
    z_bc, omega1_bc, omega3_bc = broadcast_arrays(z_arr, omega1_arr, omega3_arr)
    shape = z_bc.shape
    
    # Flatten for vectorized processing
    z_flat = z_bc.ravel()
    omega1_flat = omega1_bc.ravel()
    omega3_flat = omega3_bc.ravel()
    
    def compute_single(z_val, o1_val, o3_val, derivative=derivative):
        """Compute Weierstrass P for a single set of values."""
        
        # Check for infinite periods
        inf_omega3 = isinf(o3_val)
        inf_omega1 = isinf(o1_val)
        
        # -----------------------------
        # Special cases for infinite periods
        # -----------------------------
        if derivative == 0:
            c3 = pi**2 / (12 * o1_val**2)
            c1 = 1j * pi**2 / (12 * o3_val**2)
            
            # Value for omega3 infinite
            sin_term = sin(sqrt(3 * c3) * z_val)
            val_inf3 = -c3 + 3 * c3 * sin_term ** (-2)
            
            # Value for omega1 infinite
            sinh_term = sinh(sqrt(3 * c1) * z_val)
            val_inf1 = c1 + 3 * c1 * sinh_term ** (-2)
            
            # Choose based on which period is infinite
            val = where(inf_omega3, val_inf3, 
                           where(inf_omega1, val_inf1, 
                                    array(0.0 + 0.0j, dtype=complex128)))
            
            # Return if either period is infinite
            if any(inf_omega3 | inf_omega1):
                return val
        
        elif derivative == 1:
            c3 = pi**2 / (12 * o1_val**2)
            c1 = 1j * pi**2 / (12 * o3_val**2)
            
            # Derivative for omega3 infinite
            sqrt_3c3 = sqrt(3 * c3)
            sin_term = sin(sqrt_3c3 * z_val)
            cos_term = cos(sqrt_3c3 * z_val)
            val_inf3 = -2 * sqrt_3c3**3 * cos_term / sin_term**3
            
            # Derivative for omega1 infinite
            sqrt_3c1 = sqrt(3 * c1)
            sinh_term = sinh(sqrt_3c1 * z_val)
            cosh_term = cosh(sqrt_3c1 * z_val)
            val_inf1 = -2 * (3 * c1)**(3/2) * cosh_term / sinh_term**3
            
            val = where(inf_omega3, val_inf3,
                           where(inf_omega1, val_inf1,
                                    array(0.0 + 0.0j, dtype=complex128)))
            
            # Return if either period is infinite
            if any(inf_omega3 | inf_omega1):
                return val
        
        # -----------------------------
        # General case (neither period infinite)
        # -----------------------------
        tau = o3_val / o1_val
        nome = qfrom(tau)
        
        theta2 = jacobi_theta(2, 0.0, nome)  # uses JAX version
        theta3 = jacobi_theta(3, 0.0, nome)
        
        if derivative == 0:
            modified_z = z_val / (2 * o1_val)
            modified_in = pi * modified_z
            
            # Compute using theta functions
            theta4_val = jacobi_theta(4, modified_in, nome)
            theta1_val = jacobi_theta(1, modified_in, nome)
            
            val = (
                (
                    pi
                    * theta2
                    * theta3
                    * theta4_val
                    / theta1_val
                )
                ** 2
                - pi**2 * (theta2**4 + theta3**4) / 3
            ) / (4 * o1_val**2)
            
            return chop(val)
        
        elif derivative == 1:
            modified_in = pi * z_val / (2 * o1_val)
            
            # Compute numerator
            theta2_val = jacobi_theta(2, modified_in, nome)
            theta3_val = jacobi_theta(3, modified_in, nome)
            theta4_val = jacobi_theta(4, modified_in, nome)
            theta1_prime_0 = jacobi_theta(1, 0.0, nome, derivative=1)
            
            numerator = (
                theta2_val
                * theta3_val
                * theta4_val
                * theta1_prime_0 ** 3
            )
            
            # Compute denominator
            theta2_0 = jacobi_theta(2, 0.0, nome)
            theta3_0 = jacobi_theta(3, 0.0, nome)
            theta4_0 = jacobi_theta(4, 0.0, nome)
            theta1_val = jacobi_theta(1, modified_in, nome)
            
            denominator = (
                theta2_0
                * theta3_0
                * theta4_0
                * theta1_val ** 3
            )
            
            val = -(pi**3 / (4 * o1_val**3)) * (numerator / denominator)
            return chop(val)
        
        # -----------------------------
        # Higher derivatives (recursive)
        # -----------------------------
        elif derivative == 2:
            # Need g2 invariant
            g2 = invariants_from_periods(o1_val, o3_val)[0]
            P_val = compute_single(z_val, o1_val, o3_val)  # Call recursively for P(z)
            return 6 * P_val ** 2 - g2 / 2
        
        elif derivative == 3:
            P_val = compute_single(z_val, o1_val, o3_val)
            P_prime = compute_single(z_val, o1_val, o3_val, derivative=1)
            return 12 * P_val * P_prime
        
        elif derivative == 4:
            P_val = compute_single(z_val, o1_val, o3_val)
            P_prime = compute_single(z_val, o1_val, o3_val, derivative=1)
            P_double_prime = compute_single(z_val, o1_val, o3_val, derivative=2)
            return 12 * (P_prime ** 2 + P_val * P_double_prime)
        
        else:
            raise ValueError(f'"{derivative}" is not a valid derivative.')
    
    # Vectorize over all inputs
    results_flat = vmap(compute_single)(z_flat, omega1_flat, omega3_flat)
    
    # Reshape to original broadcast shape
    results = results_flat.reshape(shape)
    
    # Return scalar if input was scalar
    if results.shape == ():
        return results.item()
    return results

@jit
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
    z_arr = asarray(z, dtype=complex128)
    omega1_arr = asarray(omega1, dtype=complex128)
    omega3_arr = asarray(omega3, dtype=complex128)
    
    # Broadcast arrays if needed
    z_bc, omega1_bc, omega3_bc = broadcast_arrays(z_arr, omega1_arr, omega3_arr)
    shape = z_bc.shape
    
    # Flatten for vectorized processing
    z_flat = z_bc.ravel()
    omega1_flat = omega1_bc.ravel()
    omega3_flat = omega3_bc.ravel()
    
    def compute_single(z_val, o1_val, o3_val):
        """Compute inverse Weierstrass P for a single set of values."""
        # Compute roots (works for scalar or batch omega values)
        e1, e2, e3 = weierstrass_roots(o1_val, o3_val)
        
        # Apply Carlson integral: u = R_F(z - e1, z - e2, z - e3)
        result = carlson_first(z_val - e1, z_val - e2, z_val - e3)
        
        return chop(result)
    
    # Vectorize over all inputs
    results_flat = vmap(compute_single)(z_flat, omega1_flat, omega3_flat)
    
    # Reshape to original broadcast shape
    results = results_flat.reshape(shape)
    
    # Return scalar if input was scalar
    if results.shape == ():
        return results.item()
    return results

@jit
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
    z_arr = asarray(z, dtype=complex128)
    omega1_arr = asarray(omega1, dtype=complex128)
    omega3_arr = asarray(omega3, dtype=complex128)
    
    # Broadcast arrays to same shape if needed
    z_bc, omega1_bc, omega3_bc = broadcast_arrays(z_arr, omega1_arr, omega3_arr)
    shape = z_bc.shape
    
    # Flatten for vectorized processing
    z_flat = z_bc.ravel()
    omega1_flat = omega1_bc.ravel()
    omega3_flat = omega3_bc.ravel()
    
    def compute_single(z_val, o1_val, o3_val):
        """Compute Weierstrass zeta for a single set of values."""
        
        # Check for infinite periods
        inf_omega3 = isinf(o3_val)
        inf_omega1 = isinf(o1_val)
        
        # Case 1: omega3 is infinite
        if inf_omega3:
            c = pi**2 / (12 * o1_val**2)
            sqrt_3c = sqrt(3 * c)
            return c * z_val + sqrt_3c / tan(sqrt_3c * z_val)
        
        # Case 2: omega1 is infinite
        elif inf_omega1:
            c = 1j * pi**2 / (12 * o3_val**2)
            sqrt_3c = sqrt(3 * c)
            return -c * z_val + sqrt_3c / tanh(sqrt_3c * z_val)
        
        # Case 3: Normal case (neither period infinite)
        else:
            tau = o3_val / o1_val
            nome = qfrom(tau)
            
            # Get quasi-period η
            eta = periods_secondkind(o1_val, o3_val)[0]
            
            # Compute using theta functions
            v = pi * z_val / (2 * o1_val)
            theta1_val = jacobi_theta(1, v, nome)
            theta1_prime_val = jacobi_theta(1, v, nome, derivative=1)
            
            return eta * z_val / o1_val + pi * theta1_prime_val / (2 * o1_val * theta1_val)
    
    # Vectorize over all inputs
    results_flat = jax.vmap(compute_single)(z_flat, omega1_flat, omega3_flat)
    
    # Reshape to original broadcast shape
    results = results_flat.reshape(shape)
    
    # Return scalar if input was scalar
    if results.shape == ():
        return results.item()
    return results

from jax import jit

# Import the necessary functions


@jit
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
    z_arr = asarray(z, dtype=complex128)
    omega1_arr = asarray(omega1, dtype=complex128)
    omega3_arr = asarray(omega3, dtype=complex128)
    
    # Broadcast arrays to same shape if needed
    z_bc, omega1_bc, omega3_bc = broadcast_arrays(z_arr, omega1_arr, omega3_arr)
    shape = z_bc.shape
    
    # Flatten for vectorized processing
    z_flat = z_bc.ravel()
    omega1_flat = omega1_bc.ravel()
    omega3_flat = omega3_bc.ravel()
    
    def compute_single(z_val, o1_val, o3_val):
        """Compute Weierstrass sigma for a single set of values."""
        
        # Check for infinite periods
        inf_omega3 = isinf(o3_val)
        inf_omega1 = isinf(o1_val)
        
        # Case 1: omega3 is infinite
        if inf_omega3:
            c = pi**2 / (12 * o1_val**2)
            sqrt_3c = sqrt(3 * c)
            return (1.0 / sqrt_3c) * sin(sqrt_3c * z_val) * exp(c * z_val**2 / 2)
        
        # Case 2: omega1 is infinite
        elif inf_omega1:
            c = 1j * pi**2 / (12 * o3_val**2)
            sqrt_3c = sqrt(3 * c)
            return (1.0 / sqrt_3c) * sinh(sqrt_3c * z_val) * exp(-c * z_val**2 / 2)
        
        # Case 3: Normal case (neither period infinite)
        else:
            tau = o3_val / o1_val
            nome = qfrom(tau)
            
            # Get quasi-period η
            eta = periods_secondkind(o1_val, o3_val)[0]
            
            # Compute using theta functions
            v = pi * z_val / (2 * o1_val)
            theta1_v = jacobi_theta(1, v, nome)
            theta1_prime_0 = jacobi_theta(1, 0, nome, derivative=1)
            
            return (
                2 * o1_val / pi * 
                exp(eta * z_val**2 / (2 * o1_val)) * 
                theta1_v / theta1_prime_0
            )
    
    # Vectorize over all inputs
    results_flat = vmap(compute_single)(z_flat, omega1_flat, omega3_flat)
    
    # Reshape to original broadcast shape
    results = results_flat.reshape(shape)
    
    # Return scalar if input was scalar
    if results.shape == ():
        return results.item()
    return results
