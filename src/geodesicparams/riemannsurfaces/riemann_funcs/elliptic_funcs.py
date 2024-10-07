#!/usr/bin/env python3
"""
A collection of procedures for computing elliptic functions.

Functions include the Weierstrass P, sigma, and zeta, as well as the inverse
of the weierstrass P function. There are also helper functions for conversions
between the elliptic invariants g2 and g3, the half periods omega1 and 3, and the
roots of the Weierstrass cubic (4z**3 - g2z - g3) e1, e2, and e3.

"""

from mpmath import jtheta, pi, exp, sqrt, qfrom, elliprf, chop, mpc, sin, cos, isinf, sinh, cosh, cot, coth

from ..period_matrices.periods_genus1_second import periods_secondkind

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

    # Check for special values
    if isinf(omega3):
        c = pi**2 / (12 * omega1**2)
        return 2 * c, -c , -c
    elif isinf(omega1):
        c = 1j * pi**2 / (12 * omega3**2)
        return c, c, -2 * c
    # Normal result
    else:
        e1 = chop(weierstrass_P(omega1, omega1, omega3))
        e2 = chop(weierstrass_P(omega3, omega1, omega3))
        e3 = chop(weierstrass_P(-omega1 - omega3, omega1, omega3))

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

    e1, e2, e3 = weierstrass_roots(omega1, omega3)
    g2 = 2 * (e1**2 + e2**2 + e3**2)
    g3 = 4 * e1 * e2 * e3
    return g2, g3

def weierstrass_P(z, omega1, omega3, derivative = 0):
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

    z = mpc(z)

    # Special value derivatives
    if derivative == 0:
        if isinf(omega3):
            c = pi**2 / (12 * omega1**2)
            return -c + 3 * c * (sin((3 * c)**(1/2) * z))**(-2)
        elif isinf(omega1):
            c = 1j * pi**2 / (12 * omega3**2)
            return c + 3 * c * (sinh((3 * c)**(1/2) * z))**(-2)
    elif derivative == 1:
        if isinf(omega3):
            c = pi**2 / (12 * omega1**2)
            return -2 * sqrt(3 * c)**3 * cos(sqrt(3 * c) * z) / sin(sqrt(3 * c) * z)**3
        elif isinf(omega1):
            c = 1j * pi**2 / (12 * omega3**2)
            return -2 * (3 * c)**(3/2) * cosh((3 * c)**(1/2) * z) / sinh((3 * c)**(1/2) * z)**3
    elif derivative == 2:
        g2 = invariants_from_periods(omega1, omega3)[0]
        return 6 * weierstrass_P(z, omega1, omega3)**2 - g2/2
    elif derivative == 3:
        return 12 * weierstrass_P(z, omega1, omega3) * weierstrass_P(z, omega1, omega3, derivative = 1)
    elif derivative == 4:
        return 12 * (weierstrass_P(z, omega1, omega3, derivative = 1)**2 + weierstrass_P(z, omega1, omega3) * weierstrass_P(z, omega1, omega3, derivative = 2))

    # Specific values used in the computation 
    tau = omega3 / omega1
    nome = qfrom(tau = tau)
    theta2 = jtheta(2, 0, nome)
    theta3 = jtheta(3, 0, nome)

    # Normal values
    if derivative == 0:
        modified_z = z / omega1 / 2
        modified_in = pi * modified_z
        return chop(((pi * theta2 * theta3 * jtheta(4, modified_in, nome) / jtheta(1, modified_in, nome))**2 - pi**2 * (theta2**4 + theta3**4) / 3) / omega1 /omega1 / 4)
    elif derivative == 1:
        modified_in = pi*z / (2 * omega1)
        numerator = jtheta(2, modified_in, nome) * jtheta(3, modified_in, nome) * jtheta(4, modified_in, nome) * jtheta(1, 0, nome, 1)**3
        denominator = jtheta(2, 0, nome) * jtheta(3, 0, nome) * jtheta(4, 0, nome) * jtheta(1, modified_in, nome)**3
        return chop(- (pi**3 / (4 * omega1**3)) * (numerator/denominator))
    elif derivative == 2:
        g2 = invariants_from_periods(omega1, omega3)[0]
        return 6 * weierstrass_P(z, omega1, omega3)**2 - g2/2
    elif derivative == 3:
        return 12 * weierstrass_P(z, omega1, omega3) * weierstrass_P(z, omega1, omega3, derivative = 1)
    elif derivative == 4:
        return 12 * (
        weierstrass_P(z, omega1, omega3, derivative = 1)**2 + weierstrass_P(z, omega1, omega3) * weierstrass_P(z, omega1, omega3, derivative = 2)
        )
    else:
        raise ValueError(f'"{derivative}" is not a valid derivative.')

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

    z = mpc(z)

    e1, e2, e3 = weierstrass_roots(omega1, omega3)
    result = chop(elliprf(z - e1, z - e2, z - e3))

    return result

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

    z = mpc(z)

    # Special values
    if isinf(omega3):
        c = pi**2 / (12 * omega1**2)
        return c * z + (3 * c)**(1/2) * cot((3 * c)**(1/2) * z)
    elif isinf(omega1):
        c = 1j * pi**2 / (12 * omega3**2)
        return - c * z + (3 * c)**(1/2) * coth((3 * c)**(1/2) * z)
    # Normal value
    else:
        tau = omega3/omega1
        nome = qfrom(tau = tau)
        eta = periods_secondkind(omega1, omega3)[0]
        v = (pi*z)/(2*omega1)
        return eta * z/omega1 + pi * jtheta(1, v, nome, 1) / (2 * omega1 * jtheta(1, v, nome))

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

    z = mpc(z)

    # Special values
    if isinf(omega3):
        c = pi**2 / (12 * omega1**2)
        return (3 * c)**(-1/2) * sin((3 * c)**(1/2) * z) * exp(c * z**2 / 2)
    elif isinf(omega1):
        c = 1j * pi**2 / (12 * omega3**2)
        return (3 * c)**(-1/2) * sinh((3 * c)**(1/2) * z) * exp(- c * z**2 / 2)
    # Normal value
    else:
        tau = omega3/omega1
        nome = qfrom(tau = tau)
        eta = periods_secondkind(omega1, omega3)[0]
        v = pi * z /(2*omega1)
        return 2*omega1 / pi * exp(eta * z**2 / (2*omega1)) * jtheta(1, v, nome) / jtheta(1, 0, nome, 1)
