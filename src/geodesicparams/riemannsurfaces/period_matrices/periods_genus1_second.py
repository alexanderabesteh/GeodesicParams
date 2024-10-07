#!/usr/bin/env python3
"""
Procedures for computing the periods of the second kind (the integrals of the canonical
meromorphic differentials for a genus 1 Riemann surface).

NOTE: fix some things here like the precision.

"""

from mpmath import jtheta, pi, qfrom, mpc, isinf

def periods_secondkind(omega1, omega3):
    """
    Computes the periods of the second kind (i.e. the integrals of the canonical meromorphic 
    differentials for a genus 1 Riemann surface) from the half periods <omega1> and <omega3>.

    Parameters
    ----------
    omega1 : complex
        The first half period within the period lattice.
    omega3 : complex
        The second half period within the period lattice.

    Returns
    -------
    eta : complex
        The first period of the second kind.
    eta_prime : complex
        The second period of the second kind.

    """

    # Special cases
    if isinf(omega3):
        c = pi**2 / 12 * omega1**2
        return c * omega1, mpc(0, "inf")
    elif isinf(omega1):
        c = 1j * pi **2 / 12 * omega3**2
        return mpc("-inf", 0), - c * omega3
    else:
        # Normal result
        tau = omega3/omega1
        nome = qfrom(tau = tau)
        eta = - pi**2 * jtheta(1, 0, nome, 3) / (12 * omega1 * jtheta(1, 0, nome, 1))
        eta_prime = (eta * omega3 - 1/2 * pi * 1j) / omega1
        return eta, eta_prime
