#!/usr/bin/env python
"""
Procedures for computing the periods of the first kind (the half periods in the 
period lattice) from the elliptic invariants g2 and g3 from the Weierstrass cubic
4z**3 - g2z - g3.

NOTE: fix some things here like the precision.

"""

from mpmath import pi, sqrt, im, mpf, re, agm, mpc, floor, mpf, almosteq
from sympy import Poly, Symbol

from ...utilities import eval_roots

def periods_firstkind(g2, g3):
    """
    Computes the periods of the first kind (i.e. the integrals of the canonical holomorphic 
    differentials for a genus 1 Riemann surface) from the elliptic invariants <g2> and <g3>.

    Parameters
    ----------
    g2 : complex
        The elliptic invariant in the Weierstrass cubic.
    g3 : complex
        The elliptic invariant in the Weierstrass cubic.

    Returns
    -------
    omega1 : complex
        The first half period in the period lattice.     
    omega3 : complex
        The second half period in the period lattice.

    """

    g2 = mpf(g2); g3 = mpf(g3)
    discrim = floor(g2**3 - 27 * g3**2)
    x = Symbol("x")
    roots = eval_roots(Poly(4 * x**3 - g2 * x -g3).all_roots())

    # Special values for the half periods
    if discrim == 0 and g2 > 0 and g3 < 0:
        for i in range(len(roots) - 1):
            if almosteq(roots[i], roots[i + 1], abs_eps = 0.1):
                c = re(roots[i])
                break
        return mpf("inf"), (12 * c)**(-1/2) * pi * 1j
    elif discrim == 0 and g2 > 0 and g3 > 0:
        for i in range(len(roots) - 1):
            if almosteq(roots[i], roots[i + 1], abs_eps = 0.1):
                c = - re(roots[i])
                break
        return (12 * c)**(-1/2) * pi, mpc(0, "inf")
    elif g2 == 0 and g3 == 0:
        return mpf("inf"), mpc(0, "inf")
    # Normal results
    else:
        e3, e2, e1 = roots
        a = sqrt(e1 - e3)
        b = sqrt(e1 - e2)
        c = sqrt(e2 - e3)

        omega1 = pi / agm(a, b) / 2
        omega2 = pi / agm(1j * a, 1j * c) / 2

        if (re(omega2) == 0):
            omega3 = 1j * -im(omega2)
        else:
            omega3 = re(pi / agm(c, 1j * b) / 2) - 1j * im(pi / agm(c, 1j * b) / 2)

        return omega1, omega3
