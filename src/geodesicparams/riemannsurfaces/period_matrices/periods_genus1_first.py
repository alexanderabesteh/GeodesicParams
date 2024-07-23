from mpmath import jtheta, pi, sqrt, qfrom, im, mpf, re, agm, mp, mpc, floor, mpf, isinf, almosteq
from sympy import Poly, Symbol
from ...utilities import eval_roots

def periods_firstkind(g2, g3):
    g2 = mpf(g2); g3 = mpf(g3)
    discrim = floor(g2**3 - 27 * g3**2)
    x = Symbol("x")
    roots = eval_roots(Poly(4 * x**3 - g2 * x -g3).all_roots())

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

def periods_secondkind(omega1, omega3):
    if isinf(omega3):
        c = pi**2 / 12 * omega1**2
        return c * omega1, mpc(0, "inf")
    elif isinf(omega1):
        c = 1j * pi **2 / 12 * omega3**2
        return mpc("-inf", 0), - c * omega3
    else:
        tau = omega3/omega1
        nome = qfrom(tau = tau)
        eta = - pi**2 * jtheta(1, 0, nome, 3) / (12 * omega1 * jtheta(1, 0, nome, 1))
        eta_prime = (eta*omega3 - 1/2 * pi * 1j) / omega1
        return eta, eta_prime

