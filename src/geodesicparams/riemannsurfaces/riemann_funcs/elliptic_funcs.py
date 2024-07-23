from mpmath import jtheta, pi, exp, sqrt, qfrom, im, mpf, re, elliprf, agm, mp, chop, mpc, sin, cos, floor, mpf, isinf, sinh, cosh, cot, coth, almosteq
from sympy import Poly, Symbol
from ..period_matrices.periods_genus1_second import periods_secondkind
from ...utilities import eval_roots


def weierstrass_roots(omega1, omega3):
    if isinf(omega3):
        c = pi**2 / (12 * omega1**2)
        return 2 * c, -c , -c
    elif isinf(omega1):
        c = 1j * pi**2 / (12 * omega3**2)
        return c, c, -2 * c
    else:
        e1 = chop(weierstrass_P(omega1, omega1, omega3))
        e2 = chop(weierstrass_P(omega3, omega1, omega3))
        e3 = chop(weierstrass_P(-omega1 - omega3, omega1, omega3))

        return e1, e2, e3

def invariants_from_periods(omega1, omega3):
    e1, e2, e3 = weierstrass_roots(omega1, omega3)
    g2 = 2 * (e1**2 + e2**2 + e3**2)
    g3 = 4 * e1 * e2 * e3
    return g2, g3

def weierstrass_P(z, omega1, omega3, derivative = 0):
    z = mpc(z)

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
 
    tau = omega3 / omega1
    nome = qfrom(tau = tau)
    theta2 = jtheta(2, 0, nome)
    theta3 = jtheta(3, 0, nome)

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
    z = mpc(z)

    e1, e2, e3 = weierstrass_roots(omega1, omega3)
    result = chop(elliprf(z-e1, z-e2, z-e3))

    return result

def weierstrass_zeta(z, omega1, omega3):
    z = mpc(z)

    if isinf(omega3):
        c = pi**2 / (12 * omega1**2)
        return c * z + (3 * c)**(1/2) * cot((3 * c)**(1/2) * z)
    elif isinf(omega1):
        c = 1j * pi**2 / (12 * omega3**2)
        return - c * z + (3 * c)**(1/2) * coth((3 * c)**(1/2) * z)
    else:
        tau = omega3/omega1
        nome = qfrom(tau = tau)
        eta = periods_secondkind(omega1, omega3)[0]
        v = (pi*z)/(2*omega1)
        return eta * z/omega1 + pi * jtheta(1, v, nome, 1) / (2 * omega1 * jtheta(1, v, nome))

def weierstrass_sigma(z, omega1, omega3):
    z = mpc(z)

    if isinf(omega3):
        c = pi**2 / (12 * omega1**2)
        return (3 * c)**(-1/2) * sin((3 * c)**(1/2) * z) * exp(c * z**2 / 2)
    elif isinf(omega1):
        c = 1j * pi**2 / (12 * omega3**2)
        return (3 * c)**(-1/2) * sinh((3 * c)**(1/2) * z) * exp(- c * z**2 / 2)
    else:
        tau = omega3/omega1
        nome = qfrom(tau = tau)
        eta = periods_secondkind(omega1, omega3)[0]
        v = pi * z /(2*omega1)
        return 2*omega1 / pi * exp(eta * z**2 / (2*omega1)) * jtheta(1, v, nome) / jtheta(1, 0, nome, 1)
