from mpmath import jtheta, pi, qfrom, mp, mpc, isinf

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
