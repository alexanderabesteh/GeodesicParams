#!/usr/bin/env python3
"""


"""

from mpmath import exp, sin, cos, pi, re, im, mp

from theta_helper import agm_prime, theta_char, theta_genus2, diff_finies_one_step, sign_theta

def hyp_theta(z, riemannM, derivatives = [], minMax = 5):


    g = [1/2, 1/2]; h = [0, 1/2]
    g = [0, 0]; h = [0, 0]
    result = 0

    for m1 in range(-minMax, minMax + 1):
        for m2 in range(-minMax, minMax + 1):
            m = [m1, m2]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM[i, j] * (m[j] + g[j])# + 2 * z[i] + 2 * h[i]
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])
                
            if len(derivatives) == 0:
                result += exp(1j * pi * char_sum)
            elif derivatives[0] != 0 and derivatives[1] != 0:
                result += exp(1j * pi * char_sum) * (m1 + g[0])**(derivatives[0]) * (m2 + g[1])**(derivatives[1])
            elif derivatives[0] != 0:
                result += exp(1j * pi * char_sum) * (m1 + g[0])**(derivatives[0])
            elif derivatives[1] != 0:
                result += exp(1j * pi * char_sum) * (m2 + g[1])**(derivatives[1])

    if len(derivatives) != 0:
        return (2 * pi * 1j)**(derivatives[0] + derivatives[1]) * result
    else:
        return result

def kleinian_sigma(z, omega, eta, char, riemannM):


    omega_inv = omega**(-1)
    exp_part = exp(-1/2 * z.T * eta * omega_inv * z)
    theta_part = hyp_theta_genus2(omega_inv * z, riemannM, char)

    return exp_part * theta_part

def kleinian_zeta(z, omega, eta, char, riemannM, derivatives = [], minMax = 5):


    sigma = kleinian_sigma(z, omega, eta, char, riemannM)
    g = char[0]; h = char[1]

    for m1 in range(-minMax, minMax + 1):
        for m2 in range(-minMax, minMax + 1):
            m = [m1, m2]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM[i, j] * (m[j] + g[j]) 
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])

            result += exp(1j * pi * char_sum) * 2 * pi * 1j * (m1 + g[0])

    return result / sigma

def kleinian_P(z, riemannM, derivatives = []):



    result = 1

    return result

def hyp_theta_genus2(z, tau, char, precision = 53):
    

    LOW_PRECISION = 3000  # Example value for low precision threshold

    # Determine initial precision
    mp.prec = precision
    lowprec = mp.prec

    flag = 0
    if lowprec < LOW_PRECISION:
        flag = 1
    else:
        while lowprec > LOW_PRECISION:
            lowprec = (lowprec // 2) + 10

    # Low precision computation
    mp.prec = lowprec
    CC = mp.mpc
    zerolow = mp.matrix([CC("0"), CC("0")])
    zlow = mp.matrix([CC(f"{mp.re(z[0])}", f"{mp.im(z[0])}"), CC(f"{mp.re(z[1])}", f"{mp.im(z[1])}")])
    taulow = mp.matrix([[CC(f"{mp.re(tau[0][0])}", f"{mp.im(tau[0][0])}"), CC(f"{mp.re(tau[0][1])}", f"{mp.im(tau[0][1])}")], [CC(f"{mp.re(tau[1][0])}", f"{mp.im(tau[1][0])}"), CC(f"{mp.re(tau[1][1])}", f"{mp.im(tau[1][1])}")]])

    initA = []
    initB = []
    flag = 1
    if flag == 1:
        return theta_char(z, tau, char, precision)
    else:
        for i in range(4):
            initA.append(theta_genus2(i, zlow, taulow, lowprec))
            initB.append(theta_genus2(i, zerolow, taulow, lowprec))

        initA = [x**2 for x in initA]
        initB = [x**2 for x in initB]

    # Computing lambda_iwant and det_iwant
    z1sq = z[0]**2
    z2sq = z[1]**2
    twoz1z2 = (z[0] + z[1])**2 - z1sq - z2sq
    det_iwant = tau[0][1]**2 - tau[0][0] * tau[1][1]
    IPI = mp.mpc(0, 1) * mp.pi
    lambda_iwant = [
        mp.exp(IPI * z1sq / tau[0][0]),
        mp.exp(IPI * z2sq / tau[1][1]),
        mp.exp(IPI * ((z1sq * tau[1][1] + z2sq * tau[0][0] - twoz1z2 * tau[1][0]) / (-det_iwant) - 1))
    ]
    a = [initA[i] / initA[0] for i in range(4)]
    b = [initB[i] / initB[0] for i in range(4)]

    p = lowprec
    while p < precision:
        p = 2 * p
        mp.prec = p
        z = [CC(f"{mp.re(x)}", f"{mp.im(x)}") for x in z]
        tau = [[CC(f"{mp.re(x)}", f"{mp.im(x)}") for x in row] for row in tau]
        lambda_iwant = [CC(f"{mp.re(x)}", f"{mp.im(x)}") for x in lambda_iwant]
        det_iwant = CC(f"{mp.re(det_iwant)}", f"{mp.im(det_iwant)}")
        a, b = diff_finies_one_step(
            a, b, mp.matrix([z[0], z[1]]), 
            mp.matrix([[tau[0][0], tau[0][1]], [tau[1][0], tau[1][1]]]), 
            lambda_iwant, det_iwant
        )
    # Apply AGMPrime to unstick the thetas
    theta00z = agm_prime(a, b, z, mp.matrix(tau))[0]
    
    a = [mp.sqrt(a[i] / theta00z) * sign_theta(2, z, mp.matrix(tau)) for i in range(4)]
   # b = [mp.sqrt(b[i] / theta000) * sign_theta(2, zerolow, tau) for i in range(4)]

    z = mp.matrix(z)
    tau = mp.matrix(tau)
    g = mp.matrix(char[0]); h = mp.matrix(char[1])
    exp_factor = mp.exp((mp.j * mp.pi * g.T * (tau * g + 2 * z + 2 * h))[0])
    result = exp_factor * a[0]

    return result

def hyp_theta_RR(xR, xI, wR, wI, l, riemannM, minMax = 5):

    g = [1/2, 1/2]
    h = [0, 1/2]
    result = 0
    varR = [xR, wR]
    varI = [xI, wI]
    riemannR = riemannM.apply(re)
    riemannI = riemannM.apply(im)

    for m1 in range(-minMax, minMax + 1):
        for m2 in range(-minMax, minMax + 1):
            m = [m1, m2]
            char_sumExp = 0
            char_sumSin = 0

            for i in range(2):
                tau_sumI = 0
                tau_sumR = 0

                for j in range(2):
                    tau_sumI += -riemannI[i, j] * (m[j] + g[j])
                    tau_sumR += riemannR[i, j] * (m[j] + g[j])

                char_sumExp += (m[i] + g[i]) * (tau_sumI - 2 * varI[i])
                char_sumSin += (m[i] + g[i]) * (tau_sumR + 2 * varR[i] + 2 * h[i])

            result -= (exp(pi * char_sumExp) * sin(pi * char_sumSin) * 2 * pi * (m[l] + g[l]))
    return result
 
def hyp_theta_IR(xR, xI, wR, wI, l, riemannM, minMax = 5):

    g = [1/2, 1/2]; h = [0, 1/2]
    result = 0
    varR = [xR, wR]
    varI = [xI, wI]
    riemannR = riemannM.apply(re)
    riemannI = riemannM.apply(im)

    for m1 in range(-minMax, minMax + 1):
        for m2 in range(-minMax, minMax + 1):
            m = [m1, m2]
            char_sumExp = 0
            char_sumCos = 0

            for i in range(2):
                tau_sumI = 0
                tau_sumR = 0

                for j in range(2):
                    tau_sumI += -riemannI[i, j] * (m[j] + g[j])
                    tau_sumR += riemannR[i, j] * (m[j] + g[j])

                char_sumExp += (m[i] + g[i]) * (tau_sumI - 2 * varI[i])
                char_sumCos += (m[i] + g[i]) * (tau_sumR + 2 * varR[i] + 2 * h[i])
            
            result += exp(pi * char_sumExp) * cos(pi * char_sumCos) * 2 * pi * (m[l] + g[l])
    return result

def sigma1(z, riemannM, minMax = 5):

    g = [1/2, 1/2]
    h = [0, 1/2]
    result = 0

    for m1 in range(-minMax, minMax + 1):
        for m2 in range(-minMax, minMax + 1):
            m = [m1, m2]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM[i, j] * (m[j] + g[j]) 
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])

            result += exp(1j * pi * char_sum) * 2 * pi * 1j * (m1 + g[0])

    return result

def sigma2(z, riemannM, minMax = 5):

    g = [1/2, 1/2]
    h = [0, 1/2]
    result = 0

    for m1 in range(-minMax, minMax + 1):
        for m2 in range(-minMax, minMax + 1):
            m = [m1, m2]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM[i, j] * (m[j] + g[j]) 
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])
            result += exp(1j * pi * char_sum) * 2 * pi * 1j * (m2 + g[1])

    return result
