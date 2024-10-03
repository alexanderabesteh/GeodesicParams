#!/usr/bin/env python3
"""
A collection of procedures for computing hyperelliptic functions for genus a genus 2 Riemann
surface.

In particular, the functions implemented are the hyperelliptic theta function (with various 
algorithms), Kleinian sigma function, Kleinian zeta function, Kleinian P function, and the 
derivatives of the theta function and sigma functions.

TODO: remove unnecessary code and fix precision.
"""

from mpmath import exp, sin, cos, pi, re, im, mp

from theta_helper import agm_prime, theta_char, theta_genus2, diff_finies_one_step, sign_theta, derivative_factor

def hyp_theta_fourier(z, riemannM, char, derivatives = [], minMax = 5):
    """
    Computes the hyperelliptic theta function on a genus 2 Riemann surface using the Fourier
    series definition of the function. Optional derivatives can be computed.

    Parameters
    ----------
    z : list
        A list containing two complex numbers.
    riemannM : matrix
        The Riemann matrix of the Riemann surface.
    char : list
        A list containing two lists of length 2. These lists represent the g and h 
        characteristics of the theta function (the elements of these lists are either 0 or 1/2).
    derivatives : list, optional
        A list containing integers, either 0, 1, or 2. These represent the partial derivatives of the
        theta function with respect to the first and second components of the vector z, z1 and z2
        respectively. For example, [1, 2 ,1] computes the partial derivative of the theta function
        with respect to z1, z2, and finally z1. An integer of 0 means no derivative is computed.
    minMax : natural, optional
        A natural number from 5 <= minMax <= 30 (the summation bound).

    Returns
    -------
    result : complex
        The value of the hyperelliptic theta function evaluated at <z> with Riemann matrix
        <riemannM>.

    """

    g, h = char
    #g = [1/2, 1/2]; h = [0, 1/2]
    #g = [0, 0]; h = [0, 0]
    derivs_product = 1
    result = 0

    for m1 in range(-minMax, minMax + 1):
        for m2 in range(-minMax, minMax + 1):
            m = [m1, m2]
            char_sum = 0
            derivs_product = 1

            # Compute characteristics and Riemann matrix contribution
            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM[i, j] * (m[j] + g[j])# + 2 * z[i] + 2 * h[i]
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])
                
            #if len(derivatives) == 0:
             #   result += exp(1j * pi * char_sum)
            #elif derivatives[0] != 0 and derivatives[1] != 0:
            for i in derivatives:
                if i == 1:
                    derivs_product *= (m1 + g[0])
                elif i == 2:
                    derivs_product *= (m2 + g[1])

            result += exp(1j * pi * char_sum) * derivs_product 
                # result += exp(1j * pi * char_sum) * (m1 + g[0])**(derivatives[0]) * (m2 + g[1])**(derivatives[1])
           # elif derivatives[0] != 0:
                # result += exp(1j * pi * char_sum) * (m1 + g[0])**(derivatives[0])
           # elif derivatives[1] != 0:
                # result += exp(1j * pi * char_sum) * (m2 + g[1])**(derivatives[1])

        # Compute 2*pi*1j factor
        derivs_factor = derivative_factor(derivatives) 
    #if len(derivatives) != 0:
        return derivs_factor * result
        # return (2 * pi * 1j)**(derivatives[0] + derivatives[1]) * result
    #else:
     #   return result

def hyp_theta_genus2(z, tau, char, precision = 53):
    """
    Computes the hyperelliptic theta function on a genus 2 Riemann surface using the Naive
    algorithm and the high-precision algorithm found in [].

    When the precision is less than 3000 bits, the Naive algorithm is used while the
    high-precision algorithm is used above 3000 bits.

    Parameters
    ----------
    z : list
        A list containing two complex numbers.
    riemannM : matrix
        An mpmath matrix, the Riemann matrix of the Riemann surface.
    char : list
        A list containing two lists of length 2. These lists represent the g and h 
        characteristics of the theta function (the elements of these lists are either 0 or 1/2).
    precision : int, optional
        The binary precision of the computation.
    
    Returns
    -------
    result : complex
        The value of the hyperelliptic theta function evaluated at <z> with Riemann matrix
        <riemannM>.

    """
    
    LOW_PRECISION = 3000

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
    # Apply agm_prime to unstick the thetas
    theta00z = agm_prime(a, b, z, mp.matrix(tau))[0]
    
    a = [mp.sqrt(a[i] / theta00z) * sign_theta(2, z, mp.matrix(tau)) for i in range(4)]

    z = mp.matrix(z)
    tau = mp.matrix(tau)
    g = mp.matrix(char[0]); h = mp.matrix(char[1])
    exp_factor = mp.exp((mp.j * mp.pi * g.T * (tau * g + 2 * z + 2 * h))[0])
    result = exp_factor * a[0]

    return result

def hyp_theta_RR(xR, xI, wR, wI, l, riemannM, char, minMax = 5):
    """
    Computes the partial derivative of the real part of the hyperelliptic theta function
    with respect <xR> or <wR>, depending on <l>. The vector z of the theta function is split
    into two complex variables <x> and <w> (i.e. z = [xR + 1j * xI, wR + 1j * xI]).

    Parameters
    ----------
    xR : real
        The real part of the first component of z. 
    xI : real
        The imaginary part of the first component of z. 
    wR : real
        The real part of the second component of z. 
    wI : real
        The imaginary part of the second component of z. 
    l : int
        An integer, either 1 or 2 representing the partial derivative with respect to 
        <xR> for <l> = 1 and <wR> for <l> = 2.
    char : list
        A list containing two lists of length 2. These lists represent the g and h 
        characteristics of the theta function (the elements of these lists are either 0 or 1/2).
    minMax : natural, optional
        A natural number from 5 <= minMax <= 30 (the summation bound).
    
    Returns
    -------
    result : complex
        The partial deriative of the real part of the theta function with respect to <xR> or
        <wR>.

    """

    g, h = char
    #g = [1/2, 1/2]
   # h = [0, 1/2]
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

                # Riemann matrix contributions
                for j in range(2):
                    tau_sumI += -riemannI[i, j] * (m[j] + g[j])
                    tau_sumR += riemannR[i, j] * (m[j] + g[j])

                char_sumExp += (m[i] + g[i]) * (tau_sumI - 2 * varI[i])
                char_sumSin += (m[i] + g[i]) * (tau_sumR + 2 * varR[i] + 2 * h[i])

            result -= (exp(pi * char_sumExp) * sin(pi * char_sumSin) * 2 * pi * (m[l] + g[l]))
    return result
 
def hyp_theta_IR(xR, xI, wR, wI, l, riemannM, char, minMax = 5):
    """
    Computes the partial derivative of the imaginary part of the hyperelliptic theta function
    with respect <xR> or <wR>, depending on <l>. The vector z of the theta function is split
    into two complex variables <x> and <w> (i.e. z = [xR + 1j * xI, wR + 1j * xI]).

    Parameters
    ----------
    xR : real
        The real part of the first component of z. 
    xI : real
        The imaginary part of the first component of z. 
    wR : real
        The real part of the second component of z. 
    wI : real
        The imaginary part of the second component of z. 
    l : int
        An integer, either 1 or 2 representing the partial derivative with respect to 
        <xR> for <l> = 1 and <wR> for <l> = 2.
    char : list
        A list containing two lists of length 2. These lists represent the g and h 
        characteristics of the theta function (the elements of these lists are either 0 or 1/2).
    minMax : natural, optional
        A natural number from 5 <= minMax <= 30 (the summation bound).
    
    Returns
    -------
    result : complex
        The partial deriative of the real part of the theta function with respect to <xR> or
        <wR>.

    """

    g, h = char
    # g = [1/2, 1/2]; h = [0, 1/2]
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

                # Riemann matrix contributions
                for j in range(2):
                    tau_sumI += -riemannI[i, j] * (m[j] + g[j])
                    tau_sumR += riemannR[i, j] * (m[j] + g[j])

                char_sumExp += (m[i] + g[i]) * (tau_sumI - 2 * varI[i])
                char_sumCos += (m[i] + g[i]) * (tau_sumR + 2 * varR[i] + 2 * h[i])
            
            result += exp(pi * char_sumExp) * cos(pi * char_sumCos) * 2 * pi * (m[l] + g[l])
    return result

def kleinian_sigma(z, omega, eta, char, riemannM):
    """
    Evaluates the Kleinian sigma function at <z> with Riemann matrix <riemannM>.

    Parameters
    ----------
    z : list
        A list containing two complex numbers.
    omega : matrix
        The period matrix = the contour integral of the vector of canonical holomorphic 
        differentials taken along the contours that encircle the branch cuts.
    eta : matrix
        The period matrix = the contour integral of the vector of canonical meromorphic
        differentials taken along the contours that encircle the branch cuts.   
    char : list
        A list containing two lists of length 2. These lists represent the g and h 
        characteristics of the theta function (the elements of these lists are either 0 or 1/2).
    riemannM : matrix
        An mpmath matrix, the Riemann matrix of the Riemann surface.

    Returns
    -------
    result : complex
        The Kleinian sigma function evaluated at <z>.

    """

    omega_inv = omega**(-1)
    exp_part = exp(-1/2 * z.T * eta * omega_inv * z)
    theta_part = hyp_theta_genus2(omega_inv * z, riemannM, char)

    return exp_part * theta_part

def kleinian_zeta(z, omega, eta, char, riemannM, derivative, minMax = 5):
    """
    Evaluates the Kleinian zeta function at <z> with derivatives. 

    Parameters
    ----------
    z : list
        A list containing two complex numbers.
    omega : matrix
        The period matrix = the contour integral of the vector of canonical holomorphic 
        differentials taken along the contours that encircle the branch cuts.
    eta : matrix
        The period matrix = the contour integral of the vector of canonical meromorphic
        differentials taken along the contours that encircle the branch cuts.   
    char : list
        A list containing two lists of length 2. These lists represent the g and h 
        characteristics of the theta function (the elements of these lists are either 0 or 1/2).
    riemannM : matrix
        An mpmath matrix, the Riemann matrix of the Riemann surface.
    derivative : int
        An integer = 1 or 2, where 1 is the partial derivative with respect to the 
        first component of z, and 2 is with respect to the second component.
    minMax : natural, optional
        A natural number from 5 <= minMax <= 30 (the summation bound).

    Returns
    -------
    result : complex
        The value of the Kleinian zeta function at <z> with derivatives.

    """

    sigma = kleinian_sigma(z, omega, eta, char, riemannM)
    result = hyp_theta_fourier(z, riemannM, char, [derivative], minMax) 

    return result / sigma

def kleinian_P(z, omega, eta, char, riemannM, derivatives, minMax = 5):
    """
    Evaluates the Kleinian P function at <z> with three partial derivatives.
    
    Parameters
    ----------
    z : list
        A list containing two complex numbers.
    omega : matrix
        The period matrix = the contour integral of the vector of canonical holomorphic 
        differentials taken along the contours that encircle the branch cuts.
    eta : matrix
        The period matrix = the contour integral of the vector of canonical meromorphic
        differentials taken along the contours that encircle the branch cuts.   
    char : list
        A list containing two lists of length 2. These lists represent the g and h 
        characteristics of the theta function (the elements of these lists are either 0 or 1/2).
    riemannM : matrix
        An mpmath matrix, the Riemann matrix of the Riemann surface.
    derivatives : list
        A list containing three integers, where each integer = 1 or 2. An integer of 1 means the
        partial derivative with respect to the first component of z, while 2 means with respect to
        the second component of z.
    minMax : natural, optional
        A natural number from 5 <= minMax <= 30 (the summation bound).

    Returns
    -------
    result : complex
        The Kleinian P function evaluated at <z> with three partial derivatives.

    """

    sigma = kleinian_sigma(z, omega, eta, char, riemannM)

    # First partial derivatives
    sigmai = hyp_theta_fourier(z, riemannM, char, [derivatives[0], 0, 0], minMax)  
    sigmaj = hyp_theta_fourier(z, riemannM, char, [0, derivatives[1], 0], minMax)  
    sigmak = hyp_theta_fourier(z, riemannM, char, [0, 0, derivatives[2]], minMax)  

    # Second partial derivatives
    sigmaij = hyp_theta_fourier(z, riemannM, char, [derivatives[0], derivatives[1]], minMax) 
    sigmaik = hyp_theta_fourier(z, riemannM, char, [derivatives[0], 0, derivatives[2]], minMax)  
    sigmajk = hyp_theta_fourier(z, riemannM, char, [0, derivatives[1], derivatives[2]], minMax)

    # Third partial derivative
    sigmaijk = hyp_theta_fourier(z, riemannM, char, derivatives, minMax)

    result = (sigmai * sigmaj * sigmak) - (sigmaij * sigmak * sigma) - (sigmaik * sigmaj * sigma) - (sigmajk * sigmai * sigma) + (sigmaijk * sigma)

    return result / sigma**3

"""
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

"""
