#!/usr/bin/env python3
"""
A collection of procedures for computing hyperelliptic functions for genus a genus 2 Riemann
surface.

In particular, the functions implemented are the hyperelliptic theta function (with various
algorithms), Kleinian sigma function, Kleinian zeta function, Kleinian P function, and the
derivatives of the theta function and sigma functions.

TODO: remove unnecessary code and fix precision.
"""

from mpmath import cos, exp, im, mp, pi, re, sin


def naive_theta_genus2(z, tau, precision=53):
    """
    Computes the hyperelliptic theta function on a genus 2 Riemann surface using the Naive
    algorithm found in []. The Riemann matrix <tau> must also be Minkowski-reduced
    (Im(tau_3) <= Im(tau_1) <= Im(tau_2)).

    Parameters
    ----------
    z : list
        A list containing two complex numbers.
    tau : matrix
        An mpmath matrix, the Riemann matrix of the Riemann surface. The Riemann matrix
        should also satisfy Im(tau_3) <= Im(tau_1) <= Im(tau_2) (Minkowski-reduced).
    precision : int, optional
        The binary precision of the computation.

    Returns
    -------
    result : complex
        The value of the hyperelliptic theta function evaluated at <z> with Riemann matrix
        <tau>.

    """

    mp.prec = precision

    B = 2 * precision * mp.log(10) / mp.pi + 3

    q1 = mp.exp(mp.j * mp.pi * tau[0, 0])
    q1sq = q1**2
    q2 = mp.exp(mp.j * mp.pi * tau[1, 1])
    q2sq = q2**2
    q3 = mp.exp(mp.j * mp.pi * tau[0, 1])
    q3sq = q3**2

    w1 = mp.exp(mp.j * mp.pi * z[0])
    w1sq = w1**2
    w2 = mp.exp(mp.j * mp.pi * z[1])
    w2sq = w2**2

    result = mp.mpc(1)

    # n=0
    q12mminus2 = q1sq
    r1 = w1sq + 1 / w1sq
    r1m = q1 * r1
    r1mminus1 = 2
    when_to_stop = int(mp.ceil(mp.sqrt(B / mp.im(tau[0, 0]))) + 3)

    for _ in range(1, when_to_stop + 1):
        result += r1m
        q12mminus2timesq1 = q12mminus2 * q1
        bubu = r1m
        r1m = r1 * r1m * q12mminus2timesq1 - r1mminus1 * (q12mminus2**2)
        r1mminus1 = bubu
        q12mminus2 *= q1sq

    # m,n >=1
    q2to2nminus2 = 1
    q3to2n = q3sq
    s1 = w2sq + 1 / w2sq
    w1w2sq = w1sq * w2sq
    w1invw2sq = w2sq / w1sq
    q2s1 = q2 * s1
    q1r1 = q1 * r1
    q1q2 = q1 * q2
    betas = [[q1r1, q1q2 * q3sq * (w1w2sq + 1 / w1w2sq)], [mp.mpc(2), q2s1]]
    betaprimes = [
        [q1r1, (q1q2 / q3sq) * (w1invw2sq + 1 / w1invw2sq)],
        [mp.mpc(2), q2s1],
    ]

    for n in range(1, int(mp.ceil(mp.sqrt(B / mp.im(tau[1, 1])))) + 4):
        if n > 3:
            when_to_stop = B - (n - 3) ** 2 * mp.im(tau[1, 1])
        else:
            when_to_stop = B
        if when_to_stop <= 0:
            when_to_stop = 0
        when_to_stop = mp.ceil(mp.sqrt(when_to_stop / mp.im(tau[0, 0]))) + 3

        # This squared gives q**(4m-4)
        q1to2mminus2 = q1sq

        # Not betas + betaprimes (since m = 0 we only add the term once)
        term = betas[1][1]
        result += term

        alphazm = betas[0][1]
        alphaprimezm = betaprimes[0][1]
        alphazmminus1 = betas[1][1]
        alphaprimezmminus1 = betaprimes[1][1]

        for _ in range(1, int(when_to_stop) + 1):

            term = alphazm + alphaprimezm
            result += term

            r1Xq1to2mminus2Xq1 = r1 * q1to2mminus2 * q1
            bubu = alphazm
            alphazm = (
                alphazm * r1Xq1to2mminus2Xq1 * q3to2n
                - (q1to2mminus2 * q3to2n) ** 2 * alphazmminus1
            )
            alphazmminus1 = bubu
            bubu = alphaprimezm
            alphaprimezm = (
                alphaprimezm * r1Xq1to2mminus2Xq1 / q3to2n
                - (q1to2mminus2 / q3to2n) ** 2 * alphaprimezmminus1
            )
            alphaprimezmminus1 = bubu

            q1to2mminus2 *= q1sq

        q2to2nminus2 *= q2sq

        s1Xq2to2nminus2Xq2 = s1 * q2to2nminus2 * q2
        bubu = [betas[0][1], betas[1][1]]
        betas[0][1] = (
            betas[0][1] * s1Xq2to2nminus2Xq2 * q3sq
            - betas[0][0] * (q2to2nminus2 * q3sq) ** 2
        )
        betas[1][1] = betas[1][1] * s1Xq2to2nminus2Xq2 - q2to2nminus2**2 * betas[1][0]
        betas[0][0] = bubu[0]
        betas[1][0] = bubu[1]
        bubu = [betaprimes[0][1], betaprimes[1][1]]
        betaprimes[0][1] = (
            betaprimes[0][1] * s1Xq2to2nminus2Xq2 / q3sq
            - betaprimes[0][0] * (q2to2nminus2 / q3sq) ** 2
        )
        betaprimes[1][1] = (
            betaprimes[1][1] * s1Xq2to2nminus2Xq2 - q2to2nminus2**2 * betaprimes[1][0]
        )
        betaprimes[0][0] = bubu[0]
        betaprimes[1][0] = bubu[1]
        q3to2n *= q3sq

    result = mp.mpc(f"{mp.re(result)}", f"{mp.im(result)}")
    return result


def theta_char(z, tau, char, precision=53):
    """
    Computes the hyperelliptic theta function on a genus 2 Riemann surface with characteristics
    <char> using the Naive algorithm found in []. The Riemann matrix <tau> must also be
    Minkowski-reduced (Im(tau_3) <= Im(tau_1) <= Im(tau_2)).

    Parameters
    ----------
    z : list
        A list containing two complex numbers.
    tau : matrix
        An mpmath matrix, the Riemann matrix of the Riemann surface. The Riemann matrix
        should also satisfy Im(tau_3) <= Im(tau_1) <= Im(tau_2) (Minkowski-reduced).
    char : list
        A list containing two lists of length 2. These lists represent the g and h
        characteristics of the theta function (the elements of these lists are either 0 or 1/2).
    precision : int, optional
        The binary precision of the computation.

    Returns
    -------
    result : complex
        The value of the hyperelliptic theta function evaluated at <z> with Riemann matrix
        <tau> and characteristics <char>.

    """

    mp.prec = precision

    z = mp.matrix(z)
    tau = mp.matrix(tau)
    g = mp.matrix(char[0])
    h = mp.matrix(char[1])
    exp_factor = mp.exp((mp.j * mp.pi * g.T * (tau * g + 2 * z + 2 * h))[0])
    result = exp_factor * naive_theta_genus2(z + tau * g + h, tau, precision)

    return result


def derivative_factor(derivatives):
    """
    Computes the 2*pi*1j factor in front of the Fourier series definition of the theta function
    after taking partial derivatives.

    Parameters
    ----------
    derivatives : list
        A list containing the integers representing the derivatives with respect to z1 or z2.
        A 0 means no derivative is computed, while 1 and 2 represent the derivatives with respect
        to z1 and z2 respectively.

    Returns
    -------
    complex
        The derivative factor in front of the Fourier series.
    """

    count = 0

    for i in derivatives:
        if i != 0:
            count += 1

    return (2 * mp.pi * mp.j) ** count


def hyp_theta_fourier(z, riemannM, char, derivatives=[], minMax=5):
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
                    tau_sum += riemannM[i, j] * (m[j] + g[j])  # + 2 * z[i] + 2 * h[i]
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])

            for i in derivatives:
                if i == 1:
                    derivs_product *= m1 + g[0]
                elif i == 2:
                    derivs_product *= m2 + g[1]

            result += exp(1j * pi * char_sum) * derivs_product

    # Compute 2*pi*1j factor
    derivs_factor = derivative_factor(derivatives)

    return derivs_factor * result


def hyp_theta_RR(xR, xI, wR, wI, l, riemannM, char, minMax=5):
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
    # g = [1/2, 1/2]
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

            result -= (
                exp(pi * char_sumExp) * sin(pi * char_sumSin) * 2 * pi * (m[l] + g[l])
            )
    return result


def hyp_theta_IR(xR, xI, wR, wI, l, riemannM, char, minMax=5):
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

            result += (
                exp(pi * char_sumExp) * cos(pi * char_sumCos) * 2 * pi * (m[l] + g[l])
            )
    return result


def sigma1(z, riemannM, minMax=5):

    g = [1 / 2, 1 / 2]
    h = [0, 1 / 2]
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


def sigma2(z, riemannM, minMax=5):

    g = [1 / 2, 1 / 2]
    h = [0, 1 / 2]
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
