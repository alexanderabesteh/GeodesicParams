#!/usr/bin/env python3
"""
A collection of helper functions used to compute the various algorithms involved in computing
the hyperelliptic theta function on a genus 2 Riemann surface.

Algorithms for the naive theta function, high-precision theta function, and Fourier series
theta function have been implemented.

"""

from mpmath import mp
from numpy import matrix, kron, array

def naive_theta_genus2(z, tau, precision = 53):
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
    q1sq = q1 ** 2
    q2 = mp.exp(mp.j * mp.pi * tau[1, 1])
    q2sq = q2 ** 2
    q3 = mp.exp(mp.j * mp.pi * tau[0, 1])
    q3sq = q3 ** 2

    w1 = mp.exp(mp.j * mp.pi * z[0])
    w1sq = w1 ** 2
    w2 = mp.exp(mp.j * mp.pi * z[1])
    w2sq = w2 ** 2

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
        r1m = r1 * r1m * q12mminus2timesq1 - r1mminus1 * (q12mminus2 ** 2)
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
    betaprimes = [[q1r1, (q1q2 / q3sq) * (w1invw2sq + 1 / w1invw2sq)], [mp.mpc(2), q2s1]]

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
            alphazm = alphazm * r1Xq1to2mminus2Xq1 * q3to2n - (q1to2mminus2 * q3to2n) ** 2 * alphazmminus1
            alphazmminus1 = bubu
            bubu = alphaprimezm
            alphaprimezm = alphaprimezm * r1Xq1to2mminus2Xq1 / q3to2n - (q1to2mminus2 / q3to2n) ** 2 * alphaprimezmminus1
            alphaprimezmminus1 = bubu

            q1to2mminus2 *= q1sq

        q2to2nminus2 *= q2sq

        s1Xq2to2nminus2Xq2 = s1 * q2to2nminus2 * q2
        bubu = [betas[0][1], betas[1][1]]
        betas[0][1] = betas[0][1] * s1Xq2to2nminus2Xq2 * q3sq - betas[0][0] * (q2to2nminus2 * q3sq) ** 2
        betas[1][1] = betas[1][1] * s1Xq2to2nminus2Xq2 - q2to2nminus2 ** 2 * betas[1][0]
        betas[0][0] = bubu[0]
        betas[1][0] = bubu[1]
        bubu = [betaprimes[0][1], betaprimes[1][1]]
        betaprimes[0][1] = betaprimes[0][1] * s1Xq2to2nminus2Xq2 / q3sq - betaprimes[0][0] * (q2to2nminus2 / q3sq) ** 2
        betaprimes[1][1] = betaprimes[1][1] * s1Xq2to2nminus2Xq2 - q2to2nminus2 ** 2 * betaprimes[1][0]
        betaprimes[0][0] = bubu[0]
        betaprimes[1][0] = bubu[1]
        q3to2n *= q3sq
    
    result = mp.mpc(f"{mp.re(result)}", f"{mp.im(result)}")
    return result

def theta_char(z, tau, char, precision = 53):
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
    g = mp.matrix(char[0]); h = mp.matrix(char[1])
    exp_factor = mp.exp((mp.j * mp.pi * g.T * (tau * g + 2 * z + 2 * h))[0])
    result = exp_factor * naive_theta_genus2(z + tau * g + h, tau, precision)

    return result

def theta_genus2(n, z, tau, precision = 53):
    """
    Computes the hyperelliptic theta function on a genus 2 Riemann surface with characteristics
    determined by <n> using the Naive algorithm found in []. The Riemann matrix <tau> must also 
    be Minkowski-reduced (Im(tau_3) <= Im(tau_1) <= Im(tau_2)).

    Parameters
    ----------
    n : int
        An integer from 0 <= n <= 15 that determines the characteristics of the theta function.
    z : list
        A list containing two complex numbers.
    tau : matrix
        An mpmath matrix, the Riemann matrix of the Riemann surface. The Riemann matrix
        should also satisfy Im(tau_3) <= Im(tau_1) <= Im(tau_2) (Minkowski-reduced).
    precision : int, optional
        The binary precision of the computation.

    Returns
    -------
    complex
        The value of the hyperelliptic theta function evaluated at <z> with Riemann matrix
        <tau> and characteristics determined by <n>.
        
    """

    # Determine characteristics from <n>
    vec = [mp.mpf(0/2), mp.mpf(0/2), mp.mpf(0/2), mp.mpf(0/2)]
    vec[3] = mp.mpf(f"{round(n % 2)}") / 2
    n = round((n - (n % 2)) / 2)
    vec[2] = mp.mpf(f"{round(n % 2)}") / 2
    n = round((n - (n % 2)) / 2)
    vec[1] = mp.mpf(f"{round(n % 2)}") / 2
    n = round((n - (n % 2)) / 2)
    vec[0] = mp.mpf(f"{round(n % 2)}") / 2
    char = [vec[0:2], vec[2:4]]

    return theta_char(z, tau, char, precision)

def sign_theta(n, z, tau):
    """
    Computes the sign of the hyperelliptic theta function on a genus 2 Riemann surface with 
    characteristics determined by <n> using the Naive algorithm found in []. The Riemann 
    matrix <tau> must also be Minkowski-reduced (Im(tau_3) <= Im(tau_1) <= Im(tau_2)).

    Parameters
    ----------
    n : int
        An integer from 0 <= n <= 15 that determines the characteristics of the theta function.
    z : list
        A list containing two complex numbers.
    tau : matrix
        An mpmath matrix, the Riemann matrix of the Riemann surface. The Riemann matrix
        should also satisfy Im(tau_3) <= Im(tau_1) <= Im(tau_2) (Minkowski-reduced).
    
    Returns
    -------
    int
        The sign of the theta function, either +1 or -1.

    """

    prec = mp.prec
    mp.dps = 20

    z_matrix = mp.matrix([mp.mpc(f"{mp.re(z[0])}", f"{mp.im(z[0])}"), mp.mpc(f"{mp.re(z[1])}", f"{mp.im(z[1])}")])
    t_matrix = mp.matrix([[mp.mpc(f"{mp.re(tau[0, 0])}",f"{mp.im(tau[0, 0])}"), mp.mpc(f"{mp.re(tau[0, 1])}", f"{mp.im(tau[0, 1])}")], [mp.mpc(f"{mp.re(tau[1, 0])}",f"{mp.im(tau[1, 0])}"), mp.mpc(f"{mp.re(tau[1, 1])}", f"{mp.im(tau[1, 1])}")]])

    num = theta_genus2(n, z_matrix, t_matrix, mp.prec)
    den = theta_genus2(0, z_matrix, t_matrix, mp.prec)
    
    mp.prec = prec

    if mp.re(num / den) < 0:
        return -1
    else:
        return 1

def hadamard_matrix(fi, n):
    """
    Computes the nth order Hadamard matrix with the precision determined by the number of 
    bits in <fi>.

    Parameters
    ----------
    fi : complex
        A complex number to be used for determining the current working binary precision.
    n : int
        The order of the Hadamard matrix

    Returns
    -------
    result : matrix
        The nth order Hadamard matrix.

    """

    # Define the base 2x2 Hadamard matrix
    prec = mp.prec
    mp.prec = mp.re(fi).bc
    m = array([[mp.mpf(1), mp.mpf(1)], [mp.mpf(1), mp.mpf(-1)]])
    mp.prec = prec
    
    # Initialize the result as the base matrix
    result = m

    # Compute the nth order Hadamard matrix using kronecker product
    for _ in range(2, n + 1):
        result = kron(result, m)

    return result

def f(a, b, z, t):
    """
    Computes the function F in [], which returns the theta functions <a> and constants <b>
    evaluated with 2*t instead of t.

    Parameters
    ----------
    a : list
        A list containing the four theta functions squared from n = 0 to n = 3 evaluated at z.
    b : list
        A list containing the four theta constants squared from n = 0 to n = 3.
    z : list
        A list containing the values of the vector z.
    t : matrix
        The riemann matrix tau of the Riemann surface.

    Returns
    -------
    result_mults : list
        A list containing the 4 theta functions evaluated with 2*t.
    result_squares : list
        A list containing the 4 theta constants evaluated with 2*t.

    """

    n = len(a)

    # Extract with the right sign
    root_a = matrix([mp.sqrt(a[i]) * sign_theta(i, z, t) for i in range(n)]).T
    root_b = matrix([mp.sqrt(b[i]) * sign_theta(i, mp.matrix([0, 0]), t) for i in range(n)]).T

    # Hadamard stuff for optimal computation
    hadam = hadamard_matrix(a[0], 2)
    hadamard_a = hadam * root_a
    hadamard_b = hadam * root_b
    
    mults = matrix([hadamard_a[i, 0] * hadamard_b[i, 0] for i in range(n)]).T
    squares = matrix([hadamard_b[i, 0] ** 2 for i in range(n)]).T

    result_mults = mp.matrix(hadam)**-1 * mp.matrix(mults) / 4
    result_squares = mp.matrix(hadam)**-1 * mp.matrix(squares) / 4

    return result_mults, result_squares

def f_infinity(a, b, z, t):
    """
    Computes the sequence of functions F**n in [] as n approaches infinity.

    This gives the limits of the sequences a'_{i} and b'_{i}, where 0 <= i <= 3 which can be used 
    to determine lambda and mu in [].

    Parameters
    ----------
    a : list
        A list containing the four theta functions squared from n = 0 to n = 3 evaluated at z.
    b : list
        A list containing the four theta constants squared from n = 0 to n = 3. 
    z : list
        A list containing the values of the vector z.
    t : matrix
        The riemann matrix tau of the Riemann surface.

    Returns
    -------
    res : list
        A list containing two lists: the first list contains the limits of a'_{i}, while
        the second contains the limits of b'_{i}, where 0 <= i <= 3.

    """

    p = mp.prec
    myt = t
    r = a
    s = b
    res = [[r, s]]

    while mp.fabs(s[0] - s[1]) > 10**(-p + 10):
        r, s = f(r, s, z, myt)
        res.append([r, s])
        myt = 2 * myt

    r, s = f(r, s, z, myt)
    res.append([r, s])

    return res

def pow2_powN(x, n):
    """
    Raises <x> to the power of 2 <n> times.

    Parameters
    ----------
    x : complex
        The complex number to be squared.
    n : int
        Raise <x> to the 2nd power <n> times.

    Returns
    -------
    result : complex
        The value of <x> raised to the 2nd power <n> times.

    """

    r = x

    for _ in range(n):
        r = r**2

    return r

def agm_prime(a, b, z, t):
    """
    Computes lambda and mu variables in [] from the theta functions and constants.

    This is called by to_inverse to computed the Riemann matrix tau and z at a higher 
    precision.

    Parameters
    ----------
    a : list
        A list containing the four theta functions squared from n = 0 to n = 3 evaluated at z.
    b : list
        A list containing the four theta constants squared from n = 0 to n = 3. 
    z : list
        A list containing the values of the vector z.
    t : matrix
        The riemann matrix tau of the Riemann surface.
        
    Returns
    -------
    lambda_ : complex
        The complex number lambda in [].
    mu : complex
        The complex number mu in [].

    """

    R = f_infinity(a, b, z, t)
    mu = R[-1][1][0]  # R[-1] is the last [r,s]; R[-1][1] is the second element (s)
    qu = R[-1][0][0] / R[-1][1][0]
    lambda_ = pow2_powN(qu, len(R) - 1) * mu
    return lambda_, mu

def all_duplication(a, b):
    """
    Computes theta_{a, b}(z, t) given theta_{0, b}(z, t) and theta_{0, b}(0, t) (the theta
    functions and theta constants, see []).

    Parameters
    ----------
    a : list
        A list containing the four theta functions squared from n = 0 to n = 3 evaluated at z
        divided by the theta function squared at n = 0 (the first element is therefore always
        1).
    b : list
        A list containing the four theta constants squared from n = 0 to n = 3 divided by the 
        theta constant squared at n = 0 (the first element is therefore always 1).

    Returns
    -------
    thetaProducts : list
        A list containing 16 elements, theta_{a, b}(z, t).

    """

    thetaProducts = matrix([
        [a[0] * b[0], a[0] * b[1], a[0] * b[2], a[0] * b[3]],
        [a[1] * b[1], a[1] * b[0], a[1] * b[3], a[1] * b[2]],
        [a[2] * b[2], a[2] * b[3], a[2] * b[0], a[2] * b[1]],
        [a[3] * b[3], a[3] * b[2], a[3] * b[1], a[3] * b[0]]
    ])
    hadam = hadamard_matrix(a[0], 2)
    thetaProducts = (hadam * thetaProducts).tolist()
    thetaProducts =  [item for sublist in thetaProducts for item in sublist]
    thetaProducts = [elem / 4 for elem in thetaProducts]
   
    # Apply the sign change for the odd theta-constants
    for i in [5, 7, 10, 11, 13, 14]:
        thetaProducts[i] = -thetaProducts[i]
    
    return thetaProducts

def to_inverse(a, b, z, t):
    """
    Computes the values of z and tau at a high precision given the theta constants, theta
    functions (evaluated at z), z, and the Riemann matrix tau.

    Parameters
    ----------
    a : list
        A list containing the four theta functions squared from n = 0 to n = 3 evaluated at z
        divided by the theta function squared at n = 0 (the first element is therefore always
        1).
    b : list
        A list containing the four theta constants squared from n = 0 to n = 3 divided by the 
        theta constant squared at n = 0 (the first element is therefore always 1).
    z : list
        A list containing the values of the vector z.
    t : matrix
        The riemann matrix tau of the Riemann surface.

    Returns
    -------
    list
        A list containing 6 elements: the first two are the values of 1/the computed z1 and
        1/the computed z2, the third is 1/the third formula in [], and the last three are
        the computed values of the Riemann matrix tau1, tau2, and tau3.

    """

    n = len(a)

    # First step: compute 1/theta00(z)**2, 1/theta00(0)**2
    theta00z, theta000 = agm_prime(a, b, z, t)
    theta00z = 1 / theta00z
    theta000 = 1 / theta000

    # Then compute the other ones
    thetaZwithAequals0 = [a[i] * theta00z for i in range(4)]
    theta0withAequals0 = [b[i] * theta000 for i in range(4)]

    # Then compute everything at 2tau (simpler conceptually and generalizable to genus g)
    rootA = [mp.sqrt(thetaZwithAequals0[i]) * sign_theta(i, z, t) for i in range(n)]
    rootB = [mp.sqrt(theta0withAequals0[i]) * sign_theta(i, mp.matrix([[0], [0]]), t) for i in range(n)]
    sixteenThetas = all_duplication(rootA, rootB)
    sixteenThetaConstants = all_duplication(rootB, rootB)

    # Then give it to the Borchardt mean
    tt = 2 * t
    jm1 = mp.matrix([[-1 - 1 / tt[0, 0], -tt[0, 1] / tt[0, 0]],
                     [-tt[0, 1] / tt[0, 0], tt[1, 1] - tt[0, 1] ** 2 / tt[0, 0]]])
    jm2 = mp.matrix([[tt[0, 0] - tt[1, 0] ** 2 / tt[1, 1], -tt[0, 1] / tt[1, 1]],
                     [-tt[0, 1] / tt[1, 1], -1 - 1 / tt[1, 1]]])
    detdetdet = -mp.det(tt)
    jm3 = mp.matrix([[tt[0, 0] / detdetdet, (tt[0, 0] * tt[1, 1] - tt[0, 1] ** 2 - tt[0, 1]) / detdetdet],
                     [(tt[0, 0] * tt[1, 1] - tt[0, 1] ** 2 - tt[0, 1]) / detdetdet, tt[1, 1] / detdetdet]])
    zz1 = mp.matrix([[z[0, 0] / tt[0, 0]], [z[0, 0] * tt[0, 1] / tt[0, 0] - z[1, 0]]])
    zz2 = mp.matrix([[z[1, 0] * tt[0, 1] / tt[1, 1] - z[0, 0]], [z[1, 0] / tt[1, 1]]])
    zz3 = mp.zeros(2, 1)

    lambda1, mu1 = agm_prime(mp.matrix([[mp.mpf(1)], [sixteenThetas[9] / sixteenThetas[8]], 
                                       [sixteenThetas[0] / sixteenThetas[8]], [sixteenThetas[1] / sixteenThetas[8]]]),
                            mp.matrix([[mp.mpf(1)], [sixteenThetaConstants[9] / sixteenThetaConstants[8]], 
                                       [sixteenThetaConstants[0] / sixteenThetaConstants[8]], 
                                       [sixteenThetaConstants[1] / sixteenThetaConstants[8]]]), zz1, jm1)

    lambda2, mu2 = agm_prime(mp.matrix([[mp.mpf(1)], [sixteenThetas[0] / sixteenThetas[4]], 
                                       [sixteenThetas[6] / sixteenThetas[4]], [sixteenThetas[2] / sixteenThetas[4]]]),
                            mp.matrix([[mp.mpf(1)], [sixteenThetaConstants[0] / sixteenThetaConstants[4]], 
                                       [sixteenThetaConstants[6] / sixteenThetaConstants[4]], 
                                       [sixteenThetaConstants[2] / sixteenThetaConstants[4]]]), zz2, jm2)

    lambda3, mu3 = agm_prime(mp.matrix([[mp.mpf(1)], [sixteenThetas[8] / sixteenThetas[0]], 
                                       [sixteenThetas[4] / sixteenThetas[0]], [sixteenThetas[12] / sixteenThetas[0]]]),
                            mp.matrix([[mp.mpf(1)], [sixteenThetaConstants[8] / sixteenThetaConstants[0]], 
                                       [sixteenThetaConstants[4] / sixteenThetaConstants[0]], 
                                       [sixteenThetaConstants[12] / sixteenThetaConstants[0]]]), zz3, jm3)

    # Then extract the variables
    computedtau1 = mp.j / (mu1 * sixteenThetaConstants[8])
    computedtau2 = mp.j / (mu2 * sixteenThetaConstants[4])
    computedtau3 = 1 / (mu3 * sixteenThetaConstants[0])
    
    # Because we used duplication formulas 
    computedtau1 /= 2
    computedtau2 /= 2
    computedtau3 /= 4
    
    # Newton on lambda to avoid computing logs
    computedz1 = lambda1 * (-mp.j * 2 * computedtau1 * sixteenThetas[8])
    computedz2 = lambda2 * (-mp.j * 2 * computedtau2 * sixteenThetas[4])
       
    # Don't forget we're still working with 2*tau at this point
    det2tau = 1 / (-mu3 * sixteenThetaConstants[0]) 
    thirdformula = lambda3 * det2tau * sixteenThetas[0]  

    return [1 / computedz1, 1 / computedz2, 1 / thirdformula, computedtau1, computedtau2, computedtau3]

def diff_finies_one_step(a, b, z, t, lambda_iwant, det_iwant):
    """
    Computes the changes in the theta functions <a> and theta constants <b>.

    This is done by computing the Jacobian matrix of the to_inverse base results, as well as 
    perturbations determined by to_inverse with small epsilon additions.

    Parameters
    ----------
    a : list
        A list containing the four theta functions squared from n = 0 to n = 3 evaluated at z
        divided by the theta function squared at n = 0 (the first element is therefore always
        1).
    b : list
        A list containing the four theta constants squared from n = 0 to n = 3 divided by the 
        theta constant squared at n = 0 (the first element is therefore always 1).
    z : list
        A list containing the values of the vector z.
    t : matrix
        The riemann matrix tau of the Riemann surface.
    lambda_iwant : list
        A list containing the 3 values of lambda to be computed (the ones that produce the
        results we wish to find). 
    det_iwant : complex
        The determinant of the Riemann matrix to be computed in the process.

    Returns
    -------
    new_a : list
        A list containing the four theta functions from n = 0 to n = 3 evaluated at z after
        the updates.
    new_b : list
        A list containing the four theta constants from n = 0 to n = 3 after the updates.

    """

    # Set the precision
    prec = mp.prec
    mp.prec = 2 * prec

    # Convert a and b to complex field with higher precision
    newa = [mp.mpc(f"{mp.re(a[i])}", f"{mp.im(a[i])}") for i in range(len(a))]
    newb = [mp.mpc(f"{mp.re(b[i])}", f"{mp.im(b[i])}") for i in range(len(b))]
    
    my_a_of_size_3 = [newa[i] / newa[0] for i in range(1, len(newa))]
    my_b_of_size_3 = [newb[i] / newb[0] for i in range(1, len(newb))]
    
    epsilon = mp.mpc(f"10e{-prec}")
    
    # Compute the base to_inverse result
    to_inverse_base = to_inverse(
        mp.matrix([[mp.mpc(1)], [my_a_of_size_3[0]], [my_a_of_size_3[1]], [my_a_of_size_3[2]]]),
        mp.matrix([[mp.mpc(1)], [my_b_of_size_3[0]], [my_b_of_size_3[1]], [my_b_of_size_3[2]]]),
        z,
        t
    )
    
    perturb = []
    
    perturb.append(
        to_inverse(
            mp.matrix([[mp.mpc(1)], [my_a_of_size_3[0] + epsilon], [my_a_of_size_3[1]], [my_a_of_size_3[2]]]),
            mp.matrix([[mp.mpc(1)], [my_b_of_size_3[0]], [my_b_of_size_3[1]], [my_b_of_size_3[2]]]),
            z,
            t
        )
    )
    perturb.append(
        to_inverse(
            mp.matrix([[mp.mpc(1)], [my_a_of_size_3[0]], [my_a_of_size_3[1] + epsilon], [my_a_of_size_3[2]]]),
            mp.matrix([[mp.mpc(1)], [my_b_of_size_3[0]], [my_b_of_size_3[1]], [my_b_of_size_3[2]]]),
            z,
            t
        )
    )
    perturb.append(
        to_inverse(
            mp.matrix([[mp.mpc(1)], [my_a_of_size_3[0]], [my_a_of_size_3[1]], [my_a_of_size_3[2] + epsilon]]),
            mp.matrix([[mp.mpc(1)], [my_b_of_size_3[0]], [my_b_of_size_3[1]], [my_b_of_size_3[2]]]),
            z,
            t
        )
    )
    perturb.append(
        to_inverse(
            mp.matrix([[mp.mpc(1)], [my_a_of_size_3[0]], [my_a_of_size_3[1]], [my_a_of_size_3[2]]]),
            mp.matrix([[mp.mpc(1)], [my_b_of_size_3[0] + epsilon], [my_b_of_size_3[1]], [my_b_of_size_3[2]]]),
            z,
            t
        )
    )
    perturb.append(
        to_inverse(
            mp.matrix([[mp.mpc(1)], [my_a_of_size_3[0]], [my_a_of_size_3[1]], [my_a_of_size_3[2]]]),
            mp.matrix([[mp.mpc(1)], [my_b_of_size_3[0]], [my_b_of_size_3[1] + epsilon], [my_b_of_size_3[2]]]),
            z,
            t
        )
    )
    perturb.append(
        to_inverse(
            mp.matrix([[mp.mpc(1)], [my_a_of_size_3[0]], [my_a_of_size_3[1]], [my_a_of_size_3[2]]]),
            mp.matrix([[mp.mpc(1)], [my_b_of_size_3[0]], [my_b_of_size_3[1]], [my_b_of_size_3[2] + epsilon]]),
            z,
            t
        )
    )
   
    # Compute the Jacobian matrix
    jacobian = mp.matrix(6, 6)
    for i in range(6):
        for j in range(6):
            jacobian[i, j] = (perturb[j][i] - to_inverse_base[i]) / epsilon
    
    # Compute the changes needed
    changement = mp.matrix([[
        to_inverse_base[0] - lambda_iwant[0],
        to_inverse_base[1] - lambda_iwant[1],
        to_inverse_base[2] - lambda_iwant[2],
        to_inverse_base[3] - t[0, 0],
        to_inverse_base[4] - t[1, 1],
        to_inverse_base[5] - det_iwant
    ]])

    changement = changement * (jacobian.T)**-1
    
    # Return the new a and b matrices with updated values
    new_a = mp.matrix([
        1,
        my_a_of_size_3[0] - changement[0, 0],
        my_a_of_size_3[1] - changement[0, 1],
        my_a_of_size_3[2] - changement[0, 2]
    ])
    
    new_b = mp.matrix([
        1,
        my_b_of_size_3[0] - changement[0, 3],
        my_b_of_size_3[1] - changement[0, 4],
        my_b_of_size_3[2] - changement[0, 5]
    ])
   
    # Reset precision
    mp.prec = prec

    return new_a, new_b

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

    return (2 * mp.pi * mp.j)**count
