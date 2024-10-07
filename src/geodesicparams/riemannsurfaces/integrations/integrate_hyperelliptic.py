#!/usr/bin/env python
"""
Procedures for performing integrations of holomorphic and meromorphic differentials.

NOTE: fix some things here like the precision and documentation. Also integration method
for periods of second kind will be changed soon (hopefully).

"""

from mpmath import quad, matrix, exp, pi, atan2, cos, sin, mpc, mpf, binomial, mp, im, fabs
from sympy import re, Symbol, collect, degree, lambdify, sqrt, oo

from ..riemann_funcs.hyperelp_funcs import kleinian_zeta
from ...utilities import inlist, separate_zeros

def int_genus2_real_exp(zeros, lower, upper, exponent, branch, digits):
    """
    Integrates a holomorphic differential z**j / sqrt(P(z)) from <lower> to <upper>,
    where at least one of <lower> or <upper> is a real zero of the polynomial P(z).

    Parameters
    ----------
    zeros : list
        The zeros of the polynomial P(z).
    lower : float
        The lower bound of the integration.
    upper : float
        The upper bound of the integration.
    exponent : int
        The exponent j in the differential z**j / sqrt(P(z)), where j = 0 or 1.
    branch : int
        The branch of the sqrt(P(z)).
    digits : int
        The number of digits used in the computation.

    Returns
    -------
    complex
        The value of the integral.

    """
    
    if lower == upper:
        return 0
    elif lower > upper:
        lb = upper
        ub = lower
        sign = -1
    else:
        lb = lower
        ub = upper
        sign = 1

    k = inlist(lb, zeros)
    l = inlist(ub, zeros)

    if k >= 0 and l >= 0:
        realNS = separate_zeros(zeros)[0]
        if inlist(lb, realNS) + 1 == inlist(ub, realNS):
            if inlist(lb, zeros) + 1 == inlist(ub, zeros):
                tag = (lb + ub) / 2
            else:
                tag = re(zeros[inlist(lb, zeros) + 1])
                    return sign * (int_genus2_real_exp(zeros, lb, tag, exponent, branch, digits) + int_genus2_real_exp(zeros, tag, ub, exponent, branch, digits))
        else:
            raise ValueError("Invalid use")

    if k == -1 and l >= 0:
        # Integration from lb to zeros[l]
        # g is real and positive for real x in (zeros[l - 1], zeros[l]]
        g = 1
        x = Symbol("x")
        for i in range(l):
            g *= (x - zeros[i])
        for i in range(l + 1, 5):
            g *= (zeros[i] - x)
        g = collect(g.expand(), x)

        # Branch factor
        if (l + 1) % 2 > 0:
            eval_branch = exp(-pi * 1j * (branch + 1/2))
        else:
            eval_branch = exp(-pi * 1j * branch)
        u = lambda x : -2 * sqrt(ub - x)
    elif l == -1 and k >= 0:
        # Integration from zeros[k] to ub
        # g is real and positive for real x in [zeros[k], zeros[k + 1]]
        g = 1
        x = Symbol("x")
        for i in range(k):
            g *= (x - zeros[i])
        for i in range(k + 1, 5):
            g *= (zeros[i] - x)
        g = collect(g.expand(), x)

        # Branch factor
        if (k + 2) % 2 > 0:
            eval_branch = exp(-pi * 1j * (branch + 1/2))
        else:
            eval_branch = exp(-pi * 1j * branch)
        u = lambda x : 2 * sqrt(x - lb)
    else:
        raise ValueError("Invalid use")

    # Integration by parts
    q = 0
    qPrime = 0

    for i in range(0, degree(g) + 1):
        q += re(g.coeff(x, i)) * x**i
    for i in range(1, degree(g) + 1):
        qPrime += i * re(g.coeff(x, i)) * x**(i - 1)
   
    v = lambdify(x, x**(exponent) / sqrt(q), "sympy")
    vPrime = lambdify(x, -x**(exponent) / (2 * sqrt(q)**3) * qPrime + exponent * x**(exponent - 1) / sqrt(q), "sympy")
    partInt = (u(ub) * v(ub) - u(lb) * v(lb)).evalf()
    dig = digits
    methods = ["tanh-sinh", "gauss-legendre"]
    i = 0

    # Integration
    h = -1 * quad(lambda x : vPrime(x) * u(x), [lb, ub], method = methods[i])
    while isinstance(h, (mpf, mpc)) == False and dig > digits - 5:
        while isinstance(h, (mpf, mpc)) == False and i < 2:
            h = -1 * quad(lambda x : vPrime(x) * u(x), [lb, ub], method = methods[i])
            i += 1
        dig = dig -1
        i = 0

    # Check if the result of the integration is not a real or complex number
    if isinstance(h, (mpf, mpc)) == False:
        raise ValueError("Integration failed")
    if dig < digits - 1:
        print(f"WARNING in int_genus2_real_exp: digits for integration reduced to {dig + 1}")

    return sign * eval_branch * (partInt + h)

def int_genus2_complex_exp(zeros, realPart, imaPart, position, exponent, branch, digits):
    """
    Integrates a differential of the form It**j dt/ sqrt(P(<realPart> + It)) from t = 0 to
    t = <imaPart>, where I is the imaginary unit.

    This is needed to integrate canonical differentials of first and second kind.

    Parameters
    ----------
    zeros : list
        The zeros of the polynomial P.
    realPart : float
        The lower bound of the integration.
    imaPart : float
        The upper bound of the integration (assumed to be positive).
    position : int
        The index at which <realPart> - I*<imaPart> is in <zeros>.
    exponent : int
        The exponent j in the differential It**j dt/ sqrt(P(<realPart> + It)), where j = 0 
        or 1.
    branch : int
        The branch of the sqrt(P).
    digits : int
        The number of digits used in the computation.

    Returns
    -------
    complex
        The value of the integral.

    """

    x = Symbol("x")
    t = [i for i in [0, 1, 2, 3, 4] if i not in [position, position + 1]]

    g = 1
    for i in t:
        g *= (x - zeros[i])

    g = collect(g.expand(), x)

    # g has real coefficients: remove +0 * I
    coeffsg = [re(g.coeff(x, i)) for i in range(4)]

    # Separate real and imaginary parts of g and derivative of g
    iQ = lambda t : -coeffsg[3] * t**3 + t * (3 * coeffsg[3] * realPart**2 + 2 * coeffsg[2] * realPart + coeffsg[1])
    rQ = lambda t : -t**2 * (3 * coeffsg[3] * realPart + coeffsg[2]) + coeffsg[3] * realPart**3 + coeffsg[2] * realPart**2 + coeffsg[1] * realPart + coeffsg[0]
    iQPrime = lambda t : -3 * coeffsg[3] * t**2 + (3 * coeffsg[3] * realPart**2 + 2 * coeffsg[2] * realPart + coeffsg[1])
    rQPrime = lambda t : -2 * t * (3 * coeffsg[3] * realPart + coeffsg[2]) 

    methods = ["tanh-sinh", "gauss-legendre"]

    """
    For the computation of partInt you need to be careful for negative rQ(0):
    the arctan function is discontinuous if iQ(t) is negative for small t:
    it jumps from Pi for t = 0 to -pi + eps for t>0.
    Therefore, atan2(iQ(0), rQ(0)) has to be set to -pi if iQ(t) < 0 for small t.
    (This is not a problem for the integration as the discontinuity is a null set.
    """
    if exponent == 0:
        if rQ(0) > 0:
            # iQ(0) = 0
            partInt = (2 / sqrt(rQ(0))).evalf()
        else:
            if iQ(imaPart / 100) < 0:
                partInt = (2 / sqrt(-rQ(0)) * 1j).evalf()
            else:
                partInt = (2 / sqrt(-rQ(0)) * (-1j)).evalf()
    else:
        partInt = 0

    # First real part
    dig = digits
    i = 0
    a1 = -1 * quad(lambda x : x**(exponent) * sqrt(imaPart - x) / sqrt(x + imaPart)**3 
        * ((rQ(x))**2 + (iQ(x))**2)**(-1/4) * cos(-1/2 * atan2(iQ(x), rQ(x))), [0, imaPart], 
                   method = methods[i])

    # Integration
    while isinstance(a1, (mpc, mpf)) == False and dig > digits - 5:
        while isinstance(a1, (mpc, mpf)) == False and i < 2:
            a1 = -1 * quad(lambda x : x**(exponent) * sqrt(imaPart - x) / sqrt(x+imaPart)**3 
                *((rQ(x))**2 + (iQ(x))**2)**(-1/4) * cos(-1/2 * atan2(iQ(x), rQ(x))), 
                           [0, imaPart], method = methods[i])
            i += 1
        dig = dig -1
        i = 0

    if isinstance(a1, (mpf, mpc)) == False:
        raise ValueError("First integration failed")
    if dig < digits - 1:
        print(f"WARNING in int_genus2_complex_exp: digits for first integration reduced to {dig+1}")

    # First imaginary part
    dig = digits
    i = 0
    a2 = -1j * quad(lambda x : x**(exponent) * sqrt(imaPart - x) / sqrt(x + imaPart)**3 
        * ((rQ(x))**2 + (iQ(x))**2)**(-1/4) * sin(-1/2 * atan2(iQ(x), rQ(x))), [0, imaPart], 
                    method = methods[i])

    # Integration
    while isinstance(a2, (mpc, mpf)) == False and dig > digits - 5:
        while isinstance(a2, (mpc, mpf)) == False and i < 2:
            a2 = -1j * quad(lambda x : x**(exponent) * sqrt(imaPart - x) / sqrt(x + imaPart)**3 
                * ((rQ(x))**2 + (iQ(x))**2)**(-1/4) * sin(-1/2 * atan2(iQ(x), rQ(x))), 
                            [0, imaPart], method = methods[i])
            i += 1
        dig = dig -1
        i = 0

    if isinstance(a2, (mpf, mpc)) == False:
        raise ValueError("Second integration failed")
    if dig < digits - 1:
        print(f"WARNING in int_genus2_complex_exp: digits for second integration reduced to {dig+1}")

    a = a1 + a2

    # Second real part
    dig = digits
    i = 0
    b1 = -1 * quad(lambda x : x**(exponent) * sqrt(imaPart - x)/sqrt(x + imaPart) * 
        ((rQ(x))**2 + (iQ(x))**2)**(-3/4) * (cos(-3/2*atan2(iQ(x), rQ(x))) * rQPrime(x) - 
            sin(-3/2 * atan2(iQ(x), rQ(x))) * iQPrime(x)), [0, imaPart], method = methods[i])

    # Integration
    while isinstance(b1, (mpf, mpc)) == False and dig > digits - 5:
        while isinstance(b1, (mpc, mpf)) == False and i < 2:
            b1 = -1 * quad(lambda x : x**(exponent) * sqrt(imaPart - x)/sqrt(x + imaPart) * 
                ((rQ(x))**2 + (iQ(x))**2)**(-3/4) * (cos(-3/2 * atan2(iQ(x), rQ(x))) * 
                    rQPrime(x) - sin(-3/2 * atan2(iQ(x), rQ(x))) * iQPrime(x)), 
                           [0, imaPart], method = methods[i])
            i += 1
        dig = dig -1
        i = 0

    if isinstance(b1, (mpf, mpc)) == False:
        raise ValueError("Third integration failed")
    if dig < digits - 1:
        print(f"WARNING in int_genus2_complex_exp: digits for third integration reduced to {dig+1}")

    # Second imaginary part
    dig = digits
    i = 0
    b2 = -1j * quad(lambda x : x**(exponent) * sqrt(imaPart - x)/sqrt(x + imaPart) * 
        ((rQ(x))**2 + (iQ(x))**2)**(-3/4) * (sin(-3/2 * atan2(iQ(x), rQ(x))) * rQPrime(x) + 
            cos(-3/2 * atan2(iQ(x), rQ(x))) * iQPrime(x)), [0, imaPart], method = methods[i])

    # Integration
    while isinstance(b2, (mpc, mpf)) == False and dig > digits - 5:
        while isinstance(b2, (mpf, mpc)) == False and i < 2:
            b2 = -1j * quad(lambda x : x**(exponent) * sqrt(imaPart - x)/sqrt(x + imaPart) * 
                ((rQ(x))**2 + (iQ(x))**2)**(-3/4) * (sin(-3/2 * atan2(iQ(x), rQ(x))) * 
                    rQPrime(x) + cos(-3/2 * atan2(iQ(x), rQ(x))) * iQPrime(x)), [0, imaPart], 
                            method = methods[i])
            i += 1 
        dig = dig -1
        i = 0

    if isinstance(b2, (mpf, mpc)) == False:
        raise ValueError("Fourth integration failed")
    if dig < digits - 1:
        print(f"WARNING in int_genus2_complex_exp: digits for fourth integration reduced to {dig+1}")

    b = b1 + b2
   
    if exponent == 0:
        c = 0
    else:
        # Third real part
        dig = digits
        i = 0
        c1 = 2 * exponent * quad(lambda x : x**(exponent - 1) * sqrt(imaPart - x)/
            sqrt(x + imaPart) * ((rQ(x))**2 + (iQ(x))**2)**(-1/4) * cos(-1/2 * 
                atan2(iQ(x), rQ(x))), [0, imaPart], method = methods[i])

        # Integration
        while isinstance(c1, (mpc, mpf)) == False and dig > digits - 5:
            while isinstance(c1, (mpc, mpf)) == False and i < 2:
                c1 = 2 * exponent * quad(lambda x : x**(exponent - 1) * sqrt(imaPart - x)
                    /sqrt(x + imaPart) * ((rQ(x))**2 + (iQ(x))**2)**(-1/4) * cos(-1/2 * 
                        atan2(iQ(x), rQ(x))), [0, imaPart], method = methods[i])
                i += 1 
            dig = dig -1
            i = 0

        if isinstance(c1, (mpc, mpf)) == False:
            raise ValueError("Fifth integration failed")
        if dig < digits - 1:
            print(f"WARNING in int_genus2_complex_exp: digits for fifth integration reduced to {dig+1}")

        # Third imaginary part
        dig = digits
        i = 0
        c2 = 2 * exponent * 1j * quad(lambda x : x**(exponent - 1) * sqrt(imaPart - x)/
            sqrt(x + imaPart) * ((rQ(x))**2 + (iQ(x))**2)**(-1/4) * sin(-1/2 * 
                atan2(iQ(x), rQ(x))), [0, imaPart], method = methods[i])

        # Integration
        while isinstance(c1, (mpc, mpf)) == False and dig > digits - 5:
            while isinstance(c1, (mpc, mpf)) == False and i < 2:
                c2 = 2 * exponent * 1j * quad(lambda x : x**(exponent - 1) * sqrt(imaPart - x)
                    /sqrt(x + imaPart) *((rQ(x))**2 + (iQ(x))**2)**(-1/4) * sin(-1/2 * 
                        atan2(iQ(x), rQ(x))), [0, imaPart], method = methods[i])
                i += 1
            dig = dig -1
            i = 0

        if isinstance(c2, (mpc, mpf)) == False:
            raise ValueError("Sixth integration failed")
        if dig < digits - 1:
            print(f"WARNING in int_genus2_complex_exp: digits for sixth integration reduced to {dig+1}")
        
        c = c1 + c2

    return exp(-pi * 1j * branch) * 1j * (partInt + a + b + c)


def myint_genus2(zeros, lower, upper, branch, digits):
    """
    Integrates the vector of canonical holomorphic differentials dz = [1 / sqrt(P(z), 
    z / sqrt(P(z))] from <lower> to <upper>, where at least one of <lower> or <upper> is 
    a real zero of the polynomial P(z).

    Parameters
    ----------
    zeros : list
        The zeros of the polynomial P(z).
    lower : float
        The lower bound of the integration.
    upper : float
        The upper bound of the integration.
    branch : int
        The branch of the sqrt(P(z)).
    digits : int
        The number of digits used in the computation.

    Returns
    -------
    matrix
        The values of the integrals as a 2x1 mpmath matrix.

    """

    return matrix([int_genus2_real_exp(zeros, lower, upper, 0, branch, digits), int_genus2_real_exp(zeros, lower, upper, 1, branch, digits)])

def int_genus2_complex(zeros, realPart, imaPart, position, branch, digits):
    """
    Integrates the vector of canonical holomorphic differentials dz = [1 / sqrt(P(z), 
    z / sqrt(P(z))] from the real part <realPart> of a complex zero of the polynomial P(z) to
    the complex zero <realPart> + I*<imaPart> of P(z), where I is the imaginary unit.

    Parameters
    ----------
    zeros : list
        The zeros of the polynomial P.
    realPart : float
        The lower bound of the integration.
    imaPart : float
        The upper bound of the integration (assumed to be positive).
    position : int
        The index at which <realPart> - I*<imaPart> is in <zeros>.
    branch : int
        The branch of the sqrt(P).
    digits : int
        The number of digits used in the computation.

    Returns
    -------
    matrix
        The values of the integrals as a 2x1 mpmath matrix.

    """

    a = int_genus2_complex_exp(zeros, realPart, imaPart, position, 0, branch, digits)
    b = int_genus2_complex_exp(zeros, realPart, imaPart, position, 1, branch, digits)

    return matrix([a, realPart * a + 1j*b])

def eval_period(m, n, realNS, zeros, omega, component = 0):
    """
    Computes the period from one real zero of the set of zeros defining a Riemann surface 
    of genus 2 to another or to infinity.

    Parameters
    ----------
    m : int
        The index of the zero to compute the period from (realNS[<m>]).
    n : int, oo
        The index of the zero to compute the period to (realNS[<n>]). If <n> is oo, then
        the period if computed from realNS[<m>] to oo.
    realNS : list
        The real zeros of the polynomial defining the genus 2 Riemann surface.
    zeros : list
        The zeros of the polynomial defining the genus 2 Riemann surface.
    omega : matrix
        A 2x4 mpmath matrix such that period_matrix = [<periods_first>, <periods_second>] (see
        above).
    component : int, optional
        The row of the period matrix <omega> that should be used for the computation of the 
        period (can either be 0 or 1, where the default value is 0). 

    Returns
    -------
    result : complex
        The period from realNS[<m>] to realNS[<n>]. 

    """
  
    # Sort <zeros> by periodloops
    lange = len(zeros)
    if lange == 5:
        if len(realNS) == 3:
            if (im(zeros[1]) != 0 and im(zeros[2]) != 0): 
                e = [zeros[1], zeros[2], zeros[0], zeros[3], zeros[4]]
            elif (im(zeros[3]) != 0 and im(zeros[4]) != 0): 
                e = [zeros[0], zeros[1], zeros[3], zeros[4], zeros[2]]
            else:
                e = zeros
        elif len(realNS) == 1:
            if im(zeros[0]) == 0: 
                e = [zeros[1], zeros[2], zeros[3], zeros[4], zeros[0]]
            elif im(zeros[2]) == 0: 
                e  = [zeros[0], zeros[1], zeros[3], zeros[4], zeros[2]]
            else:
                e = zeros
        else:
            e = zeros

    # Sort by m < n
    if m > n:
        k = n
        l = m
        sign = -1
    else:
        k = m
        l = n
        sign = 1

    if k == -oo:
        k = l
        l = oo
        sign = -sign

    # Location of m, n in e
    K = inlist(realNS[k], e)

    if l < len(realNS):
        L = inlist(realNS[l], e)
    else:
        L = lange

    if component != 0 and component != 1:
        print(f"WARNING in eval_period: illegal component {component} changed to 1")
        component = 0

    periodlist = [omega[component, 0], omega[component, 2] - omega[component, 3], omega[component, 1], omega[component, 3], -omega[component,0] - omega[component, 1]]
    result = 0

    # Compute period
    for i in range(K, L):
        result += periodlist[i]
    result = sign * result

    return result

def branch_list_genus2(zeros, num_realNS):
    """
    Determines the branch adjacent to any zero given a list of zeros. 

    Parameters
    ----------
    zeros : list
        The 5 zeros of the polynomial defining the genus 2 Riemann surface.
    num_realNS : list
        The number of real zeros in <zeros>, which may be 1, 3, or 5.

    Returns
    -------
    erg : list
        A list containing 5 lists. Each element of the 5 lists consists of three parts: 
        The first is the branch left of the zero, the second is the zero, and the third is 
        the branch right of the zero. In the case that the second part is a complex zero, the 
        entries depend on the sign of the imaginary part. If it is negative, the first entry 
        is the branch left of the branch cut on the real axis and the third is the branch on 
        the branch cut. If it's positive, the first entry is the branch on the branch cut and 
        the third is the branch right of the branch cut on the real axis. 

    """

    if num_realNS == 5:
        erg = [[1, zeros[0], 1], [1, zeros[1], 0], [0, zeros[2], 0], [0, zeros[3], 1], [1, zeros[4], 1]]
    elif num_realNS == 3:
        # ima2Per1
        if im(zeros[0]) != 0:
            erg = [[1, zeros[0], 1], [1, zeros[1], 0], [0, zeros[2], 0], [0, zeros[3], 1], [1, zeros[4], 1]]
        # ima2Per2
        elif im(zeros[1]) != 0:
            erg = [["tbd", zeros[0], 0], [1, zeros[1], "tbd"], ["tbd", zeros[2], 0], [0, zeros[3], 1], [1, zeros[4], 1]]
        # ima2Per3
        elif im(zeros[2]) != 0:
            rea1 = re(zeros[2])
            if ((rea1-zeros[0]) * (rea1-zeros[1]) + (rea1-zeros[1]) * (rea1-zeros[4]) + (rea1-zeros[4]) * (rea1-zeros[0])).evalf() < 0:
                erg = [[1, zeros[0], 1], [1, zeros[1], 0], [0, zeros[2], 1], [1, zeros[3], 1], [1, zeros[4], 1]]
            else:
                erg = [[1, zeros[0], 1], [1, zeros[1], 0], [0, zeros[2], 0], [0, zeros[3], 1], [1, zeros[4], 1]]
        # ima2Per4
        elif im(zeros[3]) != 0:
            erg = [[1, zeros[0], 1], [1, zeros[1], 0], ["tbd", zeros[2], 1], [1, zeros[3], "tbd"], ["tbd", zeros[4], 1]]
    elif num_realNS == 1:
        # ima4Per1
        if im(zeros[4]) == 0:
            rea1 = re(zeros[0]); rea2 = re(zeros[2])
            ima1 = fabs(im(zeros[0]))
            if ((rea2-rea1)**2 + ima1**2 + 2 * (rea2-rea1) * (rea2-zeros[4])).evalf() < 0:
                erg = [[1, zeros[0], 1], [1, zeros[1], 0], [0, zeros[2], 1], [1, zeros[3], 1], [1, zeros[4], 1]]
            else:
                erg = [[1, zeros[0], 1], [1, zeros[1], 0], [0, zeros[2], 0], [0, zeros[3], 1], [1, zeros[4], 1]]
        # ima4Per2
        elif im(zeros[0]) == 0:
            erg = [["tbd", zeros[0], 1], [1, zeros[1], "tbd"], ["tbd", zeros[2], 1], [1, zeros[3], "tbd"], ["tbd", zeros[4], 1]]
        # ima4Per3
        elif im(zeros[2]) == 0:
            erg = [[1, zeros[0], 1], [1, zeros[1], 0], ["tbd", zeros[2], 1], [1, zeros[3], "tbd"], ["tbd", zeros[4], 1]]
    else:
        raise ValueError("Wrong number of real zeros.")
    return erg

def int_genus2_first(zeros, lower, upper, digits, period_matrix):
    """
    Integrates the vector of canonical holomorphic differentials dz = [1/sqrt(P(z)), 
    z/sqrt(P(z))] from <lower> to <upper>.

    Parameters
    ----------
    zeros : list
        The 5 zeros of the polynomial defining the genus 2 Riemann surface.
    lower : real
        The lower integration bound.
    upper : real
        The upper integration bound.
    digits : int
        The number of digits to use in the computation.
    periodMatrix : matrix, optional
        A 2x4 mpmath matrix such that periodMatrix = [<periods_first>, <periods_second>],
        where <periods_first> is the period matrix of the integral of holomorphic differentals
        taken along the contours around the branch cuts, while <periods_first> is the period
        matrix integrated along the contours connecting branch cuts.

    Returns
    -------
    result : list
        The value of the two integrals as a list.

    """

    if len(zeros) != 5:
        raise Exception("Invalid use; number of zeros has to be 5.")

    e = sorted(zeros, key = lambda x : re(x))
    realNS, complexNS = separate_zeros(e)
    
    if im(lower) != 0 and im(upper) != 0:
        raise ValueError("Invalid use; only real integration bounds are feasible")

    if lower == upper:
        return 0
    elif lower > upper:
        sign = -1
        lb = upper
        ub = lower
    else:
        sign = 1
        lb = lower
        ub = upper

    # ------------------ Case 1: lb and ub are adjacent real zeros
    if inlist(lb, realNS) + 1 == inlist(ub, realNS):
        if inlist(lb, e) + 1 == inlist(ub, e):
            tags = (lb + ub) / 2
        else:
            tags = re(e[inlist(lb, e) + 1])
        branch_list = branch_list_genus2(e, len(realNS))

        return sign * (myint_genus2(e, lb, tags, branch_list[inlist(lb, e)][2], digits) + myint_genus2(e, tags, ub, branch_list[inlist(ub, e)][0], digits))
    # ------------------ Case 2: only one of lb and ub is a real zero
    elif inlist(lb, realNS) >= 0 or inlist(ub, realNS) >= 0:
        if inlist(lb, realNS) == -1:
            # No real zeros between lb and ub
            if inlist(lb, sorted([lb] + realNS, key = lambda x : re(x))) == inlist(ub, realNS):
                return sign * myint_genus2(e, lb, ub, branch_list_genus2(e, len(realNS))[inlist(ub, e)][0], digits)
            else:
                raise ValueError("Invalid bounds")
        # Case 2b: lb is a real zero
        elif inlist(ub, realNS) == -1:
            # No real zeros between lb and ub
            if inlist(ub, sorted([ub] + realNS, key = lambda x : re(x))) == inlist(lb, realNS) + 1:
                return sign * myint_genus2(e, lb, ub, branch_list_genus2(e, len(realNS))[inlist(lb, e)][2], digits)
            else:
                raise ValueError("Invalid bounds")
        else:
            raise ValueError("Invalid bounds")
    # ---------------- Case 3: none of lb or ub is a real zero
    else:
        if inlist(lb, sorted([lb] + realNS, key = lambda x : re(x))) == inlist(ub, sorted([ub] + realNS, key = lambda x : re(x))):
            if ub == oo:
                return sign * (myint_genus2(e, lb, realNS[-1], branch_list_genus2(e, len(realNS))[inlist(realNS[-1], e)][2], digits) + 
                matrix([eval_period(len(realNS) - 1,oo,realNS,e,period_matrix,0), eval_period(len(realNS) - 1,oo,realNS,e,period_matrix,1)]))
            else:
                p = lambda x : (x - e[0]) * (x - e[1]) * (x - e[2]) * (x - e[3]) * (x - e[4])
                sign = sign * exp(pi * 1j * branch_list_genus2(e, len(realNS))[inlist(lb, sorted(e + [lb], key = lambda x : re(x)))][2])
                return matrix([sign * quad(lambda x : 1 / mpc(sqrt(p(x))), [lb, ub]), sign * quad(lambda x : x / mpc(sqrt(p(x))), [lb, ub])])
        else:
            raise ValueError("Invalid bounds")

def int_second_genus2(zeros, pole, periods_first, periods_second, digits, minMax = 5):
    """
    Integrates the vector of canonical meromorphic differentials from the pole in a 
    hyperelliptic integral of the third on the negative branch of the sqrt(P(pole)) to the
    pole on the positive branch of the sqrt(P(pole)), where P(x) is the 5th degree polynomial
    defining the Riemann surface.

    Parameters
    ----------
    zeros : list
        The zeros of the Riemann surface defined by a 5th degree polynomial.
    periods_first : matrix
        A 2x4 mpmath matrix, where the first 2x2 entry is the period matrix from the integral of
        the vector of canonical holomorphic differentials along the contours encircling the branch
        cuts, while the second 2x2 entry is taken along the contours connecting branch cuts.
    periods_second : matrix
        A 2x4 mpmath matrix, where the first 2x2 entry is the period matrix from the integral of
        the vector of canonical meromorphic differentials along the contours encircling the branch
        cuts, while the second 2x2 entry is taken along the contours connecting branch cuts.
    digits : int
        The number of digits used in the computation.
    minMax : natural, optional
        A natural number from 0 <= minMax <= 30, the summation bound of the theta function.

    Returns
    -------
    list
        The vector containing the integration of the canonical meromorphic differentials.

    """

    mp.digits = digits

    # Constants
    riemann_const = [[1/2, 1/2], [0, 1/2]]
    omega = periods_first[0:2, 0:2]
    eta = periods_second[0:2, 0:2]
    eta_prime = periods_second[0:2, 2:4]
    u = int_genus2_first(zeros, zeros[1], pole, digits, periods_first) + matrix(riemann_const)

    # P(pole)
    pole_zeros = 1
    for i in zeros:
        pole_zeros *= (pole - i)

    sym_funcs = sqrt(pole_zeros) / (pole - zeros[3])
    #sym_funcs = matrix([kleinian_P(u, omega, eta, riemann_const, [2, 2, 2], minMax), 0])

    zeta = matrix([kleinian_zeta(u, omega, eta, riemann_const, i, minMax) for i in range(1, 3)])
    eta_char = eta_prime*matrix(riemann_const[0]) - eta*matrix(riemann_const[1]) 

    return -2*zeta + (4 * eta_char) + sym_funcs

def myint_genus2_second(zeros, differential, lower, upper, branch, digits):
    # Subject to change

    result = 0

    for i in range(len(differential)):
        result += differential[i] * int_genus2_real_exp(zeros, lower, upper, i, branch, digits)
    return result

def int_genus2_complex_second(zeros, differential, realPart, imaPart, position, branch, digits):
    # Subject to change

    results = []
    total_sum = 0
    binomial_sum = 0

    for i in range(len(differential)):
        results.append(int_genus2_complex_exp(zeros, realPart, imaPart, position, i, branch, digits))

    for i in range(1, len(differential) + 1):
        binomial_sum = 0
        for j in range(i):
            binomial_sum += binomial(i - 1, j) * realPart**j * 1j ** (i - 1 - j) * results[(i - 1) - j]
        total_sum += differential[i - 1] * binomial_sum

    return total_sum
