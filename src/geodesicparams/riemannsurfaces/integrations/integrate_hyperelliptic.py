from mpmath import quad, matrix, exp, pi, atan2, cos, sin, mpc, mpf, binomial
from sympy import re, Symbol, collect, degree, lambdify, sqrt

from ...utilities import inlist, separate_zeros

def int_genus2_real_exp(zeros, lower, upper, exponent, branch, digits):
     
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
        realNS, complexNS = separate_zeros(zeros)
        if inlist(lb, realNS) + 1 == inlist(ub, realNS):
            if inlist(lb, zeros) + 1 == inlist(ub, zeros):
                tag = (lb + ub) / 2
            else:
                tag = re(zeros[inlist(lb, zeros) + 1])
            return sign * (int_genus2_real_exp(zeros, lb, tag, exponent, branch, digits) + int_genus2_real_exp(zeros, tag, ub, exponent, branch, digits))
        else:
            raise ValueError("Invalid use")

    if k == -1 and l >= 0:
        g = 1
        x = Symbol("x")
        for i in range(l):
            g *= (x - zeros[i])
        for i in range(l + 1, 5):
            g *= (zeros[i] - x)
        g = collect(g.expand(), x)

        if (l + 1) % 2 > 0:
            eval_branch = exp(-pi*1j*(branch+1/2))
        else:
            eval_branch = exp(-pi*1j*branch)
        u = lambda x : -2 * sqrt(ub - x)
    elif l == -1 and k >= 0:
        g = 1
        x = Symbol("x")
        for i in range(k):
            g *= (x - zeros[i])
        for i in range(k + 1, 5):
            g *= (zeros[i] - x)
        g = collect(g.expand(), x)

        if (k + 2) % 2 > 0:
            eval_branch = exp(-pi*1j*(branch+1/2))
        else:
            eval_branch = exp(-pi*1j*branch)
        u = lambda x : 2 * sqrt(x - lb)
    else:
        raise ValueError("Invalid use")

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

    h = -1 * quad(lambda x : vPrime(x) * u(x), [lb, ub], method = methods[i])
    while isinstance(h, (mpf, mpc)) == False and dig > digits - 5:
        while isinstance(h, (mpf, mpc)) == False and i < 2:
            h = -1 * quad(lambda x : vPrime(x) * u(x), [lb, ub], method = methods[i])
            i += 1
        dig = dig -1
        i = 0

    if isinstance(h, (mpf, mpc)) == False:
        raise ValueError("Integration failed")
    if dig < digits - 1:
        print(f"WARNING in int_genus2_real_exp: digits for integration reduced to {dig+1}")

    return sign * eval_branch * (partInt + h)

def int_genus2_complex_exp(zeros, realPart, imaPart, position, exponent, branch, digits):
    x = Symbol("x")
    t = [i for i in [0, 1, 2, 3, 4] if i not in [position, position + 1]]

    g = 1
    for i in t:
        g *= (x - zeros[i])

    g = collect(g.expand(), x)

    coeffsg = [re(g.coeff(x, i)) for i in range(4)]
    iQ = lambda t : -coeffsg[3] * t**3 + t * (3 * coeffsg[3] * realPart**2 + 2 * coeffsg[2] * realPart + coeffsg[1])
    rQ = lambda t : -t**2 * (3 * coeffsg[3] * realPart + coeffsg[2]) + coeffsg[3] * realPart**3 + coeffsg[2] * realPart**2 + coeffsg[1] * realPart + coeffsg[0]
    iQPrime = lambda t : -3 * coeffsg[3] * t**2 + (3 * coeffsg[3] * realPart**2 + 2 * coeffsg[2] * realPart + coeffsg[1])
    rQPrime = lambda t : -2 * t * (3 * coeffsg[3] * realPart + coeffsg[2]) 

    methods = ["tanh-sinh", "gauss-legendre"]

    if exponent == 0:
        if rQ(0) > 0:
            partInt = (2 / sqrt(rQ(0))).evalf()
        else:
            if iQ(imaPart / 100) < 0:
                partInt = (2 / sqrt(-rQ(0)) * 1j).evalf()
            else:
                partInt = (2 / sqrt(-rQ(0)) * (-1j)).evalf()
    else:
        partInt = 0

    dig = digits
    i = 0
    a1 = -1 * quad(lambda x : x**(exponent) * sqrt(imaPart-x) / sqrt(x+imaPart)**3 
        *((rQ(x))**2+(iQ(x))**2)**(-1/4) * cos(-1/2*atan2(iQ(x), rQ(x))), [0, imaPart], method = methods[i])

    while isinstance(a1, (mpc, mpf)) == False and dig > digits - 5:
        while isinstance(a1, (mpc, mpf)) == False and i < 2:
            a1 = -1 * quad(lambda x : x**(exponent) * sqrt(imaPart-x) / sqrt(x+imaPart)**3 
                *((rQ(x))**2+(iQ(x))**2)**(-1/4) * cos(-1/2*atan2(iQ(x), rQ(x))), [0, imaPart], method = methods[i])
            i += 1
        dig = dig -1
        i = 0

    if isinstance(a1, (mpf, mpc)) == False:
        raise ValueError("First integration failed")
    if dig < digits - 1:
        print(f"WARNING in int_genus2_complex_exp: digits for first integration reduced to {dig+1}")

    dig = digits
    i = 0
    a2 = -1j * quad(lambda x : x**(exponent) * sqrt(imaPart-x) / sqrt(x+imaPart)**3 
        *((rQ(x))**2+(iQ(x))**2)**(-1/4) * sin(-1/2*atan2(iQ(x),rQ(x))), [0, imaPart], method = methods[i])

    while isinstance(a2, (mpc, mpf)) == False and dig > digits - 5:
        while isinstance(a2, (mpc, mpf)) == False and i < 2:
            a2 = -1j * quad(lambda x : x**(exponent) * sqrt(imaPart-x) / sqrt(x+imaPart)**3 
                *((rQ(x))**2+(iQ(x))**2)**(-1/4) * sin(-1/2*atan2(iQ(x),rQ(x))), [0, imaPart], method = methods[i])
            i += 1
        dig = dig -1
        i = 0

    if isinstance(a2, (mpf, mpc)) == False:
        raise ValueError("Second integration failed")
    if dig < digits - 1:
        print(f"WARNING in int_genus2_complex_exp: digits for second integration reduced to {dig+1}")

    a = a1 + a2

    dig = digits
    i = 0
    b1 = -1 * quad(lambda x : x**(exponent)*sqrt(imaPart-x)/sqrt(x+imaPart) * ((rQ(x))**2+(iQ(x))**2)**(-3/4)
    *(cos(-3/2*atan2(iQ(x),rQ(x)))*rQPrime(x) -sin(-3/2*atan2(iQ(x),rQ(x)))*iQPrime(x)), [0, imaPart], method = methods[i])

    while isinstance(b1, (mpf, mpc)) == False and dig > digits - 5:
        while isinstance(b1, (mpc, mpf)) == False and i < 2:
            b1 = -1 * quad(lambda x : x**(exponent)*sqrt(imaPart-x)/sqrt(x+imaPart) * ((rQ(x))**2+(iQ(x))**2)**(-3/4)
                *(cos(-3/2*atan2(iQ(x),rQ(x)))*rQPrime(x) -sin(-3/2*atan2(iQ(x),rQ(x)))*iQPrime(x)), [0, imaPart], method = methods[i])
            i += 1
        dig = dig -1
        i = 0

    if isinstance(b1, (mpf, mpc)) == False:
        raise ValueError("Third integration failed")
    if dig < digits - 1:
        print(f"WARNING in int_genus2_complex_exp: digits for third integration reduced to {dig+1}")

    dig = digits
    i = 0
    b2 = -1j * quad(lambda x : x**(exponent)*sqrt(imaPart-x)/sqrt(x+imaPart) * ((rQ(x))**2+(iQ(x))**2)**(-3/4)
    *(sin(-3/2*atan2(iQ(x),rQ(x)))*rQPrime(x) + cos(-3/2*atan2(iQ(x),rQ(x)))*iQPrime(x)), [0, imaPart], method = methods[i])

    while isinstance(b2, (mpc, mpf)) == False and dig > digits - 5:
        while isinstance(b2, (mpf, mpc)) == False and i < 2:
            b2 = -1j * quad(lambda x : x**(exponent)*sqrt(imaPart-x)/sqrt(x+imaPart) * ((rQ(x))**2+(iQ(x))**2)**(-3/4)
                *(sin(-3/2*atan2(iQ(x),rQ(x)))*rQPrime(x) + cos(-3/2*atan2(iQ(x),rQ(x)))*iQPrime(x)), [0, imaPart], method = methods[i])
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
        dig = digits
        i = 0
        c1 = 2*exponent * quad(lambda x : x**(exponent-1)*sqrt(imaPart-x)/sqrt(x+imaPart) *((rQ(x))**2+(iQ(x))**2)**(-1/4)
            *cos(-1/2*atan2(iQ(x),rQ(x))), [0, imaPart], method = methods[i])

        while isinstance(c1, (mpc, mpf)) == False and dig > digits - 5:
            while isinstance(c1, (mpc, mpf)) == False and i < 2:
                c1 = 2*exponent * quad(lambda x : x**(exponent-1)*sqrt(imaPart-x)/sqrt(x+imaPart) *((rQ(x))**2+(iQ(x))**2)**(-1/4)
                    *cos(-1/2*atan2(iQ(x),rQ(x))), [0, imaPart], method = methods[i])
                i += 1 
            dig = dig -1
            i = 0

        if isinstance(c1, (mpc, mpf)) == False:
            raise ValueError("Fifth integration failed")
        if dig < digits - 1:
            print(f"WARNING in int_genus2_complex_exp: digits for fifth integration reduced to {dig+1}")

        dig = digits
        i = 0
        c2 = 2*exponent * 1j * quad(lambda x : x**(exponent-1)*sqrt(imaPart-x)/sqrt(x+imaPart) *((rQ(x))**2+(iQ(x))**2)**(-1/4)
            *sin(-1/2*atan2(iQ(x),rQ(x))), [0, imaPart], method = methods[i])
        while isinstance(c1, (mpc, mpf)) == False and dig > digits - 5:
            while isinstance(c1, (mpc, mpf)) == False and i < 2:
                c2 = 2*exponent * 1j * quad(lambda x : x**(exponent-1)*sqrt(imaPart-x)/sqrt(x+imaPart) *((rQ(x))**2+(iQ(x))**2)**(-1/4)
                    *sin(-1/2*atan2(iQ(x),rQ(x))), [0, imaPart], method = methods[i])
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
    return matrix([int_genus2_real_exp(zeros, lower, upper, 0, branch, digits), int_genus2_real_exp(zeros, lower, upper, 1, branch, digits)])

def int_genus2_complex(zeros, realPart, imaPart, position, branch, digits):
    a = int_genus2_complex_exp(zeros, realPart, imaPart, position, 0, branch, digits)
    b = int_genus2_complex_exp(zeros, realPart, imaPart, position, 1, branch, digits)
 
    return matrix([a, realPart * a + 1j * b])

def myint_genus2_second(zeros, differential, lower, upper, branch, digits):
    result = 0

    for i in range(len(differential)):
        result += differential[i] * int_genus2_real_exp(zeros, lower, upper, i, branch, digits)
    return result

def int_genus2_complex_second(zeros, differential, realPart, imaPart, position, branch, digits):
    
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
