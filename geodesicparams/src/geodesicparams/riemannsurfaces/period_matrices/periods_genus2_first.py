from mpmath import quad, matrix, fabs, eig, exp, pi, mpc
from sympy import re, im, Symbol, collect, lambdify, oo, sqrt

from ...utilities import inlist, separate_zeros
from ..integrations.integrate_hyperelliptic import myint_genus2, int_genus2_complex

def periods(realNS, complexNS, digits):
    realNS = sorted(realNS)

    if len(complexNS) == 0:
        return reaPer(realNS, digits)
    else:
        return imaPer(realNS, complexNS, digits)

def reaPer(realNS, digits):
    r = matrix(2, 4)
    h = myint_genus2(realNS, realNS[0], realNS[1], 1, digits)
    r[0, 0] = h[0]
    r[1, 0] = h[1]
    
    h = myint_genus2(realNS, realNS[2], realNS[3], 0, digits)
    r[0, 1] = h[0]
    r[1, 1] = h[1]

    h = myint_genus2(realNS, realNS[3], realNS[4], 1, digits)
    r[0, 3] = h[0]
    r[1, 3] = h[1]

    h = myint_genus2(realNS, realNS[1], realNS[2], 0, digits)
    r[0, 2] = h[0] + r[0, 3]
    r[1, 2] = h[1] + r[1, 3]

    return r

def imaPer(realNS, complexNS, digits):
    if len(complexNS) == 2:
        return ima2Per(realNS, complexNS, digits)
    else:
        return ima4Per(realNS, complexNS, digits)

def ima2Per(realNS, complexNS, digits):

    k = inlist(re(complexNS[0]), sorted(realNS + [re(complexNS[0])], key = lambda x : re(x)))
    if k == 0:
        return ima2Per1(realNS, complexNS, digits)
    elif k == 1:
        return ima2Per2(realNS, complexNS, digits)
    elif k == 2:
        return ima2Per3(realNS, complexNS, digits)
    elif k == 3:
        return ima2Per4(realNS, complexNS, digits)


def ima4Per(realNS, complexNS, digits):
    rea = sorted([re(i) for i in complexNS], key = lambda x : re(x))

    ima1 = fabs(im(complexNS[inlist(rea[0], [re(i) for i in complexNS])]))
    ima2 = fabs(im(complexNS[inlist(rea[2], [re(i) for i in complexNS])]))

    g = realNS[0]

    if rea[3] < g:
        return ima4Per1(realNS, rea[0], ima1, rea[2], ima2, digits)
    elif rea[3] > g and rea[1] < g:
        return ima4Per3(realNS, rea[0], ima1, rea[2], ima2, digits)
    elif rea[3] > g and rea[1] > g:
        return ima4Per2(realNS, rea[0], ima1, rea[2], ima2, digits)

def ima2Per1(realNS, complexNS, digits):
    rea = re(complexNS[0])
    ima = fabs(im(complexNS[0]))
    
    zeros = sorted(realNS + complexNS, key = lambda x : re(x))

    r = matrix(2, 4)

    h = myint_genus2(zeros, zeros[2], zeros[3], 0, digits)
    r[0, 1] = h[0]
    r[1, 1] = h[1]
    
    h = myint_genus2(zeros, zeros[3], zeros[4], 1, digits)
    r[0, 3] = h[0]
    r[1, 3] = h[1]

    a = int_genus2_complex(zeros, rea, ima, 0, -1, digits)
    r[0, 0] = 2 * re(a[0])
    r[1, 0] = 2 * re(a[1])

    h = myint_genus2(zeros, rea, zeros[2], 0, digits)
    r[0, 2] = h[0] + r[0, 3] + a[0]
    r[1, 2] = h[1] + r[1, 3] + a[1]

    return r

def ima2Per2(realNS, complexNS, digits):
    rea = re(complexNS[0])
    ima = fabs(im(complexNS[0]))
    
    zeros = sorted(realNS + complexNS, key = lambda x : re(x))

    r = matrix(2, 4)

    h = myint_genus2(zeros, zeros[3], zeros[4], 1, digits)
    r[0, 3] = h[0]
    r[1, 3] = h[1]
    
    v = myint_genus2(zeros, zeros[0], rea, 0, digits)
    h = myint_genus2(zeros, rea, zeros[3], 0, digits)
    r[0, 1] = v[0] + h[0]
    r[1, 1] = v[1] + h[1]

    a = int_genus2_complex(zeros, rea, ima, 1, -1, digits)
    r[0, 0] = 2 * re(a[0]) - 2 * v[0]
    r[1, 0] = 2 * re(a[1]) - 2 * v[1]

    #print(v[0])
    #print(v[1])
    #h = myint_genus2(zeros, rea, zeros[2], 0, digits)
    r[0, 2] = r[0, 3] - v[0] + a[0]
    r[1, 2] = r[1, 3] - v[1] + a[1]
    #print(r[0, 2])
    #print(r[1, 2])

    return r

def ima2Per3(realNS, complexNS, digits):
    rea = re(complexNS[0])
    ima = fabs(im(complexNS[0]))
    
    zeros = sorted(realNS + complexNS, key = lambda x : re(x))

    if ((rea - zeros[0]) * (rea - zeros[1]) + (rea - zeros[1]) * (rea - zeros[4]) + (rea - zeros[4]) * (rea - zeros[0])).evalf() < 0:
        a = int_genus2_complex(zeros, rea, ima, 2, -1, digits)
    else:
        a = int_genus2_complex(zeros, rea, ima, 2, 0, digits)

    r = matrix(2, 4)

    r[0, 1] = 2 * re(a[0])
    r[1, 1] = 2 * re(a[1])
    
    h = myint_genus2(zeros, zeros[0], zeros[1], 1, digits)
    r[0, 0] = h[0]
    r[1, 0] = h[1]

    h = myint_genus2(zeros, rea, zeros[4], 1, digits)
    r[0, 3] = h[0] + a[0]
    r[1, 3] = h[1] + a[1]

    h = myint_genus2(zeros, zeros[1], rea, 0, digits)
    r[0, 2] = r[0, 3] + h[0] - re(a[0]) + 1j * im(a[0])
    r[1, 2] = r[1, 3] + h[1] - re(a[1]) + 1j * im(a[1])

    return r

def ima2Per4(realNS, complexNS, digits):
    rea = re(complexNS[0])
    ima = fabs(im(complexNS[0]))
    
    zeros = sorted(realNS + complexNS, key = lambda x : re(x))

    r = matrix(2, 4)

    h = myint_genus2(zeros, zeros[0], zeros[1], 1, digits)
    r[0, 0] = h[0]
    r[1, 0] = h[1]
    
    a = int_genus2_complex(zeros, rea, ima, 3, 0, digits)
    v = myint_genus2(zeros, zeros[2], rea, 0, digits)
 
    r[0, 1] = 2 * v[0] + 2 * re(a[0])
    r[1, 1] = 2 * v[1] + 2 * re(a[1])
 
    #h = myint_genus2(zeros, zeros[2], rea, 1, digits)
    r[0, 3] = v[0] + a[0]
    r[1, 3] = v[1] + a[1]

    h = myint_genus2(zeros, zeros[1], zeros[2], 0, digits)
    r[0, 2] = h[0] + a[0] - re(a[0]) + 1j * im(a[0])
    r[1, 2] = h[1] + a[1] - re(a[1]) + 1j * im(a[1])

    return r

def ima4Per1(realNS, rea1, ima1, rea2, ima2, digits):
    r = matrix(2, 4)
    x = Symbol("x")

    zeros = [rea1 - 1j * ima1, rea1 + 1j * ima1, rea2 - 1j * ima2, rea2 + 1j * ima2, realNS[0]]
    t = 1
    p = 0

    for i in range(5):
        t *= (x - zeros[i])
    t = collect(t.expand(), x)

    for i in range(6):
        p += re(t.coeff(x, i)) * x**i
    p = lambdify(x, p)

    h1 = int_genus2_complex(zeros, rea1, ima1, 0, 1, digits)
    r[0, 0] = 2 * re(h1[0])
    r[1, 0] = 2 * re(h1[1])

    if ((rea2 - rea1)**2 + ima1**2 + 2 * (rea2 - rea1) * (rea2 - zeros[4])).evalf() < 0:
        h2 = int_genus2_complex(zeros, rea2, ima2, 2, 1, digits)
    else:
        h2 = int_genus2_complex(zeros, rea2, ima2, 2, 0, digits)

    r[0, 1] = 2 * re(h2[0])
    r[1, 1] = 2 * re(h2[1])
    
    a = myint_genus2(zeros, rea2, zeros[4], 1, digits)
    r[0, 3] = a[0] + h2[0]
    r[1, 3] = a[1] + h2[1]

    r[0, 2] = r[0, 3] + h1[0] - 1j * quad(lambda x : 1 / sqrt(-p(x)), [rea1, rea2]) - re(h2[0]) + 1j * im(h2[0])
    r[1, 2] = r[1, 3] + h1[1] - 1j * quad(lambda x : x / sqrt(-p(x)), [rea1, rea2]) - re(h2[1]) + 1j * im(h2[1])

    return r

def ima4Per2(realNS, rea1, ima1, rea2, ima2, digits):
    r = matrix(2, 4)
    zeros = [realNS[0], rea1 - 1j * ima1, rea1 + 1j * ima1, rea2 - 1j * ima2, rea2 + 1j * ima2]

    a1 = int_genus2_complex(zeros, rea1, ima1, 1, 1, digits)
    a2 = int_genus2_complex(zeros, rea2, ima2, 3, 0, digits)

    w = myint_genus2(zeros, zeros[0], rea1, 0, digits)
    r[0, 0] = 2 * re(a1[0]) - 2 * w[0]
    r[1, 0] = 2 * re(a1[1]) - 2 * w[1]

    u = myint_genus2(zeros, zeros[0], rea2, 0, digits)
    r[0, 1] = 2 * re(a2[0]) + 2 * u[0]
    r[1, 1] = 2 * re(a2[1]) + 2 * u[1]

    r[0, 3] = a2[0] + u[0]
    r[1, 3] = a2[1] + u[1]

    r[0, 2] = r[0, 3] + a1[0] - w[0] - u[0] - re(a2[0]) + 1j *im(a2[0])
    r[1, 2] = r[1, 3] + a1[1] - w[1] - u[1] - re(a2[1]) + 1j *im(a2[1])

    return r

def ima4Per3(realNS, rea1, ima1, rea2, ima2, digits):
    r = matrix(2, 4)
    zeros = [rea1 - 1j * ima1, rea1 + 1j * ima1, realNS[0], rea2 - 1j * ima2, rea2 + 1j * ima2]

    h1 = int_genus2_complex(zeros, rea1, ima1, 0, 1, digits)
    r[0, 0] = 2 * re(h1[0]) 
    r[1, 0] = 2 * re(h1[1])

    h2 = int_genus2_complex(zeros, rea2, ima2, 3, 0, digits)
    v = myint_genus2(zeros, zeros[2], rea2, 0, digits)

    r[0, 1] = 2 * re(h2[0]) + 2 * v[0]
    r[1, 1] = 2 * re(h2[1]) + 2 * v[1]

    r[0, 3] = h2[0] + v[0]
    r[1, 3] = h2[1] + v[1]

    h = myint_genus2(zeros, rea1, zeros[2], 0, digits)
    r[0, 2] = r[0, 3] + h1[0] + h[0] + v[0] + h2[0] - r[1, 0]
    r[1, 2] = r[1, 3] + h1[1] + h[1] + v[1] + h2[1] - r[1, 1]

    return r

def set_period_globals_genus2(period_matrix):

    periods_first = period_matrix[0:2, 0:2]
    periods_second = period_matrix[0:2, 2:4]

    periods_inverse = periods_first**(-1)
    riemannM = periods_inverse * periods_second
    riemannM[0, 1] = 1/2 * (riemannM[0, 1] + riemannM[1, 0])
    riemannM[1, 0] = riemannM[0, 1]
    m = eig(riemannM.apply(im))[0]
    if im(m[0]) != 0 and im(m[1]) == 0 and re(m[0]) > 0 and re(m[1]) > 0:
        raise ValueError("Imaginary part of Riemann matrix is not positive definite")
    return periods_inverse, riemannM

def eval_period(m, n, realNS, zeros, omega, component = 0):
    
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
    for i in range(K, L):
        result += periodlist[i]
    result = sign * result

    return result

def branch_list_genus2(zeros, num_realNS):
    if num_realNS == 5:
        erg = [[1, zeros[0], 1], [1, zeros[1], 0], [0, zeros[2], 0], [0, zeros[3], 1], [1, zeros[4], 1]]
    elif num_realNS == 3:
        if im(zeros[0]) != 0:
            erg = [[1, zeros[0], 1], [1, zeros[1], 0], [0, zeros[2], 0], [0, zeros[3], 1], [1, zeros[4], 1]]
        elif im(zeros[1]) != 0:
            erg = [["tbd", zeros[0], 0], [1, zeros[1], "tbd"], ["tbd", zeros[2], 0], [0, zeros[3], 1], [1, zeros[4], 1]]
        elif im(zeros[2]) != 0:
            rea1 = re(zeros[2])
            if ((rea1-zeros[0]) * (rea1-zeros[1]) + (rea1-zeros[1]) * (rea1-zeros[4]) + (rea1-zeros[4]) * (rea1-zeros[0])).evalf() < 0:
                erg = [[1, zeros[0], 1], [1, zeros[1], 0], [0, zeros[2], 1], [1, zeros[3], 1], [1, zeros[4], 1]]
            else:
                erg = [[1, zeros[0], 1], [1, zeros[1], 0], [0, zeros[2], 0], [0, zeros[3], 1], [1, zeros[4], 1]]
        elif im(zeros[3]) != 0:
            erg = [[1, zeros[0], 1], [1, zeros[1], 0], ["tbd", zeros[2], 1], [1, zeros[3], "tbd"], ["tbd", zeros[4], 1]]
    elif num_realNS == 1:
        if im(zeros[4]) == 0:
            rea1 = re(zeros[0]); rea2 = re(zeros[2])
            ima1 = fabs(im(zeros[0]))
            if ((rea2-rea1)**2 + ima1**2 + 2 * (rea2-rea1) * (rea2-zeros[4])).evalf() < 0:
                erg = [[1, zeros[0], 1], [1, zeros[1], 0], [0, zeros[2], 1], [1, zeros[3], 1], [1, zeros[4], 1]]
            else:
                erg = [[1, zeros[0], 1], [1, zeros[1], 0], [0, zeros[2], 0], [0, zeros[3], 1], [1, zeros[4], 1]]
        elif im(zeros[0]) == 0:
            erg = [["tbd", zeros[0], 1], [1, zeros[1], "tbd"], ["tbd", zeros[2], 1], [1, zeros[3], "tbd"], ["tbd", zeros[4], 1]]
        elif im(zeros[2]) == 0:
            erg = [[1, zeros[0], 1], [1, zeros[1], 0], ["tbd", zeros[2], 1], [1, zeros[3], "tbd"], ["tbd", zeros[4], 1]]
    else:
        raise ValueError("Wrong number of real zeros.")
    return erg

def int_genus2_first(zeros, lower, upper, digits, periodMatrix = None):
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

    if inlist(lb, realNS) + 1 == inlist(ub, realNS):
        if inlist(lb, e) + 1 == inlist(ub, e):
            tags = (lb + ub) / 2
        else:
            tags = re(e[inlist(lb, e) + 1])
        branch_list = branch_list_genus2(e, len(realNS))

        return sign * (myint_genus2(e, lb, tags, branch_list[inlist(lb, e)][2], digits) + myint_genus2(e, tags, ub, branch_list[inlist(ub, e)][0], digits))

    elif inlist(lb, realNS) >= 0 or inlist(ub, realNS) >= 0:
        if inlist(lb, realNS) == -1:
            if inlist(lb, sorted([lb] + realNS, key = lambda x : re(x))) == inlist(ub, realNS):
                return sign * myint_genus2(e, lb, ub, branch_list_genus2(e, len(realNS))[inlist(ub, e)][0], digits)
            else:
                raise ValueError("Invalid bounds")
        elif inlist(ub, realNS) == -1:
            if inlist(ub, sorted([ub] + realNS, key = lambda x : re(x))) == inlist(lb, realNS) + 1:
                return sign * myint_genus2(e, lb, ub, branch_list_genus2(e, len(realNS))[inlist(lb, e)][2], digits)
            else:
                raise ValueError("Invalid bounds")
        else:
            raise ValueError("Invalid bounds")
    else:
        if inlist(lb, sorted([lb] + realNS, key = lambda x : re(x))) == inlist(ub, sorted([ub] + realNS, key = lambda x : re(x))):
            if periodMatrix == None:
                periodMatrix = periods(realNS, complexNS, digits)
            if ub == oo:
                return sign * (myint_genus2(e, lb, realNS[-1], branch_list_genus2(e, len(realNS))[inlist(realNS[-1], e)][2], digits) + 
                matrix([eval_period(len(realNS) - 1,oo,realNS,e,periodMatrix,0), eval_period(len(realNS) - 1,oo,realNS,e,periodMatrix,1)]))
            else:
                p = lambda x : (x - e[0]) * (x - e[1]) * (x - e[2]) * (x - e[3]) * (x - e[4])
                sign = sign * exp(pi * 1j * branch_list_genus2(e, len(realNS))[inlist(lb, sorted(e + [lb], key = lambda x : re(x)))][2])
                return matrix([sign * quad(lambda x : 1 / mpc(sqrt(p(x))), [lb, ub]), sign * quad(lambda x : x / mpc(sqrt(p(x))), [lb, ub])])
        else:
            raise ValueError("Invalid bounds")
