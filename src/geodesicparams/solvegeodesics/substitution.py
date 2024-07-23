from mpmath import sign
from sympy import collect, lambdify, Symbol, Abs, Poly, re, degree, sign, pprint
from ..utilities import inlist, extract_multiple_elems, separate_zeros, eval_roots

def convert_deg5(polynomial, zeros):
    y = Symbol("y")

    p = polynomial
    zeros_list, mult_zeros = extract_multiple_elems(zeros)
    if len(mult_zeros) > 0:
        raise Exception("Multiple zeros in polynomial, to be done")

    p_standard = 0
    p_sym = Poly(p).gen

    coeff_5 = p.coeff(p_sym, 5)
    if coeff_5 < 0:
        p = collect(p.subs(p_sym, -y), y)
        constant = p.coeff(y, 5)
        for i in range(6):
            p_standard += re(p.coeff(y, i) / constant) * y**i
        substitution = -y
        sign = -1
    else:
        constant = coeff_5
        for i in range(6):
            p_standard += re(p.coeff(p_sym, i) / constant) * y**i
        substitution = y
        sign = 1
    return [p_standard, constant, lambdify(y, substitution), substitution, sign]

def convert_deg3(polynomial, zeros):
    y = Symbol('y')
    p = polynomial
    zeros_list, mult_zeros = extract_multiple_elems(zeros)
    if len(mult_zeros) > 0:
        raise ValueError("multiple zeros, tbd")

    a3 = p.coeff(Poly(p).gen, 3)
    a2 = p.coeff(Poly(p).gen, 2)
    sign_a3 = int(sign(a3))
    substitution = 1/a3 * (4*y - a2/3)
    p_standard =(a3**2 / 4**2 * p.subs(Poly(p).gen, substitution)).simplify()
    if Abs(p_standard.coeff(y, 2)) != 0:
        p_standard = 4*y ** 3 + p_standard.coeff(y, 1)*y + p_standard.coeff(y, 0)
    p_standard = p_standard.subs(y**4, 0)
    return [p_standard, 1, lambdify(y, substitution), substitution, sign_a3]

def convert_degeven(polynomial, zeros, badzeros):
    z = Symbol('z')
    p = polynomial
    zeros_list, mult_zeros = extract_multiple_elems(zeros)
    substitution = None

    if len(mult_zeros) > 0:
        raise ValueError("Error: multiple zeros, tbd")

    if inlist(0, zeros_list) >= 0:
        substitution = 1/z
    else:
        realNS, complexNS = separate_zeros(zeros_list)
        realz = sorted(realNS, key = lambda x : Abs(Abs(x) - 1))
        for i in range(len(realz)):
            if inlist(realz[i], badzeros) == -1:
                substitution = 1/z + realz[i]
                break
    if substitution == None:
        substitution = 1/z + realz[0]
    if degree(p) == 4:
        p_degodd = (z** 4 * p.subs(Poly(p).gen, substitution)).simplify()
        integrand = 1
        zeros_degodd = eval_roots(Poly(p_degodd).all_roots())
        data_degodd = convert_deg3(p_degodd, zeros_degodd)
        prefactor = integrand * data_degodd[1]
    elif degree(p) == 6:
        p_degodd = (z**6 * p.subs(Poly(p).gen, substitution)).simplify()
        integrand = 1 / z**2
        zeros_degodd = eval_roots(Poly(p_degodd).all_roots())
        data_degodd = convert_deg5(p_degodd, zeros_degodd)
        prefactor = (data_degodd[1] * integrand).subs(z, data_degodd[3])

    return [data_degodd[0], prefactor, lambdify(Poly(data_degodd[3]).gen, substitution.subs(z, data_degodd[3])), substitution.subs(z, data_degodd[3]), data_degodd[4]]

def convert_polynomial(polynomial, degree, zeros, badzeros):
    y = Symbol('y')

    if degree > 6:
        raise ValueError("Degree larger than 6 is not supported")
    elif degree == 6 or degree == 4:
        return convert_degeven(polynomial, zeros, badzeros)
    elif degree == 5:
        return convert_deg5(polynomial, zeros)
    elif degree == 3:
        return convert_deg3(polynomial, zeros)
    elif degree < 3:
        return [polynomial.subs(Poly(polynomial).gen, y), 1, lambdify(y, y), y, -1]

