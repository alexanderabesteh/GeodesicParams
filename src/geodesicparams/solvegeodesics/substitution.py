#!/usr/bin/env python
"""
Helper functions for converting polynomials in hyperelliptic and elliptic differential
equations into standard form.

Functions include convert_deg5, which converts a 5th degree polynomial into standard form
(y^5 + a4 * y^4 + a3 + ... + a1 * y + a0, where a1, a2, ... , a4 are real numbers);
convert_deg3, which converts a 3rd degree polynomial into the weierstrass standard form 
(4z^3 - g2z - g3); and convert_degeven, which either converts a 6th degree polynomial to a 
5th degree polynomial, or a 4th degree polynomial to a 3rd degree, which can then be passed
to convert_deg5 or convert_deg3. convert_polynomial encapsulates all these behaviours depending
on the degree of the polynomial. As of now, polynomials with multiple zeroes have yet to be
implemented.
 
"""

from mpmath import sign
from sympy import collect, lambdify, Symbol, Abs, Poly, re, degree, sign, pprint

from ..utilities import inlist, extract_multiple_elems, separate_zeros, eval_roots

def convert_deg5(polynomial, zeros):
    """
    Convert a 5th degree polynomial to its standard form, using the substitution y = - z 
    or y = z, such that the leading coefficient is positive.
    
    Parameters
    ----------
    polynomial : symbolic
        A symbolic statement representing the polynomial to be converted.
    zeros : list
        A list of 5 complex or real numbers representing the zeros of <polynomial>.
        
    Returns
    -------
    p_standard : symbolic
        The polynomial converted to standard form.
    constant : float or symbolic
        The prefactor infront of the converted polynomial in the differential equation.
    callabe
        The substitution used in the conversion as a callable lambda function.
    substitution : symbolic 
        The substitution used in the conversion as a symbolic statement.
    sign : integer
        The sign, either +1 or -1, depending on the initial direction of motion
        (not implemented yet).

    """

    y = Symbol("y")
    p = polynomial

    # Check for multiplicity (not implemented yet, tbd)
    zeros_list, mult_zeros = extract_multiple_elems(zeros)
    if len(mult_zeros) > 0:
        raise Exception("Multiple zeros in polynomial, to be done")

    p_standard = 0
    p_sym = Poly(p).gen

    coeff_5 = p.coeff(p_sym, 5)

    # Determine and apply substitution, changes whether <coeff_5> is positive or negative
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
    """
    Convert a 3rd degree polynomial to its standard form, using the substitution
    y = 1/b3 * (4*z - b2/3).
    
    Parameters
    ----------
    polynomial : symbolic
        A symbolic statement representing the polynomial to be converted.
    zeros : list
        A list of 3 complex or real numbers representing the zeros of <polynomial>.
        
    Returns
    -------
    p_standard : symbolic
        The polynomial converted to standard form.
    constant : integer
        The prefactor infront of the converted polynomial in the differential equation
        (always 1).
    callabe
        The substitution used in the conversion as a callable lambda function.
    substitution : symbolic 
        The substitution used in the conversion as a symbolic statement.
    sign : integer
        The sign, either +1 or -1, depending on the initial direction of motion.

    """

    y = Symbol('y')
    p = polynomial
    
    # Check for multiplicity (not implemented yet, tbd)
    zeros_list, mult_zeros = extract_multiple_elems(zeros)
    if len(mult_zeros) > 0:
        raise ValueError("multiple zeros, tbd")

    # Determine substitution
    a3 = p.coeff(Poly(p).gen, 3)
    a2 = p.coeff(Poly(p).gen, 2)
    sign_a3 = int(sign(a3))
    substitution = 1/a3 * (4*y - a2/3)

    # Apply substitution
    p_standard =(a3**2 / 4**2 * p.subs(Poly(p).gen, substitution)).simplify()
    if Abs(p_standard.coeff(y, 2)) != 0:
        p_standard = 4*y ** 3 + p_standard.coeff(y, 1)*y + p_standard.coeff(y, 0)
    p_standard = p_standard.subs(y**4, 0)

    return [p_standard, 1, lambdify(y, substitution), substitution, sign_a3]

def convert_degeven(polynomial, zeros, badzeros):
    """
    Convert a 4th degree polynomial to a 3rd degree polynomial or a 6th degree to a 5th 
    degree. 

    Using the substitution y = 1/z + ei, where ei is one of the roots of <polynomial>, the degree 
    of the is reduced by polynomial by 1. If the degree of the polynomial was 6, then the prefactor
    is given by f(z) = z^2.
    
    Parameters
    ----------
    polynomial : symbolic
        A symbolic statement representing the polynomial to be converted.
    zeros : list
        A list of 4 or 6 complex or real numbers representing the zeros of <polynomial>.
    badzeros : list
        A list of complex or real numbers representing the roots of <polynomial> that are
        not to be used in the conversion.
        
    Returns
    -------
    p_standard : symbolic
        The original polynomial converted to standard form.
    constant : float or symbolic
        The prefactor infront of the converted polynomial in the differential equation.
    callabe
        The substitution used in the conversion as a callable lambda function.
    substitution : symbolic 
        The substitution used in the conversion as a symbolic statement.
    sign : integer
        The sign, either +1 or -1, depending on the initial direction of motion.

    """

    z = Symbol('z')
    p = polynomial
    substitution = None
    
    # Check for multiplicity (not implemented yet, tbd)
    zeros_list, mult_zeros = extract_multiple_elems(zeros)
    if len(mult_zeros) > 0:
        raise ValueError("Error: multiple zeros, tbd")
    
    # Determine substitution
    if inlist(0, zeros_list) >= 0:
        substitution = 1/z
    else:
        realNS, complexNS = separate_zeros(zeros_list)
        # Sort real zeros by how close they are to 1 or -1
        realz = sorted(realNS, key = lambda x : Abs(Abs(x) - 1))
        for i in range(len(realz)):
            if inlist(realz[i], badzeros) == -1:
                substitution = 1/z + realz[i]
                break
    if substitution == None:
        substitution = 1/z + realz[0]

    # Apply substitution to polynomial and prefactor
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
    """
    Convert a polynomial used in a hyperelliptic or elliptic differential equation into
    standard form. 
    
    Parameters
    ----------
    polynomial : symbolic
        A symbolic statement representing the polynomial to be converted.
    degree : integer
        The degree of the polynomial.
    zeros : list
        A list of 4 or 6 complex or real numbers representing the zeros of <polynomial>.
    badzeros : list, optional
        A list of complex or real numbers representing the roots of <polynomial> that are
        not to be used in the conversion.
        
    Returns
    -------
    p_standard : symbolic
        The original polynomial converted to standard form.
    constant : float or symbolic
        The prefactor infront of the converted polynomial in the differential equation.
    callabe
        The substitution used in the conversion as a callable lambda function.
    substitution : symbolic 
        The substitution used in the conversion as a symbolic statement.
    sign : integer
        The sign, either +1 or -1, depending on the initial direction of motion.

    """

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

