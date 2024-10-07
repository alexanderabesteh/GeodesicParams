#!/usr/bin/env python
"""
Procedures for computing the periods of the first kind for a genus 2 Riemann surface.

This is done by integrating the vector of canonical holomorphic differentials along the
contours that connect the branch cuts and the contours that encircle the branch cuts.

NOTE: fix some things here like the precision and documentation.

"""

from mpmath import quad, matrix, fabs, eig
from sympy import re, im, Symbol, collect, lambdify, sqrt

from ...utilities import inlist
from ..integrations.integrate_hyperelliptic import myint_genus2, int_genus2_complex

def periods(realNS, complexNS, digits):
    """
    Computes the periods of the first kind for a genus 2 Riemann surface.

    This is done by integrating the vector of canonical holomorphic differentials along the
    contours that encircle the branch cuts and the contours connect the branch cuts.

    Parameters
    ----------
    realNS : list
        A list of real numbers, the real roots of the polynomial defining the Riemann
        surface.
    complexNS : list
        A list of complex numbers, the complex roots of the polynomial defining the Riemann 
        surface.
    digits : int
        The number of digits to be used in the computation.

    Returns
    -------
    matrix
        A 2x4 matrix: the first 2x2 matrix represents the period matrix corresponding to the
        integrations along the contours encircling the branch cuts. The second 2x2 matrix
        represents the period matrix corresponding to the integrations along the contours
        connecting the branch cuts.

    """

    realNS = sorted(realNS)

    if len(complexNS) == 0:
        return reaPer(realNS, digits)
    else:
        return imaPer(realNS, complexNS, digits)

def reaPer(realNS, digits):
    """
    Computes the periods of the first kind for a genus 2 Riemann surface for the case that
    all zeros of the polynomial defining the Riemann surface are real.

    Parameters
    ----------
    realNS : list
        A list of real numbers, the real roots of the polynomial defining the Riemann
        surface.
    digits : int
        The number of digits to be used in the computation.

    Returns
    -------
    matrix
        A 2x4 matrix: the first 2x2 matrix represents the period matrix corresponding to the
        integrations along the contours encircling the branch cuts. The second 2x2 matrix
        represents the period matrix corresponding to the integrations along the contours
        connecting the branch cuts.

    """

    r = matrix(2, 4)

    # Path A1: realNS[1]..realNS[2], negative branch
    h = myint_genus2(realNS, realNS[0], realNS[1], 1, digits)
    r[0, 0] = h[0]
    r[1, 0] = h[1]
   
    # Path A2: realNS[3]..realNS[4], positive branch
    h = myint_genus2(realNS, realNS[2], realNS[3], 0, digits)
    r[0, 1] = h[0]
    r[1, 1] = h[1]

    # Path B2: realNS[4]..realNS[5], negative branch
    h = myint_genus2(realNS, realNS[3], realNS[4], 1, digits)
    r[0, 3] = h[0]
    r[1, 3] = h[1]

    # Path B1 = B2 + B3, path B3: realNS[2]..realNS[3], positive branch
    h = myint_genus2(realNS, realNS[1], realNS[2], 0, digits)
    r[0, 2] = h[0] + r[0, 3]
    r[1, 2] = h[1] + r[1, 3]

    return r

def imaPer(realNS, complexNS, digits):
    """
    Computes the periods of the first kind for a genus 2 Riemann surface for the case that
    the zeros of the polynomial defining the Riemann surface are complex. <ima2Per> is 
    called if there are two complex zeros, and <ima4Per> is called when there are 4 complex
    zeros.

    Parameters
    ----------
    realNS : list
        A list of real numbers, the real roots of the polynomial defining the Riemann
        surface.
    complexNS : list
        A list of complex numbers, the complex roots of the polynomial defining the Riemann 
        surface.
    digits : int
        The number of digits to be used in the computation.

    Returns
    -------
    matrix
        A 2x4 matrix: the first 2x2 matrix represents the period matrix corresponding to the
        integrations along the contours encircling the branch cuts. The second 2x2 matrix
        represents the period matrix corresponding to the integrations along the contours
        connecting the branch cuts.

    """

    if len(complexNS) == 2:
        return ima2Per(realNS, complexNS, digits)
    else:
        return ima4Per(realNS, complexNS, digits)

def ima2Per(realNS, complexNS, digits):
    """
    Computes the periods of the first kind for a genus 2 Riemann surface for the case that
    two zeros of the polynomial defining the Riemann surface are complex. There are 4 different
    subcases, ima2Perk, where k is an integer related to the ordering of the zeros.

    For all <k> subcases it is necessary to compute integrals along branch cuts perpendicular 
    to the real axis. For the calculation it has to be taken into account that the real part of 
    the integrals computed along such branch cuts is symmetrical with respect to the real axis 
    but that the imaginary part is antisymmetric. Therefore, the whole integral will be real.

    Parameters
    ----------
    realNS : list
        A list of real numbers, the real roots of the polynomial defining the Riemann
        surface.
    complexNS : list
        A list of complex numbers, the complex roots of the polynomial defining the Riemann 
        surface.
    digits : int
        The number of digits to be used in the computation.

    Returns
    -------
    matrix
        A 2x4 matrix: the first 2x2 matrix represents the period matrix corresponding to the
        integrations along the contours encircling the branch cuts. The second 2x2 matrix
        represents the period matrix corresponding to the integrations along the contours
        connecting the branch cuts.

    """

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
    """
    Computes the periods of the first kind for a genus 2 Riemann surface for the case that
    four zeros z1, z2, z3, z4 of the polynomial defining the Riemann surface are complex, where
    Re(z1) = Re(z2), Re(z3) = Re(z4), and Re(z1) < Re(z3). There are 3 different subcases, 
    ima4Perk, where k is an integer related to the ordering of the zeros.

    For all <k> subcases it is necessary to compute integrals along branch cuts perpendicular 
    to the real axis. For the calculation it has to be taken into account that the real part 
    of the integrals computed along such branch cuts is symmetrical with respect to the real 
    axis but that the imaginary part is antisymmetric. Therefore, the whole integral will be 
    real.

    Parameters
    ----------
    realNS : list
        A list of real numbers, the real roots of the polynomial defining the Riemann
        surface.
    complexNS : list
        A list of complex numbers, the complex roots of the polynomial defining the Riemann 
        surface.
    digits : int
        The number of digits to be used in the computation.

    Returns
    -------
    matrix
        A 2x4 matrix: the first 2x2 matrix represents the period matrix corresponding to the
        integrations along the contours encircling the branch cuts. The second 2x2 matrix
        represents the period matrix corresponding to the integrations along the contours
        connecting the branch cuts.

    """

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
    """
    Computes the periods of the first kind for a genus 2 Riemann surface for the case that
    two zeros of the polynomial defining the Riemann surface are complex, with subcase <k> = 1. There are 4 different

    Parameters
    ----------
    realNS : list
        A list of real numbers, the real roots of the polynomial defining the Riemann
        surface.
    complexNS : list
        A list of complex numbers, the complex roots of the polynomial defining the Riemann 
        surface.
    digits : int
        The number of digits to be used in the computation.

    Returns
    -------
    matrix
        A 2x4 matrix: the first 2x2 matrix represents the period matrix corresponding to the
        integrations along the contours encircling the branch cuts. The second 2x2 matrix
        represents the period matrix corresponding to the integrations along the contours
        connecting the branch cuts.

    """

    rea = re(complexNS[0])
    ima = fabs(im(complexNS[0]))
    
    zeros = sorted(realNS + complexNS, key = lambda x : re(x))

    r = matrix(2, 4)

    # A2: zero3...zero4, pos. branch
    h = myint_genus2(zeros, zeros[2], zeros[3], 0, digits)
    r[0, 1] = h[0]
    r[1, 1] = h[1]
   
    # B2: zero4...zero5, neg. branch
    h = myint_genus2(zeros, zeros[3], zeros[4], 1, digits)
    r[0, 3] = h[0]
    r[1, 3] = h[1]

    # A1: zero2...zero1, neg. branch
    a = int_genus2_complex(zeros, rea, ima, 0, -1, digits)
    r[0, 0] = 2 * re(a[0])
    r[1, 0] = 2 * re(a[1])

    # B1 = B2 + B3: B3: zero2...zero3, pos. branch
    h = myint_genus2(zeros, rea, zeros[2], 0, digits)
    r[0, 2] = h[0] + r[0, 3] + a[0]
    r[1, 2] = h[1] + r[1, 3] + a[1]

    return r

def ima2Per2(realNS, complexNS, digits):
    """
    Computes the periods of the first kind for a genus 2 Riemann surface for the case that
    two zeros of the polynomial defining the Riemann surface are complex, with subcase <k> = 2. There are 4 different

    Parameters
    ----------
    realNS : list
        A list of real numbers, the real roots of the polynomial defining the Riemann
        surface.
    complexNS : list
        A list of complex numbers, the complex roots of the polynomial defining the Riemann 
        surface.
    digits : int
        The number of digits to be used in the computation.

    Returns
    -------
    matrix
        A 2x4 matrix: the first 2x2 matrix represents the period matrix corresponding to the
        integrations along the contours encircling the branch cuts. The second 2x2 matrix
        represents the period matrix corresponding to the integrations along the contours
        connecting the branch cuts.

    """

    rea = re(complexNS[0])
    ima = fabs(im(complexNS[0]))
    
    zeros = sorted(realNS + complexNS, key = lambda x : re(x))

    r = matrix(2, 4)

    # B2: zero4...zero5, neg. branch
    h = myint_genus2(zeros, zeros[3], zeros[4], 1, digits)
    r[0, 3] = h[0]
    r[1, 3] = h[1]
   
    # A2: (zero1...rea)**+ + (rea...zero4)**+
    v = myint_genus2(zeros, zeros[0], rea, 0, digits)
    h = myint_genus2(zeros, rea, zeros[3], 0, digits)
    r[0, 1] = v[0] + h[0]
    r[1, 1] = v[1] + h[1]

    # Computation of int_genus2_complex: neg. branch
    a = int_genus2_complex(zeros, rea, ima, 1, -1, digits)
    r[0, 0] = 2 * re(a[0]) - 2 * v[0]
    r[1, 0] = 2 * re(a[1]) - 2 * v[1]

    # A1: (ima...0)**+ + (rea...zero1)**+ + (zero1...rea)**- + (0...-ima)**+
    r[0, 2] = r[0, 3] - v[0] + a[0]
    r[1, 2] = r[1, 3] - v[1] + a[1]

    return r

def ima2Per3(realNS, complexNS, digits):
    """
    Computes the periods of the first kind for a genus 2 Riemann surface for the case that
    two zeros of the polynomial defining the Riemann surface are complex, with subcase <k> = 3. There are 4 different

    Parameters
    ----------
    realNS : list
        A list of real numbers, the real roots of the polynomial defining the Riemann
        surface.
    complexNS : list
        A list of complex numbers, the complex roots of the polynomial defining the Riemann 
        surface.
    digits : int
        The number of digits to be used in the computation.

    Returns
    -------
    matrix
        A 2x4 matrix: the first 2x2 matrix represents the period matrix corresponding to the
        integrations along the contours encircling the branch cuts. The second 2x2 matrix
        represents the period matrix corresponding to the integrations along the contours
        connecting the branch cuts.

    """

    rea = re(complexNS[0])
    ima = fabs(im(complexNS[0]))
    
    zeros = sorted(realNS + complexNS, key = lambda x : re(x))

    # For the calculation of the paths A2 and B2
    # an integration rea...rea + I*ima is needed.
    # The branch for this calculation depends on the following:
    if ((rea - zeros[0]) * (rea - zeros[1]) + (rea - zeros[1]) * (rea - zeros[4]) + (rea - zeros[4]) * (rea - zeros[0])).evalf() < 0:
        a = int_genus2_complex(zeros, rea, ima, 2, -1, digits)
    else:
        a = int_genus2_complex(zeros, rea, ima, 2, 0, digits)

    r = matrix(2, 4)

    r[0, 1] = 2 * re(a[0])
    r[1, 1] = 2 * re(a[1])
   
    # A1: zero1...zero2, neg. branch
    h = myint_genus2(zeros, zeros[0], zeros[1], 1, digits)
    r[0, 0] = h[0]
    r[1, 0] = h[1]

    # B2: rea...zero5, neg. branch
    h = myint_genus2(zeros, rea, zeros[4], 1, digits)

    # Integration rea...rea + I*ima on "back side"
    r[0, 3] = h[0] + a[0]
    r[1, 3] = h[1] + a[1]

    # B1 = B3 + B2, B3: zero2...rea, pos. branch
    # B3 = integration zeros[1]...rea + integration rea...rea - I*ima
    # integration rea...rea - I*ima = - Re(a) + I*Im(a),
    # as Re(int(rea...rea + I*ima)) = Re(int(rea...rea - I*ima)),
    # Im(int(rea...rea + I*ima)) = -Im(int(rea...rea - I*ima))
    # and the branch on rea - I*t is not the same as on rea + I*t
    h = myint_genus2(zeros, zeros[1], rea, 0, digits)
    r[0, 2] = r[0, 3] + h[0] - re(a[0]) + 1j * im(a[0])
    r[1, 2] = r[1, 3] + h[1] - re(a[1]) + 1j * im(a[1])

    return r

def ima2Per4(realNS, complexNS, digits):
    """
    Computes the periods of the first kind for a genus 2 Riemann surface for the case that
    two zeros of the polynomial defining the Riemann surface are complex, with subcase <k> = 4. There are 4 different

    Parameters
    ----------
    realNS : list
        A list of real numbers, the real roots of the polynomial defining the Riemann
        surface.
    complexNS : list
        A list of complex numbers, the complex roots of the polynomial defining the Riemann 
        surface.
    digits : int
        The number of digits to be used in the computation.

    Returns
    -------
    matrix
        A 2x4 matrix: the first 2x2 matrix represents the period matrix corresponding to the
        integrations along the contours encircling the branch cuts. The second 2x2 matrix
        represents the period matrix corresponding to the integrations along the contours
        connecting the branch cuts.

    """

    rea = re(complexNS[0])
    ima = fabs(im(complexNS[0]))
    
    zeros = sorted(realNS + complexNS, key = lambda x : re(x))

    r = matrix(2, 4)

    # A1: zero1...zero2, neg branch
    h = myint_genus2(zeros, zeros[0], zeros[1], 1, digits)
    r[0, 0] = h[0]
    r[1, 0] = h[1]
   
    # int_genus2_complex: pos. branch
    a = int_genus2_complex(zeros, rea, ima, 3, 0, digits)
    
    # A2 (=zero4...zero5 on neg. side):
    # (ima...0)**- + (rea...zero3)**- + (zero3...rea)**+ + (0...-ima)**-
    v = myint_genus2(zeros, zeros[2], rea, 0, digits)
    r[0, 1] = 2 * v[0] + 2 * re(a[0])
    r[1, 1] = 2 * v[1] + 2 * re(a[1])
 
    # B2: (ima...0)**- + (rea...zero3)**-
    r[0, 3] = v[0] + a[0]
    r[1, 3] = v[1] + a[1]

    # B1 = B2 + B3: [(ima...0)**- + (rea...e3)**-]
    # + [(e2...e3)**+ + (e3...rea)**+ + (0...-ima)**+]
    h = myint_genus2(zeros, zeros[1], zeros[2], 0, digits)
    r[0, 2] = h[0] + a[0] - re(a[0]) + 1j * im(a[0])
    r[1, 2] = h[1] + a[1] - re(a[1]) + 1j * im(a[1])

    return r

def ima4Per1(realNS, rea1, ima1, rea2, ima2, digits):
    """
    Computes the periods of the first kind for a genus 2 Riemann surface for the case that
    four zeros z1, z2, z3, z4 of the polynomial defining the Riemann surface are complex, where
    Re(z1) = Re(z2), Re(z3) = Re(z4), and Re(z1) < Re(z3), with subcase <k> = 1.

    Parameters
    ----------
    realNS : list
        A list of real numbers, the real roots of the polynomial defining the Riemann
        surface.
    rea1 : float
        The real part of the first complex zero.
    ima1 : float
        The absolute value of the imaginary part of the first complex zero. 
    rea2 : float
        The real part of the third complex zero.
    ima2 : float
        The absolute value of the imaginary part of the third complex zero.
    digits : int
        The number of digits to be used in the computation.

    Returns
    -------
    matrix
        A 2x4 matrix: the first 2x2 matrix represents the period matrix corresponding to the
        integrations along the contours encircling the branch cuts. The second 2x2 matrix
        represents the period matrix corresponding to the integrations along the contours
        connecting the branch cuts.

    """

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

    # A1: zero2...zero1 or -ima1...ima1, neg. branch
    h1 = int_genus2_complex(zeros, rea1, ima1, 0, 1, digits)
    r[0, 0] = 2 * re(h1[0])
    r[1, 0] = 2 * re(h1[1])

    # A2: zero4...zero3 or -ima2...ima2, branch depends
    if ((rea2 - rea1)**2 + ima1**2 + 2 * (rea2 - rea1) * (rea2 - zeros[4])).evalf() < 0:
        h2 = int_genus2_complex(zeros, rea2, ima2, 2, 1, digits)
    else:
        h2 = int_genus2_complex(zeros, rea2, ima2, 2, 0, digits)

    r[0, 1] = 2 * re(h2[0])
    r[1, 1] = 2 * re(h2[1])
   
    # B2: zero3...rea2 (pos. branch) + rea2...zero5 (neg. branch)
    a = myint_genus2(zeros, rea2, zeros[4], 1, digits)
    r[0, 3] = a[0] + h2[0]
    r[1, 3] = a[1] + h2[1]

    # B1 = B2 + B3, B3: zero1...rea1 (pos. branch) + rea1...rea2 (pos. branch)
    # + rea2...zero4 (neg. branch)
    r[0, 2] = r[0, 3] + h1[0] - 1j * quad(lambda x : 1 / sqrt(-p(x)), [rea1, rea2]) - re(h2[0]) + 1j * im(h2[0])
    r[1, 2] = r[1, 3] + h1[1] - 1j * quad(lambda x : x / sqrt(-p(x)), [rea1, rea2]) - re(h2[1]) + 1j * im(h2[1])

    return r

def ima4Per2(realNS, rea1, ima1, rea2, ima2, digits):
    """
    Computes the periods of the first kind for a genus 2 Riemann surface for the case that
    four zeros z1, z2, z3, z4 of the polynomial defining the Riemann surface are complex, where
    Re(z1) = Re(z2), Re(z3) = Re(z4), and Re(z1) < Re(z3), with subcase <k> = 2.

    Parameters
    ----------
    realNS : list
        A list of real numbers, the real roots of the polynomial defining the Riemann
        surface.
    rea1 : float
        The real part of the first complex zero.
    ima1 : float
        The absolute value of the imaginary part of the first complex zero. 
    rea2 : float
        The real part of the third complex zero.
    ima2 : float
        The absolute value of the imaginary part of the third complex zero.
    digits : int
        The number of digits to be used in the computation.

    Returns
    -------
    matrix
        A 2x4 matrix: the first 2x2 matrix represents the period matrix corresponding to the
        integrations along the contours encircling the branch cuts. The second 2x2 matrix
        represents the period matrix corresponding to the integrations along the contours
        connecting the branch cuts.

    """

    r = matrix(2, 4)
    zeros = [realNS[0], rea1 - 1j * ima1, rea1 + 1j * ima1, rea2 - 1j * ima2, rea2 + 1j * ima2]

    # Path rea1...rea1 + I*ima1, neg. branch
    a1 = int_genus2_complex(zeros, rea1, ima1, 1, 1, digits)
    # Path rea2...rea2 + I*ima2, pos. branch
    a2 = int_genus2_complex(zeros, rea2, ima2, 3, 0, digits)

    # Path A1: zero2...rea1 + rea1...zero1, pos. branch
    # + zero1...rea1, neg. branch + rea1...zero3
    w = myint_genus2(zeros, zeros[0], rea1, 0, digits)
    r[0, 0] = 2 * re(a1[0]) - 2 * w[0]
    r[1, 0] = 2 * re(a1[1]) - 2 * w[1]

    # Path A2: zero4...rea2 + rea2...zero1, neg. branch
    # + zero1...rea2, pos. branch + rea2...zero5
    u = myint_genus2(zeros, zeros[0], rea2, 0, digits)
    r[0, 1] = 2 * re(a2[0]) + 2 * u[0]
    r[1, 1] = 2 * re(a2[1]) + 2 * u[1]

    # Path B2:
    r[0, 3] = a2[0] + u[0]
    r[1, 3] = a2[1] + u[1]

    # Path B1 = B2 + B3, path B3:
    r[0, 2] = r[0, 3] + a1[0] - w[0] - u[0] - re(a2[0]) + 1j *im(a2[0])
    r[1, 2] = r[1, 3] + a1[1] - w[1] - u[1] - re(a2[1]) + 1j *im(a2[1])

    return r

def ima4Per3(realNS, rea1, ima1, rea2, ima2, digits):
    """
    Computes the periods of the first kind for a genus 2 Riemann surface for the case that
    four zeros z1, z2, z3, z4 of the polynomial defining the Riemann surface are complex, where
    Re(z1) = Re(z2), Re(z3) = Re(z4), and Re(z1) < Re(z3), with subcase <k> = 3.

    Parameters
    ----------
    realNS : list
        A list of real numbers, the real roots of the polynomial defining the Riemann
        surface.
    rea1 : float
        The real part of the first complex zero.
    ima1 : float
        The absolute value of the imaginary part of the first complex zero. 
    rea2 : float
        The real part of the third complex zero.
    ima2 : float
        The absolute value of the imaginary part of the third complex zero.
    digits : int
        The number of digits to be used in the computation.

    Returns
    -------
    matrix
        A 2x4 matrix: the first 2x2 matrix represents the period matrix corresponding to the
        integrations along the contours encircling the branch cuts. The second 2x2 matrix
        represents the period matrix corresponding to the integrations along the contours
        connecting the branch cuts.

    """

    r = matrix(2, 4)
    zeros = [rea1 - 1j * ima1, rea1 + 1j * ima1, realNS[0], rea2 - 1j * ima2, rea2 + 1j * ima2]

    # Vertical branch cut zero1...zero2, neg. branch:
    h1 = int_genus2_complex(zeros, rea1, ima1, 0, 1, digits)
    r[0, 0] = 2 * re(h1[0]) 
    r[1, 0] = 2 * re(h1[1])

    # Vertical branch zero4...zero5, pos. branch
    h2 = int_genus2_complex(zeros, rea2, ima2, 3, 0, digits)
    
    # A2: zero3...rea2, pos + rea2...zero4
    v = myint_genus2(zeros, zeros[2], rea2, 0, digits)
    r[0, 1] = 2 * re(h2[0]) + 2 * v[0]
    r[1, 1] = 2 * re(h2[1]) + 2 * v[1]

    # B2: zero4...zero3
    r[0, 3] = h2[0] + v[0]
    r[1, 3] = h2[1] + v[1]

    # B1 = B2 + B3, B3: zero1...rea1 + rea1...zero3
    h = myint_genus2(zeros, rea1, zeros[2], 0, digits)
    r[0, 2] = r[0, 3] + h1[0] + h[0] + v[0] + h2[0] - r[1, 0]
    r[1, 2] = r[1, 3] + h1[1] + h[1] + v[1] + h2[1] - r[1, 1]

    return r

def set_period_globals_genus2(period_matrix):
    """
    Sets the global variables <periods_inverse> and <riemannM> based on <period_matrix>, where
    <period_matrix> = [<periods_first>, <periods_second] (<periods_first> and 
    <periods_second> are 2x2 matrices such that <periods_first> are the integrals of the 
    vector of canonical holomorphic differentials along the contours that encircle the branch 
    cuts while <periods_second> are the integrals along the contours connecting the branch 
    cuts).

    Parameters
    ----------
    period_matrix : matrix
        A 2x4 mpmath matrix such that period_matrix = [<periods_first>, <periods_second>] (see
        above).

    Returns
    -------
    periods_inverse : matrix
        A 2x2 mpmath matrix representing the inverse of the matrix <periods_first>
    riemannM : matrix
        A 2x2 mpmath matrix representing the Riemann matrix tau (riemannM = <periods_inverse>
        * <periods_second).

    """

    periods_first = period_matrix[0:2, 0:2]
    periods_second = period_matrix[0:2, 2:4]

    periods_inverse = periods_first**(-1)
    riemannM = periods_inverse * periods_second
    riemannM[0, 1] = 1/2 * (riemannM[0, 1] + riemannM[1, 0])
    riemannM[1, 0] = riemannM[0, 1]

    # Check properties of riemann matrix (positive definite imaginary part)
    m = eig(riemannM.apply(im))[0]
    if im(m[0]) != 0 and im(m[1]) == 0 and re(m[0]) > 0 and re(m[1]) > 0:
        raise ValueError("Imaginary part of Riemann matrix is not positive definite")
    return periods_inverse, riemannM
