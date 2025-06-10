#!/usr/bin/env python3
"""
Procedure for computing the periods matrices of the second kind for a genus 2 Riemann surface.

References
----------
[] E. Hackmann, Geodesic equations in black hole space-times with cosmological constant (2010).
[] T. mpmath development team, mpmath: a Python library for arbitrary-precision floating-point arithmetic
    (version 1.3.0) (2023), http://mpmath.org/.
[] A. Meurer, C. P. Smith, M. Paprocki, O. ˇCert´ık, S. B. Kirpichev, M. Rocklin, A. Kumar, S. Ivanov, J. K.
    Moore, S. Singh, T. Rathnayake, S. Vig, B. E. Granger, R. P. Muller, F. Bonazzi, H. Gupta, S. Vats, F. Johans-
    son, F. Pedregosa, M. J. Curry, A. R. Terrel, v. Rouˇcka, A. Saboo, I. Fernando, S. Kulal, R. Cimrman, and
    A. Scopatz, Sympy: symbolic computing in python, PeerJ Computer Science 3, e103 (2017).

"""

from mpmath import fabs, matrix, mp, pi, quad
from sympy import Symbol, collect, im, lambdify, re, sqrt

from ...utilities import inlist
from ..integrations.integrate_hyperelliptic import (
    int_genus2_complex_second,
    myint_genus2_second,
)


def periods_second(dr1, dr2, realNS, complexNS, digits):
    """
    Computes the periods of the second kind for a genus 2 Riemann surface.

    This is done by integrating the vector of canonical meromorphic differentials along the
    contours that encircle the branch cuts and the contours connect the branch cuts.

    Parameters
    ----------
    dr1 : list
        A list representing the coefficients of the first element of the vector of
        canonical meromorphic differentials.
    dr2 : list
        A list representing the coefficients of the second element of the vector of
        canonical meromorphic differentials.
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

    if len(realNS) + len(complexNS) != 5:
        raise ValueError(f"Invalid use: len({realNS}) + len({complexNS}) has to be 5")

    realNS = sorted(realNS, key=lambda x: re(x))
    if len(complexNS) == 0:
        return -rea_second(dr1, dr2, realNS, digits)
    else:
        return -ima_second(dr1, dr2, realNS, complexNS, digits)


def rea_second(dr1, dr2, realNS, digits):
    """
    Computes the periods of the second kind for a genus 2 Riemann surface for the case that
    all zeros of the polynomial defining the Riemann surface are real.

    Parameters
    ----------
    dr1 : list
        A list representing the coefficients of the first element of the vector of
        canonical meromorphic differentials.
    dr2 : list
        A list representing the coefficients of the second element of the vector of
        canonical meromorphic differentials.
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

    eta = matrix(2, 4)

    # path A1: realNS[1]..realNS[2], negative branch
    eta[0, 0] = myint_genus2_second(realNS, dr1, realNS[0], realNS[1], 1, digits)
    eta[1, 0] = myint_genus2_second(realNS, dr2, realNS[0], realNS[1], 1, digits)

    # path A2: realNS[3]..realNS[4], positive branch
    eta[0, 1] = myint_genus2_second(realNS, dr1, realNS[2], realNS[3], 0, digits)
    eta[1, 1] = myint_genus2_second(realNS, dr2, realNS[2], realNS[3], 0, digits)

    # path B2: realNS[4]..realNS[5], negative branch
    eta[0, 3] = myint_genus2_second(realNS, dr1, realNS[3], realNS[4], 1, digits)
    eta[1, 3] = myint_genus2_second(realNS, dr2, realNS[3], realNS[4], 1, digits)

    # path B1=B2+B3, path B3: realNS[2]..realNS[3], positive branch
    eta[0, 2] = (
        myint_genus2_second(realNS, dr1, realNS[1], realNS[2], 0, digits) + eta[0, 3]
    )
    eta[1, 2] = (
        myint_genus2_second(realNS, dr2, realNS[1], realNS[2], 0, digits) + eta[1, 3]
    )

    return eta


def ima_second(dr1, dr2, realNS, complexNS, digits):
    """
    Computes the periods of the second kind for a genus 2 Riemann surface for the case that
    the zeros of the polynomial defining the Riemann surface are complex. <ima2Per> is
    called if there are two complex zeros, and <ima4Per> is called when there are 4 complex
    zeros.

    Parameters
    ----------
    dr1 : list
        A list representing the coefficients of the first element of the vector of
        canonical meromorphic differentials.
    dr2 : list
        A list representing the coefficients of the second element of the vector of
        canonical meromorphic differentials.
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
        return ima2_second(dr1, dr2, realNS, complexNS, digits)
    elif len(complexNS) == 4:
        return ima4_second(dr1, dr2, realNS, complexNS, digits)
    else:
        raise ValueError("Number of complex roots is not 0, 2, or 4")


def ima2_second(dr1, dr2, realNS, complexNS, digits):
    """
    Computes the periods of the second kind for a genus 2 Riemann surface for the case that
    two zeros of the polynomial defining the Riemann surface are complex. There are 4 different
    subcases, ima2Perk, where k is an integer related to the ordering of the zeros.

    For all <k> subcases it is necessary to compute integrals along branch cuts perpendicular
    to the real axis. For the calculation it has to be taken into account that the real part of
    the integrals computed along such branch cuts is symmetrical with respect to the real axis
    but that the imaginary part is antisymmetric. Therefore, the whole integral will be real.

    Parameters
    ----------
    dr1 : list
        A list representing the coefficients of the first element of the vector of
        canonical meromorphic differentials.
    dr2 : list
        A list representing the coefficients of the second element of the vector of
        canonical meromorphic differentials.
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

    k = inlist(
        re(complexNS[0]), sorted(realNS + [re(complexNS[0])], key=lambda x: re(x))
    )

    if k == 0:
        return ima2Per1_second(dr1, dr2, realNS, complexNS, digits)
    elif k == 1:
        return ima2Per2_second(dr1, dr2, realNS, complexNS, digits)
    elif k == 2:
        return ima2Per3_second(dr1, dr2, realNS, complexNS, digits)
    elif k == 3:
        return ima2Per4_second(dr1, dr2, realNS, complexNS, digits)


def ima4_second(dr1, dr2, realNS, complexNS, digits):
    """
    Computes the periods of the second kind for a genus 2 Riemann surface for the case that
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
    dr1 : list
        A list representing the coefficients of the first element of the vector of
        canonical meromorphic differentials.
    dr2 : list
        A list representing the coefficients of the second element of the vector of
        canonical meromorphic differentials.
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

    rea = sorted([re(i) for i in complexNS], key=lambda x: re(x))

    ima1 = fabs(im(complexNS[inlist(rea[0], [re(i) for i in complexNS])]))
    ima2 = fabs(im(complexNS[inlist(rea[2], [re(i) for i in complexNS])]))

    g = realNS[0]

    if rea[3] < g:
        return ima4Per1_second(dr1, dr2, realNS, rea[0], ima1, rea[2], ima2, digits)
    elif rea[3] > g and rea[1] < g:
        return ima4Per3_second(dr1, dr2, realNS, rea[0], ima1, rea[2], ima2, digits)
    elif rea[1] > g:
        return ima4Per2_second(dr1, dr2, realNS, rea[0], ima1, rea[2], ima2, digits)


def ima2Per1_second(dr1, dr2, realNS, complexNS, digits):
    """
    Computes the periods of the second kind for a genus 2 Riemann surface for the case that
    two zeros of the polynomial defining the Riemann surface are complex, with subcase <k> = 1. There are 4 different

    Parameters
    ----------
    dr1 : list
        A list representing the coefficients of the first element of the vector of
        canonical meromorphic differentials.
    dr2 : list
        A list representing the coefficients of the second element of the vector of
        canonical meromorphic differentials.
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

    zeros = [rea - 1j * ima, rea + 1j * ima] + realNS

    eta = matrix(2, 4)

    eta[0, 1] = myint_genus2_second(zeros, dr1, zeros[2], zeros[3], 0, digits)
    eta[1, 1] = myint_genus2_second(zeros, dr2, zeros[2], zeros[3], 0, digits)

    eta[0, 3] = myint_genus2_second(zeros, dr1, zeros[2], zeros[3], 1, digits)
    eta[1, 3] = myint_genus2_second(zeros, dr2, zeros[2], zeros[3], 1, digits)

    h1 = int_genus2_complex_second(zeros, dr1, rea, ima, 0, 1, digits)
    h2 = int_genus2_complex_second(zeros, dr2, rea, ima, 0, 1, digits)

    eta[0, 0] = 2 * re(h1)
    eta[1, 0] = 2 * re(h2)

    eta[0, 2] = (
        h1 + eta[0, 3] + myint_genus2_second(zeros, dr1, rea, zeros[2], 0, digits)
    )
    eta[1, 2] = (
        h2 + eta[1, 3] + myint_genus2_second(zeros, dr2, rea, zeros[2], 0, digits)
    )

    return eta


def ima2Per2_second(dr1, dr2, realNS, complexNS, digits):
    """
    Computes the periods of the second kind for a genus 2 Riemann surface for the case that
    two zeros of the polynomial defining the Riemann surface are complex, with subcase <k> = 2. There are 4 different

    Parameters
    ----------
    dr1 : list
        A list representing the coefficients of the first element of the vector of
        canonical meromorphic differentials.
    dr2 : list
        A list representing the coefficients of the second element of the vector of
        canonical meromorphic differentials.
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

    zeros = [realNS[0], rea - 1j * ima, rea + 1j * ima, realNS[1], realNS[2]]

    eta = matrix(2, 4)

    a1 = int_genus2_complex_second(zeros, dr1, rea, ima, 1, 1, digits)
    a2 = int_genus2_complex_second(zeros, dr2, rea, ima, 1, 1, digits)

    h1 = myint_genus2_second(zeros, dr1, zeros[0], rea, 0, digits)
    h2 = myint_genus2_second(zeros, dr2, zeros[0], rea, 0, digits)

    eta[0, 1] = h1 + myint_genus2_second(zeros, dr1, rea, zeros[3], 0, digits)
    eta[1, 1] = h2 + myint_genus2_second(zeros, dr2, rea, zeros[3], 0, digits)

    eta[0, 0] = -2 * h1 + 2 * re(a1)
    eta[1, 0] = -2 * h2 + 2 * re(a2)

    eta[0, 3] = myint_genus2_second(zeros, dr1, zeros[3], zeros[4], 1, digits)
    eta[1, 3] = myint_genus2_second(zeros, dr2, zeros[3], zeros[4], 1, digits)

    eta[0, 2] = eta[0, 3] - h1 + a1
    eta[1, 2] = eta[1, 3] - h2 + a2

    return eta


def ima2Per3_second(dr1, dr2, realNS, complexNS, digits):
    """
    Computes the periods of the second kind for a genus 2 Riemann surface for the case that
    two zeros of the polynomial defining the Riemann surface are complex, with subcase <k> = 3. There are 4 different

    Parameters
    ----------
    dr1 : list
        A list representing the coefficients of the first element of the vector of
        canonical meromorphic differentials.
    dr2 : list
        A list representing the coefficients of the second element of the vector of
        canonical meromorphic differentials.
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

    zeros = [realNS[0], realNS[1], rea - 1j * ima, rea + 1j * ima, realNS[2]]

    eta = matrix(2, 4)

    eta[0, 0] = myint_genus2_second(zeros, dr1, zeros[0], zeros[1], 1, digits)
    eta[1, 0] = myint_genus2_second(zeros, dr2, zeros[0], zeros[1], 1, digits)

    if (
        (rea - zeros[0]) * (rea - zeros[1])
        + (rea - zeros[1]) * (rea - zeros[4])
        + (rea - zeros[4]) * (rea - zeros[0])
    ).evalf() < 0:
        h1 = int_genus2_complex_second(zeros, dr1, rea, ima, 2, 1, digits)
        h2 = int_genus2_complex_second(zeros, dr2, rea, ima, 2, 1, digits)
    else:
        h1 = int_genus2_complex_second(zeros, dr1, rea, ima, 2, 0, digits)
        h2 = int_genus2_complex_second(zeros, dr2, rea, ima, 2, 0, digits)

    eta[0, 1] = 2 * re(h1)
    eta[1, 1] = 2 * re(h2)

    eta[0, 3] = h1 + myint_genus2_second(zeros, dr1, rea, zeros[4], 1, digits)
    eta[1, 3] = h2 + myint_genus2_second(zeros, dr2, rea, zeros[4], 1, digits)

    eta[0, 2] = (
        eta[0, 3]
        + myint_genus2_second(zeros, dr1, zeros[1], rea, 0, digits)
        - re(h1)
        + 1j * im(h1)
    )
    eta[1, 2] = (
        eta[1, 3]
        + myint_genus2_second(zeros, dr2, zeros[1], rea, 0, digits)
        - re(h2)
        + 1j * im(h2)
    )

    return eta


def ima2Per4_second(dr1, dr2, realNS, complexNS, digits):
    """
    Computes the periods of the second kind for a genus 2 Riemann surface for the case that
    two zeros of the polynomial defining the Riemann surface are complex, with subcase <k> = 4. There are 4 different

    Parameters
    ----------
    dr1 : list
        A list representing the coefficients of the first element of the vector of
        canonical meromorphic differentials.
    dr2 : list
        A list representing the coefficients of the second element of the vector of
        canonical meromorphic differentials.
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

    zeros = realNS + [rea - 1j * ima, rea + 1j * ima]

    eta = matrix(2, 4)

    eta[0, 0] = myint_genus2_second(zeros, dr1, zeros[0], zeros[1], 1, digits)
    eta[1, 0] = myint_genus2_second(zeros, dr2, zeros[0], zeros[1], 1, digits)

    h1 = int_genus2_complex_second(zeros, dr1, rea, ima, 3, 0, digits)
    h2 = int_genus2_complex_second(zeros, dr2, rea, ima, 3, 0, digits)

    v1 = myint_genus2_second(zeros, dr1, zeros[2], rea, 0, digits)
    v2 = myint_genus2_second(zeros, dr2, zeros[2], rea, 0, digits)

    eta[0, 1] = 2 * v1 + 2 * re(h1)
    eta[1, 1] = 2 * v2 + 2 * re(h2)

    eta[0, 3] = v1 + h1
    eta[1, 3] = v2 + h2

    eta[0, 2] = 2 * 1j * im(h1) + myint_genus2_second(
        zeros, dr1, zeros[1], zeros[2], 0, digits
    )
    eta[1, 2] = 2 * 1j * im(h2) + myint_genus2_second(
        zeros, dr2, zeros[1], zeros[2], 0, digits
    )

    return eta


def ima4Per1_second(dr1, dr2, realNS, rea1, ima1, rea2, ima2, digits):
    """
    Computes the periods of the second kind for a genus 2 Riemann surface for the case that
    four zeros z1, z2, z3, z4 of the polynomial defining the Riemann surface are complex, where
    Re(z1) = Re(z2), Re(z3) = Re(z4), and Re(z1) < Re(z3), with subcase <k> = 1.

    Parameters
    ----------
    dr1 : list
        A list representing the coefficients of the first element of the vector of
        canonical meromorphic differentials.
    dr2 : list
        A list representing the coefficients of the second element of the vector of
        canonical meromorphic differentials.
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

    eta = matrix(2, 4)
    x = Symbol("x")

    h1 = matrix(1, 2)
    h2 = matrix(1, 2)

    zeros = [
        rea1 - 1j * ima1,
        rea1 + 1j * ima1,
        rea2 - 1j * ima2,
        rea2 + 1j * ima2,
        realNS[0],
    ]
    t = 1
    p = 0

    for i in range(5):
        t *= x - zeros[i]
    t = collect(t.expand(), x)

    for i in range(6):
        p += re(t.coeff(x, i)) * x**i
    p = lambdify(x, p)

    h1[0] = int_genus2_complex_second(zeros, dr1, rea1, ima1, 0, 1, digits)
    h1[1] = int_genus2_complex_second(zeros, dr2, rea1, ima1, 0, 1, digits)

    eta[0, 0] = 2 * re(h1[0])
    eta[1, 0] = 2 * re(h1[1])

    h2[0] = int_genus2_complex_second(zeros, dr1, rea2, ima2, 2, 1, digits)
    h2[1] = int_genus2_complex_second(zeros, dr2, rea2, ima2, 2, 1, digits)

    eta[0, 1] = 2 * re(h2[0])
    eta[1, 1] = 2 * re(h2[1])

    eta[0, 3] = myint_genus2_second(zeros, dr1, rea2, zeros[4], 1, digits) + h2[0]
    eta[1, 3] = myint_genus2_second(zeros, dr2, rea2, zeros[4], 1, digits) + h2[1]

    x1 = 0
    x2 = 0

    for i in range(1, len(dr1) + 1):
        x1 += dr1[i - 1] * x ** (i - 1)
    for i in range(1, len(dr2) + 1):
        x2 += dr2[i - 1] * x ** (i - 1)

    x1 = lambdify(x, x1)
    x2 = lambdify(x, x2)

    eta[0, 2] = (
        eta[0, 3]
        + h1[0]
        - 1j * quad(lambda x: x1(x) / sqrt(-p(x)), [rea1, rea2])
        - re(h2[0])
        + 1j * im(h2[0])
    )
    eta[1, 2] = (
        eta[1, 3]
        + h1[1]
        - 1j * quad(lambda x: x2(x) / sqrt(-p(x)), [rea1, rea2])
        - re(h2[1])
        + 1j * im(h2[1])
    )

    return eta


def ima4Per2_second(dr1, dr2, realNS, rea1, ima1, rea2, ima2, digits):
    """
    Computes the periods of the second kind for a genus 2 Riemann surface for the case that
    four zeros z1, z2, z3, z4 of the polynomial defining the Riemann surface are complex, where
    Re(z1) = Re(z2), Re(z3) = Re(z4), and Re(z1) < Re(z3), with subcase <k> = 2.

    Parameters
    ----------
    dr1 : list
        A list representing the coefficients of the first element of the vector of
        canonical meromorphic differentials.
    dr2 : list
        A list representing the coefficients of the second element of the vector of
        canonical meromorphic differentials.
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

    eta = matrix(2, 4)
    h1 = matrix(1, 2)
    h2 = matrix(1, 2)

    zeros = [
        realNS[0],
        rea1 - 1j * ima1,
        rea1 + 1j * ima1,
        rea2 - 1j * ima2,
        rea2 + 1j * ima2,
    ]

    h1[0] = int_genus2_complex_second(zeros, dr1, rea1, ima1, 1, 1, digits)
    h1[1] = int_genus2_complex_second(zeros, dr2, rea1, ima1, 1, 1, digits)

    h2[0] = int_genus2_complex_second(zeros, dr1, rea2, ima2, 3, 0, digits)
    h2[1] = int_genus2_complex_second(zeros, dr2, rea2, ima2, 3, 0, digits)

    w1 = myint_genus2_second(zeros, dr1, zeros[0], rea1, 0, digits)
    w2 = myint_genus2_second(zeros, dr2, zeros[0], rea1, 0, digits)

    eta[0, 0] = 2 * re(h1[0]) - 2 * w1
    eta[1, 0] = 2 * re(h1[1]) - 2 * w2

    v1 = myint_genus2_second(zeros, dr1, zeros[0], rea2, 0, digits)
    v2 = myint_genus2_second(zeros, dr2, zeros[0], rea2, 0, digits)

    eta[0, 1] = 2 * re(h2[0]) + 2 * v1
    eta[1, 1] = 2 * re(h2[1]) + 2 * v2

    eta[0, 3] = h2[0] + v1
    eta[1, 3] = h2[1] + v2

    eta[0, 2] = eta[0, 3] + h1[0] - w1 - v1 - re(h2[0]) + 1j * im(h2[0])
    eta[1, 2] = eta[1, 3] + h1[1] - w2 - v2 - re(h2[1]) + 1j * im(h2[1])

    return eta


def ima4Per3_second(dr1, dr2, realNS, rea1, ima1, rea2, ima2, digits):
    """
    Computes the periods of the second kind for a genus 2 Riemann surface for the case that
    four zeros z1, z2, z3, z4 of the polynomial defining the Riemann surface are complex, where
    Re(z1) = Re(z2), Re(z3) = Re(z4), and Re(z1) < Re(z3), with subcase <k> = 3.

    Parameters
    ----------
    dr1 : list
        A list representing the coefficients of the first element of the vector of
        canonical meromorphic differentials.
    dr2 : list
        A list representing the coefficients of the second element of the vector of
        canonical meromorphic differentials.
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

    eta = matrix(2, 4)
    h1 = matrix(1, 2)
    h2 = matrix(1, 2)

    zeros = [
        rea1 - 1j * ima1,
        rea1 + 1j * ima1,
        realNS[0],
        rea2 - 1j * ima2,
        rea2 + 1j * ima2,
    ]

    h1[0] = int_genus2_complex_second(zeros, dr1, rea1, ima1, 0, 1, digits)
    h1[1] = int_genus2_complex_second(zeros, dr2, rea1, ima1, 0, 1, digits)

    h2[0] = int_genus2_complex_second(zeros, dr1, rea2, ima2, 3, 0, digits)
    h2[1] = int_genus2_complex_second(zeros, dr2, rea2, ima2, 3, 0, digits)

    eta[0, 0] = 2 * re(h1[0])
    eta[1, 0] = 2 * re(h1[1])

    v1 = myint_genus2_second(zeros, dr1, zeros[2], rea2, 0, digits)
    v2 = myint_genus2_second(zeros, dr2, zeros[2], rea2, 0, digits)

    eta[0, 1] = 2 * re(h2[0]) + 2 * v1
    eta[1, 1] = 2 * re(h2[1]) + 2 * v2

    eta[0, 3] = -v1 + h2[0]
    eta[1, 3] = -v2 + h2[1]

    eta[0, 2] = (
        eta[0, 3]
        + myint_genus2_second(zeros, dr1, rea1, zeros[2], 0, digits)
        + h1[0]
        - re(h2[0])
        + 1j * im(h2[0])
    )
    eta[1, 2] = (
        eta[1, 3]
        + myint_genus2_second(zeros, dr2, rea1, zeros[2], 0, digits)
        + h1[1]
        - re(h2[1])
        + 1j * im(h2[1])
    )

    return eta
