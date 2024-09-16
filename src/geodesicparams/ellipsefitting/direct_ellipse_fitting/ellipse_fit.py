#!/usr/bin/env python
"""
Procedure for computing the coefficents of an ellipse using the direct ellipse fitting
algorithm (see references).

"""

from mpmath import matrix, eig, norm

from ..matrix_functions import element_pow, element_mul

def direct_ellipse_fit(data):
    """
    Compute the coefficients of a general ellipse using the direct ellipse fitting algorithm
    (see references, tbd) that fit a set of ellipse data (i.e. Ax**2 + Bxy + Cy**2 + Dx + 
    Ey + F = 0, provided that B**2 - 4ac < 0). 

    Parameters
    ----------
    data_points : matrix
         A 2xN mpmath matrix, where the first row represents the x coordinates of the data
         points, and the second represents the y coordinates of the data points.

    Returns
    -------
    result : matrix
        A 1x6 matrix representing the coefficients of the ellipse [A, B, C, D, E, F] that
        fits the data points.
    """

    x = data[0, :].T
    y = data[1, :].T
   
    # Design matrix
    d1 = matrix(x.rows, 3)
    d2 = matrix(x.rows, 3)
   
    # Quadratic part of design matrix
    d1[:, 0] = element_pow(x, 2)
    d1[:, 1] = element_mul(x, y)
    d1[:, 2] = element_pow(y, 2)

    # Linear part of design matrix
    d2[:, 0] = x
    d2[:, 1] = y
    d2[:, 2] = 1

    # Quadratic, combined, and linear part of scatter matrix
    s1 = d1.T * d1
    s2 = d1.T * d2
    s3 = d2.T * d2

    # For obtaining a2 from a1
    t = - s3**(-1) * s2.T

    # Reduce scatter matrix
    m = s1 + s2 * t

    # Constraint matrix
    c1 = matrix([[0, 0, 2], [0, -1, 0], [2, 0, 0]])
    m = c1**(-1) * m

    # Eigensystem
    evalue, evec = eig(m)
    cond = 4 * element_mul(evec[0, :], evec[2, :]) - element_pow(evec[1, :], 2)
    
    # Find positive eigenvalue
    for i in range(cond.cols):
        if cond[0, i] > 0:
            pos = i
            break

    # Ellipse coefficients
    a1 = evec[:, pos]
    a = matrix(2, 3)
    a[0, :] = a1.T
    a[1, :] = (t * a1).T

    a = a / norm(a)

    return a
