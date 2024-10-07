#!/usr/bin/env python3
"""
Procedure for computing the coefficents of an ellipse using the direct ellipse fitting
algorithm after normalizing the coordinates (see references).

"""

from mpmath import matrix, norm

from .ellipse_fit import direct_ellipse_fit
from ..helper_functions.normalise_2d_pts import norm_2d_pts

def compute_directellipse_estimate(data_points):
    """
    Compute the coefficients of a general ellipse within the Sampson Distance that fits
    a set of ellipse data (i.e. Ax**2 + Bxy + Cy**2 + Dx + Ey + F = 0, provided that 
    B**2 - 4ac < 0). 

    Parameters
    ----------
    data_points : matrix
         A 2xN mpmath matrix, where the first row represents the x coordinates of the data
         points, and the second represents the y coordinates of the data points.

    Returns
    -------
    result : matrix
        A 1x6 matrix representing the coefficients of the ellipse [A, B, C, D, E, F] that
        fits the data points within the Sampson Distance.

    """

    x = data_points
    n = x.cols

    # Normalize points
    x_new = matrix(3, n)

    x_new[0, :] = x[0, :]
    x_new[1, :] = x[1, :]
    x_new[2, :] = 1

    x, t = norm_2d_pts(x_new)
    norm_data = x

    # Direct ellipse fit coefficients
    theta = direct_ellipse_fit(norm_data)
    theta = theta / norm(theta)
    
    # Undo coordinate normalization to return the coefficients to their cartesian form
    a = theta[0, 0]; b = theta[0, 1]; c = theta[0, 2]; d = theta[1, 0]; e = theta[1, 1]; f = theta[1, 2]
    C = matrix([[a, b / 2, d /2], [b/2, c, e/2], [d/2, e/2, f]])

    C = t.T * C * t
    aa = C[0, 0]
    bb = C[0, 1] * 2
    dd = C[0, 2] * 2
    cc = C[1, 1]
    ee = C[1, 2] * 2
    ff = C[2, 2]
    
    theta = matrix([aa, bb, cc, dd, ee, ff]).T
    theta = theta / norm(theta)

    return theta
