#!/usr/bin/env python
"""
Procedure for computing the coefficents of an ellipse within the Sampson Distance that
fits a set of data.

The primary function <compute_guaranteedellipse_estimate> returns the coefficients
of an ellipse within the Sampson Distance (i.e. Ax**2 + Bxy + Cy**2 + Dx + Ey + F = 0, 
provided that B**2 - 4ac < 0) that fits the data. The other functions are used for 
conversions, such as converting the coefficient representation into a parametric 
representation that can then be used to generate data.

"""

from mpmath import sqrt, cos, sin, atan2, matrix, norm

from ..direct_ellipse_fitting.ellipse_fit import direct_ellipse_fit
from ..helper_functions.normalise_2d_pts import norm_2d_pts
from .guaranteed_ellipse_fit import guaranteedEllipseFit

def compute_guaranteedellipse_estimate(data_points):
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

    # Initial approximation of coefficients
    theta = direct_ellipse_fit(norm_data)
    theta = theta / norm(theta)

    # Ellipse coefficients within the Sampson Distance
    theta = guaranteedEllipseFit(theta, x)
    theta = theta / norm(theta)

    # Undo coordinate normalization to return the coefficients to their cartesian form
    a = theta[0]; b = theta[1]; c = theta[2]; d = theta[3]; e = theta[4]; f = theta[5]
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

def conv_coeffs_to_axis(coeffs):
    """
    Given the coefficients of a general ellipse [A, B, C, D, E, F], return the semi-major
    and semi-minor axis.

    Parameters
    ----------
    coeffs : matrix, list
        A 1x6 mpmath matrix or a list containing 6 element: the coefficients of the general
        ellipse.

    Returns
    -------
    semi_maj : float
        The semi_major axis of the ellipse.
    semi_min : float
        The semi-minor axis of the ellipse.

    """

    a, b, c, d, e, f = coeffs

    # Conversion function
    # Sign of the square root determines which axis (+ for major, - for minor)
    convert = lambda sign : - sqrt(2 * (a * e**2 + c * d**2 - b * d * e + (b**2 - 4 * a * c) * f) * ((a + c) + sign * sqrt((a - c)**2 + b**2))) / (b**2 - 4 * a * c)
    
    semi_maj = convert(1)
    semi_min = convert(-1)

    return semi_maj, semi_min

def conv_coeffs_to_init_pos(coeffs):
    """
    Given the coefficients of a general ellipse [A, B, C, D, E, F], return the center
    coordinates of the ellipse (x_init, y_init) and the rotation angle of the ellipse
    counterclockwise around the center.

    Parameters
    ----------
    coeffs : matrix, list
        A 1x6 mpmath matrix or a list containing 6 element: the coefficients of the general
        ellipse.

    Returns
    -------
    x_init : float
        The x coordinate of the center of the ellipse.
    y_init : float
        The y coordinate of the center of the ellipse.
    rot_angle : float
        The rotation angle of the ellipse counterclockwise around the center.

    """

    a, b, c, d, e, f = coeffs
    cond = b**2 - 4 * a * c

    x_init = (2 * c * d - b * e) / cond
    y_init = (2 * a * e - b * d) / cond
    rot_angle = 1/2 * atan2(-b, c - a)

    return [x_init, y_init, rot_angle]

def parametric_rep(coeffs):
    """
    Given the coefficients of a general ellipse [A, B, C, D, E, F], return the parametric
    representation of the ellipse as a function of the eccentric anomaly t.

    Parameters
    ----------
    coeffs : matrix, list
        A 1x6 mpmath matrix or a list containing 6 element: the coefficients of the general
        ellipse.

    Returns
    -------
    x : callable
        The x coordinate in the parametric representation as a function of the eccentric anomaly
        t.
    y : callable
        The y coordinate in the parametric representation as a function of the eccentric anomaly
        t.

    """

    a, b =  conv_coeffs_to_axis(coeffs)
    x_init, y_init, rot_angle = conv_coeffs_to_init_pos(coeffs)

    # Parametric representation
    x = lambda t : a * cos(rot_angle) * cos(t) - b * sin(rot_angle) * sin(t) + x_init
    y = lambda t : a * sin(rot_angle) * cos(t) + b * cos(rot_angle) * sin(t) + y_init  

    return x, y
