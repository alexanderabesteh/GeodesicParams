#!/usr/bin/env python3
"""
Procedures for fitting an ellipse to three dimensional cartesian data points using
the guaranteed ellipse fitting method and rodrigues rotation to project the 3D data points
to the 2D fitting plane.

References
----------
[] T. mpmath development team, mpmath: a Python library for arbitrary-precision floating-point arithmetic
    (version 1.3.0) (2023), http://mpmath.org/.
[] C. R. Harris, K. J. Millman, S. J. van der Walt, R. Gom-mers, P. Virtanen, D. Cournapeau,
    E. Wieser, J. Tay-lor, S. Berg, N. J. Smith, R. Kern, M. Picus, S. Hoyer,
    M. H. van Kerkwijk, M. Brett, A. Haldane, J. F. del R´ıo, M. Wiebe, P. Peterson,
    P. G´erard-Marchant, K. Shep-pard, T. Reddy, W. Weckesser, H. Abbasi, C. Gohlke, and
    T. E. Oliphant, Array programming with NumPy, Nature 585, 357 (2020).

"""

from mpmath import findroot, linspace, matrix, pi
from numpy import arccos, array, cos, cross, dot, newaxis, sin, zeros
from numpy.linalg import norm, svd

from ..guaranteed_AML_ellipse_fit.ellipse_estimates import (
    compute_guaranteedellipse_estimate,
    parametric_rep,
)


def rodrigues_rot(data_points, n0, n1):
    """
        Rotates given points based on a starting vector <n0> and ending vector <n1>.

        Parameters
    ----------
        data_points : matrix
             A 2xN mpmath matrix, where the first row represents the x coordinates of the data
             points, and the second represents the y coordinates of the data points.
        n0 : list
            A 1x3 starting vector, which gives the axis k and angle of rotation theta.
        n1 : list
            A 1x3 ending vector, which gives the axis k and angle of rotation theta.

        Returns
        -------
        data_rot : matrix
            An Nx3 matrix containing the fitted x, y, z data points of N length.
        inc_angle : float
            The angle of rotation in the Rodrigues Rotation.

    """

    # If P is only 1d array (coords of single point), fix it to be matrix
    if data_points.ndim == 1:
        data_points = data_points[newaxis, :]

    # Get vector of rotation k and angle theta
    n0 = n0 / norm(n0)
    n1 = n1 / norm(n1)
    k = cross(n0, n1)
    k = k / norm(k)
    theta = arccos(dot(n0, n1))

    # Compute rotated points
    data_rot = zeros((len(data_points), 3))
    for i in range(len(data_points)):
        data_rot[i] = (
            data_points[i] * cos(theta)
            + cross(k, data_points[i]) * sin(theta)
            + k * dot(k, data_points[i]) * (1 - cos(theta))
        )

    return data_rot, theta


def fit_ellipse_3d(data_points, nPoints):
    """
    Given data <data_points> in the 3D cartesian plane,

    Parameters
    ----------
    data_points : matrix
         A 3xN mpmath matrix, where the first row represents the x coordinates of the data
         points, the second represents the y coordinates of the data points.
    nPoints : int
        The number of fitted data points to generate.

    Returns
    -------
    Matrix
        An Nx3 matrix containing the fitted x, y, z data points of N length.
    coeffs : matrix
        A 1x6 matrix representing the coefficients of the ellipse [A, B, C, D, E, F] that
        fits the data points within the Sampson Distance.

    """

    P_mean = data_points.mean(axis=0)
    P_centered = data_points - P_mean

    # Fitting plane by SVD for the mean-centered data
    U, s, V = svd(P_centered, full_matrices=False)

    # Normal vector of fitting plane is given by 3rd column in V
    # Note svd returns V^T, so we need to select 3rd row from V^T
    # normal on 3d plane
    normal = V[2, :]

    # Project points to coords X-Y in 2D plane
    P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])
    P_xy = matrix(P_xy[0][:, :2].T)

    # Use guaranteed ellipse estimate to fit an ellipse to set of 2d points
    coeffs = compute_guaranteedellipse_estimate(P_xy)
    # Generate n 2D points on the fitted elippse
    x, y = parametric_rep(coeffs)

    def f(t1, t2):
        return x(t1) - P_xy[0, 0], y(t2) - P_xy[1, 0]

    # This needs to be fixed
    start = findroot(f, (0, 0), tol=1e-8)
    theta_x = linspace(start[0], 2 * pi + start[0], nPoints)
    theta_y = linspace(start[1], 2 * pi + start[1], nPoints)

    x_list = [x(i) for i in theta_x]
    y_list = [y(i) for i in theta_y]

    xy = matrix(len(x_list), 2)
    xy[:, 0] = matrix(x_list)
    xy[:, 1] = matrix(y_list)

    # Convert the 2D generated points to the 3D space
    points = []
    for i in range(len(xy)):
        points.append([xy[i, 0], xy[i, 1], 0])
    points = array(points)
    rot, inc_angle = rodrigues_rot(points, [0, 0, 1], normal)
    ellipse_points_3d = rot + P_mean

    return matrix(ellipse_points_3d), coeffs, inc_angle
