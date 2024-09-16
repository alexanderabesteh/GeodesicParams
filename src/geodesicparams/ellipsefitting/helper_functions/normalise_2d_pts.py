#!/usr/bin/env python
"""
Procedure that translates and normalises a set of 2D homogeneous points so that their 
centroid is at the origin and their mean distance from the origin is sqrt(2). 

This process typically improves the conditioning of any equations used to solve homographies, 
fundamental matrices etc.

"""

from mpmath import matrix, sqrt

from ..matrix_functions import element_pow, element_div

def norm_2d_pts(pts):
    """
    Translates and normalises a set of 2D homogeneous points so that their centroid is at the 
    origin and their mean distance from the origin is sqrt(2). 

    Parameters
    ----------
    pts : matrix
        A 3xN mpmath matrix of 2D homogeneous coordinates.

    Returns
    -------
    newpts : matrix
        A 3xN mpmatrix of transformed 2D homogeneous coordinates. The scaling parameter 
        is normalised to 1 unless the point is at infinity. 
    t : matrix
        The 3x3 transformation matrix (newpts = t * pts).
    """ 
    
    n = pts.cols

    # For the finite points ensure homogeneous coords have scale of 1
    pts[0, :] = element_div(pts[0, :], pts[2, :])
    pts[1, :] = element_div(pts[1, :], pts[2, :])
    pts[2, :] = 1

    sum_cols1 = 0
    sum_cols2 = 0

    for i in range(n):
        sum_cols1 += pts[0, i]
        sum_cols2 += pts[1, i]

    # Centroid of finite points
    c = matrix([[sum_cols1 / n], [sum_cols2 / n]])

    newp = matrix(2, n)

    # Shift origin to centroid
    newp[0, :] = pts[0, :] - c[0]
    newp[1, :] = pts[1, :] - c[1]

    dist = element_pow(element_pow(newp[0, :], 2) + element_pow(newp[1, :], 2), 1/2)
    meandist = 0

    for i in range(dist.cols):
        meandist += dist[i]
    
    meandist = meandist / dist.cols
    scale = sqrt(2) / meandist

    t = matrix([[scale, 0, -scale * c[0]], [0, scale, -scale * c[1]], [0, 0, 1]])

    newpts = t * pts

    return newpts, t
