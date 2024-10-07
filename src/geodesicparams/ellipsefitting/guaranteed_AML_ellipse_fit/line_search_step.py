#!/usr/bin/env python3
"""
This function is used in the main loop of guaranteedEllipseFit in the process of minimizing an 
approximate maximum likelihood cost function of an ellipse fit to data. 

It computes an update for the parameters representing the ellipse, using the pseudo-inverse 
of a Gauss-Newton approximation of the Hessian matrix. It then performs an inexact line search 
to determine how far to move along the update direction so that the resulting fit is still an 
ellipse. 

"""

from mpmath import matrix, norm
from numpy import array
from numpy.linalg import pinv 

def lineSearchStep(struct):
    """
    This function computes an update for the parameters representing the ellipse, using the 
    pseudo-inverse of a Gauss-Newton approximation of the Hessian matrix. It then performs an 
    inexact line search to determine how far to move along the update direction so that the 
    resulting fit is still an ellipse. 

    Parameters
    ----------
    struct : EllipseData
        An EllipseData type object containing all the iteration information. 

    Returns
    -------
    struct : EllipseData
        An EllipseData type object after applying the Line Search Step method.

    """

    # Initialize variables to be updated from <struct>
    t = struct.t[:, struct.k]
    jacobian_matrix = struct.jacobian_matrix
    jacobian_matrix_barrier = struct.jacobian_matrix_barrier
    r = struct.r
    I = struct.I
    delta = struct.delta[:, struct.k]
    tolDelta = struct.tolDelta
    f = struct.f
    current_cost = struct.cost[struct.k]
    data_points = struct.data_points
    alpha = struct.alpha
    gamma = struct.gamma
    numOfPoints = struct.numberOfPoints
    hessian = struct.hessian

    temp = matrix(numOfPoints + 1, 6)
    temp[0:numOfPoints, :] = jacobian_matrix
    temp[numOfPoints, :] = jacobian_matrix_barrier

    jacob = temp.T * r
    tFt = (t.T * f * t)[0]
  
    # Compute for the new update direction
    update = - matrix(pinv(array(hessian.tolist(), dtype = float), 1e-20)) * jacob

    frac = 0.5
   
    # There is no repeat...until construct so we use a while-do
    while True:
        # Compute potential update 
        t_potential = t + frac * update
        delta = frac * update
        # Halve the step-size
        frac /= 2

        # Compute new residuals on data points
        cost = 0
        for i in range(numOfPoints):
            m = data_points[:, i]
            # Transformed data point
            ux_j = matrix([[m[0]**2, m[0] * m[1], m[1]**2, m[0], m[1], 1]]).T
            # Derivative of transformed data point
            dux_j = matrix([[2 * m[0], m[1], 0, 1, 0, 0], [0, m[0], 2 * m[1], 0, 1, 0]]).T

            # Outer product
            a = ux_j * ux_j.T
            # Use identity covs
            b = dux_j * dux_j.T

            tBt = (t_potential.T * b * t_potential)[0, 0]
            tAt = (t_potential.T * a * t_potential)[0, 0]
              
            # AML cost for i'th data point
            cost +=  tAt / tBt

        # Barrier term
        tIt = (t_potential.T * I * t_potential)[0, 0]
        tFt = (t_potential.T * f * t_potential)[0, 0]
   
        # Add the barrier term
        cost += (alpha * (tIt / tFt))**2

        # Check to see if cost function was sufficiently decreased, and whether the 
        # estimate is still an ellipse. Additonally, if the step size becomes too small we stop.
        if (t_potential.T * f * t_potential > 0 and (cost < (1 - frac * gamma) * current_cost) or norm(delta) < tolDelta):
            break

    struct.theta_update = True
    struct.t[:, struct.k + 1] = t_potential / norm(t_potential)
    struct.delta[:, struct.k + 1] = delta
    struct.cost[struct.k + 1] = cost
 
    return struct
