#!/usr/bin/env python
"""
This function is used in the main loop of guaranteedEllipseFit in the process of minimizing an 
approximate maximum likelihood cost function of an ellipse fit to data. 

It computes an update for the parameters representing the ellipse, using the method of 
Levenberg-Marquardt for non-linear optimisation. 

"""

from mpmath import matrix, norm, lu_solve

def levenbergMarquardtStep(struct):
    """
    This function computes an update for the parameters representing the ellipse, using the method 
    of Levenberg-Marquardt for non-linear optimisation. 

    Parameters
    ----------
    struct : EllipseData
        An EllipseData type object containing all the iteration information. 

    Returns
    -------
    struct : EllipseData
        An EllipseData type object after applying the Levenberg-Marquardt method.

    """
    
    # Initialize variables to be updated from <struct>
    t = struct.t[:, struct.k]
    jacobian_matrix = struct.jacobian_matrix
    jacobian_matrix_barrier = struct.jacobian_matrix_barrier
    r = struct.r
    I = struct.I
    lbd = struct.lbd
    delta = struct.delta[:, struct.k]
    damping_multiplier = struct.damping_multiplier
    f = struct.f
    current_cost = struct.cost[struct.k]
    data_points = struct.data_points
    alpha = struct.alpha
    numOfPoints = struct.numberOfPoints
   
    # Compute two potential updates for theta based on different weightings of the identity 
    # matrix.

    temp = matrix(numOfPoints + 1, 6)
    temp[0:numOfPoints, :] = jacobian_matrix
    temp[numOfPoints, :] = jacobian_matrix_barrier
    jacob = temp.T * r
   
    # Compute the new update direction
    update_a = lu_solve( - (struct.hessian + lbd * I), jacob)
    update_b = lu_solve( - (struct.hessian + (lbd / damping_multiplier) * I), jacob)

    # The potential new parameters are then 
    t_potential_a = t + update_a
    t_potential_b = t + update_b
       
    # Compute new residuals and costs based on these updates
    # ------------------------------------------------------

    cost_a = 0
    cost_b = 0
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


        t_aBt_a = (t_potential_a.T * b * t_potential_a)[0, 0]
        t_aAt_a = (t_potential_a.T * a * t_potential_a)[0, 0]

        t_bBt_b = (t_potential_b.T * b * t_potential_b)[0, 0]
        t_bAt_b = (t_potential_b.T * a * t_potential_b)[0, 0]

        # AML cost for i'th data point
        cost_a += t_aAt_a / t_aBt_a
        cost_b += t_bAt_b / t_bBt_b

    # Barrier term
    t_aIt_a = (t_potential_a.T * I * t_potential_a)[0, 0]
    t_aFt_a = (t_potential_a.T * f * t_potential_a)[0, 0]
    
    t_bIt_b = (t_potential_b.T * I * t_potential_b)[0, 0]
    t_bFt_b = (t_potential_b.T * f * t_potential_b)[0, 0]
    
    # Add the barrier term
    cost_a += (alpha * (t_aIt_a / t_aFt_a))**2;
    cost_b += (alpha * (t_bIt_b / t_bFt_b))**2;
   
    # Determine appropriate damping and if possible select an update
    if (cost_a >= current_cost and cost_b >= current_cost):
        # Neither update reduced the cost
        struct.theta_updated = False
        # No change in the cost
        struct.cost[struct.k + 1] = current_cost
        # No change in parameters
        struct.t[:, struct.k + 1] = t
        # No changes in step direction
        struct.delta[:, struct.k + 1] = delta
        # Next iteration add more Identity matrix
        struct.lbd = lbd * damping_multiplier
    elif (cost_b < current_cost):
        # Update 'b' reduced the cost function
        struct.theta_updated = True
        # Store the new cost
        struct.cost[struct.k + 1] = cost_b
        # Choose update 'b'
        struct.t[:, struct.k + 1] = t_potential_b / norm(t_potential_b)
        # Store the step direction
        struct.delta[:, struct.k + 1] = update_b
        # Next iteration add less Identity matrix
        struct.lbd = lbd / damping_multiplier
    else:
        # Update 'a' reduced the cost function
        struct.theta_updated = True
        # Store the new cost
        struct.cost[struct.k + 1] = cost_a
        # Choose update 'a'
        struct.t[:, struct.k + 1] = t_potential_a / norm(t_potential_a)
        # Store the step direction
        struct.delta[:, struct.k + 1] = update_a
        # Keep the same damping for the next iteration
        struct.lbd = lbd

    return struct
